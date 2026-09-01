"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment:
  GEMINI_API_KEY    -> Google Gemini (primary)
  OPENAI_API_KEY    -> OpenAI (fallback)
  ANTHROPIC_API_KEY -> Anthropic (fallback)
  LLM_URL           -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the default (fast) model for any provider.
LLM_MODEL_QUALITY env var sets a higher-quality model for critical steps
(resume tailoring, cover letters). Falls back to LLM_MODEL if not set.

When a model hits a 429 rate limit, the client automatically tries the
next model in the fallback chain — including cross-provider fallback to
OpenAI and Anthropic if their API keys are configured.
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

# Stage-specific knobs (all optional):
#   APPLYPILOT_MODEL_SCORE, APPLYPILOT_MODEL_TAILOR, APPLYPILOT_MODEL_COVER,
#   APPLYPILOT_MODEL_JUDGE
#   APPLYPILOT_MAX_TOKENS_SCORE, APPLYPILOT_MAX_TOKENS_TAILOR,
#   APPLYPILOT_MAX_TOKENS_COVER, APPLYPILOT_MAX_TOKENS_JUDGE
#   APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY (default: true) -- when true, the
#   claude_cli fallback tier is never added to any fallback chain built by
#   this module. Nothing in this module ever serves the auto-apply stage
#   (apply/launcher.py talks to the `claude` CLI directly, independent of
#   this file) -- everything here is score/tailor/cover/judge/discover/
#   enrich/tracking, i.e. "normal" stages. Auto-apply and this module's
#   claude_cli tier draw on the same underlying Max-plan usage window, so
#   by default normal stages don't touch it at all, leaving the full
#   window free for auto-apply. Set to "false"/"0"/"no" to restore the
#   old behavior (claude_cli as a last-resort fallback for these stages
#   too). 2026-08-20 incident: a cover run exhausted all 4 API providers,
#   cascaded to claude_cli for every remaining job, and burned a chunk of
#   the shared session window that auto-apply needed later.

log = logging.getLogger(__name__)


def _claude_cli_reserved_for_apply() -> bool:
    raw = os.environ.get("APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY", "true").strip().lower()
    return raw not in ("false", "0", "no")


# ---------------------------------------------------------------------------
# Model registry — each entry knows its provider, endpoint, and API key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    """A model with everything needed to call it."""

    name: str
    provider: str  # "gemini", "openai", "anthropic", "claude_cli", "local"
    base_url: str  # for claude_cli: the resolved path to the claude binary
    api_key: str


def _find_claude_cli() -> str | None:
    """Locate the Claude Code CLI binary, or None if not installed.

    Used as a zero-marginal-cost fallback tier: the user's existing Max plan
    subscription (OAuth login) rather than a metered ANTHROPIC_API_KEY. Checks
    PATH first, then the installer's default location, since a subprocess's
    inherited PATH doesn't always include user-local install dirs.
    """
    found = shutil.which("claude")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "claude.exe",
        Path.home() / ".local" / "bin" / "claude",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def local_openai_base_url(raw_url: str) -> str:
    """Normalize a configured local LLM URL for THIS module's OpenAI-
    compatible request construction.

    APPLYPILOT_LOCAL_LLM_URL is consumed by two independent code paths
    that expect different URL shapes:
      - local_tailor.get_local_tailoring_plan() posts directly to Ollama's
        *native* endpoint, {url}/api/chat -- it wants the bare server root
        (e.g. http://127.0.0.1:11434), no /v1.
      - This module's LLMClient is OpenAI-compatible-only: every "local"
        ModelEntry goes through _try_openai_compat, which always posts to
        {entry.base_url}/chat/completions -- exactly like it does for
        Gemini/OpenAI, whose base_urls already end in their API's versioned
        root. Ollama only serves an OpenAI-compatible surface under /v1, so
        the same bare root that's correct for /api/chat resolves to a 404
        here (POST http://host:port/chat/completions matches no route).
    A single env var can't literally satisfy both conventions at once, so
    each consumer normalizes it for its OWN endpoint style instead of
    requiring the user to guess the "right" one. This is that
    normalization for the OpenAI-compatible side: append /v1 unless
    already present, so both the documented convention (already includes
    /v1) and Ollama's bare default root work without user action.
    """
    url = raw_url.rstrip("/")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def _build_fallback_chain(
    primary_model: str, quality: bool = False, include_local: bool | None = None
) -> list[ModelEntry]:
    """Build a cross-provider fallback chain starting from the primary model.

    Gemini models come first (free tier), then OpenAI (cheap), then Anthropic.
    Only includes providers whose API keys are configured.

    include_local: whether to append the configured local model
        (APPLYPILOT_LOCAL_LLM_URL) as a last-resort entry. Defaults to
        `quality` when not given explicitly.

        2026-08-23: local (Qwen) must never be part of the FAST tier's
        chain -- that's what scoring uses (get_stage_client("score",
        quality=False)), and testing found qwen3:1.7b's scoring unreliable
        enough (missing obvious SWE experience, scoring known-9s as 7)
        that it must never silently substitute for a cloud model there. The
        QUALITY tier (tailor, cover) keeps local as the existing, deliberate
        last-resort fallback for when every cloud provider is exhausted
        (see tailor.py's DEGRADED MODE / has_cloud_available()) -- that
        behavior is unchanged. Explicitly-local operations (the
        local-first planner in local_tailor.py, `applypilot test-local`)
        don't go through this function's chain at all -- they either call
        Ollama directly or construct/override a local-only ModelEntry
        themselves, so they're unaffected either way. See CLAUDE.md
        decision #54.
    """
    if include_local is None:
        include_local = quality
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    openai_url = "https://api.openai.com/v1"
    anthropic_url = "https://api.anthropic.com"
    deepseek_url = "https://api.deepseek.com/v1"

    # Gemini chains — use verified model IDs only.
    # 2026-08-18: gemini-2.5-* / gemini-2.0-* all 404 ("no longer available to
    # new users") on this key's project; Google's error pointed at the 3.x
    # generation. gemini-3.1-pro-preview 429s (quota), so quality tier tries
    # it first but falls back fast to the working flash models.
    if quality:
        gemini_models = [
            "gemini-3.1-pro-preview",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
        ]
    else:
        gemini_models = [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
        ]

    # OpenAI fallbacks (cost-efficient)
    if quality:
        openai_models = ["gpt-4.1-mini", "gpt-4.1-nano"]
    else:
        openai_models = ["gpt-4.1-nano", "gpt-4.1-mini"]

    # Anthropic fallbacks (cost-efficient)
    if quality:
        anthropic_models = ["claude-sonnet-4-5-20250514", "claude-haiku-4-5-20251001"]
    else:
        anthropic_models = ["claude-haiku-4-5-20251001"]

    chain: list[ModelEntry] = []

    # 2026-08-22 fix: `primary_model` not being one of the known gemini_models
    # names used to be treated as "the user configured an unlisted/newer
    # Gemini model" and injected as a Gemini entry unconditionally. That's
    # correct when the caller actually intends Gemini (e.g. a genuinely new
    # Gemini model id via LLM_MODEL) but wrong whenever `primary_model`
    # happens to be the configured LOCAL model name instead (e.g. cli.py's
    # `test-local` command constructs LLMClient(local_url, local_model, ...)
    # directly) -- with GEMINI_API_KEY also set, that produced a bogus
    # "qwen3:8b (gemini)" entry that 404s against the real Gemini API before
    # ever reaching the correct "qwen3:8b (local)" entry below.
    _local_model_name = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", "").strip()
    _primary_is_local_model = bool(_local_model_name) and primary_model == _local_model_name

    # Start from the primary model in the Gemini chain
    if gemini_key:
        started = False
        for m in gemini_models:
            if m == primary_model:
                started = True
            if started:
                chain.append(ModelEntry(m, "gemini", gemini_url, gemini_key))
        # If primary wasn't found in chain, add the full chain -- but never
        # inject the primary itself as a Gemini entry when it's actually the
        # configured local model (see note above).
        if not started:
            if not _primary_is_local_model:
                chain.append(ModelEntry(primary_model, "gemini", gemini_url, gemini_key))
            for m in gemini_models:
                if m != primary_model:
                    chain.append(ModelEntry(m, "gemini", gemini_url, gemini_key))

    # DeepSeek fallbacks (cheap, OpenAI-compatible)
    if quality:
        deepseek_models = ["deepseek-chat"]
    else:
        deepseek_models = ["deepseek-chat"]

    # OpenAI fallbacks
    if openai_key:
        for m in openai_models:
            chain.append(ModelEntry(m, "openai", openai_url, openai_key))

    # DeepSeek fallbacks
    if deepseek_key:
        for m in deepseek_models:
            chain.append(ModelEntry(m, "deepseek", deepseek_url, deepseek_key))

    # Claude Code CLI fallback: uses the user's existing Max plan OAuth login
    # (flat subscription, no per-token billing) instead of a metered
    # ANTHROPIC_API_KEY. Tried before the paid Anthropic API tier since it's
    # effectively free for a Max-plan user; much slower per call (CLI startup
    # overhead, ~15s vs ~1-3s for a direct API call) so it's a fallback, not
    # the primary path.
    claude_cli_path = _find_claude_cli()
    claude_reserved = claude_cli_path and _claude_cli_reserved_for_apply()
    if claude_cli_path and not claude_reserved:
        cli_model = "sonnet" if quality else "haiku"
        chain.append(ModelEntry(cli_model, "claude_cli", claude_cli_path, ""))

    # Anthropic fallbacks
    if anthropic_key:
        for m in anthropic_models:
            chain.append(ModelEntry(m, "anthropic", anthropic_url, anthropic_key))

    # Local model fallback — appended last, after all cloud providers, but
    # BEFORE the empty-chain check so a local-only setup (no cloud keys at
    # all) is valid, PROVIDED include_local allows it (see docstring: the
    # fast/scoring tier never includes it). Configured separately from
    # LLM_URL (which sets the primary provider) so existing local-primary
    # setups are unaffected. Presence of the URL implies enabled; no
    # separate flag needed (consistent with how GEMINI_API_KEY works).
    _local_url = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "").rstrip("/")
    _local_model = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", "llama3.2")
    if include_local and _local_url:
        chain.append(ModelEntry(_local_model, "local", local_openai_base_url(_local_url), ""))

    # If nothing was added (no keys, no local), raise
    if not chain:
        reserved_note = (
            " claude_cli is installed but reserved for auto-apply "
            "(APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY=false to allow it here)."
            if claude_reserved
            else ""
        )
        _local_note = " or APPLYPILOT_LOCAL_LLM_URL" if include_local else ""
        raise RuntimeError(
            "No LLM provider configured. "
            "Set GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY"
            + _local_note
            + "."
            + reserved_note
        )

    return chain


# ---------------------------------------------------------------------------
# Provider detection (for primary model selection)
# ---------------------------------------------------------------------------


def _detect_provider(quality: bool = False, model_override: str | None = None) -> tuple[str, str, str]:
    """Return (base_url, model, api_key) for the primary provider."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")

    global_model = os.environ.get("LLM_MODEL", "")
    quality_model = os.environ.get("LLM_MODEL_QUALITY", "")

    if model_override:
        chosen_model = model_override
    elif quality and quality_model:
        chosen_model = quality_model
    else:
        chosen_model = global_model

    if gemini_key and not local_url:
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            chosen_model or "gemini-3.6-flash",
            gemini_key,
        )
    if openai_key and not local_url:
        return (
            "https://api.openai.com/v1",
            chosen_model or "gpt-4.1-nano",
            openai_key,
        )
    if local_url:
        return (
            local_url.rstrip("/"),
            chosen_model or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )
    raise RuntimeError("No LLM provider configured. Set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL.")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_TIMEOUT = 300  # seconds


# ---------------------------------------------------------------------------
# TEMPORARY diagnostic: full local-Qwen response capture
# ---------------------------------------------------------------------------
# 2026-08-23: investigating why the local qwen3:1.7b fallback is slow/times
# out frequently -- writes each successful local-model response IN FULL to
# its own file (never the main ApplyPilot log, so this doesn't flood normal
# operational logging). Remove _log_local_qwen_response, its call site in
# _try_openai_compat, and _get_qwen_diag_logger once the investigation is
# done.

_qwen_diag_logger: logging.Logger | None = None
_qwen_diag_lock = threading.Lock()


def _get_qwen_diag_logger() -> logging.Logger:
    """Lazily create a logger that writes only to
    ~/.applypilot/logs/local_qwen_output.log (propagate=False keeps it out
    of the main log handlers/file)."""
    global _qwen_diag_logger
    if _qwen_diag_logger is not None:
        return _qwen_diag_logger
    with _qwen_diag_lock:
        if _qwen_diag_logger is not None:
            return _qwen_diag_logger
        logger = logging.getLogger("applypilot.llm.qwen_diagnostic")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        log_dir = Path.home() / ".applypilot" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "local_qwen_output.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        _qwen_diag_logger = logger
        return logger


def _log_local_qwen_response(entry: "ModelEntry", messages: list[dict], text: str) -> None:
    """Record a successful local-model response to the diagnostic file.

    Only called after a real, non-null response is received (see the call
    site in _try_openai_compat) -- never on a failed/timed-out attempt.
    `messages` never contains API keys/headers (those live in `headers`,
    not the request body), so this can't leak secrets. Best-effort request
    identifier: this layer doesn't receive a job id, so we log a short
    prefix of the first user message, which is normally where job/company
    context lives in this codebase's prompts.
    """
    context = ""
    for msg in messages:
        if msg.get("role") == "user":
            context = (msg.get("content") or "").strip().replace("\n", " ")[:200]
            break
    diag = _get_qwen_diag_logger()
    diag.info(
        "=== local/%s response (%d chars) ===\ncontext: %s\n%s\n=== end response ===",
        entry.name,
        len(text),
        context or "(no user message found)",
        text,
    )


class LLMClient:
    """Multi-provider LLM client with automatic model fallback."""

    def __init__(self, base_url: str, model: str, api_key: str, quality: bool = False) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.quality = quality
        self._fallback_chain = _build_fallback_chain(model, quality=quality)
        self._client = httpx.Client(timeout=_TIMEOUT)
        # Observability (2026-08-31): which provider/model actually answered
        # the most recent successful chat() call -- read-only, additive,
        # never consulted for control flow. Lets a caller (scorer.py,
        # tailor.py, ...) log "which model handled this job" and detect
        # escalation (last_model_used != self.model, the configured
        # primary) without parsing log text or changing chat()'s str
        # return contract. None until the first successful call.
        self.last_model_used: str | None = None
        self.last_provider_used: str | None = None
        # Track which models are temporarily exhausted (store until timestamp)
        self._exhausted: dict[str, float] = {}
        # Additive, read-only classification alongside self._exhausted -- lets
        # a caller (the continuous scheduler) distinguish *why* claude_cli was
        # marked exhausted without parsing logs or changing _try_claude_cli's
        # str | None return contract. Never consulted by chat()/_try_claude_cli
        # itself for control flow, so it can't change existing behavior.
        self._exhaustion_reason: dict[str, str] = {}
        # claude_cli specifically is serialized (not just exhaustion-tracked):
        # concurrent workers can each pass the "not exhausted yet" check before
        # any of them has finished a ~3-30s subprocess call and written the
        # exhaustion flag, producing a burst of near-simultaneous claude_cli
        # invocations against a scarce, shared Max-plan session window. Gemini/
        # OpenAI/Anthropic don't need this -- they're not the reserved resource.
        self._claude_cli_lock = threading.Lock()

        chain_names = [f"{e.name} ({e.provider})" for e in self._fallback_chain]
        log.info("Fallback chain (%s): %s", "quality" if quality else "fast", " -> ".join(chain_names))

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        exclude_providers: frozenset[str] | None = None,
    ) -> str:
        """Send a chat completion request with automatic cross-provider fallback.

        exclude_providers: providers to skip for THIS call only (e.g.
        frozenset({"local"})) -- does not touch self._fallback_chain, so it
        can't race with concurrent callers sharing this client. 2026-08-23:
        added so a caller can discover "cloud is exhausted" via a fast,
        cloud-only attempt (429s reject in ~1-2s each) instead of silently
        falling through to a slow local model it didn't intend to use for
        this particular call -- see tailor.py's tailor_resume() DEGRADED
        MODE handling.
        """
        # Qwen3 /no_think handling lives in _try_openai_compat, keyed on the
        # actual entry being attempted (entry.name) rather than self.model --
        # see that method's docstring for why (self.model is the ORIGINAL
        # primary model, wrong whenever qwen is reached as a fallback from
        # Gemini/OpenAI rather than as the primary).

        # Build list of models to try: skip recently exhausted ones
        now = time.time()

        # Skip entries that are marked exhausted. Support two storage styles:
        # - legacy: stored value is the time they were marked exhausted (start time)
        #   and a short cooldown (5 minutes) applies
        # - new: stored value is an explicit 'until' timestamp (time to allow retry)
        def _is_exhausted(name: str) -> bool:
            if name not in self._exhausted:
                return False
            v = self._exhausted[name]
            if v is None:
                return False
            if v > now:
                # v is an until-timestamp in the future -> still exhausted
                return True
            # legacy mode: v is the time it was marked exhausted; cooldown = 300s
            return (now - v) < 300

        # Exclusion is a static, per-call filter (never recoverable within
        # this call) -- applied before the exhaustion/cooldown-recovery
        # logic below, which stays about STALE exhaustion timestamps, not
        # about providers deliberately left out of consideration.
        candidate_chain = (
            [e for e in self._fallback_chain if e.provider not in exclude_providers]
            if exclude_providers
            else self._fallback_chain
        )

        entries_to_try = [e for e in candidate_chain if not _is_exhausted(e.name)]
        if not entries_to_try:
            # Don't blindly clear all exhaustion — only remove short-term rate-limit
            # cooldowns (<= 5 min remaining). Long-term 24h daily-quota blocks must
            # persist so repeated batch calls don't keep hammering the same provider.
            _SHORT_COOLDOWN = 300
            for _e in list(candidate_chain):
                _v = self._exhausted.get(_e.name)
                if _v is None:
                    continue
                _remaining = (_v - now) if _v > now else max(0.0, _SHORT_COOLDOWN - (now - _v))
                if _remaining <= _SHORT_COOLDOWN:
                    self._exhausted.pop(_e.name, None)
            entries_to_try = [e for e in candidate_chain if not _is_exhausted(e.name)]
            if not entries_to_try:
                _waits = [
                    (self._exhausted[_e.name] - now) / 3600
                    for _e in candidate_chain
                    if _e.name in self._exhausted and self._exhausted[_e.name] > now
                ]
                _min_h = min(_waits) if _waits else 0.0
                # Only suggest APPLYPILOT_LOCAL_LLM_URL when local is actually
                # eligible for THIS chain (quality tier), not already set, and
                # not deliberately excluded for this call -- for the fast/
                # scoring tier, local is deliberately excluded by default
                # (see _build_fallback_chain), so suggesting it here would be
                # misleading advice that wouldn't actually add a fallback.
                _local_hint = (
                    " Set APPLYPILOT_LOCAL_LLM_URL for a local fallback."
                    if self.quality
                    and not os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "").strip()
                    and not (exclude_providers and "local" in exclude_providers)
                    else ""
                )
                raise RuntimeError(f"All LLM providers are on quota cooldown (min wait: {_min_h:.1f}h).{_local_hint}")

        for idx, entry in enumerate(entries_to_try):
            is_last = idx == len(entries_to_try) - 1
            result = self._try_entry(entry, messages, temperature, max_tokens, is_last)
            if result is not None:
                self.last_provider_used = entry.provider
                self.last_model_used = entry.name
                return result

        raise RuntimeError(
            f"All models exhausted after trying: "
            f"{[e.name for e in entries_to_try]}. "
            "Wait a few minutes for rate limits to reset."
        )

    def _try_entry(
        self, entry: ModelEntry, messages: list[dict], temperature: float, max_tokens: int, is_last: bool = False
    ) -> str | None:
        """Try a single model entry. Dispatches to the right provider."""
        if entry.provider == "anthropic":
            return self._try_anthropic(entry, messages, temperature, max_tokens, is_last)
        elif entry.provider == "claude_cli":
            return self._try_claude_cli(entry, messages, is_last)
        else:
            return self._try_openai_compat(entry, messages, temperature, max_tokens, is_last)

    def _try_openai_compat(
        self, entry: ModelEntry, messages: list[dict], temperature: float, max_tokens: int, is_last: bool = False
    ) -> str | None:
        """Try an OpenAI-compatible endpoint (Gemini, OpenAI, local)."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        # Local Ollama / llama.cpp may not require auth; omit Bearer when key is empty
        if entry.api_key:
            headers["Authorization"] = f"Bearer {entry.api_key}"
        # DeepSeek deepseek-chat has an 8192 max output token limit
        if entry.provider == "deepseek":
            max_tokens = min(max_tokens, 8192)
        # 2026-08-23 incident: a cloud-sized max_tokens (e.g. tailor's 16384)
        # reaching a CPU-bound local model -- whether via an intentional
        # local call or a cloud fallback chain that happens to cascade all
        # the way down to "local" as its last entry -- let a slow small
        # model grind for minutes even with /no_think, since the output
        # budget alone (not just prompt size) drives generation time. Clamp
        # unconditionally for any local-provider call; small, purpose-built
        # local calls (e.g. local_tailor's realization/plan prompts) already
        # request far less than this and are unaffected.
        if entry.provider == "local":
            max_tokens = min(max_tokens, int(os.environ.get("APPLYPILOT_LOCAL_LLM_MAX_TOKENS", "2048")))

        # Qwen3 optimization: disable the model's internal <think> reasoning
        # for models we only use as a fast structured-output planner. Keyed
        # on entry.name (the model actually being attempted on THIS
        # fallback tier), not self.model -- 2026-08-23 bug: the old check in
        # chat() used self.model (the ORIGINAL primary model), so /no_think
        # silently never applied when qwen was reached as a fallback from
        # Gemini/OpenAI rather than as the primary model. Builds a local
        # copy so the prefix doesn't mutate `messages`, which chat() reuses
        # for every remaining fallback attempt in the chain.
        req_messages = messages
        if "qwen" in entry.name.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                req_messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        payload = {
            "model": entry.name,
            "messages": req_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Local models may be slower; allow a configurable per-request timeout
        _req_timeout = (
            int(os.environ.get("APPLYPILOT_LOCAL_LLM_TIMEOUT", "120"))
            if entry.provider == "local"
            else None  # None = use httpx client default
        )
        for attempt in range(_MAX_RETRIES):
            try:
                post_kwargs: dict = {"json": payload, "headers": headers}
                if _req_timeout is not None:
                    post_kwargs["timeout"] = _req_timeout
                resp = self._client.post(
                    f"{entry.base_url}/chat/completions",
                    **post_kwargs,
                )
                if resp.status_code == 402:
                    # Payment Required — mark as exhausted for a full day (free-tier)
                    log.warning("%s/%s payment required (402), marking exhausted for 24h", entry.provider, entry.name)
                    self._exhausted[entry.name] = time.time() + 86400  # 24h from now
                    return None

                if resp.status_code == 400:
                    body = resp.text.lower()
                    if "api_key_invalid" in body or "api key expired" in body:
                        log.warning("%s/%s API key invalid/expired, trying next", entry.provider, entry.name)
                        self._exhausted[entry.name] = time.time()
                        return None
                    # Any other 400 (content safety, model not found, malformed prompt)
                    # — don't mark exhausted (it's per-request, not a quota), just skip
                    if not is_last:
                        log.warning("%s/%s 400 Bad Request, trying next: %.120s", entry.provider, entry.name, resp.text)
                        return None

                if resp.status_code == 404:
                    log.warning("%s/%s model not found (404), trying next", entry.provider, entry.name)
                    self._exhausted[entry.name] = time.time()
                    return None

                if resp.status_code == 429:
                    body = resp.text.lower()
                    # Distinguish daily/quota exhaustion vs transient rate limits
                    if "resource has been exhausted" in body or "quota" in body:
                        log.warning(
                            "%s/%s hit quota limit (daily), marking exhausted for 24h", entry.provider, entry.name
                        )
                        self._exhausted[entry.name] = time.time() + 86400
                        return None

                    if "rate_limit" in body:
                        # Transient rate limit — mark briefly and try next
                        log.warning("%s/%s transient rate_limit, marking exhausted for 60s", entry.provider, entry.name)
                        self._exhausted[entry.name] = time.time() + 60
                        return None

                    if attempt < _MAX_RETRIES - 1:
                        wait = 2**attempt + 1
                        log.warning(
                            "%s/%s 429 (RPM), retry in %ds (%d/%d)",
                            entry.provider,
                            entry.name,
                            wait,
                            attempt + 1,
                            _MAX_RETRIES,
                        )
                        time.sleep(wait)
                        continue
                    elif not is_last:
                        log.warning("%s/%s still 429, trying next model", entry.provider, entry.name)
                        return None
                    else:
                        resp.raise_for_status()

                if resp.status_code == 503 and attempt < _MAX_RETRIES - 1:
                    wait = 2**attempt
                    log.warning("%s/%s 503, retry in %ds", entry.provider, entry.name, wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                # Guard against malformed responses (null body, null choices, null content)
                if not isinstance(data, dict) or not data.get("choices"):
                    if not is_last:
                        log.warning("%s/%s: malformed response (no choices), trying next", entry.provider, entry.name)
                        return None
                    raise RuntimeError(
                        f"Malformed response from {entry.provider}/{entry.name}: no choices in {type(data).__name__}"
                    )
                text = data["choices"][0]["message"]["content"]
                # 2026-08-25: an empty or whitespace-only string is HTTP-level
                # success but not a usable response -- a real local/qwen3:1.7b
                # call was observed returning "" with status 200 (see
                # ~/.applypilot/logs/local_qwen_output.log). Previously only
                # `text is None` was rejected, so an empty string silently
                # returned as if it were a real answer -- for score_job() that
                # meant `_parse_score_response("")` producing a fake
                # fit_score=0 "success" instead of a retryable failure. Reject
                # both here, at the source, same fail-loud shape as the
                # null-content branch below.
                if text is None or not text.strip():
                    # Model returned null/empty/blank content (refusal, tool_call, etc.)
                    if not is_last:
                        log.warning("%s/%s: null/empty content in response, trying next", entry.provider, entry.name)
                        return None
                    raise RuntimeError(
                        f"Null/empty content from {entry.provider}/{entry.name} "
                        f"(refusal: {data['choices'][0]['message'].get('refusal', 'none')})"
                    )

                if entry.name != self.model:
                    log.info("Used fallback %s/%s (primary: %s)", entry.provider, entry.name, self.model)
                if entry.provider == "local":
                    _log_local_qwen_response(entry, req_messages, text)
                return text

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = 2**attempt
                    log.warning("%s/%s timeout, retry in %ds", entry.provider, entry.name, wait)
                    time.sleep(wait)
                    continue
                if not is_last:
                    log.warning("%s/%s timeout after retries, trying next", entry.provider, entry.name)
                    return None
                raise

            except httpx.ConnectError as exc:
                # Nothing listening at entry.base_url -- most commonly a
                # locally-configured endpoint (APPLYPILOT_LOCAL_LLM_URL)
                # that isn't running. Previously uncaught here: ConnectError
                # would abort the WHOLE fallback chain instead of falling
                # through to the next provider, even when a perfectly good
                # cloud fallback (Gemini/OpenAI) was next in line. Same
                # retry-then-fall-through shape as the timeout branch above.
                hint = " -- is the local model endpoint running?" if entry.provider == "local" else ""
                if attempt < _MAX_RETRIES - 1:
                    wait = 2**attempt
                    log.warning(
                        "%s/%s connection failed (%s), retry in %ds%s", entry.provider, entry.name, exc, wait, hint
                    )
                    time.sleep(wait)
                    continue
                if not is_last:
                    log.warning(
                        "%s/%s connection failed after retries (%s), trying next%s",
                        entry.provider,
                        entry.name,
                        exc,
                        hint,
                    )
                    return None
                raise

        return None

    def _try_anthropic(
        self, entry: ModelEntry, messages: list[dict], temperature: float, max_tokens: int, is_last: bool = False
    ) -> str | None:
        """Try the Anthropic Messages API (different format from OpenAI)."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": entry.api_key,
            "anthropic-version": "2023-06-01",
        }

        # Convert OpenAI message format to Anthropic format
        # Extract system message if present
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                api_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        # Anthropic requires at least one user message
        if not api_messages:
            return None

        payload: dict = {
            "model": entry.name,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if system_text:
            payload["system"] = system_text
        if temperature > 0:
            payload["temperature"] = temperature

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.post(
                    f"{entry.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 429:
                    body = resp.text.lower()
                    if "rate_limit" in body or "quota" in body:
                        log.warning("anthropic/%s hit rate limit, trying next", entry.name)
                        self._exhausted[entry.name] = time.time()
                        return None

                    if attempt < _MAX_RETRIES - 1:
                        wait = 2**attempt + 1
                        log.warning(
                            "anthropic/%s 429, retry in %ds (%d/%d)", entry.name, wait, attempt + 1, _MAX_RETRIES
                        )
                        time.sleep(wait)
                        continue
                    elif not is_last:
                        return None
                    else:
                        resp.raise_for_status()

                if resp.status_code == 529 and attempt < _MAX_RETRIES - 1:
                    # Anthropic overloaded
                    wait = 2**attempt + 2
                    log.warning("anthropic/%s overloaded (529), retry in %ds", entry.name, wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Extract text from Anthropic response format
                text_parts = []
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                text = "\n".join(text_parts)

                if entry.name != self.model:
                    log.info("Used fallback anthropic/%s (primary: %s)", entry.name, self.model)
                return text

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = 2**attempt
                    log.warning("anthropic/%s timeout, retry in %ds", entry.name, wait)
                    time.sleep(wait)
                    continue
                if not is_last:
                    return None
                raise

        return None

    def _try_claude_cli(self, entry: ModelEntry, messages: list[dict], is_last: bool = False) -> str | None:
        """Try the Claude Code CLI in one-shot print mode.

        Uses the user's Max plan OAuth login (subprocess env strips
        ANTHROPIC_API_KEY so it can't override that auth with API billing —
        same guard as apply/launcher.py's agent subprocess). No tools, no
        MCP config: --system-prompt fully replaces the default system prompt
        so no coding-agent framing leaks into the response, and the job text
        goes over stdin (unbounded, unlike a command-line arg).
        """
        system_text = ""
        user_parts: list[str] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                user_parts.append(msg["content"])
        user_text = "\n\n".join(user_parts)
        if not user_text.strip():
            return None

        # Serialize claude_cli access (see the lock's docstring in __init__).
        # Re-check exhaustion after acquiring: another thread may have just
        # marked this entry exhausted while we were waiting for the lock, in
        # which case there's no point spawning a doomed subprocess too.
        with self._claude_cli_lock:
            exhausted_until = self._exhausted.get(entry.name)
            if exhausted_until is not None and exhausted_until > time.time():
                return None

            cmd = [
                entry.base_url,
                "--model",
                entry.name,
                "-p",
                "--output-format",
                "text",
                "--permission-mode",
                "dontAsk",
            ]
            if system_text:
                cmd += ["--system-prompt", system_text]

            env = os.environ.copy()
            env.pop("CLAUDECODE", None)
            env.pop("CLAUDE_CODE_ENTRYPOINT", None)
            env.pop("ANTHROPIC_API_KEY", None)

            try:
                proc = subprocess.run(
                    cmd,
                    input=user_text,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    timeout=180,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                log.warning("claude_cli/%s timed out after 180s", entry.name)
                if not is_last:
                    return None
                raise RuntimeError(f"claude_cli/{entry.name} timed out")

            combined = f"{proc.stdout}\n{proc.stderr}".lower()
            # 2026-08-20 incident: Claude Code printed "You've hit your session
            # limit - resets 4pm (America/New_York)" but ApplyPilot logged an
            # EMPTY error and then retried claude_cli for every remaining job.
            # Two separate gaps caused that: (a) "session limit" didn't match
            # any of the original three keywords, so this fell through to the
            # generic failure path instead of the backoff path below; (b) the
            # diagnostic further down only ever read proc.stderr, so even the
            # generic failure path showed nothing when the message was on
            # stdout (which -p --output-format text apparently uses for this
            # kind of rejection).
            matched_limit_text = any(
                phrase in combined for phrase in ("usage limit", "session limit", "rate limit", "overloaded")
            )
            # 2026-08-19 incident: a Max-plan 5-hour usage window at 100%
            # utilization (confirmed via ~/.claude.json's cachedUsageUtilization)
            # made claude_cli/sonnet fail 50/50 tailor calls with exit 1 and
            # completely empty stdout/stderr -- no text for the check above to
            # match, so every call fell through to the hard RuntimeError below
            # instead of backing off. Verified live on 2.1.234 that this specific
            # exit-1-empty-output signature is how a fast local usage-limit
            # rejection presents in -p mode (real failures/successes both produce
            # real text). Still a heuristic, not a certainty -- some other exit-1
            # cause could in principle also produce empty output, so this is
            # logged distinctly from a text-matched limit rather than folded in
            # silently.
            empty_output_failure = proc.returncode != 0 and not proc.stdout.strip() and not proc.stderr.strip()
            if matched_limit_text or empty_output_failure:
                if empty_output_failure and not matched_limit_text:
                    log.warning(
                        "claude_cli/%s failed with exit %d and empty output; treating as "
                        "exhausted (heuristic for Max-plan usage-limit fast-fail, not a "
                        "confirmed rate-limit message)",
                        entry.name,
                        proc.returncode,
                    )
                    self._exhaustion_reason[entry.name] = "empty_output_heuristic"
                else:
                    log.warning("claude_cli/%s hit usage/session/rate limit, marking exhausted for 30min", entry.name)
                    self._exhaustion_reason[entry.name] = "quota_text_match"
                self._exhausted[entry.name] = time.time() + 1800
                return None

            if proc.returncode != 0 or not proc.stdout.strip():
                # Prefer stderr (conventional error stream) but fall back to
                # stdout -- this specific CLI/mode has been observed putting
                # rejection messages on stdout instead. Safe to log either:
                # this is the failure path, so there's no successful model
                # response (which is the only place resume/job-description
                # content would appear) to leak here -- just Claude Code's
                # own short system message about why it declined.
                diag = (proc.stderr.strip() or proc.stdout.strip())[:300]
                log.warning("claude_cli/%s failed (exit %d): %s", entry.name, proc.returncode, diag)
                if not is_last:
                    return None
                raise RuntimeError(f"claude_cli/{entry.name} failed: {diag}")

            if entry.name != self.model:
                log.info("Used fallback claude_cli/%s (primary: %s)", entry.name, self.model)
            return proc.stdout.strip()

    def claude_cli_exhaustion_reason(self, name: str) -> str | None:
        """Return the last recorded exhaustion classification for a
        claude_cli fallback-chain entry ("quota_text_match" or
        "empty_output_heuristic"), or None if it isn't currently exhausted
        or was never exhausted. Read-only; does not affect chat() routing.
        """
        if name not in self._exhausted or self._exhausted[name] <= time.time():
            return None
        return self._exhaustion_reason.get(name)

    def has_available_model(self) -> bool:
        """True if at least one entry in this client's fallback chain is not
        currently marked exhausted. Read-only capacity probe (makes no
        network/subprocess calls) -- used by the continuous scheduler to
        judge whether upstream work is worth attempting when Claude itself
        is unavailable.
        """
        now = time.time()
        return any(self._exhausted.get(e.name, 0) <= now for e in self._fallback_chain)

    def has_cloud_available(self) -> bool:
        """True if at least one non-local, non-claude_cli entry is not exhausted.

        Used by the tailor stage to select a shorter compact prompt when the
        only available model is a local one (smaller context budget than cloud).
        """
        now = time.time()
        return any(
            e.provider not in ("local", "claude_cli") and self._exhausted.get(e.name, 0) <= now
            for e in self._fallback_chain
        )

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None
_quality_instance: LLMClient | None = None
_stage_instances: dict[str, LLMClient] = {}


def _stage_model_override(stage: str) -> str | None:
    value = os.environ.get(f"APPLYPILOT_MODEL_{stage.upper()}", "").strip()
    return value or None


def get_token_limit(stage: str, default: int) -> int:
    """Read stage-specific max token override with safe fallback.

    Env key format: APPLYPILOT_MAX_TOKENS_<STAGE> (e.g. SCORE, TAILOR, COVER, JUDGE).
    """
    raw = os.environ.get(f"APPLYPILOT_MAX_TOKENS_{stage.upper()}", "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        log.warning("Invalid %s=%r; using default %d", f"APPLYPILOT_MAX_TOKENS_{stage.upper()}", raw, default)
        return default


def get_client(quality: bool = False, model_override: str | None = None) -> LLMClient:
    """Return (or create) the module-level LLMClient singleton.

    Args:
        quality: If True, return a client configured with LLM_MODEL_QUALITY
                 for critical steps like resume tailoring and cover letters.
        model_override: Optional explicit model name for this client.
    """
    global _instance, _quality_instance

    if model_override:
        key = ("quality" if quality else "fast") + ":" + model_override
        client = _stage_instances.get(key)
        if client is None:
            base_url, model, api_key = _detect_provider(quality=quality, model_override=model_override)
            log.info("LLM provider (%s override): %s  model: %s", "quality" if quality else "fast", base_url, model)
            client = LLMClient(base_url, model, api_key, quality=quality)
            _stage_instances[key] = client
        return client

    if quality:
        if _quality_instance is None:
            # Always construct a quality client when requested. The underlying
            # _detect_provider will honor LLM_MODEL_QUALITY if set, otherwise
            # fall back to sane defaults (gemini-3.6-flash by default).
            base_url, model, api_key = _detect_provider(quality=True)
            log.info("LLM quality provider: %s  model: %s", base_url, model)
            _quality_instance = LLMClient(base_url, model, api_key, quality=True)
        return _quality_instance

    if _instance is None:
        base_url, model, api_key = _detect_provider()
        log.info("LLM provider: %s  model: %s", base_url, model)
        _instance = LLMClient(base_url, model, api_key, quality=False)
    return _instance


def get_stage_client(stage: str, *, quality: bool) -> LLMClient:
    """Return a client for a pipeline stage, honoring per-stage model overrides."""
    return get_client(quality=quality, model_override=_stage_model_override(stage))


def is_local_configured() -> bool:
    """Return True if APPLYPILOT_LOCAL_LLM_URL is set (local fallback is enabled)."""
    return bool(os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "").strip())


def local_available() -> bool:
    """Probe whether the configured local LLM endpoint is reachable.

    Tries the common Ollama / llama.cpp paths. Returns False immediately when
    no URL is configured. Safe to call frequently: 4-second timeout, no retries.
    """
    url = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "").rstrip("/")
    if not url:
        return False
    for path in ("/v1/models", "/api/tags", "/models"):
        try:
            resp = httpx.get(f"{url}{path}", timeout=4.0)
            if resp.status_code < 500:
                return True
        except Exception:  # noqa: BLE001, S112 - trying multiple candidate endpoint paths (Ollama/llama.cpp conventions differ); one path failing means try the next, not abort the probe
            continue
    return False


def ask_local(prompt: str, system: str | None = None, temperature: float = 0.0, max_tokens: int = 512) -> str | None:
    """Best-effort local Ollama call for non-tailoring callers (2026-08-31:
    discovery/enrichment resilience audit) that want a last-resort fallback
    when the cloud cascade in `chat()`/`ask()` is exhausted, without pulling
    in the scoring-specific machinery in local_tailor.py.

    Mirrors local_tailor.get_local_tailoring_plan()'s proven Ollama-native
    transport (same env vars, same /api/chat shape, think=false, same
    /v1-stripping normalization) but generalized to a plain prompt/system
    pair -- callers own their own response parsing (JSON-fence stripping,
    schema validation, etc.), same as they already do for the cloud path.

    Returns None on ANY failure: not configured, unreachable, timeout,
    non-2xx, or an empty response. Never raises. Callers must already treat
    None the same as "no result" -- this is explicitly a last-resort,
    never-authoritative addition, never a second source of truth.
    """
    if not is_local_configured():
        return None
    url = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "").rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    model = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", "llama3.2")
    timeout = float(os.environ.get("APPLYPILOT_LOCAL_LLM_TIMEOUT", "60"))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "think": False,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    try:
        resp = httpx.post(f"{url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = ((data.get("message") or {}).get("content") or "").strip()
        if not text:
            log.warning("Local LLM fallback (model=%s): empty response content", model)
            return None
        return text
    except httpx.TimeoutException as exc:
        log.warning("Local LLM fallback (model=%s) timed out after %.0fs: %s", model, timeout, exc)
        return None
    except httpx.ConnectError as exc:
        log.warning("Local LLM fallback: could not reach %s (%s)", url, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - best-effort last-resort fallback; any other failure just means "no local result", never propagates
        log.warning("Local LLM fallback failed: %s: %s", type(exc).__name__, exc)
        return None
