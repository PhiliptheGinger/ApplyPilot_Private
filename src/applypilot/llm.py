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
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

# Stage-specific knobs (all optional):
#   APPLYPILOT_MODEL_SCORE, APPLYPILOT_MODEL_TAILOR, APPLYPILOT_MODEL_COVER,
#   APPLYPILOT_MODEL_JUDGE
#   APPLYPILOT_MAX_TOKENS_SCORE, APPLYPILOT_MAX_TOKENS_TAILOR,
#   APPLYPILOT_MAX_TOKENS_COVER, APPLYPILOT_MAX_TOKENS_JUDGE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — each entry knows its provider, endpoint, and API key
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    """A model with everything needed to call it."""
    name: str
    provider: str           # "gemini", "openai", "anthropic", "claude_cli", "local"
    base_url: str            # for claude_cli: the resolved path to the claude binary
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


def _build_fallback_chain(primary_model: str, quality: bool = False) -> list[ModelEntry]:
    """Build a cross-provider fallback chain starting from the primary model.

    Gemini models come first (free tier), then OpenAI (cheap), then Anthropic.
    Only includes providers whose API keys are configured.
    """
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

    # Start from the primary model in the Gemini chain
    if gemini_key:
        started = False
        for m in gemini_models:
            if m == primary_model:
                started = True
            if started:
                chain.append(ModelEntry(m, "gemini", gemini_url, gemini_key))
        # If primary wasn't found in chain, add full chain
        if not started:
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
    if claude_cli_path:
        cli_model = "sonnet" if quality else "haiku"
        chain.append(ModelEntry(cli_model, "claude_cli", claude_cli_path, ""))

    # Anthropic fallbacks
    if anthropic_key:
        for m in anthropic_models:
            chain.append(ModelEntry(m, "anthropic", anthropic_url, anthropic_key))

    # If nothing was added (no keys), raise
    if not chain:
        raise RuntimeError(
            "No LLM provider configured. "
            "Set GEMINI_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, or ANTHROPIC_API_KEY."
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
    raise RuntimeError(
        "No LLM provider configured. "
        "Set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_TIMEOUT = 300  # seconds


class LLMClient:
    """Multi-provider LLM client with automatic model fallback."""

    def __init__(self, base_url: str, model: str, api_key: str,
                 quality: bool = False) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.quality = quality
        self._fallback_chain = _build_fallback_chain(model, quality=quality)
        self._client = httpx.Client(timeout=_TIMEOUT)
        # Track which models are temporarily exhausted (store until timestamp)
        self._exhausted: dict[str, float] = {}

        chain_names = [f"{e.name} ({e.provider})" for e in self._fallback_chain]
        log.info("Fallback chain (%s): %s",
                 "quality" if quality else "fast", " -> ".join(chain_names))

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request with automatic cross-provider fallback."""
        # Qwen3 optimization
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

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

        entries_to_try = [e for e in self._fallback_chain if not _is_exhausted(e.name)]
        if not entries_to_try:
            self._exhausted.clear()
            entries_to_try = list(self._fallback_chain)

        for idx, entry in enumerate(entries_to_try):
            is_last = (idx == len(entries_to_try) - 1)
            result = self._try_entry(entry, messages, temperature, max_tokens, is_last)
            if result is not None:
                return result

        raise RuntimeError(
            f"All models exhausted after trying: "
            f"{[e.name for e in entries_to_try]}. "
            "Wait a few minutes for rate limits to reset."
        )

    def _try_entry(self, entry: ModelEntry, messages: list[dict],
                   temperature: float, max_tokens: int,
                   is_last: bool = False) -> str | None:
        """Try a single model entry. Dispatches to the right provider."""
        if entry.provider == "anthropic":
            return self._try_anthropic(entry, messages, temperature, max_tokens, is_last)
        elif entry.provider == "claude_cli":
            return self._try_claude_cli(entry, messages, is_last)
        else:
            return self._try_openai_compat(entry, messages, temperature, max_tokens, is_last)

    def _try_openai_compat(self, entry: ModelEntry, messages: list[dict],
                           temperature: float, max_tokens: int,
                           is_last: bool = False) -> str | None:
        """Try an OpenAI-compatible endpoint (Gemini, OpenAI, local)."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {entry.api_key}",
        }
        # DeepSeek deepseek-chat has an 8192 max output token limit
        if entry.provider == "deepseek":
            max_tokens = min(max_tokens, 8192)
        payload = {
            "model": entry.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.post(
                    f"{entry.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 402:
                    # Payment Required — mark as exhausted for a full day (free-tier)
                    log.warning("%s/%s payment required (402), marking exhausted for 24h",
                                entry.provider, entry.name)
                    self._exhausted[entry.name] = time.time() + 86400  # 24h from now
                    return None

                if resp.status_code == 400:
                    body = resp.text.lower()
                    if "api_key_invalid" in body or "api key expired" in body:
                        log.warning("%s/%s API key invalid/expired, trying next",
                                    entry.provider, entry.name)
                        self._exhausted[entry.name] = time.time()
                        return None
                    # Any other 400 (content safety, model not found, malformed prompt)
                    # — don't mark exhausted (it's per-request, not a quota), just skip
                    if not is_last:
                        log.warning("%s/%s 400 Bad Request, trying next: %.120s",
                                    entry.provider, entry.name, resp.text)
                        return None

                if resp.status_code == 404:
                    log.warning("%s/%s model not found (404), trying next",
                                entry.provider, entry.name)
                    self._exhausted[entry.name] = time.time()
                    return None

                if resp.status_code == 429:
                    body = resp.text.lower()
                    # Distinguish daily/quota exhaustion vs transient rate limits
                    if "resource has been exhausted" in body or "quota" in body:
                        log.warning("%s/%s hit quota limit (daily), marking exhausted for 24h",
                                    entry.provider, entry.name)
                        self._exhausted[entry.name] = time.time() + 86400
                        return None

                    if "rate_limit" in body:
                        # Transient rate limit — mark briefly and try next
                        log.warning("%s/%s transient rate_limit, marking exhausted for 60s",
                                    entry.provider, entry.name)
                        self._exhausted[entry.name] = time.time() + 60
                        return None

                    if attempt < _MAX_RETRIES - 1:
                        wait = 2 ** attempt + 1
                        log.warning("%s/%s 429 (RPM), retry in %ds (%d/%d)",
                                    entry.provider, entry.name, wait,
                                    attempt + 1, _MAX_RETRIES)
                        time.sleep(wait)
                        continue
                    elif not is_last:
                        log.warning("%s/%s still 429, trying next model",
                                    entry.provider, entry.name)
                        return None
                    else:
                        resp.raise_for_status()

                if resp.status_code == 503 and attempt < _MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    log.warning("%s/%s 503, retry in %ds", entry.provider, entry.name, wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                # Guard against malformed responses (null body, null choices, null content)
                if not isinstance(data, dict) or not data.get("choices"):
                    if not is_last:
                        log.warning("%s/%s: malformed response (no choices), trying next",
                                    entry.provider, entry.name)
                        return None
                    raise RuntimeError(
                        f"Malformed response from {entry.provider}/{entry.name}: "
                        f"no choices in {type(data).__name__}"
                    )
                text = data["choices"][0]["message"]["content"]
                if text is None:
                    # Model returned null content (refusal, tool_call, etc.)
                    if not is_last:
                        log.warning("%s/%s: null content in response, trying next",
                                    entry.provider, entry.name)
                        return None
                    raise RuntimeError(
                        f"Null content from {entry.provider}/{entry.name} "
                        f"(refusal: {data['choices'][0]['message'].get('refusal', 'none')})"
                    )

                if entry.name != self.model:
                    log.info("Used fallback %s/%s (primary: %s)",
                             entry.provider, entry.name, self.model)
                return text

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    log.warning("%s/%s timeout, retry in %ds",
                                entry.provider, entry.name, wait)
                    time.sleep(wait)
                    continue
                if not is_last:
                    log.warning("%s/%s timeout after retries, trying next",
                                entry.provider, entry.name)
                    return None
                raise

        return None

    def _try_anthropic(self, entry: ModelEntry, messages: list[dict],
                       temperature: float, max_tokens: int,
                       is_last: bool = False) -> str | None:
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
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

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
                        wait = 2 ** attempt + 1
                        log.warning("anthropic/%s 429, retry in %ds (%d/%d)",
                                    entry.name, wait, attempt + 1, _MAX_RETRIES)
                        time.sleep(wait)
                        continue
                    elif not is_last:
                        return None
                    else:
                        resp.raise_for_status()

                if resp.status_code == 529 and attempt < _MAX_RETRIES - 1:
                    # Anthropic overloaded
                    wait = 2 ** attempt + 2
                    log.warning("anthropic/%s overloaded (529), retry in %ds",
                                entry.name, wait)
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
                    log.info("Used fallback anthropic/%s (primary: %s)",
                             entry.name, self.model)
                return text

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    log.warning("anthropic/%s timeout, retry in %ds",
                                entry.name, wait)
                    time.sleep(wait)
                    continue
                if not is_last:
                    return None
                raise

        return None

    def _try_claude_cli(self, entry: ModelEntry, messages: list[dict],
                        is_last: bool = False) -> str | None:
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

        cmd = [entry.base_url, "--model", entry.name, "-p",
               "--output-format", "text", "--permission-mode", "dontAsk"]
        if system_text:
            cmd += ["--system-prompt", system_text]

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        env.pop("ANTHROPIC_API_KEY", None)

        try:
            proc = subprocess.run(
                cmd, input=user_text, capture_output=True, text=True,
                encoding="utf-8", errors="replace", env=env, timeout=180,
            )
        except subprocess.TimeoutExpired:
            log.warning("claude_cli/%s timed out after 180s", entry.name)
            if not is_last:
                return None
            raise RuntimeError(f"claude_cli/{entry.name} timed out")

        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        matched_limit_text = "usage limit" in combined or "rate limit" in combined or "overloaded" in combined
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
        empty_output_failure = (
            proc.returncode != 0 and not proc.stdout.strip() and not proc.stderr.strip()
        )
        if matched_limit_text or empty_output_failure:
            if empty_output_failure and not matched_limit_text:
                log.warning(
                    "claude_cli/%s failed with exit %d and empty output; treating as "
                    "exhausted (heuristic for Max-plan usage-limit fast-fail, not a "
                    "confirmed rate-limit message)",
                    entry.name, proc.returncode,
                )
            else:
                log.warning("claude_cli/%s hit usage/rate limit, marking exhausted for 30min", entry.name)
            self._exhausted[entry.name] = time.time() + 1800
            return None

        if proc.returncode != 0 or not proc.stdout.strip():
            log.warning("claude_cli/%s failed (exit %d): %s",
                        entry.name, proc.returncode, proc.stderr[:300])
            if not is_last:
                return None
            raise RuntimeError(f"claude_cli/{entry.name} failed: {proc.stderr[:300]}")

        if entry.name != self.model:
            log.info("Used fallback claude_cli/%s (primary: %s)", entry.name, self.model)
        return proc.stdout.strip()

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
        log.warning("Invalid %s=%r; using default %d",
                    f"APPLYPILOT_MAX_TOKENS_{stage.upper()}", raw, default)
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
            log.info("LLM provider (%s override): %s  model: %s",
                     "quality" if quality else "fast", base_url, model)
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
