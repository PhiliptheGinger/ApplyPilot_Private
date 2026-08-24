"""Claude Code CLI availability state machine for the continuous scheduler.

Two independent Claude Code CLI invocation paths exist in this codebase:
llm.py's `_try_claude_cli` (one-shot fallback for score/tailor/cover, reserved
away from the default fallback chain per APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY)
and apply/launcher.py's `run_job` (the real agentic auto-apply session). This
module tracks availability for the latter -- the one the continuous scheduler
actually needs to gate apply on -- via a simple in-process signal that
`run_job` reports into (`record_apply_exhaustion` / `record_apply_success`).

~/.claude.json's undocumented `cachedUsageUtilization` block is NEVER used to
invent availability or unavailability on its own. It is strictly a best-effort
hint for *when* an already-observed exhaustion might clear. Confirmed live on
this machine: a cache can be 32+ hours stale with a `resets_at` timestamp
already in the past -- reading that naively would report a reset time that
already happened. Every read path here treats absence, staleness, malformed
data, and past timestamps identically: "no usable reset estimate."
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

AVAILABLE = "AVAILABLE"
EXHAUSTED_KNOWN_RESET = "EXHAUSTED_KNOWN_RESET"
EXHAUSTED_UNKNOWN_RESET = "EXHAUSTED_UNKNOWN_RESET"
AUTH_FAILURE = "AUTH_FAILURE"
TRANSIENT_ERROR = "TRANSIENT_ERROR"

_USAGE_WINDOWS = ("five_hour", "seven_day")


@dataclass
class ClaudeAvailability:
    state: str
    reset_estimate: datetime | None
    reset_source: str | None  # "cached_usage_state" or None
    cache_age_seconds: float | None
    binding_window: str | None  # "five_hour" | "seven_day" | None
    detail: str


# ---------------------------------------------------------------------------
# ~/.claude.json best-effort cache reading (never authoritative)
# ---------------------------------------------------------------------------


def _default_cache_path() -> Path:
    return Path.home() / ".claude.json"


def _parse_iso(value) -> datetime | None:
    """Parse an ISO-8601 timestamp, returning None on anything unusable."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def read_cached_usage_state(cache_path: Path | None = None) -> tuple[dict | None, float | None]:
    """Return (cachedUsageUtilization dict, age_seconds) or (None, None).

    Missing file, missing key, unreadable/malformed JSON, or a missing/
    malformed/future-skewed fetchedAtMs all collapse to a usable-but-ageless
    dict (age None) or fully absent (None, None) -- callers must treat a
    None age as "freshness unknown, therefore not usable."
    """
    path = cache_path or _default_cache_path()
    try:
        if not path.exists():
            return None, None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None, None

    if not isinstance(raw, dict):
        return None, None
    cached = raw.get("cachedUsageUtilization")
    if not isinstance(cached, dict):
        return None, None

    fetched_ms = cached.get("fetchedAtMs")
    if not isinstance(fetched_ms, (int, float)):
        return cached, None

    age = time.time() - (fetched_ms / 1000.0)
    if age < 0:
        # Clock skew or a bogus future timestamp -- can't trust freshness.
        return cached, None
    return cached, age


def binding_window(cached: dict, now: datetime) -> tuple[str, datetime] | None:
    """Return (window_name, resets_at) for whichever usage window is both
    saturated (>=100%) AND has a valid FUTURE resets_at -- i.e. the window
    that is actually blocking Claude right now, not just whichever
    resets_at happens to be numerically smallest. If both five_hour and
    seven_day qualify, the soonest future reset among the qualifying ones
    wins. Returns None if neither window qualifies.
    """
    utilization = cached.get("utilization")
    if not isinstance(utilization, dict):
        return None

    candidates: list[tuple[str, datetime]] = []
    for window_name in _USAGE_WINDOWS:
        window = utilization.get(window_name)
        if not isinstance(window, dict):
            continue
        pct = window.get("utilization")
        if not isinstance(pct, (int, float)) or pct < 100:
            continue
        resets_at = _parse_iso(window.get("resets_at"))
        if resets_at is None or resets_at <= now:
            continue  # missing, malformed, or already in the past
        candidates.append((window_name, resets_at))

    if not candidates:
        return None
    return min(candidates, key=lambda c: c[1])


# ---------------------------------------------------------------------------
# Observed-exhaustion signal, reported by apply/launcher.py::run_job
#
# 2026-08-21 fix: this was previously a single timestamp
# (_apply_exhausted_until) that played two incompatible roles at once --
# "is Claude currently believed exhausted" AND "when has the cooldown
# expired." Once time.time() passed that timestamp, _apply_signal() would
# report (False, None) -- i.e. AVAILABLE -- purely because a clock ticked
# past a fixed 30-minute mark, with no actual evidence Claude recovered.
# Worse, that silent flip could fire well before a scheduler-computed
# EXHAUSTED_KNOWN_RESET estimate (e.g. a real 3-hour session limit),
# undermining the whole point of tracking a reset estimate.
#
# Fixed by splitting the two concerns: `_apply_exhausted`/
# `_apply_exhaustion_reason` are now durable -- they change ONLY on an
# explicit record_apply_exhaustion()/record_apply_success() call, never
# from time passing. `_apply_probe_after` is purely a "when is it worth
# trying again" hint, consulted only by probe_due() (used by ClaudeGate to
# let periodic real attempts through even while nominally paused -- since
# an actual successful Claude call is the ONLY thing that can ever clear
# the durable flag, something has to be allowed to attempt one).
# ---------------------------------------------------------------------------

_apply_signal_lock = threading.Lock()
_apply_exhausted: bool = False
_apply_exhaustion_reason: str | None = None
_apply_probe_after: float | None = None


def record_apply_exhaustion(reason: str, cooldown_seconds: float = 1800) -> None:
    """Record that the apply-side Claude Code CLI just showed a real,
    observed exhaustion signal (not derived from the cache). `reason` is a
    short machine tag, e.g. "session_limit" or "empty_output".

    This is durable: it does NOT auto-clear after `cooldown_seconds`. That
    window only controls when `probe_due()` starts allowing a real retry
    attempt -- recovery itself is only ever established by a subsequent
    record_apply_success() call (see probe_due()'s docstring).
    """
    global _apply_exhausted, _apply_exhaustion_reason, _apply_probe_after
    with _apply_signal_lock:
        _apply_exhausted = True
        _apply_exhaustion_reason = reason
        _apply_probe_after = time.time() + cooldown_seconds
    log.info("Apply-side Claude signal: exhausted (reason=%s, next probe eligible in %ss)", reason, cooldown_seconds)


def record_apply_success() -> None:
    """Clear any recorded exhaustion -- a run_job invocation just completed
    without hitting a quota/session-limit condition, proving Claude is
    currently usable regardless of what the cache says. This is the ONLY
    way the durable exhausted flag is ever cleared.
    """
    global _apply_exhausted, _apply_exhaustion_reason, _apply_probe_after
    with _apply_signal_lock:
        was_exhausted = _apply_exhausted
        _apply_exhausted = False
        _apply_exhaustion_reason = None
        _apply_probe_after = None
    if was_exhausted:
        log.info("Apply-side Claude signal: cleared (usable again -- confirmed by a successful call)")


def _apply_signal() -> tuple[bool, str | None]:
    """Durable exhaustion belief -- does NOT expire with the passage of
    time. See probe_due() for whether it's time to attempt a real retry.
    """
    with _apply_signal_lock:
        return _apply_exhausted, _apply_exhaustion_reason


def probe_due() -> bool:
    """True if Claude is currently believed exhausted AND enough time has
    passed that a real attempt is worth making.

    This is the ONLY mechanism that can lead to recovery -- unlike the
    pre-fix behavior, reaching this point never itself flips any state to
    available. It just tells the apply gate (ClaudeGate.is_paused) to let
    exactly the next real attempt through. That attempt's own outcome
    (record_apply_success or another record_apply_exhaustion) is what
    actually changes the durable belief.
    """
    with _apply_signal_lock:
        if not _apply_exhausted or _apply_probe_after is None:
            return False
        return time.time() >= _apply_probe_after


def reset_apply_signal_for_tests() -> None:
    """Test-only helper: force the module back to a clean AVAILABLE state."""
    global _apply_exhausted, _apply_exhaustion_reason, _apply_probe_after
    with _apply_signal_lock:
        _apply_exhausted = False
        _apply_exhaustion_reason = None
        _apply_probe_after = None


# ---------------------------------------------------------------------------
# `claude auth status` -- corroborating signal only, rate-limited
# ---------------------------------------------------------------------------

_auth_check_lock = threading.Lock()
_auth_check_cache: dict = {"at": 0.0, "logged_in": None}


def _run_claude_auth_status() -> bool | None:
    try:
        proc = subprocess.run(
            ["claude", "auth", "status", "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:  # noqa: BLE001 - claude CLI subprocess probe is inherently unreliable; degrade to "unknown" (None), not a crash
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return None
    if "loggedIn" not in data:
        return None
    return bool(data["loggedIn"])


def _get_auth_status(min_interval_seconds: float = 300, runner=None) -> bool | None:
    """Cached, rate-limited `claude auth status` check. Returns True/False
    if freshly (or cache-recently) known, None if never successfully
    checked -- None must NEVER be treated as a failure, only as "unknown."
    """
    global _auth_check_cache
    now = time.time()
    with _auth_check_lock:
        if _auth_check_cache["at"] > 0 and (now - _auth_check_cache["at"]) < min_interval_seconds:
            return _auth_check_cache["logged_in"]

    check = runner or _run_claude_auth_status
    logged_in = check()
    with _auth_check_lock:
        _auth_check_cache = {"at": now, "logged_in": logged_in}
    return logged_in


def reset_auth_cache_for_tests() -> None:
    global _auth_check_cache
    with _auth_check_lock:
        _auth_check_cache = {"at": 0.0, "logged_in": None}


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------

_TRANSIENT_REASON_TAGS = {"transient_error"}
_AUTH_REASON_TAGS = {"auth_failure"}


def check_claude_availability(
    *,
    cache_path: Path | None = None,
    cache_max_age_seconds: float = 600,
    now: datetime | None = None,
    observed: tuple[bool, str | None] | None = None,
    check_auth: bool = True,
    auth_min_interval_seconds: float = 300,
    auth_runner=None,
) -> ClaudeAvailability:
    """Determine current Claude availability for the apply worker.

    `observed` overrides the module-level apply signal (mainly for tests);
    when None, the real signal recorded by run_job via
    `record_apply_exhaustion`/`record_apply_success` is used.
    """
    now = now or datetime.now(UTC)
    observed_exhausted, observed_reason = observed if observed is not None else _apply_signal()

    if check_auth:
        logged_in = _get_auth_status(auth_min_interval_seconds, runner=auth_runner)
        if logged_in is False:
            return ClaudeAvailability(
                state=AUTH_FAILURE,
                reset_estimate=None,
                reset_source=None,
                cache_age_seconds=None,
                binding_window=None,
                detail="claude auth status reports loggedIn=false",
            )

    if not observed_exhausted:
        return ClaudeAvailability(
            state=AVAILABLE,
            reset_estimate=None,
            reset_source=None,
            cache_age_seconds=None,
            binding_window=None,
            detail="no observed exhaustion",
        )

    if observed_reason in _AUTH_REASON_TAGS:
        return ClaudeAvailability(
            state=AUTH_FAILURE,
            reset_estimate=None,
            reset_source=None,
            cache_age_seconds=None,
            binding_window=None,
            detail=f"observed auth failure: {observed_reason}",
        )
    if observed_reason in _TRANSIENT_REASON_TAGS:
        return ClaudeAvailability(
            state=TRANSIENT_ERROR,
            reset_estimate=None,
            reset_source=None,
            cache_age_seconds=None,
            binding_window=None,
            detail=f"observed transient error: {observed_reason}",
        )

    # Real exhaustion observed (session_limit / empty_output / etc). Try to
    # get a best-effort reset estimate from a fresh, valid cache -- never
    # fabricate one from stale or missing data.
    cached, cache_age = read_cached_usage_state(cache_path)
    binding = None
    if cached is not None and cache_age is not None and cache_age <= cache_max_age_seconds:
        binding = binding_window(cached, now)

    if binding is not None:
        window_name, resets_at = binding
        return ClaudeAvailability(
            state=EXHAUSTED_KNOWN_RESET,
            reset_estimate=resets_at,
            reset_source="cached_usage_state",
            cache_age_seconds=cache_age,
            binding_window=window_name,
            detail=f"exhausted ({observed_reason}); reset estimated from {window_name} cache",
        )

    stale_note = ""
    if cached is not None and cache_age is not None and cache_age > cache_max_age_seconds:
        stale_note = f"; cached usage state stale (age: {cache_age / 3600:.1f}h)"
    elif cached is None:
        stale_note = "; no cache available"
    return ClaudeAvailability(
        state=EXHAUSTED_UNKNOWN_RESET,
        reset_estimate=None,
        reset_source=None,
        cache_age_seconds=cache_age,
        binding_window=None,
        detail=f"exhausted ({observed_reason}), no usable reset estimate{stale_note}",
    )


# ---------------------------------------------------------------------------
# In-process gate consulted by apply/orchestrator.py's continuous worker loop
# ---------------------------------------------------------------------------


class ClaudeGate:
    """Defaults to never-paused, so standalone `applypilot apply --continuous`
    (no scheduler running) is completely unaffected. Only a running
    `applypilot run-continuous` scheduler ever calls `.set(...)`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused = False
        self._state = AVAILABLE

    def set(self, availability: ClaudeAvailability) -> None:
        with self._lock:
            self._paused = availability.state != AVAILABLE
            self._state = availability.state

    def is_paused(self) -> bool:
        """False (proceed) when genuinely available, OR when a probe is
        due -- exhaustion is durable (see probe_due's docstring), so
        without this a scheduler-gated continuous worker would block
        acquire_job() forever once exhausted, since nothing would ever be
        allowed to make the real attempt that could prove recovery.
        """
        with self._lock:
            paused = self._paused
        if not paused:
            return False
        return not probe_due()

    def is_probe_attempt(self) -> bool:
        """True if the gate is currently letting an attempt through
        specifically BECAUSE a probe is due -- Claude is still durably
        believed exhausted, and this attempt is the one that gets to
        establish whether it has actually recovered. False when genuinely
        available (nothing to probe) or when still paused (is_paused()
        would be True and nothing should be attempted at all).

        Callers that proceed after `is_paused()` returns False should
        check this too and carry the result explicitly into the actual
        attempt (rather than re-deriving it later), so a probe that fails
        via a generic/timeout path -- which does not itself produce a
        billing/session-limit signal to re-record -- knows to re-arm the
        probe window instead of silently leaving it expired.
        """
        with self._lock:
            paused = self._paused
        return paused and probe_due()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def reset_for_tests(self) -> None:
        with self._lock:
            self._paused = False
            self._state = AVAILABLE


gate = ClaudeGate()
