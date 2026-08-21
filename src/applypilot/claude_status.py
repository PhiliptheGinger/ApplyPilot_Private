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
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc
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
    reset_source: str | None       # "cached_usage_state" or None
    cache_age_seconds: float | None
    binding_window: str | None     # "five_hour" | "seven_day" | None
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
# ---------------------------------------------------------------------------

_apply_signal_lock = threading.Lock()
_apply_exhausted_until: float | None = None
_apply_exhaustion_reason: str | None = None


def record_apply_exhaustion(reason: str, cooldown_seconds: float = 1800) -> None:
    """Record that the apply-side Claude Code CLI just showed a real,
    observed exhaustion signal (not derived from the cache). `reason` is a
    short machine tag, e.g. "session_limit" or "empty_output".
    """
    global _apply_exhausted_until, _apply_exhaustion_reason
    with _apply_signal_lock:
        _apply_exhausted_until = time.time() + cooldown_seconds
        _apply_exhaustion_reason = reason
    log.info("Apply-side Claude signal: exhausted (reason=%s, cooldown=%ss)",
             reason, cooldown_seconds)


def record_apply_success() -> None:
    """Clear any recorded exhaustion -- a run_job invocation just completed
    without hitting a quota/session-limit condition, proving Claude is
    currently usable regardless of what the cache says.
    """
    global _apply_exhausted_until, _apply_exhaustion_reason
    was_exhausted = _apply_exhausted_until is not None and _apply_exhausted_until > time.time()
    with _apply_signal_lock:
        _apply_exhausted_until = None
        _apply_exhaustion_reason = None
    if was_exhausted:
        log.info("Apply-side Claude signal: cleared (usable again)")


def _apply_signal() -> tuple[bool, str | None]:
    with _apply_signal_lock:
        if _apply_exhausted_until is not None and _apply_exhausted_until > time.time():
            return True, _apply_exhaustion_reason
        return False, None


def reset_apply_signal_for_tests() -> None:
    """Test-only helper: force the module back to a clean AVAILABLE state."""
    global _apply_exhausted_until, _apply_exhaustion_reason
    with _apply_signal_lock:
        _apply_exhausted_until = None
        _apply_exhaustion_reason = None


# ---------------------------------------------------------------------------
# `claude auth status` -- corroborating signal only, rate-limited
# ---------------------------------------------------------------------------

_auth_check_lock = threading.Lock()
_auth_check_cache: dict = {"at": 0.0, "logged_in": None}


def _run_claude_auth_status() -> bool | None:
    try:
        proc = subprocess.run(
            ["claude", "auth", "status", "--output-format", "json"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
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
    observed_exhausted, observed_reason = (
        observed if observed is not None else _apply_signal()
    )

    if check_auth:
        logged_in = _get_auth_status(auth_min_interval_seconds, runner=auth_runner)
        if logged_in is False:
            return ClaudeAvailability(
                state=AUTH_FAILURE, reset_estimate=None, reset_source=None,
                cache_age_seconds=None, binding_window=None,
                detail="claude auth status reports loggedIn=false",
            )

    if not observed_exhausted:
        return ClaudeAvailability(
            state=AVAILABLE, reset_estimate=None, reset_source=None,
            cache_age_seconds=None, binding_window=None,
            detail="no observed exhaustion",
        )

    if observed_reason in _AUTH_REASON_TAGS:
        return ClaudeAvailability(
            state=AUTH_FAILURE, reset_estimate=None, reset_source=None,
            cache_age_seconds=None, binding_window=None,
            detail=f"observed auth failure: {observed_reason}",
        )
    if observed_reason in _TRANSIENT_REASON_TAGS:
        return ClaudeAvailability(
            state=TRANSIENT_ERROR, reset_estimate=None, reset_source=None,
            cache_age_seconds=None, binding_window=None,
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
            state=EXHAUSTED_KNOWN_RESET, reset_estimate=resets_at,
            reset_source="cached_usage_state", cache_age_seconds=cache_age,
            binding_window=window_name,
            detail=f"exhausted ({observed_reason}); reset estimated from {window_name} cache",
        )

    stale_note = ""
    if cached is not None and cache_age is not None and cache_age > cache_max_age_seconds:
        stale_note = f"; cached usage state stale (age: {cache_age / 3600:.1f}h)"
    elif cached is None:
        stale_note = "; no cache available"
    return ClaudeAvailability(
        state=EXHAUSTED_UNKNOWN_RESET, reset_estimate=None, reset_source=None,
        cache_age_seconds=cache_age, binding_window=None,
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
        with self._lock:
            return self._paused

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def reset_for_tests(self) -> None:
        with self._lock:
            self._paused = False
            self._state = AVAILABLE


gate = ClaudeGate()
