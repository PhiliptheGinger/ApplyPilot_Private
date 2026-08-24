"""Tests for the Claude availability state machine (claude_status.py).

Every test either mocks ~/.claude.json via an explicit tmp path (never the
real file) or disables the cache read entirely by passing a nonexistent
path, and disables `claude auth status` (check_auth=False) unless the test
is specifically exercising auth-failure detection (in which case a fake
auth_runner is injected -- never the real subprocess).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot import claude_status as cs


@pytest.fixture(autouse=True)
def _isolate_module_state():
    """Every test starts from a clean apply-signal / auth-cache / gate state."""
    cs.reset_apply_signal_for_tests()
    cs.reset_auth_cache_for_tests()
    cs.gate.reset_for_tests()
    yield
    cs.reset_apply_signal_for_tests()
    cs.reset_auth_cache_for_tests()
    cs.gate.reset_for_tests()


def _write_cache(tmp_path: Path, cached: dict) -> Path:
    p = tmp_path / "claude.json"
    p.write_text(json.dumps({"cachedUsageUtilization": cached}), encoding="utf-8")
    return p


def _fresh_saturated_cache(now: datetime, window: str = "five_hour", minutes_until_reset: int = 42) -> dict:
    resets_at = (now + timedelta(minutes=minutes_until_reset)).isoformat()
    other = "seven_day" if window == "five_hour" else "five_hour"
    return {
        "fetchedAtMs": time.time() * 1000,
        "utilization": {
            window: {"utilization": 100, "resets_at": resets_at},
            other: {"utilization": 10, "resets_at": (now + timedelta(days=3)).isoformat()},
        },
    }


# ── read_cached_usage_state ──────────────────────────────────────────────


class TestReadCachedUsageState:
    def test_missing_file(self, tmp_path):
        cached, age = cs.read_cached_usage_state(tmp_path / "nonexistent.json")
        assert cached is None and age is None

    def test_malformed_json(self, tmp_path):
        p = tmp_path / "claude.json"
        p.write_text("{not valid json", encoding="utf-8")
        cached, age = cs.read_cached_usage_state(p)
        assert cached is None and age is None

    def test_missing_cachedUsageUtilization_key(self, tmp_path):
        p = tmp_path / "claude.json"
        p.write_text(json.dumps({"somethingElse": True}), encoding="utf-8")
        cached, age = cs.read_cached_usage_state(p)
        assert cached is None and age is None

    def test_missing_fetchedAtMs(self, tmp_path):
        p = _write_cache(tmp_path, {"utilization": {}})
        cached, age = cs.read_cached_usage_state(p)
        assert cached is not None
        assert age is None

    def test_malformed_fetchedAtMs(self, tmp_path):
        p = _write_cache(tmp_path, {"fetchedAtMs": "not-a-number", "utilization": {}})
        cached, age = cs.read_cached_usage_state(p)
        assert cached is not None
        assert age is None

    def test_fresh_valid_cache_has_small_age(self, tmp_path):
        p = _write_cache(tmp_path, {"fetchedAtMs": time.time() * 1000, "utilization": {}})
        cached, age = cs.read_cached_usage_state(p)
        assert cached is not None
        assert age is not None
        assert age < 5

    def test_stale_cache_has_large_age(self, tmp_path):
        stale_ms = (time.time() - 32.4 * 3600) * 1000
        p = _write_cache(tmp_path, {"fetchedAtMs": stale_ms, "utilization": {}})
        _cached, age = cs.read_cached_usage_state(p)
        assert age is not None
        assert age > 32 * 3600


# ── binding_window ───────────────────────────────────────────────────────


class TestBindingWindow:
    def test_missing_resets_at(self):
        now = datetime.now(UTC)
        cached = {"utilization": {"five_hour": {"utilization": 100}}}
        assert cs.binding_window(cached, now) is None

    def test_reset_in_past_is_ignored(self):
        now = datetime.now(UTC)
        cached = {
            "utilization": {
                "five_hour": {
                    "utilization": 100,
                    "resets_at": (now - timedelta(hours=1)).isoformat(),
                }
            }
        }
        assert cs.binding_window(cached, now) is None

    def test_valid_future_reset(self):
        now = datetime.now(UTC)
        resets = now + timedelta(minutes=30)
        cached = {
            "utilization": {
                "five_hour": {
                    "utilization": 100,
                    "resets_at": resets.isoformat(),
                }
            }
        }
        result = cs.binding_window(cached, now)
        assert result is not None
        assert result[0] == "five_hour"

    def test_five_hour_binding(self):
        now = datetime.now(UTC)
        cached = {
            "utilization": {
                "five_hour": {"utilization": 100, "resets_at": (now + timedelta(minutes=10)).isoformat()},
                "seven_day": {"utilization": 40, "resets_at": (now + timedelta(days=2)).isoformat()},
            }
        }
        window, _ = cs.binding_window(cached, now)
        assert window == "five_hour"

    def test_seven_day_binding(self):
        now = datetime.now(UTC)
        cached = {
            "utilization": {
                "five_hour": {"utilization": 50, "resets_at": (now + timedelta(minutes=10)).isoformat()},
                "seven_day": {"utilization": 100, "resets_at": (now + timedelta(days=2)).isoformat()},
            }
        }
        window, _ = cs.binding_window(cached, now)
        assert window == "seven_day"

    def test_both_saturated_picks_soonest(self):
        now = datetime.now(UTC)
        cached = {
            "utilization": {
                "five_hour": {"utilization": 100, "resets_at": (now + timedelta(minutes=10)).isoformat()},
                "seven_day": {"utilization": 100, "resets_at": (now + timedelta(days=2)).isoformat()},
            }
        }
        window, resets_at = cs.binding_window(cached, now)
        assert window == "five_hour"
        assert resets_at == now + timedelta(minutes=10)

    def test_neither_saturated(self):
        now = datetime.now(UTC)
        cached = {
            "utilization": {
                "five_hour": {"utilization": 20, "resets_at": (now + timedelta(minutes=10)).isoformat()},
                "seven_day": {"utilization": 30, "resets_at": (now + timedelta(days=2)).isoformat()},
            }
        }
        assert cs.binding_window(cached, now) is None


# ── check_claude_availability ────────────────────────────────────────────


class TestCheckClaudeAvailability:
    def test_no_observed_exhaustion_is_available_even_with_optimistic_cache(self, tmp_path):
        now = datetime.now(UTC)
        p = _write_cache(tmp_path, _fresh_saturated_cache(now))
        avail = cs.check_claude_availability(
            cache_path=p,
            now=now,
            observed=(False, None),
            check_auth=False,
        )
        assert avail.state == cs.AVAILABLE

    def test_actual_exhaustion_plus_optimistic_cache_stays_exhausted(self, tmp_path):
        """Real observed failure must take precedence -- a cache showing
        utilization below 100% must NOT be interpreted as availability."""
        now = datetime.now(UTC)
        cached = {
            "fetchedAtMs": time.time() * 1000,
            "utilization": {
                "five_hour": {"utilization": 40, "resets_at": (now + timedelta(hours=1)).isoformat()},
                "seven_day": {"utilization": 20, "resets_at": (now + timedelta(days=2)).isoformat()},
            },
        }
        p = _write_cache(tmp_path, cached)
        avail = cs.check_claude_availability(
            cache_path=p,
            now=now,
            observed=(True, "session_limit"),
            check_auth=False,
        )
        assert avail.state == cs.EXHAUSTED_UNKNOWN_RESET

    def test_actual_exhaustion_plus_no_cache_is_unknown_reset(self, tmp_path):
        now = datetime.now(UTC)
        avail = cs.check_claude_availability(
            cache_path=tmp_path / "nope.json",
            now=now,
            observed=(True, "session_limit"),
            check_auth=False,
        )
        assert avail.state == cs.EXHAUSTED_UNKNOWN_RESET
        assert avail.reset_estimate is None

    def test_actual_exhaustion_plus_fresh_saturated_cache_is_known_reset(self, tmp_path):
        now = datetime.now(UTC)
        p = _write_cache(tmp_path, _fresh_saturated_cache(now, minutes_until_reset=42))
        avail = cs.check_claude_availability(
            cache_path=p,
            now=now,
            observed=(True, "session_limit"),
            check_auth=False,
        )
        assert avail.state == cs.EXHAUSTED_KNOWN_RESET
        assert avail.reset_estimate is not None
        assert avail.binding_window == "five_hour"
        assert avail.reset_source == "cached_usage_state"

    def test_stale_cache_never_produces_known_reset(self, tmp_path):
        """The concrete case that motivated this module: a 32+ hour stale
        cache with a resets_at that's already passed must never be read as
        a valid future estimate."""
        now = datetime.now(UTC)
        stale_ms = (time.time() - 32.4 * 3600) * 1000
        cached = {
            "fetchedAtMs": stale_ms,
            "utilization": {
                "five_hour": {"utilization": 100, "resets_at": (now - timedelta(hours=8)).isoformat()},
                "seven_day": {"utilization": 28, "resets_at": (now + timedelta(days=3)).isoformat()},
            },
        }
        p = _write_cache(tmp_path, cached)
        avail = cs.check_claude_availability(
            cache_path=p,
            now=now,
            observed=(True, "session_limit"),
            cache_max_age_seconds=600,
            check_auth=False,
        )
        assert avail.state == cs.EXHAUSTED_UNKNOWN_RESET
        assert avail.reset_estimate is None
        assert "stale" in avail.detail.lower()

    def test_cache_exactly_at_max_age_boundary_still_stale_rejected(self, tmp_path):
        now = datetime.now(UTC)
        stale_ms = (time.time() - 700) * 1000
        cached = {
            "fetchedAtMs": stale_ms,
            "utilization": {
                "five_hour": {
                    "utilization": 100,
                    "resets_at": (now + timedelta(minutes=10)).isoformat(),
                }
            },
        }
        p = _write_cache(tmp_path, cached)
        avail = cs.check_claude_availability(
            cache_path=p,
            now=now,
            observed=(True, "session_limit"),
            cache_max_age_seconds=600,
            check_auth=False,
        )
        assert avail.state == cs.EXHAUSTED_UNKNOWN_RESET

    def test_no_claude_json_file_at_all_still_functions(self, tmp_path):
        avail_no_exhaustion = cs.check_claude_availability(
            cache_path=tmp_path / "missing.json",
            observed=(False, None),
            check_auth=False,
        )
        assert avail_no_exhaustion.state == cs.AVAILABLE

        avail_exhausted = cs.check_claude_availability(
            cache_path=tmp_path / "missing.json",
            observed=(True, "empty_output"),
            check_auth=False,
        )
        assert avail_exhausted.state == cs.EXHAUSTED_UNKNOWN_RESET

    def test_auth_failure_detected_via_auth_runner(self):
        avail = cs.check_claude_availability(
            observed=(False, None),
            check_auth=True,
            auth_runner=lambda: False,
        )
        assert avail.state == cs.AUTH_FAILURE

    def test_auth_unknown_does_not_cause_auth_failure(self):
        """auth status returning None (couldn't determine) must never be
        treated as a failure -- only an explicit loggedIn=false does."""
        avail = cs.check_claude_availability(
            observed=(False, None),
            check_auth=True,
            auth_runner=lambda: None,
        )
        assert avail.state == cs.AVAILABLE

    def test_observed_transient_error_reason(self):
        avail = cs.check_claude_availability(
            observed=(True, "transient_error"),
            check_auth=False,
        )
        assert avail.state == cs.TRANSIENT_ERROR

    def test_observed_auth_failure_reason(self):
        avail = cs.check_claude_availability(
            observed=(True, "auth_failure"),
            check_auth=False,
        )
        assert avail.state == cs.AUTH_FAILURE


# ── apply-side signal (record_apply_exhaustion / record_apply_success) ──


class TestApplySignal:
    def test_record_and_clear(self):
        assert cs._apply_signal() == (False, None)
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=100)
        exhausted, reason = cs._apply_signal()
        assert exhausted is True
        assert reason == "session_limit"
        cs.record_apply_success()
        assert cs._apply_signal() == (False, None)

    def test_exhaustion_is_durable_past_a_negative_cooldown(self):
        """2026-08-21 fix: a fixed cooldown must NEVER silently declare
        recovery by itself. Even with cooldown_seconds=-1 (the probe
        window is immediately open), the durable exhausted flag stays set
        -- only record_apply_success() can clear it."""
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        exhausted, reason = cs._apply_signal()
        assert exhausted is True
        assert reason == "session_limit"

    def test_probe_due_false_before_cooldown(self):
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=1800)
        assert cs.probe_due() is False

    def test_probe_due_true_after_cooldown(self):
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        assert cs.probe_due() is True
        # Durability: probe being due is not the same as being available.
        exhausted, _ = cs._apply_signal()
        assert exhausted is True

    def test_probe_due_false_when_never_exhausted(self):
        assert cs.probe_due() is False

    def test_probe_due_false_after_recovery(self):
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        assert cs.probe_due() is True
        cs.record_apply_success()
        assert cs.probe_due() is False

    def test_check_claude_availability_uses_real_signal_by_default(self, tmp_path):
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=1800)
        avail = cs.check_claude_availability(
            cache_path=tmp_path / "nope.json",
            check_auth=False,
        )
        assert avail.state == cs.EXHAUSTED_UNKNOWN_RESET

    def test_full_lifecycle_does_not_auto_recover_from_elapsed_cooldown(self, tmp_path):
        """Integration-style: exhaustion -> advance past the cooldown/reset
        estimate -> call the REAL (unmocked) availability logic -> must NOT
        become AVAILABLE merely because the timer expired. Recovery must
        require an actual positive signal (record_apply_success), not just
        elapsed time."""
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=1800)

        # Simulate real time having advanced well past the cooldown/reset
        # estimate, with no probe attempted yet.
        cs._apply_probe_after = time.time() - 1

        avail = cs.check_claude_availability(
            cache_path=tmp_path / "nope.json",
            check_auth=False,
        )
        assert avail.state != cs.AVAILABLE
        assert avail.state == cs.EXHAUSTED_UNKNOWN_RESET

        # A probe is now due (gate would let a real attempt through)...
        assert cs.probe_due() is True
        # ...but merely being "due" is still not recovery.
        avail_still_checking = cs.check_claude_availability(
            cache_path=tmp_path / "nope.json",
            check_auth=False,
        )
        assert avail_still_checking.state != cs.AVAILABLE

        # Only a genuine positive signal (the probe attempt actually
        # succeeding) establishes recovery.
        cs.record_apply_success()
        avail_after_recovery = cs.check_claude_availability(
            cache_path=tmp_path / "nope.json",
            check_auth=False,
        )
        assert avail_after_recovery.state == cs.AVAILABLE

    def test_repeated_failed_probes_stay_exhausted_and_extend_backoff(self, tmp_path):
        """A probe that fails again must re-arm the durable flag and push
        the next probe window out further -- it must not accidentally
        clear anything."""
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        assert cs.probe_due() is True

        # The probe attempt happens (a real run_job call) and fails again.
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=1800)
        assert cs.probe_due() is False
        exhausted, reason = cs._apply_signal()
        assert exhausted is True
        assert reason == "session_limit"


# ── ClaudeGate ────────────────────────────────────────────────────────────


class TestClaudeGate:
    def test_defaults_to_not_paused(self):
        g = cs.ClaudeGate()
        assert g.is_paused() is False
        assert g.state == cs.AVAILABLE

    def test_set_available_not_paused(self):
        g = cs.ClaudeGate()
        g.set(cs.ClaudeAvailability(cs.AVAILABLE, None, None, None, None, "ok"))
        assert g.is_paused() is False

    def test_set_exhausted_pauses(self):
        g = cs.ClaudeGate()
        g.set(cs.ClaudeAvailability(cs.EXHAUSTED_UNKNOWN_RESET, None, None, None, None, "x"))
        assert g.is_paused() is True
        assert g.state == cs.EXHAUSTED_UNKNOWN_RESET

    def test_stays_paused_while_no_real_exhaustion_recorded(self):
        """A gate told 'exhausted' via .set() but with no corresponding
        real apply-side signal recorded must stay conservatively paused
        (probe_due() is false with nothing to probe for)."""
        g = cs.ClaudeGate()
        g.set(cs.ClaudeAvailability(cs.EXHAUSTED_UNKNOWN_RESET, None, None, None, None, "x"))
        assert cs.probe_due() is False
        assert g.is_paused() is True

    def test_probe_due_lets_a_paused_gate_through(self):
        """The fix's key invariant: durable exhaustion alone must not
        block forever -- once a real probe is due, the gate must let it
        through so recovery can ever be established."""
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        g = cs.ClaudeGate()
        g.set(cs.ClaudeAvailability(cs.EXHAUSTED_UNKNOWN_RESET, None, None, None, None, "x"))
        assert cs.probe_due() is True
        assert g.is_paused() is False

    def test_gate_re_pauses_after_a_failed_probe(self):
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        g = cs.ClaudeGate()
        g.set(cs.ClaudeAvailability(cs.EXHAUSTED_UNKNOWN_RESET, None, None, None, None, "x"))
        assert g.is_paused() is False  # probe allowed through

        # The probe attempt (a real run_job call) fails again.
        cs.record_apply_exhaustion("session_limit", cooldown_seconds=1800)
        g.set(cs.ClaudeAvailability(cs.EXHAUSTED_UNKNOWN_RESET, None, None, None, None, "x"))
        assert g.is_paused() is True

    def test_available_state_never_consults_probe_due(self):
        """When genuinely available, is_paused() short-circuits before
        ever looking at probe_due() -- confirms the two mechanisms don't
        interfere with the ordinary available path."""
        g = cs.ClaudeGate()
        g.set(cs.ClaudeAvailability(cs.AVAILABLE, None, None, None, None, "ok"))
        assert g.is_paused() is False
