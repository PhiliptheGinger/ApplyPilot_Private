"""Tests for the continuous scheduler (scheduler.py).

Pure-function tests for plan_upstream_work need no DB. run_once tests use
the tmp_db fixture plus fully injected run_pipeline_fn/availability_fn so
nothing here ever hits a real API, the real ~/.claude.json, or a real
subprocess.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot import scheduler as sched
from applypilot.claude_status import (
    AVAILABLE,
    EXHAUSTED_KNOWN_RESET,
    EXHAUSTED_UNKNOWN_RESET,
    ClaudeAvailability,
)


@pytest.fixture(autouse=True)
def _isolate_gate_state():
    """run_once() mutates the module-level claude_status.gate singleton as
    a side effect -- reset it around every test so this file's tests never
    leak gate state into each other or into other test modules."""
    sched.gate.reset_for_tests()
    yield
    sched.gate.reset_for_tests()


def _throughput(count: int, rate: float) -> dict:
    return {"count": count, "window_minutes": 60, "rate_per_minute": rate}


def _avail(state: str, reset_estimate=None, binding_window=None) -> ClaudeAvailability:
    return ClaudeAvailability(
        state=state,
        reset_estimate=reset_estimate,
        reset_source=None,
        cache_age_seconds=None,
        binding_window=binding_window,
        detail="test",
    )


# ── plan_upstream_work ───────────────────────────────────────────────────


class TestPlanUpstreamWork:
    def test_ready_below_target_produces_need(self):
        plan = sched.plan_upstream_work(
            ready=1,
            target=5,
            minutes_available=None,
            max_batch=20,
            safety_margin=0.5,
            tailor_throughput=_throughput(5, 1.0),
            cover_throughput=_throughput(5, 1.0),
        )
        assert plan["need"] == 4
        assert plan["tailor"] == 4
        assert plan["cover"] == 4

    def test_ready_at_target_produces_nothing(self):
        plan = sched.plan_upstream_work(
            ready=5,
            target=5,
            minutes_available=None,
            max_batch=20,
            safety_margin=0.5,
            tailor_throughput=_throughput(5, 1.0),
            cover_throughput=_throughput(5, 1.0),
        )
        assert plan == {"tailor": 0, "cover": 0, "need": 0}

    def test_ready_above_target_produces_nothing(self):
        plan = sched.plan_upstream_work(
            ready=9,
            target=5,
            minutes_available=None,
            max_batch=20,
            safety_margin=0.5,
            tailor_throughput=_throughput(5, 1.0),
            cover_throughput=_throughput(5, 1.0),
        )
        assert plan["tailor"] == 0 and plan["cover"] == 0

    def test_known_reset_capacity_calculation(self):
        # 40 minutes left, tailor 0.81/min, cover 4.1/min, margin 0.5:
        # tailor_capacity = int(40*0.81*0.5) = 16, cover_capacity = int(40*4.1*0.5) = 82
        # bottlenecked by tailor(16), but need=4 wins.
        plan = sched.plan_upstream_work(
            ready=1,
            target=5,
            minutes_available=40,
            max_batch=20,
            safety_margin=0.5,
            tailor_throughput=_throughput(5, 0.81),
            cover_throughput=_throughput(5, 4.1),
        )
        assert plan["need"] == 4
        assert plan["tailor"] == 4
        assert plan["cover"] == 4

    def test_known_reset_capacity_bottlenecks_below_need(self):
        # Only 5 minutes left, low throughput -- capacity below need.
        plan = sched.plan_upstream_work(
            ready=0,
            target=5,
            minutes_available=5,
            max_batch=20,
            safety_margin=0.5,
            tailor_throughput=_throughput(5, 0.2),
            cover_throughput=_throughput(5, 4.0),
        )
        # tailor_capacity = int(5*0.2*0.5) = 0
        assert plan["tailor"] == 0
        assert plan["cover"] == 0

    def test_huge_minutes_left_still_obeys_max_batch(self):
        plan = sched.plan_upstream_work(
            ready=0,
            target=1000,
            minutes_available=100000,
            max_batch=20,
            safety_margin=0.5,
            tailor_throughput=_throughput(5, 10.0),
            cover_throughput=_throughput(5, 10.0),
        )
        assert plan["tailor"] == 20
        assert plan["cover"] == 20

    def test_unbounded_available_state_still_obeys_max_batch(self):
        plan = sched.plan_upstream_work(
            ready=0,
            target=1000,
            minutes_available=None,
            max_batch=20,
            safety_margin=0.5,
            tailor_throughput=_throughput(5, 10.0),
            cover_throughput=_throughput(5, 10.0),
        )
        assert plan["tailor"] == 20
        assert plan["cover"] == 20

    def test_insufficient_throughput_history_uses_conservative_fallback(self):
        # count < MIN_OBSERVATIONS_FOR_MEASURED_RATE=3 -> ignore the
        # (absurdly high) measured rate, use the conservative fallback.
        plan = sched.plan_upstream_work(
            ready=0,
            target=5,
            minutes_available=10,
            max_batch=20,
            safety_margin=1.0,
            tailor_throughput=_throughput(1, 999.0),
            cover_throughput=_throughput(1, 999.0),
        )
        expected = int(10 * sched.CONSERVATIVE_FALLBACK_RATES["tailor"] * 1.0)
        assert plan["tailor"] == min(expected, 5)

    def test_exactly_three_observations_is_meaningful(self):
        plan = sched.plan_upstream_work(
            ready=0,
            target=100,
            minutes_available=10,
            max_batch=20,
            safety_margin=1.0,
            tailor_throughput=_throughput(3, 5.0),
            cover_throughput=_throughput(3, 5.0),
        )
        # measured rate used: int(10*5.0*1.0) = 50, capped by max_batch=20
        assert plan["tailor"] == 20

    def test_safety_margin_scales_capacity(self):
        plan_full = sched.plan_upstream_work(
            ready=0,
            target=100,
            minutes_available=10,
            max_batch=100,
            safety_margin=1.0,
            tailor_throughput=_throughput(5, 1.0),
            cover_throughput=_throughput(5, 1.0),
        )
        plan_half = sched.plan_upstream_work(
            ready=0,
            target=100,
            minutes_available=10,
            max_batch=100,
            safety_margin=0.5,
            tailor_throughput=_throughput(5, 1.0),
            cover_throughput=_throughput(5, 1.0),
        )
        assert plan_half["tailor"] < plan_full["tailor"]

    def test_zero_minutes_available_produces_no_work(self):
        plan = sched.plan_upstream_work(
            ready=0,
            target=5,
            minutes_available=0,
            max_batch=20,
            safety_margin=1.0,
            tailor_throughput=_throughput(5, 5.0),
            cover_throughput=_throughput(5, 5.0),
        )
        assert plan["tailor"] == 0
        assert plan["cover"] == 0

    def test_negative_minutes_available_clamped_to_zero(self):
        plan = sched.plan_upstream_work(
            ready=0,
            target=5,
            minutes_available=-10,
            max_batch=20,
            safety_margin=1.0,
            tailor_throughput=_throughput(5, 5.0),
            cover_throughput=_throughput(5, 5.0),
        )
        assert plan["tailor"] == 0
        assert plan["cover"] == 0


# ── run_once ──────────────────────────────────────────────────────────────


class _RecordingPipeline:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"stages": [], "errors": {}, "elapsed": 0.0}


class TestRunOnce:
    def _cfg(self, **overrides) -> sched.SchedulerConfig:
        base = {
            "ready_buffer": 5, "ready_buffer_unknown": 2, "poll_interval": 60,
            "cache_max_age": 600, "max_batch": 20, "safety_margin": 0.5,
        }
        base.update(overrides)
        return sched.SchedulerConfig(**base)

    def test_fresh_database_requery_no_cached_job_list(self, tmp_db, seed_job):
        """run_once must never thread a specific job list through to
        run_pipeline_fn -- only stage + limit -- so a newly discovered
        higher-scoring job is naturally included on the very next call
        (get_jobs_by_stage/acquire_job re-query fresh every time)."""
        conn = tmp_db()
        seed_job(conn, url_suffix="a", fit_score=8, state="tailored")
        pipeline_fn = _RecordingPipeline()
        cfg = self._cfg()

        sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(AVAILABLE),
            api_capacity_fn=lambda: True,
        )

        for call in pipeline_fn.calls:
            assert "job" not in call
            assert "url" not in call
            assert "jobs" not in call

    def test_available_state_targets_ready_buffer(self, tmp_db, seed_job):
        conn = tmp_db()
        pipeline_fn = _RecordingPipeline()
        cfg = self._cfg(ready_buffer=5, ready_buffer_unknown=2)

        result = sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(AVAILABLE),
            api_capacity_fn=lambda: True,
        )
        assert result["target"] == 5

    def test_unknown_reset_uses_smaller_target(self, tmp_db, seed_job):
        conn = tmp_db()
        pipeline_fn = _RecordingPipeline()
        cfg = self._cfg(ready_buffer=5, ready_buffer_unknown=2)

        result = sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(EXHAUSTED_UNKNOWN_RESET),
            api_capacity_fn=lambda: True,
        )
        assert result["target"] == 2

    def test_unknown_reset_skips_upstream_when_no_api_capacity(self, tmp_db, seed_job):
        conn = tmp_db()
        pipeline_fn = _RecordingPipeline()
        cfg = self._cfg()

        result = sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(EXHAUSTED_UNKNOWN_RESET),
            api_capacity_fn=lambda: False,
        )
        assert result["plan"]["tailor"] == 0
        assert result["plan"]["cover"] == 0
        tailor_calls = [c for c in pipeline_fn.calls if c.get("stages") == ["tailor"]]
        assert tailor_calls == []

    def test_stale_reset_does_not_bound_by_stale_timestamp(self, tmp_db, seed_job):
        """availability_fn reporting EXHAUSTED_UNKNOWN_RESET (as it must for
        a stale cache) means run_once computes minutes_available=None, not
        a value derived from an expired timestamp."""
        conn = tmp_db()
        pipeline_fn = _RecordingPipeline()
        cfg = self._cfg()

        result = sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(EXHAUSTED_UNKNOWN_RESET),
            api_capacity_fn=lambda: True,
        )
        assert result["minutes_available"] is None

    def test_reevaluates_after_estimated_reset_passes_without_assuming_recovery(self, tmp_db, seed_job):
        """If the availability_fn (as check_claude_availability legitimately
        would once a previously-estimated resets_at has passed with no new
        evidence Claude recovered) reports EXHAUSTED_UNKNOWN_RESET again,
        run_once must not silently treat it as AVAILABLE."""
        conn = tmp_db()
        pipeline_fn = _RecordingPipeline()
        cfg = self._cfg()

        past_estimate = datetime.now(UTC) - timedelta(minutes=5)
        # Simulate: first cycle thought it knew the reset time...
        result1 = sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(
                EXHAUSTED_KNOWN_RESET, reset_estimate=past_estimate + timedelta(minutes=5)
            ),
            api_capacity_fn=lambda: True,
        )
        assert result1["availability"].state == EXHAUSTED_KNOWN_RESET

        # ...but the estimate has now passed and Claude is still exhausted --
        # a correctly-implemented availability_fn reports UNKNOWN, not
        # AVAILABLE, and run_once must respect that rather than assume
        # recovery from elapsed time alone.
        result2 = sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(EXHAUSTED_UNKNOWN_RESET),
            api_capacity_fn=lambda: True,
        )
        assert result2["availability"].state == EXHAUSTED_UNKNOWN_RESET
        assert result2["target"] == cfg.ready_buffer_unknown

    def test_known_reset_minutes_available_computed_from_estimate(self, tmp_db, seed_job):
        conn = tmp_db()
        pipeline_fn = _RecordingPipeline()
        cfg = self._cfg()
        now = datetime.now(UTC)
        reset_at = now + timedelta(minutes=42)

        result = sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(EXHAUSTED_KNOWN_RESET, reset_estimate=reset_at),
            api_capacity_fn=lambda: True,
            now=now,
        )
        assert result["minutes_available"] == pytest.approx(42, abs=0.1)

    def test_gate_is_updated_from_availability(self, tmp_db, seed_job):
        conn = tmp_db()
        pipeline_fn = _RecordingPipeline()
        cfg = self._cfg()

        sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(EXHAUSTED_UNKNOWN_RESET),
            api_capacity_fn=lambda: True,
        )
        assert sched.gate.is_paused() is True
        assert sched.gate.state == EXHAUSTED_UNKNOWN_RESET

        sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=pipeline_fn,
            availability_fn=lambda **k: _avail(AVAILABLE),
            api_capacity_fn=lambda: True,
        )
        assert sched.gate.is_paused() is False
        sched.gate.reset_for_tests()


# ── single-instance protection ───────────────────────────────────────────


class TestSchedulerLock:
    def test_second_instance_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sched, "_RUN_STATE_DIR", tmp_path)
        monkeypatch.setattr(sched, "_SCHEDULER_PID_FILE", tmp_path / "scheduler.pid")

        sched.acquire_scheduler_lock()
        try:
            with pytest.raises(sched.SchedulerAlreadyRunning):
                sched.acquire_scheduler_lock()
        finally:
            sched.release_scheduler_lock()

    def test_stale_pid_reclaimed(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "scheduler.pid"
        monkeypatch.setattr(sched, "_RUN_STATE_DIR", tmp_path)
        monkeypatch.setattr(sched, "_SCHEDULER_PID_FILE", pid_file)

        # A PID essentially guaranteed not to be alive.
        pid_file.write_text("999999999", encoding="utf-8")
        sched.acquire_scheduler_lock()  # must not raise
        sched.release_scheduler_lock()

    def test_release_then_reacquire(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sched, "_RUN_STATE_DIR", tmp_path)
        monkeypatch.setattr(sched, "_SCHEDULER_PID_FILE", tmp_path / "scheduler.pid")

        sched.acquire_scheduler_lock()
        sched.release_scheduler_lock()
        sched.acquire_scheduler_lock()  # must not raise
        sched.release_scheduler_lock()


# ── sleep-interval selection ──────────────────────────────────────────────


class TestNextSleepSeconds:
    def _cfg(self) -> sched.SchedulerConfig:
        return sched.SchedulerConfig(poll_interval=60)

    def test_available_uses_poll_interval(self):
        backoff = sched._UnknownResetBackoff()
        result = {"availability": _avail(AVAILABLE)}
        assert sched._next_sleep_seconds(result, self._cfg(), backoff) == 60

    def test_known_reset_uses_min_of_poll_and_minutes_left(self):
        """Reset is imminent (30s away) -- must wake up sooner than the
        normal 60s poll interval so it doesn't sleep past the reset."""
        backoff = sched._UnknownResetBackoff()
        result = {"availability": _avail(EXHAUSTED_KNOWN_RESET), "minutes_available": 0.5}
        assert sched._next_sleep_seconds(result, self._cfg(), backoff) == 30

    def test_known_reset_does_not_shorten_poll_for_a_distant_reset(self):
        """Reset is farther away than the normal poll interval -- stick to
        the normal cadence rather than waiting the full remaining time."""
        backoff = sched._UnknownResetBackoff()
        result = {"availability": _avail(EXHAUSTED_KNOWN_RESET), "minutes_available": 2}
        assert sched._next_sleep_seconds(result, self._cfg(), backoff) == 60

    def test_unknown_reset_backoff_ladder_advances(self):
        backoff = sched._UnknownResetBackoff()
        result = {"availability": _avail(EXHAUSTED_UNKNOWN_RESET)}
        cfg = self._cfg()
        first = sched._next_sleep_seconds(result, cfg, backoff)
        second = sched._next_sleep_seconds(result, cfg, backoff)
        third = sched._next_sleep_seconds(result, cfg, backoff)
        assert (first, second, third) == (300, 600, 1200)

    def test_unknown_reset_backoff_caps(self):
        backoff = sched._UnknownResetBackoff()
        result = {"availability": _avail(EXHAUSTED_UNKNOWN_RESET)}
        cfg = self._cfg()
        for _ in range(10):
            value = sched._next_sleep_seconds(result, cfg, backoff)
        assert value == 1800

    def test_backoff_resets_when_available_again(self):
        backoff = sched._UnknownResetBackoff()
        cfg = self._cfg()
        sched._next_sleep_seconds({"availability": _avail(EXHAUSTED_UNKNOWN_RESET)}, cfg, backoff)
        sched._next_sleep_seconds({"availability": _avail(EXHAUSTED_UNKNOWN_RESET)}, cfg, backoff)
        sched._next_sleep_seconds({"availability": _avail(AVAILABLE)}, cfg, backoff)
        first_after_reset = sched._next_sleep_seconds({"availability": _avail(EXHAUSTED_UNKNOWN_RESET)}, cfg, backoff)
        assert first_after_reset == 300
