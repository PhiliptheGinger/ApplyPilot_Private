"""Tests for the continuous scheduler's durable run log.

Covers: `applypilot run-continuous` writes a dedicated, timestamped log file
under ~/.applypilot/logs/ (mocked to a tmp dir in every test here); cycle
and decision records land in it in a machine-parseable form; the apply-side
Claude exhaustion/recovery signal is captured; and repeated setup/teardown
(across cycles, and across separate runs) never leaves duplicate handlers
on the loggers involved.

None of this changes scheduler planning/apply behavior -- these tests only
exercise the new logging plumbing (_setup_continuous_file_logging /
_teardown_continuous_file_logging / _log_decision) plus the two new log
lines in claude_status.record_apply_exhaustion/record_apply_success.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot import claude_status as cs
from applypilot import config
from applypilot import scheduler as sched


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(config, "LOG_DIR", log_dir)
    cs.reset_apply_signal_for_tests()
    sched.gate.reset_for_tests()
    yield
    cs.reset_apply_signal_for_tests()
    sched.gate.reset_for_tests()
    # Defensive cleanup: if a test failed before its own teardown call ran,
    # don't let a stray FileHandler leak into later tests/files.
    for name in sched._CONTINUOUS_LOG_LOGGER_NAMES:
        logger_ = logging.getLogger(name)
        for h in list(logger_.handlers):
            if isinstance(h, logging.FileHandler):
                logger_.removeHandler(h)
                h.close()


def _avail(state: str, **kw) -> cs.ClaudeAvailability:
    base = {
        "state": state, "reset_estimate": None, "reset_source": None,
        "cache_age_seconds": None, "binding_window": None, "detail": "test",
    }
    base.update(kw)
    return cs.ClaudeAvailability(**base)


class _RecordingPipeline:
    def __call__(self, **kwargs):
        return {"stages": [], "errors": {}, "elapsed": 0.0}


def _cfg(**overrides) -> sched.SchedulerConfig:
    base = {
        "ready_buffer": 5, "ready_buffer_unknown": 2, "poll_interval": 60,
        "cache_max_age": 600, "max_batch": 20, "safety_margin": 0.5,
    }
    base.update(overrides)
    return sched.SchedulerConfig(**base)


def _log_text() -> str:
    files = list(config.LOG_DIR.glob("continuous_*.log"))
    assert len(files) == 1, f"expected exactly one continuous_*.log, found {files}"
    return files[0].read_text(encoding="utf-8")


def _decision_payload(text: str, event: str) -> dict:
    line = next(l for l in text.splitlines() if f"DECISION {event}" in l)
    json_part = line.split(f"DECISION {event}", 1)[1].strip()
    return json.loads(json_part)


# ── setup/teardown ─────────────────────────────────────────────────────


class TestSetupTeardown:
    def test_creates_a_persistent_log_file(self):
        handler = sched._setup_continuous_file_logging()
        try:
            log_files = list(config.LOG_DIR.glob("continuous_*.log"))
            assert len(log_files) == 1
            assert log_files[0].exists()
        finally:
            sched._teardown_continuous_file_logging(handler)

    def test_log_filename_matches_expected_pattern(self):
        handler = sched._setup_continuous_file_logging()
        try:
            name = next(iter(config.LOG_DIR.glob("continuous_*.log"))).name
            assert re.match(r"^continuous_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log$", name)
        finally:
            sched._teardown_continuous_file_logging(handler)

    def test_handler_attached_to_all_expected_loggers(self):
        handler = sched._setup_continuous_file_logging()
        try:
            for name in sched._CONTINUOUS_LOG_LOGGER_NAMES:
                assert handler in logging.getLogger(name).handlers
        finally:
            sched._teardown_continuous_file_logging(handler)

    def test_teardown_removes_handler_from_all_loggers(self):
        handler = sched._setup_continuous_file_logging()
        sched._teardown_continuous_file_logging(handler)
        for name in sched._CONTINUOUS_LOG_LOGGER_NAMES:
            assert handler not in logging.getLogger(name).handlers

    def test_no_handler_duplication_across_repeated_setup_teardown(self):
        """Simulates several separate `run-continuous` invocations in the
        same process -- repeated setup/teardown must never leave more than
        one FileHandler attached to a given logger at a time."""
        for _ in range(3):
            handler = sched._setup_continuous_file_logging()
            for name in sched._CONTINUOUS_LOG_LOGGER_NAMES:
                file_handlers = [h for h in logging.getLogger(name).handlers if isinstance(h, logging.FileHandler)]
                assert len(file_handlers) == 1
            sched._teardown_continuous_file_logging(handler)
        for name in sched._CONTINUOUS_LOG_LOGGER_NAMES:
            file_handlers = [h for h in logging.getLogger(name).handlers if isinstance(h, logging.FileHandler)]
            assert file_handlers == []

    def test_multiple_cycles_within_one_setup_do_not_duplicate_handler(self, tmp_db, seed_job):
        """The realistic case: one run_continuous invocation, many run_once
        cycles -- setup happens exactly once, so repeated cycles must never
        add a second handler."""
        conn = tmp_db()
        handler = sched._setup_continuous_file_logging()
        try:
            for _ in range(3):
                sched.run_once(
                    _cfg(),
                    conn=conn,
                    run_pipeline_fn=_RecordingPipeline(),
                    availability_fn=lambda **k: _avail(cs.AVAILABLE),
                    api_capacity_fn=lambda: True,
                )
            file_handlers = [
                h for h in logging.getLogger("applypilot.scheduler").handlers if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1
        finally:
            sched._teardown_continuous_file_logging(handler)


# ── decision records ─────────────────────────────────────────────────────


class TestDecisionRecords:
    def test_cycle_writes_expected_decision_events(self, tmp_db, seed_job):
        conn = tmp_db()
        handler = sched._setup_continuous_file_logging()
        try:
            sched.run_once(
                _cfg(),
                conn=conn,
                run_pipeline_fn=_RecordingPipeline(),
                availability_fn=lambda **k: _avail(cs.AVAILABLE),
                api_capacity_fn=lambda: True,
            )
        finally:
            sched._teardown_continuous_file_logging(handler)

        text = _log_text()
        for event in (
            "cycle_start",
            "claude_availability",
            "apply_gate",
            "discover_result",
            "enrich_result",
            "score_planned",
            "ready_queue",
            "upstream_plan",
            "cycle_end",
        ):
            assert f"DECISION {event}" in text, f"missing DECISION {event} in log"

    def test_claude_availability_record_is_parseable_json(self, tmp_db, seed_job):
        conn = tmp_db()
        handler = sched._setup_continuous_file_logging()
        try:
            sched.run_once(
                _cfg(),
                conn=conn,
                run_pipeline_fn=_RecordingPipeline(),
                availability_fn=lambda **k: _avail(
                    cs.EXHAUSTED_KNOWN_RESET,
                    reset_estimate=datetime(2026, 8, 20, 19, 0, tzinfo=UTC),
                    binding_window="five_hour",
                ),
                api_capacity_fn=lambda: True,
            )
        finally:
            sched._teardown_continuous_file_logging(handler)

        payload = _decision_payload(_log_text(), "claude_availability")
        assert payload["state"] == cs.EXHAUSTED_KNOWN_RESET
        assert payload["binding_window"] == "five_hour"
        assert "2026-08-20" in payload["reset_estimate"]

    def test_upstream_plan_records_throughput_and_plan(self, tmp_db, seed_job):
        conn = tmp_db()
        handler = sched._setup_continuous_file_logging()
        try:
            sched.run_once(
                _cfg(ready_buffer=5),
                conn=conn,
                run_pipeline_fn=_RecordingPipeline(),
                availability_fn=lambda **k: _avail(cs.AVAILABLE),
                api_capacity_fn=lambda: True,
            )
        finally:
            sched._teardown_continuous_file_logging(handler)

        payload = _decision_payload(_log_text(), "upstream_plan")
        assert "plan" in payload
        assert "tailor" in payload["plan"]
        assert "cover" in payload["plan"]
        assert "tailor_throughput" in payload
        assert "cover_throughput" in payload

    def test_apply_gate_reflects_pause_state(self, tmp_db, seed_job):
        conn = tmp_db()
        handler = sched._setup_continuous_file_logging()
        try:
            sched.run_once(
                _cfg(),
                conn=conn,
                run_pipeline_fn=_RecordingPipeline(),
                availability_fn=lambda **k: _avail(cs.EXHAUSTED_UNKNOWN_RESET),
                api_capacity_fn=lambda: True,
            )
        finally:
            sched._teardown_continuous_file_logging(handler)

        payload = _decision_payload(_log_text(), "apply_gate")
        assert payload["paused"] is True
        assert payload["state"] == cs.EXHAUSTED_UNKNOWN_RESET

    def test_ready_queue_records_ready_and_target(self, tmp_db, seed_job):
        conn = tmp_db()
        handler = sched._setup_continuous_file_logging()
        try:
            sched.run_once(
                _cfg(ready_buffer_unknown=2),
                conn=conn,
                run_pipeline_fn=_RecordingPipeline(),
                availability_fn=lambda **k: _avail(cs.EXHAUSTED_UNKNOWN_RESET),
                api_capacity_fn=lambda: True,
            )
        finally:
            sched._teardown_continuous_file_logging(handler)

        payload = _decision_payload(_log_text(), "ready_queue")
        assert payload["target"] == 2
        assert payload["ready"] == 0


# ── apply-side Claude signal logging ──────────────────────────────────────


class TestApplySignalLogging:
    def test_exhaustion_signal_is_logged(self):
        handler = sched._setup_continuous_file_logging()
        try:
            cs.record_apply_exhaustion("session_limit", cooldown_seconds=100)
        finally:
            sched._teardown_continuous_file_logging(handler)

        text = _log_text()
        assert "exhausted" in text.lower()
        assert "session_limit" in text

    def test_recovery_signal_is_logged_only_when_previously_exhausted(self):
        handler = sched._setup_continuous_file_logging()
        try:
            # No prior exhaustion -- must not log a spurious "cleared" line.
            cs.record_apply_success()
            text_before = _log_text()
            assert "cleared" not in text_before.lower()

            cs.record_apply_exhaustion("empty_output", cooldown_seconds=100)
            cs.record_apply_success()
            text_after = _log_text()
            assert "cleared" in text_after.lower()
        finally:
            sched._teardown_continuous_file_logging(handler)


# ── run_continuous start/stop banner ──────────────────────────────────────


class TestRunContinuousBanner:
    def test_run_start_and_stop_are_logged_with_config(self, monkeypatch):
        """Run-start must record the effective config (so a run can be
        reconstructed without guessing what flags were used) and shutdown
        must be logged whether stopped cleanly or by Ctrl+C."""
        monkeypatch.setattr(sched, "acquire_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched, "release_scheduler_lock", lambda: None)
        monkeypatch.setattr(
            sched,
            "run_once",
            lambda cfg, last_discover_at=None: (_ for _ in ()).throw(KeyboardInterrupt),
        )

        cfg = _cfg(no_continuous_apply=True, ready_buffer=7)
        sched.run_continuous(cfg)

        text = _log_text()
        assert "run-continuous starting" in text
        assert "ready_buffer" in text and "7" in text
        assert "stopped by user" in text.lower()
        assert "shutdown complete" in text.lower()

    def test_unhandled_exception_is_logged_and_reraised(self, monkeypatch):
        """2026-08-28: a single cycle failure no longer immediately kills
        run_continuous -- it's caught, logged, and retried after a backoff
        (see TestRunContinuousCycleFailureHandling below). A PERSISTENT
        failure (every cycle raises) still eventually surfaces and re-raises
        once the consecutive-failure ceiling is reached -- this test now
        proves that end-to-end behavior survives, updated to mock
        time.sleep so it stays fast/deterministic rather than actually
        sleeping through every retry's backoff."""
        monkeypatch.setattr(sched, "acquire_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched, "release_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched.time, "sleep", lambda _seconds: None)

        calls = {"n": 0}

        def _boom(cfg, *, last_discover_at=None):
            calls["n"] += 1
            raise ValueError("boom")

        monkeypatch.setattr(sched, "run_once", _boom)

        cfg = _cfg(no_continuous_apply=True)
        with pytest.raises(ValueError):
            sched.run_continuous(cfg)

        assert calls["n"] == sched._MAX_CONSECUTIVE_CYCLE_FAILURES
        text = _log_text()
        assert "crashed with an unhandled exception" in text.lower()
        assert "shutdown complete" in text.lower()

    def test_no_continuous_apply_flag_is_logged(self, monkeypatch):
        monkeypatch.setattr(sched, "acquire_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched, "release_scheduler_lock", lambda: None)
        monkeypatch.setattr(
            sched,
            "run_once",
            lambda cfg, last_discover_at=None: (_ for _ in ()).throw(KeyboardInterrupt),
        )

        cfg = _cfg(no_continuous_apply=True)
        sched.run_continuous(cfg)

        text = _log_text()
        assert "NOT started" in text


# ── run_continuous bounded per-cycle failure handling ──────────────────────
#
# 2026-08-28: previously ANY exception from run_once propagated straight out
# of the while-loop, killing the entire continuous supervisor on a single
# transient DB/network/API/malformed-response failure. Now a cycle failure
# is caught, logged, and retried after a backoff (reusing the existing
# _TRANSIENT_ERROR_BACKOFF_SECONDS constant); only
# _MAX_CONSECUTIVE_CYCLE_FAILURES in a row without an intervening success
# causes a genuine exit. KeyboardInterrupt/SystemExit are explicitly never
# treated as cycle failures.


class TestRunContinuousCycleFailureHandling:
    def test_transient_failure_is_recovered_and_counter_resets(self, tmp_db, seed_job, monkeypatch):
        """First cycle raises, second cycle succeeds (proving the counter
        reset and the normal success path -- log line, _next_sleep_seconds,
        last_discover_at update -- are unaffected), third cycle stops the
        loop via KeyboardInterrupt so the test terminates deterministically."""
        monkeypatch.setattr(sched, "acquire_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched, "release_scheduler_lock", lambda: None)
        sleeps: list[float] = []
        monkeypatch.setattr(sched.time, "sleep", lambda seconds: sleeps.append(seconds))

        conn = tmp_db()
        cfg = _cfg(no_continuous_apply=True)
        # A real, fully-shaped result dict (format_status_block/_next_sleep_seconds
        # read several keys) -- built via the real run_once rather than hand-rolled,
        # so this test can't drift from that function's actual return shape.
        good_result = sched.run_once(
            cfg,
            conn=conn,
            run_pipeline_fn=_RecordingPipeline(),
            availability_fn=lambda **k: _avail(cs.AVAILABLE),
            api_capacity_fn=lambda: True,
        )

        calls = {"n": 0}

        def _flaky(cfg, *, last_discover_at=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("transient boom")
            if calls["n"] == 2:
                return good_result
            raise KeyboardInterrupt

        monkeypatch.setattr(sched, "run_once", _flaky)

        sched.run_continuous(cfg)

        assert calls["n"] == 3
        assert sched._TRANSIENT_ERROR_BACKOFF_SECONDS in sleeps
        text = _log_text()
        assert "consecutive failure 1/5" in text.lower()
        assert "cycle 2 finished" in text.lower()  # the recovered cycle completed normally

    def test_persistent_failure_stops_after_ceiling_with_backoff(self, monkeypatch):
        monkeypatch.setattr(sched, "acquire_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched, "release_scheduler_lock", lambda: None)
        sleeps: list[float] = []
        monkeypatch.setattr(sched.time, "sleep", lambda seconds: sleeps.append(seconds))

        calls = {"n": 0}

        def _always_boom(cfg, *, last_discover_at=None):
            calls["n"] += 1
            raise ValueError("persistent boom")

        monkeypatch.setattr(sched, "run_once", _always_boom)

        cfg = _cfg(no_continuous_apply=True)
        with pytest.raises(ValueError, match="persistent boom"):
            sched.run_continuous(cfg)

        # Retried up to the ceiling, not forever, and not just once.
        assert calls["n"] == sched._MAX_CONSECUTIVE_CYCLE_FAILURES
        # A backoff sleep precedes every retry except the final (ceiling-reaching) failure.
        assert sleeps.count(sched._TRANSIENT_ERROR_BACKOFF_SECONDS) == sched._MAX_CONSECUTIVE_CYCLE_FAILURES - 1

        text = _log_text()
        assert "reached the limit" in text.lower()
        assert "crashed with an unhandled exception" in text.lower()
        assert f"failed (consecutive failure {sched._MAX_CONSECUTIVE_CYCLE_FAILURES}/{sched._MAX_CONSECUTIVE_CYCLE_FAILURES})" in text.lower()

    def test_keyboard_interrupt_is_not_counted_as_a_cycle_failure(self, monkeypatch):
        monkeypatch.setattr(sched, "acquire_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched, "release_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched.time, "sleep", lambda seconds: None)

        calls = {"n": 0}

        def _interrupt(cfg, *, last_discover_at=None):
            calls["n"] += 1
            raise KeyboardInterrupt

        monkeypatch.setattr(sched, "run_once", _interrupt)

        cfg = _cfg(no_continuous_apply=True)
        sched.run_continuous(cfg)  # must exit cleanly, not raise

        assert calls["n"] == 1  # never retried
        text = _log_text()
        assert "consecutive failure" not in text.lower()
        assert "stopped by user" in text.lower()

    def test_system_exit_is_not_counted_as_a_cycle_failure(self, monkeypatch):
        monkeypatch.setattr(sched, "acquire_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched, "release_scheduler_lock", lambda: None)
        monkeypatch.setattr(sched.time, "sleep", lambda seconds: None)

        calls = {"n": 0}

        def _exit(cfg, *, last_discover_at=None):
            calls["n"] += 1
            raise SystemExit(1)

        monkeypatch.setattr(sched, "run_once", _exit)

        cfg = _cfg(no_continuous_apply=True)
        with pytest.raises(SystemExit):
            sched.run_continuous(cfg)

        assert calls["n"] == 1  # never retried -- SystemExit must propagate immediately
        text = _log_text()
        assert "consecutive failure" not in text.lower()
