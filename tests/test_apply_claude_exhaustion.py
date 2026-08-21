"""Tests for the apply-side Claude exhaustion fix (section 10 of the
2026-08-20 continuous-scheduler design).

apply/launcher.py::run_job is a completely separate Claude Code CLI
invocation path from llm.py's _try_claude_cli (full agentic streaming
session vs. one-shot -p) with its own, previously weaker, exhaustion
detection that only recognized API-billing wording ("credit balance is too
low") and not the real Max/Pro session-limit wording ("You've hit your
session limit"). These tests cover the extracted pure classifier
(_classify_claude_apply_exhaustion) and the orchestrator-level dispatch
that must release the job without consuming an apply attempt or marking it
permanently failed.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply.launcher import _classify_claude_apply_exhaustion
from applypilot import claude_status


@pytest.fixture(autouse=True)
def _isolate_claude_status_state():
    claude_status.reset_apply_signal_for_tests()
    claude_status.gate.reset_for_tests()
    yield
    claude_status.reset_apply_signal_for_tests()
    claude_status.gate.reset_for_tests()


# ── _classify_claude_apply_exhaustion (pure) ─────────────────────────────

class TestClassifyClaudeApplyExhaustion:
    def test_explicit_session_limit_text(self):
        output = "You've hit your session limit · resets 4pm (America/New_York)"
        assert _classify_claude_apply_exhaustion(output, 1) == "session_limit"

    def test_usage_limit_text(self):
        assert _classify_claude_apply_exhaustion("Error: usage limit reached", 1) == "session_limit"

    def test_rate_limit_text(self):
        assert _classify_claude_apply_exhaustion("rate limit exceeded", 1) == "session_limit"

    def test_overloaded_text(self):
        assert _classify_claude_apply_exhaustion("the model is overloaded", 1) == "session_limit"

    def test_billing_credit_balance_text(self):
        assert _classify_claude_apply_exhaustion(
            "Error: Your credit balance is too low", 1) == "billing"

    def test_billing_insufficient_credits_text(self):
        assert _classify_claude_apply_exhaustion(
            "insufficient credits to complete request", 1) == "billing"

    def test_empty_output_nonzero_exit(self):
        assert _classify_claude_apply_exhaustion("", 1) == "empty_output"

    def test_empty_output_but_zero_exit_is_not_exhaustion(self):
        """Zero exit with empty output isn't the failure signature -- a
        genuinely empty successful run (shouldn't normally happen, but the
        classifier must not misfire on it) is not exhaustion."""
        assert _classify_claude_apply_exhaustion("", 0) is None

    def test_whitespace_only_output_nonzero_exit_is_empty_output(self):
        assert _classify_claude_apply_exhaustion("   \n\n  ", 1) == "empty_output"

    def test_genuine_error_text_is_not_exhaustion(self):
        assert _classify_claude_apply_exhaustion(
            "Error: could not find element on page", 1) is None

    def test_genuine_success_is_not_exhaustion(self):
        assert _classify_claude_apply_exhaustion("RESULT:APPLIED", 0) is None

    def test_session_limit_wins_over_billing_if_both_present(self):
        """Ambiguity case: if both phrasings somehow appear, the real
        temporary-condition wording takes priority so a genuine session
        limit is never misclassified as a permanent billing failure."""
        output = "session limit hit; credit balance is too low"
        assert _classify_claude_apply_exhaustion(output, 1) == "session_limit"

    def test_case_insensitive(self):
        assert _classify_claude_apply_exhaustion("SESSION LIMIT REACHED", 1) == "session_limit"
        assert _classify_claude_apply_exhaustion("CREDIT BALANCE IS TOO LOW", 1) == "billing"


# ── orchestrator dispatch: session exhaustion must not burn an attempt ────

class _OneShotAcquire:
    """Returns `job` once, then None forever -- simulates queue exhaustion
    after a single acquisition so the worker loop terminates cleanly."""

    def __init__(self, job):
        self._job = job
        self._served = False

    def __call__(self, **kwargs):
        if self._served:
            return None
        self._served = True
        return self._job


class _DummyChrome:
    pid = 424242


class TestOrchestratorSessionExhaustionDispatch:
    def _job(self):
        return {
            "url": "https://example.com/job/1",
            "title": "Software Engineer",
            "application_url": "https://example.com/job/1",
            "site": "acme",
        }

    def _patch_common(self, monkeypatch, orch, launcher, *, run_job_result):
        monkeypatch.setattr(orch, "_probe_for_reconnect", lambda *a, **k: (None, None))
        monkeypatch.setattr(orch, "detect_ats", lambda *a, **k: None)
        monkeypatch.setattr(orch, "launch_chrome", lambda *a, **k: _DummyChrome())
        monkeypatch.setattr(orch, "cleanup_worker", lambda *a, **k: None)
        monkeypatch.setattr(orch, "add_event", lambda *a, **k: None)
        monkeypatch.setattr(orch, "update_state", lambda *a, **k: None)
        monkeypatch.setattr(orch, "_SESSION_LIMIT_BACKOFF_SECONDS", 0)
        # _log_failed_attempt/_record_job_history do real file/DB I/O against
        # config.APP_DIR -- no-op them so these tests never touch the real
        # ~/.applypilot directory.
        monkeypatch.setattr(orch, "_log_failed_attempt", lambda *a, **k: None)
        monkeypatch.setattr(orch, "_record_job_history", lambda *a, **k: None)
        monkeypatch.setattr(launcher, "run_job", lambda *a, **k: run_job_result)
        monkeypatch.setattr(launcher, "_stop_event", threading.Event())

    def test_session_exhausted_releases_lock_without_permanent_failure(self, monkeypatch):
        import applypilot.apply.orchestrator as orch
        import applypilot.apply.launcher as launcher

        job = self._job()
        release_calls: list[str] = []
        mark_result_calls: list[tuple] = []

        self._patch_common(monkeypatch, orch, launcher,
                            run_job_result=("failed:claude_session_exhausted", 1234, []))
        monkeypatch.setattr(launcher, "acquire_job", _OneShotAcquire(job))
        monkeypatch.setattr(launcher, "release_lock", lambda url: release_calls.append(url))
        monkeypatch.setattr(launcher, "mark_result",
                             lambda *a, **k: mark_result_calls.append((a, k)))

        applied, failed = orch._worker_loop_body(
            worker_id=0, limit=1, target_url=None, min_score=8, max_score=None,
            max_age_days=14, headless=True, model="sonnet", dry_run=True,
            apply_engine="claude", fresh_sessions=False, applied=0, failed=0,
            continuous=False, jobs_done=0, empty_polls=0, port=9333,
        )

        assert release_calls == [job["url"]]
        assert mark_result_calls == []  # no permanent-failure write, no apply_attempts increment
        assert failed == 0
        assert applied == 0

    def test_genuine_permanent_failure_still_behaves_as_before(self, monkeypatch):
        """Control case: an unrelated permanent failure (e.g. not_a_job_application)
        must still go through mark_result(permanent=True) exactly as before --
        the new branch must not have weakened this path."""
        import applypilot.apply.orchestrator as orch
        import applypilot.apply.launcher as launcher

        job = self._job()
        release_calls: list[str] = []
        mark_result_calls: list[tuple] = []

        self._patch_common(monkeypatch, orch, launcher,
                            run_job_result=("failed:not_a_job_application", 1234, []))
        monkeypatch.setattr(launcher, "acquire_job", _OneShotAcquire(job))
        monkeypatch.setattr(launcher, "release_lock", lambda url: release_calls.append(url))
        monkeypatch.setattr(launcher, "mark_result",
                             lambda *a, **k: mark_result_calls.append((a, k)))

        applied, failed = orch._worker_loop_body(
            worker_id=0, limit=1, target_url=None, min_score=8, max_score=None,
            max_age_days=14, headless=True, model="sonnet", dry_run=True,
            apply_engine="claude", fresh_sessions=False, applied=0, failed=0,
            continuous=False, jobs_done=0, empty_polls=0, port=9333,
        )

        assert len(mark_result_calls) == 1
        args, kwargs = mark_result_calls[0]
        assert args[0] == job["url"]
        assert args[1] == "failed"
        assert kwargs.get("permanent") is True or (len(args) > 3 and args[3] is True)
        assert failed == 1

    def test_credits_exhausted_billing_still_stops_worker_as_before(self, monkeypatch):
        """Control case: genuine billing exhaustion must still be permanent
        and still stop the whole worker (_stop_event.set()) -- unchanged."""
        import applypilot.apply.orchestrator as orch
        import applypilot.apply.launcher as launcher

        job = self._job()
        mark_result_calls: list[tuple] = []

        self._patch_common(monkeypatch, orch, launcher,
                            run_job_result=("failed:credits_exhausted", 1234, []))
        monkeypatch.setattr(launcher, "acquire_job", _OneShotAcquire(job))
        monkeypatch.setattr(launcher, "release_lock", lambda url: None)
        monkeypatch.setattr(launcher, "mark_result",
                             lambda *a, **k: mark_result_calls.append((a, k)))

        applied, failed = orch._worker_loop_body(
            worker_id=0, limit=1, target_url=None, min_score=8, max_score=None,
            max_age_days=14, headless=True, model="sonnet", dry_run=True,
            apply_engine="claude", fresh_sessions=False, applied=0, failed=0,
            continuous=False, jobs_done=0, empty_polls=0, port=9333,
        )

        assert len(mark_result_calls) == 1
        assert mark_result_calls[0][1].get("permanent") is True
        assert failed == 1
        assert launcher._stop_event.is_set()


# ── gate prevents acquisition ────────────────────────────────────────────

class _StopAfterOneWait:
    """A fake stop_event: is_set() is False until wait() is called once,
    which flips it True and returns True -- lets a continuous loop run
    exactly one gated iteration in a test before exiting cleanly."""

    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def wait(self, timeout=None):
        self._set = True
        return True

    def clear(self):
        self._set = False


class TestGatePreventsAcquisition:
    def test_paused_gate_never_calls_acquire_job(self, monkeypatch):
        import applypilot.apply.orchestrator as orch
        import applypilot.apply.launcher as launcher

        acquire_calls = {"n": 0}

        def fake_acquire_job(**kwargs):
            acquire_calls["n"] += 1
            return None

        monkeypatch.setattr(orch, "_probe_for_reconnect", lambda *a, **k: (None, None))
        monkeypatch.setattr(orch, "add_event", lambda *a, **k: None)
        monkeypatch.setattr(orch, "update_state", lambda *a, **k: None)
        monkeypatch.setattr(orch, "POLL_INTERVAL", 0)
        monkeypatch.setattr(launcher, "acquire_job", fake_acquire_job)
        monkeypatch.setattr(launcher, "_stop_event", _StopAfterOneWait())

        claude_status.gate.set(claude_status.ClaudeAvailability(
            state=claude_status.EXHAUSTED_UNKNOWN_RESET, reset_estimate=None,
            reset_source=None, cache_age_seconds=None, binding_window=None,
            detail="test",
        ))

        applied, failed = orch._worker_loop_body(
            worker_id=0, limit=0, target_url=None, min_score=8, max_score=None,
            max_age_days=14, headless=True, model="sonnet", dry_run=True,
            apply_engine="claude", fresh_sessions=False, applied=0, failed=0,
            continuous=True, jobs_done=0, empty_polls=0, port=9333,
        )

        assert acquire_calls["n"] == 0

    def test_standalone_apply_unaffected_when_gate_never_set(self, monkeypatch):
        """The invariant from section 14: applypilot apply --continuous with
        no scheduler running must behave exactly as before -- the gate
        defaults to never-paused, so acquire_job is called normally."""
        import applypilot.apply.orchestrator as orch
        import applypilot.apply.launcher as launcher

        acquire_calls = {"n": 0}

        def fake_acquire_job(**kwargs):
            acquire_calls["n"] += 1
            return None  # queue empty

        monkeypatch.setattr(orch, "_probe_for_reconnect", lambda *a, **k: (None, None))
        monkeypatch.setattr(orch, "add_event", lambda *a, **k: None)
        monkeypatch.setattr(orch, "update_state", lambda *a, **k: None)
        monkeypatch.setattr(launcher, "acquire_job", fake_acquire_job)
        monkeypatch.setattr(launcher, "_stop_event", threading.Event())

        assert claude_status.gate.is_paused() is False  # default state, no scheduler ever ran

        applied, failed = orch._worker_loop_body(
            worker_id=0, limit=1, target_url=None, min_score=8, max_score=None,
            max_age_days=14, headless=True, model="sonnet", dry_run=True,
            apply_engine="claude", fresh_sessions=False, applied=0, failed=0,
            continuous=False, jobs_done=0, empty_polls=0, port=9333,
        )

        assert acquire_calls["n"] == 1
