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

import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ── ClaudeGate.is_probe_attempt() ────────────────────────────────────────

class TestClaudeGateIsProbeAttempt:
    def test_false_when_genuinely_available(self):
        g = claude_status.ClaudeGate()
        g.set(claude_status.ClaudeAvailability(
            claude_status.AVAILABLE, None, None, None, None, "ok"))
        assert g.is_probe_attempt() is False

    def test_false_when_paused_with_no_real_signal(self):
        """Told 'exhausted' via .set() but with no corresponding real
        apply-side signal recorded -- probe_due() is false, so this isn't
        a probe (nothing is being attempted at all; is_paused() is True)."""
        g = claude_status.ClaudeGate()
        g.set(claude_status.ClaudeAvailability(
            claude_status.EXHAUSTED_UNKNOWN_RESET, None, None, None, None, "x"))
        assert g.is_probe_attempt() is False

    def test_true_when_probe_due(self):
        claude_status.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        g = claude_status.ClaudeGate()
        g.set(claude_status.ClaudeAvailability(
            claude_status.EXHAUSTED_UNKNOWN_RESET, None, None, None, None, "x"))
        assert g.is_paused() is False
        assert g.is_probe_attempt() is True


# ── run_job(): a failed probe must re-arm, not silently expire ──────────

def _fake_timeout_proc(cmd):
    """subprocess.Popen replacement whose .wait() raises TimeoutExpired
    after an empty stdout stream, matching a hung/never-responding claude
    invocation without needing a real subprocess."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = iter([])  # no stream-json lines
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd=cmd, timeout=300)
    proc.poll.return_value = 0  # skip the _kill_process_tree cleanup path
    proc.pid = 424242
    return proc


class TestRunJobProbeRearm:
    """Covers the fix: run_job's subprocess.TimeoutExpired/generic-Exception
    handlers previously did nothing to the exhaustion signal. When the
    caller explicitly marks the attempt as a probe (is_probe=True) and it
    fails via one of these generic paths -- which carry no billing/
    session-limit text of their own to re-detect -- the probe window must
    be re-armed via the existing claude_status machinery so probe_due()
    doesn't stay stuck true forever. Durable exhaustion must stay set the
    whole time, and the state must never read as AVAILABLE.
    """

    def _job(self):
        return {
            "url": "https://example.com/job/1",
            "title": "Software Engineer",
            "application_url": "https://example.com/job/1",
            "site": "acme",
            "fit_score": 9,
        }

    def _run_with_mocked_subprocess(self, monkeypatch, tmp_path, *, is_probe: bool):
        import applypilot.apply.launcher as launcher
        from applypilot import config as cfg

        monkeypatch.setattr(cfg, "APP_DIR", tmp_path)
        monkeypatch.setattr(cfg, "LOG_DIR", tmp_path / "logs")
        monkeypatch.setattr(cfg, "APPLY_WORKER_DIR", tmp_path / "apply-workers")
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "apply-workers" / "worker-0").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(launcher, "reset_worker_dir", lambda worker_id: None)
        monkeypatch.setattr(launcher, "_reset_browser_tabs", lambda port: None)
        monkeypatch.setattr(launcher, "_refresh_gmail_token", lambda: True)
        monkeypatch.setattr(launcher, "_make_mcp_config", lambda port, worker_id=0: {})
        monkeypatch.setattr(launcher, "_activate_agent_tab", lambda *a, **k: None)
        monkeypatch.setattr(launcher.prompt_mod, "build_prompt", lambda **k: "prompt text")

        def _fake_popen(cmd, **kwargs):
            return _fake_timeout_proc(cmd)

        monkeypatch.setattr(launcher.subprocess, "Popen", _fake_popen)

        return launcher.run_job(
            self._job(), port=9333, worker_id=0, model="sonnet",
            dry_run=True, apply_engine="claude", is_probe=is_probe,
        )

    def test_full_probe_rearm_lifecycle(self, monkeypatch, tmp_path):
        """1. record exhaustion  2. advance to probe_due  3. run a probe
        that times out (generic-failure path)  4. exhaustion still durable
        5. probe window re-armed  6. immediate second probe_due() is false
        7. state is never AVAILABLE.
        """
        # 1 + 2: a prior exhaustion whose cooldown has already elapsed.
        claude_status.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        assert claude_status.probe_due() is True

        # 3: the probe attempt times out (generic-failure path, no
        # billing/session-limit text of its own).
        result, duration_ms, screening_qs = self._run_with_mocked_subprocess(
            monkeypatch, tmp_path, is_probe=True)
        assert result == "failed:timeout"

        # 4: durable exhaustion is untouched (still true).
        exhausted, reason = claude_status._apply_signal()
        assert exhausted is True

        # 5 + 6: the probe window was re-armed -- NOT left in the past --
        # so an immediate re-check is false again.
        assert claude_status.probe_due() is False

        # 7: never reported as AVAILABLE at any point in this lifecycle.
        avail = claude_status.check_claude_availability(
            cache_path=tmp_path / "no-cache.json", check_auth=False)
        assert avail.state != claude_status.AVAILABLE

    def test_non_probe_timeout_does_not_touch_exhaustion_signal(self, monkeypatch, tmp_path):
        """Control case: an ordinary (non-probe) attempt that times out
        must retain the EXISTING behavior -- no exhaustion signal touched
        at all, since Claude was never durably exhausted to begin with."""
        assert claude_status._apply_signal() == (False, None)

        result, duration_ms, screening_qs = self._run_with_mocked_subprocess(
            monkeypatch, tmp_path, is_probe=False)
        assert result == "failed:timeout"

        assert claude_status._apply_signal() == (False, None)
        assert claude_status.probe_due() is False

    def test_probe_generic_exception_also_rearms(self, monkeypatch, tmp_path):
        """Same contract for the generic Exception path (not just
        TimeoutExpired) -- e.g. a broken pipe writing to stdin."""
        import applypilot.apply.launcher as launcher

        claude_status.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        assert claude_status.probe_due() is True

        def _fake_popen(cmd, **kwargs):
            proc = MagicMock()
            proc.stdin = MagicMock()
            proc.stdin.write.side_effect = OSError("broken pipe")
            return proc

        import applypilot.apply.launcher as launcher_mod
        from applypilot import config as cfg
        monkeypatch.setattr(cfg, "APP_DIR", tmp_path)
        monkeypatch.setattr(cfg, "LOG_DIR", tmp_path / "logs")
        monkeypatch.setattr(cfg, "APPLY_WORKER_DIR", tmp_path / "apply-workers")
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "apply-workers" / "worker-0").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(launcher_mod, "reset_worker_dir", lambda worker_id: None)
        monkeypatch.setattr(launcher_mod, "_reset_browser_tabs", lambda port: None)
        monkeypatch.setattr(launcher_mod, "_refresh_gmail_token", lambda: True)
        monkeypatch.setattr(launcher_mod, "_make_mcp_config", lambda port, worker_id=0: {})
        monkeypatch.setattr(launcher_mod.prompt_mod, "build_prompt", lambda **k: "prompt text")
        monkeypatch.setattr(launcher_mod.subprocess, "Popen", _fake_popen)

        result, duration_ms, screening_qs = launcher.run_job(
            self._job(), port=9333, worker_id=0, model="sonnet",
            dry_run=True, apply_engine="claude", is_probe=True,
        )

        assert result.startswith("failed:")
        exhausted, _ = claude_status._apply_signal()
        assert exhausted is True
        assert claude_status.probe_due() is False

    def test_probe_success_not_undone_by_post_success_crash(self, monkeypatch, tmp_path):
        """Final pre-merge audit regression: a probe whose Claude invocation
        genuinely succeeds (real RESULT:APPLIED output) must have that
        recovery preserved even when unrelated post-success bookkeeping
        (mark_qa_outcome, as if hitting a still-locked SQLite DB after
        commit_with_retry's own retries were exhausted) raises afterward.
        record_apply_success() already ran by that point -- the crash must
        NOT re-arm exhaustion just because is_probe was True.
        """
        import applypilot.apply.launcher as launcher_mod
        import applypilot.database as database_mod
        import json as _json
        import sqlite3
        from applypilot import config as cfg

        # 1: Claude starts durably exhausted.
        exhausted_before, _ = claude_status._apply_signal()
        assert exhausted_before is False  # clean fixture state first...
        claude_status.record_apply_exhaustion("session_limit", cooldown_seconds=-1)
        exhausted_before, _ = claude_status._apply_signal()
        assert exhausted_before is True

        # 2: the probe window is open.
        assert claude_status.probe_due() is True

        monkeypatch.setattr(cfg, "APP_DIR", tmp_path)
        monkeypatch.setattr(cfg, "LOG_DIR", tmp_path / "logs")
        monkeypatch.setattr(cfg, "APPLY_WORKER_DIR", tmp_path / "apply-workers")
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "apply-workers" / "worker-0").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(launcher_mod, "reset_worker_dir", lambda worker_id: None)
        monkeypatch.setattr(launcher_mod, "_reset_browser_tabs", lambda port: None)
        monkeypatch.setattr(launcher_mod, "_refresh_gmail_token", lambda: True)
        monkeypatch.setattr(launcher_mod, "_make_mcp_config", lambda port, worker_id=0: {})
        monkeypatch.setattr(launcher_mod, "_activate_agent_tab", lambda *a, **k: None)
        monkeypatch.setattr(launcher_mod.prompt_mod, "build_prompt", lambda **k: "prompt text")

        # 3: Claude's own subprocess genuinely succeeds -- real stream-json
        # output containing RESULT:APPLIED, normal exit code.
        def _fake_popen(cmd, **kwargs):
            proc = MagicMock()
            proc.stdin = MagicMock()
            applied_line = _json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "RESULT:APPLIED"}]},
            })
            proc.stdout = iter([applied_line])
            proc.wait = MagicMock(return_value=None)
            proc.returncode = 0
            proc.poll.return_value = 0
            proc.pid = 424244
            return proc

        monkeypatch.setattr(launcher_mod.subprocess, "Popen", _fake_popen)

        # 5: realistic post-success bookkeeping (mark_qa_outcome, called
        # from the RESULT:APPLIED branch since the job has a url) raises.
        monkeypatch.setattr(
            database_mod, "mark_qa_outcome",
            MagicMock(side_effect=sqlite3.OperationalError("database is locked")),
        )

        result, duration_ms, screening_qs = launcher_mod.run_job(
            self._job(), port=9333, worker_id=0, model="sonnet",
            dry_run=True, apply_engine="claude", is_probe=True,
        )

        # 4 + 6: record_apply_success() was reached (verified below via the
        # durable state), and the crash still surfaces as a real failure --
        # it isn't silently swallowed.
        assert result.startswith("failed:")
        assert "locked" in result.lower()

        # 7 + 8: durable exhaustion was cleared by the genuine success and
        # MUST stay cleared -- the post-success crash must not re-arm it,
        # and the system must not report exhausted/anything but AVAILABLE.
        exhausted_after, reason_after = claude_status._apply_signal()
        assert exhausted_after is False
        assert reason_after is None

        avail = claude_status.check_claude_availability(
            cache_path=tmp_path / "no-cache.json", check_auth=False)
        assert avail.state == claude_status.AVAILABLE
