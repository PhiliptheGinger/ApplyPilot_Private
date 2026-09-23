"""Orchestrator-level dispatch tests for the human-first flow's three
outcomes (CLAUDE.md Future Work item 40), mirroring
test_apply_claude_exhaustion.py's TestOrchestratorSessionExhaustionDispatch
style: drive _worker_loop_body(human_first=True) directly with a fake
acquire + a fake hitl.run_human_first, and assert what it did (mark_result/
release_lock/run_job calls, applied/failed counters) rather than what it
returned in prose.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class _OneShotAcquire:
    """Returns a job once, then None — so _worker_loop_body's per-job loop
    naturally exits after exactly one iteration (queue empty)."""

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

    def poll(self):
        return None


def _job():
    return {
        "url": "https://www.linkedin.com/jobs/view/4380167634",
        "title": "Software Engineer",
        "application_url": None,
        "site": "linkedin",
        "fit_score": 9,
    }


def _patch_common(monkeypatch, orch, launcher):
    monkeypatch.setattr(orch, "_probe_for_reconnect", lambda *a, **k: (None, None))
    import applypilot.enrichment.detail as _detail

    monkeypatch.setattr(_detail, "precheck_expired", lambda *a, **k: False)
    monkeypatch.setattr(orch, "detect_ats", lambda *a, **k: None)
    monkeypatch.setattr(orch, "launch_chrome", lambda *a, **k: _DummyChrome())
    monkeypatch.setattr(orch, "cleanup_worker", lambda *a, **k: None)
    monkeypatch.setattr(orch, "add_event", lambda *a, **k: None)
    monkeypatch.setattr(orch, "update_state", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_log_failed_attempt", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_record_job_history", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_stop_event", threading.Event())


def _call_worker_loop_body(orch, human_first_job=None):
    job = human_first_job or _job()
    return orch._worker_loop_body(
        worker_id=0,
        limit=1,
        target_url=None,
        min_score=8,
        max_score=None,
        max_age_days=14,
        headless=True,
        model="sonnet",
        dry_run=True,
        apply_engine="claude",
        fresh_sessions=False,
        applied=0,
        failed=0,
        continuous=False,
        jobs_done=0,
        empty_polls=0,
        port=9333,
        human_first=True,
    )


class TestHumanFirstDispatch:
    def test_applied_outcome_marks_applied_without_run_job(self, monkeypatch):
        import applypilot.apply.orchestrator as orch
        from applypilot.apply import hitl, launcher

        job = _job()
        mark_result_calls: list[tuple] = []
        run_job_calls: list[tuple] = []

        _patch_common(monkeypatch, orch, launcher)
        monkeypatch.setattr(launcher, "acquire_human_first_linkedin_job", _OneShotAcquire(job))
        monkeypatch.setattr(launcher, "mark_result", lambda *a, **k: mark_result_calls.append((a, k)))
        monkeypatch.setattr(launcher, "run_job", lambda *a, **k: run_job_calls.append((a, k)) or ("applied", 0, []))
        monkeypatch.setattr(hitl, "run_human_first", lambda **k: ("applied", job))

        applied, failed = _call_worker_loop_body(orch)

        assert run_job_calls == [], "human-applied-directly must never spawn a Claude automation session"
        assert mark_result_calls and mark_result_calls[0][0][:2] == (job["url"], "applied")
        assert applied == 1
        assert failed == 0

    def test_released_outcome_releases_lock_without_run_job(self, monkeypatch):
        import applypilot.apply.orchestrator as orch
        from applypilot.apply import hitl, launcher

        job = _job()
        release_calls: list[str] = []
        run_job_calls: list[tuple] = []

        _patch_common(monkeypatch, orch, launcher)
        monkeypatch.setattr(launcher, "acquire_human_first_linkedin_job", _OneShotAcquire(job))
        monkeypatch.setattr(launcher, "release_lock", lambda url: release_calls.append(url))
        monkeypatch.setattr(launcher, "run_job", lambda *a, **k: run_job_calls.append((a, k)) or ("applied", 0, []))
        monkeypatch.setattr(hitl, "run_human_first", lambda **k: ("released", job))

        applied, failed = _call_worker_loop_body(orch)

        assert run_job_calls == [], "a released (timed-out) human-first job must never fall through to automation"
        assert release_calls == [job["url"]]
        assert applied == 0
        assert failed == 0

    def test_handoff_outcome_calls_run_job_with_skip_tab_reset(self, monkeypatch):
        import applypilot.apply.orchestrator as orch
        from applypilot.apply import hitl, launcher

        job = _job()
        handoff_job = dict(job, application_url="https://boards.greenhouse.io/acme/jobs/99")
        run_job_calls: list[tuple] = []
        mark_result_calls: list[tuple] = []

        _patch_common(monkeypatch, orch, launcher)
        monkeypatch.setattr(launcher, "acquire_human_first_linkedin_job", _OneShotAcquire(job))
        monkeypatch.setattr(launcher, "mark_result", lambda *a, **k: mark_result_calls.append((a, k)))

        def _fake_run_job(job_arg, **kwargs):
            run_job_calls.append((job_arg, kwargs))
            return "applied", 500, []

        monkeypatch.setattr(launcher, "run_job", _fake_run_job)
        monkeypatch.setattr(hitl, "run_human_first", lambda **k: ("handoff", handoff_job))

        applied, failed = _call_worker_loop_body(orch)

        assert len(run_job_calls) == 1
        called_job, called_kwargs = run_job_calls[0]
        assert called_job["application_url"] == handoff_job["application_url"]
        assert called_kwargs["skip_tab_reset"] is True
        assert "LinkedIn" in (called_kwargs.get("extra_context") or "")
        assert applied == 1  # fake run_job returned "applied" -> flows through the normal branch
        assert mark_result_calls and mark_result_calls[0][0][:2] == (job["url"], "applied")

    def test_stopped_outcome_breaks_without_acquiring_again(self, monkeypatch):
        import applypilot.apply.orchestrator as orch
        from applypilot.apply import hitl, launcher

        job = _job()
        acquire_calls: list[int] = []

        def _acquire(**kwargs):
            acquire_calls.append(1)
            return job

        run_job_calls: list[tuple] = []

        _patch_common(monkeypatch, orch, launcher)
        monkeypatch.setattr(launcher, "acquire_human_first_linkedin_job", _acquire)
        monkeypatch.setattr(launcher, "run_job", lambda *a, **k: run_job_calls.append((a, k)) or ("applied", 0, []))
        monkeypatch.setattr(hitl, "run_human_first", lambda **k: ("stopped", job))

        applied, failed = _call_worker_loop_body(orch)

        assert run_job_calls == []
        assert len(acquire_calls) == 1, "must not loop back and acquire a second job after a stop signal"
        assert applied == 0
        assert failed == 0
