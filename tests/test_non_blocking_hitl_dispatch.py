"""Orchestrator-level dispatch tests for non-blocking HITL (CLAUDE.md Future
Work item 69's addendum, built 2026-09-25), mirroring
test_human_first_orchestrator_dispatch.py's style: drive
_worker_loop_body(non_blocking_hitl=True) directly with fake acquire/run_job/
dispatch_hitl, and assert what it did (mark_result/release_lock calls,
applied/failed counters, whether it proceeded to a second job) rather than
re-testing hitl.dispatch_hitl's own internals (covered in
test_dispatch_hitl.py).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import session_pool


class _DummyChrome:
    pid = 424242

    def poll(self):
        return None


def _job(n: int, **overrides) -> dict:
    job = {
        "url": f"https://example.com/job/{n}",
        "title": f"Job {n}",
        "application_url": f"https://example.com/job/{n}",
        "site": "workday",
        "fit_score": 9,
    }
    job.update(overrides)
    return job


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


def _run_worker_loop_body(orch, *, limit=2, non_blocking_hitl=True):
    return orch._worker_loop_body(
        worker_id=0,
        limit=limit,
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
        port=9222,
        non_blocking_hitl=non_blocking_hitl,
    )


def setup_function():
    session_pool.reset_for_tests()


class TestNonBlockingHitlDispatch:
    def test_backgrounded_job_does_not_block_the_next_acquisition(self, monkeypatch):
        import applypilot.apply.orchestrator as orch
        from applypilot.apply import launcher

        job1, job2 = _job(1), _job(2)
        remaining = iter([job1, job2, None])
        monkeypatch.setattr(launcher, "acquire_job", lambda **k: next(remaining))

        run_job_calls: list[str] = []

        def _fake_run_job(job_arg, **kwargs):
            run_job_calls.append(job_arg["url"])
            if job_arg["url"] == job1["url"]:
                return "needs_human:captcha", 100, []
            return "applied", 200, []

        _patch_common(monkeypatch, orch, launcher)
        monkeypatch.setattr(launcher, "run_job", _fake_run_job)

        dispatch_calls: list[dict] = []

        def _fake_dispatch_hitl(**kwargs):
            dispatch_calls.append(kwargs)
            return "backgrounded", None

        monkeypatch.setattr(orch, "dispatch_hitl", _fake_dispatch_hitl)

        mark_result_calls: list[tuple] = []
        monkeypatch.setattr(launcher, "mark_result", lambda *a, **k: mark_result_calls.append((a, k)))

        applied, failed = _run_worker_loop_body(orch)

        assert run_job_calls == [job1["url"], job2["url"]], "must proceed to job 2 without waiting on job 1"
        assert len(dispatch_calls) == 1
        assert dispatch_calls[0]["non_blocking"] is True
        assert mark_result_calls and mark_result_calls[0][0][:2] == (job2["url"], "applied")
        # job1's own eventual fate is the (mocked-away) background thread's
        # job -- only job2's synchronous "applied" counts here.
        assert applied == 1
        assert failed == 0

    def test_falls_back_to_normal_relaunch_flow_when_dispatch_reports_blocking(self, monkeypatch):
        """When dispatch_hitl's own cap is full, it runs synchronously and
        returns ("blocking", outcome) -- this must be handled identically to
        the pre-existing direct _run_hitl call (relaunch loop continues)."""
        import applypilot.apply.orchestrator as orch
        from applypilot.apply import launcher

        job1 = _job(1)
        remaining = iter([job1, None])
        monkeypatch.setattr(launcher, "acquire_job", lambda **k: next(remaining))

        run_job_calls: list[str] = []

        def _fake_run_job(job_arg, **kwargs):
            run_job_calls.append(job_arg["url"])
            return "needs_human:captcha", 100, []

        _patch_common(monkeypatch, orch, launcher)
        monkeypatch.setattr(launcher, "run_job", _fake_run_job)

        def _fake_dispatch_hitl(**kwargs):
            return "blocking", ("applied", 500, [])

        monkeypatch.setattr(orch, "dispatch_hitl", _fake_dispatch_hitl)

        mark_result_calls: list[tuple] = []
        monkeypatch.setattr(launcher, "mark_result", lambda *a, **k: mark_result_calls.append((a, k)))

        applied, failed = _run_worker_loop_body(orch, limit=1)

        assert mark_result_calls and mark_result_calls[0][0][:2] == (job1["url"], "applied")
        assert applied == 1
        assert failed == 0

    def test_default_off_uses_run_hitl_directly_not_dispatch_hitl(self, monkeypatch):
        """non_blocking_hitl=False (the default) must never touch
        dispatch_hitl at all -- zero behavior change for existing callers."""
        import applypilot.apply.orchestrator as orch
        from applypilot.apply import launcher

        job1 = _job(1)
        remaining = iter([job1, None])
        monkeypatch.setattr(launcher, "acquire_job", lambda **k: next(remaining))

        _patch_common(monkeypatch, orch, launcher)
        monkeypatch.setattr(launcher, "run_job", lambda job_arg, **k: ("needs_human:captcha", 100, []))

        dispatch_calls = []
        monkeypatch.setattr(orch, "dispatch_hitl", lambda **k: dispatch_calls.append(k) or ("blocking", None))
        monkeypatch.setattr(orch, "_run_hitl", lambda *a, **k: ("applied", 500, []))

        mark_result_calls: list[tuple] = []
        monkeypatch.setattr(launcher, "mark_result", lambda *a, **k: mark_result_calls.append((a, k)))

        applied, failed = _run_worker_loop_body(orch, limit=1, non_blocking_hitl=False)

        assert dispatch_calls == [], "dispatch_hitl must not be called at all when non_blocking_hitl is off"
        assert mark_result_calls and mark_result_calls[0][0][:2] == (job1["url"], "applied")
        assert applied == 1
