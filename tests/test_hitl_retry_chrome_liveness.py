"""Regression test for the 2026-09-20 HITL-retry Chrome-liveness check
(decision #173).

Root cause (confirmed live via a real apply run): `_run_hitl`'s post-resume
retry loop (step 10) used to call `run_job` up to 3 times without ever
checking whether Chrome was actually still reachable. A real run found
Chrome's CDP port go unreachable mid-session (traced to Windows'
ScheduledDefrag task + RestartManager coordinating app closures during a
disk-optimization pass) -- every subsequent retry attempt failed in
~seconds against a dead browser, burning real cost for zero chance of
success. The pre-existing wait-loop crash recovery (`chrome_proc.poll()`)
only catches an outright process exit, not a frozen-but-still-running
Chrome, so it didn't catch this case either.

`_run_hitl` now checks CDP reachability (`_wait_for_cdp_ready`) before every
retry attempt, tries exactly one relaunch if it's down, and gives up
cleanly (not 3 blind attempts) if the relaunch doesn't come back.

These tests exercise the retry loop directly (all other `_run_hitl` side
effects -- the HTTP listener, CDP banner injection, Node done-watcher,
desktop notification, DB writes -- are mocked/no-op'd) via a fake
already-set `threading.Event` so the wait phase returns immediately.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import hitl


class _ImmediateEvent:
    """Stand-in for threading.Event whose .wait() returns True at once,
    so _run_hitl's wait-for-human loop exits on the first iteration
    without ever needing a real Node watcher / banner click."""

    def is_set(self) -> bool:
        return True

    def set(self) -> None:
        pass

    def wait(self, timeout=None) -> bool:
        return True


def _patch_common(monkeypatch):
    monkeypatch.setattr(hitl, "mark_needs_human", lambda *a, **k: None)
    monkeypatch.setattr(hitl, "_start_hitl_listener", lambda *a, **k: 7380)
    monkeypatch.setattr(hitl, "_inject_banner_for_worker", lambda *a, **k: None)
    monkeypatch.setattr(hitl, "notify_human_needed", lambda *a, **k: None)
    monkeypatch.setattr(hitl, "reset_needs_human", lambda *a, **k: None)
    monkeypatch.setattr(hitl, "_stop_hitl_listener", lambda *a, **k: None)
    monkeypatch.setattr(hitl, "_unregister_waiting", lambda *a, **k: None)
    monkeypatch.setattr(hitl, "_register_waiting", lambda *a, **k: None)
    monkeypatch.setattr(hitl.threading, "Event", lambda: _ImmediateEvent())

    class _FakeLock:
        def acquire(self, blocking=False):
            return False

        def release(self):
            pass

    monkeypatch.setattr(hitl, "_stdin_fallback_lock", _FakeLock())
    monkeypatch.setattr(hitl.time, "sleep", lambda *a, **k: None)

    import applypilot.apply.human_review as human_review_mod

    monkeypatch.setattr(human_review_mod, "_start_done_watcher", lambda *a, **k: None)

    import applypilot.database as db_mod

    monkeypatch.setattr(db_mod, "get_qa", lambda *a, **k: None)
    monkeypatch.setattr(db_mod, "close_connection", lambda *a, **k: None)

    from applypilot.apply import launcher

    monkeypatch.setattr(launcher, "_worker_state", {}, raising=False)
    monkeypatch.setattr(launcher, "_worker_state_lock", threading.Lock(), raising=False)


def _job():
    return {"title": "Test Job", "site": "workday", "url": "https://example.com/job/1", "fit_score": 9}


def test_retry_proceeds_normally_when_chrome_stays_reachable(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(hitl, "_wait_for_cdp_ready", lambda *a, **k: True)

    from applypilot.apply import launcher

    calls = []

    def fake_run_job(job, **kwargs):
        calls.append(job["url"])
        return "applied", 1000, []

    monkeypatch.setattr(launcher, "run_job", fake_run_job)

    result = hitl._run_hitl(
        worker_id=0,
        port=9222,
        job=_job(),
        reason="form_stuck",
        instructions="do the thing",
        navigate_url="https://example.com/job/1",
        duration_ms=500,
    )

    assert len(calls) == 1, "CDP reachable the whole time -- must not relaunch Chrome or skip run_job"
    assert result[0] == "applied"


def test_retry_gives_up_cleanly_when_chrome_never_comes_back(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.setattr(hitl, "_wait_for_cdp_ready", lambda *a, **k: False)

    relaunch_calls = []
    monkeypatch.setattr(hitl, "launch_chrome", lambda *a, **k: relaunch_calls.append(1) or None)

    from applypilot.apply import launcher

    run_job_calls = []

    def fake_run_job(job, **kwargs):
        run_job_calls.append(job["url"])
        return "applied", 1000, []

    monkeypatch.setattr(launcher, "run_job", fake_run_job)

    result = hitl._run_hitl(
        worker_id=0,
        port=9222,
        job=_job(),
        reason="form_stuck",
        instructions="do the thing",
        navigate_url="https://example.com/job/1",
        duration_ms=500,
    )

    assert run_job_calls == [], "Chrome never reachable -- run_job must never be called"
    assert len(relaunch_calls) == 1, "must attempt exactly one relaunch, not retry 3x blindly"
    assert result[0] == "failed:browser_unreachable"


def test_retry_relaunches_once_then_succeeds(monkeypatch):
    """Chrome is down on attempt 1's pre-check, the relaunch works, and the
    subsequent run_job call succeeds -- confirms the relaunch path doesn't
    just give up unconditionally."""
    _patch_common(monkeypatch)

    cdp_calls = {"n": 0}

    def fake_cdp_ready(*a, **k):
        cdp_calls["n"] += 1
        # First check (pre-attempt) fails; second check (post-relaunch) succeeds.
        return cdp_calls["n"] > 1

    monkeypatch.setattr(hitl, "_wait_for_cdp_ready", fake_cdp_ready)
    monkeypatch.setattr(hitl, "launch_chrome", lambda *a, **k: object())

    from applypilot.apply import launcher

    run_job_calls = []

    def fake_run_job(job, **kwargs):
        run_job_calls.append(job["url"])
        return "applied", 1000, []

    monkeypatch.setattr(launcher, "run_job", fake_run_job)

    result = hitl._run_hitl(
        worker_id=0,
        port=9222,
        job=_job(),
        reason="form_stuck",
        instructions="do the thing",
        navigate_url="https://example.com/job/1",
        duration_ms=500,
    )

    assert len(run_job_calls) == 1, "relaunch succeeded -- run_job should proceed normally"
    assert result[0] == "applied"
