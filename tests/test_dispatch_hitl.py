"""Regression tests for hitl.dispatch_hitl (CLAUDE.md Future Work item 69's
addendum, built 2026-09-25 per explicit user request to stop blocking a
worker's whole loop on a single needs_human pause).

Reuses test_hitl_retry_chrome_liveness.py's exact mocking pattern for
_run_hitl's internals (HTTP listener, CDP banner injection, Node watcher,
desktop notification, DB writes) via a fake already-set threading.Event so
the wait phase returns immediately -- these tests are about dispatch_hitl's
OWN blocking-vs-background branching and bookkeeping, not re-testing
_run_hitl's internals (already covered elsewhere).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import hitl, session_pool

# Captured BEFORE any monkeypatching happens: _patch_common patches
# hitl.threading.Event / hitl.time.sleep, but hitl.threading and hitl.time
# ARE the real global threading/time modules (same object, not a copy), so
# those patches leak into anything else in the process using threading.Event
# or time.sleep for the rest of the test -- including this test file's OWN
# synchronization primitives if they're constructed AFTER _patch_common runs.
# Using these captured-before-patch references keeps this test's own
# synchronization real and unaffected. See CLAUDE.md decisions #187/#190 for
# the same lesson learned twice before this test file existed.
_RealEvent = threading.Event
_real_sleep = time.sleep


class _ImmediateEvent(threading.Event):
    """Stand-in for threading.Event whose .wait() returns True at once.

    Must subclass the REAL threading.Event (bound at class-definition time,
    before any monkeypatching) rather than being a bare fake object --
    patching hitl.threading.Event module-wide also redirects
    threading.Thread.__init__'s own internal `_started` bookkeeping Event
    (same module object everywhere), so a non-Event fake breaks thread
    startup itself with "threads can only be started once". See CLAUDE.md
    decisions #187/#190 for the same lesson learned twice before.
    """

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
    monkeypatch.setattr(hitl, "_wait_for_cdp_ready", lambda *a, **k: True)
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

    def fake_run_job(job, **kwargs):
        return "applied", 1000, []

    monkeypatch.setattr(launcher, "run_job", fake_run_job)


def _job():
    return {"title": "Test Job", "site": "workday", "url": "https://example.com/job/1", "fit_score": 9}


def _call_dispatch(**overrides):
    kwargs = {
        "worker_id": 0,
        "port": 9222,
        "job": _job(),
        "reason": "form_stuck",
        "instructions": "do the thing",
        "navigate_url": "https://example.com/job/1",
        "duration_ms": 500,
    }
    kwargs.update(overrides)
    return hitl.dispatch_hitl(**kwargs)


def setup_function():
    session_pool.reset_for_tests()


def test_default_is_blocking_identical_to_calling_run_hitl_directly(monkeypatch):
    _patch_common(monkeypatch)
    mode, outcome = _call_dispatch()
    assert mode == "blocking"
    assert outcome[0] == "applied"
    assert session_pool.active_background_count() == 0


def test_non_blocking_returns_immediately_and_completes_in_background(monkeypatch):
    done = _RealEvent()
    captured = {}

    def on_complete(outcome):
        captured["outcome"] = outcome
        done.set()

    _patch_common(monkeypatch)
    mode, outcome = _call_dispatch(non_blocking=True, on_background_complete=on_complete)

    assert mode == "backgrounded"
    assert outcome is None

    assert done.wait(timeout=5.0), "background thread never called on_background_complete"
    assert captured["outcome"][0] == "applied"
    # The thread decrements after _run_hitl returns, before calling the
    # callback -- give it a moment to finish that final bookkeeping step.
    for _ in range(50):
        if session_pool.active_background_count() == 0:
            break
        _real_sleep(0.01)
    assert session_pool.active_background_count() == 0


def test_falls_back_to_blocking_when_cap_already_reached(monkeypatch):
    _patch_common(monkeypatch)
    for _ in range(session_pool.MAX_CONCURRENT_BACKGROUND_HITL):
        session_pool.increment()

    calls = []
    mode, outcome = _call_dispatch(non_blocking=True, on_background_complete=lambda o: calls.append(o))

    assert mode == "blocking"
    assert outcome[0] == "applied"
    assert calls == [], "on_background_complete must not fire on the synchronous fallback path"


def test_on_background_complete_exception_does_not_crash_the_thread(monkeypatch):
    _patch_common(monkeypatch)

    def bad_callback(outcome):
        raise RuntimeError("boom")

    mode, _ = _call_dispatch(non_blocking=True, on_background_complete=bad_callback)
    assert mode == "backgrounded"

    for _ in range(50):
        if session_pool.active_background_count() == 0:
            break
        _real_sleep(0.01)
    assert session_pool.active_background_count() == 0, "a crashing callback must still release the pool slot"
