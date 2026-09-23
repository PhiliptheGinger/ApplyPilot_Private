"""Regression test for a real, live-caught bug (2026-09-23): `_run_hitl`'s
terminal-stdin "Done" fallback silently auto-dismissed the VERY FIRST
needs_human pause whenever stdin wasn't a live interactive terminal (e.g.
`applypilot apply` launched under a background/non-interactive process).

Root cause: `sys.stdin.readline().strip().lower()` reduces BOTH a real
interactive Enter keypress ("\n" -> "") AND true EOF on a closed/non-
interactive stdin (returns "" directly, no blocking) to the same empty
string -- and empty string was one of the values treated as "done, resume
the agent". A real live apply run hit a genuine `needs_human:
sms_verification` pause (Walmart careers, a real live job), and the page
closed again within ~15 seconds with no human action at all -- confirmed
via direct code read this class of bug, not a coincidence.

Fixed by checking the RAW (pre-strip) readline() return value first: only
a truly empty read (zero bytes -- real EOF) is skipped; a real Enter
keypress ("\n") still falls through to the normal done-matching logic.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import hitl


class _FastEvent(threading.Event):
    """threading.Event with a capped wait() timeout, so the test doesn't
    have to sit through the real 5s poll interval. A real SUBCLASS (not a
    from-scratch reimplementation that calls threading.Event() itself) --
    `class _FastEvent(threading.Event)` binds the REAL Event class as its
    base at module-import time, before any monkeypatching runs, so
    instantiating one can't recurse into the (later-patched)
    `threading.Event` factory the way a wrapper calling `threading.Event()`
    internally would."""

    def wait(self, timeout=None) -> bool:
        return super().wait(timeout=min(timeout or 0.05, 0.05))


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
    monkeypatch.setattr(hitl.time, "sleep", lambda *a, **k: None)

    import applypilot.apply.human_review as human_review_mod

    monkeypatch.setattr(human_review_mod, "_start_done_watcher", lambda *a, **k: None)

    import applypilot.database as db_mod

    monkeypatch.setattr(db_mod, "get_qa", lambda *a, **k: None)
    monkeypatch.setattr(db_mod, "close_connection", lambda *a, **k: None)

    from applypilot.apply import launcher

    monkeypatch.setattr(launcher, "_worker_state", {}, raising=False)
    monkeypatch.setattr(launcher, "_worker_state_lock", threading.Lock(), raising=False)

    run_job_calls: list[str] = []

    def fake_run_job(job, **kwargs):
        run_job_calls.append(job["url"])
        return "applied", 1000, []

    monkeypatch.setattr(launcher, "run_job", fake_run_job)
    return run_job_calls


def _job():
    return {"title": "Personal Shopper", "site": "linkedin", "url": "https://example.com/job/1", "fit_score": 10}


def _run_hitl_with_controlled_events(monkeypatch, stdin_text: str):
    """Runs _run_hitl in a background thread with a real (unpatched)
    stop_event and a fast-polling fake for the function's OWN internal
    hitl_event, so the test can observe whether the stdin reader fired it
    without needing to sit through the real 5s poll interval.

    Returns (created_events, run_job_calls, thread). Event-creation order
    is: [0] the outer worker Thread's own internal `_started` bookkeeping
    (created before `_run_hitl` even starts running), [1] `_run_hitl`'s
    own `hitl_event` (what this test actually cares about), [2] the
    stdin-reader daemon Thread's own `_started` (created after, at step
    7b) -- `threading.Thread.__init__` creates an Event internally too,
    since `threading.Event` was patched module-wide, not just for
    `_run_hitl`'s own direct calls.
    """
    # Create the real stop_event BEFORE patching threading.Event, since
    # hitl.threading IS the real threading module (same object) -- patching
    # it after would make this call return a _FastEvent too.
    real_stop_event = threading.Event()

    run_job_calls = _patch_common(monkeypatch)

    created_events: list[_FastEvent] = []

    def _factory():
        e = _FastEvent()
        created_events.append(e)
        return e

    monkeypatch.setattr(hitl.threading, "Event", _factory)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))

    def _run():
        hitl._run_hitl(
            worker_id=0,
            port=9222,
            job=_job(),
            reason="sms_verification",
            instructions="enter the code",
            navigate_url="https://careers.example.com/verify",
            duration_ms=500,
            stop_event=real_stop_event,
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # Give the daemon stdin-reader thread a real chance to hit EOF/read a
    # line and (if the bug were present) fire hitl_event.
    time.sleep(0.3)
    real_stop_event.set()
    t.join(timeout=2)
    return created_events, run_job_calls, t


class TestStdinFallbackEofHandling:
    def test_eof_stdin_does_not_auto_dismiss_the_pause(self, monkeypatch):
        created_events, run_job_calls, t = _run_hitl_with_controlled_events(monkeypatch, stdin_text="")

        assert not t.is_alive(), "the worker thread must finish once stop_event is set, not hang"
        assert len(created_events) >= 2, "expected Thread._started + _run_hitl.hitl_event"
        assert not created_events[1].is_set(), (
            "EOF on a non-interactive stdin must NOT be treated as a real "
            "human pressing Enter to confirm done -- this is exactly the "
            "live-caught bug (2026-09-23) that silently closed a real "
            "sms_verification pause within seconds of it appearing"
        )
        # The wait loop exited via stop_event, not via a (bogus) done
        # signal, so it should NOT have proceeded into the retry loop.
        assert run_job_calls == []

    def test_real_enter_keypress_still_confirms_done(self, monkeypatch):
        """Control case: a genuine interactive Enter keypress ("\\n") must
        still work exactly as before -- the fix must not break the real
        use case it's meant to preserve."""
        created_events, run_job_calls, t = _run_hitl_with_controlled_events(monkeypatch, stdin_text="\n")

        assert not t.is_alive()
        assert created_events
        assert created_events[1].is_set(), "a real Enter keypress must still be treated as 'done'"

    def test_typed_done_still_works(self, monkeypatch):
        created_events, run_job_calls, t = _run_hitl_with_controlled_events(monkeypatch, stdin_text="done\n")

        assert not t.is_alive()
        assert created_events
        assert created_events[1].is_set()
