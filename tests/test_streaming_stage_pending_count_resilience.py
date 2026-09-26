"""Regression tests for CLAUDE.md decision #211 (2026-09-26): `_run_stage_
streaming`'s per-poll `_count_pending(...)` call used to be completely
unguarded, directly inside a bare `threading.Thread(..., daemon=True)`'s
loop with no try/except anywhere above it in `_run_streaming`'s call chain
either.

Real incident this closes: the overnight score stage silently stopped
being invoked at all after its last successful pass, ~9 hours before
anyone noticed, with zero traceback anywhere in the persisted log (Python's
default unhandled-thread-exception hook goes to stderr, never through this
module's own `log`) -- exactly the shape an uncaught `sqlite3.
OperationalError: database is locked` from `_count_pending` would produce,
given this machine's already-documented (decision #183) real, sustained
multi-stage write contention. The only visible trace of a thread dying
this way is `_run_streaming`'s watcher loop printing "Completed: <stage>"
to console once `t.is_alive()` goes False -- worded identically to (and,
from the log alone, indistinguishable from) a genuine finish.

Fixed the same way as every other transient-DB-error path in this
codebase: catch, log, back off, keep looping -- never let one bad poll
permanently kill a stage for the rest of the run.
"""

from __future__ import annotations

import threading

import applypilot.pipeline as pipeline


def _new_tracker_with_discover_done() -> pipeline._StageTracker:
    """A tracker where 'discover' (enrich's own upstream) is already
    marked done, so once pending settles to 0 the loop can naturally
    conclude "no more work" and exit cleanly via mark_done -- isolating
    the test to the _count_pending resilience behavior itself, not the
    upstream-waiting logic."""
    tracker = pipeline._StageTracker()
    tracker.mark_done("discover", {"status": "ok"})
    return tracker


def test_transient_count_pending_exception_is_retried_not_fatal(monkeypatch):
    calls = {"n": 0}

    def _flaky_count_pending(stage, min_score, max_age_days):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        return 0  # settles to "no work" on every call after the first

    monkeypatch.setattr(pipeline, "_count_pending", _flaky_count_pending)
    monkeypatch.setattr(pipeline, "_wait_or_stop", lambda stop_event, timeout: False)

    tracker = _new_tracker_with_discover_done()
    stop_event = threading.Event()

    # Must not raise -- the whole point of the fix.
    pipeline._run_stage_streaming("enrich", tracker, stop_event)

    assert calls["n"] >= 2, "the count check must be retried after the transient failure, not abandoned"
    assert tracker.is_done("enrich"), "the stage must still reach its normal mark_done completion"


def test_persistent_count_pending_exception_never_crashes_the_stage_thread(monkeypatch):
    """Defense in depth: even if the underlying condition NEVER clears
    (e.g. a sustained, not-just-transient lock), the stage loop must still
    be alive and controllable via the normal stop mechanism -- not stuck
    in a way that could only be ended by the thread crashing."""
    calls = {"n": 0}

    def _always_raises(stage, min_score, max_age_days):
        calls["n"] += 1
        raise RuntimeError("database is locked")

    stop_event = threading.Event()

    def _stop_after_a_few(stop_ev, timeout):
        if calls["n"] >= 3:
            stop_ev.set()
        return stop_ev.is_set()

    monkeypatch.setattr(pipeline, "_count_pending", _always_raises)
    monkeypatch.setattr(pipeline, "_wait_or_stop", _stop_after_a_few)

    tracker = _new_tracker_with_discover_done()

    # Must not raise, and must actually terminate (proving the loop is
    # still responsive to stop_event even while every count check fails).
    pipeline._run_stage_streaming("enrich", tracker, stop_event)

    assert calls["n"] >= 3
    assert tracker.is_done("enrich")


def test_count_pending_exception_is_logged_not_silently_swallowed(monkeypatch, caplog):
    """The real incident had ZERO trace in the persisted log -- the fix
    must actually route the failure through this module's own logger, not
    just avoid crashing."""
    import logging

    calls = {"n": 0}

    def _flaky_count_pending(stage, min_score, max_age_days):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        return 0

    monkeypatch.setattr(pipeline, "_count_pending", _flaky_count_pending)
    monkeypatch.setattr(pipeline, "_wait_or_stop", lambda stop_event, timeout: False)

    tracker = _new_tracker_with_discover_done()
    stop_event = threading.Event()

    with caplog.at_level(logging.ERROR, logger="applypilot.pipeline"):
        pipeline._run_stage_streaming("enrich", tracker, stop_event)

    assert any("pending-count check failed" in r.message for r in caplog.records)
