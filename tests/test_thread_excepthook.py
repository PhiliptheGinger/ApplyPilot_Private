"""Regression tests for CLAUDE.md decision #211/Future Work item 70a
(2026-09-26): a global `threading.excepthook` override, installed once at
CLI import time in `cli.py`, so ANY background thread's unhandled
exception -- not just the one call site decision #211 fixed directly --
gets routed through this application's own logger instead of Python's
default (stderr-only, never captured by the per-run FileHandler) behavior.
"""

from __future__ import annotations

import logging
import threading

import applypilot.cli as cli


def test_excepthook_is_actually_installed():
    """The real wiring, not just the function existing -- confirms
    threading.excepthook was reassigned at import time."""
    assert threading.excepthook is cli._log_unhandled_thread_exception


def _make_args(exc_type, exc_value, thread=None):
    try:
        raise exc_value
    except exc_type:
        import sys

        tb = sys.exc_info()[2]
    return threading.ExceptHookArgs((exc_type, exc_value, tb, thread))


def test_real_exception_is_logged_with_thread_name_and_traceback(caplog):
    fake_thread = threading.Thread(name="stage-score")
    args = _make_args(RuntimeError, RuntimeError("database is locked"), thread=fake_thread)

    with caplog.at_level(logging.ERROR, logger="applypilot"):
        cli._log_unhandled_thread_exception(args)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "stage-score" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_missing_thread_name_does_not_crash_the_hook(caplog):
    """args.thread can legitimately be None -- the hook itself must never
    raise (a hook that crashes while handling a crash would be worse than
    the silence it's meant to fix)."""
    args = _make_args(RuntimeError, RuntimeError("boom"), thread=None)

    with caplog.at_level(logging.ERROR, logger="applypilot"):
        cli._log_unhandled_thread_exception(args)

    assert len(caplog.records) == 1
    assert "<unknown thread>" in caplog.records[0].getMessage()


def test_system_exit_delegates_to_default_hook_not_logged_as_an_error(monkeypatch, caplog):
    """A deliberate thread-level SystemExit is normal control flow, not a
    crash worth alarming on -- must fall through to Python's own default
    handling instead of being logged as an error."""
    called = {"default_hook": False}
    monkeypatch.setattr(threading, "__excepthook__", lambda args: called.__setitem__("default_hook", True))

    args = _make_args(SystemExit, SystemExit(0), thread=threading.Thread(name="x"))

    with caplog.at_level(logging.ERROR, logger="applypilot"):
        cli._log_unhandled_thread_exception(args)

    assert called["default_hook"] is True
    assert len(caplog.records) == 0


def test_real_thread_crash_is_captured_end_to_end(caplog):
    """Not just the function in isolation -- an actual crashing
    threading.Thread, driven through the REAL global threading.excepthook
    assignment, must produce a log record. This is the literal shape of
    the real incident (decision #211): a bare daemon thread raising with
    nothing above it to catch the exception."""

    def _boom():
        raise RuntimeError("database is locked")

    t = threading.Thread(target=_boom, name="stage-enrich-test", daemon=True)

    with caplog.at_level(logging.ERROR, logger="applypilot"):
        t.start()
        t.join(timeout=5.0)

    assert not t.is_alive()
    matching = [r for r in caplog.records if "stage-enrich-test" in r.getMessage()]
    assert matching, "the real thread crash must be logged, not silently lost"
    assert matching[0].exc_info[0] is RuntimeError
