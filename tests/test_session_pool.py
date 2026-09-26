"""Regression tests for apply/session_pool.py (CLAUDE.md Future Work item 69's
addendum, built 2026-09-25): the bookkeeping behind non-blocking HITL.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import threading
import time

from applypilot.apply import session_pool


def setup_function():
    session_pool.reset_for_tests()


def test_synthetic_ids_are_unique_and_above_base():
    seen = {session_pool.allocate_synthetic_session_id() for _ in range(20)}
    assert len(seen) == 20, "every allocation must be unique"
    assert all(i >= session_pool._SYNTHETIC_ID_BASE for i in seen)


def test_cap_enforced_and_reversible():
    assert session_pool.can_spawn_background() is True
    for _ in range(session_pool.MAX_CONCURRENT_BACKGROUND_HITL):
        session_pool.increment()
    assert session_pool.can_spawn_background() is False
    assert session_pool.active_background_count() == session_pool.MAX_CONCURRENT_BACKGROUND_HITL

    session_pool.decrement()
    assert session_pool.can_spawn_background() is True


def test_decrement_never_goes_negative():
    session_pool.decrement()
    session_pool.decrement()
    assert session_pool.active_background_count() == 0


def test_wait_for_drain_returns_true_immediately_when_nothing_outstanding():
    assert session_pool.wait_for_drain(timeout=1.0) is True


def test_wait_for_drain_waits_for_a_real_decrement_then_returns_true():
    session_pool.increment()

    def _decrement_soon():
        time.sleep(0.1)
        session_pool.decrement()

    threading.Thread(target=_decrement_soon, daemon=True).start()

    start = time.monotonic()
    assert session_pool.wait_for_drain(timeout=5.0, poll_interval=0.02) is True
    assert time.monotonic() - start < 5.0, "should return as soon as the count drains, not wait out the full timeout"


def test_wait_for_drain_times_out_honestly_when_never_drained():
    session_pool.increment()
    assert session_pool.wait_for_drain(timeout=0.2, poll_interval=0.05) is False
    session_pool.reset_for_tests()
