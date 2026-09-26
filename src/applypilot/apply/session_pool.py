"""Bookkeeping for non-blocking HITL pauses (CLAUDE.md Future Work item 69's
addendum, built 2026-09-25 per explicit user request: "Go ahead and build
it.").

When a job escalates to `needs_human`, a worker can dispatch the existing
wait-and-resume logic (`hitl._run_hitl`) onto a detached background thread
instead of blocking its own loop, then immediately move on to acquire its
next job under a freshly-allocated synthetic session identity.

Real prior art behind this design (Cloudflare Browser Run's HITL handoff,
UiPath Action Center's suspend/resume queue items -- see CLAUDE.md decision
log) converges on decoupling a paused SESSION's lifecycle from the WORKER
that noticed it. This module is the minimal real implementation of that
idea for ApplyPilot: rather than a full session pool where "any free
worker reclaims a paused session," the background thread that already
owns a paused session's Chrome/profile/HITL-listener resources simply
finishes that ONE job's lifecycle on its own once a human resolves it --
no other worker ever needs to "reclaim" anything, since chrome.py/hitl.py's
existing functions are already purely keyed by a plain int id with no
assumption that the id corresponds to a live ThreadPoolExecutor thread.
Confirmed live in code (not assumed) before this was built: Chrome is
already killed and relaunched fresh per job (`cleanup_worker` in
chrome.py's own per-job `finally` block), not held alive for a worker's
whole `--limit` lifetime -- only the on-disk profile directory persists
across jobs for a given id. So a backgrounded pause's Chrome staying alive
a bit longer than usual, while the freed worker launches a genuinely
separate Chrome under a new synthetic id, is the same *shape* of operation
this codebase already does constantly, just with two processes briefly
overlapping instead of one following another.

Bounded: MAX_CONCURRENT_BACKGROUND_HITL caps how many needs_human pauses
can be backgrounded at once, since each one holds an extra live Chrome
process open on this machine's known-constrained resources (decision
#131: ~460MB free RAM at idle, mechanical HDD, no GPU). Once the cap is
hit, callers fall back to the existing (safe, already-proven) blocking
behavior for that occurrence -- never worse than today.
"""

from __future__ import annotations

import itertools
import threading
import time

MAX_CONCURRENT_BACKGROUND_HITL = 2

# Synthetic session ids start well above any realistic --workers count so
# they can never collide with a real worker_id's own port/profile/extension
# resources -- all of chrome.py's/hitl.py's per-id resources are derived by
# simple integer arithmetic or string formatting from the id (BASE_CDP_PORT
# + id, HITL_LISTEN_BASE_PORT + id, CHROME_WORKER_DIR / f"worker-{id}"), so
# any never-before-used integer is automatically safe to use as a session id
# without touching those modules at all.
_SYNTHETIC_ID_BASE = 1000
_next_synthetic_id = itertools.count(_SYNTHETIC_ID_BASE)
_synthetic_id_lock = threading.Lock()

_active_count = 0
_active_lock = threading.Lock()


def allocate_synthetic_session_id() -> int:
    """Return a fresh, never-repeated synthetic session id for this process."""
    with _synthetic_id_lock:
        return next(_next_synthetic_id)


def can_spawn_background() -> bool:
    """Whether a new backgrounded HITL wait is currently allowed under the cap."""
    with _active_lock:
        return _active_count < MAX_CONCURRENT_BACKGROUND_HITL


def active_background_count() -> int:
    with _active_lock:
        return _active_count


def increment() -> None:
    global _active_count
    with _active_lock:
        _active_count += 1


def decrement() -> None:
    global _active_count
    with _active_lock:
        _active_count = max(0, _active_count - 1)


def wait_for_drain(timeout: float = 20.0, poll_interval: float = 0.25) -> bool:
    """Block (this thread only -- never a live worker thread) until every
    currently-backgrounded HITL job has finished, or `timeout` elapses.

    Real correctness gap this closes: `_worker_loop_body` returns its
    `(applied, failed)` counts to its caller (`main()`, via
    `ThreadPoolExecutor`'s `future.result()`) the moment its own while loop
    exits -- but a job it backgrounded onto a detached thread (CLAUDE.md
    Future Work item 69's addendum) may still be running at that exact
    moment, and mutating those same counts via `nonlocal` AFTER the
    function has already returned does not retroactively fix what was
    already handed back to the caller. A worker that backgrounded at least
    one job must wait here, right before its own return, so its reported
    totals are accurate -- not skip waiting just because ITS OWN queue is
    empty. Returns True if fully drained, False if `timeout` was hit with
    jobs still outstanding (a real, honest possibility if a human simply
    hasn't clicked Done yet -- the caller's own shutdown path is
    responsible for deciding what happens next, this function only reports
    which case occurred).
    """
    deadline = time.monotonic() + timeout
    while active_background_count() > 0:
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval)
    return True


def reset_for_tests() -> None:
    """Test-only: reset the active-count bookkeeping between test cases.

    Does NOT reset the synthetic-id counter -- id uniqueness across the
    whole process lifetime is the actual safety property being relied on,
    resetting it would defeat the point even in tests.
    """
    global _active_count
    with _active_lock:
        _active_count = 0
