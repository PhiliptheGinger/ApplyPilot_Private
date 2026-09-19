"""Regression test for a real, confirmed production hang (2026-09-19).

`worker_loop` used to re-derive `continuous = limit == 0` internally,
which is only correct when the CLI's real `--continuous` flag put every
worker in continuous mode on purpose. `orchestrator.main()`'s
`executor.submit(worker_loop, ...)` call never passed the actual
`continuous` CLI flag through at all -- so when a BOUNDED `--limit` split
unevenly across workers (e.g. `--limit 3` over 5 workers: base=0, extra=3
-> limits=[1,1,1,0,0]), the two workers whose fair share was 0 read that
as "run forever, polling every 60s" instead of "nothing assigned, exit
now." Since `ThreadPoolExecutor.shutdown(wait=True)` waits on every
future, those two permanently-polling workers blocked the entire
`applypilot apply` pipeline from ever finishing -- confirmed live via
py-spy: real worker threads sitting in `_stop_event.wait(timeout=
POLL_INTERVAL)` for hours with zero visible progress, since the
"queue empty"/poll messages go to the in-memory dashboard (`add_event`),
not the captured console stream.

Fix: `worker_loop` now accepts an explicit `continuous` parameter (passed
through from `main()`'s real flag), falling back to the old `limit == 0`
heuristic only when a caller doesn't pass it -- so a bounded run's
zero-share workers correctly see `continuous=False` and exit immediately.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import orchestrator


def _run_with_timeout(fn, *, timeout: float, **kwargs):
    """Runs `fn(**kwargs)` in a background thread and returns
    (finished_in_time, result). Used to detect an infinite-loop
    regression deterministically instead of letting a real hang stall
    the test suite forever."""
    result: dict = {}

    def _target():
        result["value"] = fn(**kwargs)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return not t.is_alive(), result.get("value")


class TestZeroShareWorkerExitsInBoundedMode:
    def _patch_common(self, monkeypatch):
        monkeypatch.setattr(orchestrator, "_probe_for_reconnect", lambda *a, **k: (None, None))
        monkeypatch.setattr(orchestrator, "add_event", lambda *a, **k: None)
        monkeypatch.setattr(orchestrator, "update_state", lambda *a, **k: None)
        from applypilot.apply import launcher

        monkeypatch.setattr(launcher, "acquire_job", lambda *a, **k: None)
        monkeypatch.setattr(launcher, "_stop_event", threading.Event())
        monkeypatch.setattr(launcher, "_start_worker_listener", lambda *a, **k: None)
        monkeypatch.setattr(launcher, "_stop_worker_listener", lambda *a, **k: None)

    def test_limit_zero_with_explicit_continuous_false_exits_immediately(self, monkeypatch):
        """The fix: a worker given limit=0 as its bounded-run fair share
        (continuous=False passed explicitly) must return quickly, not
        poll forever."""
        self._patch_common(monkeypatch)
        finished, value = _run_with_timeout(
            orchestrator.worker_loop,
            timeout=15.0,  # real call takes ~5-6s (import overhead); well short of POLL_INTERVAL's 60s
            worker_id=3,
            limit=0,
            continuous=False,
        )
        assert finished, "worker_loop(limit=0, continuous=False) did not return -- regression reintroduced"
        assert value == (0, 0)

    def test_limit_zero_with_no_continuous_arg_falls_back_to_old_heuristic(self, monkeypatch):
        """Backward-compat: a caller that doesn't pass `continuous` at all
        (any caller besides orchestrator.main()) keeps the original
        limit==0-means-continuous behavior -- this test intentionally
        does NOT assert it exits quickly, since true continuous mode is
        supposed to poll forever until told to stop."""
        self._patch_common(monkeypatch)
        from applypilot.apply import launcher

        stop_event = threading.Event()
        monkeypatch.setattr(launcher, "_stop_event", stop_event)

        finished, _value = _run_with_timeout(
            orchestrator.worker_loop,
            timeout=1.0,
            worker_id=4,
            limit=0,
        )
        # Still polling after 1s (POLL_INTERVAL is 60s) -- confirms the
        # fallback heuristic still produces continuous mode when no
        # caller opts into the explicit flag.
        assert not finished
        stop_event.set()  # let the thread's next wait() wake up and exit
