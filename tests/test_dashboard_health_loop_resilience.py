"""Regression tests for CLAUDE.md decision #211's audit (Future Work item
70b, 2026-09-26): `dashboard.py::_health_check_loop` had NO exception guard
at all -- a single raise from `_check_chrome_health` (or anything else in
one poll iteration) would have silently killed the whole health-check
thread for the rest of the run, the same shape as the pipeline.py stream-
stage gap decision #211 fixed directly. Lower real-world severity than
that one (`chrome_ok` only feeds the dashboard display, nothing functional
reads it), but the same audit found it and the same fix applies: catch,
log, keep polling.
"""

from __future__ import annotations

import logging

import applypilot.apply.dashboard as dashboard


def test_transient_check_chrome_health_exception_does_not_kill_the_loop(monkeypatch):
    calls = {"n": 0}

    def _flaky_check(worker_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("socket error")
        return True

    monkeypatch.setattr(dashboard, "_check_chrome_health", _flaky_check)
    monkeypatch.setattr(dashboard, "_worker_states", {0: dashboard.WorkerState(worker_id=0)})

    wait_calls = {"n": 0}

    class _FakeStop:
        def wait(self, timeout=None):
            wait_calls["n"] += 1
            return wait_calls["n"] > 3  # a few real iterations, then stop cleanly

    monkeypatch.setattr(dashboard, "_health_stop", _FakeStop())

    # Must not raise -- the whole point of the fix.
    dashboard._health_check_loop()

    assert calls["n"] >= 2, "the health check must be retried after the transient failure, not abandoned"


def test_check_chrome_health_exception_is_logged_not_silently_swallowed(caplog):
    def _always_raises(worker_id):
        raise RuntimeError("socket error")

    import applypilot.apply.dashboard as dashboard_mod

    orig_check = dashboard_mod._check_chrome_health
    orig_states = dict(dashboard_mod._worker_states)
    orig_stop = dashboard_mod._health_stop

    try:
        dashboard_mod._check_chrome_health = _always_raises
        dashboard_mod._worker_states = {0: dashboard_mod.WorkerState(worker_id=0)}

        wait_calls = {"n": 0}

        class _FakeStop:
            def wait(self, timeout=None):
                wait_calls["n"] += 1
                return wait_calls["n"] > 1

        dashboard_mod._health_stop = _FakeStop()

        with caplog.at_level(logging.ERROR, logger="applypilot.apply.dashboard"):
            dashboard_mod._health_check_loop()

        assert any("Chrome health-check loop" in r.message for r in caplog.records)
    finally:
        dashboard_mod._check_chrome_health = orig_check
        dashboard_mod._worker_states = orig_states
        dashboard_mod._health_stop = orig_stop
