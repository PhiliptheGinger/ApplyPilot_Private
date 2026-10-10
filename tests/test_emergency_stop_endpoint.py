"""Integration test: POST /api/emergency-stop -- the real one-click STOP
button raised live 2026-10-10, after the user had to say "Stop!!!" twice in
one session and wait for an external process-kill each time. Same real-
HTTPServer-in-a-thread pattern as test_action_log_endpoint.py.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TEST_WORKER_ID = 93


@pytest.fixture
def worker_server():
    from applypilot.apply import launcher

    port = launcher._start_worker_listener(TEST_WORKER_ID)
    time.sleep(0.1)

    yield port

    launcher._stop_worker_listener(TEST_WORKER_ID)
    with launcher._worker_state_lock:
        launcher._worker_state.pop(TEST_WORKER_ID, None)
    with launcher._claude_lock:
        launcher._claude_procs.pop(TEST_WORKER_ID, None)
    # _stop_event is a module-level global shared by every test in the
    # process -- leaving it set would silently break every later test that
    # relies on the worker loop actually running (e.g. "queue empty, stop
    # was requested" firing immediately). Must reset, not just trust the
    # next test to clear it first.
    launcher._stop_event.clear()


def _post(port: int, path: str) -> bytes:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read()


def test_sets_the_global_stop_event(worker_server):
    from applypilot.apply import launcher

    assert not launcher._stop_event.is_set()

    _post(worker_server, "/api/emergency-stop")

    assert launcher._stop_event.is_set()


def test_kills_a_live_claude_subprocess(worker_server):
    """The actual point of this button: a live, running process must
    actually die, not just get a polite stop signal it can ignore."""
    from applypilot.apply import launcher

    dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with launcher._claude_lock:
        launcher._claude_procs[TEST_WORKER_ID] = dummy
    try:
        assert dummy.poll() is None  # confirmed alive before the test

        _post(worker_server, "/api/emergency-stop")

        deadline = time.monotonic() + 5
        while dummy.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert dummy.poll() is not None, "the live subprocess was still running after emergency-stop"
    finally:
        if dummy.poll() is None:
            dummy.kill()
            dummy.wait()


def test_response_reports_which_workers_were_stopped(worker_server):
    from applypilot.apply import launcher

    dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with launcher._claude_lock:
        launcher._claude_procs[TEST_WORKER_ID] = dummy
    try:
        import json

        body = _post(worker_server, "/api/emergency-stop")
        result = json.loads(body)
        assert result["ok"] is True
        assert TEST_WORKER_ID in result["stopped_workers"]
    finally:
        if dummy.poll() is None:
            dummy.kill()
            dummy.wait()


def test_does_not_crash_with_no_live_processes(worker_server):
    """The common case: nothing is actually running when STOP is clicked
    (e.g. waiting for a human, not mid-automation). Must not error."""
    import json

    body = _post(worker_server, "/api/emergency-stop")
    result = json.loads(body)
    assert result["ok"] is True
    assert result["stopped_workers"] == []


def test_updates_worker_status_for_dashboard_visibility(worker_server):
    from applypilot.apply import launcher

    _post(worker_server, "/api/emergency-stop")

    with launcher._worker_state_lock:
        status = launcher._worker_state[TEST_WORKER_ID]["status"]
    assert status == "stopped_by_user"
