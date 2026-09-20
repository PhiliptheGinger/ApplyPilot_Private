"""Regression test for the 2026-09-19 Chrome-launch fix (decisions #165-166):
`launch_chrome()` used to return after a blind `time.sleep(3)` with no check
that the CDP debug port was actually reachable. Live 5-worker concurrency
testing showed individual Chrome launches taking 3.7s-10.6s under real
contention, so a fixed 3s wait can return before Chrome is ready.
`_wait_for_cdp_ready` polls the real port instead of guessing a fixed delay.

2026-09-20 (decision #176): strengthened to also open a real WebSocket CDP
session and round-trip a `Target.getTargets` command before declaring the
port ready -- the original HTTP-only check only proved the discovery
endpoint was up, not that the browser was actually accepting CDP sessions
(the real thing Playwright MCP needs). These tests fake the WS layer via
monkeypatch (a real WS server is unnecessary weight for a unit test) rather
than the HTTP-only fakes the pre-#176 version of this file used.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import websocket  # type: ignore

from applypilot.apply.chrome import _wait_for_cdp_ready


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"webSocketDebuggerUrl": "ws://127.0.0.1:1/fake"}).encode())

    def log_message(self, *args):
        pass  # keep test output quiet


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeWs:
    """Stands in for a real CDP WebSocket session: replies to whatever
    request id it's sent with a well-formed {"id": ..., "result": {}}."""

    def __init__(self):
        self._last_id = None

    def send(self, payload):
        self._last_id = json.loads(payload)["id"]

    def recv(self):
        return json.dumps({"id": self._last_id, "result": {"targetInfos": []}})

    def close(self):
        pass


def _patch_working_ws(monkeypatch):
    monkeypatch.setattr(websocket, "create_connection", lambda url, timeout=None: _FakeWs())


class TestWaitForCdpReady:
    def test_returns_true_once_port_responds(self, monkeypatch):
        _patch_working_ws(monkeypatch)
        port = _free_port()
        server = http.server.HTTPServer(("127.0.0.1", port), _OkHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            start = time.time()
            assert _wait_for_cdp_ready(port, timeout=5.0, poll_interval=0.1) is True
            assert time.time() - start < 5.0
        finally:
            server.shutdown()

    def test_returns_true_even_if_port_is_slow_to_come_up(self, monkeypatch):
        _patch_working_ws(monkeypatch)
        port = _free_port()
        server_holder: dict = {}

        def _delayed_start():
            time.sleep(1.0)
            server = http.server.HTTPServer(("127.0.0.1", port), _OkHandler)
            server_holder["server"] = server
            server.serve_forever()

        t = threading.Thread(target=_delayed_start, daemon=True)
        t.start()
        try:
            # A blind sleep(1) or less would have missed this -- confirms the
            # poll loop keeps trying past a naive fixed short delay.
            assert _wait_for_cdp_ready(port, timeout=5.0, poll_interval=0.1) is True
        finally:
            server_holder.get("server") and server_holder["server"].shutdown()

    def test_returns_false_when_nothing_ever_listens(self):
        port = _free_port()  # guaranteed free, nothing bound to it
        start = time.time()
        assert _wait_for_cdp_ready(port, timeout=1.0, poll_interval=0.1) is False
        assert time.time() - start >= 1.0

    def test_returns_false_when_http_is_up_but_ws_never_accepts(self, monkeypatch):
        """The real gap decision #176 closed: HTTP discovery responding is
        not sufficient on its own -- if the WS endpoint refuses every
        connection attempt, this must keep polling (and eventually give up),
        not declare victory on the HTTP response alone."""
        monkeypatch.setattr(
            websocket,
            "create_connection",
            lambda url, timeout=None: (_ for _ in ()).throw(OSError("WS refused")),
        )
        port = _free_port()
        server = http.server.HTTPServer(("127.0.0.1", port), _OkHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            start = time.time()
            assert _wait_for_cdp_ready(port, timeout=1.0, poll_interval=0.1) is False
            assert time.time() - start >= 1.0
        finally:
            server.shutdown()
