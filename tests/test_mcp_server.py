"""Persistent Playwright MCP server per worker (decision #235)."""

from __future__ import annotations

import pytest

from applypilot.apply import launcher, mcp_server


class _FakeProc:
    def __init__(self, alive: bool = True):
        self.alive = alive
        self.pid = 4242

    def poll(self):
        return None if self.alive else 1


@pytest.fixture
def fake_servers(monkeypatch):
    """Replace process start/kill/probe with fakes; returns a record of starts and kills."""
    record = {"starts": [], "kills": [], "answering": True}

    def _start(worker_id, cdp_port):
        srv = mcp_server._Server(proc=_FakeProc(), port=mcp_server.mcp_port(worker_id), cdp_port=cdp_port)
        record["starts"].append((worker_id, cdp_port))
        return srv

    def _kill(srv):
        srv.proc.alive = False
        record["kills"].append(srv.port)

    monkeypatch.setattr(mcp_server, "_start", _start)
    monkeypatch.setattr(mcp_server, "_kill", _kill)
    monkeypatch.setattr(mcp_server, "_is_answering", lambda port, timeout=2.0: record["answering"])
    monkeypatch.setattr(mcp_server, "_servers", {})
    return record


def test_started_once_and_reused_across_jobs(fake_servers):
    url1 = mcp_server.ensure_playwright_mcp(0, 9222)
    url2 = mcp_server.ensure_playwright_mcp(0, 9222)
    assert url1 == url2 == "http://localhost:8931/mcp"
    assert fake_servers["starts"] == [(0, 9222)]


def test_each_worker_gets_its_own_port(fake_servers):
    assert mcp_server.ensure_playwright_mcp(1, 9223) == "http://localhost:8932/mcp"
    assert mcp_server.ensure_playwright_mcp(1001, 10223) == "http://localhost:9932/mcp"


def test_restarted_when_unhealthy(fake_servers):
    mcp_server.ensure_playwright_mcp(0, 9222)
    fake_servers["answering"] = False
    mcp_server.ensure_playwright_mcp(0, 9222)
    assert len(fake_servers["starts"]) == 2
    assert fake_servers["kills"] == [8931]


def test_restarted_when_process_died(fake_servers):
    mcp_server.ensure_playwright_mcp(0, 9222)
    mcp_server._servers[0].proc.alive = False
    mcp_server.ensure_playwright_mcp(0, 9222)
    assert len(fake_servers["starts"]) == 2


def test_restart_flag_and_cdp_port_change_restart(fake_servers):
    mcp_server.ensure_playwright_mcp(0, 9222)
    mcp_server.ensure_playwright_mcp(0, 9222, restart=True)
    mcp_server.ensure_playwright_mcp(0, 9300)
    assert fake_servers["starts"] == [(0, 9222), (0, 9222), (0, 9300)]


def test_returns_none_when_start_fails(fake_servers, monkeypatch):
    monkeypatch.setattr(mcp_server, "_start", lambda *a: None)
    assert mcp_server.ensure_playwright_mcp(0, 9222) is None
    assert 0 not in mcp_server._servers


def test_stop_all_kills_every_server(fake_servers):
    mcp_server.ensure_playwright_mcp(0, 9222)
    mcp_server.ensure_playwright_mcp(1, 9223)
    mcp_server.stop_all()
    assert sorted(fake_servers["kills"]) == [8931, 8932]
    assert mcp_server._servers == {}


def test_server_args_match_stdio_browser_flags():
    args = mcp_server.server_args(8931, 9222, (1280, 900), "UA/1.0")
    assert mcp_server.PLAYWRIGHT_MCP_PACKAGE in args
    assert "--port=8931" in args
    assert "--cdp-endpoint=http://localhost:9222" in args
    assert "--viewport-size=1280x900" in args
    assert "--user-agent=UA/1.0" in args


def test_mcp_config_http_entry(monkeypatch):
    monkeypatch.setattr("applypilot.apply.chrome._get_real_user_agent", lambda: "UA")
    cfg = launcher._make_mcp_config(9222, worker_id=0, mcp_url="http://localhost:8931/mcp")
    assert cfg["mcpServers"]["playwright"] == {"type": "http", "url": "http://localhost:8931/mcp"}
    assert cfg["mcpServers"]["gmail"]["command"] == "npx"


def test_mcp_config_stdio_fallback(monkeypatch):
    monkeypatch.setattr("applypilot.apply.chrome._get_real_user_agent", lambda: "UA")
    cfg = launcher._make_mcp_config(9222, worker_id=0)
    entry = cfg["mcpServers"]["playwright"]
    assert entry["command"] == "npx"
    assert mcp_server.PLAYWRIGHT_MCP_PACKAGE in entry["args"]
    assert "--cdp-endpoint=http://localhost:9222" in entry["args"]
