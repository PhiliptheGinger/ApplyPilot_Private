"""Regression test for the 2026-09-19 Playwright-MCP-connect retry (decisions
#165-167) and the 2026-09-20 retry-count bump (decision #171): `run_job` used
to treat a failed Playwright MCP handshake (reported in the session's own
`system:init` event as `{"name": "playwright", "status": "failed"}`) as an
ordinary job failure, burning a whole apply attempt on what live testing
showed is often a transient race. `run_job` now retries the Claude subprocess
spawn up to twice more (same already-running Chrome, same MCP config, 3
attempts total) before giving up -- bumped from 1 retry (2 attempts total)
after live testing on 2026-09-20 found the single retry could still hit the
same race twice in a row.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import launcher


def _init_line(playwright_status: str) -> str:
    return json.dumps(
        {
            "type": "system",
            "subtype": "init",
            "mcp_servers": [{"name": "playwright", "status": playwright_status}, {"name": "gmail", "status": "connected"}],
        }
    )


def _result_line(text: str) -> str:
    return json.dumps({"type": "result", "usage": {}, "total_cost_usd": 0.01, "num_turns": 1, "result": text})


class _FakeProc:
    def __init__(self, lines: list[str]):
        self.stdout = iter(lines)
        self.stdin = io.StringIO()
        self.returncode = 0
        self.pid = 4242

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


@pytest.fixture
def _patched_run_job(monkeypatch, tmp_path):
    from applypilot import config

    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(config, "APPLY_WORKER_DIR", tmp_path)
    (tmp_path / "worker-0").mkdir(exist_ok=True)

    monkeypatch.setattr(launcher, "_reset_browser_tabs", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "reset_worker_dir", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_refresh_gmail_token", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_make_mcp_config", lambda *a, **k: {})
    monkeypatch.setattr(launcher, "_activate_agent_tab", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "get_state", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "update_state", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "add_event", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "detect_ats", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_parse_account_created", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_parse_qa_lines", lambda *a, **k: None)
    monkeypatch.setattr(launcher.prompt_mod, "build_prompt", lambda **k: "AGENT PROMPT")

    from applypilot import claude_status

    monkeypatch.setattr(claude_status, "record_apply_success", lambda *a, **k: None)
    monkeypatch.setattr(launcher.time, "sleep", lambda *a, **k: None)  # skip the real 2s inter-retry delay
    return tmp_path


def _job():
    return {"title": "Test Job", "site": "greenhouse", "url": "https://example.com/job/1", "fit_score": 9}


class TestMcpConnectRetry:
    def test_retries_once_and_succeeds_on_second_attempt(self, _patched_run_job, monkeypatch):
        attempt1 = _FakeProc([_init_line("failed"), '{"type":"assistant","message":{"content":[{"type":"text","text":"no tools"}]}}', _result_line("RESULT:FAILED:browser_tool_unavailable")])
        attempt2 = _FakeProc([_init_line("connected"), _result_line("RESULT:APPLIED")])
        procs = [attempt1, attempt2]
        popen_calls = []

        def _fake_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            return procs.pop(0)

        monkeypatch.setattr(launcher.subprocess, "Popen", _fake_popen)

        status, _duration_ms, _screening = launcher.run_job(_job(), port=9222, worker_id=0)

        assert len(popen_calls) == 2, "expected to stop retrying once the second attempt succeeds"
        assert status == "applied"

    def test_retries_twice_and_succeeds_on_third_attempt(self, _patched_run_job, monkeypatch):
        attempt1 = _FakeProc([_init_line("failed"), _result_line("RESULT:FAILED:browser_tool_unavailable")])
        attempt2 = _FakeProc([_init_line("failed"), _result_line("RESULT:FAILED:browser_tool_unavailable")])
        attempt3 = _FakeProc([_init_line("connected"), _result_line("RESULT:APPLIED")])
        procs = [attempt1, attempt2, attempt3]
        popen_calls = []

        def _fake_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            return procs.pop(0)

        monkeypatch.setattr(launcher.subprocess, "Popen", _fake_popen)

        status, _duration_ms, _screening = launcher.run_job(_job(), port=9222, worker_id=0)

        assert len(popen_calls) == 3, "expected two retries (three total Popen calls) before giving up"
        assert status == "applied"

    def test_no_retry_when_mcp_connects_on_first_attempt(self, _patched_run_job, monkeypatch):
        attempt1 = _FakeProc([_init_line("connected"), _result_line("RESULT:FAILED:some_unrelated_reason")])
        procs = [attempt1]
        popen_calls = []

        def _fake_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            return procs.pop(0)

        monkeypatch.setattr(launcher.subprocess, "Popen", _fake_popen)

        status, _duration_ms, _screening = launcher.run_job(_job(), port=9222, worker_id=0)

        assert len(popen_calls) == 1, "must not retry when Playwright connected fine the first time"
        assert status == "failed:some_unrelated_reason"

    def test_gives_up_after_max_attempts_still_failing(self, _patched_run_job, monkeypatch):
        attempt1 = _FakeProc([_init_line("failed"), _result_line("RESULT:FAILED:browser_tool_unavailable")])
        attempt2 = _FakeProc([_init_line("failed"), _result_line("RESULT:FAILED:browser_tool_unavailable")])
        attempt3 = _FakeProc([_init_line("failed"), _result_line("RESULT:FAILED:browser_tool_unavailable")])
        procs = [attempt1, attempt2, attempt3]
        popen_calls = []

        def _fake_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            return procs.pop(0)

        monkeypatch.setattr(launcher.subprocess, "Popen", _fake_popen)

        status, _duration_ms, _screening = launcher.run_job(_job(), port=9222, worker_id=0)

        assert len(popen_calls) == 3, "must not retry more than twice (bounded, not infinite)"
        assert status == "failed:browser_tool_unavailable"
