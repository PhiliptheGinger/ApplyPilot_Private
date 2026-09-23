"""Regression test for the 2026-09-22 apply-agent tool-restriction fix.

Real, live incident: a worker whose Playwright MCP connection failed did
NOT report RESULT:FAILED like its sibling workers -- it used its
unrestricted Bash/PowerShell/Write tool access (part of Claude Code's full
default toolset; the apply-agent's `--disallowedTools` list never touched
these) to write and run its own separate `playwright.async_api` script,
attempting real account creation and form submission on Motorola's live
Workday site, entirely outside `inject_dry_run_gate()` (which is only
injected into the sanctioned Chrome-for-Testing/CDP session). Caught live
before it reached a real Submit click -- verified via the actual last
screenshot (still on an empty step-1 form) and a DB check (no new account
row) -- but the gap was real.

Fixed by switching from a blacklist (`--disallowedTools`) to an explicit
whitelist (`--allowedTools`): the apply-agent now gets ONLY the
`mcp__playwright__browser_*` tools it actually uses plus the 3 Gmail
read/send tools -- no Bash, PowerShell, Write, Edit, Read, WebFetch, or
WebSearch, so there is no way to write or run an alternate automation
script regardless of what the model decides to try when its sanctioned
tools fail.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import launcher


def _init_line() -> str:
    return json.dumps({"type": "system", "subtype": "init", "mcp_servers": [{"name": "playwright", "status": "connected"}]})


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
    monkeypatch.setattr(launcher, "_parse_account_created", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_parse_qa_lines", lambda *a, **k: None)
    monkeypatch.setattr(launcher.prompt_mod, "build_prompt", lambda **k: "AGENT PROMPT")
    monkeypatch.setattr(launcher, "detect_ats", lambda url: None)

    from applypilot import claude_status

    monkeypatch.setattr(claude_status, "record_apply_success", lambda *a, **k: None)
    monkeypatch.setattr(launcher.time, "sleep", lambda *a, **k: None)
    return tmp_path


def _job(url="https://job-boards.greenhouse.io/acme/jobs/1"):
    return {"title": "Test Job", "site": "greenhouse", "url": url, "fit_score": 9}


def _capture_popen(monkeypatch):
    popen_calls = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return _FakeProc([_init_line(), _result_line("RESULT:FAILED:some_reason")])

    monkeypatch.setattr(launcher.subprocess, "Popen", _fake_popen)
    return popen_calls


def _extract_tool_list(cmd: list[str], flag: str) -> set[str]:
    idx = cmd.index(flag)
    return set(cmd[idx + 1].split(","))


class TestApplyAgentToolAllowlist:
    def test_allowedTools_flag_is_present(self, _patched_run_job, monkeypatch):
        calls = _capture_popen(monkeypatch)
        launcher.run_job(_job(), port=9222, worker_id=0, model="sonnet")
        assert "--allowedTools" in calls[0]

    def test_allowlist_contains_only_expected_browser_and_gmail_tools(self, _patched_run_job, monkeypatch):
        calls = _capture_popen(monkeypatch)
        launcher.run_job(_job(), port=9222, worker_id=0, model="sonnet")
        allowed = _extract_tool_list(calls[0], "--allowedTools")

        assert allowed == {
            "mcp__playwright__browser_click",
            "mcp__playwright__browser_close",
            "mcp__playwright__browser_console_messages",
            "mcp__playwright__browser_drag",
            "mcp__playwright__browser_drop",
            "mcp__playwright__browser_evaluate",
            "mcp__playwright__browser_file_upload",
            "mcp__playwright__browser_fill_form",
            "mcp__playwright__browser_handle_dialog",
            "mcp__playwright__browser_hover",
            "mcp__playwright__browser_navigate",
            "mcp__playwright__browser_navigate_back",
            "mcp__playwright__browser_network_request",
            "mcp__playwright__browser_network_requests",
            "mcp__playwright__browser_press_key",
            "mcp__playwright__browser_resize",
            "mcp__playwright__browser_run_code_unsafe",
            "mcp__playwright__browser_select_option",
            "mcp__playwright__browser_snapshot",
            "mcp__playwright__browser_tabs",
            "mcp__playwright__browser_take_screenshot",
            "mcp__playwright__browser_type",
            "mcp__playwright__browser_wait_for",
            "mcp__gmail__search_emails",
            "mcp__gmail__read_email",
            "mcp__gmail__send_email",
        }

    def test_general_purpose_escape_hatch_tools_are_never_allowed(self, _patched_run_job, monkeypatch):
        """The exact tools the real 2026-09-22 incident used to write and run
        an unauthorized, ungated Playwright script must never appear in the
        allowlist, regardless of future edits to it."""
        calls = _capture_popen(monkeypatch)
        launcher.run_job(_job(), port=9222, worker_id=0, model="sonnet")
        allowed = _extract_tool_list(calls[0], "--allowedTools")

        forbidden = {"Bash", "PowerShell", "Write", "Edit", "Read", "WebFetch", "WebSearch", "Task", "Skill", "NotebookEdit"}
        assert allowed & forbidden == set()

    def test_browser_install_still_excluded(self, _patched_run_job, monkeypatch):
        """browser_install restarts the browser in CDP mode, breaking the
        session (decision predates this fix) -- confirm it's excluded by
        the new allowlist too, not just the old blacklist."""
        calls = _capture_popen(monkeypatch)
        launcher.run_job(_job(), port=9222, worker_id=0, model="sonnet")
        allowed = _extract_tool_list(calls[0], "--allowedTools")

        assert "mcp__playwright__browser_install" not in allowed

    def test_disallowedTools_still_present_as_defense_in_depth(self, _patched_run_job, monkeypatch):
        calls = _capture_popen(monkeypatch)
        launcher.run_job(_job(), port=9222, worker_id=0, model="sonnet")
        assert "--disallowedTools" in calls[0]
        disallowed = _extract_tool_list(calls[0], "--disallowedTools")
        assert "mcp__gmail__draft_email" in disallowed
