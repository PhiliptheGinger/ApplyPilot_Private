"""Regression test for the 2026-09-19 speed escalation trigger: `run_job`
downgrades from the CLI-default "sonnet" to "haiku" when a fresh
successful_paths memo already exists for the job's ATS (a known-shape
page doesn't need sonnet's extra reasoning budget), mirroring the
fast/slow escalation pattern already proven for scoring (decisions
#76-81). An explicit non-default --model choice is never overridden.
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


class TestHaikuEscalation:
    def test_uses_haiku_when_fresh_memo_exists(self, _patched_run_job, monkeypatch):
        monkeypatch.setattr(launcher, "detect_ats", lambda url: "greenhouse")
        import applypilot.apply.successful_paths as sp

        monkeypatch.setattr(sp, "load_path", lambda ats: {"ats_slug": ats, "steps": []})
        calls = _capture_popen(monkeypatch)

        launcher.run_job(_job(), port=9222, worker_id=0, model="sonnet")

        assert "--model" in calls[0]
        assert calls[0][calls[0].index("--model") + 1] == "haiku"

    def test_stays_sonnet_when_no_memo(self, _patched_run_job, monkeypatch):
        monkeypatch.setattr(launcher, "detect_ats", lambda url: "greenhouse")
        import applypilot.apply.successful_paths as sp

        monkeypatch.setattr(sp, "load_path", lambda ats: None)
        calls = _capture_popen(monkeypatch)

        launcher.run_job(_job(), port=9222, worker_id=0, model="sonnet")

        assert calls[0][calls[0].index("--model") + 1] == "sonnet"

    def test_explicit_model_choice_is_never_overridden(self, _patched_run_job, monkeypatch):
        monkeypatch.setattr(launcher, "detect_ats", lambda url: "greenhouse")
        import applypilot.apply.successful_paths as sp

        monkeypatch.setattr(sp, "load_path", lambda ats: {"ats_slug": ats, "steps": []})
        calls = _capture_popen(monkeypatch)

        launcher.run_job(_job(), port=9222, worker_id=0, model="opus")

        assert calls[0][calls[0].index("--model") + 1] == "opus"
