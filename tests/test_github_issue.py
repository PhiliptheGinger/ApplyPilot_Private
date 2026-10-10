"""FW47(a) (2026-10-09): in-banner bug-report icon files a GitHub issue."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.tracking import github_issue


class TestFileBugReport:
    def test_off_without_token(self, monkeypatch):
        monkeypatch.delenv("APPLYPILOT_GITHUB_TOKEN", raising=False)
        ok, detail = github_issue.file_bug_report("something broke")
        assert ok is False
        assert "APPLYPILOT_GITHUB_TOKEN" in detail

    def test_fails_cleanly_when_repo_unresolvable(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(github_issue, "_resolve_repo", lambda: None)
        ok, detail = github_issue.file_bug_report("something broke")
        assert ok is False
        assert "repo" in detail.lower()

    def test_successful_post_returns_issue_url(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(github_issue, "_resolve_repo", lambda: "owner/repo")

        captured = {}

        class _FakeResponse:
            status_code = 201

            def json(self):
                return {"html_url": "https://github.com/owner/repo/issues/42"}

        def _fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _FakeResponse()

        import httpx

        monkeypatch.setattr(httpx, "post", _fake_post)

        ok, detail = github_issue.file_bug_report("buttons unresponsive", context={"Worker": "W0"})

        assert ok is True
        assert detail == "https://github.com/owner/repo/issues/42"
        assert captured["url"] == "https://api.github.com/repos/owner/repo/issues"
        assert captured["headers"]["Authorization"] == "Bearer fake-token"
        assert "buttons unresponsive" in captured["json"]["title"]
        assert "W0" in captured["json"]["body"]

    def test_api_error_reported_not_raised(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(github_issue, "_resolve_repo", lambda: "owner/repo")

        class _FakeResponse:
            status_code = 401
            text = "Bad credentials"

        import httpx

        monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse())

        ok, detail = github_issue.file_bug_report("test")
        assert ok is False
        assert "401" in detail

    def test_network_failure_reported_not_raised(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_GITHUB_TOKEN", "fake-token")
        monkeypatch.setattr(github_issue, "_resolve_repo", lambda: "owner/repo")

        import httpx

        def _raise(*a, **k):
            raise httpx.ConnectError("no network")

        monkeypatch.setattr(httpx, "post", _raise)

        ok, detail = github_issue.file_bug_report("test")
        assert ok is False
        assert "no network" in detail.lower() or "connect" in detail.lower()


class TestResolveRepo:
    def test_parses_https_remote(self, monkeypatch):
        import subprocess

        class _Result:
            stdout = "https://github.com/PhiliptheGinger/ApplyPilot_Private.git\n"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        assert github_issue._resolve_repo() == "PhiliptheGinger/ApplyPilot_Private"

    def test_parses_ssh_remote(self, monkeypatch):
        import subprocess

        class _Result:
            stdout = "git@github.com:PhiliptheGinger/ApplyPilot_Private.git\n"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        assert github_issue._resolve_repo() == "PhiliptheGinger/ApplyPilot_Private"

    def test_no_remote_returns_none(self, monkeypatch):
        import subprocess

        def _raise(*a, **k):
            raise subprocess.CalledProcessError(1, "git")

        monkeypatch.setattr(subprocess, "run", _raise)
        assert github_issue._resolve_repo() is None
