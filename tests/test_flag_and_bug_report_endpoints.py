"""Integration tests: POST /api/flag-job and /api/bug-report (FW47) via the
always-on worker server -- same real-HTTPServer pattern as
test_action_log_endpoint.py.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TEST_WORKER_ID = 92


@pytest.fixture
def worker_server():
    from applypilot.apply import launcher

    # _start_worker_listener builds its own fresh `state` dict and installs
    # it into _worker_state -- any pre-seeding before this call is clobbered,
    # so job/status must be set on the dict it actually creates, afterward.
    port = launcher._start_worker_listener(TEST_WORKER_ID)
    time.sleep(0.1)
    with launcher._worker_state_lock:
        launcher._worker_state[TEST_WORKER_ID]["job"] = {
            "title": "Test Job",
            "url": "https://example.com/job/1",
        }
        launcher._worker_state[TEST_WORKER_ID]["status"] = "applying"

    yield port

    launcher._stop_worker_listener(TEST_WORKER_ID)
    with launcher._worker_state_lock:
        launcher._worker_state.pop(TEST_WORKER_ID, None)


def _post(port: int, path: str, payload: dict) -> bytes:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read()


def _post_json(port: int, path: str, payload: dict) -> dict:
    body = _post(port, path, payload)
    return json.loads(body) if body else {}


class TestFlagJobEndpoint:
    def test_writes_reason_to_db(self, tmp_db, seed_job, worker_server):
        conn = tmp_db()
        row = seed_job(conn, url="https://example.com/job/flagme")

        _post(worker_server, "/api/flag-job", {"url": row["url"], "reason": "score too high"})

        db_row = conn.execute(
            "SELECT user_flagged_reason, user_flagged_at FROM jobs WHERE url = ?", (row["url"],)
        ).fetchone()
        assert db_row["user_flagged_reason"] == "score too high"
        assert db_row["user_flagged_at"] is not None

    def test_empty_reason_still_records_a_flag(self, tmp_db, seed_job, worker_server):
        """An empty reason (user just clicked Submit) should still land a
        flag, not silently no-op -- the job is visibly flagged either way."""
        conn = tmp_db()
        row = seed_job(conn, url="https://example.com/job/flagme2")

        _post(worker_server, "/api/flag-job", {"url": row["url"], "reason": ""})

        db_row = conn.execute("SELECT user_flagged_reason FROM jobs WHERE url = ?", (row["url"],)).fetchone()
        assert db_row["user_flagged_reason"]

    def test_missing_url_returns_400(self, worker_server):
        import urllib.error

        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(worker_server, "/api/flag-job", {"reason": "no url given"})
        assert exc.value.code == 400


class TestBugReportEndpoint:
    def test_delegates_to_github_issue_with_worker_context(self, monkeypatch, worker_server):
        from applypilot.tracking import github_issue

        captured = {}

        def _fake_file_bug_report(description, context=None):
            captured["description"] = description
            captured["context"] = context
            return True, "https://github.com/owner/repo/issues/7"

        monkeypatch.setattr(github_issue, "file_bug_report", _fake_file_bug_report)

        result = _post_json(worker_server, "/api/bug-report", {"description": "banner froze"})

        assert result == {"ok": True, "detail": "https://github.com/owner/repo/issues/7"}
        assert captured["description"] == "banner froze"
        assert captured["context"]["Job"] == "Test Job"
        assert captured["context"]["Job URL"] == "https://example.com/job/1"
        assert captured["context"]["Worker"] == f"W{TEST_WORKER_ID}"

    def test_surfaces_failure_detail(self, monkeypatch, worker_server):
        from applypilot.tracking import github_issue

        monkeypatch.setattr(
            github_issue, "file_bug_report", lambda description, context=None: (False, "APPLYPILOT_GITHUB_TOKEN not set")
        )

        result = _post_json(worker_server, "/api/bug-report", {"description": "x"})

        assert result == {"ok": False, "detail": "APPLYPILOT_GITHUB_TOKEN not set"}
