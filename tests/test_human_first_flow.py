"""Tests for hitl.run_human_first (CLAUDE.md Future Work item 40).

Covers the three outcomes a human can produce after being shown the
human-first banner (applied directly / handed off to automation / released
after timeout), plus the stop-event short-circuit — mirroring
test_apply_mcp_connect_retry.py's mock-based style rather than driving a
real Chrome/Node process.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import hitl, launcher


def _job(url="https://www.linkedin.com/jobs/view/4380167634"):
    return {"url": url, "title": "Test Job", "site": "linkedin", "fit_score": 9}


@pytest.fixture
def _patched_banner(monkeypatch):
    """No real Chrome/Node: banner injection, navigation, and the CSP-
    fallback watcher are all no-ops that just need to not crash."""
    monkeypatch.setattr(hitl, "launch_chrome", lambda *a, **k: None)
    # NOTE: deliberately NOT patching time.sleep here -- hitl.time IS the
    # global `time` module (plain `import time`), so monkeypatching
    # hitl.time.sleep patches time.sleep process-wide, including this
    # test file's own delay-based thread synchronization below. The real
    # ~1s post-navigate sleep in run_human_first is harmless to eat per test.

    import applypilot.apply.human_review as human_review

    monkeypatch.setattr(human_review, "_navigate_chrome", lambda *a, **k: True)
    monkeypatch.setattr(human_review, "_inject_human_first_banner", lambda *a, **k: True)
    monkeypatch.setattr(human_review, "_start_human_first_done_watcher", lambda *a, **k: None)


@pytest.fixture
def _worker_state(monkeypatch):
    """Register worker 0 in launcher's always-on state dict, exactly like
    _start_worker_listener does in production — run_human_first's listener
    registration (_start_human_first_listener) requires this to exist."""
    with launcher._worker_state_lock:
        launcher._worker_state[0] = {}
    yield
    with launcher._worker_state_lock:
        launcher._worker_state.pop(0, None)


def _fire_banner_click(worker_id: int, outcome: str, url: str | None = None, delay: float = 1.3) -> None:
    """Simulate the banner's HTTP POST /api/human-first/{hash} — exactly
    what _handle_human_first does, just from a background thread instead
    of a real HTTP request."""

    def _fire():
        time.sleep(delay)
        with launcher._worker_state_lock:
            state = launcher._worker_state.get(worker_id)
        assert state is not None, "worker state not registered before banner click fired"
        state["human_first_result"] = {"outcome": outcome, "url": url}
        evt = state.get("human_first_event")
        assert evt is not None, "run_human_first hadn't registered its event yet"
        evt.set()

    threading.Thread(target=_fire, daemon=True).start()


class TestRunHumanFirst:
    def test_applied_outcome(self, tmp_db, _patched_banner, _worker_state):
        tmp_db()
        job = _job()
        _fire_banner_click(0, "applied")

        outcome, returned_job = hitl.run_human_first(worker_id=0, port=9999, job=job, poll_interval=0.05)

        assert outcome == "applied"
        assert returned_job["url"] == job["url"]

    def test_handoff_outcome_updates_application_url(self, tmp_db, seed_job, _patched_banner, _worker_state):
        conn = tmp_db()
        row = seed_job(
            conn,
            url="https://www.linkedin.com/jobs/view/4380167634",
            application_url=None,
            state="manual_only",
        )
        job = dict(row)
        ats_url = "https://boards.greenhouse.io/acme/jobs/99"
        _fire_banner_click(0, "handoff", url=ats_url)

        outcome, returned_job = hitl.run_human_first(worker_id=0, port=9999, job=job, poll_interval=0.05)

        assert outcome == "handoff"
        assert returned_job["application_url"] == ats_url
        # Persisted to the DB too — a real, useful side effect independent
        # of this job's own eventual outcome (Future Work item 38).
        db_row = conn.execute("SELECT application_url FROM jobs WHERE url = ?", (job["url"],)).fetchone()
        assert db_row["application_url"] == ats_url

    def test_released_on_timeout(self, tmp_db, _patched_banner, _worker_state):
        tmp_db()
        job = _job()
        # No banner click fired — the wait must time out and release, not hang.

        outcome, returned_job = hitl.run_human_first(
            worker_id=0, port=9999, job=job, timeout_seconds=0.05, poll_interval=0.02
        )

        assert outcome == "released"
        assert returned_job["url"] == job["url"]

    def test_stopped_short_circuits_immediately(self, tmp_db, _patched_banner, _worker_state):
        tmp_db()
        job = _job()
        stop_event = threading.Event()
        stop_event.set()

        outcome, _ = hitl.run_human_first(
            worker_id=0, port=9999, job=job, stop_event=stop_event, poll_interval=0.02
        )

        assert outcome == "stopped"

    def test_unrecognized_signal_is_treated_as_released(self, tmp_db, _patched_banner, _worker_state):
        """A signal missing outcome/url (e.g. a malformed POST body) must
        never be silently treated as a green light to proceed with
        automation — default to the safe (released) outcome."""
        tmp_db()
        job = _job()
        _fire_banner_click(0, "handoff", url=None)  # handoff claimed but no URL captured

        outcome, _ = hitl.run_human_first(worker_id=0, port=9999, job=job, poll_interval=0.05)

        assert outcome == "released"

    def test_listener_state_cleaned_up_after_return(self, tmp_db, _patched_banner, _worker_state):
        tmp_db()
        job = _job()
        _fire_banner_click(0, "applied")

        hitl.run_human_first(worker_id=0, port=9999, job=job, poll_interval=0.05)

        with launcher._worker_state_lock:
            state = launcher._worker_state.get(0)
        assert state["human_first_event"] is None
        assert state["human_first_job_hash"] is None

    def test_banner_injection_retried_after_initial_failure(self, tmp_db, _worker_state, monkeypatch):
        """Real, live-caught bug (2026-09-23): injecting into a real, heavy
        LinkedIn page right after navigation lost the race against the page
        still loading -- the first attempt failed, and with no retry the
        human was left staring at a banner-less page for the full timeout,
        indistinguishable from a hang. A retry on the very next poll cycle
        must pick it up instead of silently giving up."""
        import applypilot.apply.human_review as human_review

        monkeypatch.setattr(hitl, "launch_chrome", lambda *a, **k: None)
        monkeypatch.setattr(human_review, "_navigate_chrome", lambda *a, **k: True)
        monkeypatch.setattr(human_review, "_start_human_first_done_watcher", lambda *a, **k: None)

        calls: list[int] = []

        def _flaky_inject(*a, **k):
            calls.append(1)
            return len(calls) >= 2  # fails once, succeeds on the retry

        monkeypatch.setattr(human_review, "_inject_human_first_banner", _flaky_inject)

        tmp_db()
        job = _job()
        _fire_banner_click(0, "applied", delay=2.5)  # after the 1s navigate-sleep + one 0.1s poll retry

        outcome, _ = hitl.run_human_first(worker_id=0, port=9999, job=job, poll_interval=0.1)

        assert outcome == "applied"
        assert len(calls) >= 2, "banner injection must be retried, not attempted only once"
