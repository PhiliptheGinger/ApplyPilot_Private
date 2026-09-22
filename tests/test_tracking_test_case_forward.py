"""Regression tests for the test-case email-forwarding feature (decision #175's
outstanding ask, closed out 2026-09-22): a job deliberately marked `test_case=1`
(decision #175's low-score-job apply-mechanics testing mechanism) should have
any matched application-response email forwarded to the candidate's own inbox
with a "This job was applied for test purposes only" note -- and the original
sender must NEVER see any trace of this notification (no reply, no cc, no
threadId/inReplyTo tying it to their message).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


def _make_email(**overrides) -> dict:
    email = {
        "id": "msg-1",
        "thread_id": "thread-1",
        "subject": "Thank you for applying to Acme",
        "sender": "noreply@acme.com",
        "sender_name": "Acme Careers",
        "date": datetime.now(UTC).isoformat(),
        "snippet": "We received your application.",
        "body": "We received your application and will be in touch.",
    }
    email.update(overrides)
    return email


def test_forward_sends_only_to_self_with_test_note(monkeypatch):
    from applypilot.tracking import _forward_test_case_notification

    captured = {}

    async def fake_send_email(to, subject, body):
        captured["to"] = to
        captured["subject"] = subject
        captured["body"] = body
        return True, "Email sent successfully with ID: fake123"

    monkeypatch.setattr(
        "applypilot.config.load_profile",
        lambda: {"personal": {"email": "candidate@example.com"}},
    )
    monkeypatch.setattr("applypilot.tracking.gmail_client.send_email", fake_send_email)

    job = {"url": "https://boards.greenhouse.io/acme/jobs/1", "title": "SDR", "company": "acme"}
    email = _make_email()

    _forward_test_case_notification(job, email, "confirmation")

    assert captured["to"] == ["candidate@example.com"], "must go ONLY to the candidate's own address"
    assert "noreply@acme.com" not in captured["to"]
    assert "This job was applied for test purposes only." in captured["body"]
    assert "[TEST CASE]" in captured["subject"]


def test_forward_noop_when_profile_has_no_email(monkeypatch, caplog):
    from applypilot.tracking import _forward_test_case_notification

    called = {"n": 0}

    async def fake_send_email(to, subject, body):
        called["n"] += 1
        return True, "ok"

    monkeypatch.setattr("applypilot.config.load_profile", lambda: {"personal": {}})
    monkeypatch.setattr("applypilot.tracking.gmail_client.send_email", fake_send_email)

    job = {"url": "https://boards.greenhouse.io/acme/jobs/1", "title": "SDR", "company": "acme"}
    _forward_test_case_notification(job, _make_email(), "confirmation")

    assert called["n"] == 0


def test_process_classified_email_forwards_for_test_case_job(tmp_db, seed_job, monkeypatch):
    from applypilot.tracking import _process_classified_email

    conn = tmp_db()
    job = seed_job(
        conn,
        url_suffix="test-case-forward",
        state="applied",
        test_case=1,
        applied_at=datetime.now(UTC).isoformat(),
    )

    captured = {}

    async def fake_send_email(to, subject, body):
        captured["called"] = True
        captured["to"] = to
        return True, "ok"

    monkeypatch.setattr(
        "applypilot.config.load_profile",
        lambda: {"personal": {"email": "candidate@example.com"}},
    )
    monkeypatch.setattr("applypilot.tracking.gmail_client.send_email", fake_send_email)
    monkeypatch.setattr(
        "applypilot.tracking.matcher.match_email_to_job",
        lambda email, applied_jobs: {"job_url": job["url"], "score": 100, "signals": ["forced"]},
    )

    applied_jobs = [{"url": job["url"], "title": job["title"], "company": job["company"], "test_case": 1}]
    result = {"classification": "confirmation", "people": [], "dates": [], "action_items": [], "summary": ""}
    counters = {"matched": 0, "stubs": 0}

    _process_classified_email(_make_email(), result, applied_jobs, dry_run=False, conn=conn, counters=counters)

    assert captured.get("called") is True
    assert captured["to"] == ["candidate@example.com"]


def test_process_classified_email_does_not_forward_for_real_job(tmp_db, seed_job, monkeypatch):
    from applypilot.tracking import _process_classified_email

    conn = tmp_db()
    job = seed_job(
        conn,
        url_suffix="real-job-no-forward",
        state="applied",
        test_case=0,
        applied_at=datetime.now(UTC).isoformat(),
    )

    called = {"n": 0}

    async def fake_send_email(to, subject, body):
        called["n"] += 1
        return True, "ok"

    monkeypatch.setattr("applypilot.tracking.gmail_client.send_email", fake_send_email)
    monkeypatch.setattr(
        "applypilot.tracking.matcher.match_email_to_job",
        lambda email, applied_jobs: {"job_url": job["url"], "score": 100, "signals": ["forced"]},
    )

    applied_jobs = [{"url": job["url"], "title": job["title"], "company": job["company"], "test_case": 0}]
    result = {"classification": "confirmation", "people": [], "dates": [], "action_items": [], "summary": ""}
    counters = {"matched": 0, "stubs": 0}

    _process_classified_email(_make_email(), result, applied_jobs, dry_run=False, conn=conn, counters=counters)

    assert called["n"] == 0


def test_process_classified_email_dry_run_never_forwards(tmp_db, seed_job, monkeypatch):
    from applypilot.tracking import _process_classified_email

    conn = tmp_db()
    job = seed_job(
        conn,
        url_suffix="dry-run-no-forward",
        state="applied",
        test_case=1,
        applied_at=datetime.now(UTC).isoformat(),
    )

    called = {"n": 0}

    async def fake_send_email(to, subject, body):
        called["n"] += 1
        return True, "ok"

    monkeypatch.setattr("applypilot.tracking.gmail_client.send_email", fake_send_email)
    monkeypatch.setattr(
        "applypilot.tracking.matcher.match_email_to_job",
        lambda email, applied_jobs: {"job_url": job["url"], "score": 100, "signals": ["forced"]},
    )

    applied_jobs = [{"url": job["url"], "title": job["title"], "company": job["company"], "test_case": 1}]
    result = {"classification": "confirmation", "people": [], "dates": [], "action_items": [], "summary": ""}
    counters = {"matched": 0, "stubs": 0}

    _process_classified_email(_make_email(), result, applied_jobs, dry_run=True, conn=conn, counters=counters)

    assert called["n"] == 0
