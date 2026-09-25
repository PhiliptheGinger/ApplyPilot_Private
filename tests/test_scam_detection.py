"""Regression tests for the application-response-email scam detector
(CLAUDE.md Future Work item 34, built 2026-09-25).

Grounded in the real Forestar Group incident (decision #174): a fake
interview invite sent from a personal Gmail address impersonating a real
company, bundling several unrelated generic job titles. Requires at
least 2 independent signal categories to co-occur, per the module's own
documented false-positive caution (a real check against this pipeline's
job-posting corpus found several intuitive single-keyword scam phrases
are dangerous false-positive traps on legitimate text).
"""

from __future__ import annotations

from applypilot.tracking.scam_detection import detect_scam_signals


def _make_email(**overrides) -> dict:
    email = {
        "sender": "noreply@acme.com",
        "subject": "Thank you for applying to Acme",
        "body": "We received your application and will be in touch.",
    }
    email.update(overrides)
    return email


def test_legitimate_corporate_email_flags_nothing():
    email = _make_email(
        sender="careers@acme.com",
        subject="Interview invitation - Acme Corp",
        body="We'd like to invite you to interview for the Software Engineer role.",
    )
    assert detect_scam_signals(email) == []


def test_real_forestar_style_scam_flags_two_signals():
    email = _make_email(
        sender="sarahgross201@gmail.com",
        subject="You're Invited to Interview with Forestar Group",
        body=(
            "On behalf of Forestar Group, we are pleased to invite you to interview "
            "for one of these roles: Remote Data Entry Clerk, Customer Service "
            "Representative, and Administrative positions. This is strictly "
            "work-from-home and requires training."
        ),
    )
    signals = detect_scam_signals(email)
    assert "personal_email_claiming_corporate" in signals
    assert "bundled_generic_titles" in signals


def test_single_signal_alone_is_not_enough():
    # Personal-domain sender but no corporate-claiming language at all --
    # a real friend/personal contact, not a scam signal on its own.
    email = _make_email(sender="friend@gmail.com", subject="hey", body="lunch tomorrow?")
    assert detect_scam_signals(email) == []


def test_personal_domain_alone_without_corporate_claim_is_not_enough():
    email = _make_email(
        sender="someone@yahoo.com",
        subject="Quick question",
        body="Just following up on our chat.",
    )
    assert detect_scam_signals(email) == []


def test_upfront_payment_plus_urgency_flags():
    email = _make_email(
        sender="hr@totallyrealcompany.com",
        subject="Job Offer - Act Now",
        body=(
            "Congratulations, you are hired! You must purchase your own equipment "
            "before starting. Please respond within 24 hours to confirm."
        ),
    )
    signals = detect_scam_signals(email)
    assert "upfront_payment_language" in signals
    assert "urgency_pressure" in signals


def test_no_sender_or_body_does_not_crash():
    assert detect_scam_signals({}) == []


def test_process_classified_email_stores_scam_signals_in_extracted_data(tmp_db, seed_job, monkeypatch):
    import json
    from datetime import UTC, datetime

    from applypilot.tracking import _process_classified_email

    conn = tmp_db()
    job = seed_job(conn, url_suffix="scam-flag-test", state="applied", applied_at=datetime.now(UTC).isoformat())

    monkeypatch.setattr(
        "applypilot.tracking.matcher.match_email_to_job",
        lambda email, applied_jobs: {"job_url": job["url"], "score": 100, "signals": ["forced"]},
    )

    applied_jobs = [{"url": job["url"], "title": job["title"], "company": job["company"], "test_case": 0}]
    result = {"classification": "confirmation", "people": [], "dates": [], "action_items": [], "summary": ""}
    counters: dict = {"matched": 0, "stubs": 0}
    email = {
        "id": "msg-scam-1",
        "sender": "sarahgross201@gmail.com",
        "subject": "You're Invited to Interview",
        "body": (
            "On behalf of the company, we are pleased to invite you to interview for "
            "Remote Data Entry Clerk, Customer Service Representative, and Administrative roles."
        ),
    }

    _process_classified_email(email, result, applied_jobs, dry_run=False, conn=conn, counters=counters)

    row = conn.execute("SELECT extracted_data FROM tracking_emails WHERE email_id=?", ("msg-scam-1",)).fetchone()
    stored = json.loads(row["extracted_data"])
    assert "personal_email_claiming_corporate" in stored["scam_signals"]
    assert "bundled_generic_titles" in stored["scam_signals"]
    assert counters["scam_flagged"] == 1
