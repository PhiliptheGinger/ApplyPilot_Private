"""Regression tests for `mark_job_as_test_case` (decision #175's reusable
architecture, added 2026-09-22): a CLI/DB-level tool to keep marking real,
genuinely low-scoring jobs as test cases for apply-mechanics testing, instead
of the one-off DB surgery decision #175 originally used. The whole point is a
job the candidate genuinely wouldn't want (the pipeline's own scorer already
said so) -- so this deliberately only accepts jobs already in state=low_score,
never invents a fake low score or a fake application_url.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


def test_marks_low_score_job_and_reuses_approved_materials(tmp_db, seed_job):
    from applypilot.database import mark_job_as_test_case

    conn = tmp_db()
    seed_job(
        conn,
        url_suffix="donor",
        state="applied",
        apply_status="applied",
        applied_at=datetime.now(UTC).isoformat(),
        tailored_resume_path="/tmp/approved_resume.pdf",
        cover_letter_path="/tmp/approved_cover.pdf",
        test_case=0,
    )
    target = seed_job(
        conn,
        url_suffix="target",
        state="low_score",
        fit_score=2,
        tailored_resume_path=None,
        cover_letter_path=None,
        application_url="https://boards.greenhouse.io/acme/jobs/99",
    )

    updated = mark_job_as_test_case(target["url"], conn)

    assert updated["test_case"] == 1
    assert updated["state"] == "ready_to_apply"
    assert updated["tailored_resume_path"] == "/tmp/approved_resume.pdf"
    assert updated["cover_letter_path"] == "/tmp/approved_cover.pdf"


def test_rejects_job_not_in_low_score_state(tmp_db, seed_job):
    from applypilot.database import mark_job_as_test_case

    conn = tmp_db()
    target = seed_job(conn, url_suffix="wrong-state", state="scored", fit_score=9)

    with pytest.raises(ValueError, match="not 'low_score'"):
        mark_job_as_test_case(target["url"], conn)


def test_rejects_job_with_no_application_url(tmp_db, seed_job):
    from applypilot.database import mark_job_as_test_case

    conn = tmp_db()
    seed_job(
        conn,
        url_suffix="donor2",
        state="applied",
        apply_status="applied",
        applied_at=datetime.now(UTC).isoformat(),
        tailored_resume_path="/tmp/approved_resume.pdf",
        cover_letter_path="/tmp/approved_cover.pdf",
        test_case=0,
    )
    target = seed_job(conn, url_suffix="no-url", state="low_score", fit_score=2, application_url=None)

    with pytest.raises(ValueError, match="application_url"):
        mark_job_as_test_case(target["url"], conn)


def test_rejects_when_no_donor_available(tmp_db, seed_job):
    from applypilot.database import mark_job_as_test_case

    conn = tmp_db()
    target = seed_job(
        conn,
        url_suffix="no-donor",
        state="low_score",
        fit_score=2,
        application_url="https://boards.greenhouse.io/acme/jobs/1",
    )

    with pytest.raises(ValueError, match="approved resume"):
        mark_job_as_test_case(target["url"], conn)


def test_raises_for_unknown_url(tmp_db):
    from applypilot.database import mark_job_as_test_case

    conn = tmp_db()
    with pytest.raises(ValueError, match="No job found"):
        mark_job_as_test_case("https://example.com/does-not-exist", conn)
