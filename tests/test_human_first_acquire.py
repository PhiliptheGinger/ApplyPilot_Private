"""Tests for acquire_human_first_linkedin_job (CLAUDE.md Future Work item 40).

LinkedIn jobs have no application_url (jobspy no longer exposes a direct-
apply link for them at all — see Future Work item 38), so they dead-end in
`manual_only` via the normal acquire_job sweep (decision #172) and are
otherwise never re-selected. This is the acquisition side of the human-first
apply flow that reclaims them: a human clicks Apply on the real LinkedIn
page instead.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def seed_linkedin_job(seed_job):
    """Override the conftest fixture: a realistic manual_only LinkedIn row
    with no application_url, matching what decision #172's sweep actually
    produces."""

    def _seed(conn, **overrides):
        overrides.setdefault("state", "manual_only")
        overrides.setdefault("application_url", None)
        overrides.setdefault(
            "url",
            f"https://www.linkedin.com/jobs/view/{overrides.pop('li_id', '4380167634')}",
        )
        return seed_job(conn, **overrides)

    return _seed


def test_acquires_manual_only_linkedin_job(tmp_db, seed_linkedin_job):
    from applypilot.apply.launcher import acquire_human_first_linkedin_job
    from applypilot.database import current_state

    conn = tmp_db()
    seed_linkedin_job(conn, fit_score=9)

    job = acquire_human_first_linkedin_job(worker_id=0)

    assert job is not None
    assert "linkedin.com/jobs/view" in job["url"]
    assert current_state(conn, job["url"]) == "applying"


def test_skips_non_linkedin_manual_only_job(tmp_db, seed_job):
    from applypilot.apply.launcher import acquire_human_first_linkedin_job

    conn = tmp_db()
    seed_job(conn, state="manual_only", application_url=None, url="https://example.com/job/manual-ats-1")

    job = acquire_human_first_linkedin_job(worker_id=0)

    assert job is None


def test_skips_linkedin_job_that_already_has_application_url(tmp_db, seed_linkedin_job):
    """A LinkedIn row that already has a real application_url isn't part of
    the dead-end backlog this flow exists for — the normal apply queue
    already handles it."""
    from applypilot.apply.launcher import acquire_human_first_linkedin_job

    conn = tmp_db()
    seed_linkedin_job(conn, application_url="https://boards.greenhouse.io/acme/jobs/1")

    job = acquire_human_first_linkedin_job(worker_id=0)

    assert job is None


def test_skips_job_without_tailored_resume(tmp_db, seed_linkedin_job):
    from applypilot.apply.launcher import acquire_human_first_linkedin_job

    conn = tmp_db()
    seed_linkedin_job(conn, tailored_resume_path=None)

    job = acquire_human_first_linkedin_job(worker_id=0)

    assert job is None


def test_respects_bounded_retry_cap(tmp_db, seed_linkedin_job):
    """Mirrors decision #168's MAX_TRANSIENT_APPLY_RETRIES pattern: a job
    that's already timed out on an unattended human a few times shouldn't
    keep resurfacing forever."""
    from applypilot.apply.launcher import acquire_human_first_linkedin_job

    conn = tmp_db()
    seed_linkedin_job(conn, apply_attempts=3)

    job = acquire_human_first_linkedin_job(worker_id=0)

    assert job is None


def test_also_acquires_orphaned_ready_to_apply_linkedin_job(tmp_db, seed_linkedin_job):
    """A job released back to ready_to_apply after a human-first timeout
    (hitl.run_human_first's "released" outcome, via the normal release_lock
    path) must still be re-acquireable by THIS function, not just the
    normal acquire_job — otherwise a session running --human-first on its
    own would silently orphan it forever (never manual_only, never has a
    real application_url for the normal queue either)."""
    from applypilot.apply.launcher import acquire_human_first_linkedin_job

    conn = tmp_db()
    seed_linkedin_job(conn, state="ready_to_apply")

    job = acquire_human_first_linkedin_job(worker_id=0)

    assert job is not None


def test_picks_highest_score_first(tmp_db, seed_linkedin_job):
    from applypilot.apply.launcher import acquire_human_first_linkedin_job

    conn = tmp_db()
    seed_linkedin_job(conn, li_id="1111", fit_score=6)
    seed_linkedin_job(conn, li_id="2222", fit_score=9)

    job = acquire_human_first_linkedin_job(worker_id=0)

    assert job["fit_score"] == 9


def test_manual_only_to_applying_is_a_legal_transition(tmp_db, seed_linkedin_job):
    """database.py's VALID_TRANSITIONS must allow this override edge —
    decision #172's sweep otherwise makes manual_only a dead end."""
    from applypilot.database import VALID_TRANSITIONS, transition_state

    assert "applying" in VALID_TRANSITIONS["manual_only"]

    conn = tmp_db()
    row = seed_linkedin_job(conn)

    ok = transition_state(conn, row["url"], "applying", reason="test")

    assert ok is True
