"""Retry-time deterministic eligibility recheck at the acquire_job() choke point.

2026-08-27 audit finding #9: acquire_job()'s existing defense-in-depth
(_auto_reject_title) only re-validates seniority. A job scored before a
later eligibility rule existed -- the concrete historical case is a federal
ML-Ops job requiring TS/SCI, scored before the 652cfb6 clearance gate
existed -- can be resurrected back into the application path via
reset_failed(), an HTTP reset, HITL requeue, or the stale-lock sweep,
without ever re-running scorer._check_ineligible(). These tests verify the
new recheck added directly to acquire_job() (the single choke point both
the --url and bulk selection paths converge on) closes that gap without
weakening the existing seniority guard or normal eligible-job acquisition.
"""

from __future__ import annotations

from applypilot.database import current_state


def _setup_apply_env(monkeypatch) -> None:
    """Quiet the company-cap loader so it doesn't interfere with selection
    under test (mirrors test_phase3_state_resurrection.py's helper)."""
    from applypilot import config

    monkeypatch.setattr(config, "get_company_limit", lambda key: (-1, 30), raising=False)


_TS_SCI_DESCRIPTION = (
    "This role supports a US federal ML Ops program. Candidates must currently "
    "hold an active TS/SCI clearance to be considered."
)


def _ready_job(seed_job, conn, **overrides):
    defaults = {
        "state": "ready_to_apply",
        "fit_score": 9,
        "tailored_resume_path": "/tmp/resume.pdf",
        "cover_letter_path": "/tmp/cover.pdf",
        "application_url": "https://boards.greenhouse.io/acme/jobs/1",
        "company": "acme",
    }
    defaults.update(overrides)
    return seed_job(conn, **defaults)


def test_retroactively_ineligible_job_rejected_at_bulk_acquire(tmp_db, seed_job, monkeypatch):
    """A ready_to_apply job whose current description requires TS/SCI must
    be archived instead of acquired, via normal bulk candidate selection."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = _ready_job(
        seed_job,
        conn,
        url_suffix="elig-tssci-bulk",
        title="Software Development Engineer - ML Ops (US Federal)",
        full_description=_TS_SCI_DESCRIPTION,
    )
    url = row["url"]

    job = acquire_job(min_score=9, max_age_days=0)

    assert job is None
    assert current_state(conn, url) == "archived"
    stored = conn.execute("SELECT apply_status FROM jobs WHERE url = ?", (url,)).fetchone()
    assert stored["apply_status"] != "in_progress", "Ineligible job must never reach an application attempt"


def test_retroactively_ineligible_job_rejected_at_targeted_url_acquire(tmp_db, seed_job, monkeypatch):
    """The same recheck must fire through the targeted --url acquisition path."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = _ready_job(
        seed_job,
        conn,
        url_suffix="elig-tssci-targeted",
        title="Software Development Engineer - ML Ops (US Federal)",
        full_description=_TS_SCI_DESCRIPTION,
    )
    url = row["url"]

    job = acquire_job(target_url=url)

    assert job is None
    assert current_state(conn, url) == "archived"


def test_ineligible_job_never_reaches_apply_status_in_progress(tmp_db, seed_job, monkeypatch):
    """Explicit confirmation the rejection happens strictly before the
    apply_status='in_progress' / state='applying' claim writes further down
    acquire_job() -- the recheck must short-circuit before any application
    attempt is marked as started."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = _ready_job(
        seed_job,
        conn,
        url_suffix="elig-no-attempt",
        title="Software Development Engineer - ML Ops (US Federal)",
        full_description=_TS_SCI_DESCRIPTION,
    )
    url = row["url"]

    acquire_job(min_score=9, max_age_days=0)

    row_after = conn.execute(
        "SELECT state, apply_status, last_attempted_at FROM jobs WHERE url = ?", (url,)
    ).fetchone()
    assert row_after["state"] == "archived"
    assert row_after["apply_status"] != "in_progress"


def test_bulk_acquire_skips_ineligible_and_picks_eligible_candidate(tmp_db, seed_job, monkeypatch):
    """Positive control mixed with the rejection: among two ready_to_apply
    candidates, the retroactively-ineligible one is archived and the
    genuinely eligible one is still acquired in the same call (acquire_job's
    existing recursive skip-and-retry shape, reused for this new check)."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    bad = _ready_job(
        seed_job,
        conn,
        url_suffix="elig-mixed-bad",
        title="Software Development Engineer - ML Ops (US Federal)",
        full_description=_TS_SCI_DESCRIPTION,
        fit_score=10,
        company="fed-corp",
        application_url="https://boards.greenhouse.io/fedcorp/jobs/1",
    )
    good = _ready_job(
        seed_job,
        conn,
        url_suffix="elig-mixed-good",
        title="Software Engineer",
        full_description="A normal US-remote backend engineering role.",
        fit_score=9,
        company="normalco",
        application_url="https://boards.greenhouse.io/normalco/jobs/2",
    )

    job = acquire_job(min_score=9, max_age_days=0)

    assert job is not None
    assert job["url"] == good["url"]
    assert current_state(conn, bad["url"]) == "archived"
    assert current_state(conn, good["url"]) == "applying"


def test_eligible_job_still_acquires_normally(tmp_db, seed_job, monkeypatch):
    """Positive control: a genuinely eligible ready_to_apply job is
    unaffected by the new recheck."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = _ready_job(
        seed_job,
        conn,
        url_suffix="elig-normal",
        title="Software Engineer",
        full_description="A normal US-remote backend engineering role.",
    )
    url = row["url"]

    job = acquire_job(min_score=9, max_age_days=0)

    assert job is not None and job["url"] == url
    assert current_state(conn, url) == "applying"


def test_existing_seniority_title_rejection_still_works(tmp_db, seed_job, monkeypatch):
    """Regression guard: the pre-existing _auto_reject_title seniority check
    (independent of this new recheck) must remain intact and still fire
    first for a senior-titled job."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = _ready_job(
        seed_job,
        conn,
        url_suffix="elig-senior-title",
        title="Senior Software Engineer",
        full_description="A normal US-remote backend engineering role.",
    )
    url = row["url"]

    job = acquire_job(min_score=9, max_age_days=0)

    assert job is None
    assert current_state(conn, url) == "archived"
