"""Tests for acquire_job's duplicate-application-URL guard and the
NULL-application_url filter.

Covers two regressions caught during a live apply run:

1. **Duplicate application URLs.** LinkedIn (and other aggregators) repost
   the same Greenhouse / Lever / Ashby / Workday posting under multiple
   listing URLs, all pointing at the same upstream `application_url`.
   acquire_job must only fire once per `application_url` even when the
   queue has many discovered rows for the same posting.

2. **NULL application_url leakage.** Aggregator-discovered jobs whose
   enrichment never extracted a real apply URL would slip through, and
   the agent would try to "apply" by navigating to the listing page,
   typically getting confused or producing junk applies. Such jobs must
   never be acquired — they belong in `manual_only` instead.

Each test isolates one decision in `acquire_job` so a regression
flips a single assertion rather than the entire test file.
"""

from __future__ import annotations


def _setup_apply_env(monkeypatch) -> None:
    """Quiet the company-cap loader and avoid touching the user's profile."""
    from applypilot import config

    # company_limits.yaml — return wide-open defaults so the cap doesn't
    # interfere with the dedup/NULL-url decisions under test.
    monkeypatch.setattr(
        config, "get_company_limit", lambda key: (-1, 30), raising=False,
    )


def test_acquire_skips_job_when_other_row_already_applied(
    tmp_db, seed_job, monkeypatch
):
    """Two queue rows → same application_url → applying once locks the
    other row out, even though they have different listing URLs."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()

    shared_apply_url = "https://job-boards.greenhouse.io/overstory/jobs/4600037101"

    # First listing — already applied.
    seed_job(
        conn, url_suffix="li-original",
        url="https://www.linkedin.com/jobs/view/4377839705",
        application_url=shared_apply_url,
        apply_status="applied",
        fit_score=10, company="overstory",
    )
    # Second listing — different URL, same upstream application_url.
    seed_job(
        conn, url_suffix="li-repost",
        url="https://www.linkedin.com/jobs/view/4380167634",
        application_url=shared_apply_url,
        apply_status=None,
        fit_score=10, company="overstory",
    )

    job = acquire_job(min_score=10, max_age_days=0)
    assert job is None, (
        "Expected no acquireable jobs — the only candidate's "
        "application_url is already in flight on a sibling row, "
        "got: %r" % (job,)
    )


def test_acquire_skips_when_other_row_in_progress(tmp_db, seed_job, monkeypatch):
    """`in_progress` and `needs_human` should also block sibling rows."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    shared = "https://job-boards.greenhouse.io/temporaltechnologies/jobs/4290103007"
    seed_job(
        conn, url_suffix="lock-1",
        url="https://www.linkedin.com/jobs/view/1",
        application_url=shared,
        apply_status="in_progress",
        fit_score=10, company="temporal",
    )
    seed_job(
        conn, url_suffix="lock-2",
        url="https://www.linkedin.com/jobs/view/2",
        application_url=shared,
        apply_status=None,
        fit_score=10, company="temporal",
    )

    assert acquire_job(min_score=10, max_age_days=0) is None


def test_acquire_does_not_block_self(tmp_db, seed_job, monkeypatch):
    """A row's own row should never block itself — sanity check on the
    `j2.url != j.url` clause inside the NOT EXISTS subquery."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    seed_job(
        conn, url_suffix="solo",
        url="https://www.linkedin.com/jobs/view/solo",
        application_url="https://boards.greenhouse.io/acme/jobs/123",
        apply_status=None,
        fit_score=10, company="acme",
    )
    job = acquire_job(min_score=10, max_age_days=0)
    assert job is not None, "Solo candidate must be acquireable"
    assert job["url"] == "https://www.linkedin.com/jobs/view/solo"


def test_acquire_skips_jobs_with_null_application_url(
    tmp_db, seed_job, monkeypatch
):
    """Jobs without an extracted application_url must never be acquired —
    the agent has nothing to navigate to."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    seed_job(
        conn, url_suffix="no-apply-url",
        url="https://www.linkedin.com/jobs/view/no-apply-url",
        application_url=None,
        apply_status=None,
        fit_score=10, company="acme",
    )
    seed_job(
        conn, url_suffix="empty-apply-url",
        url="https://www.linkedin.com/jobs/view/empty",
        application_url="",
        apply_status=None,
        fit_score=10, company="acme",
    )

    assert acquire_job(min_score=10, max_age_days=0) is None


def test_acquire_picks_valid_when_others_are_null(
    tmp_db, seed_job, monkeypatch
):
    """When multiple candidates exist and only one has a valid
    application_url, that's the one that fires."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    seed_job(
        conn, url_suffix="bad-1",
        url="https://www.linkedin.com/jobs/view/bad-1",
        application_url=None,
        apply_status=None,
        fit_score=10, company="acme",
    )
    seed_job(
        conn, url_suffix="bad-2",
        url="https://www.linkedin.com/jobs/view/bad-2",
        application_url="",
        apply_status=None,
        fit_score=10, company="acme",
    )
    seed_job(
        conn, url_suffix="good",
        url="https://www.linkedin.com/jobs/view/good",
        application_url="https://boards.greenhouse.io/acme/jobs/9",
        apply_status=None,
        fit_score=10, company="acme",
    )

    job = acquire_job(min_score=10, max_age_days=0)
    assert job is not None
    assert job["url"] == "https://www.linkedin.com/jobs/view/good"
