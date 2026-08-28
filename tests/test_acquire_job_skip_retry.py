"""Regression tests for the 2026-08-28 acquire_job recursion→loop conversion.

acquire_job() used to recurse into a fresh `return acquire_job(...)` call at
three skip-and-retry sites (retroactive ineligibility recheck, defensive
missing-application_url, manual-ATS domain match). Each site commits its
rejected candidate's state transition BEFORE recursing, so the row can never
be reselected -- recursion depth was bounded by how many `ready_to_apply`
candidates simultaneously satisfied a rejection condition (the bulk path's
own `LIMIT 100`), not literally unbounded, but with no explicit ceiling.

Fix: the three recursive calls became `return _SKIP_CANDIDATE` inside
`_acquire_job_one_attempt`, and `acquire_job` itself is now a small bounded
loop (`MAX_ACQUIRE_ATTEMPTS = 150`, a pathological-loop safety ceiling, not
an expected termination mechanism) that retries on that sentinel.

The ineligibility-recheck site (Site 1) already had coverage in
`test_acquire_job_eligibility_recheck.py`, which continues to pass
unmodified against the new loop -- proving the conversion is behavior-
preserving for that path. This file closes the coverage gap the 2026-08-28
investigation found for the other two sites (Site 2 had SQL-filter-level
coverage only, never the in-code defensive branch; Site 3 had zero
coverage), plus the loop's own ceiling/termination behavior.
"""

from __future__ import annotations

from applypilot.database import current_state


def _setup_apply_env(monkeypatch) -> None:
    """Quiet the company-cap loader (mirrors the other acquire_job test files)."""
    from applypilot import config

    monkeypatch.setattr(config, "get_company_limit", lambda key: (-1, 30), raising=False)


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


# ---------------------------------------------------------------------------
# Site 2 — defensive missing-application_url check (bulk path)
# ---------------------------------------------------------------------------


def test_bulk_acquire_skips_blank_application_url_and_picks_next_candidate(tmp_db, seed_job, monkeypatch):
    """A whitespace-only application_url passes the SQL filter
    (`!= ''` is true for `' '`) but fails the in-code `.strip()` defensive
    check -- this is the only way to reach Site 2's branch via the bulk
    (non-target_url) path, since the SQL WHERE clause already excludes
    genuinely NULL/empty values before Python ever sees the row."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    bad = _ready_job(
        seed_job,
        conn,
        url_suffix="skip-blank-url",
        application_url=" ",
        fit_score=10,
        company="blankco",
    )
    good = _ready_job(
        seed_job,
        conn,
        url_suffix="skip-blank-good",
        application_url="https://boards.greenhouse.io/goodco/jobs/1",
        fit_score=9,
        company="goodco",
    )

    job = acquire_job(min_score=9, max_age_days=0)

    assert job is not None and job["url"] == good["url"]
    assert current_state(conn, bad["url"]) == "manual_only"
    assert current_state(conn, good["url"]) == "applying"


# ---------------------------------------------------------------------------
# Site 3 — manual-ATS domain match (target_url path only: the bulk path's own
# SQL already excludes manual_ats domains via the same config source
# is_manual_ats() reads, so this in-code branch is unreachable in the bulk
# path -- confirmed by reading both call sites' predicates).
# ---------------------------------------------------------------------------


def test_targeted_acquire_skips_manual_ats_url(tmp_db, seed_job, monkeypatch):
    _setup_apply_env(monkeypatch)
    from applypilot import config
    from applypilot.apply.launcher import acquire_job

    monkeypatch.setattr(
        config,
        "load_sites_config",
        lambda: {"manual_ats": ["manualcorp.example.com"]},
    )

    conn = tmp_db()
    row = _ready_job(
        seed_job,
        conn,
        url_suffix="manual-ats-target",
        application_url="https://manualcorp.example.com/jobs/apply/1",
        fit_score=9,
    )

    job = acquire_job(target_url=row["url"], min_score=9, max_age_days=0)

    assert job is None
    assert current_state(conn, row["url"]) == "manual_only"


# ---------------------------------------------------------------------------
# Loop termination and ceiling behavior
# ---------------------------------------------------------------------------


def test_skip_then_next_candidate_terminates_in_a_handful_of_attempts(tmp_db, seed_job, monkeypatch):
    """Legitimate skip-then-next-candidate behavior must resolve in a
    handful of iterations, nowhere near MAX_ACQUIRE_ATTEMPTS -- the ceiling
    is a pathological-loop safety net, not an expected termination path."""
    _setup_apply_env(monkeypatch)
    import applypilot.apply.launcher as launcher_mod

    conn = tmp_db()
    bad = _ready_job(seed_job, conn, url_suffix="quick-bad", application_url=" ", fit_score=10, company="badco")
    good = _ready_job(
        seed_job,
        conn,
        url_suffix="quick-good",
        application_url="https://boards.greenhouse.io/goodco/jobs/2",
        fit_score=9,
        company="goodco",
    )

    call_count = {"n": 0}
    real_attempt = launcher_mod._acquire_job_one_attempt

    def _counting_attempt(*args, **kwargs):
        call_count["n"] += 1
        return real_attempt(*args, **kwargs)

    monkeypatch.setattr(launcher_mod, "_acquire_job_one_attempt", _counting_attempt)

    job = launcher_mod.acquire_job(min_score=9, max_age_days=0)

    assert job is not None and job["url"] == good["url"]
    assert current_state(conn, bad["url"]) == "manual_only"
    assert call_count["n"] == 2, (
        f"Expected exactly 2 attempts (one skip, one success), got {call_count['n']} -- "
        "a legitimate skip-then-next-candidate run must not approach the ceiling"
    )


def test_ceiling_exceeded_returns_none_and_logs_warning(tmp_db, seed_job, monkeypatch, caplog):
    """If every candidate in a pathologically large skip-run is rejected,
    acquire_job must give up (None + warning) once MAX_ACQUIRE_ATTEMPTS is
    exhausted, never loop forever (recursion's old failure mode would have
    been RecursionError, not an infinite loop -- either way, this proves
    the loop now fails safely and predictably instead)."""
    _setup_apply_env(monkeypatch)
    import applypilot.apply.launcher as launcher_mod

    monkeypatch.setattr(launcher_mod, "MAX_ACQUIRE_ATTEMPTS", 2)

    conn = tmp_db()
    for i in range(3):
        _ready_job(
            seed_job,
            conn,
            url_suffix=f"ceiling-bad-{i}",
            application_url=" ",
            fit_score=10,
            company=f"badco{i}",
        )

    with caplog.at_level("WARNING"):
        job = launcher_mod.acquire_job(min_score=9, max_age_days=0)

    assert job is None
    assert any("exceeded" in r.message and "2" in r.message for r in caplog.records), (
        f"Expected a warning mentioning the exceeded ceiling; got {[r.message for r in caplog.records]}"
    )
