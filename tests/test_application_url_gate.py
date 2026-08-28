"""The pending_tailor / no-application_url gate.

Root cause (2026-08-27 investigation): neither `pending_tailor` nor
`pending_cover` require `application_url`, and `_mark_cover_result`'s
success path transitions straight to `ready_to_apply` without checking it
either -- so a job with no usable `application_url` (predominantly
LinkedIn/Easy Apply postings, where no static apply link exists to
extract) burns a tailor AND a cover-letter LLM call even though
`acquire_job` can never auto-submit it (bulk selection already requires a
nonempty `application_url`; the targeted --url path's defensive check
routes it to `manual_only` before any browser attempt).

Design: this is an automation-eligibility gap, not a "job is bad" gap --
the job may be perfectly legitimate, we just structurally cannot submit to
it. So the fix must NOT archive these jobs (which would also hide their
listing from a human). Instead, `database.redirect_jobs_missing_
application_url` diverts them straight to `manual_only` -- the same state
`acquire_job` already uses for this exact condition (mirrors its column-
update convention: apply_status='manual', apply_error='no application_url',
apply_category='manual_only') -- from `run_tailoring()`, before any
tailor/cover LLM spend. `manual_only` is a real, non-terminal state the
dashboard already renders as a visible "archive tab" card with the job's
original listing URL intact (view.py's `_ARCHIVED_CATEGORIES` handling),
so the job stays discoverable/browsable for a human to apply to manually.

This required two new legal transitions in VALID_TRANSITIONS
(`scored -> manual_only`, `tailor_failed -> manual_only`) since neither
previously permitted it -- chosen over `force=True` so the state machine
keeps validating this as a real, intentional transition rather than
accumulating another bypass (the 2026-08-27 audit already flagged
`force=True` as "the norm, not the exception" as an existing weakness).

These tests exercise the behavioral invariant end to end: candidate
selection (`pending_tailor`), the diversion itself, `run_tailoring()`
wiring (via the same `_tailor_one_job` stub pattern test_batch_identity.py
already uses -- no real LLM/docx work), retry-attempt semantics, the audit
trail, and dashboard visibility -- not just "the new function does what it
does."
"""

from __future__ import annotations

from applypilot.database import current_state


def _stub_tailor_one_job(job, resume_text, profile, doc_format="docx"):
    return {
        "url": job["url"],
        "title": job["title"],
        "site": job.get("site", ""),
        "status": "approved",
        "attempts": 1,
        "path": "/tmp/fake.txt",
        "pdf_path": None,
        "auto_approved_by_facts": False,
    }


# ---------------------------------------------------------------------------
# 1-4. Selection / exclusion invariant at the pending_tailor boundary
# ---------------------------------------------------------------------------


def test_scored_job_with_valid_url_remains_pending_tailor_eligible(tmp_db, seed_job):
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    job = seed_job(
        conn, url_suffix="valid-url", state="scored", fit_score=9, tailored_resume_path=None,
        application_url="https://boards.greenhouse.io/acme/jobs/1",
    )

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    assert [r["url"] for r in rows] == [job["url"]]


def test_scored_job_with_null_url_excluded_after_redirect(tmp_db, seed_job):
    """The behavioral invariant, not just the filter: after run_tailoring's
    diversion step runs once, the job never reappears in pending_tailor on
    any later call -- it has permanently left the ('scored','tailor_failed')
    pool pending_tailor selects from."""
    from applypilot.database import get_jobs_by_stage, redirect_jobs_missing_application_url

    conn = tmp_db()
    job = seed_job(
        conn, url_suffix="null-url", state="scored", fit_score=9, tailored_resume_path=None, application_url=None
    )

    candidates = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    assert [r["url"] for r in candidates] == [job["url"]]  # still selectable before diversion

    kept = redirect_jobs_missing_application_url(conn, candidates, reason="test")
    assert kept == []

    rows_after = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    assert rows_after == []


def test_scored_job_with_empty_string_url_excluded_after_redirect(tmp_db, seed_job):
    from applypilot.database import get_jobs_by_stage, redirect_jobs_missing_application_url

    conn = tmp_db()
    job = seed_job(
        conn, url_suffix="empty-url", state="scored", fit_score=9, tailored_resume_path=None, application_url=""
    )

    candidates = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    kept = redirect_jobs_missing_application_url(conn, candidates, reason="test")

    assert kept == []
    assert get_jobs_by_stage(conn, stage="pending_tailor", min_score=8) == []
    assert current_state(conn, job["url"]) == "manual_only"


def test_mixed_pool_only_diverts_the_url_less_job(tmp_db, seed_job):
    """The exclusion must not accidentally exclude normal jobs with valid
    URLs -- a realistic mixed pool keeps the good one and only the good
    one."""
    from applypilot.database import get_jobs_by_stage, redirect_jobs_missing_application_url

    conn = tmp_db()
    good = seed_job(
        conn, url_suffix="mixed-good", state="scored", fit_score=9, tailored_resume_path=None,
        application_url="https://boards.greenhouse.io/acme/jobs/2",
    )
    bad = seed_job(
        conn, url_suffix="mixed-bad", state="scored", fit_score=9, tailored_resume_path=None, application_url=None
    )

    candidates = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    kept = redirect_jobs_missing_application_url(conn, candidates, reason="test")

    assert [r["url"] for r in kept] == [good["url"]]
    assert current_state(conn, good["url"]) == "scored"
    assert current_state(conn, bad["url"]) == "manual_only"


# ---------------------------------------------------------------------------
# 5. Manual-view path is preserved (dashboard visibility, listing URL intact)
# ---------------------------------------------------------------------------


def test_diverted_job_stays_visible_via_manual_only_dashboard_classification(tmp_db, seed_job):
    """A diverted job is not archived/hidden: view.py's _classify_job must
    still surface it (as a browsable 'manual_only' archive-tab card), and
    its original listing URL (the `url` column, distinct from the missing
    `application_url`) must remain untouched so a human can navigate to it
    manually."""
    from applypilot.database import get_jobs_by_stage, redirect_jobs_missing_application_url
    from applypilot.view import _classify_job

    conn = tmp_db()
    job = seed_job(
        conn,
        url_suffix="manual-view",
        url="https://www.linkedin.com/jobs/view/manual-view",
        state="scored",
        fit_score=9,
        tailored_resume_path=None,
        application_url=None,
    )

    candidates = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    redirect_jobs_missing_application_url(conn, candidates, reason="test")

    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    assert row["url"] == "https://www.linkedin.com/jobs/view/manual-view"
    assert not (row["application_url"] or "")

    stage, tab = _classify_job(row)
    assert stage == "manual_only"
    assert tab == "archive"


# ---------------------------------------------------------------------------
# 6. State transition + retry-attempt semantics
# ---------------------------------------------------------------------------


def test_diversion_does_not_increment_tailor_attempts(tmp_db, seed_job):
    """This is a definitive automation-eligibility exclusion, not a failed
    attempt -- it must not consume the tailor_attempts retry budget the way
    a genuine tailoring failure does (_mark_tailor_result's failure path)."""
    from applypilot.database import get_jobs_by_stage, redirect_jobs_missing_application_url

    conn = tmp_db()
    job = seed_job(
        conn, url_suffix="attempts", state="scored", fit_score=9, tailored_resume_path=None,
        application_url=None, tailor_attempts=0,
    )

    candidates = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    redirect_jobs_missing_application_url(conn, candidates, reason="test")

    attempts = conn.execute("SELECT tailor_attempts FROM jobs WHERE url = ?", (job["url"],)).fetchone()[0]
    assert (attempts or 0) == 0


def test_diversion_writes_a_consistent_audit_row(tmp_db, seed_job):
    from applypilot.database import get_jobs_by_stage, redirect_jobs_missing_application_url

    conn = tmp_db()
    job = seed_job(
        conn, url_suffix="audit", state="tailor_failed", fit_score=9, tailored_resume_path=None, application_url=""
    )

    candidates = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    redirect_jobs_missing_application_url(conn, candidates, reason="no application_url — excluded before tailoring")

    transitions = conn.execute(
        "SELECT from_state, to_state, reason FROM job_state_transitions WHERE job_url = ? ORDER BY id DESC",
        (job["url"],),
    ).fetchall()
    assert transitions[0]["from_state"] == "tailor_failed"
    assert transitions[0]["to_state"] == "manual_only"
    assert transitions[0]["reason"] == "no application_url — excluded before tailoring"

    row = conn.execute(
        "SELECT apply_status, apply_error, apply_category FROM jobs WHERE url = ?", (job["url"],)
    ).fetchone()
    assert row["apply_status"] == "manual"
    assert row["apply_error"] == "no application_url"
    assert row["apply_category"] == "manual_only"


def test_diverted_job_is_not_resurrected_by_a_later_pending_tailor_run(tmp_db, seed_job):
    """Confirms manual_only is a genuine terminal-for-tailoring state, not
    just a one-off filter: a second, independent pending_tailor/redirect
    pass finds nothing left to divert."""
    from applypilot.database import get_jobs_by_stage, redirect_jobs_missing_application_url

    conn = tmp_db()
    job = seed_job(
        conn, url_suffix="no-resurrect", state="scored", fit_score=9, tailored_resume_path=None, application_url=None
    )

    first_pass = redirect_jobs_missing_application_url(
        conn, get_jobs_by_stage(conn, stage="pending_tailor", min_score=8), reason="test"
    )
    second_pass = redirect_jobs_missing_application_url(
        conn, get_jobs_by_stage(conn, stage="pending_tailor", min_score=8), reason="test"
    )

    assert first_pass == []
    assert second_pass == []
    assert current_state(conn, job["url"]) == "manual_only"


# ---------------------------------------------------------------------------
# run_tailoring() integration -- _tailor_one_job stubbed, no real LLM/docx
# work, mirrors test_batch_identity.py's established pattern.
# ---------------------------------------------------------------------------


def test_run_tailoring_diverts_url_less_job_and_still_tailors_the_valid_one(tmp_db, seed_job, monkeypatch, tmp_path):
    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    good = seed_job(
        conn, url_suffix="rt-good", state="scored", fit_score=9, tailored_resume_path=None,
        application_url="https://boards.greenhouse.io/acme/jobs/3", company="acme",
    )
    bad = seed_job(
        conn, url_suffix="rt-bad", state="scored", fit_score=9, tailored_resume_path=None,
        application_url=None, company="othercorp",
    )

    calls: list[str] = []

    def tracking_stub(job, resume_text, profile, doc_format="docx"):
        calls.append(job["url"])
        return _stub_tailor_one_job(job, resume_text, profile, doc_format)

    monkeypatch.setattr(tailor_mod, "_tailor_one_job", tracking_stub)
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    result = tailor_mod.run_tailoring(min_score=8, limit=10)

    assert calls == [good["url"]], "The url-less job must never reach the tailoring call at all"
    assert result["approved"] == 1
    assert current_state(conn, good["url"]) == "tailored"
    assert current_state(conn, bad["url"]) == "manual_only"


def test_run_tailoring_makes_no_llm_call_when_only_url_less_jobs_are_pending(tmp_db, seed_job, monkeypatch, tmp_path):
    """The zero-candidate case: if every pending_tailor candidate lacks an
    application_url, run_tailoring must divert all of them and return
    early without ever invoking the (stubbed, would-be-real) tailoring
    call."""
    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    job = seed_job(
        conn, url_suffix="rt-only-bad", state="scored", fit_score=9, tailored_resume_path=None, application_url=None
    )

    calls: list[str] = []
    monkeypatch.setattr(tailor_mod, "_tailor_one_job", lambda *a, **k: calls.append(a[0]["url"]))
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    result = tailor_mod.run_tailoring(min_score=8, limit=10)

    assert calls == []
    assert result["approved"] == 0
    assert current_state(conn, job["url"]) == "manual_only"
