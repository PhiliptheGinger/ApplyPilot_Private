"""Phase 3 state-machine integrity regression tests.

Covers the resurrection paths found in the 2026-08-27 investigation: a job
deliberately archived for an eligibility/title/seniority/ethical reason
must not resume autonomous pipeline work merely because legacy
`apply_status` data, a bulk recovery sweep, or a stale in-flight worker
bypasses the canonical `state` column.

Each test isolates one call site so a regression flips a single
assertion. Two tests reproduce the actual historical incidents found live
in the database (anonymized/minimal fixtures, same transition sequence):

- PayPal "Software-Engineer_R0137191": archived -> ready_to_apply via
  stale-lock recovery in acquire_job().
- PayPal "Sr-Software-Engineer_R0137197": a stale in-flight tailoring
  completion wrote onto an already-archived job.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from applypilot.database import current_state, transition_state


def _setup_apply_env(monkeypatch) -> None:
    """Quiet the company-cap loader so it doesn't interfere with selection
    under test (mirrors test_acquire_job_dedup.py's helper)."""
    from applypilot import config

    monkeypatch.setattr(config, "get_company_limit", lambda key: (-1, 30), raising=False)


def _stale_timestamp(minutes_ago: int = 40) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()


# ---------------------------------------------------------------------------
# 1. reset_failed()
# ---------------------------------------------------------------------------


def test_reset_failed_does_not_resurrect_archived_job(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="rf-archived", state="archived", apply_status="failed")
    url = row["url"]

    from applypilot.apply.launcher import reset_failed

    reset_failed()

    assert current_state(conn, url) == "archived"


def test_reset_failed_still_resurrects_apply_failed_job(tmp_db, seed_job):
    """Positive control: the legitimate retry candidate is unaffected."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="rf-failed", state="apply_failed", apply_status="failed")
    url = row["url"]

    from applypilot.apply.launcher import reset_failed

    count = reset_failed()

    assert count == 1
    assert current_state(conn, url) == "ready_to_apply"


def test_reset_failed_does_not_resurrect_manual_only_job(tmp_db, seed_job):
    """manual_only is a deliberate categorization, not a failure -- reset_failed
    must not touch it even though its legacy apply_status ('manual') used to
    satisfy the old broad predicate."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="rf-manual", state="manual_only", apply_status="manual")
    url = row["url"]

    from applypilot.apply.launcher import reset_failed

    reset_failed()

    assert current_state(conn, url) == "manual_only"


# ---------------------------------------------------------------------------
# 2. Stale-lock recovery in acquire_job()
# ---------------------------------------------------------------------------


def test_stale_lock_recovery_does_not_resurrect_archived_job(tmp_db, seed_job, monkeypatch):
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="sl-archived",
        state="archived",
        apply_status="in_progress",
        last_attempted_at=_stale_timestamp(),
        fit_score=10,
    )
    url = row["url"]

    job = acquire_job(min_score=10, max_age_days=0)

    assert job is None
    assert current_state(conn, url) == "archived"


def test_stale_lock_recovery_still_recovers_applying_job(tmp_db, seed_job, monkeypatch):
    """Positive control: a genuinely stuck 'applying' job is still released.

    The same acquire_job() call that releases the stale lock also runs its
    normal candidate-selection pass, so a released job with a qualifying
    fit_score is immediately eligible to be re-acquired -- that's expected,
    not a bug. What matters here is that the release itself succeeded
    (the job is reachable/selectable again at all) rather than staying
    permanently stuck at 'applying' with a dead lock.
    """
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="sl-applying",
        state="applying",
        apply_status="in_progress",
        last_attempted_at=_stale_timestamp(),
        fit_score=10,
    )
    url = row["url"]

    job = acquire_job(min_score=10, max_age_days=0)

    assert job is not None and job["url"] == url, "Stale-locked job should have been released and reacquired"
    assert current_state(conn, url) == "applying"


def test_reset_needs_human_then_crash_is_recovered_by_stale_lock_sweep(tmp_db, seed_job, monkeypatch):
    """End-to-end proof for the 2026-08-28 HITL-resume fix: before it,
    apply.hitl.reset_needs_human() cleared apply_status to NULL while
    moving state to 'applying', so a job abandoned (worker/process dies)
    between the HITL resume and the resumed application finishing could
    never match this sweep's `WHERE apply_status = 'in_progress'` clause --
    permanently stuck, exactly the fingerprint of the real live stuck row
    (simplyhired.com "Software Engineer - AI Trainer", stuck since
    2026-08-21). Now reset_needs_human sets the same apply_status=
    'in_progress' + last_attempted_at markers acquire_job's own normal
    acquisition sets, via the shared _mark_job_actively_applying helper --
    so a row it leaves behind is recoverable exactly like any other stale
    'applying' lock, verified here through the REAL reset_needs_human call,
    not a hand-rolled fixture shaped like its output.
    """
    _setup_apply_env(monkeypatch)
    from applypilot.apply.hitl import reset_needs_human
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="nh-crash-recovery",
        state="needs_human",
        apply_status="needs_human",
        fit_score=10,
    )
    url = row["url"]

    # 1. HITL resume: the real code path that used to leave this row
    # permanently unrecoverable.
    count = reset_needs_human(url=url, worker_id=4)
    assert count == 1
    assert current_state(conn, url) == "applying"
    row_after_resume = conn.execute("SELECT apply_status, agent_id FROM jobs WHERE url = ?", (url,)).fetchone()
    assert row_after_resume["apply_status"] == "in_progress"
    assert row_after_resume["agent_id"] == "worker-4"

    # 2. Simulate the worker/process dying before the resumed application
    # finishes: time passes with no further progress, so last_attempted_at
    # (freshly set by reset_needs_human above) ages past the sweep's
    # 30-minute threshold. Nothing else about the row changes -- this is
    # exactly what "crashed mid-resume" looks like in the DB.
    conn.execute(
        "UPDATE jobs SET last_attempted_at = ? WHERE url = ?",
        (_stale_timestamp(), url),
    )
    conn.commit()

    # 3. The sweep must now recognize and recover it.
    job = acquire_job(min_score=10, max_age_days=0)

    assert job is not None and job["url"] == url, (
        "A job abandoned after an HITL resume must be recoverable by the stale-lock "
        "sweep, not permanently stuck -- this would have failed (job is None) before "
        "the fix, since apply_status stayed NULL instead of 'in_progress'"
    )
    assert current_state(conn, url) == "applying"


def test_paypal_r0137191_stale_lock_regression(tmp_db, seed_job, monkeypatch):
    """Reproduces the real historical sequence (anonymized): a job cycles
    through applying/apply_failed a few times, is archived by a concurrent
    scoring-fix sweep while its legacy apply_status is still 'in_progress'
    from the interrupted attempt, and must NOT be resurrected by the next
    acquire_job() call."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="paypal-r0137191",
        title="Software Engineer",
        state="applying",
        apply_status="in_progress",
        last_attempted_at=_stale_timestamp(minutes_ago=1),
        fit_score=8,
    )
    url = row["url"]

    # Simulate the interrupted attempt: apply_status stays 'in_progress'
    # (a crashed/killed worker never got to clear it) while a concurrent
    # process force-archives the job for an unrelated policy reason.
    transition_state(conn, url, "archived", reason="requeue_after_scoring_fix_test", force=True)
    conn.execute(
        "UPDATE jobs SET last_attempted_at = ? WHERE url = ?",
        (_stale_timestamp(minutes_ago=40), url),
    )
    conn.commit()

    job = acquire_job(min_score=8, max_age_days=0)

    assert job is None
    assert current_state(conn, url) == "archived"


# ---------------------------------------------------------------------------
# 3. HITL startup re-queue
# ---------------------------------------------------------------------------


def test_hitl_startup_requeue_does_not_resurrect_archived_job(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="nh-archived", state="archived", apply_status="needs_human")
    url = row["url"]

    from applypilot.apply.orchestrator import requeue_needs_human_from_previous_session

    count = requeue_needs_human_from_previous_session(conn)

    assert count == 0
    assert current_state(conn, url) == "archived"


def test_hitl_startup_requeue_still_recovers_needs_human_job(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="nh-parked", state="needs_human", apply_status="needs_human")
    url = row["url"]

    from applypilot.apply.orchestrator import requeue_needs_human_from_previous_session

    count = requeue_needs_human_from_previous_session(conn)

    assert count == 1
    assert current_state(conn, url) == "ready_to_apply"


# ---------------------------------------------------------------------------
# 4. HTTP reset action
# ---------------------------------------------------------------------------


def test_http_reset_refuses_archived_job(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="http-archived", state="archived", apply_status="failed")
    url = row["url"]

    from applypilot.apply.launcher import _http_reset_job

    refused_state = _http_reset_job(conn, url)

    assert refused_state == "archived"
    assert current_state(conn, url) == "archived"


def test_http_reset_refuses_manual_only_job(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="http-manual", state="manual_only", apply_status="manual")
    url = row["url"]

    from applypilot.apply.launcher import _http_reset_job

    refused_state = _http_reset_job(conn, url)

    assert refused_state == "manual_only"
    assert current_state(conn, url) == "manual_only"


def test_http_reset_allows_apply_failed_job(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="http-failed", state="apply_failed", apply_status="failed")
    url = row["url"]

    from applypilot.apply.launcher import _http_reset_job

    refused_state = _http_reset_job(conn, url)

    assert refused_state is None
    assert current_state(conn, url) == "ready_to_apply"


# ---------------------------------------------------------------------------
# 5/6. Tailoring and cover-letter completion guards + in-flight claim states
# ---------------------------------------------------------------------------


def test_tailor_claim_succeeds_from_scored(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="tc-scored", state="scored")
    url = row["url"]

    claimed = transition_state(conn, url, "tailoring", reason="claimed for tailoring")

    assert claimed is True
    assert current_state(conn, url) == "tailoring"


def test_tailor_claim_fails_when_already_archived(tmp_db, seed_job):
    """The claim step itself is the first line of defense: a job archived
    between pending_tailor's SELECT and the claim UPDATE is dropped here."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="tc-archived", state="archived")
    url = row["url"]

    claimed = transition_state(conn, url, "tailoring", reason="claimed for tailoring")

    assert claimed is False
    assert current_state(conn, url) == "archived"


def test_tailor_claim_succeeds_from_tailor_failed(tmp_db, seed_job):
    """pending_tailor's other selectable source state (bounded-retry
    design, decision #65) can also be legally claimed."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="tc-retry", state="tailor_failed")
    url = row["url"]

    claimed = transition_state(conn, url, "tailoring", reason="claimed for tailoring")

    assert claimed is True
    assert current_state(conn, url) == "tailoring"


def test_tailor_completion_happy_path(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="tc-happy", state="tailoring", tailored_resume_path=None)
    url = row["url"]

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "approved", "/tmp/resume_x.docx")

    assert current_state(conn, url) == "tailored"
    stored = conn.execute("SELECT tailored_resume_path FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert stored == "/tmp/resume_x.docx"


def test_tailor_completion_failure_path(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="tc-fail", state="tailoring", tailored_resume_path=None)
    url = row["url"]

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "failed_validation", None)

    assert current_state(conn, url) == "tailor_failed"


def test_tailor_completion_skips_stale_write_when_archived_mid_flight(tmp_db, seed_job):
    """A worker claims a job (state='tailoring'), but a concurrent process
    archives it before the LLM call returns. The completion write must not
    overwrite tailored_resume_path or the state."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="tc-stale", state="tailoring", tailored_resume_path=None)
    url = row["url"]

    # Concurrent archival while the worker is "mid-flight".
    transition_state(conn, url, "archived", reason="title_pattern_reject:test", force=True)

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "approved", "/tmp/should_not_land.docx")

    assert current_state(conn, url) == "archived"
    stored = conn.execute("SELECT tailored_resume_path FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert stored is None, "Stale tailor completion must not write a resume path onto an archived job"


def test_paypal_r0137197_stale_tailoring_completion_regression(tmp_db, seed_job):
    """Reproduces the real historical sequence (anonymized): a job is
    selected and claimed for tailoring while eligible, gets archived by a
    concurrent title-pattern sweep while the LLM call is in flight, and the
    stale worker's completion must not produce a real tailored artifact."""
    conn = tmp_db()
    row = seed_job(
        conn, url_suffix="paypal-r0137197", title="Sr Software Engineer", state="scored", tailored_resume_path=None
    )
    url = row["url"]

    # run_tailoring's claim step (this is the part of the race the Phase 3
    # fix closes off first).
    assert transition_state(conn, url, "tailoring", reason="claimed for tailoring") is True

    # Concurrent title-pattern sweep archives it while the LLM call runs.
    transition_state(conn, url, "archived", reason="title_pattern_reject:auto_apply", force=True)

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "approved", "/tmp/sr_swe_should_not_land.docx")

    assert current_state(conn, url) == "archived"
    stored = conn.execute("SELECT tailored_resume_path FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert stored is None


def test_cover_claim_succeeds_from_tailored(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="cc-tailored", state="tailored")
    url = row["url"]

    claimed = transition_state(conn, url, "cover_writing", reason="claimed for cover writing")

    assert claimed is True
    assert current_state(conn, url) == "cover_writing"


def test_cover_claim_fails_when_already_archived(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="cc-archived", state="archived")
    url = row["url"]

    claimed = transition_state(conn, url, "cover_writing", reason="claimed for cover writing")

    assert claimed is False
    assert current_state(conn, url) == "archived"


def test_cover_claim_succeeds_from_cover_failed(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="cc-retry", state="cover_failed")
    url = row["url"]

    claimed = transition_state(conn, url, "cover_writing", reason="claimed for cover writing")

    assert claimed is True
    assert current_state(conn, url) == "cover_writing"


def test_cover_completion_happy_path(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="cc-happy", state="cover_writing", cover_letter_path=None)
    url = row["url"]

    from applypilot.scoring.cover_letter import _mark_cover_result

    _mark_cover_result(conn, url, "/tmp/cover_x.docx")

    assert current_state(conn, url) == "ready_to_apply"
    stored = conn.execute("SELECT cover_letter_path FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert stored == "/tmp/cover_x.docx"


def test_cover_completion_failure_path(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="cc-fail", state="cover_writing", cover_letter_path=None)
    url = row["url"]

    from applypilot.scoring.cover_letter import _mark_cover_result

    _mark_cover_result(conn, url, None, error="LLM exhausted")

    assert current_state(conn, url) == "cover_failed"


def test_cover_completion_skips_stale_write_when_archived_mid_flight(tmp_db, seed_job):
    """The most dangerous stale-completion case in the investigation: the
    success path lands directly on ready_to_apply with no intermediate
    failure state."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="cc-stale", state="cover_writing", cover_letter_path=None)
    url = row["url"]

    transition_state(conn, url, "archived", reason="ethical_exclusion:test", force=True)

    from applypilot.scoring.cover_letter import _mark_cover_result

    _mark_cover_result(conn, url, "/tmp/should_not_land_CL.docx")

    assert current_state(conn, url) == "archived"
    stored = conn.execute("SELECT cover_letter_path FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert stored is None, "Stale cover completion must not write ready_to_apply onto an archived job"


# ---------------------------------------------------------------------------
# 7. Application-result completion guard
# ---------------------------------------------------------------------------


def test_apply_mark_result_happy_path(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="ar-happy", state="applying")
    url = row["url"]

    from applypilot.apply.launcher import mark_result

    mark_result(url, "applied")

    assert current_state(conn, url) == "applied"
    applied_at = conn.execute("SELECT applied_at FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert applied_at is not None


def test_apply_mark_result_skips_stale_write_when_state_changed_mid_flight(tmp_db, seed_job):
    """A worker acquires a job (state='applying'), but a concurrent process
    changes its state (e.g. archives it) before the apply subprocess
    reports its result. The completion write must not mark it applied."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="ar-stale", state="applying")
    url = row["url"]

    transition_state(conn, url, "archived", reason="concurrent policy change", force=True)

    from applypilot.apply.launcher import mark_result

    mark_result(url, "applied")

    assert current_state(conn, url) == "archived"
    applied_at = conn.execute("SELECT applied_at FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert applied_at is None, "Stale apply completion must not mark an archived job as applied"


# ---------------------------------------------------------------------------
# 8. Enrichment completion guard
# ---------------------------------------------------------------------------


def test_enrichment_completion_happy_path(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="ec-happy", state="discovered", full_description=None)
    url = row["url"]

    from applypilot.enrichment.detail import _mark_enrich_result

    _mark_enrich_result(
        conn,
        url,
        status="ok",
        full_description="A real fetched description.",
        application_url=None,
        error=None,
        tier=1,
        retry_count=0,
    )

    assert current_state(conn, url) == "enriched"
    stored = conn.execute("SELECT full_description FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert stored == "A real fetched description."


def test_enrichment_completion_skips_stale_write_when_archived_mid_flight(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="ec-stale", state="discovered", full_description=None)
    url = row["url"]

    transition_state(conn, url, "archived", reason="duplicate detected mid-scrape", force=True)

    from applypilot.enrichment.detail import _mark_enrich_result

    _mark_enrich_result(
        conn,
        url,
        status="ok",
        full_description="Should not land.",
        application_url=None,
        error=None,
        tier=1,
        retry_count=0,
    )

    assert current_state(conn, url) == "archived"
    stored = conn.execute("SELECT full_description FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert stored is None, "Stale enrichment completion must not write a description onto an archived job"


# ---------------------------------------------------------------------------
# 9. Targeted apply --url
# ---------------------------------------------------------------------------


def test_targeted_apply_url_requires_ready_to_apply(tmp_db, seed_job, monkeypatch):
    """Decision: apply --url means 'manually trigger a job that already
    passed the pipeline and is ready to apply' -- not a general
    eligibility-bypass. An archived job (with a stale tailored_resume_path
    from before it was archived) must not be selectable."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="tgt-archived",
        state="archived",
        fit_score=9,
        application_url="https://job-boards.greenhouse.io/acme/jobs/999",
    )
    url = row["url"]

    job = acquire_job(target_url=url, min_score=1, max_age_days=0)

    assert job is None
    assert current_state(conn, url) == "archived"


def test_targeted_apply_url_selects_ready_to_apply_job(tmp_db, seed_job, monkeypatch):
    """Positive control: a genuinely ready job is still selectable by URL."""
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="tgt-ready",
        state="ready_to_apply",
        fit_score=9,
        application_url="https://job-boards.greenhouse.io/acme/jobs/1000",
    )
    url = row["url"]

    job = acquire_job(target_url=url, min_score=1, max_age_days=0)

    assert job is not None
    assert job["url"] == url


def test_targeted_apply_url_does_not_select_manual_only_job(tmp_db, seed_job, monkeypatch):
    _setup_apply_env(monkeypatch)
    from applypilot.apply.launcher import acquire_job

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="tgt-manual",
        state="manual_only",
        fit_score=9,
        application_url="https://job-boards.greenhouse.io/acme/jobs/1001",
    )
    url = row["url"]

    job = acquire_job(target_url=url, min_score=1, max_age_days=0)

    assert job is None
    assert current_state(conn, url) == "manual_only"
