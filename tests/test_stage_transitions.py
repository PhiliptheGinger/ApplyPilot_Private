"""TDD tests for state-machine transitions emitted by tailor, cover letter,
and enrichment stages.

All tests are written BEFORE the implementation so they fail first.
After implementing the changes, all should pass.
"""

from applypilot.database import current_state, state_history, transition_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_with_state(conn, seed_job, suffix, state, **extra):
    """Seed a job then force it into the given state."""
    row = seed_job(conn, url_suffix=suffix, **extra)
    url = row["url"]
    transition_state(conn, url, state, reason="test setup", force=True)
    conn.commit()
    return url


# ---------------------------------------------------------------------------
# A1 — tailor.py: _flush_tailor_results (via _mark_tailor_result helper)
# ---------------------------------------------------------------------------


def test_tailor_success_transitions_to_tailored(tmp_db, seed_job):
    """Approved tailor result must transition state to 'tailored'."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "TA1", "tailoring", fit_score=9, full_description="x" * 300, tailored_resume_path=None
    )

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "approved", "/tmp/resume.pdf", attempts=1)
    conn.commit()

    assert current_state(conn, url) == "tailored"


def test_tailor_success_writes_audit_row(tmp_db, seed_job):
    """Approved result must appear in the audit log with to_state='tailored'."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "TA2", "tailoring", fit_score=9, full_description="x" * 300, tailored_resume_path=None
    )

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "approved", "/tmp/resume.pdf", attempts=2)
    conn.commit()

    history = state_history(conn, url)
    assert any(h["to_state"] == "tailored" for h in history), (
        f"Expected 'tailored' in history, got: {[h['to_state'] for h in history]}"
    )


def test_tailor_failed_transitions_to_tailor_failed(tmp_db, seed_job):
    """Any non-approved tailor result must transition to 'tailor_failed'."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "TA3", "tailoring", fit_score=9, full_description="x" * 300, tailored_resume_path=None
    )

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "failed_validation", None, attempts=3)
    conn.commit()

    assert current_state(conn, url) == "tailor_failed"


def test_tailor_failed_judge_transitions_to_tailor_failed(tmp_db, seed_job):
    """failed_judge status must also transition to 'tailor_failed'."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "TA4", "tailoring", fit_score=9, full_description="x" * 300, tailored_resume_path=None
    )

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "failed_judge", None, attempts=2)
    conn.commit()

    assert current_state(conn, url) == "tailor_failed"


def test_tailor_attempts_increments_even_on_failure(tmp_db, seed_job):
    """tailor_attempts counter must increment even when the tailor fails."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "TA5", "tailoring", fit_score=8, full_description="x" * 300, tailored_resume_path=None
    )
    # Baseline: no attempt yet
    row = conn.execute("SELECT COALESCE(tailor_attempts, 0) FROM jobs WHERE url=?", (url,)).fetchone()
    before = row[0]

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "error", None, attempts=1)
    conn.commit()

    row = conn.execute("SELECT COALESCE(tailor_attempts, 0) FROM jobs WHERE url=?", (url,)).fetchone()
    after = row[0]
    assert after == before + 1, f"Expected attempts to increment from {before} to {before + 1}, got {after}"


def test_tailor_attempts_increments_on_success_too(tmp_db, seed_job):
    """tailor_attempts must also increment on approved (after path is written)."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "TA6", "tailoring", fit_score=9, full_description="x" * 300, tailored_resume_path=None
    )

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "approved", "/tmp/resume.pdf", attempts=1)
    conn.commit()

    row = conn.execute("SELECT COALESCE(tailor_attempts, 0) FROM jobs WHERE url=?", (url,)).fetchone()
    assert row[0] >= 1, f"Expected tailor_attempts >= 1, got {row[0]}"


def test_tailor_provider_unavailable_does_not_increment_attempts(tmp_db, seed_job):
    """Temporary provider outages are retryable and should not burn attempts."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "TA8", "tailoring", fit_score=9, full_description="x" * 300, tailored_resume_path=None
    )
    conn.execute("UPDATE jobs SET tailor_attempts=2 WHERE url=?", (url,))

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "provider_unavailable", None, attempts=1)
    conn.commit()

    row = conn.execute("SELECT COALESCE(tailor_attempts, 0) FROM jobs WHERE url=?", (url,)).fetchone()
    assert row[0] == 2
    assert current_state(conn, url) == "tailor_failed"


def test_tailor_retry_path_from_tailor_failed_via_claim(tmp_db, seed_job):
    """Retry path: pending_tailor re-selects a 'tailor_failed' row and
    run_tailoring's claim step legally re-enters 'tailoring' (tailor_failed
    -> tailoring is a valid VALID_TRANSITIONS source, see decision #65's
    bounded-retry design) before _mark_tailor_result completes it.

    2026-08-27 (Phase 3 state-machine hardening): _mark_tailor_result now
    requires state == 'tailoring' at completion time (closes the stale-
    worker resurrection race -- see PayPal "Sr-Software-Engineer_R0137197"
    regression), so this test seeds the post-claim state directly rather
    than the pre-claim 'tailor_failed' state it used to. force=True is no
    longer needed for this transition since tailoring -> tailored is
    already a legal VALID_TRANSITIONS destination once claimed.
    """
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "TA7", "tailoring", fit_score=9, full_description="x" * 300, tailored_resume_path=None
    )

    from applypilot.scoring.tailor import _mark_tailor_result

    _mark_tailor_result(conn, url, "approved", "/tmp/retry.pdf", attempts=2)
    conn.commit()

    assert current_state(conn, url) == "tailored"


# ---------------------------------------------------------------------------
# A2 — cover_letter.py: _flush_cover_results (via _mark_cover_result helper)
# ---------------------------------------------------------------------------


def test_cover_success_transitions_to_ready_to_apply(tmp_db, seed_job):
    """Successful cover letter must transition state to 'ready_to_apply'."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "CL1", "cover_writing", tailored_resume_path="/tmp/resume.pdf", cover_letter_path=None
    )

    from applypilot.scoring.cover_letter import _mark_cover_result

    _mark_cover_result(conn, url, "/tmp/cover.pdf", error=None)
    conn.commit()

    assert current_state(conn, url) == "ready_to_apply"


def test_cover_success_writes_audit_row(tmp_db, seed_job):
    """Successful cover result must appear in audit log with 'ready_to_apply'."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "CL2", "cover_writing", tailored_resume_path="/tmp/resume.pdf", cover_letter_path=None
    )

    from applypilot.scoring.cover_letter import _mark_cover_result

    _mark_cover_result(conn, url, "/tmp/cover.pdf", error=None)
    conn.commit()

    history = state_history(conn, url)
    assert any(h["to_state"] == "ready_to_apply" for h in history)


def test_cover_failure_transitions_to_cover_failed(tmp_db, seed_job):
    """Failed cover generation (no path) must transition to 'cover_failed'."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "CL3", "cover_writing", tailored_resume_path="/tmp/resume.pdf", cover_letter_path=None
    )

    from applypilot.scoring.cover_letter import _mark_cover_result

    _mark_cover_result(conn, url, None, error="LLM refused")
    conn.commit()

    assert current_state(conn, url) == "cover_failed"


def test_cover_failure_writes_audit_row(tmp_db, seed_job):
    """Failed cover result must appear in audit log with 'cover_failed'."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "CL4", "cover_writing", tailored_resume_path="/tmp/resume.pdf", cover_letter_path=None
    )

    from applypilot.scoring.cover_letter import _mark_cover_result

    _mark_cover_result(conn, url, None, error="timeout")
    conn.commit()

    history = state_history(conn, url)
    assert any(h["to_state"] == "cover_failed" for h in history)


def test_cover_provider_unavailable_does_not_increment_attempts(tmp_db, seed_job):
    """Provider-unavailable cover failures should be retryable without burning attempts."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "CL5", "cover_writing", tailored_resume_path="/tmp/resume.pdf", cover_letter_path=None
    )
    conn.execute("UPDATE jobs SET cover_attempts=2 WHERE url=?", (url,))

    from applypilot.scoring.cover_letter import _mark_cover_result

    _mark_cover_result(conn, url, None, error="cloud_cover_generation_exhausted: All models exhausted")
    conn.commit()

    row = conn.execute("SELECT COALESCE(cover_attempts, 0) FROM jobs WHERE url=?", (url,)).fetchone()
    assert row[0] == 2
    assert current_state(conn, url) == "cover_failed"

    history = state_history(conn, url)
    assert history[0]["reason"] == "provider_unavailable"


def test_cover_validation_failure_still_increments_attempts(tmp_db, seed_job):
    """Non-provider errors keep existing attempt-accounting behavior."""
    conn = tmp_db()
    url = _seed_with_state(
        conn, seed_job, "CL6", "cover_writing", tailored_resume_path="/tmp/resume.pdf", cover_letter_path=None
    )
    conn.execute("UPDATE jobs SET cover_attempts=2 WHERE url=?", (url,))

    from applypilot.scoring.cover_letter import _mark_cover_result

    _mark_cover_result(conn, url, None, error="validation failed: too short")
    conn.commit()

    row = conn.execute("SELECT COALESCE(cover_attempts, 0) FROM jobs WHERE url=?", (url,)).fetchone()
    assert row[0] == 3
    assert current_state(conn, url) == "cover_failed"


# ---------------------------------------------------------------------------
# A3 — enrichment/detail.py: inline UPDATE blocks
# ---------------------------------------------------------------------------


def test_enrich_success_transitions_to_enriched(tmp_db, seed_job):
    """Successful enrichment must transition state to 'enriched'."""
    conn = tmp_db()
    url = _seed_with_state(conn, seed_job, "EN1", "discovered", full_description=None)

    from applypilot.enrichment.detail import _mark_enrich_result

    _mark_enrich_result(
        conn,
        url,
        status="ok",
        full_description="Full job text here " * 30,
        application_url="https://boards.greenhouse.io/acme/jobs/1",
        error=None,
        tier=1,
        retry_count=0,
    )
    conn.commit()

    assert current_state(conn, url) == "enriched"


def test_enrich_success_writes_audit_row(tmp_db, seed_job):
    """Successful enrichment must appear in audit log with 'enriched'."""
    conn = tmp_db()
    url = _seed_with_state(conn, seed_job, "EN2", "discovered", full_description=None)

    from applypilot.enrichment.detail import _mark_enrich_result

    _mark_enrich_result(
        conn,
        url,
        status="ok",
        full_description="Job description " * 20,
        application_url=None,
        error=None,
        tier=2,
        retry_count=0,
    )
    conn.commit()

    history = state_history(conn, url)
    assert any(h["to_state"] == "enriched" for h in history)


def test_enrich_permanent_failure_transitions_to_enrich_failed(tmp_db, seed_job):
    """Permanent enrichment failure must transition state to 'enrich_failed'."""
    conn = tmp_db()
    url = _seed_with_state(conn, seed_job, "EN3", "discovered", full_description=None)

    from applypilot.enrichment.detail import _mark_enrich_result

    _mark_enrich_result(
        conn,
        url,
        status="error",
        full_description=None,
        application_url=None,
        error="HTTP 404: page not found",
        tier=None,
        retry_count=0,
    )
    conn.commit()

    assert current_state(conn, url) == "enrich_failed"


def test_enrich_expired_failure_transitions_to_enrich_failed(tmp_db, seed_job):
    """Expired job (410 Gone) must also transition to 'enrich_failed'."""
    conn = tmp_db()
    url = _seed_with_state(conn, seed_job, "EN4", "discovered", full_description=None)

    from applypilot.enrichment.detail import _mark_enrich_result

    _mark_enrich_result(
        conn,
        url,
        status="error",
        full_description=None,
        application_url=None,
        error="HTTP 410: Gone",
        tier=None,
        retry_count=0,
    )
    conn.commit()

    assert current_state(conn, url) == "enrich_failed"


def test_enrich_retriable_failure_stays_in_discovered(tmp_db, seed_job):
    """Retriable enrichment error must NOT change state (stays 'discovered')."""
    conn = tmp_db()
    url = _seed_with_state(conn, seed_job, "EN5", "discovered", full_description=None)

    from applypilot.enrichment.detail import _mark_enrich_result

    # "timeout" matches _RETRIABLE_PATTERNS → no transition expected
    _mark_enrich_result(
        conn,
        url,
        status="error",
        full_description=None,
        application_url=None,
        error="HTTP 503: Service unavailable timeout",
        tier=None,
        retry_count=0,
    )
    conn.commit()

    assert current_state(conn, url) == "discovered"


def test_enrich_partial_success_transitions_to_enriched(tmp_db, seed_job):
    """'partial' status (description only, no apply URL) still means enriched."""
    conn = tmp_db()
    url = _seed_with_state(conn, seed_job, "EN6", "discovered", full_description=None)

    from applypilot.enrichment.detail import _mark_enrich_result

    _mark_enrich_result(
        conn,
        url,
        status="partial",
        full_description="Some description " * 10,
        application_url=None,
        error=None,
        tier=3,
        retry_count=0,
    )
    conn.commit()

    assert current_state(conn, url) == "enriched"


def test_scrape_detail_page_handles_patchright_timeout():
    """A Patchright navigation timeout should be treated as a normal per-job error."""
    from patchright.sync_api import TimeoutError as PatchrightTimeoutError

    class _Page:
        def __init__(self):
            self.goto_kwargs = None

        def goto(self, *args, **kwargs):
            self.goto_kwargs = kwargs
            raise PatchrightTimeoutError("Page.goto: Timeout 45000ms exceeded.")

        def wait_for_load_state(self, *args, **kwargs):
            raise AssertionError("should not reach wait_for_load_state when goto times out")

    from applypilot.enrichment.detail import scrape_detail_page

    page = _Page()
    result = scrape_detail_page(page, "https://example.com/job")

    assert page.goto_kwargs["wait_until"] == "domcontentloaded"
    assert result["status"] == "error"
    assert result["error"] == "timeout"


# ---------------------------------------------------------------------------
# A4 — database.py: VALID_TRANSITIONS direct edges (Fix 3)
# ---------------------------------------------------------------------------


def test_scored_to_tailored_allowed_without_force(tmp_db, seed_job):
    """scored → tailored must be a legal direct edge (no force needed)."""
    conn = tmp_db()
    url = _seed_with_state(conn, seed_job, "ST1", "scored", fit_score=9, full_description="x" * 300)
    assert transition_state(conn, url, "tailored", reason="direct") is True
    assert current_state(conn, url) == "tailored"


def test_scored_to_tailor_failed_allowed_without_force(tmp_db, seed_job):
    """scored → tailor_failed must be a legal direct edge (no force needed)."""
    conn = tmp_db()
    url = _seed_with_state(conn, seed_job, "ST2", "scored", fit_score=9, full_description="x" * 300)
    assert transition_state(conn, url, "tailor_failed", reason="direct") is True
    assert current_state(conn, url) == "tailor_failed"


def test_tailor_failed_to_tailored_allowed_without_force(tmp_db, seed_job):
    """tailor_failed → tailored must be a legal direct edge (retry success)."""
    conn = tmp_db()
    url = _seed_with_state(conn, seed_job, "ST3", "tailor_failed", fit_score=9, full_description="x" * 300)
    assert transition_state(conn, url, "tailored", reason="retry success") is True
    assert current_state(conn, url) == "tailored"
