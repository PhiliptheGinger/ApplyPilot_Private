"""Regression tests for the 2026-08-25 `pending_tailor` state-selection fix.

Root cause (traced via VALID_TRANSITIONS, not assumed): `get_jobs_by_stage`'s
"pending_tailor" WHERE clause had NO `state` check at all -- only
fit_score/tailored_resume_path/tailor_attempts/eligibility. A job archived
for ANY reason (title-reject, revalidate-seniority, non_us_only, ...)
remained selectable forever as long as those other columns still matched,
since archiving doesn't change fit_score or tailored_resume_path. Combined
with `reject_jobs_by_title_patterns` not protecting `tailor_failed`, this
produced an observed live infinite oscillation: archived -> (re-selected by
pending_tailor) -> tailor_failed(seniority_mismatch) -> (re-archived by
--auto-reject-titles) -> archived -> ... five times in ~90 seconds for one
real job before its tailor_attempts budget ran out.

Fix: positive-select only the states where VALID_TRANSITIONS
(database.py) actually permits progression to "tailoring":
  - "scored" -> "tailoring" is legal (the primary ready state)
  - "tailor_failed" -> "tailoring" is legal (the existing bounded-retry
    design, tailor_attempts < 5 -- must remain selectable)
  - "archived" -> frozenset() (zero legal outgoing transitions anywhere in
    the state machine) -- must never be reachable from this query again
"""

from __future__ import annotations


def test_archived_job_never_returned_by_pending_tailor(tmp_db, seed_job):
    """The exact bug: an archived job with a qualifying fit_score and no
    tailored resume must never be selected again -- archived is the state
    machine's one true terminal state (VALID_TRANSITIONS["archived"] ==
    frozenset())."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    seed_job(conn, fit_score=9, tailored_resume_path=None, state="archived")

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    assert rows == []


def test_tailor_failed_job_is_returned_within_its_retry_budget(tmp_db, seed_job):
    """tailor_failed legally transitions back to tailoring
    (VALID_TRANSITIONS["tailor_failed"] includes "tailoring") -- this is
    the existing, intentional bounded-retry mechanism
    (tailor_attempts < 5) and must keep working exactly as designed."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    job = seed_job(conn, fit_score=9, tailored_resume_path=None, state="tailor_failed", tailor_attempts=2)

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    assert [r["url"] for r in rows] == [job["url"]]


def test_tailor_failed_job_excluded_once_attempts_exhausted(tmp_db, seed_job):
    """The existing tailor_attempts < 5 cap is unchanged by this fix --
    exhausted retries stop being selected regardless of state."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    seed_job(conn, fit_score=9, tailored_resume_path=None, state="tailor_failed", tailor_attempts=5)

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    assert rows == []


def test_genuinely_eligible_scored_job_is_returned(tmp_db, seed_job):
    """The baseline positive case: a scored, high-fit, undescribed-as-
    tailored, description-present job must still be selected -- the fix
    must not be so conservative it breaks the normal path."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    job = seed_job(conn, fit_score=9, full_description="A full description.", tailored_resume_path=None, state="scored")

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    assert [r["url"] for r in rows] == [job["url"]]


def test_high_fit_score_alone_does_not_bypass_state_exclusion(tmp_db, seed_job):
    """An excluded job cannot reach pending_tailor merely because it has a
    high fit_score and no tailored_resume_path -- state is now a real,
    independent gate, not something a high score can compensate for. Covers
    every OTHER state a job could plausibly sit in besides archived/
    tailor_failed/scored, confirming the fix is a genuine positive
    allowlist, not a narrow archived-only patch."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    excluded_states = [
        "discovered",
        "enriched",
        "enrich_failed",
        "score_failed",
        "low_score",
        "tailoring",
        "tailored",
        "cover_writing",
        "cover_failed",
        "ready_to_apply",
        "applying",
        "applied",
        "apply_failed",
        "needs_human",
        "manual_only",
        "responded",
        "interview",
        "offer",
        "rejected",
        "ghosted",
    ]
    for i, state in enumerate(excluded_states):
        seed_job(conn, url_suffix=f"excluded-{i}", fit_score=9, tailored_resume_path=None, state=state)

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    assert rows == []


def test_mixed_pool_returns_only_the_two_eligible_states(tmp_db, seed_job):
    """Realistic mixed pool (mirrors the live incident's actual shape --
    mostly archived, a couple genuinely retry-eligible) -- only scored and
    tailor_failed rows come back, nothing else."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    archived = seed_job(conn, url_suffix="archived", fit_score=9, tailored_resume_path=None, state="archived")
    scored = seed_job(conn, url_suffix="scored", fit_score=9, tailored_resume_path=None, state="scored")
    retry = seed_job(
        conn, url_suffix="retry", fit_score=9, tailored_resume_path=None, state="tailor_failed", tailor_attempts=1
    )
    tailored = seed_job(
        conn, url_suffix="already-tailored", fit_score=9, tailored_resume_path="/tmp/done.docx", state="tailored"
    )

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8)
    urls = {r["url"] for r in rows}
    assert urls == {scored["url"], retry["url"]}
    assert archived["url"] not in urls
    assert tailored["url"] not in urls
