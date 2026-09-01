"""Regression tests for the 2026-08-25 `pending_score` state-selection fix.

Same class of bug as `pending_tailor` (see test_pending_tailor_state_
selection.py's docstring), found by the LLM-architecture audit: `get_jobs_
by_stage`'s "pending_score" WHERE clause had NO `state` check at all --
only fit_score/score_error/score_attempts/score_next_retry_at. A job
archived for ANY reason (title-reject, revalidate-seniority, non_us_only,
...) with fit_score still NULL remained selectable forever, since archiving
doesn't set fit_score/score_error. Live-measured impact before this fix:
2,080 archived rows were wrongly re-selectable, 46 of which would have
reached a real (wasted) cloud LLM call, and all 2,080 would have had their
archived fit_score/score_reasoning/eligibility silently overwritten by
_flush_score_batch's unconditional UPDATE (see test_score_batch_archived_
guard.py for that half of the fix).

Fix: positive-select only the states VALID_TRANSITIONS (database.py)
actually permits progression from:
  - "enriched" -> "scored"/"low_score" is the primary path
  - "score_failed" -> "scored" is legal (VALID_TRANSITIONS["score_failed"]
    == frozenset({"scored", "archived"})) -- the existing give-up-after-5-
    attempts state, which a later fix/config change could legitimately
    re-score, but no query previously re-selected it
  - "archived" -> frozenset() (zero legal outgoing transitions anywhere in
    the state machine) -- must never be reachable from this query again
"""

from __future__ import annotations


def test_archived_job_never_returned_by_pending_score(tmp_db, seed_job):
    """The exact bug: an archived job with fit_score still NULL must never
    be selected again -- archived is the state machine's one true terminal
    state (VALID_TRANSITIONS["archived"] == frozenset())."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    seed_job(conn, fit_score=None, full_description="x", state="archived")

    rows = get_jobs_by_stage(conn, stage="pending_score")
    assert rows == []


def test_score_failed_job_is_returned_within_its_retry_budget(tmp_db, seed_job):
    """score_failed legally transitions back to scored
    (VALID_TRANSITIONS["score_failed"] includes "scored") -- a job parked
    there after exhausting MAX_SCORE_RETRIES must still be reachable by a
    later run (e.g. after a scoring-prompt/config fix), same as tailor's
    bounded-retry design."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    job = seed_job(
        conn,
        fit_score=None,
        full_description="x",
        state="score_failed",
        score_error="LLM error: boom",
        score_attempts=2,
    )

    rows = get_jobs_by_stage(conn, stage="pending_score")
    assert [r["url"] for r in rows] == [job["url"]]


def test_score_failed_job_excluded_once_attempts_exhausted(tmp_db, seed_job):
    """The existing score_attempts < 5 (MAX_SCORE_RETRIES) cap is unchanged
    by this fix -- exhausted retries stop being selected regardless of
    state."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    seed_job(
        conn,
        fit_score=None,
        full_description="x",
        state="score_failed",
        score_error="LLM error: boom",
        score_attempts=5,
    )

    rows = get_jobs_by_stage(conn, stage="pending_score")
    assert rows == []


def test_genuinely_eligible_enriched_job_is_returned(tmp_db, seed_job):
    """The baseline positive case: a freshly-enriched, never-scored job must
    still be selected -- the fix must not be so conservative it breaks the
    normal path."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    job = seed_job(conn, fit_score=None, full_description="A full description.", state="enriched")

    rows = get_jobs_by_stage(conn, stage="pending_score")
    assert [r["url"] for r in rows] == [job["url"]]


def test_other_states_do_not_bypass_state_exclusion(tmp_db, seed_job):
    """A job with fit_score NULL and no score_error cannot reach
    pending_score merely by having those two columns unset -- state is now
    a real, independent gate. Covers every OTHER state a job could
    plausibly sit in besides archived/score_failed/enriched, confirming the
    fix is a genuine positive allowlist, not a narrow archived-only patch."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    excluded_states = [
        "discovered",
        "enrich_failed",
        "scored",
        "low_score",
        "tailoring",
        "tailor_failed",
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
        seed_job(conn, url_suffix=f"excluded-{i}", fit_score=None, full_description="x", state=state)

    rows = get_jobs_by_stage(conn, stage="pending_score")
    assert rows == []


def test_mixed_pool_returns_only_the_two_eligible_states(tmp_db, seed_job):
    """Realistic mixed pool (mirrors the live incident's actual shape --
    mostly archived, a couple genuinely re-scoreable) -- only enriched and
    score_failed rows come back, nothing else."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    archived = seed_job(conn, url_suffix="archived", fit_score=None, full_description="x", state="archived")
    enriched = seed_job(conn, url_suffix="enriched", fit_score=None, full_description="x", state="enriched")
    retry = seed_job(
        conn,
        url_suffix="retry",
        fit_score=None,
        full_description="x",
        state="score_failed",
        score_error="LLM error: boom",
        score_attempts=1,
    )
    scored = seed_job(conn, url_suffix="already-scored", fit_score=9, full_description="x", state="scored")

    rows = get_jobs_by_stage(conn, stage="pending_score")
    urls = {r["url"] for r in rows}
    assert urls == {enriched["url"], retry["url"]}
    assert archived["url"] not in urls
    assert scored["url"] not in urls


# ---------------------------------------------------------------------------
# 2026-08-31: retry-starvation fix.
#
# Real-data finding: after the Aug 19 cloud-quota event, 3,884 jobs became
# retry-eligible (score_error set, retry window since expired), but the
# existing ORDER BY (fit_score DESC NULLS LAST, then newest-per-site-first)
# never discriminated between them and fresh jobs, since fit_score is NULL
# for the entire unscored population -- so newest-first always won. Live
# evidence: the Aug 30 continuous session scored 756 fresh jobs (91.9%) and
# only 67 old-backlog jobs (8.1%); 0 of the 3,884 retry-eligible jobs were
# rescored. These tests lock down the fix: a retry-eligible job must sort
# ahead of ANY fresh job, regardless of how much newer the fresh job is.
# ---------------------------------------------------------------------------


def test_retry_eligible_job_sorts_before_much_newer_fresh_job(tmp_db, seed_job):
    """The exact starvation scenario: an old, retry-eligible job must be
    returned before a much more recently discovered, never-attempted job."""
    from datetime import UTC, datetime, timedelta

    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    old_retry = seed_job(
        conn,
        url_suffix="old-retry",
        fit_score=None,
        full_description="x",
        state="enriched",
        score_error="LLM error: quota",
        score_attempts=1,
        score_next_retry_at=(datetime.now(UTC) - timedelta(days=11)).isoformat(),
        discovered_at=(datetime.now(UTC) - timedelta(days=12)).isoformat(),
    )
    fresh = seed_job(
        conn,
        url_suffix="fresh",
        fit_score=None,
        full_description="x",
        state="enriched",
        score_error=None,
        score_attempts=0,
        discovered_at=datetime.now(UTC).isoformat(),
    )

    rows = get_jobs_by_stage(conn, stage="pending_score")
    urls = [r["url"] for r in rows]
    assert urls == [old_retry["url"], fresh["url"]]


def test_retry_tier_ordered_by_most_overdue_first(tmp_db, seed_job):
    """Deterministic ordering WITHIN the retry-eligible tier: the job whose
    retry window opened longest ago goes first, not insertion order."""
    from datetime import UTC, datetime, timedelta

    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    recently_due = seed_job(
        conn,
        url_suffix="recently-due",
        fit_score=None,
        full_description="x",
        state="enriched",
        score_error="LLM error: quota",
        score_attempts=1,
        score_next_retry_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
    )
    long_overdue = seed_job(
        conn,
        url_suffix="long-overdue",
        fit_score=None,
        full_description="x",
        state="enriched",
        score_error="LLM error: quota",
        score_attempts=1,
        score_next_retry_at=(datetime.now(UTC) - timedelta(days=11)).isoformat(),
    )

    rows = get_jobs_by_stage(conn, stage="pending_score")
    urls = [r["url"] for r in rows]
    assert urls == [long_overdue["url"], recently_due["url"]]


def test_retry_not_yet_due_is_still_excluded_entirely(tmp_db, seed_job):
    """The fix must not weaken the existing cooldown -- a job whose retry
    window hasn't opened yet must not be selected at all, regardless of
    ordering."""
    from datetime import UTC, datetime, timedelta

    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    not_yet_due = seed_job(
        conn,
        url_suffix="not-yet-due",
        fit_score=None,
        full_description="x",
        state="enriched",
        score_error="LLM error: quota",
        score_attempts=1,
        score_next_retry_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )

    rows = get_jobs_by_stage(conn, stage="pending_score")
    assert rows == []
    assert not_yet_due  # seeded successfully; just never selectable yet


def test_exhausted_retry_budget_still_excluded(tmp_db, seed_job):
    """Unaffected by the ordering change: a job at/above the retry-attempt
    ceiling stays excluded, priority tier or not."""
    from datetime import UTC, datetime, timedelta

    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    exhausted = seed_job(
        conn,
        url_suffix="exhausted",
        fit_score=None,
        full_description="x",
        state="enriched",
        score_error="LLM error: quota",
        score_attempts=5,
        score_next_retry_at=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
    )

    rows = get_jobs_by_stage(conn, stage="pending_score")
    assert rows == []
    assert exhausted


def test_fresh_only_population_still_uses_recency_ordering(tmp_db, seed_job):
    """Regression guard: with no retry-eligible jobs present at all, the
    pre-existing newest-per-site-first behavior for fresh jobs is
    unchanged (the new CASE tier is a no-op when nothing has score_error)."""
    from datetime import UTC, datetime, timedelta

    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    older = seed_job(
        conn,
        url_suffix="older-fresh",
        fit_score=None,
        full_description="x",
        state="enriched",
        score_error=None,
        site="acme",
        discovered_at=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
    )
    newer = seed_job(
        conn,
        url_suffix="newer-fresh",
        fit_score=None,
        full_description="x",
        state="enriched",
        score_error=None,
        site="acme",
        discovered_at=datetime.now(UTC).isoformat(),
    )

    rows = get_jobs_by_stage(conn, stage="pending_score")
    urls = [r["url"] for r in rows]
    assert urls == [newer["url"], older["url"]]


def test_other_stage_ordering_unaffected_by_retry_tier(tmp_db, seed_job):
    """The shared ORDER BY change must not alter a DIFFERENT stage's
    result -- a "scored" job with a stale, already-cleared score_error
    (the real post-success state per _flush_score_batch) sorts purely by
    fit_score, same as before this fix."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    lower = seed_job(conn, url_suffix="lower", fit_score=5, full_description="x", state="scored", score_error=None)
    higher = seed_job(conn, url_suffix="higher", fit_score=9, full_description="x", state="scored", score_error=None)

    rows = get_jobs_by_stage(conn, stage="scored", min_score=1)
    urls = [r["url"] for r in rows]
    assert urls == [higher["url"], lower["url"]]
