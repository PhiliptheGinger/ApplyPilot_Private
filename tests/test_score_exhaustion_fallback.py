"""Regression tests for CLAUDE.md decision #119 (2026-09-11): once a job has
exhausted MAX_SCORE_RETRIES cloud attempts, `_flush_score_batch` now tries
the local deterministic-fallback scorer ONE time as a genuine last resort
before permanently giving up (`state="score_failed"`).

This narrows, rather than reverses, decision #76's original "manual
invocation only" caution for scoring.deterministic_fallback -- it only
fires at the exact moment a job would otherwise die forever, never during
the normal fast-cloud-scoring happy path. Found via a real production run
(decision #117) that left 62 real jobs one bad-quota-timing-roll away from
permanent failure with no automatic recovery at all.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch


def _now() -> str:
    return datetime.now(UTC).isoformat()


def test_exhausted_job_rescued_by_local_fallback_instead_of_failing(tmp_db, seed_job):
    """The core fix: a job on its 5th failed cloud attempt gets one local
    rescue attempt instead of being marked score_failed. On success it ends
    up looking exactly like a normal successful score, tagged with the
    fallback's own score_method so it stays revalidation-eligible."""
    from applypilot.scoring.scorer import MAX_SCORE_RETRIES, _flush_score_batch

    conn = tmp_db()
    job = seed_job(conn, fit_score=None, state="enriched", score_attempts=MAX_SCORE_RETRIES)

    fake_result = {
        "score": 6,
        "keywords": "",
        "reasoning": "[deterministic fallback, model=qwen3:8b] family=it_or_tech_support years_required=None cs_degree_required=False",
        "eligibility": "eligible",
    }
    with patch(
        "applypilot.scoring.deterministic_fallback.score_job_deterministic",
        return_value=fake_result,
    ) as mock_fallback:
        batch = [{"url": job["url"], "score": None, "keywords": "", "reasoning": "", "error": "LLM error: quota cooldown"}]
        _flush_score_batch(conn, batch, _now())

    assert mock_fallback.called
    row = conn.execute(
        "SELECT fit_score, state, score_method, score_attempts, score_next_retry_at, score_error FROM jobs WHERE url = ?",
        (job["url"],),
    ).fetchone()
    assert row["fit_score"] == 6
    assert row["state"] == "low_score"  # below the funnel's min_score=8
    assert row["score_method"] == "deterministic_fallback"
    assert row["score_attempts"] == 0
    assert row["score_next_retry_at"] is None
    assert row["score_error"] is None


def test_exhausted_job_still_fails_permanently_when_local_fallback_also_unavailable(tmp_db, seed_job):
    """Defense in depth: if the local model itself is unreachable/erroring,
    the original give-up behavior must still fire unchanged -- this new
    rescue path is a genuine last resort, not a new single point of
    failure that could silently swallow the score_failed transition."""
    from applypilot.scoring.scorer import MAX_SCORE_RETRIES, _flush_score_batch

    conn = tmp_db()
    job = seed_job(conn, fit_score=None, state="enriched", score_attempts=MAX_SCORE_RETRIES)

    with patch(
        "applypilot.scoring.deterministic_fallback.score_job_deterministic",
        side_effect=RuntimeError("local model unreachable"),
    ):
        batch = [{"url": job["url"], "score": None, "keywords": "", "reasoning": "", "error": "LLM error: quota cooldown"}]
        _flush_score_batch(conn, batch, _now())

    row = conn.execute(
        "SELECT fit_score, state, score_attempts, score_next_retry_at FROM jobs WHERE url = ?",
        (job["url"],),
    ).fetchone()
    assert row["fit_score"] is None
    assert row["state"] == "score_failed"
    assert row["score_attempts"] == MAX_SCORE_RETRIES + 1
    assert row["score_next_retry_at"] is None


def test_job_below_retry_limit_never_triggers_local_fallback(tmp_db, seed_job):
    """The rescue path must ONLY fire at genuine exhaustion -- a job still
    within its normal retry budget schedules the usual backoff and never
    calls the (comparatively expensive) local model at all."""
    from applypilot.scoring.scorer import _flush_score_batch

    conn = tmp_db()
    job = seed_job(conn, fit_score=None, state="enriched", score_attempts=2)

    with patch("applypilot.scoring.deterministic_fallback.score_job_deterministic") as mock_fallback:
        batch = [{"url": job["url"], "score": None, "keywords": "", "reasoning": "", "error": "LLM error: quota cooldown"}]
        _flush_score_batch(conn, batch, _now())

    mock_fallback.assert_not_called()
    row = conn.execute(
        "SELECT fit_score, state, score_attempts, score_next_retry_at FROM jobs WHERE url = ?",
        (job["url"],),
    ).fetchone()
    assert row["fit_score"] is None
    assert row["state"] == "enriched"
    assert row["score_attempts"] == 3
    assert row["score_next_retry_at"] is not None


def test_exhausted_job_rescued_into_ineligible_archival_when_fallback_says_so(tmp_db, seed_job):
    """The rescue path reuses the SAME success-path logic as a normal
    score, so a fallback verdict of non_us_only (etc.) correctly archives
    the job -- not just a bare fit_score write."""
    from applypilot.scoring.scorer import MAX_SCORE_RETRIES, _flush_score_batch

    conn = tmp_db()
    job = seed_job(conn, fit_score=None, state="enriched", score_attempts=MAX_SCORE_RETRIES)

    fake_result = {
        "score": 2,
        "keywords": "",
        "reasoning": "Ineligible: non-US location field: Toronto, Ontario, Canada.",
        "eligibility": "non_us_only",
    }
    with patch(
        "applypilot.scoring.deterministic_fallback.score_job_deterministic",
        return_value=fake_result,
    ):
        batch = [{"url": job["url"], "score": None, "keywords": "", "reasoning": "", "error": "LLM error: quota cooldown"}]
        _flush_score_batch(conn, batch, _now())

    row = conn.execute(
        "SELECT fit_score, state, eligibility, score_method FROM jobs WHERE url = ?",
        (job["url"],),
    ).fetchone()
    assert row["fit_score"] == 2
    assert row["state"] == "archived"
    assert row["eligibility"] == "non_us_only"
    assert row["score_method"] == "deterministic_fallback"
