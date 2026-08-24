"""Regression tests for the 2026-08-25 `_flush_score_batch` archived-row
overwrite guard (paired with the `pending_score` state-selection fix in
test_pending_score_state_selection.py).

Found by the LLM-architecture audit: `_flush_score_batch`'s UPDATE of
fit_score/score_reasoning/eligibility ran unconditionally, before the state
transition was even attempted. Checking `transition_state`'s return value
alone would not have been enough to guard it: the non_us_only branch calls
`transition_state(..., force=True)`, which bypasses validation entirely,
and even without force, an archived->archived self-transition is trivially
"allowed" (to_state == from_state) -- neither rejects an already-archived
row. `_flush_score_batch` now looks up the job's real current state first
and skips the write entirely when it's already "archived" (the state
machine's one true terminal state), which is the only guard that actually
closes both branches. This also defends the `--rescore` CLI path, which
selects on `full_description IS NOT NULL` alone with no state filter at
all, so an archived row can still reach this function directly even after
the `pending_score` query-side fix.
"""

from __future__ import annotations

from datetime import UTC, datetime


def _now() -> str:
    return datetime.now(UTC).isoformat()


def test_archived_job_score_and_reasoning_are_not_overwritten(tmp_db, seed_job):
    """The exact bug: an archived job re-scored via a stale selection (e.g.
    --rescore) must not have its fit_score/score_reasoning/eligibility
    clobbered, and must not leave the terminal "archived" state."""
    from applypilot.scoring.scorer import _flush_score_batch

    conn = tmp_db()
    job = seed_job(
        conn,
        fit_score=None,
        full_description="x",
        state="archived",
        score_reasoning="original archived reasoning",
        eligibility="non_us_only",
    )

    batch = [
        {
            "url": job["url"],
            "score": 8,
            "keywords": "python, aws",
            "reasoning": "a fresh, wrongly-recomputed score",
            "eligibility": "eligible",
        }
    ]
    _flush_score_batch(conn, batch, _now())

    row = conn.execute(
        "SELECT fit_score, score_reasoning, eligibility, state FROM jobs WHERE url = ?",
        (job["url"],),
    ).fetchone()
    assert row["fit_score"] is None
    assert row["score_reasoning"] == "original archived reasoning"
    assert row["eligibility"] == "non_us_only"
    assert row["state"] == "archived"


def test_archived_job_not_overwritten_even_when_new_score_is_non_us_only(tmp_db, seed_job):
    """The force=True branch (eligibility == "non_us_only") is the loophole
    a naive "check transition_state's return value" guard would miss --
    transition_state(..., force=True) always succeeds, and archived->
    archived is trivially allowed anyway. Must still be blocked by the
    current-state lookup."""
    from applypilot.scoring.scorer import _flush_score_batch

    conn = tmp_db()
    job = seed_job(
        conn,
        fit_score=None,
        full_description="x",
        state="archived",
        score_reasoning="original archived reasoning",
    )

    batch = [
        {
            "url": job["url"],
            "score": 2,
            "keywords": "",
            "reasoning": "Ineligible: non-US geography.",
            "eligibility": "non_us_only",
        }
    ]
    _flush_score_batch(conn, batch, _now())

    row = conn.execute(
        "SELECT fit_score, score_reasoning, state FROM jobs WHERE url = ?",
        (job["url"],),
    ).fetchone()
    assert row["fit_score"] is None
    assert row["score_reasoning"] == "original archived reasoning"
    assert row["state"] == "archived"


def test_genuinely_enriched_job_is_still_scored_normally(tmp_db, seed_job):
    """Contrast case: the guard must not be so broad it blocks legitimate
    scoring of a non-archived job -- the normal write path is unaffected."""
    from applypilot.scoring.scorer import _flush_score_batch

    conn = tmp_db()
    job = seed_job(conn, fit_score=None, full_description="x", state="enriched")

    batch = [
        {
            "url": job["url"],
            "score": 9,
            "keywords": "python",
            "reasoning": "strong match",
            "eligibility": "eligible",
        }
    ]
    _flush_score_batch(conn, batch, _now())

    row = conn.execute(
        "SELECT fit_score, score_reasoning, state FROM jobs WHERE url = ?",
        (job["url"],),
    ).fetchone()
    assert row["fit_score"] == 9
    assert "strong match" in row["score_reasoning"]
    assert row["state"] == "scored"
