"""Regression test for the 2026-08-30 tailor/cover per-company cap index fix.

Real incident: `run_tailoring`'s and `run_cover_letters`' per-company cap
queries (tailor.py/cover_letter.py) filter on `tailored_resume_path`/
`cover_letter_path IS NOT NULL` and run on EVERY streaming stage pass -- as
often as every ~10s for the life of a long-running `--stream` run. Confirmed
live via `EXPLAIN QUERY PLAN` against the real production DB (24,653 rows /
236 MB) that this was an unindexed `SCAN jobs` -- real, growing I/O
amplification on a machine whose HDD has shown sustained 100% active time
under load. Fix: two partial indexes on `discovered_at`, one per predicate,
added in `database.ensure_columns` (idempotent, runs on every `init_db`).
"""

from __future__ import annotations


def _index_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    return {r["name"] if hasattr(r, "keys") else r[0] for r in rows}


def test_ensure_columns_creates_the_tailored_and_cover_cap_indexes(tmp_db):
    """The exact two indexes the cap queries need must exist after init --
    tmp_db's factory already calls get_connection() -> init_db() ->
    ensure_columns() once to create the fixture DB."""
    conn = tmp_db()
    names = _index_names(conn)
    assert "idx_jobs_tailored_cap" in names
    assert "idx_jobs_cover_cap" in names


def test_ensure_columns_is_idempotent_for_the_new_indexes(tmp_db):
    """Calling ensure_columns again (as every init_db() call does) must not
    raise on the already-existing partial indexes."""
    from applypilot.database import ensure_columns

    conn = tmp_db()
    ensure_columns(conn)  # must not raise
    names = _index_names(conn)
    assert "idx_jobs_tailored_cap" in names
    assert "idx_jobs_cover_cap" in names


def test_tailored_cap_query_uses_the_partial_index_not_a_full_scan(tmp_db, seed_job):
    """Reproduces the real-incident query shape (tailor.py's per-company cap
    aggregate) against a fixture DB with enough tailored rows for SQLite's
    planner to prefer the partial index over a full scan (mirrors the live
    EXPLAIN QUERY PLAN check: 'SCAN jobs' before the fix, 'SEARCH jobs USING
    INDEX idx_jobs_tailored_cap' after)."""
    conn = tmp_db()
    for i in range(200):
        seed_job(
            conn,
            fit_score=9,
            tailored_resume_path=f"/tmp/resume_{i}.txt" if i % 20 == 0 else None,
            company=f"company-{i}",
            state="tailored",
        )

    plan = conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT LOWER(company) AS key, COUNT(*) AS n
        FROM jobs
        WHERE tailored_resume_path IS NOT NULL
          AND state != 'archived'
          AND company IS NOT NULL AND TRIM(company) != ''
          AND discovered_at > datetime('now', '-14 days')
        GROUP BY key
        """
    ).fetchall()
    detail = " | ".join(row[3] if not hasattr(row, "keys") else row["detail"] for row in plan)
    assert "idx_jobs_tailored_cap" in detail
    assert "SCAN jobs" not in detail
