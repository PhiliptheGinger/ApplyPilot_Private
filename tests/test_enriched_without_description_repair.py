"""E8 (2026-10-07): jobs stuck as 'enriched' with no full_description.

smartextract used to insert them; they could never be scored and every
enrichment pass re-scraped them and discarded the result.
"""

from applypilot.database import repair_enriched_without_description
from applypilot.enrichment.detail import _mark_enrich_result


def _state(conn, url):
    return conn.execute("SELECT state FROM jobs WHERE url = ?", (url,)).fetchone()[0]


def test_stuck_rows_move_back_to_discovered_and_enrichment_then_completes(tmp_db, seed_job):
    conn = tmp_db()
    stuck = seed_job(conn, url_suffix="stuck", state="enriched", full_description=None)
    conn.execute("UPDATE jobs SET detail_scraped_at = NULL WHERE url = ?", (stuck["url"],))
    conn.commit()

    assert repair_enriched_without_description(conn) == 1
    assert _state(conn, stuck["url"]) == "discovered"
    reason = conn.execute(
        "SELECT reason FROM job_state_transitions WHERE job_url = ? ORDER BY id DESC LIMIT 1", (stuck["url"],)
    ).fetchone()[0]
    assert "E8" in reason

    # Before the repair this completion was skipped as "stale".
    _mark_enrich_result(
        conn,
        stuck["url"],
        status="ok",
        full_description="A real, long job description. " * 20,
        application_url=None,
        error=None,
        tier=2,
        retry_count=0,
    )
    conn.commit()
    row = conn.execute("SELECT state, full_description FROM jobs WHERE url = ?", (stuck["url"],)).fetchone()
    assert row[0] == "enriched" and row[1]


def test_real_enriched_rows_untouched_and_repair_is_idempotent(tmp_db, seed_job):
    conn = tmp_db()
    real = seed_job(conn, url_suffix="real", state="enriched", full_description="Real description. " * 20)
    assert repair_enriched_without_description(conn) == 0
    assert _state(conn, real["url"]) == "enriched"


def test_init_db_runs_the_repair(tmp_db, monkeypatch):
    import applypilot.database as database

    calls = []
    monkeypatch.setattr(database, "repair_enriched_without_description", lambda conn=None: calls.append(1) or 0)
    tmp_db()
    assert calls
