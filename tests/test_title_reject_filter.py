from pathlib import Path

from applypilot.database import get_connection, init_db, reject_jobs_by_title_patterns


def _seed_job(conn, *, url: str, title: str, applied: bool = False) -> None:
    conn.execute(
        "INSERT INTO jobs (url, title, discovered_at, state, applied_at) VALUES (?, ?, datetime('now'), 'discovered', ?)",
        (url, title, "2026-01-01T00:00:00+00:00" if applied else None),
    )
    conn.commit()


def test_reject_titles_dry_run_only_reports_matches(tmp_path: Path):
    db = tmp_path / "jobs.db"
    conn = init_db(db)

    _seed_job(conn, url="https://example.com/s1", title="Senior Backend Engineer")
    _seed_job(conn, url="https://example.com/j1", title="Junior Backend Engineer")

    result = reject_jobs_by_title_patterns([r"\bsenior\b"], conn=conn, dry_run=True)

    assert result["matched"] == 1
    assert result["updated"] == 0

    row = conn.execute("SELECT state, apply_category FROM jobs WHERE url = ?", ("https://example.com/s1",)).fetchone()
    assert row["state"] == "discovered"
    assert row["apply_category"] is None


def test_reject_titles_archives_matching_unapplied_jobs(tmp_path: Path):
    db = tmp_path / "jobs.db"
    conn = init_db(db)

    _seed_job(conn, url="https://example.com/s1", title="Senior Platform Engineer")
    _seed_job(conn, url="https://example.com/s2", title="Staff Data Engineer")
    _seed_job(conn, url="https://example.com/j1", title="Junior QA Engineer")
    _seed_job(conn, url="https://example.com/a1", title="Senior Site Reliability Engineer", applied=True)

    result = reject_jobs_by_title_patterns([r"\bsenior\b", r"\bstaff\b"], conn=conn, dry_run=False)

    assert result["matched"] == 2
    assert result["updated"] == 2

    archived = conn.execute(
        "SELECT url, state, apply_category FROM jobs WHERE url IN (?, ?)",
        ("https://example.com/s1", "https://example.com/s2"),
    ).fetchall()
    assert all(r["state"] == "archived" for r in archived)
    assert all(r["apply_category"] == "archived_ineligible" for r in archived)

    junior = conn.execute("SELECT state FROM jobs WHERE url = ?", ("https://example.com/j1",)).fetchone()
    assert junior["state"] == "discovered"

    applied = conn.execute("SELECT state FROM jobs WHERE url = ?", ("https://example.com/a1",)).fetchone()
    assert applied["state"] == "discovered"
