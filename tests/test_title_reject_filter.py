from pathlib import Path

from applypilot.database import get_connection, get_jobs_by_stage, init_db, reject_jobs_by_title_patterns


def _seed_job(conn, *, url: str, title: str, applied: bool = False, state: str | None = None, fit_score: int | None = 9, tailored_resume_path: str | None = None, cover_letter_path: str | None = None) -> None:
    conn.execute(
        "INSERT INTO jobs (url, title, discovered_at, state, applied_at, fit_score, tailored_resume_path, cover_letter_path, full_description) VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)",
        (
            url,
            title,
            state or "discovered",
            "2026-01-01T00:00:00+00:00" if applied else None,
            fit_score,
            tailored_resume_path,
            cover_letter_path,
            "A full description.",
        ),
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


def test_reject_titles_does_not_archive_tailored_jobs(tmp_path: Path):
    db = tmp_path / "jobs.db"
    conn = init_db(db)

    _seed_job(conn, url="https://example.com/tailored-senior", title="Senior Platform Engineer", state="tailored",
              tailored_resume_path="/tmp/resume.txt")

    result = reject_jobs_by_title_patterns([r"\bsenior\b"], conn=conn, dry_run=False)

    assert result["matched"] == 0
    assert result["updated"] == 0
    row = conn.execute("SELECT state, apply_category, apply_error FROM jobs WHERE url = ?", ("https://example.com/tailored-senior",)).fetchone()
    assert row["state"] == "tailored"
    assert row["apply_category"] is None
    assert row["apply_error"] is None


def test_reject_titles_does_not_archive_cover_failed_or_ready_to_apply(tmp_path: Path):
    db = tmp_path / "jobs.db"
    conn = init_db(db)

    _seed_job(conn, url="https://example.com/cover-failed-senior", title="Senior Platform Engineer", state="cover_failed",
              fit_score=9, tailored_resume_path="/tmp/resume.txt")
    _seed_job(conn, url="https://example.com/ready-senior", title="Senior Platform Engineer", state="ready_to_apply",
              fit_score=9, tailored_resume_path="/tmp/resume.txt", cover_letter_path="/tmp/cover.txt")

    result = reject_jobs_by_title_patterns([r"\bsenior\b"], conn=conn, dry_run=False)

    assert result["matched"] == 0
    assert result["updated"] == 0
    cover_failed = conn.execute("SELECT state FROM jobs WHERE url = ?", ("https://example.com/cover-failed-senior",)).fetchone()
    ready = conn.execute("SELECT state FROM jobs WHERE url = ?", ("https://example.com/ready-senior",)).fetchone()
    assert cover_failed["state"] == "cover_failed"
    assert ready["state"] == "ready_to_apply"


def test_pending_cover_only_selects_valid_states(tmp_path: Path):
    db = tmp_path / "jobs.db"
    conn = init_db(db)

    _seed_job(conn, url="https://example.com/pending-tailored", title="Software Engineer", state="tailored",
              fit_score=9, tailored_resume_path="/tmp/resume.txt", cover_letter_path=None)
    _seed_job(conn, url="https://example.com/pending-cover-failed", title="Software Engineer", state="cover_failed",
              fit_score=9, tailored_resume_path="/tmp/resume.txt", cover_letter_path=None)
    _seed_job(conn, url="https://example.com/archive-should-skip", title="Software Engineer", state="archived",
              fit_score=9, tailored_resume_path="/tmp/resume.txt", cover_letter_path=None)

    rows = get_jobs_by_stage(conn, stage="pending_cover", min_score=8, max_age_days=0)
    urls = {r["url"] for r in rows}

    assert "https://example.com/pending-tailored" in urls
    assert "https://example.com/pending-cover-failed" in urls
    assert "https://example.com/archive-should-skip" not in urls
