"""Regression test for decision #179 (2026-09-21): `run_tailoring`'s
job-claim loop used to call `transition_state()` directly with no
retry-on-locked handling, unlike every other write path in tailor.py (see
`_flush_tailor_results`'s own `write_with_retry` usage). Confirmed live
during a real overnight `run --stream` session: concurrent discover/enrich
writes held SQLite's write lock often enough that EVERY tailor claim
attempt for a long stretch crashed instantly with `sqlite3.OperationalError:
database is locked`, blocking tailoring entirely for that whole window.

The fix wraps the claim loop in the same `write_with_retry` helper the rest
of this module already uses -- these tests confirm a transient lock error
during claiming is retried (not fatal) and the job still gets tailored.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch


def _fake_tailor_one_job(job, resume_text, profile, doc_format="docx"):
    return {
        "url": job["url"],
        "title": job["title"],
        "site": job.get("site", ""),
        "status": "approved",
        "attempts": 1,
        "path": "/tmp/fake.txt",
        "pdf_path": None,
        "auto_approved_by_facts": False,
    }


def test_transient_lock_during_claim_is_retried_not_fatal(tmp_db, seed_job, monkeypatch, tmp_path):
    import applypilot.database as db_mod
    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    job = seed_job(
        conn, url_suffix="lock-retry", state="scored", fit_score=9, tailored_resume_path=None,
        application_url="https://boards.greenhouse.io/acme/jobs/1",
    )

    real_transition_state = db_mod.transition_state
    calls = {"n": 0}

    def _flaky_transition_state(conn, url, to_state, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1 and to_state == "tailoring":
            raise sqlite3.OperationalError("database is locked")
        return real_transition_state(conn, url, to_state, **kwargs)

    monkeypatch.setattr(db_mod, "transition_state", _flaky_transition_state)
    monkeypatch.setattr(tailor_mod, "_tailor_one_job", _fake_tailor_one_job)
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    result = tailor_mod.run_tailoring(min_score=8, limit=3)

    assert result["approved"] == 1, "the job must still get tailored after the claim retry succeeds"
    assert calls["n"] >= 2, "must have actually retried the claim after the simulated lock error"
    row = conn.execute("SELECT state FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    assert row["state"] == "tailored"


def test_persistent_lock_during_claim_does_not_crash_run_tailoring(tmp_db, seed_job, monkeypatch, tmp_path):
    """If the lock never clears (all retries exhausted), write_with_retry's
    own bounded-retry behavior applies -- run_tailoring must propagate a
    clean, catchable error rather than the whole pipeline process dying on
    an unhandled sqlite3.OperationalError (matching the real traceback seen
    live, which crashed the calling `run --stream` pass every single time)."""
    import applypilot.database as db_mod
    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    seed_job(
        conn, url_suffix="lock-persistent", state="scored", fit_score=9, tailored_resume_path=None,
        application_url="https://boards.greenhouse.io/acme/jobs/2",
    )

    def _always_locked(conn, url, to_state, **kwargs):
        if to_state == "tailoring":
            raise sqlite3.OperationalError("database is locked")
        raise AssertionError("should never reach a non-claim transition_state call in this test")

    monkeypatch.setattr(db_mod, "transition_state", _always_locked)
    monkeypatch.setattr(tailor_mod, "_tailor_one_job", _fake_tailor_one_job)
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)
    monkeypatch.setattr(tailor_mod.time, "sleep", lambda *a, **k: None)  # skip real backoff delays

    import pytest

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        tailor_mod.run_tailoring(min_score=8, limit=3)
