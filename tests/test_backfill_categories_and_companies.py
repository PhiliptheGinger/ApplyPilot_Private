"""Regression tests for two real gaps found while auditing init_db()'s
backfill chain (2026-09-26), the same morning as decision #209's
`backfill_states` fix, per the user's own "look at the whole program for
this sort of bug" request:

1. `backfill_categories` had the IDENTICAL missing-lock-retry-wrapper gap
   as `backfill_states` -- a bare `conn.execute(...)` loop, called from
   the exact same `init_db()` chain, that could silently crash whichever
   scraper called `init_db()` under real sustained write contention
   (decisions #179/#182/#193/#209 are all the same bug shape, just
   different call sites).

2. `backfill_companies` was a fully-implemented, otherwise-correct
   function with NO CALLER ANYWHERE in the codebase -- confirmed by
   grepping the whole src/ tree. Since `company` is only ever set once at
   insert time, any job whose `application_url` arrived LATER (enrichment
   finding the real link post-discovery, or a human-first hand-off
   persisting a captured URL) could never get backfilled -- a real,
   live-confirmed gap (14,634 real DB rows affected at the moment this was
   found). Fixed by wiring it into init_db() alongside backfill_states/
   backfill_categories, with the same lock-retry protection added
   proactively rather than repeating decision #209's mistake of shipping
   it unguarded.
"""

from __future__ import annotations

import sqlite3

import pytest


class _FlakyConnProxy:
    """Forwards everything to a real sqlite3.Connection, except `execute`,
    which raises 'database is locked' for the first `fail_times` UPDATE
    calls (SELECTs always pass through untouched). Same pattern as
    decisions #182/#209's own regression tests."""

    def __init__(self, real_conn: sqlite3.Connection, fail_times: int):
        self._real = real_conn
        self._fail_times = fail_times
        self.update_calls = 0

    def execute(self, sql, params=()):
        if sql.strip().upper().startswith("UPDATE"):
            if self.update_calls < self._fail_times:
                self.update_calls += 1
                raise sqlite3.OperationalError("database is locked")
            self.update_calls += 1
        return self._real.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


# ── backfill_categories ─────────────────────────────────────────────────


def test_backfill_categories_derives_applied(tmp_db, seed_job):
    from applypilot.database import backfill_categories

    conn = tmp_db()
    job = seed_job(conn, url_suffix="cat-applied", apply_status="applied", apply_category=None)

    updated = backfill_categories(conn)

    assert updated == 1
    row = conn.execute("SELECT apply_category FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    assert row["apply_category"] == "applied"


def test_backfill_categories_skips_rows_with_no_signal(tmp_db, seed_job):
    from applypilot.database import backfill_categories

    conn = tmp_db()
    seed_job(conn, url_suffix="cat-none", apply_status=None, apply_error=None, apply_category=None)

    assert backfill_categories(conn) == 0


def test_backfill_categories_transient_lock_is_retried_not_discarded(tmp_db, seed_job):
    from applypilot.database import backfill_categories

    conn = tmp_db()
    job = seed_job(conn, url_suffix="cat-lock-retry", apply_status="applied", apply_category=None)
    proxy = _FlakyConnProxy(conn, fail_times=1)

    updated = backfill_categories(proxy)

    assert updated == 1
    assert proxy.update_calls == 2, "1 failed attempt + 1 successful UPDATE on the full-batch retry"
    row = conn.execute("SELECT apply_category FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    assert row["apply_category"] == "applied"


def test_backfill_categories_persistent_lock_propagates_cleanly(tmp_db, seed_job, monkeypatch):
    from applypilot.database import backfill_categories
    import applypilot.database as db_mod

    conn = tmp_db()
    seed_job(conn, url_suffix="cat-lock-persist", apply_status="applied", apply_category=None)
    proxy = _FlakyConnProxy(conn, fail_times=999)
    monkeypatch.setattr(db_mod.time, "sleep", lambda *a, **k: None)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        backfill_categories(proxy)


# ── backfill_companies ───────────────────────────────────────────────────


def test_backfill_companies_extracts_from_ashby_url(tmp_db, seed_job):
    from applypilot.database import backfill_companies

    conn = tmp_db()
    job = seed_job(
        conn,
        url_suffix="co-ashby",
        application_url="https://jobs.ashbyhq.com/ramp/29d188c7-0529-4bc8-a04a-57902df2a7ae",
        company=None,
    )

    updated = backfill_companies(conn)

    assert updated == 1
    row = conn.execute("SELECT company FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    assert row["company"] == "ramp"


def test_backfill_companies_skips_rows_with_no_application_url(tmp_db, seed_job):
    from applypilot.database import backfill_companies

    conn = tmp_db()
    seed_job(conn, url_suffix="co-none", application_url=None, company=None)

    assert backfill_companies(conn) == 0


def test_backfill_companies_skips_rows_already_having_a_company(tmp_db, seed_job):
    from applypilot.database import backfill_companies

    conn = tmp_db()
    seed_job(
        conn,
        url_suffix="co-existing",
        application_url="https://jobs.ashbyhq.com/notion/x",
        company="already-set",
    )

    assert backfill_companies(conn) == 0
    row = conn.execute(
        "SELECT company FROM jobs WHERE url = 'https://example.com/job/co-existing'"
    ).fetchone()
    assert row["company"] == "already-set"


def test_backfill_companies_transient_lock_is_retried_not_discarded(tmp_db, seed_job):
    from applypilot.database import backfill_companies

    conn = tmp_db()
    job = seed_job(
        conn,
        url_suffix="co-lock-retry",
        application_url="https://job-boards.greenhouse.io/pendo/jobs/8596202002",
        company=None,
    )
    proxy = _FlakyConnProxy(conn, fail_times=1)

    updated = backfill_companies(proxy)

    assert updated == 1
    assert proxy.update_calls == 2
    row = conn.execute("SELECT company FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    assert row["company"] == "pendo"


def test_backfill_companies_persistent_lock_propagates_cleanly(tmp_db, seed_job, monkeypatch):
    from applypilot.database import backfill_companies
    import applypilot.database as db_mod

    conn = tmp_db()
    seed_job(
        conn,
        url_suffix="co-lock-persist",
        application_url="https://jobs.ashbyhq.com/ramp/x",
        company=None,
    )
    proxy = _FlakyConnProxy(conn, fail_times=999)
    monkeypatch.setattr(db_mod.time, "sleep", lambda *a, **k: None)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        backfill_companies(proxy)


def test_init_db_calls_backfill_companies(tmp_db, seed_job, monkeypatch):
    """The real bug: backfill_companies existed but had no caller anywhere.
    This pins the actual wiring, not just the function's own logic."""
    from applypilot.database import init_db

    conn = tmp_db()
    seed_job(
        conn,
        url_suffix="co-initdb",
        application_url="https://jobs.ashbyhq.com/ramp/x",
        company=None,
    )
    conn.commit()

    init_db()  # re-runs full init against the same tmp-patched DB_PATH

    row = conn.execute(
        "SELECT company FROM jobs WHERE url = 'https://example.com/job/co-initdb'"
    ).fetchone()
    assert row["company"] == "ramp"
