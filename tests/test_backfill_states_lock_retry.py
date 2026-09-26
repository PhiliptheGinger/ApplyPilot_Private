"""Regression test (2026-09-25/26 overnight log check): `backfill_states`'s
per-row writes (`UPDATE jobs`, `INSERT job_state_transitions`) used to run as
bare `conn.execute(...)` calls with no lock-retry handling at all, followed by
a single trailing `commit_with_retry(conn)` -- the same missing-wrapper gap
already fixed elsewhere for `_mark_enrich_result` (#182), `run_tailoring`'s
claim loop (#179), and `recover_stale_claims` (#193).

Confirmed live in a real overnight `run --stream` session: `init_db()` calls
`backfill_states()` unconditionally at the very top of *every* discovery
scraper's entry point (`ats_common.run_ats_crawl`, and each of
workday.py/amazon.py/costco.py/builtin.py/smartextract.py/hackernews.py
directly). Under the sustained multi-stage write contention this file's own
decision #183 already documents as a genuine, unresolved capacity ceiling on
this machine, a single locked UPDATE inside `backfill_states` aborted the
ENTIRE calling scraper immediately -- 9 distinct scrapers (Workday,
Greenhouse, Lever, Ashby, Amazon, Costco, BuiltIn, Smart extract, HN
discovery) failed this way within an 11-minute window in one real log.

Fixed by wrapping the whole per-row loop in one `write_with_retry`-guarded
closure (same pattern as #179/#182/#193): a lock mid-batch rolls back and
retries the WHOLE batch from scratch, safe here because nothing is committed
until the closure returns successfully, and the closure recomputes `counts`
from the same already-fetched `candidates` list every time.
"""

from __future__ import annotations

import sqlite3

import pytest


class _FlakyConnProxy:
    """Forwards everything to a real sqlite3.Connection, except `execute`,
    which raises 'database is locked' for the first `fail_times` UPDATE
    calls (SELECTs always pass through untouched)."""

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


def test_transient_lock_on_backfill_write_is_retried_not_discarded(tmp_db, seed_job):
    from applypilot.database import backfill_states, current_state

    conn = tmp_db()
    job = seed_job(conn, url_suffix="backfill-lock-retry", fit_score=3)
    proxy = _FlakyConnProxy(conn, fail_times=1)

    counts = backfill_states(proxy)

    assert proxy.update_calls == 2, "1 failed UPDATE attempt + 1 successful UPDATE on the full-batch retry"
    assert counts == {"low_score": 1}
    assert current_state(conn, job["url"]) == "low_score"


def test_persistent_lock_propagates_cleanly_not_silently(tmp_db, seed_job, monkeypatch):
    """If the lock never clears, write_with_retry's own bounded retry
    exhausts and raises -- matching the real behavior confirmed live
    (`DB write locked: giving up after 8 attempts`) rather than silently
    swallowing the failure, returning a bogus empty result, or hanging."""
    from applypilot.database import backfill_states
    import applypilot.database as db_mod

    conn = tmp_db()
    seed_job(conn, url_suffix="backfill-lock-persistent", fit_score=3)
    proxy = _FlakyConnProxy(conn, fail_times=999)  # never succeeds

    monkeypatch.setattr(db_mod.time, "sleep", lambda *a, **k: None)  # skip real backoff delays

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        backfill_states(proxy)


def test_backfill_with_no_candidates_never_touches_write_with_retry(tmp_db, seed_job, monkeypatch):
    """Nothing to backfill must stay a pure no-op -- confirms the `if
    candidates:` guard still short-circuits correctly after the refactor."""
    from applypilot.database import backfill_states, write_with_retry

    conn = tmp_db()
    seed_job(
        conn,
        url_suffix="backfill-noop",
        fit_score=9,
        tailored_resume_path="/tmp/r.docx",
        cover_letter_path="/tmp/c.docx",
        application_url="https://ex.com/apply",
    )
    backfill_states(conn)  # first call does the real backfill

    calls = []
    import applypilot.database as db_mod

    monkeypatch.setattr(db_mod, "write_with_retry", lambda *a, **k: calls.append(1) or write_with_retry(*a, **k))

    second = backfill_states(conn)

    assert second == {}
    assert calls == [], "no candidates left -> write_with_retry must not be invoked at all"
