"""Regression test for decision #181/#182 (2026-09-21/22): `_mark_enrich_result`'s
write used to call `conn.execute(...)`/`transition_state(...)` directly with no
retry-on-locked handling. The caller's own `commit_with_retry(conn)` (right
after this function returns) only protects the final COMMIT -- it does
nothing for a lock hit during the UPDATE statements themselves, which is
exactly where the real `sqlite3.OperationalError: database is locked`
exceptions were actually raised.

Confirmed live over a real multi-day continuous `run --stream` session:
700+ site-batch crashes across dozens of companies, each one discarding a
real, already-fetched job description and aborting every remaining job in
that site's batch for the pass. Fixed by wrapping the whole write (both the
success and error branches) in `write_with_retry`, matching decision #179's
identical fix for `run_tailoring`'s claim loop.

`sqlite3.Connection`'s `execute` attribute is read-only (C extension type),
so a thin forwarding proxy stands in for the real connection to simulate
transient lock errors on UPDATE statements only.
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
        if sql.strip().upper().startswith("UPDATE") and self.update_calls < self._fail_times:
            self.update_calls += 1
            raise sqlite3.OperationalError("database is locked")
        if sql.strip().upper().startswith("UPDATE"):
            self.update_calls += 1
        return self._real.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_transient_lock_on_success_write_is_retried_not_discarded(tmp_db, seed_job):
    from applypilot.enrichment.detail import _mark_enrich_result

    conn = tmp_db()
    job = seed_job(conn, url_suffix="lock-retry-ok", state="discovered", full_description=None)
    proxy = _FlakyConnProxy(conn, fail_times=1)

    _mark_enrich_result(
        proxy,
        job["url"],
        status="ok",
        full_description="Real, successfully-fetched description.",
        application_url="https://example.com/apply",
        error=None,
        tier=1,
        retry_count=0,
    )

    assert proxy.update_calls == 3, "1 failed attempt + 2 successful UPDATEs on retry (description + transition_state)"
    row = conn.execute("SELECT full_description, state FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    assert row["full_description"] == "Real, successfully-fetched description."
    assert row["state"] == "enriched"


def test_transient_lock_on_error_write_is_retried_not_discarded(tmp_db, seed_job):
    from applypilot.enrichment.detail import _mark_enrich_result

    conn = tmp_db()
    job = seed_job(conn, url_suffix="lock-retry-err", state="discovered", full_description=None)
    proxy = _FlakyConnProxy(conn, fail_times=1)

    _mark_enrich_result(
        proxy,
        job["url"],
        status="error",
        full_description=None,
        application_url=None,
        error="timeout",
        tier=None,
        retry_count=0,
    )

    assert proxy.update_calls == 2
    row = conn.execute("SELECT detail_error, enrich_attempts FROM jobs WHERE url = ?", (job["url"],)).fetchone()
    assert row["detail_error"] == "timeout"
    assert row["enrich_attempts"] == 1


def test_persistent_lock_propagates_cleanly_not_silently(tmp_db, seed_job, monkeypatch):
    """If the lock never clears, write_with_retry's own bounded retry
    exhausts and raises -- matching the real behavior confirmed live
    (`DB write locked: giving up after 8 attempts`) rather than silently
    swallowing the failure or hanging forever."""
    from applypilot.enrichment.detail import _mark_enrich_result

    conn = tmp_db()
    job = seed_job(conn, url_suffix="lock-persistent", state="discovered", full_description=None)
    proxy = _FlakyConnProxy(conn, fail_times=999)  # never succeeds

    import applypilot.database as db_mod

    monkeypatch.setattr(db_mod.time, "sleep", lambda *a, **k: None)  # skip real backoff delays

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _mark_enrich_result(
            proxy,
            job["url"],
            status="ok",
            full_description="Would-be content.",
            application_url=None,
            error=None,
            tier=1,
            retry_count=0,
        )
