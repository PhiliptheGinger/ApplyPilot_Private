"""Regression test for the 2026-08-30 `synchronous=NORMAL` fix.

Real finding: `get_connection()` set `journal_mode=WAL` and `busy_timeout`
but never touched `synchronous`, so every connection silently ran on
SQLite's compiled-in default (FULL) -- confirmed live via `PRAGMA
synchronous` against the real production DB. FULL fsyncs on every WAL
write; on this machine's mechanical HDD that is a real, non-cached
platter-flush per commit. SQLite's own documentation guarantees WAL mode +
synchronous=NORMAL cannot corrupt the database -- only the most recent
commit(s) can be lost on an OS crash/power loss, a risk ApplyPilot already
tolerates via its stale-claim recovery mechanisms (`recover_stale_claims`,
the `applying`-lock sweep in `acquire_job`). A repo-wide grep for
"synchronous" found zero prior references in source or tests, confirming
no existing code or test depended on FULL.
"""

from __future__ import annotations


def test_get_connection_sets_synchronous_normal(tmp_db):
    conn = tmp_db()
    mode = conn.execute("PRAGMA synchronous").fetchone()
    value = mode[0] if not hasattr(mode, "keys") else mode["synchronous"]
    assert value == 1  # SQLite reports NORMAL as integer 1 (FULL=2, OFF=0)


def test_get_connection_still_sets_wal_mode_and_busy_timeout(tmp_db):
    """Regression guard -- the synchronous fix must not disturb the
    pre-existing WAL/busy_timeout setup on the same connection."""
    conn = tmp_db()
    journal = conn.execute("PRAGMA journal_mode").fetchone()
    journal_value = journal[0] if not hasattr(journal, "keys") else journal["journal_mode"]
    assert journal_value == "wal"

    busy = conn.execute("PRAGMA busy_timeout").fetchone()
    busy_value = busy[0] if not hasattr(busy, "keys") else busy["timeout"]
    assert busy_value == 30000
