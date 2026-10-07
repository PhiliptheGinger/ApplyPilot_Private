"""write_with_retry must roll back when the write function raises a
non-lock error.

2026-10-07 audit: write_with_retry only rolled back on "database is
locked". Any other exception raised after the function had already
written (e.g. transition_state's ValueError("Job not found") for a row
deleted mid-batch) re-raised with the transaction still open. Connections
are cached per thread for the process lifetime, so that open transaction
kept SQLite's single write lock until the same thread happened to commit
or roll back -- every other writer in the process (and the apply process)
waited out busy_timeout and then failed with "database is locked". This is
a candidate root cause for the sustained lock exhaustion in decisions
#183/#192/#213.
"""

import sqlite3

import pytest

from applypilot.database import write_with_retry


def _make_db(path):
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO t (id, v) VALUES (1, 'a')")
    conn.commit()
    return conn


def test_non_lock_error_rolls_back_and_releases_write_lock(tmp_path):
    db = tmp_path / "x.db"
    writer = _make_db(db)

    def write_then_fail():
        writer.execute("UPDATE t SET v = 'partial' WHERE id = 1")
        raise ValueError("Job not found: https://example.com/gone")

    with pytest.raises(ValueError):
        write_with_retry(writer, write_then_fail)

    assert not writer.in_transaction

    other = sqlite3.connect(db, timeout=0.2)
    other.execute("UPDATE t SET v = 'other' WHERE id = 1")  # must not raise "database is locked"
    other.commit()
    assert other.execute("SELECT v FROM t WHERE id = 1").fetchone()[0] == "other"


def test_partial_write_is_discarded(tmp_path):
    db = tmp_path / "x.db"
    writer = _make_db(db)

    def write_then_fail():
        writer.execute("UPDATE t SET v = 'partial' WHERE id = 1")
        raise KeyError("boom")

    with pytest.raises(KeyError):
        write_with_retry(writer, write_then_fail)

    assert writer.execute("SELECT v FROM t WHERE id = 1").fetchone()[0] == "a"


def test_non_lock_operational_error_also_rolls_back(tmp_path):
    db = tmp_path / "x.db"
    writer = _make_db(db)

    def write_then_bad_sql():
        writer.execute("UPDATE t SET v = 'partial' WHERE id = 1")
        writer.execute("SELECT no_such_column FROM t")

    with pytest.raises(sqlite3.OperationalError):
        write_with_retry(writer, write_then_bad_sql)

    assert not writer.in_transaction


def test_store_qa_failure_does_not_leave_transaction_open(tmp_db, monkeypatch):
    """store_qa swallows errors (best-effort cache write). It must not
    swallow them with its INSERT still uncommitted on the cached connection."""
    import applypilot.database as database

    conn = tmp_db()

    def failing_commit(c, *a, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(database, "commit_with_retry", failing_commit)

    assert database.store_qa("Are you authorized to work in the US?", "Yes", conn=conn) is None
    assert not conn.in_transaction
