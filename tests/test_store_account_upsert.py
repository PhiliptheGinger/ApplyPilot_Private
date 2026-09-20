"""Regression test for the 2026-09-20 store_account upsert fix.

Root cause (confirmed live): store_account did a blind INSERT with no
dedup, so a single real Zillow account got 6 identical rows across 3 real
apply attempts on different days -- always the same domain/email/password.
Two contributing causes: (a) a single real run's ACCOUNT_CREATED line gets
parsed twice within one call (Claude Code's own "result" message duplicates
already-streamed assistant text), and (b) genuinely separate real attempts
on different days each re-create the same account. store_account now
upserts on (domain, email) so both are idempotent -- one row per account,
always reflecting the latest known password.
"""

from __future__ import annotations


def _account(**overrides):
    base = {
        "site": "Zillow Group",
        "domain": "zillow.wd5.myworkdayjobs.com",
        "email": "philipthegingermclaughlin@gmail.com",
        "password": "Aryeo#2026Bdr",
        "login_method": "email",
    }
    base.update(overrides)
    return base


def test_repeated_identical_account_does_not_duplicate(tmp_db):
    from applypilot.database import store_account

    conn = tmp_db()
    for _ in range(3):
        store_account(conn, _account(), job_url="https://example.com/job/1")

    rows = conn.execute(
        "SELECT * FROM accounts WHERE domain = ? AND email = ?",
        ("zillow.wd5.myworkdayjobs.com", "philipthegingermclaughlin@gmail.com"),
    ).fetchall()
    assert len(rows) == 1, f"expected exactly 1 row after 3 identical stores, got {len(rows)}"


def test_updated_password_overwrites_existing_row_not_appends(tmp_db):
    from applypilot.database import store_account

    conn = tmp_db()
    store_account(conn, _account(password="OldPass#1"), job_url="https://example.com/job/1")
    store_account(conn, _account(password="NewPass#2"), job_url="https://example.com/job/2")

    rows = conn.execute(
        "SELECT password, job_url FROM accounts WHERE domain = ? AND email = ?",
        ("zillow.wd5.myworkdayjobs.com", "philipthegingermclaughlin@gmail.com"),
    ).fetchall()
    assert len(rows) == 1, f"expected exactly 1 row, got {len(rows)}"
    assert rows[0]["password"] == "NewPass#2"
    assert rows[0]["job_url"] == "https://example.com/job/2"


def test_different_domains_or_emails_get_separate_rows(tmp_db):
    from applypilot.database import store_account

    conn = tmp_db()
    store_account(conn, _account(domain="zillow.wd5.myworkdayjobs.com"))
    store_account(conn, _account(domain="motorolasolutions.wd5.myworkdayjobs.com"))
    store_account(conn, _account(email="someoneelse@gmail.com"))

    rows = conn.execute("SELECT domain, email FROM accounts").fetchall()
    assert len(rows) == 3, f"expected 3 distinct rows for 3 distinct (domain, email) pairs, got {len(rows)}"
