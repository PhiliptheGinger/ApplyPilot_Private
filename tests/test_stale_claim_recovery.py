"""Stale in-flight claim recovery for the 'tailoring' / 'cover_writing' states.

These are claim states: run_tailoring()/run_cover_letters() transition a
job into them right before starting a (possibly multi-minute) LLM call, and
only the matching completion path (_mark_tailor_result / _mark_cover_result)
ever moves the job back out. If the worker process dies mid-call, the job
is stranded forever -- neither pending_tailor nor pending_cover selects
'tailoring'/'cover_writing' rows.

database.recover_stale_claims() closes this using the existing
job_state_transitions audit history rather than a new schema column: the
invariant documented on transition_state() (jobs.state always equals the
latest transition row's to_state for that job) means the claim timestamp is
simply the `at` of the most recent transition into the job's current state.

These tests exercise recover_stale_claims() directly (deterministic, no
LLM calls, no sleeping -- transition timestamps are backdated explicitly),
plus two lightweight integration checks confirming run_tailoring() and
run_cover_letters() actually invoke it before selecting new candidates.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from applypilot.database import current_state, recover_stale_claims


def _at(minutes_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_transition(conn, url, from_state, to_state, minutes_ago, reason="test"):
    conn.execute(
        "INSERT INTO job_state_transitions (job_url, from_state, to_state, at, reason, metadata) "
        "VALUES (?, ?, ?, ?, ?, NULL)",
        (url, from_state, to_state, _at(minutes_ago), reason),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Core recovery behavior
# ---------------------------------------------------------------------------


def test_stale_tailoring_recovers_to_tailor_failed(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="stale-tailoring", state="tailoring")
    url = row["url"]
    _insert_transition(conn, url, "scored", "tailoring", minutes_ago=40)

    recovered = recover_stale_claims(conn, "tailoring", "tailor_failed", "tailor_attempts")

    assert recovered == [url]
    assert current_state(conn, url) == "tailor_failed"


def test_fresh_tailoring_remains_untouched(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="fresh-tailoring", state="tailoring")
    url = row["url"]
    _insert_transition(conn, url, "scored", "tailoring", minutes_ago=5)

    recovered = recover_stale_claims(conn, "tailoring", "tailor_failed", "tailor_attempts")

    assert recovered == []
    assert current_state(conn, url) == "tailoring"


def test_stale_cover_writing_recovers_to_cover_failed(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="stale-cover", state="cover_writing")
    url = row["url"]
    _insert_transition(conn, url, "tailored", "cover_writing", minutes_ago=40)

    recovered = recover_stale_claims(conn, "cover_writing", "cover_failed", "cover_attempts")

    assert recovered == [url]
    assert current_state(conn, url) == "cover_failed"


def test_fresh_cover_writing_remains_untouched(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="fresh-cover", state="cover_writing")
    url = row["url"]
    _insert_transition(conn, url, "tailored", "cover_writing", minutes_ago=5)

    recovered = recover_stale_claims(conn, "cover_writing", "cover_failed", "cover_attempts")

    assert recovered == []
    assert current_state(conn, url) == "cover_writing"


# ---------------------------------------------------------------------------
# Race safety
# ---------------------------------------------------------------------------


def test_job_already_moved_out_of_claim_state_is_not_recovered(tmp_db, seed_job):
    """A job that legitimately completed (state now 'tailored') must not be
    touched, even though its history contains an old 'tailoring' entry from
    the claim that preceded its real completion."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="moved-on", state="tailored", tailored_resume_path="/tmp/r.docx")
    url = row["url"]
    _insert_transition(conn, url, "scored", "tailoring", minutes_ago=40)
    _insert_transition(conn, url, "tailoring", "tailored", minutes_ago=1)

    recovered = recover_stale_claims(conn, "tailoring", "tailor_failed", "tailor_attempts")

    assert recovered == []
    assert current_state(conn, url) == "tailored"


def test_uses_latest_claim_not_a_stale_historical_one(tmp_db, seed_job):
    """A job retried into 'tailoring' a second time (tailor_failed ->
    tailoring, decision #65's bounded-retry design) must be judged by its
    latest (fresh) claim timestamp, not the old one from its first,
    already-failed attempt."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="retried", state="tailoring")
    url = row["url"]
    _insert_transition(conn, url, "scored", "tailoring", minutes_ago=90)
    _insert_transition(conn, url, "tailoring", "tailor_failed", minutes_ago=60)
    _insert_transition(conn, url, "tailor_failed", "tailoring", minutes_ago=2)

    recovered = recover_stale_claims(conn, "tailoring", "tailor_failed", "tailor_attempts")

    assert recovered == []
    assert current_state(conn, url) == "tailoring"


def test_job_with_no_transition_history_is_not_recovered(tmp_db, seed_job):
    """Defensive: seed_job() inserts directly into `jobs`, bypassing
    transition_state(), so a test (or a real pre-migration row) with no
    job_state_transitions rows at all has an unknown claim timestamp. The
    correlated subquery returns NULL in that case; recover_stale_claims
    must treat "unknown" as "not stale" rather than always-recoverable."""
    conn = tmp_db()
    row = seed_job(conn, url_suffix="no-history", state="tailoring")
    url = row["url"]

    recovered = recover_stale_claims(conn, "tailoring", "tailor_failed", "tailor_attempts")

    assert recovered == []
    assert current_state(conn, url) == "tailoring"


# ---------------------------------------------------------------------------
# Attempt/retry semantics + audit trail
# ---------------------------------------------------------------------------


def test_recovery_increments_attempts_counter(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="attempts", state="tailoring", tailor_attempts=2)
    url = row["url"]
    _insert_transition(conn, url, "tailor_failed", "tailoring", minutes_ago=40)

    recover_stale_claims(conn, "tailoring", "tailor_failed", "tailor_attempts")

    attempts = conn.execute("SELECT tailor_attempts FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert attempts == 3


def test_untouched_job_attempts_counter_unchanged(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="attempts-fresh", state="tailoring", tailor_attempts=2)
    url = row["url"]
    _insert_transition(conn, url, "tailor_failed", "tailoring", minutes_ago=5)

    recover_stale_claims(conn, "tailoring", "tailor_failed", "tailor_attempts")

    attempts = conn.execute("SELECT tailor_attempts FROM jobs WHERE url = ?", (url,)).fetchone()[0]
    assert attempts == 2


def test_recovery_writes_a_consistent_audit_row(tmp_db, seed_job):
    conn = tmp_db()
    row = seed_job(conn, url_suffix="audit", state="cover_writing")
    url = row["url"]
    _insert_transition(conn, url, "tailored", "cover_writing", minutes_ago=45)

    recover_stale_claims(conn, "cover_writing", "cover_failed", "cover_attempts", reason="stale claim test")

    transitions = conn.execute(
        "SELECT from_state, to_state, reason FROM job_state_transitions WHERE job_url = ? ORDER BY id DESC",
        (url,),
    ).fetchall()
    assert transitions[0]["from_state"] == "cover_writing"
    assert transitions[0]["to_state"] == "cover_failed"
    assert transitions[0]["reason"] == "stale claim test"
    assert current_state(conn, url) == transitions[0]["to_state"]


# ---------------------------------------------------------------------------
# Wired into the pipeline entry points
# ---------------------------------------------------------------------------


def test_run_tailoring_recovers_stale_claim_before_selecting(tmp_db, seed_job):
    from applypilot.scoring.tailor import run_tailoring

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="run-tailoring-stale",
        state="tailoring",
        fit_score=1,  # below min_score so it is NOT re-selected by pending_tailor this same run
        tailored_resume_path=None,
    )
    url = row["url"]
    _insert_transition(conn, url, "scored", "tailoring", minutes_ago=40)

    result = run_tailoring(min_score=8, max_age_days=0)

    assert current_state(conn, url) == "tailor_failed"
    assert result["approved"] == 0


def test_run_cover_letters_recovers_stale_claim_before_selecting(tmp_db, seed_job):
    from applypilot.scoring.cover_letter import run_cover_letters

    conn = tmp_db()
    row = seed_job(
        conn,
        url_suffix="run-cover-stale",
        state="cover_writing",
        fit_score=1,  # below min_score so it is NOT re-selected by pending_cover this same run
        tailored_resume_path="/tmp/resume.docx",
        cover_letter_path=None,
    )
    url = row["url"]
    _insert_transition(conn, url, "tailored", "cover_writing", minutes_ago=40)

    run_cover_letters(min_score=8, max_age_days=0)

    assert current_state(conn, url) == "cover_failed"
