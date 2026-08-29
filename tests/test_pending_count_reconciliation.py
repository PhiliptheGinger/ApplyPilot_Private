"""Regression tests for the 2026-08-28 `_PENDING_SQL` / `get_jobs_by_stage`
reconciliation fix.

Root cause (traced live, not assumed): `pipeline._count_pending` used its own,
independently-maintained `_PENDING_SQL` dict for "enrich"/"score"/"tailor"/
"cover" that never gained the `state`/`eligibility` gates `get_jobs_by_stage`'s
"pending_score"/"pending_tailor"/"pending_cover" conditions picked up on
2026-08-25 (decision #65). A job terminally `archived` for ANY reason
(title-reject, revalidate-seniority, non_us_only, ...) kept counting as
"pending" forever in `_count_pending`, even though `get_jobs_by_stage` (what
the real stage runners actually use to select work) correctly excludes it --
`archived` is the state machine's one true terminal state
(`VALID_TRANSITIONS["archived"] == frozenset()`).

Live-measured impact before this fix: score 14,558 vs 11,770 actually
selectable, tailor 4 vs 0, cover 40 vs 0 -- 100% of every sampled gap row was
`state='archived'`. This isn't cosmetic: `pipeline._run_stage_streaming`'s
`--stream` poll loop only exits a stage once `_count_pending(stage) == 0`
*and* upstream is done; a permanent `archived` phantom count meant that once
the real backlog drained, `pending` could never reach 0 again, so the loop
would poll (and back off) forever instead of terminating.

Fix: `database.count_jobs_by_stage(conn, stage, ...)` shares the exact same
`_STAGE_CONDITIONS` dict / `_build_stage_where()` helper `get_jobs_by_stage`
uses, via a cheap `SELECT COUNT(*)` (deliberately never `SELECT *` through
`get_jobs_by_stage`'s `ROW_NUMBER()` window-function query, which would be
wasteful for a poll-loop count against a multi-thousand-row backlog).
`pipeline._count_pending` now delegates "enrich"/"score"/"tailor"/"cover" to
it via `_CANONICAL_PENDING_STAGE`. "pdf" has no canonical `get_jobs_by_stage`
equivalent (its selection criterion, `tailored_resume_path LIKE '%.txt'`,
isn't a state-machine concept) and deliberately stays on the original,
unchanged `_PENDING_SQL` query.
"""

from __future__ import annotations

import pytest

# (pipeline stage name, canonical get_jobs_by_stage/count_jobs_by_stage name)
_CANONICAL_STAGES = [
    ("score", "pending_score"),
    ("tailor", "pending_tailor"),
    ("cover", "pending_cover"),
    ("enrich", "pending_detail"),
]


def _seed_baseline_selectable(conn, seed_job, stage: str):
    """Seed exactly one genuinely selectable job for `stage` (mirrors the
    positive-case fixtures in test_pending_{score,tailor}_state_selection.py)."""
    if stage == "score":
        return seed_job(conn, fit_score=None, full_description="x", state="enriched")
    if stage == "tailor":
        return seed_job(conn, fit_score=9, full_description="x", tailored_resume_path=None, state="scored")
    if stage == "cover":
        return seed_job(
            conn,
            fit_score=9,
            full_description="x",
            tailored_resume_path="/tmp/resume.docx",
            cover_letter_path=None,
            state="tailored",
        )
    if stage == "enrich":
        return seed_job(conn, detail_scraped_at=None)
    raise AssertionError(stage)


def _seed_archived_phantom(conn, seed_job, stage: str):
    """Seed a job that the OLD `_PENDING_SQL` predicate would have counted
    (same fit_score/description/path/attempts shape as the baseline) but
    that is terminally `archived` -- the exact live bug."""
    if stage == "score":
        return seed_job(conn, fit_score=None, full_description="x", state="archived")
    if stage == "tailor":
        return seed_job(conn, fit_score=9, full_description="x", tailored_resume_path=None, state="archived")
    if stage == "cover":
        return seed_job(
            conn,
            fit_score=9,
            full_description="x",
            tailored_resume_path="/tmp/resume.docx",
            cover_letter_path=None,
            state="archived",
        )
    if stage == "enrich":
        return seed_job(conn, detail_scraped_at=None, state="archived")
    raise AssertionError(stage)


# ── count/fetch parity ──────────────────────────────────────────────────────


@pytest.mark.parametrize("stage,canonical", _CANONICAL_STAGES)
def test_count_matches_fetch_for_baseline_selectable_job(tmp_db, seed_job, stage, canonical):
    """count_jobs_by_stage, get_jobs_by_stage, and _count_pending must all
    agree on the same genuinely-selectable job -- the parity this whole fix
    exists to guarantee."""
    from applypilot.database import count_jobs_by_stage, get_jobs_by_stage
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    _seed_baseline_selectable(conn, seed_job, stage)

    n_count = count_jobs_by_stage(conn, canonical)
    n_fetch = len(get_jobs_by_stage(conn, stage=canonical, limit=0))
    n_pending = _count_pending(stage)

    assert n_count == n_fetch == n_pending == 1


@pytest.mark.parametrize("stage,canonical", [s for s in _CANONICAL_STAGES if s[0] != "enrich"])
def test_count_matches_fetch_for_archived_phantom_row(tmp_db, seed_job, stage, canonical):
    """The exact bug: a job shaped exactly like the baseline positive case
    but terminally archived must be excluded by count_jobs_by_stage,
    get_jobs_by_stage, AND _count_pending alike -- not just the fetch path
    (which decision #65 already fixed) but the count path this fix adds.
    "enrich"/"pending_detail" is excluded here: unlike score/tailor/cover, it
    has no `state` gate at all (pre-existing, unrelated to this fix -- see
    test_pending_detail_has_no_state_gate_parity_is_still_preserved below)."""
    from applypilot.database import count_jobs_by_stage, get_jobs_by_stage
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    _seed_archived_phantom(conn, seed_job, stage)

    n_count = count_jobs_by_stage(conn, canonical)
    n_fetch = len(get_jobs_by_stage(conn, stage=canonical, limit=0))
    n_pending = _count_pending(stage)

    assert n_count == n_fetch == n_pending == 0


def test_pending_detail_has_no_state_gate_parity_is_still_preserved(tmp_db, seed_job):
    """"pending_detail" (unlike pending_score/tailor/cover) never gained a
    `state` check -- there was no live divergence for "enrich" to fix (see
    investigation report), so this fix intentionally leaves that predicate
    itself unchanged. What it DOES guarantee is that count_jobs_by_stage,
    get_jobs_by_stage, and _count_pending all agree with EACH OTHER on this
    stage too, since they now share the exact same WHERE clause -- even
    though, pre-existing and out of scope, an archived job with
    detail_scraped_at still NULL is (unchanged) still counted."""
    from applypilot.database import count_jobs_by_stage, get_jobs_by_stage
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    _seed_archived_phantom(conn, seed_job, "enrich")

    n_count = count_jobs_by_stage(conn, "pending_detail")
    n_fetch = len(get_jobs_by_stage(conn, stage="pending_detail", limit=0))
    n_pending = _count_pending("enrich")

    assert n_count == n_fetch == n_pending == 1


# ── retry-eligible failed states within budget ──────────────────────────────


def test_count_pending_tailor_includes_retry_eligible_tailor_failed(tmp_db, seed_job):
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    seed_job(conn, fit_score=9, full_description="x", tailored_resume_path=None, state="tailor_failed", tailor_attempts=2)

    assert _count_pending("tailor") == 1


def test_count_pending_cover_includes_retry_eligible_cover_failed(tmp_db, seed_job):
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    seed_job(
        conn,
        fit_score=9,
        full_description="x",
        tailored_resume_path="/tmp/resume.docx",
        cover_letter_path=None,
        state="cover_failed",
        cover_attempts=2,
    )

    assert _count_pending("cover") == 1


def test_count_pending_score_includes_retry_eligible_score_failed(tmp_db, seed_job):
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    seed_job(
        conn,
        fit_score=None,
        full_description="x",
        state="score_failed",
        score_error="LLM error: boom",
        score_attempts=2,
    )

    assert _count_pending("score") == 1


# ── exhausted attempts ───────────────────────────────────────────────────────


def test_count_pending_tailor_excludes_exhausted_tailor_failed(tmp_db, seed_job):
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    seed_job(conn, fit_score=9, full_description="x", tailored_resume_path=None, state="tailor_failed", tailor_attempts=5)

    assert _count_pending("tailor") == 0


def test_count_pending_cover_excludes_exhausted_cover_failed(tmp_db, seed_job):
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    seed_job(
        conn,
        fit_score=9,
        full_description="x",
        tailored_resume_path="/tmp/resume.docx",
        cover_letter_path=None,
        state="cover_failed",
        cover_attempts=5,
    )

    assert _count_pending("cover") == 0


def test_count_pending_score_excludes_exhausted_score_failed(tmp_db, seed_job):
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    seed_job(
        conn,
        fit_score=None,
        full_description="x",
        state="score_failed",
        score_error="LLM error: boom",
        score_attempts=5,
    )

    assert _count_pending("score") == 0


# ── eligibility gate ─────────────────────────────────────────────────────────


def test_count_pending_tailor_excludes_non_us_only(tmp_db, seed_job):
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    seed_job(
        conn,
        fit_score=9,
        full_description="x",
        tailored_resume_path=None,
        state="scored",
        eligibility="non_us_only",
    )

    assert _count_pending("tailor") == 0


def test_count_pending_cover_excludes_non_us_only(tmp_db, seed_job):
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    seed_job(
        conn,
        fit_score=9,
        full_description="x",
        tailored_resume_path="/tmp/resume.docx",
        cover_letter_path=None,
        state="tailored",
        eligibility="non_us_only",
    )

    assert _count_pending("cover") == 0


# ── streaming termination ───────────────────────────────────────────────────


@pytest.mark.parametrize("stage", ["score", "tailor", "cover"])
def test_count_pending_zero_when_only_archived_phantoms_remain(tmp_db, seed_job, stage):
    """This is what actually fixes `_run_stage_streaming`'s non-termination:
    that function's loop treats `_count_pending(stage) == 0` (with upstream
    done) as "this stage is finished". Before this fix, a permanently
    `archived` phantom row kept `_count_pending` above zero forever even
    after all real work drained, so the loop could never reach its own exit
    condition and would poll (and back off) indefinitely instead."""
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    _seed_archived_phantom(conn, seed_job, stage)
    # A second, differently-shaped archived row for good measure -- the fix
    # must generalize, not just handle the one exact fixture shape.
    _seed_archived_phantom(conn, seed_job, stage)

    assert _count_pending(stage) == 0


# ── pdf fallback unchanged ───────────────────────────────────────────────────


def test_pdf_fallback_still_uses_the_original_standalone_query(tmp_db, seed_job):
    """"pdf" has no canonical get_jobs_by_stage equivalent (its criterion,
    a `.txt` path suffix, isn't a state-machine concept) and is a deliberate
    exception left on the original _PENDING_SQL query -- unaffected by this
    fix, including its permissiveness (no state/eligibility gate at all)."""
    from applypilot.pipeline import _count_pending

    conn = tmp_db()
    # Even an archived job with a .txt tailored resume is still counted --
    # proving "pdf" genuinely bypassed the reconciliation, not incidentally.
    seed_job(conn, tailored_resume_path="/tmp/resume.txt", state="archived")
    seed_job(conn, tailored_resume_path="/tmp/resume.docx", state="scored")

    assert _count_pending("pdf") == 1


def test_pdf_stage_has_no_canonical_mapping(tmp_db, seed_job):
    from applypilot.pipeline import _CANONICAL_PENDING_STAGE

    assert "pdf" not in _CANONICAL_PENDING_STAGE


# ── performance guard: count path must never do SELECT * ──────────────────


@pytest.mark.parametrize("stage,canonical", _CANONICAL_STAGES)
def test_count_jobs_by_stage_issues_select_count_not_select_star(tmp_db, seed_job, stage, canonical):
    """Guards against a future "helpful" refactor silently reintroducing the
    expensive path (calling get_jobs_by_stage with limit=0 just to count),
    which would re-add a SELECT * + ROW_NUMBER() window-function scan over
    the full matched set to every streaming poll / scheduler cycle."""
    from applypilot.database import count_jobs_by_stage

    conn = tmp_db()
    _seed_baseline_selectable(conn, seed_job, stage)

    # sqlite3.Connection.execute is a read-only C attribute -- can't be
    # monkeypatched directly on the instance. set_trace_callback is the
    # sqlite3 module's own hook for observing every SQL statement a
    # connection runs.
    executed: list[str] = []
    conn.set_trace_callback(executed.append)
    try:
        count_jobs_by_stage(conn, canonical)
    finally:
        conn.set_trace_callback(None)

    select_statements = [sql for sql in executed if sql.strip().upper().startswith("SELECT")]
    assert len(select_statements) == 1
    sql = select_statements[0].strip().upper()
    assert sql.startswith("SELECT COUNT(*)")
    assert "ROW_NUMBER" not in sql


# ── get_stats()['unscored'] reconciliation ──────────────────────────────────
#
# 2026-08-28: get_stats()'s "unscored" field was a THIRD independently-
# maintained "pending score" predicate (`full_description IS NOT NULL AND
# fit_score IS NULL`, no `state` filter) -- the same phantom-archived-row
# bug this whole module's fix already closed for pipeline._count_pending,
# just a separate, unreconciled copy. It's user-facing: `applypilot status`
# displays it as "Pending scoring". Fixed by delegating to
# count_jobs_by_stage(conn, "pending_score", max_age_days=0) -- the
# max_age_days=0 preserves the field's pre-existing no-age-filtering
# semantics exactly.


def test_get_stats_unscored_matches_canonical_pending_score(tmp_db, seed_job):
    """Parity test across a realistic mixed pool: get_stats()['unscored']
    must equal count_jobs_by_stage(conn, "pending_score") exactly, on a
    fixture that includes the exact historical phantom-row shape (an
    archived job with fit_score still NULL) alongside genuinely pending
    rows -- reproducing the live 14,558-vs-11,770 divergence at test scale."""
    from applypilot.database import count_jobs_by_stage, get_stats

    conn = tmp_db()
    # Archived phantom -- the exact bug. Must NOT be counted.
    seed_job(conn, url_suffix="stats-archived", fit_score=None, full_description="x", state="archived")
    # Genuinely pending (enriched, never scored). Must be counted.
    seed_job(conn, url_suffix="stats-enriched", fit_score=None, full_description="x", state="enriched")
    # score_failed within retry budget. Must be counted.
    seed_job(
        conn,
        url_suffix="stats-retry",
        fit_score=None,
        full_description="x",
        state="score_failed",
        score_error="LLM error: boom",
        score_attempts=1,
    )
    # score_failed exhausted. Must NOT be counted.
    seed_job(
        conn,
        url_suffix="stats-exhausted",
        fit_score=None,
        full_description="x",
        state="score_failed",
        score_error="LLM error: boom",
        score_attempts=5,
    )
    # Already scored. Must NOT be counted.
    seed_job(conn, url_suffix="stats-scored", fit_score=9, full_description="x", state="scored")

    stats = get_stats(conn)
    canonical = count_jobs_by_stage(conn, "pending_score", max_age_days=0)

    assert stats["unscored"] == canonical == 2, (
        f"get_stats()['unscored']={stats['unscored']} vs canonical={canonical} -- "
        "these must never diverge again; expected exactly the 2 genuinely-pending rows"
    )
