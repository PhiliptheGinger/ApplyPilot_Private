"""Tests for sequential-pipeline batch identity: a `tailor` stage that
follows `score` in the same run must operate on exactly the batch `score`
selected, not independently re-query and pull in unrelated already-eligible
jobs sitting in the DB from earlier runs -- and the same for `cover`
following `tailor`.

Covers:
- get_jobs_by_stage(urls=...): the low-level restriction primitive.
- run_tailoring(job_ids=...): the edge case where only some of the carried
  batch meets the tailoring threshold.
- run_tailoring() without job_ids: standalone behavior is unchanged.
- run_scoring()'s job_urls in its return dict (what pipeline.py carries
  forward).
- run_cover_letters(job_ids=...): the real-world regression this file's
  bottom section reproduces -- a `cover` stage with no batch of its own
  independently re-querying pending_cover and picking up unrelated
  already-eligible jobs from earlier runs (a recurring, many-times-reposted
  LinkedIn listing), generating 3 cover letters when tailor had only
  produced 1 job this run.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def seed_job(seed_job):
    """Override the conftest fixture: this file's tests are almost all
    about `pending_tailor`, which (as of the 2026-08-25 state-selection
    fix) only matches state IN ('scored', 'tailor_failed') -- default to
    'scored' so existing call sites that never cared about state (the bug
    this file predates) keep exercising the same selection logic they did
    before. setdefault (not a blanket override) preserves the explicit
    state="tailored" already passed by this file's own pending_cover
    tests further down."""

    def _seed(conn, **overrides):
        overrides.setdefault("state", "scored")
        return seed_job(conn, **overrides)

    return _seed


# ---------------------------------------------------------------------------
# get_jobs_by_stage(urls=...)
# ---------------------------------------------------------------------------


def test_get_jobs_by_stage_urls_restricts_to_exact_batch(tmp_db, seed_job):
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    a = seed_job(conn, url_suffix="a", fit_score=9, tailored_resume_path=None)
    b = seed_job(conn, url_suffix="b", fit_score=9, tailored_resume_path=None)
    seed_job(conn, url_suffix="unrelated-prior-batch", fit_score=9, tailored_resume_path=None)

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8, urls=[a["url"], b["url"]])
    urls = {r["url"] for r in rows}
    assert urls == {a["url"], b["url"]}


def test_get_jobs_by_stage_empty_urls_list_restricts_to_nothing(tmp_db, seed_job):
    """None (no restriction) and [] (restrict to nothing) must behave
    differently -- an empty carried batch must not fall back to
    independent selection."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    seed_job(conn, fit_score=9, tailored_resume_path=None)

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8, urls=[])
    assert rows == []

    rows_unrestricted = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8, urls=None)
    assert len(rows_unrestricted) == 1


def test_get_jobs_by_stage_urls_still_applies_eligibility(tmp_db, seed_job):
    """The edge case from the bug report: a carried batch of 3 where only
    1 meets min_score must yield only that 1, not 3 (backfilled) or 0."""
    from applypilot.database import get_jobs_by_stage

    conn = tmp_db()
    high = seed_job(conn, url_suffix="high", fit_score=9, tailored_resume_path=None)
    low1 = seed_job(conn, url_suffix="low1", fit_score=2, tailored_resume_path=None)
    low2 = seed_job(conn, url_suffix="low2", fit_score=2, tailored_resume_path=None)
    batch = [high["url"], low1["url"], low2["url"]]

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8, urls=batch)
    assert [r["url"] for r in rows] == [high["url"]]


# ---------------------------------------------------------------------------
# run_tailoring(job_ids=...) integration -- _tailor_one_job is stubbed out
# (no real LLM/docx work) so these exercise batch selection, not tailoring
# quality.
# ---------------------------------------------------------------------------


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


def test_run_tailoring_with_job_ids_restricts_to_batch(tmp_db, seed_job, monkeypatch, tmp_path):
    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    in_batch = seed_job(conn, url_suffix="in-batch", fit_score=9, tailored_resume_path=None)
    seed_job(conn, url_suffix="unrelated-prior-batch", fit_score=9, tailored_resume_path=None)

    calls: list[str] = []

    def tracking_fake(job, resume_text, profile, doc_format="docx"):
        calls.append(job["url"])
        return _fake_tailor_one_job(job, resume_text, profile, doc_format)

    monkeypatch.setattr(tailor_mod, "_tailor_one_job", tracking_fake)
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    result = tailor_mod.run_tailoring(min_score=8, limit=3, job_ids=[in_batch["url"]])

    assert calls == [in_batch["url"]]
    assert result["approved"] == 1


def test_run_tailoring_job_ids_edge_case_only_eligible_from_batch(
    tmp_db,
    seed_job,
    monkeypatch,
    tmp_path,
):
    """Score selects 3 jobs; only 1 meets min_score. Tailor must process
    only that 1 -- not backfill the other 2 slots with the unrelated
    previously-high-scoring job sitting in the DB from an earlier run."""
    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    eligible = seed_job(conn, url_suffix="eligible", fit_score=9, tailored_resume_path=None)
    low1 = seed_job(conn, url_suffix="low1", fit_score=2, tailored_resume_path=None)
    low2 = seed_job(conn, url_suffix="low2", fit_score=2, tailored_resume_path=None)
    unrelated = seed_job(conn, url_suffix="unrelated-prior-batch", fit_score=9, tailored_resume_path=None)

    calls: list[str] = []

    def tracking_fake(job, resume_text, profile, doc_format="docx"):
        calls.append(job["url"])
        return _fake_tailor_one_job(job, resume_text, profile, doc_format)

    monkeypatch.setattr(tailor_mod, "_tailor_one_job", tracking_fake)
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    batch = [eligible["url"], low1["url"], low2["url"]]
    result = tailor_mod.run_tailoring(min_score=8, limit=3, job_ids=batch)

    assert calls == [eligible["url"]]
    assert unrelated["url"] not in calls
    assert result["approved"] == 1


def test_run_tailoring_empty_job_ids_tailors_nothing(tmp_db, seed_job, monkeypatch, tmp_path):
    """If score selected zero jobs this run, tailor must also select zero
    -- not fall back to independent selection and pick up unrelated
    already-eligible jobs."""
    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    seed_job(conn, fit_score=9, tailored_resume_path=None)  # would be independently eligible

    calls: list[str] = []
    monkeypatch.setattr(tailor_mod, "_tailor_one_job", lambda *a, **k: calls.append(a[0]["url"]))
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    result = tailor_mod.run_tailoring(min_score=8, limit=3, job_ids=[])

    assert calls == []
    assert result == {"approved": 0, "failed": 0, "errors": 0, "elapsed": 0.0, "job_urls": []}


def test_run_tailoring_without_job_ids_uses_independent_selection(
    tmp_db,
    seed_job,
    monkeypatch,
    tmp_path,
):
    """Standalone `applypilot run tailor` (no preceding `score` stage in
    this run) must keep selecting independently -- unaffected by the
    job_ids parameter's default."""
    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    job = seed_job(conn, fit_score=9, tailored_resume_path=None)

    calls: list[str] = []

    def tracking_fake(j, resume_text, profile, doc_format="docx"):
        calls.append(j["url"])
        return _fake_tailor_one_job(j, resume_text, profile, doc_format)

    monkeypatch.setattr(tailor_mod, "_tailor_one_job", tracking_fake)
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    result = tailor_mod.run_tailoring(min_score=8, limit=3)  # job_ids defaults to None

    assert calls == [job["url"]]
    assert result["approved"] == 1


def test_run_tailoring_logs_carried_batch_size(tmp_db, seed_job, monkeypatch, tmp_path, caplog):
    import logging

    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    eligible = seed_job(conn, url_suffix="eligible", fit_score=9, tailored_resume_path=None)
    low = seed_job(conn, url_suffix="low", fit_score=2, tailored_resume_path=None)

    monkeypatch.setattr(tailor_mod, "_tailor_one_job", _fake_tailor_one_job)
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    with caplog.at_level(logging.INFO, logger="applypilot.scoring.tailor"):
        tailor_mod.run_tailoring(min_score=8, job_ids=[eligible["url"], low["url"]])

    assert any("carried batch of 2" in r.message for r in caplog.records)


def test_run_tailoring_logs_local_first_enabled_banner(
    tmp_db,
    seed_job,
    monkeypatch,
    tmp_path,
    caplog,
):
    import logging

    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    seed_job(conn, fit_score=9, tailored_resume_path=None)

    monkeypatch.setenv("APPLYPILOT_LOCAL_PLAN", "1")
    monkeypatch.setattr(tailor_mod, "_tailor_one_job", _fake_tailor_one_job)
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    with caplog.at_level(logging.INFO, logger="applypilot.scoring.tailor"):
        tailor_mod.run_tailoring(min_score=8)

    banner_lines = [r.message for r in caplog.records if "local-first: enabled" in r.message]
    assert len(banner_lines) == 1


# ---------------------------------------------------------------------------
# run_scoring()'s job_urls -- what pipeline.py carries forward
# ---------------------------------------------------------------------------


def test_run_scoring_returns_job_urls_for_the_batch_it_selected(
    tmp_db,
    seed_job,
    monkeypatch,
):
    import applypilot.scoring.scorer as scorer_mod

    conn = tmp_db()
    # 2026-08-25: pending_score now requires state IN ('enriched',
    # 'score_failed') -- this file's local seed_job override defaults to
    # state="scored" (for the pending_tailor tests above), which is wrong
    # for a pending_score selection, so override it explicitly here.
    a = seed_job(conn, url_suffix="a", fit_score=None, full_description="x", state="enriched")
    b = seed_job(conn, url_suffix="b", fit_score=None, full_description="x", state="enriched")

    def fake_score_job(resume_text, job, profile=None):
        return {"score": 9, "keywords": "", "reasoning": ""}

    monkeypatch.setattr(scorer_mod, "score_job", fake_score_job)
    monkeypatch.setattr(scorer_mod, "load_profile", dict)
    monkeypatch.setattr(scorer_mod, "render_profile_reference", lambda profile: "Resume text.")

    result = scorer_mod.run_scoring(limit=2)

    assert set(result["job_urls"]) == {a["url"], b["url"]}
    assert result["scored"] == 2


# ---------------------------------------------------------------------------
# pipeline._run_sequential: carries score's batch into the following tailor
# stage, and leaves standalone `tailor` runs unaffected.
# ---------------------------------------------------------------------------


def test_run_sequential_carries_score_batch_into_tailor():
    from applypilot import pipeline

    calls: dict[str, dict] = {}

    def fake_score(**kwargs):
        calls["score"] = kwargs
        return {"status": "ok", "job_ids": ["https://example.com/a", "https://example.com/b"]}

    def fake_tailor(**kwargs):
        calls["tailor"] = kwargs
        return {"status": "ok"}

    with patch.object(pipeline, "_STAGE_RUNNERS", {"score": fake_score, "tailor": fake_tailor}):
        pipeline._run_sequential(["score", "tailor"], min_score=8, limit=3)

    assert calls["tailor"]["job_ids"] == ["https://example.com/a", "https://example.com/b"]


def test_run_sequential_standalone_tailor_gets_no_job_ids():
    """A `tailor`-only run (no `score` stage in this invocation) must not
    receive a job_ids kwarg at all -- preserving independent selection."""
    from applypilot import pipeline

    calls: dict[str, dict] = {}

    def fake_tailor(**kwargs):
        calls["tailor"] = kwargs
        return {"status": "ok"}

    with patch.object(pipeline, "_STAGE_RUNNERS", {"tailor": fake_tailor}):
        pipeline._run_sequential(["tailor"], min_score=8, limit=3)

    assert "job_ids" not in calls["tailor"]


def test_run_sequential_score_error_does_not_carry_stale_job_ids():
    """If the score stage errors out, tailor must fall back to independent
    selection rather than being restricted to a nonexistent/partial batch."""
    from applypilot import pipeline

    calls: dict[str, dict] = {}

    def failing_score(**kwargs):
        return {"status": "error: boom"}

    def fake_tailor(**kwargs):
        calls["tailor"] = kwargs
        return {"status": "ok"}

    with patch.object(pipeline, "_STAGE_RUNNERS", {"score": failing_score, "tailor": fake_tailor}):
        pipeline._run_sequential(["score", "tailor"], min_score=8, limit=3)

    assert "job_ids" not in calls["tailor"]


# ---------------------------------------------------------------------------
# run_tailoring()'s job_urls -- what pipeline.py carries forward to `cover`
# ---------------------------------------------------------------------------


def test_run_tailoring_returns_job_urls_for_the_eligible_after_cap_batch(
    tmp_db,
    seed_job,
    monkeypatch,
    tmp_path,
):
    import applypilot.scoring.tailor as tailor_mod

    conn = tmp_db()
    a = seed_job(conn, url_suffix="a", fit_score=9, tailored_resume_path=None)
    seed_job(conn, url_suffix="low", fit_score=2, tailored_resume_path=None)  # excluded

    monkeypatch.setattr(tailor_mod, "_tailor_one_job", _fake_tailor_one_job)
    monkeypatch.setattr(tailor_mod, "load_profile", dict)
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    result = tailor_mod.run_tailoring(min_score=8, limit=3)

    assert result["job_urls"] == [a["url"]]


# ---------------------------------------------------------------------------
# run_cover_letters(job_ids=...) integration -- the real-world regression.
#
# Reported scenario: a 3-job score->tailor->cover run tailored exactly 1
# job, but the cover stage (which had no batch of its own) independently
# re-queried pending_cover and picked up 2 UNRELATED already-eligible jobs
# left over from earlier runs -- a recurring LinkedIn listing reposted
# under different URLs, each independently scored/tailored on a different
# day -- generating 3 cover letters instead of 1. _cover_one_job is
# stubbed out (no real LLM/docx work) so these exercise batch selection,
# not cover-letter quality.
# ---------------------------------------------------------------------------


def _fake_cover_one_job(job, resume_text, profile, doc_format="docx"):
    return {
        "url": job["url"],
        "title": job["title"],
        "site": job.get("site", ""),
        "path": "/tmp/fake_cl.txt",
        "pdf_path": None,
        "error": None,
    }


def test_run_cover_letters_with_job_ids_restricts_to_batch(
    tmp_db,
    seed_job,
    monkeypatch,
    tmp_path,
):
    import applypilot.scoring.cover_letter as cover_mod

    conn = tmp_db()
    in_batch = seed_job(
        conn,
        url_suffix="in-batch",
        fit_score=9,
        state="tailored",
        tailored_resume_path="/tmp/r.docx",
        cover_letter_path=None,
    )
    seed_job(
        conn,
        url_suffix="unrelated-prior-batch",
        fit_score=9,
        state="tailored",
        tailored_resume_path="/tmp/r2.docx",
        cover_letter_path=None,
    )

    calls: list[str] = []

    def tracking_fake(job, resume_text, profile, doc_format="docx"):
        calls.append(job["url"])
        return _fake_cover_one_job(job, resume_text, profile, doc_format)

    monkeypatch.setattr(cover_mod, "_cover_one_job", tracking_fake)
    monkeypatch.setattr(cover_mod, "load_profile", dict)
    monkeypatch.setattr(cover_mod, "COVER_LETTER_DIR", tmp_path)

    result = cover_mod.run_cover_letters(min_score=8, limit=3, job_ids=[in_batch["url"]])

    assert calls == [in_batch["url"]]
    assert result["generated"] == 1


def test_run_cover_letters_reproduces_and_fixes_the_recurring_listing_regression(
    tmp_db,
    seed_job,
    monkeypatch,
    tmp_path,
):
    """The exact reported 3-job scenario: two rows for a recurring listing
    (same title, different URLs) were tailored on earlier days and are
    sitting in pending_cover; tailor's carried batch from THIS run contains
    only the one job it just tailored. Cover must generate exactly 1
    letter -- not backfill to 3 using the unrelated backlog."""
    import applypilot.scoring.cover_letter as cover_mod

    conn = tmp_db()

    old_1 = seed_job(
        conn,
        url_suffix="recurring-listing-old-1",
        title="Seasonal, Operations Technical Specialist (Part Time)",
        fit_score=9,
        state="tailored",
        tailored_resume_path="/tmp/old1.docx",
        cover_letter_path=None,
    )
    old_2 = seed_job(
        conn,
        url_suffix="recurring-listing-old-2",
        title="Seasonal, Operations Technical Specialist (Part Time)",
        fit_score=9,
        state="tailored",
        tailored_resume_path="/tmp/old2.docx",
        cover_letter_path=None,
    )
    new_job = seed_job(
        conn,
        url_suffix="recurring-listing-new",
        title="Seasonal, Operations Technical Specialist (Part Time)",
        fit_score=9,
        state="tailored",
        tailored_resume_path="/tmp/new.docx",
        cover_letter_path=None,
    )
    # Without a job_ids restriction, all three are genuinely pending_cover
    # eligible -- confirms the fixture reproduces the reported shape before
    # asserting the fix.
    from applypilot.database import get_jobs_by_stage

    unrestricted = get_jobs_by_stage(conn, stage="pending_cover", min_score=8)
    assert len(unrestricted) == 3

    calls: list[str] = []

    def tracking_fake(job, resume_text, profile, doc_format="docx"):
        calls.append(job["url"])
        return _fake_cover_one_job(job, resume_text, profile, doc_format)

    monkeypatch.setattr(cover_mod, "_cover_one_job", tracking_fake)
    monkeypatch.setattr(cover_mod, "load_profile", dict)
    monkeypatch.setattr(cover_mod, "COVER_LETTER_DIR", tmp_path)

    # This is what pipeline._run_sequential now carries forward: only the
    # URL tailor actually produced this run.
    result = cover_mod.run_cover_letters(min_score=8, limit=3, job_ids=[new_job["url"]])

    assert calls == [new_job["url"]]
    assert result["generated"] == 1
    assert old_1["url"] not in calls
    assert old_2["url"] not in calls


def test_run_cover_letters_empty_job_ids_generates_nothing(
    tmp_db,
    seed_job,
    monkeypatch,
    tmp_path,
):
    import applypilot.scoring.cover_letter as cover_mod

    conn = tmp_db()
    seed_job(
        conn, fit_score=9, state="tailored", tailored_resume_path="/tmp/r.docx", cover_letter_path=None
    )  # would be independently eligible

    calls: list[str] = []
    monkeypatch.setattr(cover_mod, "_cover_one_job", lambda *a, **k: calls.append(a[0]["url"]))
    monkeypatch.setattr(cover_mod, "load_profile", dict)
    monkeypatch.setattr(cover_mod, "COVER_LETTER_DIR", tmp_path)

    result = cover_mod.run_cover_letters(min_score=8, limit=3, job_ids=[])

    assert calls == []
    assert result["generated"] == 0


def test_run_cover_letters_without_job_ids_uses_independent_selection(
    tmp_db,
    seed_job,
    monkeypatch,
    tmp_path,
):
    """Standalone `applypilot run cover` (no preceding `tailor` stage in
    this run) must keep selecting independently."""
    import applypilot.scoring.cover_letter as cover_mod

    conn = tmp_db()
    job = seed_job(conn, fit_score=9, state="tailored", tailored_resume_path="/tmp/r.docx", cover_letter_path=None)

    calls: list[str] = []

    def tracking_fake(j, resume_text, profile, doc_format="docx"):
        calls.append(j["url"])
        return _fake_cover_one_job(j, resume_text, profile, doc_format)

    monkeypatch.setattr(cover_mod, "_cover_one_job", tracking_fake)
    monkeypatch.setattr(cover_mod, "load_profile", dict)
    monkeypatch.setattr(cover_mod, "COVER_LETTER_DIR", tmp_path)

    result = cover_mod.run_cover_letters(min_score=8, limit=3)  # job_ids defaults to None

    assert calls == [job["url"]]
    assert result["generated"] == 1


def test_run_cover_letters_logs_carried_batch_size(
    tmp_db,
    seed_job,
    monkeypatch,
    tmp_path,
    caplog,
):
    import logging

    import applypilot.scoring.cover_letter as cover_mod

    conn = tmp_db()
    eligible = seed_job(
        conn,
        url_suffix="eligible",
        fit_score=9,
        state="tailored",
        tailored_resume_path="/tmp/r.docx",
        cover_letter_path=None,
    )
    low = seed_job(
        conn,
        url_suffix="low",
        fit_score=2,
        state="tailored",
        tailored_resume_path="/tmp/r2.docx",
        cover_letter_path=None,
    )

    monkeypatch.setattr(cover_mod, "_cover_one_job", _fake_cover_one_job)
    monkeypatch.setattr(cover_mod, "load_profile", dict)
    monkeypatch.setattr(cover_mod, "COVER_LETTER_DIR", tmp_path)

    with caplog.at_level(logging.INFO, logger="applypilot.scoring.cover_letter"):
        cover_mod.run_cover_letters(min_score=8, job_ids=[eligible["url"], low["url"]])

    assert any("carried batch of 2" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# pipeline._run_sequential: carries tailor's batch into the following cover
# stage, and leaves standalone `cover` runs unaffected.
# ---------------------------------------------------------------------------


def test_run_sequential_carries_tailor_batch_into_cover():
    from applypilot import pipeline

    calls: dict[str, dict] = {}

    def fake_tailor(**kwargs):
        calls["tailor"] = kwargs
        return {"status": "ok", "job_ids": ["https://example.com/new-job"]}

    def fake_cover(**kwargs):
        calls["cover"] = kwargs
        return {"status": "ok"}

    with patch.object(pipeline, "_STAGE_RUNNERS", {"tailor": fake_tailor, "cover": fake_cover}):
        pipeline._run_sequential(["tailor", "cover"], min_score=8, limit=3)

    assert calls["cover"]["job_ids"] == ["https://example.com/new-job"]


def test_run_sequential_full_chain_carries_score_through_tailor_to_cover():
    """The exact reported pipeline: enrich -> score -> tailor -> cover.
    Each stage's carried batch must be independently correct -- tailor
    restricted to score's batch, cover restricted to tailor's (smaller,
    post-eligibility) batch, not score's original batch."""
    from applypilot import pipeline

    calls: dict[str, dict] = {}

    def fake_score(**kwargs):
        calls["score"] = kwargs
        return {"status": "ok", "job_ids": ["a", "b", "c"]}

    def fake_tailor(**kwargs):
        calls["tailor"] = kwargs
        # Only 1 of the 3 scored jobs was actually eligible/tailored.
        return {"status": "ok", "job_ids": ["a"]}

    def fake_cover(**kwargs):
        calls["cover"] = kwargs
        return {"status": "ok"}

    with patch.object(pipeline, "_STAGE_RUNNERS", {"score": fake_score, "tailor": fake_tailor, "cover": fake_cover}):
        pipeline._run_sequential(["score", "tailor", "cover"], min_score=8, limit=3)

    assert calls["tailor"]["job_ids"] == ["a", "b", "c"]
    assert calls["cover"]["job_ids"] == ["a"]


def test_run_sequential_standalone_cover_gets_no_job_ids():
    """A `cover`-only run (no `tailor` stage in this invocation) must not
    receive a job_ids kwarg at all -- preserving independent selection."""
    from applypilot import pipeline

    calls: dict[str, dict] = {}

    def fake_cover(**kwargs):
        calls["cover"] = kwargs
        return {"status": "ok"}

    with patch.object(pipeline, "_STAGE_RUNNERS", {"cover": fake_cover}):
        pipeline._run_sequential(["cover"], min_score=8, limit=3)

    assert "job_ids" not in calls["cover"]


def test_run_sequential_tailor_error_does_not_carry_stale_job_ids_to_cover():
    """If the tailor stage errors out, cover must fall back to independent
    selection rather than being restricted to a nonexistent/partial batch."""
    from applypilot import pipeline

    calls: dict[str, dict] = {}

    def failing_tailor(**kwargs):
        return {"status": "error: boom"}

    def fake_cover(**kwargs):
        calls["cover"] = kwargs
        return {"status": "ok"}

    with patch.object(pipeline, "_STAGE_RUNNERS", {"tailor": failing_tailor, "cover": fake_cover}):
        pipeline._run_sequential(["tailor", "cover"], min_score=8, limit=3)

    assert "job_ids" not in calls["cover"]
