"""Tests for sequential-pipeline batch identity: a `tailor` stage that
follows `score` in the same run must operate on exactly the batch `score`
selected, not independently re-query and pull in unrelated already-eligible
jobs sitting in the DB from earlier runs.

Covers:
- get_jobs_by_stage(urls=...): the low-level restriction primitive.
- run_tailoring(job_ids=...): the edge case where only some of the carried
  batch meets the tailoring threshold.
- run_tailoring() without job_ids: standalone behavior is unchanged.
- run_scoring()'s job_urls in its return dict (what pipeline.py carries
  forward).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# get_jobs_by_stage(urls=...)
# ---------------------------------------------------------------------------

def test_get_jobs_by_stage_urls_restricts_to_exact_batch(tmp_db, seed_job):
    from applypilot.database import get_jobs_by_stage
    conn = tmp_db()
    a = seed_job(conn, url_suffix="a", fit_score=9, tailored_resume_path=None)
    b = seed_job(conn, url_suffix="b", fit_score=9, tailored_resume_path=None)
    seed_job(conn, url_suffix="unrelated-prior-batch", fit_score=9, tailored_resume_path=None)

    rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=8,
                             urls=[a["url"], b["url"]])
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
        "url": job["url"], "title": job["title"], "site": job.get("site", ""),
        "status": "approved", "attempts": 1, "path": "/tmp/fake.txt", "pdf_path": None,
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
    monkeypatch.setattr(tailor_mod, "load_profile", lambda: {})
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    result = tailor_mod.run_tailoring(min_score=8, limit=3, job_ids=[in_batch["url"]])

    assert calls == [in_batch["url"]]
    assert result["approved"] == 1


def test_run_tailoring_job_ids_edge_case_only_eligible_from_batch(
    tmp_db, seed_job, monkeypatch, tmp_path,
):
    """Score selects 3 jobs; only 1 meets min_score. Tailor must process
    only that 1 -- not backfill the other 2 slots with the unrelated
    previously-high-scoring job sitting in the DB from an earlier run."""
    import applypilot.scoring.tailor as tailor_mod
    conn = tmp_db()
    eligible = seed_job(conn, url_suffix="eligible", fit_score=9, tailored_resume_path=None)
    low1 = seed_job(conn, url_suffix="low1", fit_score=2, tailored_resume_path=None)
    low2 = seed_job(conn, url_suffix="low2", fit_score=2, tailored_resume_path=None)
    unrelated = seed_job(conn, url_suffix="unrelated-prior-batch", fit_score=9,
                         tailored_resume_path=None)

    calls: list[str] = []

    def tracking_fake(job, resume_text, profile, doc_format="docx"):
        calls.append(job["url"])
        return _fake_tailor_one_job(job, resume_text, profile, doc_format)

    monkeypatch.setattr(tailor_mod, "_tailor_one_job", tracking_fake)
    monkeypatch.setattr(tailor_mod, "load_profile", lambda: {})
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
    monkeypatch.setattr(tailor_mod, "_tailor_one_job",
                        lambda *a, **k: calls.append(a[0]["url"]))
    monkeypatch.setattr(tailor_mod, "load_profile", lambda: {})
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    result = tailor_mod.run_tailoring(min_score=8, limit=3, job_ids=[])

    assert calls == []
    assert result == {"approved": 0, "failed": 0, "errors": 0, "elapsed": 0.0}


def test_run_tailoring_without_job_ids_uses_independent_selection(
    tmp_db, seed_job, monkeypatch, tmp_path,
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
    monkeypatch.setattr(tailor_mod, "load_profile", lambda: {})
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
    monkeypatch.setattr(tailor_mod, "load_profile", lambda: {})
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    with caplog.at_level(logging.INFO, logger="applypilot.scoring.tailor"):
        tailor_mod.run_tailoring(min_score=8, job_ids=[eligible["url"], low["url"]])

    assert any("carried batch of 2" in r.message for r in caplog.records)


def test_run_tailoring_logs_local_first_enabled_banner(
    tmp_db, seed_job, monkeypatch, tmp_path, caplog,
):
    import logging
    import applypilot.scoring.tailor as tailor_mod
    conn = tmp_db()
    seed_job(conn, fit_score=9, tailored_resume_path=None)

    monkeypatch.setenv("APPLYPILOT_LOCAL_PLAN", "1")
    monkeypatch.setattr(tailor_mod, "_tailor_one_job", _fake_tailor_one_job)
    monkeypatch.setattr(tailor_mod, "load_profile", lambda: {})
    monkeypatch.setattr(tailor_mod, "TAILORED_DIR", tmp_path)

    with caplog.at_level(logging.INFO, logger="applypilot.scoring.tailor"):
        tailor_mod.run_tailoring(min_score=8)

    banner_lines = [r.message for r in caplog.records if "local-first: enabled" in r.message]
    assert len(banner_lines) == 1


# ---------------------------------------------------------------------------
# run_scoring()'s job_urls -- what pipeline.py carries forward
# ---------------------------------------------------------------------------

def test_run_scoring_returns_job_urls_for_the_batch_it_selected(
    tmp_db, seed_job, monkeypatch,
):
    import applypilot.scoring.scorer as scorer_mod

    conn = tmp_db()
    a = seed_job(conn, url_suffix="a", fit_score=None, full_description="x")
    b = seed_job(conn, url_suffix="b", fit_score=None, full_description="x")

    def fake_score_job(resume_text, job):
        return {"score": 9, "keywords": "", "reasoning": ""}

    monkeypatch.setattr(scorer_mod, "score_job", fake_score_job)
    monkeypatch.setattr(scorer_mod, "RESUME_PATH", _FakeResumePath())

    result = scorer_mod.run_scoring(limit=2)

    assert set(result["job_urls"]) == {a["url"], b["url"]}
    assert result["scored"] == 2


class _FakeResumePath:
    def read_text(self, encoding="utf-8"):
        return "Resume text."


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
