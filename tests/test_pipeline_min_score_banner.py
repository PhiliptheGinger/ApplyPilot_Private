"""Regression test for the 2026-08-30 fix: run_pipeline()'s banner used to
print "Min score: N" unconditionally, even for a stage list (e.g. `score`
alone) where min_score is never actually consumed -- _run_score has no
min_score parameter at all; only _run_tailor/_run_cover read it (their
pending_tailor/pending_cover `fit_score >= ?` filter). The banner now only
shows the line when at least one requested stage actually uses it.

Does not change --min-score's semantics -- purely a display fix, verified
here via dry_run=True (no DB writes, no LLM calls).
"""

from __future__ import annotations

from applypilot.pipeline import run_pipeline


def test_score_only_run_hides_min_score_line(tmp_db, capsys):
    tmp_db()
    run_pipeline(stages=["score"], min_score=8, dry_run=True)
    out = capsys.readouterr().out
    assert "Min score:" not in out


def test_tailor_only_run_shows_min_score_line(tmp_db, capsys):
    tmp_db()
    run_pipeline(stages=["tailor"], min_score=8, dry_run=True)
    out = capsys.readouterr().out
    assert "Min score: 8" in out


def test_cover_only_run_shows_min_score_line(tmp_db, capsys):
    tmp_db()
    run_pipeline(stages=["cover"], min_score=8, dry_run=True)
    out = capsys.readouterr().out
    assert "Min score: 8" in out


def test_score_and_tailor_run_shows_min_score_line(tmp_db, capsys):
    """A multi-stage run where only SOME stages use min_score must still
    show it -- the line communicates the tailor sub-stage's behavior, not
    a blanket claim about every requested stage."""
    tmp_db()
    run_pipeline(stages=["score", "tailor"], min_score=8, dry_run=True)
    out = capsys.readouterr().out
    assert "Min score: 8" in out


def test_discover_enrich_only_run_hides_min_score_line(tmp_db, capsys):
    tmp_db()
    run_pipeline(stages=["discover", "enrich"], min_score=8, dry_run=True)
    out = capsys.readouterr().out
    assert "Min score:" not in out
