"""Regression tests for the 2026-08-30 streaming progress-accounting fix.

Real incident: a `run score tailor cover --stream --limit 1` session
repeatedly logged "Stage 'score' pass N had pending=... but progress=0;
backing off 10s" -- 13 consecutive passes observed live, every one of them
immediately after successfully scoring a job. Two independent, compounding
bugs, both confirmed by direct code reading AND by watching a real
streaming run:

1. `_run_stage_streaming`'s progress-detection key list (now `_PROGRESS_KEYS`)
   never checked for `run_scoring()`'s actual success-count key ("scored")
   -- only "errors" happened to match.

2. The REAL root cause of what was actually observed live: `_run_score`,
   `_run_tailor`, and `_run_cover` (the wrapper functions
   `_run_stage_streaming` actually calls, via `_STAGE_RUNNERS`) each
   discarded their inner `run_scoring()`/`run_tailoring()`/
   `run_cover_letters()` call's real stats dict and returned only a bare
   `{"status": "ok", ...}` placeholder. Fixing bug #1 alone was NOT
   sufficient -- the real numbers never reached `_stage_progress` in the
   first place, because these wrapper functions threw them away one layer
   up. This was caught only by watching a real streaming run continue to
   log "progress=0" even after `_PROGRESS_KEYS` was fixed; a purely static
   read of `run_scoring()`'s own return dict (which DOES have "scored")
   was not enough to catch that the wrapper never forwards it.
   `_run_enrich` had the identical pattern (discarding
   `run_enrichment()`'s real "processed"/"ok"/"partial"/"error" stats for
   a `{"status": "ok"}` placeholder) and needed the same fix.

A false progress=0 on every successful pass means a needless 10s backoff
after EVERY batch, not just when a stage genuinely has no work -- directly
contributing to the "streaming mode processed one job at a time, very
slowly" symptom from the same incident (a 10s sleep tacked onto every
otherwise-fast pass, indefinitely, for as long as the backlog lasts).
"""

from __future__ import annotations

from applypilot.pipeline import (
    _PROGRESS_KEYS,
    _run_cover,
    _run_enrich,
    _run_pdf,
    _run_score,
    _run_tailor,
    _stage_progress,
)


class TestStageProgress:
    def test_run_scoring_style_result_counts_as_progress(self):
        """The exact real-incident shape: run_scoring()'s own return dict."""
        result = {"scored": 1, "errors": 0, "elapsed": 4.2, "distribution": [], "job_urls": ["https://x"]}
        assert _stage_progress(result) == 1

    def test_run_cover_letters_style_result_counts_as_progress(self):
        result = {"generated": 1, "rejected": 0, "errors": 0, "elapsed": 2.1}
        assert _stage_progress(result) == 1

    def test_run_cover_letters_rejected_also_counts_as_progress(self):
        """A rejected-by-validation cover letter still consumed a real LLM
        call and produced a real terminal state (cover_failed) -- it is not
        "no work happened", the same way tailor's "failed" already counts."""
        result = {"generated": 0, "rejected": 1, "errors": 0, "elapsed": 1.5}
        assert _stage_progress(result) == 1

    def test_run_tailoring_style_result_counts_as_progress(self):
        """Already worked before this fix -- regression guard."""
        result = {"approved": 1, "failed": 0, "errors": 0, "elapsed": 30.0, "job_urls": []}
        assert _stage_progress(result) == 1

    def test_genuinely_empty_pass_is_zero_progress(self):
        """Cap-blocked / all-filtered pass -- must still correctly report no
        progress so the backoff-to-avoid-hot-looping behavior is preserved."""
        result = {"scored": 0, "errors": 0, "elapsed": 0.01, "distribution": [], "job_urls": []}
        assert _stage_progress(result) == 0

    def test_errors_alone_still_counts_as_progress(self):
        """A pass that hit real errors did real work (attempted jobs, burned
        LLM calls) -- must not be indistinguishable from an empty pass."""
        result = {"scored": 0, "errors": 3, "elapsed": 1.0, "distribution": [], "job_urls": []}
        assert _stage_progress(result) == 3

    def test_progress_keys_cover_every_real_runner_success_key(self):
        """Direct guard against the exact bug class: each real runner's own
        success-count key(s) must be present in the shared list."""
        for key in ("scored", "approved", "failed", "generated", "rejected", "errors"):
            assert key in _PROGRESS_KEYS


class TestRunEnrichForwardsRealStats:
    def test_run_enrich_does_not_collapse_stats_to_a_placeholder(self, monkeypatch):
        """_run_enrich must forward run_enrichment()'s real stats dict (its
        own docstring: "processed, ok, partial, error, tiers") instead of
        discarding it for {"status": "ok"} -- otherwise every successful
        enrich pass looks like zero progress to the streaming coordinator."""
        import applypilot.enrichment.detail as detail_mod

        fake_stats = {"processed": 5, "ok": 4, "partial": 1, "error": 0, "tiers": {}}
        monkeypatch.setattr(detail_mod, "run_enrichment", lambda workers=1: fake_stats)

        result = _run_enrich(workers=1)

        assert result["processed"] == 5
        assert result["ok"] == 4
        assert result["status"] == "ok"
        assert _stage_progress(result) > 0

    def test_run_enrich_still_reports_errors_on_exception(self, monkeypatch):
        import applypilot.enrichment.detail as detail_mod

        def _boom(workers=1):
            raise RuntimeError("scrape failed")

        monkeypatch.setattr(detail_mod, "run_enrichment", _boom)

        result = _run_enrich(workers=1)
        assert result["status"].startswith("error")


class TestRunScoreForwardsRealStats:
    def test_run_score_does_not_collapse_stats_to_a_placeholder(self, monkeypatch):
        """The exact real-incident bug: _run_score used to return only
        {"status": "ok", "job_ids": [...]}, discarding run_scoring()'s real
        "scored"/"errors" counts. Confirmed live: a real streaming run kept
        logging false progress=0 even after _PROGRESS_KEYS was fixed,
        because this wrapper threw the real numbers away first."""
        import applypilot.scoring.scorer as scorer_mod

        fake_result = {
            "scored": 1,
            "errors": 0,
            "elapsed": 9.4,
            "distribution": [],
            "job_urls": ["https://example.com/job1"],
        }
        monkeypatch.setattr(
            scorer_mod, "run_scoring", lambda workers=1, max_age_days=None, limit=0: fake_result
        )

        result = _run_score(workers=1, limit=1)

        assert result["scored"] == 1
        assert result["status"] == "ok"
        assert result["job_ids"] == ["https://example.com/job1"]
        assert _stage_progress(result) > 0

    def test_run_score_still_reports_errors_on_exception(self, monkeypatch):
        import applypilot.scoring.scorer as scorer_mod

        def _boom(workers=1, max_age_days=None, limit=0):
            raise RuntimeError("LLM provider down")

        monkeypatch.setattr(scorer_mod, "run_scoring", _boom)

        result = _run_score(workers=1, limit=1)
        assert result["status"].startswith("error")


class TestRunTailorForwardsRealStats:
    def test_run_tailor_does_not_collapse_stats_to_a_placeholder(self, monkeypatch):
        import applypilot.scoring.tailor as tailor_mod

        fake_result = {
            "approved": 1,
            "failed": 0,
            "errors": 0,
            "elapsed": 30.0,
            "job_urls": ["https://example.com/job1"],
        }
        monkeypatch.setattr(
            tailor_mod,
            "run_tailoring",
            lambda min_score=None, max_age_days=None, limit=20, workers=1, doc_format="docx", job_ids=None: fake_result,
        )

        result = _run_tailor(limit=1)

        assert result["approved"] == 1
        assert result["status"] == "ok"
        assert _stage_progress(result) > 0

    def test_run_tailor_still_reports_errors_on_exception(self, monkeypatch):
        import applypilot.scoring.tailor as tailor_mod

        def _boom(**kwargs):
            raise RuntimeError("tailoring crashed")

        monkeypatch.setattr(tailor_mod, "run_tailoring", _boom)

        result = _run_tailor(limit=1)
        assert result["status"].startswith("error")


class TestRunCoverForwardsRealStats:
    def test_run_cover_does_not_collapse_stats_to_a_placeholder(self, monkeypatch):
        """_run_cover used to discard run_cover_letters()'s return value
        entirely (didn't even assign it to a variable)."""
        import applypilot.scoring.cover_letter as cover_mod

        fake_result = {"generated": 1, "rejected": 0, "errors": 0, "elapsed": 2.1}
        monkeypatch.setattr(
            cover_mod,
            "run_cover_letters",
            lambda min_score=None, max_age_days=None, limit=20, workers=1, doc_format="docx", job_ids=None: fake_result,
        )

        result = _run_cover(limit=1)

        assert result["generated"] == 1
        assert result["status"] == "ok"
        assert _stage_progress(result) > 0

    def test_run_cover_still_reports_errors_on_exception(self, monkeypatch):
        import applypilot.scoring.cover_letter as cover_mod

        def _boom(**kwargs):
            raise RuntimeError("cover generation crashed")

        monkeypatch.setattr(cover_mod, "run_cover_letters", _boom)

        result = _run_cover(limit=1)
        assert result["status"].startswith("error")


class TestRunPdfForwardsRealStats:
    def test_run_pdf_does_not_collapse_stats_to_a_placeholder(self, monkeypatch):
        """Same bug class as the other four wrappers above -- _run_pdf
        discarded batch_convert()'s real int count and returned a bare
        {"status": "ok"} placeholder, which reads as zero progress to the
        streaming coordinator no matter how many files were converted."""
        import applypilot.scoring.pdf as pdf_mod

        monkeypatch.setattr(pdf_mod, "batch_convert", lambda doc_format="docx": 3)

        result = _run_pdf(doc_format="docx")

        assert result["processed"] == 3
        assert result["status"] == "ok"
        assert _stage_progress(result) > 0

    def test_run_pdf_still_reports_errors_on_exception(self, monkeypatch):
        import applypilot.scoring.pdf as pdf_mod

        def _boom(doc_format="docx"):
            raise RuntimeError("conversion crashed")

        monkeypatch.setattr(pdf_mod, "batch_convert", _boom)

        result = _run_pdf(doc_format="docx")
        assert result["status"].startswith("error")
