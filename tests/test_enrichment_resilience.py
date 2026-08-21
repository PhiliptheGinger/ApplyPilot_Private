"""Regression tests for enrichment batch resilience.

2026-08-21 incident: a single job (a "Built In" listing) timing out during
enrichment killed the entire batch -- `scrape_site_batch`'s per-job loop
had no try/except around `scrape_detail_page`, so an uncaught exception
(most plausibly a Playwright/Patchright timeout from
`collect_detail_intelligence`'s unguarded `page.title()` /
`query_selector_all()` calls) propagated all the way out through
`_run_detail_scraper` and `run_enrichment`, aborting every remaining job
for every remaining site in that run.

These tests exercise the fix directly against `scrape_site_batch` with a
mocked Playwright (`sync_playwright`) and a mocked `scrape_detail_page`
raising on one job -- no real browser is launched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.enrichment.detail import (
    _classify_detail_error,
    _mark_enrich_result,
    _run_detail_scraper,
    scrape_site_batch,
)


def _fake_sync_playwright():
    """A context-manager mock standing in for patchright.sync_api.sync_playwright()."""
    page = MagicMock()
    context = MagicMock()
    context.new_page.return_value = page
    browser = MagicMock()
    browser.new_context.return_value = context
    p = MagicMock()
    p.chromium.launch.return_value = browser

    cm = MagicMock()
    cm.__enter__.return_value = p
    cm.__exit__.return_value = False
    return cm


class TestScrapeSiteBatchSurvivesPerJobCrash:
    def test_one_job_timeout_does_not_abort_the_batch(self, tmp_db, seed_job):
        conn = tmp_db()
        job1 = seed_job(conn, url_suffix="crashes", title="Job That Times Out",
                        site="BuiltIn Remote", full_description=None)
        job2 = seed_job(conn, url_suffix="succeeds", title="Job That Succeeds",
                        site="BuiltIn Remote", full_description=None)

        def _side_effect(page, url):
            if url == job1["url"]:
                raise TimeoutError("Page.title: Timeout 30000ms exceeded.")
            return {
                "status": "ok",
                "full_description": "A real description.",
                "application_url": "https://example.com/apply",
                "tier_used": 1,
                "elapsed": 0.5,
            }

        with patch("applypilot.enrichment.detail.sync_playwright",
                   return_value=_fake_sync_playwright()), \
             patch("applypilot.enrichment.detail.scrape_detail_page",
                   side_effect=_side_effect):
            stats = scrape_site_batch(
                conn, "BuiltIn Remote",
                [(job1["url"], job1["title"]), (job2["url"], job2["title"])],
                delay=0,
            )

        # Both jobs processed -- the crash did not stop the loop.
        assert stats["processed"] == 2
        assert stats["ok"] == 1
        assert stats["error"] == 1

        row1 = conn.execute(
            "SELECT detail_error, detail_error_category, enrich_attempts, "
            "full_description FROM jobs WHERE url = ?", (job1["url"],),
        ).fetchone()
        assert row1["detail_error"] is not None
        assert "Timeout" in row1["detail_error"]
        assert row1["full_description"] is None

        row2 = conn.execute(
            "SELECT full_description, detail_error, detail_scraped_at "
            "FROM jobs WHERE url = ?", (job2["url"],),
        ).fetchone()
        assert row2["full_description"] == "A real description."
        assert row2["detail_error"] is None
        assert row2["detail_scraped_at"] is not None

    def test_crashed_job_classified_retriable_not_permanent(self, tmp_db, seed_job):
        """The crash must route through the SAME existing retry/backoff
        classification as any other enrichment error -- a transient
        timeout must not permanently give up after a single occurrence."""
        conn = tmp_db()
        job = seed_job(conn, url_suffix="crashes2", title="Slow Page",
                       site="BuiltIn Remote", full_description=None)

        with patch("applypilot.enrichment.detail.sync_playwright",
                   return_value=_fake_sync_playwright()), \
             patch("applypilot.enrichment.detail.scrape_detail_page",
                   side_effect=TimeoutError("Timeout 30000ms exceeded.")):
            scrape_site_batch(conn, "BuiltIn Remote", [(job["url"], job["title"])], delay=0)

        row = conn.execute(
            "SELECT detail_error_category, enrich_attempts, enrich_next_retry_at, state "
            "FROM jobs WHERE url = ?", (job["url"],),
        ).fetchone()
        assert row["detail_error_category"] == "retriable"
        assert row["enrich_attempts"] == 1
        assert row["enrich_next_retry_at"] is not None
        # Retriable errors don't transition state -- job stays retryable.
        assert row["state"] != "enrich_failed"

    def test_all_jobs_crashing_still_processes_every_one(self, tmp_db, seed_job):
        """Even a site where every job crashes must not lose jobs -- each
        one gets its own recorded failure, not a single aborted batch."""
        conn = tmp_db()
        jobs = [
            seed_job(conn, url_suffix=f"allcrash-{i}", title=f"Job {i}",
                     site="BuiltIn Remote", full_description=None)
            for i in range(3)
        ]

        with patch("applypilot.enrichment.detail.sync_playwright",
                   return_value=_fake_sync_playwright()), \
             patch("applypilot.enrichment.detail.scrape_detail_page",
                   side_effect=RuntimeError("boom")):
            stats = scrape_site_batch(
                conn, "BuiltIn Remote",
                [(j["url"], j["title"]) for j in jobs], delay=0,
            )

        assert stats["processed"] == 3
        assert stats["error"] == 3
        for j in jobs:
            row = conn.execute(
                "SELECT detail_error FROM jobs WHERE url = ?", (j["url"],),
            ).fetchone()
            assert row["detail_error"] is not None

    def test_successfully_enriched_job_not_reselected_by_pending_query(self, tmp_db, seed_job):
        """After a successful enrichment write, the job must fall outside
        _run_detail_scraper's own pending-jobs WHERE clause (detail_scraped_at
        IS NULL OR retriable-and-due) so it is never reprocessed. This
        exercises the existing, unmodified selection query directly."""
        conn = tmp_db()
        job = seed_job(conn, url_suffix="already-done", title="Done Job",
                       site="BuiltIn Remote", full_description=None)

        _mark_enrich_result(
            conn, job["url"], status="ok",
            full_description="Already enriched.",
            application_url="https://example.com/apply",
            error=None, tier=1, retry_count=0,
        )
        conn.commit()

        pending = conn.execute(
            "SELECT url FROM jobs WHERE url = ? AND ("
            "  detail_scraped_at IS NULL "
            "  OR (detail_error_category = 'retriable' "
            "      AND (enrich_next_retry_at IS NULL OR datetime(enrich_next_retry_at) <= datetime('now')))"
            ")", (job["url"],),
        ).fetchall()
        assert pending == []


class TestRunDetailScraperSurvivesSiteLevelCrash:
    def test_one_site_batch_crash_does_not_block_other_sites(self, tmp_db, seed_job):
        """A whole-batch-level crash (e.g. browser launch failure) for one
        site must not prevent other queued sites from being processed."""
        conn = tmp_db()
        seed_job(conn, url_suffix="crashy-site", title="Job A",
                 site="CrashySite", full_description=None)
        seed_job(conn, url_suffix="fine-site", title="Job B",
                 site="FineSite", full_description=None)

        fine_stats = {"processed": 1, "ok": 1, "partial": 0, "error": 0,
                      "tiers": {1: 1, 2: 0, 3: 0}}

        def _side_effect(conn_arg, site, jobs, delay=2.0, max_jobs=None):
            if site == "CrashySite":
                raise RuntimeError("browser launch failed")
            return fine_stats

        with patch("applypilot.enrichment.detail.scrape_site_batch",
                   side_effect=_side_effect):
            total = _run_detail_scraper(conn)

        assert total["processed"] == 1
        assert total["ok"] == 1

    def test_workers2_one_site_crash_does_not_block_other_sites(self, tmp_db, seed_job):
        """Same scenario as above, but through the workers>1
        ThreadPoolExecutor path -- previously an exception from one
        future's .result() call would propagate out of the
        `for future in as_completed(...)` loop, abandoning collection of
        every other future's (possibly already-finished, successful)
        result too."""
        conn = tmp_db()
        seed_job(conn, url_suffix="crashy-site-w2", title="Job A",
                 site="CrashySite", full_description=None)
        seed_job(conn, url_suffix="fine-site-w2", title="Job B",
                 site="FineSite", full_description=None)

        fine_stats = {"processed": 1, "ok": 1, "partial": 0, "error": 0,
                      "tiers": {1: 1, 2: 0, 3: 0}}

        def _side_effect(conn_arg, site, jobs, delay=2.0, max_jobs=None):
            if site == "CrashySite":
                raise RuntimeError("browser launch failed")
            return fine_stats

        with patch("applypilot.enrichment.detail.scrape_site_batch",
                   side_effect=_side_effect):
            total = _run_detail_scraper(conn, workers=2)

        # The crashed site contributed nothing; the successful site's
        # results are fully preserved despite running concurrently with
        # the crash.
        assert total["processed"] == 1
        assert total["ok"] == 1
        assert total["error"] == 0


class TestClassifyDetailErrorCaseInsensitive:
    def test_capitalized_timeout_error_classified_retriable(self):
        category, next_retry = _classify_detail_error(
            "TimeoutError: Timeout 30000ms exceeded.", current_retry_count=0)
        assert category == "retriable"
        assert next_retry is not None

    def test_lowercase_timeout_still_classified_retriable(self):
        category, _ = _classify_detail_error("connection timeout", current_retry_count=0)
        assert category == "retriable"

    def test_unrelated_error_still_permanent(self):
        category, next_retry = _classify_detail_error("no data extracted", current_retry_count=0)
        assert category == "permanent"
        assert next_retry is None

    def test_http_404_still_expired(self):
        category, _ = _classify_detail_error("HTTP 404", current_retry_count=0)
        assert category == "expired"

    def test_max_retries_exceeded_is_permanent_regardless_of_message(self):
        category, next_retry = _classify_detail_error("timeout", current_retry_count=5)
        assert category == "permanent"
        assert next_retry is None
