"""Tests for the pre-apply expired-posting check (CLAUDE.md Future Work item 31).

Decision #156 found real Claude Code Pro-plan usage burned on jobs that had
expired between being queued and `apply` actually reaching them. Enrichment
already classifies expired postings (HTTP 404/410/451, or a same-site
redirect all the way to the root path) but only once, days earlier via a
full Playwright page load. `precheck_expired` reuses those same two signals
via a plain HTTP request so `apply` can skip a dead posting for near-zero
cost, right before it would otherwise spawn Chrome + a Claude agent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx

from applypilot.enrichment.detail import precheck_expired


def _fake_response(status_code: int, final_url: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.url = final_url
    return resp


class TestPrecheckExpired:
    def test_http_404_is_expired(self):
        with patch("httpx.get", return_value=_fake_response(404, "https://jobs.example.com/posting/123")):
            assert precheck_expired("https://jobs.example.com/posting/123") is True

    def test_http_410_is_expired(self):
        with patch("httpx.get", return_value=_fake_response(410, "https://jobs.example.com/posting/123")):
            assert precheck_expired("https://jobs.example.com/posting/123") is True

    def test_redirect_to_site_root_is_expired(self):
        # Real WWR shape (decision #77/#156's motivating case): HTTP 200,
        # but the final URL is the bare homepage of the same site.
        with patch("httpx.get", return_value=_fake_response(200, "https://weworkremotely.com/")):
            assert precheck_expired("https://weworkremotely.com/remote-jobs/some-stale-posting") is True

    def test_redirect_to_different_site_root_is_not_expired(self):
        # Same path shape ("/") but a DIFFERENT host -- e.g. an ATS-hosted
        # posting redirecting through an auth/tracking domain. Must not be
        # treated as the same-site "posting is gone" signal.
        with patch("httpx.get", return_value=_fake_response(200, "https://sso.example.com/")):
            assert precheck_expired("https://jobs.example.com/posting/123") is False

    def test_live_posting_is_not_expired(self):
        with patch(
            "httpx.get",
            return_value=_fake_response(200, "https://jobs.example.com/posting/123"),
        ):
            assert precheck_expired("https://jobs.example.com/posting/123") is False

    def test_timeout_is_not_treated_as_expired(self):
        # Ambiguous/inconclusive outcomes must fall through to False -- a
        # live job wrongly skipped costs a real application, which is a
        # bigger, less reversible cost than one wasted Claude apply attempt.
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            assert precheck_expired("https://jobs.example.com/posting/123") is False

    def test_connection_error_is_not_treated_as_expired(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
            assert precheck_expired("https://jobs.example.com/posting/123") is False

    def test_retriable_status_is_not_treated_as_expired(self):
        # A transient 503 is a retriable enrichment signal elsewhere in this
        # module, not a permanent "gone" signal -- must not be conflated.
        with patch("httpx.get", return_value=_fake_response(503, "https://jobs.example.com/posting/123")):
            assert precheck_expired("https://jobs.example.com/posting/123") is False
