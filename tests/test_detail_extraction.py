"""Regression tests for the 2026-09-07 enrichment fixes (CLAUDE.md decision
#75/Future Work item 5): WeWorkRemotely selector coverage and the
redirect-to-site-root expired-listing detection in `scrape_detail_page`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.enrichment.detail import (
    DESCRIPTION_SELECTORS,
    _looks_like_boilerplate,
    scrape_detail_page,
)


class TestWwrSelectorsPresent:
    def test_lis_container_job_selector_added(self):
        assert ".lis-container__job" in DESCRIPTION_SELECTORS
        assert ".lis-container" in DESCRIPTION_SELECTORS


class TestLooksLikeBoilerplate:
    def test_wwr_ad_widget_detected(self):
        assert _looks_like_boilerplate(
            "PRODUCTIVITY\nReplace All Your Work Tools\nAll your tasks, docs, chat, and AI in one place.\nStart Free\nPROMOTED"
        )

    def test_intel_workday_shell_detected(self):
        assert _looks_like_boilerplate(
            "Intel's official careers website. Find your next job and take on projects that shape tomorrow's technology."
        )

    def test_real_description_not_flagged(self):
        assert not _looks_like_boilerplate(
            "About the role: Samsara sits at the center of hardware, software, AI, and the physical world."
        )

    def test_none_not_flagged(self):
        assert not _looks_like_boilerplate(None)


def _mock_page(final_url: str, status: int = 200):
    page = MagicMock()
    resp = MagicMock()
    resp.status = status
    page.goto.return_value = resp
    page.url = final_url
    return page


class TestRedirectToSiteRootTreatedAsExpired:
    def test_wwr_stale_url_redirected_to_homepage_is_expired(self):
        """Real, live-verified case: a WWR posting that expired since
        discovery returns HTTP 200 but silently redirects all the way to
        the bare site root (the generic homepage, ad widget included) --
        no 404/410 anywhere in the response chain."""
        page = _mock_page("https://weworkremotely.com/")
        result = scrape_detail_page(page, "https://weworkremotely.com/remote-jobs/some-expired-posting")
        assert result["error"] == "HTTP 404"
        assert result["full_description"] is None

    def test_same_url_with_trailing_slash_root_also_expired(self):
        page = _mock_page("https://weworkremotely.com")
        result = scrape_detail_page(page, "https://weworkremotely.com/remote-jobs/another-expired-one")
        assert result["error"] == "HTTP 404"

    def test_real_job_page_not_falsely_flagged_as_expired(self):
        """Regression guard: a normal detail page (final URL path deeper
        than the domain root) must proceed to real extraction, not get
        misclassified as expired."""
        page = _mock_page("https://weworkremotely.com/remote-jobs/samsara-staff-software-engineer")
        page.query_selector.return_value = None
        page.query_selector_all.return_value = []
        page.evaluate.return_value = ""
        result = scrape_detail_page(page, "https://weworkremotely.com/remote-jobs/samsara-staff-software-engineer")
        assert result["error"] != "HTTP 404"

    def test_cross_domain_redirect_not_treated_as_expired(self):
        """A redirect to a DIFFERENT domain's root (e.g. an ATS bounce)
        isn't the same-site expiry signal this check targets -- must not be
        misclassified."""
        page = _mock_page("https://some-ats.example.com/")
        page.query_selector.return_value = None
        page.query_selector_all.return_value = []
        page.evaluate.return_value = ""
        result = scrape_detail_page(page, "https://weworkremotely.com/remote-jobs/redirects-elsewhere")
        assert result["error"] != "HTTP 404"
