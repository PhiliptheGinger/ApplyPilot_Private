"""Tests for the embedded-ATS URL canonicalizer."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.discovery.url_normalize import (
    canonicalize_application_url,
    GREENHOUSE_HOST_SLUGS,
)


# ── gh_jid rewrites ──────────────────────────────────────────────────────────

def test_databricks_gh_jid_to_canonical():
    src = "https://www.databricks.com/company/careers/engineering/some-role-6779084002?gh_jid=6779084002"
    assert canonicalize_application_url(src) == \
        "https://job-boards.greenhouse.io/databricks/jobs/6779084002"


def test_stripe_gh_jid_to_canonical():
    src = "https://stripe.com/jobs/search?gh_jid=7393169"
    assert canonicalize_application_url(src) == \
        "https://job-boards.greenhouse.io/stripe/jobs/7393169"


def test_pinterest_careers_to_canonical():
    src = "https://www.pinterestcareers.com/jobs/?gh_jid=7685119"
    assert canonicalize_application_url(src) == \
        "https://job-boards.greenhouse.io/pinterest/jobs/7685119"


def test_airbnb_subdomain_host_match():
    src = "https://careers.airbnb.com/positions/7744247?gh_jid=7744247"
    assert canonicalize_application_url(src) == \
        "https://job-boards.greenhouse.io/airbnb/jobs/7744247"


def test_extra_query_params_dropped():
    src = "https://cast.ai/careers/apply/?gh_jid=4136062009&gh_src=5rago2ff9us&urlHash=NeaV"
    assert canonicalize_application_url(src) == \
        "https://job-boards.greenhouse.io/castai/jobs/4136062009"


# ── unchanged paths ──────────────────────────────────────────────────────────

def test_already_canonical_unchanged():
    src = "https://job-boards.greenhouse.io/databricks/jobs/6779084002"
    assert canonicalize_application_url(src) == src


def test_old_greenhouse_host_unchanged():
    # boards.greenhouse.io (legacy) should also not be rewritten — already on
    # the right host.
    src = "https://boards.greenhouse.io/andurilindustries/jobs/4754841007"
    assert canonicalize_application_url(src) == src


def test_no_gh_jid_unchanged():
    src = "https://www.databricks.com/company/careers/some-role"
    assert canonicalize_application_url(src) == src


def test_unknown_host_unchanged():
    # Not in GREENHOUSE_HOST_SLUGS — leave alone rather than guess wrong.
    src = "https://example-startup.dev/apply?gh_jid=12345"
    assert canonicalize_application_url(src) == src


def test_non_numeric_gh_jid_unchanged():
    # Defensive: gh_jid should always be numeric in real Greenhouse URLs.
    src = "https://databricks.com/jobs?gh_jid=abc"
    assert canonicalize_application_url(src) == src


def test_relative_url_unchanged():
    assert canonicalize_application_url("/jobs/123?gh_jid=456") == "/jobs/123?gh_jid=456"


def test_empty_string_unchanged():
    assert canonicalize_application_url("") == ""


def test_none_unchanged():
    # canonicalize_application_url is documented to return the input unchanged
    # when it can't or shouldn't rewrite.  None falls into that bucket.
    assert canonicalize_application_url(None) is None


# ── slug map sanity ──────────────────────────────────────────────────────────

def test_slug_map_has_top_employers():
    """Regression: don't accidentally drop the high-volume hosts."""
    for host in ("databricks.com", "stripe.com", "pinterestcareers.com",
                 "careers.airbnb.com"):
        assert host in GREENHOUSE_HOST_SLUGS, f"missing {host}"
