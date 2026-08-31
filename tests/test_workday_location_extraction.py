"""Tests for the Workday `bulletFields` location fallback (2026-08-30 fix).

`_extract_location()` recovers a posting's location from `bulletFields[1]`
only when `locationsText` is absent/blank -- see workday.py's module
comment for the live-API evidence (Accenture and others omit
`locationsText` entirely but do populate `bulletFields` as
`[requisition_id, location]`). This does not change what `_location_ok()`
accepts or rejects; it only gives it something real to evaluate instead of
an always-blank field.
"""

from unittest.mock import patch

from applypilot.discovery.workday import _extract_location, _location_ok, search_employer

# NC-style accept/reject lists matching this candidate's real searches.yaml
# shape (location.accept_patterns / location_reject_non_remote), used to
# prove the recovered location flows into an unchanged _location_ok().
ACCEPT_LOCS = ["Greensboro", "North Carolina", "NC", "Raleigh", "Charlotte", "Remote", "United States", "US"]
REJECT_LOCS = ["Seattle", "Bay Area", "San Francisco"]


class TestExtractLocationFallback:
    """locationsText absent/blank -> bulletFields[1]."""

    def test_locations_text_absent_falls_back_to_bullet_fields(self):
        j = {"title": "Technical Support L2", "bulletFields": ["R00330861", "Sofia"]}
        assert _extract_location(j) == "Sofia"

    def test_locations_text_blank_string_falls_back_to_bullet_fields(self):
        j = {"locationsText": "", "bulletFields": ["R00330861", "Sofia"]}
        assert _extract_location(j) == "Sofia"

    def test_real_observed_example_sofia(self):
        # Verbatim from the live CXS probe.
        j = {"bulletFields": ["R00330861", "Sofia"]}
        assert _extract_location(j) == "Sofia"

    def test_us_style_location(self):
        # No US example appeared in the live 20-result sample, but the
        # schema is uniform per-tenant ([requisition_id, location]) --
        # this is the same shape with a US value, not a new parsing path.
        j = {"bulletFields": ["R00123456", "Charlotte, NC"]}
        assert _extract_location(j) == "Charlotte, NC"

    def test_paris(self):
        j = {"bulletFields": ["R00299696", "Paris"]}
        assert _extract_location(j) == "Paris"

    def test_london(self):
        j = {"bulletFields": ["14463822", "London"]}
        assert _extract_location(j) == "London"

    def test_mumbai(self):
        j = {"bulletFields": ["ATCI-5048875-S1898194", "Mumbai"]}
        assert _extract_location(j) == "Mumbai"

    def test_sydney(self):
        j = {"bulletFields": ["14586828", "Sydney"]}
        assert _extract_location(j) == "Sydney"

    def test_buenos_aires(self):
        j = {"bulletFields": ["R00086230", "Buenos Aires"]}
        assert _extract_location(j) == "Buenos Aires"

    def test_multi_part_location_string(self):
        # bulletFields[1] can itself contain multiple comma-separated
        # parts (building/campus names) -- passed through verbatim, same
        # as locationsText always has been; no re-splitting/parsing.
        j = {"bulletFields": ["14295710", "Cyberjaya, Century Square"]}
        assert _extract_location(j) == "Cyberjaya, Century Square"


class TestPrecedence:
    def test_populated_locations_text_wins_over_bullet_fields(self):
        j = {"locationsText": "Charlotte, NC", "bulletFields": ["123", "Paris"]}
        assert _extract_location(j) == "Charlotte, NC"


class TestMalformedBulletFields:
    """Missing/malformed bulletFields must never invent a location --
    result should be whatever locationsText already was ("" or None)."""

    def test_no_bullet_fields_key(self):
        assert _extract_location({"title": "X"}) == ""

    def test_empty_bullet_fields_list(self):
        assert _extract_location({"bulletFields": []}) == ""

    def test_bullet_fields_single_element(self):
        # Just the requisition ID, no location paired with it.
        assert _extract_location({"bulletFields": ["REQ"]}) == ""

    def test_bullet_fields_none(self):
        assert _extract_location({"bulletFields": None}) == ""

    def test_bullet_fields_non_list_string(self):
        assert _extract_location({"bulletFields": "Sofia"}) == ""

    def test_bullet_fields_non_list_dict(self):
        assert _extract_location({"bulletFields": {"1": "Sofia"}}) == ""

    def test_second_field_empty_string(self):
        assert _extract_location({"bulletFields": ["R123", ""]}) == ""

    def test_second_field_whitespace_only(self):
        assert _extract_location({"bulletFields": ["R123", "   "]}) == ""

    def test_second_field_non_string(self):
        assert _extract_location({"bulletFields": ["R123", None]}) == ""

    def test_second_field_numeric(self):
        assert _extract_location({"bulletFields": ["R123", 12345]}) == ""

    def test_locations_text_explicit_none_falls_back(self):
        # A key present with value None (not just absent) must be treated
        # the same as absent/blank -- `if loc:` is falsy for None too.
        j = {"locationsText": None, "bulletFields": ["R00330861", "Sofia"]}
        assert _extract_location(j) == "Sofia"


class TestNoUrlOrTitleParsing:
    """The fix must never fall back to parsing externalPath or title text
    -- only locationsText and bulletFields are authoritative."""

    def test_ignores_external_path_and_title_when_no_usable_source(self):
        j = {
            "title": "Technicien Industrialisation F/H - Toulouse",
            "externalPath": "/job/Blagnac/Aircraft-Manager-F-H_R00319084",
        }
        assert _extract_location(j) == ""

    def test_ignores_external_path_even_when_bullet_fields_present(self):
        # bulletFields is authoritative; externalPath's embedded city must
        # never be preferred or blended in, even when it disagrees.
        j = {
            "externalPath": "/job/Blagnac/Aircraft-Manager-F-H_R00319084",
            "bulletFields": ["R00319084", "Paris"],
        }
        assert _extract_location(j) == "Paris"


class TestLocationOkUnchanged:
    """_location_ok() itself is untouched -- these lock down that the
    recovered location is evaluated by the exact same, pre-existing
    policy, and that the policy's own behavior did not shift."""

    def test_still_blank_location_is_kept_unfiltered(self):
        # No locationsText, no usable bulletFields -- _extract_location
        # returns "", and _location_ok's own "unknown -- keep it" rule
        # (unchanged) still applies.
        loc = _extract_location({"bulletFields": ["REQ"]})
        assert loc == ""
        assert _location_ok(loc, ACCEPT_LOCS, REJECT_LOCS) is True

    def test_recovered_us_location_is_accepted(self):
        loc = _extract_location({"bulletFields": ["R123", "Charlotte, NC"]})
        assert _location_ok(loc, ACCEPT_LOCS, REJECT_LOCS) is True

    def test_recovered_non_us_location_is_rejected(self):
        loc = _extract_location({"bulletFields": ["R00299696", "Paris"]})
        assert _location_ok(loc, ACCEPT_LOCS, REJECT_LOCS) is False

    def test_recovered_remote_location_is_accepted(self):
        loc = _extract_location({"bulletFields": ["R1", "Remote"]})
        assert _location_ok(loc, ACCEPT_LOCS, REJECT_LOCS) is True


class TestSearchEmployerIntegration:
    """End-to-end through search_employer(): the CXS response's
    bulletFields reaches the returned job dict's "location" key and the
    existing location_filter, with no extra network call and no DB
    write."""

    @patch("applypilot.discovery.workday.workday_search")
    def test_blank_locations_text_job_gets_bullet_fields_location(self, mock_search):
        mock_search.return_value = {
            "total": 1,
            "jobPostings": [
                {
                    "title": "Technical Support L2",
                    "externalPath": "/job/Sofia/Technical-Support-L2_R00330861",
                    "postedOn": "Posted 18 Days Ago",
                    "bulletFields": ["R00330861", "Sofia"],
                    # No "locationsText" key at all -- matches the live
                    # CXS response shape observed for the affected tenants.
                }
            ],
        }
        employer = {"name": "TestCo", "base_url": "https://testco.wd1.myworkdayjobs.com", "tenant": "testco", "site_id": "Careers"}
        jobs = search_employer(
            "testco",
            employer,
            "technical support",
            location_filter=True,
            accept_locs=ACCEPT_LOCS,
            reject_locs=REJECT_LOCS,
        )
        # Sofia is neither an accept nor a reject match -- falls through
        # to _location_ok's final "no match -- reject unknown" branch.
        assert jobs == []

    @patch("applypilot.discovery.workday.workday_search")
    def test_blank_locations_text_us_job_survives_filter(self, mock_search):
        mock_search.return_value = {
            "total": 1,
            "jobPostings": [
                {
                    "title": "Desktop Support Technician",
                    "externalPath": "/job/Charlotte-North-Carolina/Desktop-Support_R00999999",
                    "bulletFields": ["R00999999", "Charlotte, NC"],
                }
            ],
        }
        employer = {"name": "TestCo", "base_url": "https://testco.wd1.myworkdayjobs.com", "tenant": "testco", "site_id": "Careers"}
        jobs = search_employer(
            "testco",
            employer,
            "desktop support",
            location_filter=True,
            accept_locs=ACCEPT_LOCS,
            reject_locs=REJECT_LOCS,
        )
        assert len(jobs) == 1
        assert jobs[0]["location"] == "Charlotte, NC"

    @patch("applypilot.discovery.workday.workday_search")
    def test_populated_locations_text_unaffected_by_bullet_fields(self, mock_search):
        # Regression guard: a tenant that already populates locationsText
        # (the majority, pre-existing-working case) must be completely
        # unaffected by this change, even if bulletFields is also present
        # and disagrees.
        mock_search.return_value = {
            "total": 1,
            "jobPostings": [
                {
                    "title": "Software Engineer",
                    "locationsText": "Raleigh, NC",
                    "bulletFields": ["R1", "Paris"],
                }
            ],
        }
        employer = {"name": "TestCo", "base_url": "https://testco.wd1.myworkdayjobs.com", "tenant": "testco", "site_id": "Careers"}
        jobs = search_employer(
            "testco",
            employer,
            "software engineer",
            location_filter=True,
            accept_locs=ACCEPT_LOCS,
            reject_locs=REJECT_LOCS,
        )
        assert len(jobs) == 1
        assert jobs[0]["location"] == "Raleigh, NC"
