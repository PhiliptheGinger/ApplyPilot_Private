"""Shared discovery location filter (applypilot.discovery.location_filter).

2026-10-07: replaced four drifted `_location_ok` copies. These tests pin the
two behavior gaps the consolidation closed, plus the existing contract.
"""

from applypilot.discovery import ashby, greenhouse, jobspy, lever, smartextract, workday
from applypilot.discovery.location_filter import load_location_filter, location_ok


class TestLoadLocationFilter:
    def test_documented_keys(self):
        cfg = {"location": {"accept_patterns": ["NC", "Remote"], "reject_patterns": ["India"]}}
        assert load_location_filter(cfg) == (["NC", "Remote"], ["India"])

    def test_legacy_top_level_keys_still_read(self):
        cfg = {"location_accept": ["Greensboro"], "location_reject_non_remote": ["Seattle"]}
        assert load_location_filter(cfg) == (["Greensboro"], ["Seattle"])

    def test_both_shapes_merge_without_duplicates(self):
        cfg = {
            "location": {"accept_patterns": ["NC", "Raleigh"], "reject_patterns": ["India"]},
            "location_accept": ["nc", "Charlotte"],
            "location_reject_non_remote": ["Seattle", "india"],
        }
        assert load_location_filter(cfg) == (["NC", "Raleigh", "Charlotte"], ["India", "Seattle"])

    def test_missing_and_malformed_config(self):
        assert load_location_filter({}) == ([], [])
        assert load_location_filter({"location": "Remote"}) == ([], [])
        assert load_location_filter({"location": {"accept_patterns": None}}) == ([], [])


class TestLocationOk:
    def test_blank_location_kept(self):
        assert location_ok(None, ["NC"], ["India"]) is True
        assert location_ok("", ["NC"], ["India"]) is True

    def test_remote_kept_even_if_rejected_word_present(self):
        assert location_ok("Remote - India", ["NC"], ["India"]) is True

    def test_reject_is_word_boundary(self):
        assert location_ok("Indianapolis, IN", ["IN"], ["India"]) is True
        assert location_ok("Mumbai, India", ["India"], ["India"]) is False

    def test_empty_accept_means_no_preference(self):
        # Previously jobspy/workday/smartextract dropped this; greenhouse kept it.
        assert location_ok("Austin, TX", [], []) is True
        assert location_ok("Austin, TX", [], ["Austin"]) is False

    def test_accept_substring(self):
        assert location_ok("Greensboro, NC", ["Greensboro"], []) is True
        assert location_ok("Austin, TX", ["Greensboro"], []) is False


def test_every_scraper_uses_the_shared_filter():
    for mod in (jobspy, workday, smartextract, greenhouse):
        assert mod._location_ok is location_ok
    # lever and ashby import greenhouse's name
    assert lever._location_ok is location_ok
    assert ashby._location_ok is location_ok


def test_ats_crawl_passes_reject_list(monkeypatch, tmp_path):
    """Greenhouse/Lever/Ashby crawls used to ignore reject patterns entirely."""
    from applypilot import config
    from applypilot.discovery import ats_common

    monkeypatch.setattr(
        config,
        "load_search_config",
        lambda: {"location": {"accept_patterns": ["NC"], "reject_patterns": ["India"]}},
    )
    monkeypatch.setattr(ats_common, "get_connection", lambda: None, raising=False)
    monkeypatch.setattr(ats_common, "init_db", lambda: None, raising=False)

    seen = {}

    def fake_scrape(slug, emp, accept_locs, reject_locs=None):
        seen["accept"] = accept_locs
        seen["reject"] = reject_locs
        raise RuntimeError("stop after capturing filter args")

    ats_common.run_ats_crawl("Test", "test", "test_api", {"acme": {"name": "Acme"}}, fake_scrape)
    assert seen == {"accept": ["NC"], "reject": ["India"]}


class TestWholeWordAccept:
    """2026-10-07: accept patterns were a substring test."""

    def test_short_codes_do_not_match_inside_words(self):
        assert location_ok("San Francisco, CA", ["NC"], []) is False
        assert location_ok("Austin, TX", ["US"], []) is False
        assert location_ok("Chicago, IL", ["CA"], []) is False

    def test_short_codes_still_match_as_words(self):
        assert location_ok("Greensboro, NC", ["NC"], []) is True
        assert location_ok("Denver, CO, US", ["US"], []) is True

    def test_punctuated_pattern(self):
        assert location_ok("Washington, D.C.", ["Washington, D.C."], []) is True


def test_scorer_location_signal_uses_whole_words(monkeypatch):
    from applypilot import config
    from applypilot.scoring.deterministic_fallback import classify_location_signal

    monkeypatch.setattr(
        config,
        "load_search_config",
        lambda: {"location": {"accept_patterns": ["Greensboro", "NC"], "reject_patterns": ["India"]}},
    )
    assert classify_location_signal({"location": "Greensboro, NC"}) == "accepted"
    assert classify_location_signal({"location": "San Francisco, CA"}) == "away"
    assert classify_location_signal({"location": "Mumbai, India"}) == "away"
    assert classify_location_signal({"location": "Remote"}) == "remote"
