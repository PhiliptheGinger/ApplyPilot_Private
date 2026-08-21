"""Regression tests for false-positive exclusion bugs found in a 2026-08-21
audit of every title/location/description filter in the pipeline.

The canonical seniority guard (applypilot.eligibility) was confirmed clean
-- these tests are NOT about that module. They cover two adjacent, legitimately
separate mechanisms that were using plain substring matching instead of
word-boundary matching, which could wrongly reject legitimate jobs:

1. scoring/scorer.py's searches.yaml `exclude_titles` handling: raw
   `re.search(pattern, title)` on plain config words let "VP " match
   inside "MVP", "head" match inside "Overhead Crane Technician", etc.
2. discovery/{jobspy,workday,smartextract}.py's `_location_ok()` reject-list
   check: plain `pattern in location.lower()` let reject-pattern "India"
   match "Indianapolis, IN".
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.scoring.scorer import _check_ineligible


def _job(title, location="Remote (US)", description="A full-time US-remote position."):
    return {"title": title, "location": location, "full_description": description}


_LIVE_EXCLUDE_TITLES = [
    "senior", "sr.", "staff", "principal", "lead", "manager", "head",
    "senior director", "director", "VP ", "vice president", "chief",
    "architect", "distinguished", "fellow", "intern", "internship",
    "co-op", "clearance required", "TS/SCI",
]


def _cfg(exclude_titles=None):
    return {"exclude_titles": exclude_titles if exclude_titles is not None else _LIVE_EXCLUDE_TITLES}


# ── scorer.py exclude_titles: word-boundary fix ──────────────────────────

class TestExcludeTitlesWordBoundary:
    @patch("applypilot.scoring.scorer.load_search_config")
    def test_mvp_title_not_excluded_by_vp_pattern(self, mock_cfg):
        """'VP ' in exclude_titles must not match the substring inside 'MVP'."""
        mock_cfg.return_value = _cfg()
        assert _check_ineligible(_job("MVP Product Engineer")) is None

    @patch("applypilot.scoring.scorer.load_search_config")
    def test_overhead_technician_not_excluded_by_head_pattern(self, mock_cfg):
        """'head' in exclude_titles must not match the substring inside
        'Overhead' -- a real blue-collar/install title matching the
        candidate's background, not a management role."""
        mock_cfg.return_value = _cfg()
        assert _check_ineligible(_job("Overhead Crane Technician")) is None

    @patch("applypilot.scoring.scorer.load_search_config")
    def test_leadframe_technician_not_excluded_by_lead_pattern(self, mock_cfg):
        """'lead' in exclude_titles must not match the substring inside
        'Leadframe' (an electronics manufacturing term, not seniority)."""
        mock_cfg.return_value = _cfg()
        assert _check_ineligible(_job("Leadframe Assembly Technician")) is None

    @patch("applypilot.scoring.scorer.load_search_config")
    def test_genuine_config_exclusion_still_excluded(self, mock_cfg):
        """Regression guard: fixing the false positive must not weaken real
        exclude_titles matches. Uses 'co-op' rather than a seniority word
        (VP/head/lead/etc.) deliberately -- those are ALSO independently
        caught by the canonical seniority_disqualifier() (checked earlier
        in _check_ineligible), so a title like 'VP of Engineering' would
        still be rejected even with the exclude_titles matching completely
        broken, proving nothing about this fix in isolation. 'co-op' has no
        overlap with SENIORITY_TITLE_PATTERN, so this can only be the
        exclude_titles path -- confirmed via the reason string too."""
        mock_cfg.return_value = _cfg()
        reason = _check_ineligible(_job("Software Engineering Co-op"))
        assert reason is not None
        assert "search configuration" in reason
        assert "seniority" not in reason.lower()

    @patch("applypilot.scoring.scorer.load_search_config")
    def test_genuine_clearance_phrase_still_excluded(self, mock_cfg):
        """Multi-word exclude_titles entries ('clearance required') must
        still match as a phrase after the word-boundary fix."""
        mock_cfg.return_value = _cfg()
        assert _check_ineligible(_job("Software Engineer (Clearance Required)")) is not None

    @patch("applypilot.scoring.scorer.load_search_config")
    def test_ts_sci_token_still_excluded(self, mock_cfg):
        mock_cfg.return_value = _cfg()
        assert _check_ineligible(_job("Software Engineer - TS/SCI")) is not None

    @patch("applypilot.scoring.scorer.load_search_config")
    def test_trailing_space_config_entry_normalized(self, mock_cfg):
        """A config entry with the live config's manual trailing-space
        word-boundary hack (e.g. 'VP ') must be stripped and still match
        as a genuine word after normalization. Uses 'intern ' rather than
        'VP ' -- 'VP' is also independently caught by the canonical
        seniority_disqualifier(), so a 'VP '-based version of this test
        would pass even if this fix's strip()+\\b-wrapping were broken.
        'intern' has no overlap with SENIORITY_TITLE_PATTERN, so a match
        here can only come from the exclude_titles path."""
        mock_cfg.return_value = _cfg(exclude_titles=["intern "])
        reason = _check_ineligible(_job("Software Engineering Intern"))
        assert reason is not None
        assert "search configuration" in reason

    @patch("applypilot.scoring.scorer.load_search_config")
    def test_title_reject_patterns_still_treated_as_real_regex(self, mock_cfg):
        """title_reject_patterns (distinct from exclude_titles) are
        genuine user-supplied regex and must be unaffected by this fix."""
        mock_cfg.return_value = {
            "exclude_titles": [],
            "title_reject_patterns": [r"\bcontractor\b"],
        }
        assert _check_ineligible(_job("Software Engineer Contractor")) is not None
        assert _check_ineligible(_job("Software Engineer")) is None


# ── discovery _location_ok: word-boundary fix on reject lists ────────────

class TestLocationRejectWordBoundary:
    """accept always includes a pattern that matches the test location, so
    the accept-list's own (unmodified, out-of-scope) substring semantics
    can never be what determines the result -- isolates the reject-list
    word-boundary fix as the only variable under test."""

    def test_jobspy_indianapolis_not_rejected_by_india_pattern(self):
        from applypilot.discovery.jobspy import _location_ok
        assert _location_ok("Indianapolis, IN", accept=["IN"], reject=["india"]) is True

    def test_jobspy_mumbai_india_still_rejected(self):
        from applypilot.discovery.jobspy import _location_ok
        assert _location_ok("Mumbai, India", accept=["india"], reject=["india"]) is False

    def test_workday_indianapolis_not_rejected_by_india_pattern(self):
        from applypilot.discovery.workday import _location_ok
        assert _location_ok("Indianapolis, IN", accept=["IN"], reject=["india"]) is True

    def test_workday_mumbai_india_still_rejected(self):
        from applypilot.discovery.workday import _location_ok
        assert _location_ok("Mumbai, India", accept=["india"], reject=["india"]) is False

    def test_smartextract_indianapolis_not_rejected_by_india_pattern(self):
        from applypilot.discovery.smartextract import _location_ok
        assert _location_ok("Indianapolis, IN", accept=["IN"], reject=["india"]) is True

    def test_smartextract_mumbai_india_still_rejected(self):
        from applypilot.discovery.smartextract import _location_ok
        assert _location_ok("Mumbai, India", accept=["india"], reject=["india"]) is False

    def test_jobspy_france_not_rejected_by_nc_pattern(self):
        """'NC' as a reject pattern must not match the substring inside
        'France'."""
        from applypilot.discovery.jobspy import _location_ok
        assert _location_ok("Paris, France", accept=["france"], reject=["nc"]) is True

    def test_jobspy_north_carolina_still_rejected_by_nc_pattern(self):
        from applypilot.discovery.jobspy import _location_ok
        assert _location_ok("Charlotte, NC", accept=["nc"], reject=["nc"]) is False
