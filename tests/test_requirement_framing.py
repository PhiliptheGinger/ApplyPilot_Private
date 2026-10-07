"""FW28 requirement-framing classifier (decision #236)."""

from __future__ import annotations

import pytest

from applypilot.scoring import deterministic_fallback as df
from applypilot.scoring import requirement_framing as rf


def _frames(text):
    return [(m.years, m.frame) for m in rf.classify_years_mentions(text)]


class TestHardRequirements:
    @pytest.mark.parametrize(
        "text,years",
        [
            ("What you'll need\n- 5+ years of commercial experience with Go\n", 5),
            ("Who you are:\n- 3-5 years of experience in IT support", 3),
            ("Minimum Requirements:\n- Two years of teller experience", 2),
            ("We are seeking a candidate with 10+ years of experience leading platform teams.", 10),
            ("You must have at least 2 years of retail experience.", 2),
            ("We need someone with 4 years of HVAC experience.", 4),
        ],
    )
    def test_agreeing_signals_make_a_hard_requirement(self, text, years):
        assert rf.framed_years_required(text) == years


class TestNotHard:
    @pytest.mark.parametrize(
        "text",
        [
            "Seeking 10+ years in fintech.",  # one signal only: unchanged
            "You have 4+ years building web apps.",  # one signal only
            "Nice to have: 3+ years with Kubernetes",
            "You have 4+ years of Python experience (preferred).",
            "Preferred Qualifications\n- 5 years of Rust",
            "Our team has 15 years of experience building banks.",
            "Founded in 1991, more than 30 years later we serve 2 million customers.",
            "Must have a 4-year degree in any field.",
            "Applicants must be at least 18 years of age.",
            "Benefits\n- 401k match after 1 year of service\n- Sabbatical after 5 years",
            "Responsibilities\n- Lead a team of 5 engineers over the next 2 years",
            "The ideal candidate has 6 years of experience.",
            "Requirements: 3 years experience is a plus",
            "Requirements:\n- Typically requires a degree and less than 2 years of prior relevant experience",
            "Diverse Experiences\nWe value people.\n\n- 3+ years of professional software development experience",
        ],
    )
    def test_returns_none(self, text):
        assert rf.framed_years_required(text) is None, _frames(text)

    def test_preferred_section_after_required_section(self):
        text = "Requirements\n- Customer service skills\n\nPreferred qualifications\n- 3 years of sales\n"
        assert _frames(text) == [(3, rf.PREFERRED)]


class TestImplicitSeniority:
    def test_two_authority_asks(self):
        text = "As a founding engineer you will set the technical direction and mentor other engineers."
        assert rf.implicit_seniority(text) == ["founding", "technical_direction", "mentor_engineers"]

    def test_company_section_ignored(self):
        assert rf.implicit_seniority("About us\nOur founders set the technical direction.\nYou will fix bugs.") == []

    def test_preferred_sentence_ignored(self):
        assert rf.implicit_seniority("Experience mentoring other engineers is a plus.") == []

    def test_one_ask_is_not_senior(self):
        assert not rf.is_implicitly_senior("You will mentor other engineers.")


def test_years_regex_matches_extractor():
    assert rf._YEARS_MENTION_RE.pattern == df._YEARS_MENTION_RE.pattern


class TestScorerIntegration:
    DESC = "What you'll need\n- 5+ years of commercial experience with Go\n"

    def test_shadow_by_default_notes_but_does_not_change(self, monkeypatch):
        monkeypatch.delenv("APPLYPILOT_FRAMING_CLASSIFIER", raising=False)
        years, score, note = df.apply_requirement_framing(self.DESC, "it_or_tech_support", None, 9)
        assert (years, score) == (None, 9)
        assert note == "[shadow] framing: years_required=5, would score 5"

    def test_on_applies(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_FRAMING_CLASSIFIER", "on")
        years, score, note = df.apply_requirement_framing(self.DESC, "it_or_tech_support", None, 9)
        assert (years, score) == (5, 5)
        assert note == "framing: years_required=5, capped at 5"

    def test_never_overrides_extractor_years(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_FRAMING_CLASSIFIER", "on")
        assert df.apply_requirement_framing(self.DESC, "it_or_tech_support", 1, 7) == (1, 7, "")

    def test_never_raises_score(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_FRAMING_CLASSIFIER", "on")
        _y, score, _n = df.apply_requirement_framing(self.DESC, "specialized_or_other", None, 3)
        assert score == 3

    def test_off(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_FRAMING_CLASSIFIER", "off")
        assert df.apply_requirement_framing(self.DESC, "it_or_tech_support", None, 9) == (None, 9, "")

    def test_implied_seniority_caps_software(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_FRAMING_CLASSIFIER", "on")
        desc = "As a founding engineer you will set the technical direction for our platform."
        _y, score, note = df.apply_requirement_framing(desc, "software_engineering", None, 5)
        assert score == 3 and note.startswith("framing: implied seniority")
        # Other families are not capped by implied seniority.
        assert df.apply_requirement_framing(desc, "it_or_tech_support", None, 9)[1] == 9
