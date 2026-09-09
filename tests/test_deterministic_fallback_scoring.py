"""Tests for the deterministic/local scoring fallback (CLAUDE.md decision
#76): quota-outage-only rescue scorer for jobs stuck on a Gemini/OpenAI
quota-cooldown error. Real accuracy validated externally at n=52-58 clean
labeled jobs (see decision #76 / data/experiments/
deterministic_score_fallback_20260906/) -- these tests cover the
production wiring (regex extraction, the deterministic combine table,
quota-cooldown detection, and the explicit-invocation DB read/write paths),
not model accuracy, which can't be unit-tested without a real Ollama call.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.scoring.deterministic_fallback import (
    SCORE_METHOD,
    _AMBIGUOUS_TITLE_RE,
    deterministic_combine,
    extract_cs_degree_required,
    extract_years_required,
    is_quota_cooldown_error,
    revalidate_deterministic_fallback_scores,
    run_deterministic_fallback_scoring,
    score_job_deterministic,
)


class TestExtractYearsRequired:
    def test_required_years_extracted(self):
        assert extract_years_required("You must have 3 years of professional experience required.") == 3

    def test_preferred_years_not_counted(self):
        """The real Desktop Support regression (decision #76): '3 years'
        mentioned only as preferred must not read as a hard requirement."""
        assert extract_years_required("3 years of experience preferred.") is None

    def test_no_mention_returns_none(self):
        assert extract_years_required("A great opportunity for a motivated individual.") is None

    def test_smallest_qualifying_number_taken(self):
        text = "5 years of professional experience required. Also 2 years of experience required in Python."
        assert extract_years_required(text) == 2

    def test_empty_description(self):
        assert extract_years_required("") is None
        assert extract_years_required(None) is None

    def test_years_under_minimum_qualifications_header_counts(self):
        """2026-09-08 (decision #82): real regression found via a manual
        accuracy spot-check of a live scoring batch -- a real Sourcegraph/
        Tenable "Security Engineer" posting stated its requirement as
        "MINIMUM QUALIFICATIONS\n\nBachelor's degree with 8+ years of
        hands-on experience with Tenable.io..." with no inline "required"/
        "must have"/"minimum of" phrase near the number, so the old
        inline-only context check missed it entirely -- the job scored a
        false-positive 9/10 instead of being caught by the years gate."""
        text = (
            "MINIMUM QUALIFICATIONS\n\n"
            "Bachelor's degree with 8+ years of hands-on experience with Tenable.io, "
            "Tenable.sc, and related enterprise vulnerability management tools."
        )
        assert extract_years_required(text) == 8

    def test_years_under_required_qualifications_header_counts(self):
        text = "REQUIRED QUALIFICATIONS\n- 5 years of experience with AWS infrastructure\n- Strong Python skills"
        assert extract_years_required(text) == 5

    def test_years_under_basic_qualifications_header_counts(self):
        text = "Basic Qualifications\n- 3 years of experience in a customer-facing technical role"
        assert extract_years_required(text) == 3

    def test_years_under_preferred_header_still_not_counted(self):
        """Real 'Sales Engineer' case found in the same batch: a
        'Preferred Qualifications:' section explicitly states '5 years
        industry experience, preferred' -- this is genuinely optional, and
        must still return None even with the new header-based check."""
        text = (
            "Minimum Qualifications:\n"
            "- Electrical Design/Engineering Background\n"
            "- Communication Proficiency\n"
            "Preferred Qualifications:\n"
            "- 5 years industry experience, preferred\n"
        )
        assert extract_years_required(text) is None

    def test_years_mention_after_required_section_ends_not_counted(self):
        """A years-mention appearing well past the required section (e.g.
        in a benefits/compensation blurb near the end of a long posting)
        must not be swept in just because SOME required-qualifications
        header exists earlier in the text."""
        text = (
            "Required Qualifications:\n- Strong communication skills\n- Team player\n"
            + ("filler text. " * 400)
            + "Employees with 10 years of tenure receive additional PTO."
        )
        assert extract_years_required(text) is None


class TestExtractCsDegreeRequired:
    def test_cs_degree_required_detected(self):
        assert extract_cs_degree_required("Bachelor's degree in Computer Science required.")

    def test_unrelated_degree_not_flagged(self):
        assert not extract_cs_degree_required("Bachelor's degree in Business Administration required.")

    def test_no_degree_mention(self):
        assert not extract_cs_degree_required("No degree required, just enthusiasm.")


class TestDeterministicCombine:
    def test_unclassified_family_is_neutral(self):
        assert deterministic_combine(None, None, False) == 5

    def test_specialized_or_other_is_conservative(self):
        assert deterministic_combine("specialized_or_other", None, False) == 3

    def test_hands_on_repair_no_years_is_optimistic(self):
        assert deterministic_combine("hands_on_repair_or_trade", None, False) == 9

    def test_hands_on_repair_one_year_is_borderline(self):
        assert deterministic_combine("hands_on_repair_or_trade", 1, False) == 7

    def test_hands_on_repair_two_plus_years_drops(self):
        assert deterministic_combine("hands_on_repair_or_trade", 2, False) == 5

    def test_cs_degree_caps_direct_match_families(self):
        assert deterministic_combine("customer_facing_or_sales", None, True) == 5

    def test_software_engineering_zero_years_is_entry_level(self):
        assert deterministic_combine("software_engineering", 0, False) == 7

    def test_software_engineering_unknown_years_is_uncertain_not_optimistic(self):
        """Real DevOps Engineer regression (decision #76): ownership/scope
        language implies seniority the years-regex can't see -- default
        must stay neutral, not assume entry-level."""
        assert deterministic_combine("software_engineering", None, False) == 5

    def test_software_engineering_years_required_drops_score(self):
        assert deterministic_combine("software_engineering", 3, False) == 3


class TestIsQuotaCooldownError:
    def test_matches_real_error_string(self):
        assert is_quota_cooldown_error("LLM error: All LLM providers are on quota cooldown (min wait: 3.2h).")

    def test_other_errors_not_matched(self):
        assert not is_quota_cooldown_error("LLM error: response did not contain a parseable SCORE line")

    def test_none_not_matched(self):
        assert not is_quota_cooldown_error(None)


class TestScoreJobDeterministic:
    def test_ineligible_job_short_circuits_without_llm_call(self):
        job = {"title": "X", "site": "Y", "full_description": "desc"}
        with (
            patch("applypilot.scoring.deterministic_fallback._check_ineligible", return_value="non-US role"),
            patch("applypilot.scoring.deterministic_fallback.classify_family") as mock_classify,
        ):
            result = score_job_deterministic(job, profile={})
        mock_classify.assert_not_called()
        assert result["score"] == 2
        assert "Ineligible" in result["reasoning"]

    def test_eligible_job_calls_family_classifier_and_tags_reasoning(self):
        job = {
            "title": "Maintenance Technician",
            "site": "Acme",
            "full_description": "General maintenance work.",
        }
        with (
            patch("applypilot.scoring.deterministic_fallback._check_ineligible", return_value=None),
            patch("applypilot.scoring.deterministic_fallback.local_only_client", return_value=object()),
            patch(
                "applypilot.scoring.deterministic_fallback.classify_family",
                return_value="hands_on_repair_or_trade",
            ),
            patch(
                "applypilot.scoring.deterministic_fallback.classify_compensation",
                return_value={"status": "unknown"},
            ),
        ):
            result = score_job_deterministic(job, profile={}, model="qwen3:1.7b")
        # base 9 from the family table, -2 compensation-unknown penalty
        assert result["score"] == 7
        assert "deterministic fallback" in result["reasoning"]
        assert "qwen3:1.7b" in result["reasoning"]


class TestAmbiguousTitleEscalation:
    """2026-09-07 (Future Work item 2): opt-in title-keyword escalation,
    built from real observed qwen3:1.7b-vs-8b disagreement titles at n=52
    (self-consistency was tried and rejected -- see the module docstring
    note). Off by default (escalate_model=None)."""

    @pytest.mark.parametrize(
        "title",
        [
            "Maintenance Technician",
            "Field Service Technician",
            "Mechanical Assembler 1",
            "Composites Technician",
            "Embedded C Software Engineer",
        ],
    )
    def test_real_disagreement_titles_match(self, title):
        assert _AMBIGUOUS_TITLE_RE.search(title)

    def test_ordinary_title_does_not_match(self):
        assert not _AMBIGUOUS_TITLE_RE.search("Software Engineer")

    def test_bare_technician_no_longer_escalates(self):
        """2026-09-08 (decision #81): "technician" was removed from the
        regex after a bootstrap re-validation (using only existing data,
        no new model calls) found it fired on 11/52 real titles but
        contributed ZERO unique disagreement catches -- every case it
        matched was also independently caught by a more specific
        alternative (composites/field service/maintenance/assembler/
        embedded). Verified removing it: escalation volume 17/52 -> 13/52
        with IDENTICAL hybrid gate agreement (84.6%, unchanged). Titles
        that are ONLY "technician" with no other real signal (e.g. "Field
        Technician", "Desktop/End User Support Technician") must no
        longer escalate; titles matching a still-real keyword must."""
        assert not _AMBIGUOUS_TITLE_RE.search("Field Technician")
        assert not _AMBIGUOUS_TITLE_RE.search("Desktop/End User Support Technician")
        assert not _AMBIGUOUS_TITLE_RE.search("Multi-Skilled Technician")
        # sanity: a title combining "technician" with a still-real keyword
        # must still match (via the other keyword, not "technician" itself).
        assert _AMBIGUOUS_TITLE_RE.search("Composites Technician")
        assert _AMBIGUOUS_TITLE_RE.search("Maintenance Technician")

    def test_no_escalate_model_uses_primary_model_regardless_of_title(self):
        job = {"title": "Maintenance Technician", "site": "Acme", "full_description": "desc"}
        with (
            patch("applypilot.scoring.deterministic_fallback._check_ineligible", return_value=None),
            patch("applypilot.scoring.deterministic_fallback.local_only_client", return_value=object()) as mock_client,
            patch("applypilot.scoring.deterministic_fallback.classify_family", return_value="hands_on_repair_or_trade"),
            patch("applypilot.scoring.deterministic_fallback.classify_compensation", return_value={"status": "stated"}),
        ):
            score_job_deterministic(job, profile={}, model="qwen3:1.7b")
        mock_client.assert_called_once_with("qwen3:1.7b")

    def test_matching_title_escalates_to_slow_model(self):
        job = {"title": "Maintenance Technician", "site": "Acme", "full_description": "desc"}
        with (
            patch("applypilot.scoring.deterministic_fallback._check_ineligible", return_value=None),
            patch("applypilot.scoring.deterministic_fallback.local_only_client", return_value=object()) as mock_client,
            patch("applypilot.scoring.deterministic_fallback.classify_family", return_value="hands_on_repair_or_trade"),
            patch("applypilot.scoring.deterministic_fallback.classify_compensation", return_value={"status": "stated"}),
        ):
            result = score_job_deterministic(job, profile={}, model="qwen3:1.7b", escalate_model="qwen3:8b")
        mock_client.assert_called_once_with("qwen3:8b")
        assert "escalated" in result["reasoning"]

    def test_non_matching_title_does_not_escalate_even_when_escalate_model_given(self):
        job = {"title": "Software Engineer", "site": "Acme", "full_description": "desc"}
        with (
            patch("applypilot.scoring.deterministic_fallback._check_ineligible", return_value=None),
            patch("applypilot.scoring.deterministic_fallback.local_only_client", return_value=object()) as mock_client,
            patch("applypilot.scoring.deterministic_fallback.classify_family", return_value="software_engineering"),
            patch("applypilot.scoring.deterministic_fallback.classify_compensation", return_value={"status": "stated"}),
        ):
            result = score_job_deterministic(job, profile={}, model="qwen3:1.7b", escalate_model="qwen3:8b")
        mock_client.assert_called_once_with("qwen3:1.7b")
        assert "escalated" not in result["reasoning"]


class TestRunDeterministicFallbackScoring:
    def test_only_quota_cooldown_stuck_jobs_are_scored(self, tmp_db, seed_job):
        conn = tmp_db()
        stuck = seed_job(
            conn,
            url_suffix="quota-stuck",
            title="Maintenance Technician",
            fit_score=None,
            score_error="LLM error: All LLM providers are on quota cooldown (min wait: 2.0h).",
            state="enriched",
        )
        other_error = seed_job(
            conn,
            url_suffix="other-error",
            title="Software Engineer",
            fit_score=None,
            score_error="LLM error: response did not contain a parseable SCORE line",
            state="enriched",
        )
        never_scored = seed_job(
            conn,
            url_suffix="never-scored",
            title="Software Engineer",
            fit_score=None,
            score_error=None,
            state="enriched",
        )

        with (
            patch("applypilot.scoring.deterministic_fallback._check_ineligible", return_value=None),
            patch("applypilot.scoring.deterministic_fallback.local_only_client", return_value=object()),
            patch(
                "applypilot.scoring.deterministic_fallback.classify_family",
                return_value="hands_on_repair_or_trade",
            ),
            patch(
                "applypilot.scoring.deterministic_fallback.classify_compensation",
                return_value={"status": "stated", "subtype": "annual"},
            ),
            patch("applypilot.config.load_profile", return_value={}),
        ):
            result = run_deterministic_fallback_scoring(conn=conn, model="qwen3:1.7b")

        assert result["candidates"] == 1
        assert result["scored"] == 1

        row = conn.execute("SELECT fit_score, score_method FROM jobs WHERE url = ?", (stuck["url"],)).fetchone()
        assert row["fit_score"] == 9
        assert row["score_method"] == SCORE_METHOD

        for untouched in (other_error, never_scored):
            row = conn.execute("SELECT fit_score, score_method FROM jobs WHERE url = ?", (untouched["url"],)).fetchone()
            assert row["fit_score"] is None
            assert row["score_method"] is None

    def test_no_candidates_is_a_clean_noop(self, tmp_db, seed_job):
        conn = tmp_db()
        seed_job(conn, url_suffix="clean", fit_score=9, score_error=None)

        with patch("applypilot.config.load_profile", return_value={}):
            result = run_deterministic_fallback_scoring(conn=conn)

        assert result["candidates"] == 0
        assert result["scored"] == 0


class TestRevalidateDeterministicFallbackScores:
    def test_scored_fallback_row_reset_for_real_rescoring(self, tmp_db, seed_job):
        conn = tmp_db()
        job = seed_job(
            conn,
            url_suffix="fallback-scored",
            fit_score=9,
            score_method=SCORE_METHOD,
            state="scored",
            scored_at="2026-09-07T00:00:00+00:00",
        )

        result = revalidate_deterministic_fallback_scores(conn=conn)

        assert result["matched"] == 1
        assert result["updated"] == 1
        row = conn.execute(
            "SELECT fit_score, score_method, state, scored_at FROM jobs WHERE url = ?", (job["url"],)
        ).fetchone()
        assert row["fit_score"] is None
        assert row["score_method"] is None
        assert row["scored_at"] is None
        assert row["state"] == "enriched"

    def test_archived_ineligible_fallback_row_untouched(self, tmp_db, seed_job):
        """An ineligible fallback verdict came from the same deterministic
        pre-filter the real LLM path runs first -- nothing to revalidate."""
        conn = tmp_db()
        job = seed_job(
            conn,
            url_suffix="fallback-archived",
            fit_score=2,
            score_method=SCORE_METHOD,
            state="archived",
        )

        result = revalidate_deterministic_fallback_scores(conn=conn)

        assert result["matched"] == 0
        row = conn.execute("SELECT fit_score, state FROM jobs WHERE url = ?", (job["url"],)).fetchone()
        assert row["fit_score"] == 2
        assert row["state"] == "archived"

    def test_real_llm_scored_row_untouched(self, tmp_db, seed_job):
        conn = tmp_db()
        job = seed_job(conn, url_suffix="real-llm-scored", fit_score=9, score_method=None, state="scored")

        result = revalidate_deterministic_fallback_scores(conn=conn)

        assert result["matched"] == 0
        row = conn.execute("SELECT fit_score FROM jobs WHERE url = ?", (job["url"],)).fetchone()
        assert row["fit_score"] == 9

    def test_dry_run_reports_without_modifying(self, tmp_db, seed_job):
        conn = tmp_db()
        job = seed_job(conn, url_suffix="dry-run-fallback", fit_score=7, score_method=SCORE_METHOD, state="low_score")

        result = revalidate_deterministic_fallback_scores(conn=conn, dry_run=True)

        assert result["matched"] == 1
        assert result["updated"] == 0
        row = conn.execute("SELECT fit_score, state FROM jobs WHERE url = ?", (job["url"],)).fetchone()
        assert row["fit_score"] == 7
        assert row["state"] == "low_score"
