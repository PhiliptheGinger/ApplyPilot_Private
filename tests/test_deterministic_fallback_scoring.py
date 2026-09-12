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
    _CLINICAL_LICENSE_TITLE_RE,
    classify_family,
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

    def test_explicit_range_takes_lower_bound(self):
        """2026-09-09: real Truist "Wealth Support Specialist I" posting --
        "Entry Level / 1 - 3 years in the role" (en dash) was extracting
        the UPPER bound (3), under-crediting a candidate who meets the
        range's actual floor. Must take the lower bound, consistent with
        this function's own "smallest qualifying number" design intent."""
        text = "Requirements:\n1. Entry Level / 1 – 3 years in the role as a Support Specialist."
        assert extract_years_required(text) == 1

    def test_explicit_range_with_hyphen_and_plus_takes_lower_bound(self):
        text = "Required Qualifications:\n- 2-3+ years of professional experience"
        assert extract_years_required(text) == 2

    def test_decimal_years_rounds_up(self):
        """2026-09-09 (decision #88/#89): real Affirm "Software Engineer
        II" posting -- "What We Look For\n- You have a total of 1.5+
        years of experience as a software engineer" -- the old
        \\d{1,2}-only pattern matched nothing at this position at all.
        Rounds UP (ceil), not down: a 1.5-year bar is closer to "almost 2
        years" than "just 1". Also exercises the "What We Look For"
        header, a second real gap found while verifying THIS fix against
        the real posting (that specific header wasn't recognized either)."""
        text = "What We Look For\n- You have a total of 1.5+ years of experience as a software engineer."
        assert extract_years_required(text) == 2

    def test_decimal_years_exact_half_year_rounds_up_from_whole(self):
        text = "Requirements:\n- 3.5 years of professional experience required"
        assert extract_years_required(text) == 4

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

    def test_spelled_out_number_with_parenthetical_digit_counts(self):
        """2026-09-09: real Clear Street "Backend Software Engineer"
        regression, found via a real Gemini-vs-local comparison run --
        "at least eight (8) years of professional experience" has no
        digit directly followed by whitespace+"years" (")" sits in
        between), so the old regex never matched it at all."""
        text = "Requirements:\n- You have at least eight (8) years of professional experience implementing services."
        assert extract_years_required(text) == 8

    def test_plus_years_without_experience_word_counts(self):
        """2026-09-09: real Cash App "Software Engineer" regression --
        "5+ years working on complex systems" has no "experience" word
        anywhere nearby, which the old regex hard-required. A bare "N+"
        is itself a strong enough signal without it."""
        text = "You Have\n- 5+ years working on complex systems and delivering quality software."
        assert extract_years_required(text) == 5

    def test_bare_years_under_required_header_counts_without_any_verb_list(self):
        """2026-09-09, revised design (user request: generalize beyond an
        ever-growing inventory of experience-indicating verbs): a bare
        "N years <anything>" now qualifies purely via required-section
        anchoring, with no "+" or "experience" word needed at all -- real
        Cash App case, "working on" is just one of infinitely many real
        verb phrasings this must not depend on enumerating."""
        text = "Requirements:\n- 5 years working on complex systems."
        assert extract_years_required(text) == 5

    def test_bare_years_with_no_required_signal_at_all_still_returns_none(self):
        """The generalization above only fires when a required-section
        header or inline required-context phrase is present -- a bare
        "N years" with NEITHER (no header, no "required"/"must have"/
        "minimum of" nearby) must still return None."""
        text = "This role involves working with data pipelines. 5 years is a nice round number."
        assert extract_years_required(text) is None

    def test_yrs_abbreviation_counts(self):
        """Real postings commonly abbreviate "years" as "yrs"."""
        text = "Required Qualifications:\n- 5 yrs of professional experience"
        assert extract_years_required(text) == 5

    def test_spelled_out_number_word_counts(self):
        """2026-09-09: real, recurring Truist "Part Time Universal Banker"
        template (multiple branch postings, same text) -- "Two years of
        teller or cash handling or client service experience" -- no digit
        at all, so no numeric pattern could ever have matched."""
        text = "Required Qualifications:\n1. High school diploma\n2. Two years of teller or cash handling experience"
        assert extract_years_required(text) == 2

    def test_or_more_years_counts(self):
        """2026-09-09: real Boeing "Systems Engineer (Comm & Networks)"
        posting -- "Level 4: 9 or more years of related work experience" --
        this exact job previously scored 9/10 despite explicitly being a
        Senior/multi-tier role requiring 3-14+ years, because neither "or
        more" (only literal "+") nor the compound Boeing header format
        (see test_boeing_compound_qualifications_header below) matched."""
        text = "Basic Qualifications (Required Skills/Experience):\n9 or more years of related work experience"
        assert extract_years_required(text) == 9

    def test_minimum_years_without_of_counts(self):
        """2026-09-09: real RTX "Supplier Performance Specialist" posting
        -- "minimum 5 years prior relevant experience" -- the old
        _REQUIRED_CONTEXT_RE only recognized the literal phrase "minimum
        of", missing this extremely common variant entirely."""
        text = "Typically requires: a degree and minimum 5 years prior relevant experience."
        assert extract_years_required(text) == 5

    def test_minimum_years_requires_tight_proximity_not_a_distant_header(self):
        """2026-09-09: real false positive caught before shipping -- a
        naive first fix (adding bare "minimum" to the shared, wide-window
        _REQUIRED_CONTEXT_RE) let a real Sherwin-Williams "Minimum
        Requirements:" SECTION HEADER (an unrelated line) vouch for a
        completely different, unrelated "eighteen (18) years of age"
        mention 60+ chars away. "minimum N years" must require "minimum"
        immediately before the number, not just present somewhere in a
        wide symmetric window."""
        text = "Minimum Requirements:\n- Must be at least eighteen (18) years of age\n- Legally authorized to work in the US"
        assert extract_years_required(text) is None

    def test_years_of_age_never_counts_as_experience(self):
        """A second, independent safety net for the same real regression
        -- "years of age" must never count as an experience requirement
        regardless of any proximity coincidence with "minimum"/"required"."""
        text = "Requirements:\n- Must be at least eighteen (18) years of age\n- 5 years of professional experience required"
        assert extract_years_required(text) == 5

    def test_years_or_older_never_counts_as_experience(self):
        """2026-09-09 (decision #92): a real Avionics Technician posting
        phrases the same universal minimum-age requirement as "Must be 18
        years or older" rather than "18 years of age" -- must not count."""
        text = "Required Qualifications\n- Must be 18 years or older\n- High school diploma or equivalent"
        assert extract_years_required(text) is None

    def test_about_company_header_ends_required_section(self):
        """2026-09-09 (decision #92): a real Rooms To Go "Furniture
        Service Tech" posting has no "Preferred Qualifications"-style
        closing header, so the 1500-char fallback window swept in an
        unrelated "About Rooms To Go" company-history blurb ("Founded in
        1991... More than 30 years later...") sitting right after the
        requirements bullet list -- "30 years" wrongly counted as an
        experience requirement. An "About <Company>" header (not "About
        You", which is itself a required-section start header) must end
        the required-context zone."""
        text = (
            "What We're Looking For\n"
            "- A clean driving record\n"
            "- Self-motivated\n"
            "About Rooms To Go\n"
            "Founded in 1991. More than 30 years later and now America's #1 furniture retailer."
        )
        assert extract_years_required(text) is None

    def test_full_time_education_years_never_counts_as_experience(self):
        """2026-09-09 (decision #91 follow-up): real Accenture India/
        Philippines template, "A 15 years full time education is
        required" -- an EDUCATION-duration marker (India's convention
        for "equivalent of a bachelor's degree"), not a professional-
        experience requirement, but the literal word "required" sits
        right next to it. Must not count even though it satisfies the
        inline required-context check."""
        text = "Minimum 3 year(s) of experience is required\nEducational Qualification : BTech\nA 15 years full time education is required."
        assert extract_years_required(text) == 3

    def test_boeing_compound_qualifications_header_counts(self):
        """2026-09-09: real Boeing template repeated across many postings
        in the same batch -- "Basic Qualifications (Required Skills and
        Experience):" -- never matched the old header pattern, which
        required the header to be JUST "Basic Qualifications" with
        nothing else on the line."""
        text = "Basic Qualifications (Required Skills and Experience):\n5+ years work-related experience with a Bachelors degree"
        assert extract_years_required(text) == 5

    def test_markdown_bold_wrapped_header_counts(self):
        """2026-09-09 (decision #92): real SunTech Medical "Technical
        Support Repair Technician" posting -- its "Minimum Qualifications"
        header was rendered as "**Minimum Qualifications  \\n  \\n**", a
        Markdown-to-text conversion artifact (bold markers + hard-line-
        break trailing spaces splitting the closing "**" onto its own
        line). Must still be recognized as a required-section header."""
        text = "**Minimum Qualifications  \n  \n**\n\n  * 3+ years working in electronics manufacturing facility"
        assert extract_years_required(text) == 3

    def test_what_were_looking_for_header_counts(self):
        """Distinct from the already-handled bare "What We Look For" --
        real Accenture "Infra & Cloud Technical Coordinator" posting uses
        "What We're Looking For" (with the apostrophe-'re)."""
        text = "What We're Looking For\nAt least 5 years of experience in infrastructure projects."
        assert extract_years_required(text) == 5

    def test_typically_requires_header_counts(self):
        text = "Typically requires:\n8 years of relevant industry experience."
        assert extract_years_required(text) == 8

    def test_hyphenated_years_separator_counts(self):
        text = "Requirements:\n- 5-years of professional experience required"
        assert extract_years_required(text) == 5

    def test_license_renewal_years_mention_inside_required_section_excluded(self):
        """2026-09-09: generalizing to bare "N years" (no verb-list
        dependency) opens a real false-positive class the old
        "...experience" requirement blocked structurally -- a years
        mention that has nothing to do with professional experience at
        all (license renewal cadence here) sitting inside an otherwise-
        recognized required-qualifications section. A short, closed
        negative list catches this instead of trying to positively
        enumerate every real experience-verb phrasing."""
        text = (
            "Requirements:\n"
            "- Valid driver's license, renewed within the last 2 years\n"
            "- 5 years of professional experience in the field\n"
        )
        assert extract_years_required(text) == 5

    def test_alternative_credential_noun_not_treated_as_non_experience_context(self):
        """2026-09-09: real Boeing "Machine Repair Mechanic" false-
        NEGATIVE caught before shipping -- a first version of the negative
        list included bare "certificat\\w*"/"licen[cs]\\w*" as triggers,
        which wrongly excluded this entirely valid requirement: "1+ years
        of related experience OR A COMPLETED TECHNICAL CERTIFICATE / post-
        secondary degree" uses "certificate" as an alternative-credential
        noun in the SAME bullet as the real years requirement, not as a
        renewal-cadence signal. The negative list must key on the
        renewal/validity-period framing (renew/valid for/expire/...), not
        the bare credential noun."""
        text = (
            "Qualifications You Must Have\n\n"
            "· High School or General Equivalency Diploma\n\n"
            "· 1+ years of related experience or a completed technical certificate "
            "/ post-secondary degree in a related field\n\n"
            "Qualifications We Prefer\n\n"
            "· Work experience"
        )
        assert extract_years_required(text) == 1

    def test_non_experience_context_check_is_scoped_to_own_line(self):
        """The negative-context check must only look at the mention's OWN
        line, not a flat character radius -- otherwise a genuinely
        qualifying mention on one bullet can be wrongly excluded just
        because an unrelated bullet nearby (e.g. a license-renewal
        mention) happens to fall within a flat character window."""
        text = "Requirements:\n- 5 years of professional experience required\n- valid license, renewed every 2 years\n"
        assert extract_years_required(text) == 5

    def test_inline_required_context_does_not_bleed_from_next_line_header(self):
        """2026-09-09: real AMG Tech Support "IT and Telecom Field
        Technicians" posting -- "...OR 5+ years verifiable field
        experience in I.T./Telecom\\nRequired Equipment & Qualifications\\n..."
        -- both mentions sit under a "Preferred Skills & Experience"
        header with no real hard requirement, but the word "Required"
        starting an entirely different, unrelated section (equipment, not
        experience) on the very NEXT line wrongly vouched for the "5+
        years" mention via the old flat +60-char trailing window. The
        inline required-context window's trailing side is now bounded to
        the mention's own line."""
        text = (
            "Preferred Skills & Experience\n"
            "- At least 1 year of I.T. or Telecom experience, and one of the following:\n"
            "- A+ Certification\n"
            "- OR 5+ years verifiable field experience in I.T./Telecom\n"
            "Required Equipment & Qualifications\n"
            "- Reliable personal vehicle\n"
        )
        assert extract_years_required(text) is None

    def test_same_line_trailing_required_still_counts(self):
        """Regression guard for the fix above -- "required" on the SAME
        line as the mention (the most common real phrasing, "X years of
        experience required") must still count."""
        text = "5 years of professional experience required."
        assert extract_years_required(text) == 5

    def test_requirements_colon_header_counts(self):
        """2026-09-09: real Clear Street case -- "Requirements:" as its
        own line, no "Qualifications" wording at all."""
        text = "Requirements:\n- 8 years of professional experience required in the field."
        assert extract_years_required(text) == 8

    def test_about_you_header_counts(self):
        """2026-09-09: real Vercel "Software Engineer, Workflows" case --
        "About You:" as its own line."""
        text = "About You:\n- You have at least 5+ years of relevant experience"
        assert extract_years_required(text) == 5

    def test_you_have_header_without_colon_counts(self):
        """2026-09-09: real Cash App case -- "You Have" as its own line,
        with no trailing colon at all."""
        text = "You Have\n- 5+ years working on complex systems and delivering quality software."
        assert extract_years_required(text) == 5

    def test_bare_qualifications_header_without_colon_counts(self):
        """2026-09-09: real BuiltIn "Software Engineer, Front-End" case --
        a bare "Qualifications" header (no Minimum/Required/Basic prefix,
        no colon)."""
        text = "Qualifications\n-\n7 years of experience in front-end development required."
        assert extract_years_required(text) == 7

    def test_requirements_as_ordinary_prose_noun_not_treated_as_header(self):
        """The new bare-word headers ("requirements"/"qualifications")
        must not fire just because the word appears mid-sentence in
        ordinary prose -- only when it's the ENTIRE content of its own
        line, matching how a real section header is formatted."""
        text = "There are no formal requirements for this role.\n5 years of tenure earns a bonus."
        assert extract_years_required(text) is None

    def test_no_length_limit_past_old_6000_char_window(self):
        """2026-09-09: real BuiltIn "Software Engineer, Front-End"
        regression -- its bare "Qualifications" header + "7 years of
        experience" sits at character 3196 of a 6208-char posting (a long
        company-intro/responsibilities section precedes it), past the old
        3000-char cutoff every extraction function in this module used.
        Rather than just raising the cutoff again, removed it entirely --
        this is a pure regex scan, not an LLM call, so there's no cost
        reason to truncate (unlike classify_family's real LLM prompt,
        which stays bounded at DESCRIPTION_WINDOW). Proven with text well
        past even the old 6000-char bump, not just past the original
        3000."""
        text = ("filler. " * 1000) + "Qualifications\n- 7 years of experience in the field required."
        assert len(text) > 6000
        assert extract_years_required(text) == 7


class TestExtractCsDegreeRequired:
    def test_cs_degree_required_detected(self):
        assert extract_cs_degree_required("Bachelor's degree in Computer Science required.")

    def test_unrelated_degree_not_flagged(self):
        assert not extract_cs_degree_required("Bachelor's degree in Business Administration required.")

    def test_no_degree_mention(self):
        assert not extract_cs_degree_required("No degree required, just enthusiasm.")

    def test_degree_under_minimum_qualifications_header_counts(self):
        """2026-09-09: real Cisco "Cloud Engineer" regression found via a
        manual accuracy spot-check of the 2026-09-08 backlog run -- the
        posting states "Minimum Qualifications\n\nBachelor's degree in
        computer science, Computer Engineering, or a related technical
        field" with no inline "required" phrase near it, the same
        section-header gap decision #82 already fixed for
        extract_years_required but never ported to this sibling function.
        The job scored a false-positive 9/10 instead of being caught by
        the degree gate."""
        text = (
            "Minimum Qualifications\n\n"
            "Bachelor’s degree in computer science, Computer Engineering, "
            "or a related technical field.\n\n"
            "Preferred Qualifications\n\nExperience with Kubernetes."
        )
        assert extract_cs_degree_required(text)

    def test_degree_under_preferred_header_not_counted(self):
        text = (
            "Required Qualifications\n- Strong communication skills\n\n"
            "Preferred Qualifications\n- Bachelor's degree in Computer Science"
        )
        assert not extract_cs_degree_required(text)

    def test_curly_apostrophe_inline_required_detected(self):
        """The old regex's straight-quote-only ``'?`` silently failed to
        match a Unicode right-single-quote apostrophe (the character real
        scraped postings actually use), breaking the \\s+ boundary right
        after "Bachelor" -- a pre-existing bug independent of the
        section-header gap above, caught by the same real Cisco text."""
        assert extract_cs_degree_required("A Bachelor’s degree in Computer Science is required.")


class TestClinicalLicenseTitleOverride:
    """2026-09-09: real regression found via a manual accuracy spot-check
    of the 2026-09-08 backlog run -- a Sana "Physician" posting and an
    Eagle Family Medicine "Medical Assistant or LPN" posting both got
    FAMILY: customer_facing_or_sales (score 9) from qwen3:1.7b, despite
    _FAMILY_SYSTEM already explicitly listing "clinical work" under
    specialized_or_other -- the model didn't reliably honor its own
    instruction against patient/compassion language that reads like
    customer-service vocabulary. Fixed with a deterministic override that
    bypasses the LLM call entirely for unambiguous licensed-clinical
    titles."""

    def test_physician_title_matches_override(self):
        assert _CLINICAL_LICENSE_TITLE_RE.search("Physician")

    def test_medical_assistant_or_lpn_title_matches_override(self):
        assert _CLINICAL_LICENSE_TITLE_RE.search("Medical Assistant or LPN - Eagle Family Medicine @ Triad")

    def test_unrelated_title_does_not_match(self):
        assert not _CLINICAL_LICENSE_TITLE_RE.search("Desktop Support Technician")

    def test_classify_family_short_circuits_without_llm_call(self):
        job = {"title": "Physician", "site": "Sana", "full_description": "Primary care physician role."}
        result = classify_family(client=object(), job=job)
        assert result == "specialized_or_other"

    def test_classify_family_llm_path_unaffected_for_normal_titles(self):
        job = {"title": "Desktop Support Technician", "site": "Acme", "full_description": "Fix computers."}
        mock_client = type("C", (), {"chat": lambda self, *a, **k: "FAMILY: it_or_tech_support"})()
        result = classify_family(client=mock_client, job=job)
        assert result == "it_or_tech_support"


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

    def test_software_engineering_one_year_lands_in_5_6_band(self):
        """2026-09-09 (decision #89): real n=27 Claude-vs-Gemini comparison
        (decision #88) found this branch used to flatten every years>=1
        to a single score of 3, but the written rubric's own prose (and
        both Claude's and Gemini's real independent scores) put "~1 year,
        rest learnable" in the 5-6 band, not 3-4."""
        assert deterministic_combine("software_engineering", 1, False) == 5

    def test_software_engineering_two_years_lands_in_3_4_band(self):
        assert deterministic_combine("software_engineering", 2, False) == 3

    def test_software_engineering_three_plus_years_lands_in_1_2_band(self):
        """Real n=27 evidence: every 5+/6+/7/8/10+-year software posting
        scored a 1 by both Claude and Gemini independently -- the rubric's
        own "1-2: ...3+ years..." band, which this lookup table had never
        actually implemented (it used to return a flat 3 regardless of
        whether the requirement was 1 year or 10)."""
        assert deterministic_combine("software_engineering", 3, False) == 1
        assert deterministic_combine("software_engineering", 8, False) == 1


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

    def test_scope_all_unscored_touches_every_unscored_enriched_job(self, tmp_db, seed_job):
        """2026-09-11 (decision #129): scope='all_unscored' is a deliberate
        widening of the default quota_cooldown-only scope -- confirms it
        touches jobs the default scope explicitly leaves alone (never
        attempted at all, or failed for a differently-worded reason)."""
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
            score_error="LLM error: All models exhausted after trying: ['gemini-3.6-flash'].",
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
            result = run_deterministic_fallback_scoring(conn=conn, model="qwen3:1.7b", scope="all_unscored")

        assert result["candidates"] == 3
        assert result["scored"] == 3

        for touched in (stuck, other_error, never_scored):
            row = conn.execute("SELECT fit_score, score_method FROM jobs WHERE url = ?", (touched["url"],)).fetchone()
            assert row["fit_score"] is not None
            assert row["score_method"] == SCORE_METHOD

    def test_unknown_scope_raises(self, tmp_db, seed_job):
        conn = tmp_db()
        with pytest.raises(ValueError, match="Unknown scope"):
            run_deterministic_fallback_scoring(conn=tmp_db(), scope="bogus")

    def test_no_candidates_is_a_clean_noop(self, tmp_db, seed_job):
        conn = tmp_db()
        seed_job(conn, url_suffix="clean", fit_score=9, score_error=None)

        with patch("applypilot.config.load_profile", return_value={}):
            result = run_deterministic_fallback_scoring(conn=conn)

        assert result["candidates"] == 0
        assert result["scored"] == 0

    def test_fast_and_escalate_jobs_are_grouped_not_interleaved(self, tmp_db, seed_job):
        """2026-09-09: real log timing from the 2026-09-08 backlog run
        showed escalated (qwen3:8b) calls costing 139-420s EACH, recurring
        throughout the run rather than paying a one-time warmup cost --
        consistent with Ollama evicting/reloading a model every time the
        requested model changes on this machine. The DB query returns jobs
        in whatever order they were inserted, which can interleave
        ambiguous-title (escalated) jobs among ordinary ones; this test
        seeds them interleaved and asserts every ``model`` call happens
        before every ``escalate_model`` call, regardless of DB order."""
        conn = tmp_db()
        # Interleaved on purpose: fast, escalate, fast, escalate, fast.
        titles = [
            "Software Engineer",  # fast
            "Maintenance Technician",  # escalate
            "Desktop Support Technician",  # fast
            "Composites Technician",  # escalate
            "Systems Administrator",  # fast
        ]
        for i, title in enumerate(titles):
            seed_job(
                conn,
                url_suffix=f"job-{i}",
                title=title,
                fit_score=None,
                score_error="LLM error: All LLM providers are on quota cooldown (min wait: 1.0h).",
                state="enriched",
            )

        model_call_order: list[str] = []

        def fake_local_only_client(model: str):
            model_call_order.append(model)
            return object()

        with (
            patch("applypilot.scoring.deterministic_fallback._check_ineligible", return_value=None),
            patch("applypilot.scoring.deterministic_fallback.local_only_client", side_effect=fake_local_only_client),
            patch("applypilot.scoring.deterministic_fallback.classify_family", return_value="specialized_or_other"),
            patch(
                "applypilot.scoring.deterministic_fallback.classify_compensation",
                return_value={"status": "unknown"},
            ),
            patch("applypilot.config.load_profile", return_value={}),
        ):
            result = run_deterministic_fallback_scoring(
                conn=conn, model="qwen3:1.7b", escalate_model="qwen3:8b", limit=0
            )

        assert result["scored"] == 5
        assert model_call_order == ["qwen3:1.7b"] * 3 + ["qwen3:8b"] * 2

    def test_no_escalate_model_processes_in_original_order(self, tmp_db, seed_job):
        """Without escalate_model, there's nothing to group -- jobs must
        still process in the DB's own order, not silently reordered."""
        conn = tmp_db()
        for i, title in enumerate(["Maintenance Technician", "Software Engineer", "Composites Technician"]):
            seed_job(
                conn,
                url_suffix=f"job-{i}",
                title=title,
                fit_score=None,
                score_error="LLM error: All LLM providers are on quota cooldown (min wait: 1.0h).",
                state="enriched",
            )

        scored_order: list[str] = []

        def fake_flush(conn, batch, now, score_method=None):
            scored_order.append(batch[0]["url"])

        with (
            patch("applypilot.scoring.deterministic_fallback._check_ineligible", return_value=None),
            patch("applypilot.scoring.deterministic_fallback.local_only_client", return_value=object()),
            patch("applypilot.scoring.deterministic_fallback.classify_family", return_value="specialized_or_other"),
            patch(
                "applypilot.scoring.deterministic_fallback.classify_compensation",
                return_value={"status": "unknown"},
            ),
            patch("applypilot.scoring.deterministic_fallback._flush_score_batch", side_effect=fake_flush),
            patch("applypilot.config.load_profile", return_value={}),
        ):
            run_deterministic_fallback_scoring(conn=conn, model="qwen3:1.7b", limit=0)

        assert scored_order == [f"https://example.com/job/job-{i}" for i in range(3)]


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
