"""Tests for the pre-filter patterns in scorer.py.

Validated 2026-04-23 against 5,938 historical scored jobs, then revised
2026-08-18: the candidate profile has no professional software engineering
experience, so Junior/Entry-Level/New-Grad/Trainee/Apprentice/Graduate titles
are the actual target level, not noise to reject. Internships/co-ops remain
excluded via searches.yaml exclude_titles (candidate wants full-time work).
"""

from unittest.mock import patch

import pytest

from applypilot.scoring.scorer import _check_ineligible


def _job(title="Senior Software Engineer", location="Remote", description="US-remote position."):
    return {"title": title, "location": location, "full_description": description}


# ── Seniority: junior/entry-level titles are ELIGIBLE (candidate's real level) ──


def test_junior_title_not_rejected():
    assert _check_ineligible(_job(title="Junior Software Engineer")) is None


def test_intern_title_rejected():
    assert _check_ineligible(_job(title="Platform Engineering Intern")) is not None


def test_internship_title_rejected():
    assert _check_ineligible(_job(title="Software Engineering Internship")) is not None


def test_fresher_title_not_rejected():
    assert _check_ineligible(_job(title="Fresher Software Engineer")) is None


def test_entry_level_title_not_rejected():
    assert _check_ineligible(_job(title="Software Engineer I - Entry Level")) is None


def test_new_grad_title_not_rejected():
    assert _check_ineligible(_job(title="New Grad Software Engineer")) is None


def test_trainee_title_not_rejected():
    assert _check_ineligible(_job(title="Software Engineer Trainee")) is None


def test_apprentice_title_not_rejected():
    assert _check_ineligible(_job(title="Software Apprentice")) is None


# ── Sales-adjacency: occupation alone no longer blocks (2026-08-25) ──────
# Candidate policy: "take anything in my fields so long as it pays well and
# doesn't try to screw me." These occupations now reach the LLM, which
# evaluates them on actual requirements/competitiveness, not title alone.


def test_sales_engineer_not_blocked_by_title():
    assert _check_ineligible(_job(title="Sales Engineer")) is None


def test_solutions_engineer_not_blocked_by_title():
    assert _check_ineligible(_job(title="Solutions Engineer")) is None


def test_presales_not_blocked_by_title():
    assert _check_ineligible(_job(title="Presales Engineer")) is None


def test_customer_success_engineer_not_blocked_by_title():
    assert _check_ineligible(_job(title="Customer Success Engineer")) is None


def test_senior_sales_titles_still_blocked_by_seniority_not_occupation():
    """Removing the sales-occupation patterns must not weaken the separate,
    unrelated seniority gate -- these remain rejected, but via
    eligibility.seniority_disqualifier, not the (now-removed) sales patterns."""
    assert _check_ineligible(_job(title="Senior Sales Engineer")) is not None
    assert _check_ineligible(_job(title="Senior Presales Engineer")) is not None
    assert _check_ineligible(_job(title="Senior Customer Success Engineer")) is not None


# ── Additional safe patterns (validated 2026-04-23 round 2) ─────────


def test_graduate_title_not_rejected():
    assert _check_ineligible(_job(title="Graduate Developer")) is None
    assert _check_ineligible(_job(title="Graduate Software Engineer")) is None


def test_recruiter_title_not_blocked_by_occupation():
    assert _check_ineligible(_job(title="Technical Recruiter")) is None
    assert _check_ineligible(_job(title="Talent Acquisition Partner")) is None
    assert _check_ineligible(_job(title="Talent Scout")) is None
    assert _check_ineligible(_job(title="Talent Sourcer")) is None


def test_account_manager_title_not_blocked_by_occupation():
    assert _check_ineligible(_job(title="Enterprise Account Executive")) is None


# ── IC/customer-facing manager-title exception (2026-08-25 follow-up) ────
# The broad bare-"manager" seniority rule (decision #64) was investigated
# specifically for four industry-standard IC/customer-facing title families:
# Account Manager, Customer Success Manager, Technical Account Manager,
# Sales Manager. A real posting (Zillow/Aryeo "Technical Customer Success
# Manager, API") explicitly reads "you will own a portfolio of... customers"
# with no team-management language and only a 2+ year requirement -- exactly
# the case-by-case judgment the LLM scorer exists to make. See
# eligibility._IC_CUSTOMER_FACING_MANAGER_TITLES for the exact, deliberately
# narrow (not a general "manager" exemption) implementation.
#
# These tests mock load_search_config() to isolate the code-level exemption
# from this machine's live ~/.applypilot/searches.yaml, which independently
# ALSO lists "manager" in its own user-editable exclude_titles (a genuinely
# separate mechanism from eligibility.SENIORITY_TITLE_PATTERN -- see
# test_config_exclude_titles_is_a_separate_mechanism_not_touched_by_this_change
# below for the unmocked, live-environment-coupled proof that this second
# gate still exists and was deliberately left untouched, since modifying a
# user's config file is out of scope for this change).


def _cfg_no_manager_exclusion(exclude_titles=None):
    return {
        "exclude_titles": exclude_titles
        if exclude_titles is not None
        else ["senior", "sr.", "staff", "principal", "lead", "head", "director", "vp ", "chief", "intern", "co-op"]
    }


@patch("applypilot.scoring.scorer.load_search_config")
def test_account_manager_reaches_scoring(mock_cfg):
    """Proves the exception reaches the actual production scoring path
    (_check_ineligible), not just the eligibility.seniority_disqualifier
    regex in isolation -- per the explicit instruction that a title
    reaching scoring must not be silently re-rejected by another prefilter."""
    mock_cfg.return_value = _cfg_no_manager_exclusion()
    assert _check_ineligible(_job(title="Account Manager")) is None


@patch("applypilot.scoring.scorer.load_search_config")
def test_customer_success_manager_reaches_scoring(mock_cfg):
    mock_cfg.return_value = _cfg_no_manager_exclusion()
    assert _check_ineligible(_job(title="Customer Success Manager")) is None


@patch("applypilot.scoring.scorer.load_search_config")
def test_technical_account_manager_reaches_scoring(mock_cfg):
    mock_cfg.return_value = _cfg_no_manager_exclusion()
    assert _check_ineligible(_job(title="Technical Account Manager")) is None


@patch("applypilot.scoring.scorer.load_search_config")
def test_sales_manager_reaches_scoring(mock_cfg):
    mock_cfg.return_value = _cfg_no_manager_exclusion()
    assert _check_ineligible(_job(title="Sales Manager")) is None


@patch("applypilot.scoring.scorer.load_search_config")
def test_business_segment_modifiers_do_not_defeat_the_exemption(mock_cfg):
    """Regional/Strategic/Enterprise/Channel prefixes are not seniority
    signals -- the exact-four-family approach deliberately does not try to
    infer seniority from them (per the investigation's own guidance)."""
    mock_cfg.return_value = _cfg_no_manager_exclusion()
    assert _check_ineligible(_job(title="Regional Sales Manager")) is None
    assert _check_ineligible(_job(title="Enterprise Account Manager")) is None
    assert _check_ineligible(_job(title="Strategic Account Manager")) is None


@patch("applypilot.scoring.scorer.load_search_config")
def test_independently_senior_modifiers_still_block_the_exempt_families(mock_cfg):
    """The exemption only suppresses the bare "manager" token -- any OTHER
    SENIORITY_TITLE_PATTERN token anywhere in the title still disqualifies
    exactly as before, so a genuinely senior variant of an otherwise-exempt
    family is never let through."""
    mock_cfg.return_value = _cfg_no_manager_exclusion()
    assert _check_ineligible(_job(title="Senior Account Manager")) is not None
    assert _check_ineligible(_job(title="Senior Customer Success Manager")) is not None
    assert _check_ineligible(_job(title="Lead Technical Account Manager")) is not None
    assert _check_ineligible(_job(title="Director of Account Management")) is not None
    assert _check_ineligible(_job(title="Head of Customer Success")) is not None


@patch("applypilot.scoring.scorer.load_search_config")
def test_non_exempt_manager_families_remain_blocked(mock_cfg):
    """The exemption is exactly four families, not a general "manager"
    exemption -- Engineering/Product/Project/Program/Operations Manager
    (and bare "Manager" alone) remain fully disqualifying."""
    mock_cfg.return_value = _cfg_no_manager_exclusion()
    assert _check_ineligible(_job(title="Engineering Manager")) is not None
    assert _check_ineligible(_job(title="Product Manager")) is not None
    assert _check_ineligible(_job(title="Project Manager")) is not None
    assert _check_ineligible(_job(title="Program Manager")) is not None
    assert _check_ineligible(_job(title="Operations Manager")) is not None
    assert _check_ineligible(_job(title="Manager")) is not None


def test_config_exclude_titles_is_a_separate_mechanism_not_touched_by_this_change():
    """Unmocked (uses this machine's real ~/.applypilot/searches.yaml):
    documents that a genuinely separate, user-editable config layer
    (scorer.py's exclude_titles check, sourced from searches.yaml) may
    independently still exclude "manager" -- this is NOT the seniority gate
    this change modified, and editing a user's config file is out of scope
    here. If this assertion ever starts failing because the live config no
    longer lists "manager", that's a config change on this machine, not a
    regression in this exemption."""
    from applypilot.config import load_search_config

    cfg = load_search_config() or {}
    excluded = [str(t).strip().lower() for t in (cfg.get("exclude_titles") or [])]
    if "manager" in excluded:
        reason = _check_ineligible(_job(title="Account Manager"))
        assert reason is not None
        assert "search configuration" in reason.lower()


def test_senior_recruiter_and_account_titles_still_blocked_by_seniority():
    """Same non-weakening guarantee as the sales-adjacency case above."""
    assert _check_ineligible(_job(title="Senior Talent Acquisition Partner")) is not None
    assert _check_ineligible(_job(title="Senior Account Manager")) is not None


def test_designer_title_rejected():
    assert _check_ineligible(_job(title="Senior UX Designer")) is not None
    assert _check_ineligible(_job(title="Product Designer")) is not None
    assert _check_ineligible(_job(title="Graphic Designer")) is not None


def test_mobile_only_title_rejected():
    assert _check_ineligible(_job(title="Senior Android Engineer")) is not None
    assert _check_ineligible(_job(title="iOS Engineer")) is not None
    assert _check_ineligible(_job(title="Mobile Engineer")) is not None


def test_legacy_stack_title_rejected():
    assert _check_ineligible(_job(title="Salesforce Developer")) is not None
    assert _check_ineligible(_job(title="Apex Developer")) is not None
    assert _check_ineligible(_job(title="Mainframe Engineer (COBOL)")) is not None
    assert _check_ineligible(_job(title="Senior TIBCO Developer")) is not None


def test_senior_backend_python_rejected():
    """Seniority exclusions apply even when the technical stack matches."""
    assert _check_ineligible(_job(title="Senior Backend Engineer, Python")) is not None
    assert _check_ineligible(_job(title="Staff Platform Engineer - Go")) is not None
    assert _check_ineligible(_job(title="Principal Software Engineer")) is not None


# ── Regional sales tags ─────────────────────────────────────────────


def test_latam_in_title_rejected():
    # Rejected via the LATAM geography token, not the (now-removed) Account
    # Manager occupation token -- "Account Manager" alone no longer blocks.
    assert _check_ineligible(_job(title="Account Manager, LATAM")) is not None


def test_mena_in_title_rejected():
    assert _check_ineligible(_job(title="Engineering Lead, MENA")) is not None


def test_anz_in_title_rejected():
    assert _check_ineligible(_job(title="Senior DevOps Engineer, ANZ")) is not None


def test_nordics_in_title_rejected():
    assert _check_ineligible(_job(title="Staff Engineer - Nordics")) is not None


def test_only_hiring_in_title_rejected():
    assert _check_ineligible(_job(title="Only hiring in Vietnam | Senior Engineer")) is not None


# ── New non-US countries in location ────────────────────────────────


@pytest.mark.parametrize(
    "loc",
    [
        "Brazil Remote Work",
        "Remote — São Paulo, Brazil",
        "Mexico City, Mexico",
        "Buenos Aires, Argentina",
        "Vietnam",
        "Remote - Japan",
        "Thailand",
        "Philippines",
        "Jakarta, Indonesia",
        "Seoul, Korea",
        "Taipei, Taiwan",
        "Cairo, Egypt",
        "Nairobi, Kenya",
        "Johannesburg, South Africa",
        "Tel Aviv, Israel",
        "Istanbul, Turkey",
        "Lisbon, Portugal",
        "Dublin, Ireland",
        "Copenhagen, Denmark",
        "Stockholm, Sweden",
        "Oslo, Norway",
        "Helsinki, Finland",
        "Brussels, Belgium",
        "Zurich, Switzerland",
        "Vienna, Austria",
        "Bucharest, Romania",
        "Budapest, Hungary",
    ],
)
def test_non_us_country_in_location_rejected(loc):
    assert _check_ineligible(_job(location=loc)) is not None


# ── LLM rubric text (2026-08-25 sales/recruiting policy realignment) ─
# Regression guard: the rubric must not tell the LLM to score sales/
# recruiting/marketing/etc. low merely because of occupation.


def test_prompt_rubric_no_longer_auto_rejects_sales_recruiting_by_occupation():
    from applypilot.scoring.scorer import SCORE_PROMPT_TEMPLATE

    assert "recruiting, design, marketing, product" not in SCORE_PROMPT_TEMPLATE
    assert "not by title alone" in SCORE_PROMPT_TEMPLATE
    assert "SALES" in SCORE_PROMPT_TEMPLATE
    assert "RECRUITING" in SCORE_PROMPT_TEMPLATE


# ── Must NOT reject legit US roles ──────────────────────────────────


def test_senior_us_remote_rejected():
    assert _check_ineligible(_job(title="Senior Software Engineer", location="Remote (US)")) is not None


def test_staff_engineer_rejected():
    assert _check_ineligible(_job(title="Staff Software Engineer", location="Seattle, WA")) is not None


def test_principal_engineer_rejected():
    assert _check_ineligible(_job(title="Principal Platform Engineer", location="San Francisco, CA")) is not None


def test_us_role_with_global_office_mention_not_rejected():
    """A non-senior US role mentioning a global office should remain eligible.

    The description pre-filter is narrow — requires explicit regional restrictions
    like 'Remote (Europe)' or 'EMEA only', not a casual office mention.
    """
    desc = "We're a US-based company with offices in San Francisco, London, and Tokyo. "
    desc += "This role is US-remote. You'll work on distributed systems."
    assert _check_ineligible(_job(title="Software Engineer", description=desc)) is None


def test_associate_director_rejected():
    assert _check_ineligible(_job(title="Associate Director of Engineering")) is not None


# ── 2026-04-30: Buried-in-description non-US restrictions ───────────
#
# Twilio's Greenhouse postings (jobs/7662058 and 7767260) titled their
# roles "Senior Software Engineer" with location "Remote", but the
# country restriction was buried in the description body like:
#   "This role will be remote and based in the UK."
#   "This role will be remote and based in Ontario, BC or Alberta, Canada."
# The prior 800-char head-only scan missed those. Bumped to 6000 + new
# patterns. Each test below corresponds to a real failed-apply URL.


def test_twilio_uk_remote_buried_in_description_rejected():
    """Twilio jobs/7662058 — Senior SWE with UK location buried in desc."""
    desc = (
        "About Twilio: We are looking for an exceptional Senior Software "
        "Engineer to join our team. "
        * 30  # pushes the restriction past 800 chars
        + " The mission of this team is to build the foundational platform. "
        + "This role will be remote and based in the UK."
    )
    result = _check_ineligible(
        _job(
            title="Senior Software Engineer",
            location="Remote",
            description=desc,
        )
    )
    assert result is not None
    assert "non-US geography in description" in result


def test_twilio_canada_provinces_rejected():
    """Twilio jobs/7767260 — L3 SWE with Canada-province restriction."""
    desc = (
        "About Twilio: "
        + "blah " * 200
        + " This role will be remote and based in Ontario, British Columbia or Alberta, Canada."
    )
    result = _check_ineligible(
        _job(
            title="Software Engineer (L3) Infrastructure",
            location="Remote",
            description=desc,
        )
    )
    assert result is not None


def test_based_in_uk_rejected():
    """Variant: 'based in the UK' anywhere in first 6000 chars."""
    desc = "We are a global team. " * 50 + "Candidates must be based in the UK."
    assert _check_ineligible(_job(description=desc)) is not None


def test_right_to_work_uk_rejected():
    """JD mentions 'right to work in the UK' question."""
    desc = "Senior role. " * 100 + "You must have the right to work in the United Kingdom."
    assert _check_ineligible(_job(description=desc)) is not None


def test_ist_timezone_rejected():
    """India Standard Time requirement."""
    desc = "Engineering role. " * 50 + "Candidates must work IST timezone hours."
    assert _check_ineligible(_job(description=desc)) is not None


# ── _parse_score_response: ELIGIBILITY field extraction ─────────────


def test_parser_extracts_eligibility_eligible():
    from applypilot.scoring.scorer import _parse_score_response

    response = """ELIGIBILITY: eligible
SCORE: 9
KEYWORDS: Python, Go, Kubernetes
REASONING: Strong stack match, US remote role."""
    parsed = _parse_score_response(response)
    assert parsed["score"] == 9
    assert parsed["eligibility"] == "eligible"


def test_parser_extracts_eligibility_non_us_only():
    from applypilot.scoring.scorer import _parse_score_response

    response = """ELIGIBILITY: non_us_only
SCORE: 8
KEYWORDS: backend
REASONING: UK-only remote role."""
    parsed = _parse_score_response(response)
    assert parsed["eligibility"] == "non_us_only"


def test_parser_eligibility_with_brackets():
    """The prompt template shows [eligible|non_us_only] — make sure
    ``ELIGIBILITY: [non_us_only]`` (the LLM keeping the brackets) parses."""
    from applypilot.scoring.scorer import _parse_score_response

    response = "ELIGIBILITY: [non_us_only]\nSCORE: 7\nKEYWORDS: foo\nREASONING: bar"
    parsed = _parse_score_response(response)
    assert parsed["eligibility"] == "non_us_only"


def test_parser_missing_eligibility_defaults_to_eligible():
    """Older models that omit the field — default to eligible (no false bans)."""
    from applypilot.scoring.scorer import _parse_score_response

    response = "SCORE: 8\nKEYWORDS: foo\nREASONING: bar"
    parsed = _parse_score_response(response)
    assert parsed["eligibility"] == "eligible"


def test_parser_handles_markdown_bold_fields():
    """Regression test for 2026-08-18: the claude_cli fallback tier answers in
    markdown ("**SCORE: 1**") rather than plain lines. The old startswith()
    check missed every field entirely, silently defaulting score to 0 for 9 of
    25 jobs before this was caught. Also covers multi-paragraph reasoning with
    a markdown preamble before the four required fields, which claude_cli
    produces despite being told not to."""
    from applypilot.scoring.scorer import _parse_score_response

    response = (
        "I'll evaluate this job against the candidate's profile.\n\n"
        "## GEOGRAPHY CHECK\n\nNo restriction found.\n\n"
        "**ELIGIBILITY: eligible**\n\n"
        "**SCORE: 7**\n\n"
        "**KEYWORDS:** Help Desk, CompTIA A+, Troubleshooting\n\n"
        "**REASONING:** Strong match on certification and hands-on background."
    )
    parsed = _parse_score_response(response)
    assert parsed["score"] == 7
    assert parsed["eligibility"] == "eligible"
    assert parsed["keywords"] == "Help Desk, CompTIA A+, Troubleshooting"
    assert parsed["reasoning"] == "Strong match on certification and hands-on background."


# ── _parse_score_response: unparseable responses must yield score=None, ────
# ── never a fake 0 ───────────────────────────────────────────────────────
#
# 2026-08-25 fix: score used to default to 0 whenever no "SCORE: <digits>"
# line could be found -- indistinguishable from a real parsed score, since
# nothing downstream could tell "the LLM said 0" apart from "nothing was
# parseable". score_job()/_flush_score_batch's success/failure branching is
# keyed on `score is not None`, so a fake 0 was silently written to the DB
# as a successful fit_score=0 with score_error cleared. None now signals
# "could not parse" so the existing failure path (retry/backoff, eventually
# score_failed) handles it instead of a redesign.


def test_parser_empty_response_returns_none_score():
    from applypilot.scoring.scorer import _parse_score_response

    parsed = _parse_score_response("")
    assert parsed["score"] is None


def test_parser_whitespace_only_response_returns_none_score():
    from applypilot.scoring.scorer import _parse_score_response

    parsed = _parse_score_response("   \n\t  ")
    assert parsed["score"] is None


def test_parser_refusal_prose_returns_none_score():
    """A conversational refusal/non-conforming answer with no SCORE: line
    at all -- the model ignored the output-format instruction entirely."""
    from applypilot.scoring.scorer import _parse_score_response

    parsed = _parse_score_response("I cannot assist with evaluating this job posting.")
    assert parsed["score"] is None


def test_parser_malformed_score_value_returns_none_score():
    """SCORE: present but its value isn't digits -- the regex requires
    \\d+, so this must not match and fall through to None, not silently
    coerce/crash."""
    from applypilot.scoring.scorer import _parse_score_response

    parsed = _parse_score_response("SCORE: not_a_number\nREASONING: unclear")
    assert parsed["score"] is None


def test_parser_truncated_response_before_score_line_returns_none_score():
    """A response cut off mid-generation before ever reaching the SCORE:
    line (e.g. token budget exhausted) -- must not silently default."""
    from applypilot.scoring.scorer import _parse_score_response

    parsed = _parse_score_response(
        "Let me evaluate this candidate against the job requirements in detail. "
        "First, looking at the technical stack overlap between the resume and "
        "the posting, I notice several relevant"  # cut off, no SCORE: line ever appears
    )
    assert parsed["score"] is None


def test_parser_legitimate_score_zero_is_impossible_by_contract():
    """Documents an important invariant relied on by the None-means-
    unparseable fix: the prompt's own contract is "SCORE: [1-10]", and the
    parser clamps any matched digit into that range (max(1, min(10, ...))),
    so a literal "SCORE: 0" in the raw text is clamped UP to 1, never left
    as 0. Under the current contract, 0 can never be a legitimately parsed
    score -- confirming the None sentinel doesn't collide with any real
    value this parser can produce."""
    from applypilot.scoring.scorer import _parse_score_response

    parsed = _parse_score_response("SCORE: 0\nREASONING: literal zero in the text")
    assert parsed["score"] == 1


def test_parser_legitimate_nonzero_score_is_unaffected():
    from applypilot.scoring.scorer import _parse_score_response

    parsed = _parse_score_response("ELIGIBILITY: eligible\nSCORE: 3\nKEYWORDS: none\nREASONING: weak fit.")
    assert parsed["score"] == 3


# ── Ethical exclusions (defense/weapons/surveillance/policing) ──────────
# Regression test for the 2026-08-18 bug: detail.py's ethical-keyword loader
# called a config function that didn't exist and was never even called, so
# exclude_description_keywords (military/defense contractor/weapons/etc.) was
# silently a no-op end to end. Enforcement now lives in _check_ineligible.


def test_defense_contractor_description_rejected(monkeypatch):
    import applypilot.scoring.scorer as scorer_mod

    monkeypatch.setattr(
        scorer_mod,
        "load_search_config",
        lambda: {"exclude_description_keywords": ["defense contractor", "autonomous weapons"]},
    )
    job = _job(
        title="Software Engineer",
        description="We are a leading defense contractor building next-generation systems.",
    )
    assert scorer_mod._check_ineligible(job) is not None


def test_non_defense_description_not_rejected_by_ethical_filter(monkeypatch):
    import applypilot.scoring.scorer as scorer_mod

    monkeypatch.setattr(
        scorer_mod,
        "load_search_config",
        lambda: {"exclude_description_keywords": ["defense contractor", "autonomous weapons"]},
    )
    job = _job(title="Software Engineer", description="We build SaaS billing software.")
    assert scorer_mod._check_ineligible(job) is None


# ── Ethics config policy-critical fallback (2026-08-27 fix) ─────────────
# Regression tests for the audit's P0-2 finding: a 2026-08-25 rewrite of the
# live ~/.applypilot/searches.yaml dropped the exclude_description_keywords
# key entirely (not an explicit []), silently disabling the whole
# military/weapons/surveillance/policing exclusion policy end to end (289
# ethical_exclusion archives on 2026-08-18, none since). load_search_config()
# only falls back to the packaged example when searches.yaml is *absent*
# entirely, so a present-but-incomplete file never triggered that fallback.


def test_ethics_keywords_fall_back_to_packaged_defaults_when_key_absent(monkeypatch):
    import applypilot.scoring.scorer as scorer_mod

    # A live config that exists and has other keys, but genuinely omits
    # exclude_description_keywords -- the exact 2026-08-25 failure mode.
    monkeypatch.setattr(scorer_mod, "load_search_config", lambda: {"exclude_titles": ["senior"]})
    monkeypatch.setattr(
        scorer_mod,
        "load_packaged_default_search_config",
        lambda: {"exclude_description_keywords": ["defense contractor", "autonomous weapons"]},
    )
    job = _job(title="Software Engineer", description="We are a leading defense contractor.")
    reason = scorer_mod._check_ineligible(job)
    assert reason is not None
    assert "ethical exclusion" in reason.lower()


def test_ethics_keywords_explicit_empty_list_is_honored_not_backfilled(monkeypatch):
    """An explicit exclude_description_keywords: [] means the key IS
    present -- a deliberate "disable this policy" choice -- and must NOT be
    treated the same as "key absent" / silently backfilled from packaged
    defaults."""
    import applypilot.scoring.scorer as scorer_mod

    monkeypatch.setattr(scorer_mod, "load_search_config", lambda: {"exclude_description_keywords": []})
    monkeypatch.setattr(
        scorer_mod,
        "load_packaged_default_search_config",
        lambda: {"exclude_description_keywords": ["defense contractor"]},
    )
    job = _job(title="Software Engineer", description="We are a leading defense contractor.")
    assert scorer_mod._check_ineligible(job) is None


def test_ethics_keywords_active_through_real_config_loading_path(monkeypatch):
    """Unmocked load_search_config/load_packaged_default_search_config --
    proves the fallback is wired through the REAL config-loading functions,
    not just reachable in isolation. Uses a job title with no other
    disqualifying signal so only the ethics check can be responsible for
    the result."""
    from applypilot.config import load_packaged_default_search_config, load_search_config
    from applypilot.scoring.scorer import _check_ineligible

    live_cfg = load_search_config() or {}
    if "exclude_description_keywords" in live_cfg:
        pytest.skip("live searches.yaml explicitly sets exclude_description_keywords; nothing to fall back from")

    packaged = load_packaged_default_search_config().get("exclude_description_keywords") or []
    if not packaged:
        pytest.skip("packaged searches.example.yaml has no exclude_description_keywords to fall back to")

    job = _job(title="Software Engineer", description="We are a leading defense contractor.")
    reason = _check_ineligible(job)
    assert reason is not None
    assert "ethical exclusion" in reason.lower()


def test_ethics_keyword_bare_dod_does_not_match_inside_unrelated_word(monkeypatch):
    """Audit finding: a bare "dod" substring check would match inside
    unrelated words like "Dodge" (a common vehicle brand -- relevant given
    this candidate's automotive/warehouse-adjacent work history). Matching
    must be word-boundary, not substring."""
    import applypilot.scoring.scorer as scorer_mod

    monkeypatch.setattr(scorer_mod, "load_search_config", lambda: {"exclude_description_keywords": ["dod"]})
    job = _job(
        title="Parts Specialist",
        description="Experience with Dodge, Chrysler, and Jeep vehicles preferred.",
    )
    assert scorer_mod._check_ineligible(job) is None


def test_ethics_keyword_bare_dod_still_matches_standalone_token(monkeypatch):
    import applypilot.scoring.scorer as scorer_mod

    monkeypatch.setattr(scorer_mod, "load_search_config", lambda: {"exclude_description_keywords": ["dod"]})
    job = _job(title="Software Engineer", description="Must have DoD contracting experience.")
    reason = scorer_mod._check_ineligible(job)
    assert reason is not None
    assert "ethical exclusion" in reason.lower()


def test_packaged_default_ethics_keywords_do_not_include_bare_military():
    """2026-08-27 real-data finding, not hypothetical: live-DB measurement
    against 24,325 described jobs found bare "military" was the ONLY
    matching ethics keyword on 5,017 rows, and a random sample of those was
    ~92% ordinary EEO/non-discrimination boilerplate ("...disability
    status, military veteran status, sexual orientation...") or veteran-
    hiring language on non-defense postings -- including an ordinary
    Greensboro, NC "Deskside Support Technician" listing, exactly the class
    of role this candidate wants preserved. Genuine defense employers
    remain caught by the many more specific terms (defense industry/
    contractor, dod, department of defense, missile, warfighter, combat
    systems, etc.), which showed no comparable false-positive pattern in
    the same measurement. The packaged default list is the fallback source
    scorer._check_ineligible uses when a user's searches.yaml omits
    exclude_description_keywords entirely -- it must not reintroduce this
    false-positive term."""
    from applypilot.config import load_packaged_default_search_config

    keywords = [
        str(k).strip().lower()
        for k in (load_packaged_default_search_config().get("exclude_description_keywords") or [])
    ]
    assert "military" not in keywords


def test_eeo_boilerplate_military_veteran_clause_not_ethics_excluded(monkeypatch):
    """Direct regression for the real Accenture/linkedin boilerplate found
    in the live DB: a job whose only "military" mention is the standard
    EEO non-discrimination clause must not be archived as a defense/
    weapons/surveillance/policing exclusion."""
    import applypilot.scoring.scorer as scorer_mod

    monkeypatch.setattr(
        scorer_mod,
        "load_search_config",
        lambda: {
            "exclude_description_keywords": [
                k for k in scorer_mod.load_packaged_default_search_config()["exclude_description_keywords"]
            ]
        },
    )
    job = _job(
        title="Deskside Support Technician - Greensboro, NC",
        location="Greensboro, NC",
        description=(
            "We are an equal opportunity employer. All employment decisions shall be made without "
            "regard to age, race, creed, color, religion, sex, national origin, ancestry, disability "
            "status, military veteran status, sexual orientation, gender identity or expression, "
            "genetic information, marital status, citizenship status or any other basis as protected "
            "by federal, state, or local law."
        ),
    )
    assert scorer_mod._check_ineligible(job) is None


# ── Advanced-degree requirement (candidate has a Bachelor's only) ───────
# Regression test for the 2026-08-19 incident: a PayPal "Software Engineer"
# posting (plain title, no seniority signal) required a Master's degree in
# its description body. Scored 8 under the pre-a6f72a6 rubric and sat in
# ready_to_apply with a tailored resume/cover letter already generated.

_NO_ADVANCED_DEGREE_PROFILE = {
    "education": [{"official_degree": "Bachelor of Arts in Media Studies"}],
}
_MASTERS_PROFILE = {
    "education": [{"official_degree": "Master of Science in Computer Science"}],
}


def test_masters_required_curly_apostrophe_rejected():
    # Real-world scrape artifact: source HTML used a curly apostrophe (U+2019),
    # not straight ASCII -- the regex must handle both.
    job = _job(
        title="Software Engineer",
        description="Minimum Requirements: Master’s degree, or foreign equivalent, "
        "in Computer Science, Engineering, or a closely related field.",
    )
    assert _check_ineligible(job, _NO_ADVANCED_DEGREE_PROFILE) is not None


def test_masters_required_straight_apostrophe_rejected():
    job = _job(
        title="Software Engineer",
        description="Required Qualifications: Master's degree in Computer Science required.",
    )
    assert _check_ineligible(job, _NO_ADVANCED_DEGREE_PROFILE) is not None


def test_masters_preferred_not_rejected():
    # "Preferred" is not a hard requirement -- must not fire.
    job = _job(
        title="Software Engineer",
        description="Bachelor's degree required; Master's degree preferred.",
    )
    assert _check_ineligible(job, _NO_ADVANCED_DEGREE_PROFILE) is None


def test_bachelors_or_masters_not_rejected():
    # Either degree clears the bar -- must not fire.
    job = _job(
        title="Software Engineer",
        description="Requires a Bachelor's or Master's degree in a related field.",
    )
    assert _check_ineligible(job, _NO_ADVANCED_DEGREE_PROFILE) is None


def test_masters_required_but_candidate_has_one_not_rejected():
    job = _job(
        title="Software Engineer",
        description="Minimum Requirements: Master's degree, or foreign equivalent, in Computer Science.",
    )
    assert _check_ineligible(job, _MASTERS_PROFILE) is None


def test_masters_required_without_profile_not_rejected():
    # profile=None (the default) preserves prior behavior -- this guard only
    # activates when a profile is actually supplied.
    job = _job(
        title="Software Engineer",
        description="Minimum Requirements: Master's degree, or foreign equivalent, in Computer Science.",
    )
    assert _check_ineligible(job) is None
