"""Tests for the deterministic commission-only compensation hard gate in
scorer.py.

2026-08-28 investigation: no deterministic backstop existed anywhere for
commission-only compensation -- only prose in SCORE_PROMPT_TEMPLATE
instructing the LLM to score "100% commission/1099 structure" low,
entirely dependent on LLM adherence. Live DB calibration (24,325 described
jobs) found exactly ONE unambiguous commission-only posting ("New Mortgage
Loan Officer Turn Your Network Into Real Income"), sitting unscored, and
386+ legitimate base+commission mentions that must NOT be rejected. See
scorer.py's _COMMISSION_ONLY_PATTERN module comment for the full design
rationale (deliberately narrow: only explicit exclusivity language --
100%, only, solely, straight, no base/guaranteed -- never bare
"commission", and 1099 is never its own signal).
"""

from __future__ import annotations

import pytest

from applypilot.scoring.scorer import _check_ineligible


def _job(title="Sales Representative", location="Remote (US)", description="A full-time US-remote sales position."):
    return {"title": title, "location": location, "full_description": description}


# ── High-confidence positives: explicit exclusivity language ─────────────


@pytest.mark.parametrize(
    "description",
    [
        "This is a 100% commission business with real upside and no income caps.",
        "Compensation structure: 100%commission (no draw).",
        "This role is commission only.",
        "This is a commission-only position.",
        "Sales reps here work on straight commission.",
        "Reps are paid solely on commission.",
        "Reps are compensated solely by commission, no base.",
        "Compensation is commission only, uncapped earning potential.",
        "There is no base salary for this role -- 100% of pay comes from closed deals.",
        "This position offers no base pay; all income is performance-driven.",
        "There is no guaranteed salary in this role; you earn what you close.",
    ],
)
def test_commission_only_language_rejected(description):
    reason = _check_ineligible(_job(description=description))
    assert reason is not None
    assert "commission" in reason.lower()


def test_1099_with_commission_only_language_rejected():
    """1099 alone must never trigger (see negative controls below), but
    when it genuinely co-occurs with real commission-only wording the
    phrase is caught the same way any other commission-only phrase is --
    no separate 1099-combinator logic exists or is needed."""
    description = "1099 commission-only role -- no salary, no guarantees, just your book of business."
    reason = _check_ineligible(_job(description=description))
    assert reason is not None
    assert "commission" in reason.lower()


# ── Negative controls: legitimate base + commission ───────────────────────


@pytest.mark.parametrize(
    "description",
    [
        "Base salary of $50,000 plus uncapped commission.",
        "$20/hour plus commission.",
        "Competitive base + commission.",
        "Commission eligible after base salary.",
        "This role will include a 30% commission/variable along with base salary.",
        "Hourly compensation with incentivized commission structure.",
    ],
)
def test_base_plus_commission_not_rejected(description):
    assert _check_ineligible(_job(description=description)) is None


def test_commission_based_without_exclusivity_not_rejected():
    description = "This is a commission-based role with strong earning potential."
    assert _check_ineligible(_job(description=description)) is None


def test_commission_structure_without_exclusivity_not_rejected():
    description = "We offer an uncapped commission structure with a guaranteed income feature."
    assert _check_ineligible(_job(description=description)) is None


def test_uncapped_commission_without_exclusivity_not_rejected():
    description = "Uncapped commission with significant accelerators for exceeding targets."
    assert _check_ineligible(_job(description=description)) is None


# ── Negative controls: "commission" as a substring of "commissioning" ────


@pytest.mark.parametrize(
    "description",
    [
        "You will assemble, install, troubleshoot, test, commission, and service HVAC systems.",
        "Support equipment installations, upgrades, and decommissioning activities.",
        "This role is not commissioning-focused, not controls-programming-heavy.",
        "Own the commissioning of new manufacturing equipment and processes.",
    ],
)
def test_commissioning_engineering_sense_not_rejected(description):
    assert _check_ineligible(_job(title="Commissioning Engineer", description=description)) is None


# ── Negative controls: 1099 alone, no exclusivity language ───────────────


@pytest.mark.parametrize(
    "description",
    [
        "Paid hourly while on site. 1099 contractor position. Travel pay included.",
        "Contract Type: Independent Contractor-1099. Start Date: Immediate.",
        "Knowledge of AP controls, supplier onboarding, 1099 reporting and payment fraud prevention.",
        "W2 or 1099 options available depending on your preference.",
    ],
)
def test_1099_alone_not_rejected(description):
    assert _check_ineligible(_job(description=description)) is None


# ── Negative controls: commission mentioned about someone else / duties ──


def test_commission_as_a_job_duty_not_rejected():
    """A Payroll/People-Ops role whose DUTY is processing commissions for
    a sales team -- not a description of this job's own pay structure.
    Title is deliberately non-senior (no Manager/Director/Lead) so this
    test isolates the commission gate alone, not the unrelated seniority
    gate that also lives in _check_ineligible()."""
    description = (
        "Own end-to-end payroll operations including base salary changes, equity grants, "
        "bonus calculations, and commission processing, ensuring data integrity and compliance."
    )
    assert _check_ineligible(_job(title="Payroll Operations Coordinator", description=description)) is None


def test_candidates_prior_commission_experience_not_rejected():
    """Describes the CANDIDATE's past work environment, not this job's
    compensation structure -- must not be misread as a self-description."""
    description = (
        "You have operated in a commission or performance-incentivized sales environment, "
        "not just a salary-based one, and know how to manage a pipeline against a quota."
    )
    assert _check_ineligible(_job(description=description)) is None


def test_legitimate_compensation_boilerplate_not_rejected():
    """The single most common commission-adjacent string in the live
    corpus (1,778 rows share a close variant of this sentence) -- a
    standard corporate compensation-range disclosure that explicitly
    includes salary, not a commission-only posting."""
    description = (
        "Compensation ranges reflect salary and commission compensation (when applicable) "
        "across several geographic markets."
    )
    assert _check_ineligible(_job(description=description)) is None


# ── Buried disclosure: full_description is scanned, not just the first
# ── 6,000-character desc_head other checks use ───────────────────────────


def test_commission_only_language_buried_past_6000_chars_still_caught():
    padding = "This role involves cross-functional collaboration and stakeholder communication. " * 90
    assert len(padding) > 6000
    description = padding + "To be clear: this position is commission only, with no base salary whatsoever."
    reason = _check_ineligible(_job(description=description))
    assert reason is not None
    assert "commission" in reason.lower()


def test_ordinary_description_buried_past_6000_chars_not_rejected():
    """Positive control for the buried-disclosure test above: padding
    alone (no commission-only language anywhere) must not misfire."""
    padding = "This role involves cross-functional collaboration and stakeholder communication. " * 90
    assert len(padding) > 6000
    assert _check_ineligible(_job(description=padding)) is None


# ── Plain / vague mentions must not false-positive ────────────────────────


def test_plain_job_with_no_commission_mention_not_rejected():
    assert _check_ineligible(_job()) is None


def test_bare_commission_word_not_rejected():
    description = "This role includes a commission component in addition to standard pay."
    assert _check_ineligible(_job(description=description)) is None


# ── Regression: the real confirmed live example ───────────────────────────


def test_mortgage_loan_officer_live_regression_rejected():
    """Real description text (excerpt) from the one confirmed live
    commission-only posting found during the 2026-08-28 investigation:
    state='enriched', unscored -- about to reach the LLM scorer with zero
    guard before this gate."""
    description = (
        "ity relationships, anything that means you're not starting from zero contacts. "
        "This matters more than mortgage experience.\n"
        "-  You're a self-starter. Nobody is going to hand you a list of leads to dial. "
        "We teach you how to generate your own business, but you have to be someone who "
        "makes things happen.\n"
        "-  You're genuinely coachable. You'll be learning a new craft.\n"
        "-  You're ready to work. This is a 100% commission business with real upside and "
        "no income caps. The ceiling is high, but you have to climb.\n"
        "This is NOT for you if you're looking for a salary, a lead list handed to you, or "
        "a company that just collects new licensees and leaves them to sink."
    )
    reason = _check_ineligible(_job(title="New Mortgage Loan Officer Turn Your Network Into Real Income", description=description))
    assert reason is not None
    assert "commission" in reason.lower()


def test_mortgage_loan_officer_regression_never_reaches_llm():
    """The strongest guarantee, matching test_clearance_gate.py's
    equivalent test: the LLM client must never even be constructed for the
    disqualified job."""
    from unittest.mock import patch

    import applypilot.scoring.scorer as scorer_mod

    description = "This is a 100% commission business with real upside and no income caps."
    job = {
        "title": "New Mortgage Loan Officer Turn Your Network Into Real Income",
        "site": "Talent.com",
        "location": "Remote (US)",
        "full_description": description,
    }

    with patch.object(scorer_mod, "get_stage_client") as mock_get_client:
        result = scorer_mod.score_job(resume_text="Resume text.", job=job, profile={"education": []})

    mock_get_client.assert_not_called()
    assert result["score"] == 2
    assert result["eligibility"] == "non_us_only"  # reused generic reject signal, same as clearance/seniority
    assert "commission" in result["reasoning"].lower()
