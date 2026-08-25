"""Tests for the deterministic security-clearance hard gate in scorer.py.

2026-08-25 regression: "Software Development Engineer - ML Ops (US Federal)"
(a real Workday ML Runtime posting) scored fit_score=8 and reached
apply_failed -- an application was actually attempted -- despite the
description explicitly naming TS/SCI w/CI Poly. The scoring prompt's own
"clearance roles are OUT OF SCOPE, score 1-2" instruction was never applied;
there was no deterministic backstop for it the way geography and seniority
have one. See scorer.py's _TS_SCI_PATTERN / _CLEARANCE_REQUIRED_PATTERN
module comment for the two-tier design rationale (TS/SCI is disqualifying on
a bare mention; generic "clearance" / "Secret" / "Top Secret" requires an
explicit hard-requirement qualifier, since those tiers are often phrased as
"ability to obtain," which this gate deliberately does not treat as
disqualifying on its own).
"""

from __future__ import annotations

import pytest

from applypilot.scoring.scorer import _check_ineligible


def _job(title="Software Engineer", location="Remote (US)", description="A full-time US-remote position."):
    return {"title": title, "location": location, "full_description": description}


# ── TS/SCI: disqualifying on a bare mention, any framing ────────────────


@pytest.mark.parametrize(
    "description",
    [
        "Candidates must currently hold an active TS/SCI clearance.",
        "This position requires TS/SCI.",
        "An active TS/SCI is required to start.",
    ],
)
def test_ts_sci_required_rejected(description):
    reason = _check_ineligible(_job(description=description))
    assert reason is not None
    assert "clearance" in reason.lower()


def test_ts_sci_with_ci_poly_rejected():
    description = "This role requires an active TS/SCI with CI Poly clearance."
    assert _check_ineligible(_job(description=description)) is not None


def test_ts_sci_w_ci_poly_slash_rejected():
    description = "Candidates must hold TS/SCI w/CI Poly."
    assert _check_ineligible(_job(description=description)) is not None


def test_ts_sci_bare_mention_without_required_wording_still_rejected():
    """TS/SCI is disqualifying even when framed softly ('may require',
    'ability to obtain', 'preferred') -- this is the exact framing the real
    regression job used, and is why TS/SCI gets bare-mention treatment while
    generic clearance/Secret/Top-Secret do not (see module docstring)."""
    description = "This role may require a security clearance at the TS/SCI level. Preferred but not mandatory."
    reason = _check_ineligible(_job(description=description))
    assert reason is not None
    assert "clearance" in reason.lower()


# ── Top Secret / Secret / generic "active clearance": require an explicit
# ── hard-requirement qualifier, not just a bare mention ──────────────────


def test_top_secret_clearance_required_rejected():
    description = "Top Secret clearance is required for this position."
    assert _check_ineligible(_job(description=description)) is not None


def test_secret_clearance_required_rejected():
    description = "Secret clearance required."
    assert _check_ineligible(_job(description=description)) is not None


def test_active_security_clearance_required_rejected():
    description = "Active security clearance required to be considered for this role."
    assert _check_ineligible(_job(description=description)) is not None


def test_current_security_clearance_required_rejected():
    description = "Applicants must currently hold a current security clearance."
    assert _check_ineligible(_job(description=description)) is not None


def test_must_possess_active_clearance_rejected():
    description = "You must possess an active security clearance on day one."
    assert _check_ineligible(_job(description=description)) is not None


# ── Ability-to-obtain framing: NOT automatically classified as required ──


def test_ability_to_obtain_generic_clearance_not_rejected():
    """Per the existing policy's own ambiguity, 'ability to obtain' is
    deliberately not treated as an active-requirement signal for the
    generic/Secret tier -- many employers extend this to uncleared
    candidates."""
    description = "Applicants must have the ability to obtain and maintain a U.S. government issued security clearance."
    assert _check_ineligible(_job(description=description)) is None


def test_ability_to_obtain_secret_clearance_not_rejected():
    description = "Candidates should be eligible to obtain a Secret clearance if selected."
    assert _check_ineligible(_job(description=description)) is None


# ── Ordinary / vague mentions must not false-positive ────────────────────


def test_vague_company_boilerplate_not_rejected():
    description = (
        "We're a diverse team across many industries. Several of our teams work with "
        "government clients and some of those roles require clearance depending on the client."
    )
    assert _check_ineligible(_job(description=description)) is None


def test_unrelated_use_of_clearance_word_not_rejected():
    description = "All returned equipment goes through a customs clearance process before resale."
    assert _check_ineligible(_job(description=description)) is None


def test_clearance_sale_not_rejected():
    description = "Check out our clearance sale on refurbished laptops for the team."
    assert _check_ineligible(_job(description=description)) is None


def test_plain_job_with_no_clearance_mention_not_rejected():
    assert _check_ineligible(_job()) is None


# ── Regression: the real confirmed-bug job ────────────────────────────────


def test_workday_ml_ops_us_federal_regression_rejected():
    """Real description text (truncated) from the live regression job:
    fit_score=8, state=apply_failed -- an application was actually attempted
    against a role naming TS/SCI w/CI Poly, despite the scoring prompt's own
    'clearance roles are OUT OF SCOPE' rule."""
    description = (
        "Workday's Public Sector team will support one or more direct or indirect contracts "
        "with the U.S. Federal Government which, due to federal government security "
        "requirements, mandates that all Workday personnel working on the contracts be "
        "United States citizens (naturalized or native).\n\n"
        "About You\n\n"
        "This role may require a security clearance at the TS/SCI w/CI Poly level. Applicants "
        "must have the ability to obtain and maintain a U.S. government issued security "
        "clearance. An active TS/SCI w/CI Poly is preferred.\n\n"
        "We need creative and dedicated Software Engineers, like you, who really want to move "
        "the needle."
    )
    reason = _check_ineligible(_job(title="Software Development Engineer - ML Ops (US Federal)", description=description))
    assert reason is not None
    assert "clearance" in reason.lower()


def test_workday_ml_ops_regression_never_reaches_llm():
    """The strongest guarantee, matching the existing seniority-gate test
    pattern (TestScoreCannotBypassSeniority in test_eligibility.py): the LLM
    client must never even be constructed for the disqualified job, so no
    LLM response -- however high a score it might produce -- can reach the
    apply funnel."""
    from unittest.mock import patch

    import applypilot.scoring.scorer as scorer_mod

    description = (
        "This role may require a security clearance at the TS/SCI w/CI Poly level. Applicants "
        "must have the ability to obtain and maintain a U.S. government issued security "
        "clearance. An active TS/SCI w/CI Poly is preferred."
    )
    job = {
        "title": "Software Development Engineer - ML Ops (US Federal)",
        "site": "Workday",
        "location": "USA.VA.Reston",
        "full_description": description,
    }

    with patch.object(scorer_mod, "get_stage_client") as mock_get_client:
        result = scorer_mod.score_job(resume_text="Resume text.", job=job, profile={"education": []})

    mock_get_client.assert_not_called()
    assert result["score"] == 2
    assert result["eligibility"] == "non_us_only"  # reused generic reject signal, same as seniority
    assert "clearance" in result["reasoning"].lower()
