"""Tests for the 2026-08-30 compensation classification/estimation layer
(``scoring.compensation``) and its integration into ``scorer.score_job()``.

Follow-up to the 2026-08-29/30 compensation investigations: the existing
commission-only gate (``scorer._COMMISSION_ONLY_PATTERN`` /
``_check_ineligible``) was found to correctly reject genuine commission-only
postings and correctly allow base+commission/ordinary salaried postings --
but a posting with NO compensation information at all was completely
invisible to scoring, neither rejected nor flagged. This module adds a
deterministic classification ("stated"/"estimated"/"unknown"/
"explicitly_absent") and a small, explainable scoring adjustment for the
"unknown"/"estimated" cases, without touching the commission-only gate at
all -- ``classify_compensation`` imports and reuses the exact same
``_COMMISSION_ONLY_PATTERN`` scorer.py already uses, so there is only ever
one source of truth for "explicitly_absent".
"""

from __future__ import annotations

from unittest.mock import patch

from applypilot.scoring.compensation import (
    classify_compensation,
    compensation_score_adjustment,
)


def _job(title="IT Support Specialist", site="SomeCorp", salary=None, description="", url="https://example.com/job"):
    return {"title": title, "site": site, "salary": salary, "full_description": description, "url": url}


# ── Stated compensation (requirements 1-3, 6-8) ──────────────────────────


class TestStatedClassification:
    def test_explicit_annual_range(self):
        result = classify_compensation(_job(description="$60,000-$75,000/year"))
        assert result["status"] == "stated"
        assert result["subtype"] == "annual"
        assert result["stated_min"] == 60000
        assert result["stated_max"] == 75000
        assert result["stated_unit"] == "annual"

    def test_explicit_hourly(self):
        result = classify_compensation(_job(description="$20/hour"))
        assert result["status"] == "stated"
        assert result["subtype"] == "hourly"
        assert result["stated_min"] == 20
        assert result["stated_max"] == 20

    def test_range_with_dollar_sign_only_on_first_number(self):
        """Real 2026-08-30 finding: Intel's own posting formats a range as
        "$60,450.00-85,340.00" -- only the first number carries "$". The
        second side must still be read as the range's upper bound, not
        dropped (which previously collapsed this to a misleading single
        point value of $60,450)."""
        result = classify_compensation(_job(description="$60,450.00-85,340.00 USD"))
        assert result["status"] == "stated"
        assert result["stated_min"] == 60450
        assert result["stated_max"] == 85340

    def test_intel_annual_salary_mislabeled_hourly_role_is_read_as_annual(self):
        """Exact real production text (Intel, "Module Equipment Technician
        - Metrology"): the posting's own template labels the SAME figure
        both "Annual Salary Range" and "(Hourly Role)" in one sentence.
        $60,450-$85,340 is nowhere near a plausible hourly wage -- must
        resolve to annual, not "$60,450.00/hour"."""
        description = (
            "Annual Salary Range for jobs which could be performed in the US: "
            "$60,450.00-85,340.00 USD (Hourly Role)"
        )
        result = classify_compensation(_job(description=description))
        assert result["status"] == "stated"
        assert result["stated_unit"] == "annual"
        assert result["stated_min"] == 60450
        assert result["stated_max"] == 85340

    def test_genuine_hourly_rate_near_the_plausibility_ceiling_still_works(self):
        """Contrast case for the Intel fix's plausibility ceiling: a real,
        modestly-paid hourly rate mentioned alongside the bare word
        "hourly" elsewhere in the same posting must still classify as
        hourly -- the ceiling only rejects figures nowhere near a
        realistic wage, it must not become suspicious of ordinary ones."""
        result = classify_compensation(
            _job(description="This is an hourly position. Pay: $45.00/hour.")
        )
        assert result["status"] == "stated"
        assert result["subtype"] == "hourly"
        assert result["stated_min"] == 45

    def test_base_plus_commission(self):
        result = classify_compensation(_job(description="Starting base salary $45K + commission"))
        assert result["status"] == "stated"
        assert result["subtype"] == "base_plus_commission"
        assert result["stated_min"] == 45000

    def test_base_plus_bonus(self):
        result = classify_compensation(_job(description="Base salary $80,000 plus annual performance bonus"))
        assert result["status"] == "stated"
        assert result["subtype"] == "base_plus_bonus"

    def test_standalone_signon_bonus_is_not_stated_compensation(self):
        """Real 2026-08-29 finding: a one-time sign-on bonus figure is not
        base pay and must not be read as one -- with no other pay
        information in the posting, this must classify as unknown, not
        "$3,000/year"."""
        result = classify_compensation(
            _job(description="External candidates will receive a sign-on bonus of $3,000.")
        )
        assert result["status"] == "unknown"

    def test_real_base_pay_still_found_alongside_a_signon_bonus_mention(self):
        """Contrast case: a genuine base salary figure elsewhere in the
        same posting must still be found even when a sign-on bonus is
        mentioned nearby -- only the bonus figure itself is excluded."""
        result = classify_compensation(
            _job(description="Base salary $80,000 plus a $3,000 sign-on bonus.")
        )
        assert result["status"] == "stated"
        assert result["stated_min"] == 80000
        assert result["stated_max"] == 80000

    def test_vague_phrase_is_not_stated(self):
        """'competitive salary' has no number to anchor on -- must not be
        treated as confirmed compensation."""
        for phrase in ["competitive salary", "generous compensation", "attractive package", "market-rate pay"]:
            result = classify_compensation(_job(description=f"We offer a {phrase} and great benefits."))
            assert result["status"] == "unknown", phrase

    def test_no_compensation_anywhere_is_unknown(self):
        result = classify_compensation(_job(description="A job doing things for a customer."))
        assert result["status"] == "unknown"

    def test_ambiguous_commission_heavy_language_with_real_base_not_explicitly_absent(self):
        """Real 2026-08-29 finding (AT&T Field Sales Rep posting): heavy
        repeated 'uncapped commission' language must not be mistaken for
        commission-only when the posting also states real base pay."""
        description = (
            "Our new Field Sales Representatives earn between $60,300 to $100,000, "
            "including the hourly rate and our uncapped commission opportunities."
        )
        result = classify_compensation(_job(description=description))
        assert result["status"] == "stated"
        assert result["subtype"] == "base_plus_commission"


# ── Explicitly absent (requirements 4-5) -- delegates to the existing gate ─


class TestExplicitlyAbsentDelegatesToExistingGate:
    def test_100_percent_commission(self):
        result = classify_compensation(_job(description="This is a 100% commission business with no income caps."))
        assert result["status"] == "explicitly_absent"

    def test_commission_only_wording(self):
        result = classify_compensation(_job(description="This role is commission only."))
        assert result["status"] == "explicitly_absent"

    def test_base_plus_commission_never_misclassified_as_absent(self):
        """Negative control paired with the two positives above."""
        result = classify_compensation(_job(description="Base salary of $50,000 plus uncapped commission."))
        assert result["status"] != "explicitly_absent"


# ── Estimation (requirements 9-10) ────────────────────────────────────────


class TestEstimation:
    def test_sufficient_same_employer_evidence_produces_estimate(self, tmp_db, seed_job):
        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="comp1",
            title="IT Help Desk Specialist",
            site="First Bank",
            salary=None,
            full_description="Pay: $40,000 annually. Help desk role.",
        )
        seed_job(
            conn,
            url_suffix="comp2",
            title="IT Help Desk Specialist II",
            site="First Bank",
            salary=None,
            full_description="Pay: $45,000 annually. Help desk role.",
        )
        target = seed_job(
            conn,
            url_suffix="target",
            title="IT Help Desk Specialist",
            site="First Bank",
            salary=None,
            full_description="No pay stated in this posting.",
        )

        result = classify_compensation(dict(target), conn=conn)
        assert result["status"] == "estimated"
        assert result["estimated_unit"] == "annual"
        assert result["estimated_min"] == 40000
        assert result["estimated_max"] == 45000
        assert result["estimated_confidence"] == "medium"
        assert result["estimate_basis"]

    def test_insufficient_evidence_stays_unknown_no_fabricated_estimate(self, tmp_db, seed_job):
        """Only ONE comparable exists -- below MIN_COMPARABLES_FOR_ESTIMATE
        (2) -- must abstain rather than guess from a single data point."""
        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="onlyone",
            title="IT Help Desk Specialist",
            site="First Bank",
            salary=None,
            full_description="Pay: $40,000 annually.",
        )
        target = seed_job(
            conn,
            url_suffix="target2",
            title="IT Help Desk Specialist",
            site="First Bank",
            salary=None,
            full_description="No pay stated in this posting.",
        )
        result = classify_compensation(dict(target), conn=conn)
        assert result["status"] == "unknown"
        assert "estimated_min" not in result

    def test_aggregator_site_never_produces_an_estimate(self, tmp_db, seed_job):
        """Even with 3+ same-title comparables, a known aggregator site
        (LinkedIn here) is never treated as an employer identity -- `site`
        for those rows is the platform, not the employer, and `company` is
        essentially never populated for them in the live database either
        (confirmed during the 2026-08-30 investigation)."""
        conn = tmp_db()
        for i in range(3):
            seed_job(
                conn,
                url_suffix=f"agg{i}",
                title="IT Help Desk Specialist",
                site="linkedin",
                salary=None,
                full_description=f"Pay: ${40000 + i * 1000} annually.",
            )
        target = seed_job(
            conn,
            url_suffix="agg_target",
            title="IT Help Desk Specialist",
            site="linkedin",
            salary=None,
            full_description="No pay stated.",
        )
        result = classify_compensation(dict(target), conn=conn)
        assert result["status"] == "unknown"

    def test_no_conn_skips_estimation_and_stays_unknown(self):
        """Cheap callers (e.g. the dashboard, classifying many rows per
        render) pass conn=None deliberately to avoid a DB query per row."""
        result = classify_compensation(_job(description="No pay stated."), conn=None)
        assert result["status"] == "unknown"

    def test_one_time_signon_bonus_never_pollutes_a_comparable(self, tmp_db, seed_job):
        """Real 2026-08-29 production bug: a Raytheon 'External candidates
        will receive a sign-on bonus of $3,000' posting was misread as
        stated annual pay, and its irrelevant $3,000 figure combined with
        a genuine $48,176-$90,571 Avionics Technician comparable to
        produce a nonsensical $3,000-$90,571 "estimate" for an unrelated
        Apprentice Avionics Technician posting. Only ONE genuine
        comparable remains once the bonus figure is correctly excluded --
        below MIN_COMPARABLES_FOR_ESTIMATE (2) -- so this must now abstain
        to "unknown", not produce any range at all."""
        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="bonus_only",
            title="Assembly Solder Technician III",
            site="Raytheon (RTX)",
            salary=None,
            full_description="External candidates will receive a sign-on bonus of $3,000. Assemble and repair units.",
        )
        seed_job(
            conn,
            url_suffix="genuine_comparable",
            title="Avionics Electronics Technician II",
            site="Raytheon (RTX)",
            salary=None,
            full_description="Pay: $48,176 - $90,571 annually. Avionics technician role.",
        )
        target = seed_job(
            conn,
            url_suffix="apprentice_target",
            title="Apprentice Avionics Technician",
            site="Raytheon (RTX)",
            salary=None,
            full_description="No pay stated. Avionics technician apprenticeship.",
        )
        result = classify_compensation(dict(target), conn=conn)
        assert result["status"] == "unknown"

    def test_incoherent_wide_spread_abstains_even_with_enough_comparables(self, tmp_db, seed_job):
        """Real 2026-08-29 production bug (contrast case): even when every
        contributing number is individually a real, correctly-parsed
        figure (not corrupted data), a spread this wide (30x) is not a
        defensible "estimate" at any confidence level -- must abstain
        rather than publish it with a confidence label attached."""
        conn = tmp_db()
        seed_job(
            conn, url_suffix="wide1", title="Software Engineer, Team A", site="BigCo",
            salary=None, full_description="Pay: $30,000 annually.",
        )
        seed_job(
            conn, url_suffix="wide2", title="Software Engineer, Team B", site="BigCo",
            salary=None, full_description="Pay: $900,000 annually.",
        )
        target = seed_job(
            conn, url_suffix="wide_target", title="Software Engineer, Team C", site="BigCo",
            salary=None, full_description="No pay stated.",
        )
        result = classify_compensation(dict(target), conn=conn)
        assert result["status"] == "unknown"

    def test_genuinely_wide_but_plausible_spread_still_produces_an_estimate(self, tmp_db, seed_job):
        """Contrast case for the ceiling above: a real ~2x spread (e.g.
        OpenAI's actual live Software Engineer postings, which legitimately
        range from $230K to $490K across different roles) must still
        surface as a usable medium-confidence estimate -- the ceiling
        exists to catch incoherent noise, not ordinary real-world variance."""
        conn = tmp_db()
        seed_job(
            conn, url_suffix="plausible1", title="Software Engineer, Ad Formats", site="BigCo",
            salary=None, full_description="Pay: $230,000 annually.",
        )
        seed_job(
            conn, url_suffix="plausible2", title="Software Engineer, API SDK", site="BigCo",
            salary=None, full_description="Pay: $293,000 - $385,000 annually.",
        )
        target = seed_job(
            conn, url_suffix="plausible_target", title="Software Engineer, Delivery", site="BigCo",
            salary=None, full_description="No pay stated.",
        )
        result = classify_compensation(dict(target), conn=conn)
        assert result["status"] == "estimated"
        assert result["estimated_confidence"] == "medium"


# ── Score adjustment (deterministic, small, explainable) ─────────────────


class TestCompensationScoreAdjustment:
    def test_stated_has_no_adjustment(self):
        adjustment, note = compensation_score_adjustment({"status": "stated"})
        assert adjustment == 0
        assert note == ""

    def test_unknown_has_a_small_penalty_and_an_uncertainty_not_quality_message(self):
        """The note must frame this as ApplyPilot's own uncertainty, never
        as a claim about the job's actual pay quality -- it may reference
        the phrase "pays poorly" only to explicitly DENY it (matching the
        brief's own example), never assert it."""
        adjustment, note = compensation_score_adjustment({"status": "unknown"})
        assert adjustment < 0
        assert "uncertainty" in note.lower()
        assert "not a claim that the job pays poorly" in note.lower()

    def test_estimated_high_confidence_has_no_or_tiny_penalty(self):
        comp = {
            "status": "estimated",
            "estimated_min": 50000,
            "estimated_max": 55000,
            "estimated_unit": "annual",
            "estimated_confidence": "high",
            "estimate_basis": "5 comparables",
        }
        adjustment, note = compensation_score_adjustment(comp)
        assert adjustment == 0
        assert "estimated" in note.lower()
        assert "not employer-stated" in note.lower()

    def test_estimated_medium_confidence_penalty_smaller_than_unknown(self):
        comp = {
            "status": "estimated",
            "estimated_min": 50000,
            "estimated_max": 55000,
            "estimated_unit": "annual",
            "estimated_confidence": "medium",
            "estimate_basis": "2 comparables",
        }
        estimated_adjustment, _ = compensation_score_adjustment(comp)
        unknown_adjustment, _ = compensation_score_adjustment({"status": "unknown"})
        assert estimated_adjustment < 0
        assert estimated_adjustment > unknown_adjustment

    def test_explicitly_absent_has_no_adjustment_here(self):
        """The deterministic gate already rejects this case before any
        score exists to adjust -- this function is never actually reached
        for "explicitly_absent" in the real score_job() flow, but must not
        do anything harmful if called directly with that status."""
        adjustment, _note = compensation_score_adjustment({"status": "explicitly_absent"})
        assert adjustment == 0


# ── score_job() integration ────────────────────────────────────────────


class TestScoreJobIntegration:
    def _mock_llm(self, response="ELIGIBILITY: eligible\nSCORE: 8\nKEYWORDS: help desk\nREASONING: Strong fit."):
        from unittest.mock import MagicMock

        import applypilot.scoring.scorer as scorer_mod

        patcher = patch.object(scorer_mod, "get_stage_client")
        mock_get_client = patcher.start()
        mock_client = MagicMock()
        mock_client.chat.return_value = response
        mock_get_client.return_value = mock_client
        return patcher

    def test_stated_compensation_no_penalty(self):
        import applypilot.scoring.scorer as scorer_mod

        job = {
            "title": "IT Help Desk Analyst",
            "site": "SomeCorp",
            "location": "Remote (US)",
            "salary": None,
            "full_description": "Pay: $50,000-$60,000 annually. Help desk role.",
            "url": "https://example.com/j1",
        }
        patcher = self._mock_llm()
        try:
            result = scorer_mod.score_job(resume_text="Resume text.", job=job, profile={"education": []})
        finally:
            patcher.stop()

        assert result["score"] == 8
        assert result["compensation"]["status"] == "stated"
        assert result["reasoning"] == "Strong fit."

    def test_unknown_compensation_gets_penalized_and_annotated(self):
        import applypilot.scoring.scorer as scorer_mod

        job = {
            "title": "IT Help Desk Analyst",
            "site": "SomeCorp",
            "location": "Remote (US)",
            "salary": None,
            "full_description": "Provide help desk support. No pay stated in this posting.",
            "url": "https://example.com/j2",
        }
        patcher = self._mock_llm()
        try:
            result = scorer_mod.score_job(resume_text="Resume text.", job=job, profile={"education": []})
        finally:
            patcher.stop()

        assert result["score"] < 8
        assert result["compensation"]["status"] == "unknown"
        assert "uncertainty" in result["reasoning"].lower()
        assert result["eligibility"] == "eligible"

    def test_commission_only_still_never_reaches_llm(self):
        """Eligibility requirement: commission-only jobs cannot proceed --
        must remain true after this change, exactly as before it."""
        import applypilot.scoring.scorer as scorer_mod

        job = {
            "title": "New Mortgage Loan Officer",
            "site": "Talent.com",
            "location": "Remote (US)",
            "salary": None,
            "full_description": "This is a 100% commission business with real upside and no income caps.",
            "url": "https://example.com/j3",
        }
        with patch.object(scorer_mod, "get_stage_client") as mock_get_client:
            result = scorer_mod.score_job(resume_text="Resume text.", job=job, profile={"education": []})
        mock_get_client.assert_not_called()
        assert result["compensation"]["status"] == "explicitly_absent"
        assert result["eligibility"] == "ineligible_other"

    def test_score_never_drops_below_floor_of_one(self):
        import applypilot.scoring.scorer as scorer_mod

        job = {
            "title": "IT Help Desk Analyst",
            "site": "SomeCorp",
            "location": "Remote (US)",
            "salary": None,
            "full_description": "No pay stated in this posting.",
            "url": "https://example.com/j4",
        }
        patcher = self._mock_llm(
            response="ELIGIBILITY: eligible\nSCORE: 1\nKEYWORDS: help desk\nREASONING: Weak fit."
        )
        try:
            result = scorer_mod.score_job(resume_text="Resume text.", job=job, profile={"education": []})
        finally:
            patcher.stop()
        assert result["score"] >= 1


# ── Eligibility / stage-selection interaction (not the same as scoring) ──


class TestEligibilityUnaffectedByCompensationUncertainty:
    """Requirement: unknown-compensation jobs must not accidentally become
    ineligible solely because pay is missing -- the compensation penalty
    is a SCORE adjustment, never an eligibility category, and must not
    prevent an otherwise-qualifying job from being selected downstream."""

    def test_unknown_compensation_job_still_selectable_for_tailoring(self, tmp_db, seed_job):
        from applypilot.database import get_jobs_by_stage

        conn = tmp_db()
        job = seed_job(
            conn,
            fit_score=8,
            full_description="No pay stated in this posting.",
            salary=None,
            state="scored",
            eligibility="eligible",
            tailored_resume_path=None,
        )
        rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=0)
        urls = {r["url"] for r in rows}
        assert job["url"] in urls

    def test_explicitly_absent_job_excluded_from_tailoring(self, tmp_db, seed_job):
        from applypilot.database import get_jobs_by_stage

        conn = tmp_db()
        job = seed_job(
            conn,
            fit_score=2,
            full_description="This is a 100% commission business.",
            salary=None,
            state="archived",
            eligibility="ineligible_other",
            tailored_resume_path=None,
        )
        rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=0)
        urls = {r["url"] for r in rows}
        assert job["url"] not in urls
