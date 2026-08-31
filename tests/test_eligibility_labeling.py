"""Regression tests for the 2026-08-29 eligibility-labeling fix.

score_job()'s pre-filter branch used to hardcode eligibility="non_us_only"
for EVERY _check_ineligible() rejection reason -- seniority, title/search-
config exclusion, advanced-degree, clearance, commission-only, and ethics
reasons all got mislabeled as a geography rejection. Live measurement: of
7,155 "non_us_only" rows, 2,784 (39%) were actually non-geography (2,592
title-pattern, 187 seniority, 5 degree/ethics).

Fix: scorer._classify_ineligibility() maps _check_ineligible()'s reason
string to one of scorer.DETERMINISTIC_INELIGIBLE_VALUES ("non_us_only",
"seniority_mismatch", "title_excluded", "ineligible_other"). Behavior that
must be preserved exactly: ALL four categories still terminally archive the
job (scorer._flush_score_batch), and are still excluded from
pending_tailor/pending_cover/pending_apply (database.py's existing positive
`eligibility IS NULL OR eligibility = 'eligible'` checks -- unchanged,
verified to need no code change since they already exclude any non-
"eligible" value). The state-transition reason now carries the real
descriptive text instead of the hardcoded literal "non_us_only
employer/role".

This does NOT touch the scoring algorithm/prompt/calibration, and does NOT
backfill the 2,784 already-mislabeled historical rows -- new/re-scored jobs
only.
"""

from __future__ import annotations

import pytest

from applypilot.scoring.scorer import DETERMINISTIC_INELIGIBLE_VALUES, _classify_ineligibility

# ── _classify_ineligibility: pure string classification ─────────────────────


class TestClassifyIneligibility:
    def test_non_us_geography_in_title(self):
        assert _classify_ineligibility("non-US geography in title: Software Engineer (Remote - India)") == (
            "non_us_only"
        )

    def test_non_us_location_field(self):
        assert _classify_ineligibility("non-US location field: Sydney, New South Wales, Australia") == "non_us_only"

    def test_non_us_geography_in_description(self):
        assert _classify_ineligibility("non-US geography in description: must be based in Toronto") == "non_us_only"

    def test_seniority_title_match(self):
        reason = "seniority title match: 'Senior' in 'Senior Software Engineer'"
        assert _classify_ineligibility(reason) == "seniority_mismatch"

    def test_title_excluded_by_search_configuration(self):
        reason = "title excluded by search configuration: VP of Engineering"
        assert _classify_ineligibility(reason) == "title_excluded"

    @pytest.mark.parametrize(
        "reason",
        [
            "advanced degree required, candidate has none: PhD in Computer Science",
            "security clearance required: active TS/SCI clearance",
            "commission-only compensation: 100% commission, no base",
            "ethical exclusion keyword matched: 'defense industry'",
        ],
    )
    def test_other_reasons_fall_into_catch_all(self, reason):
        """Low-volume reasons (degree/clearance/commission/ethics -- 5 rows
        total in the live measurement) deliberately share one catch-all
        category rather than each getting a dedicated value, per the
        'do not invent unnecessary categories' design constraint."""
        assert _classify_ineligibility(reason) == "ineligible_other"

    def test_every_classification_is_a_recognized_value(self):
        """No classification path can produce a string outside the fixed
        DETERMINISTIC_INELIGIBLE_VALUES set -- guards against a future new
        _check_ineligible() reason silently escaping the catch-all."""
        reasons = [
            "non-US location field: Berlin, Germany",
            "seniority title match: 'Staff' in 'Staff Engineer'",
            "title excluded by search configuration: Recruiter",
            "advanced degree required, candidate has none: MS required",
            "some entirely new reason text never seen before",
        ]
        for reason in reasons:
            assert _classify_ineligibility(reason) in DETERMINISTIC_INELIGIBLE_VALUES


# ── _flush_score_batch: archiving + transition-reason behavior ──────────────
#
# Follows the seed_job() + batch dict + _flush_score_batch() + row-assertion
# pattern already established in test_score_batch_archived_guard.py.


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


class TestFlushScoreBatchArchivingAllCategories:
    """Requirement 4: all deterministic ineligible categories are still
    archived, exactly like the old non_us_only-only branch did."""

    @pytest.mark.parametrize(
        "eligibility_value",
        ["non_us_only", "seniority_mismatch", "title_excluded", "ineligible_other"],
    )
    def test_each_category_routes_to_archived(self, tmp_db, seed_job, eligibility_value):
        from applypilot.scoring.scorer import _flush_score_batch

        conn = tmp_db()
        job = seed_job(conn, fit_score=None, full_description="x", state="enriched")

        batch = [
            {
                "url": job["url"],
                "score": 2,
                "keywords": "",
                "reasoning": f"Ineligible: some {eligibility_value} reason.",
                "eligibility": eligibility_value,
            }
        ]
        _flush_score_batch(conn, batch, _now())

        row = conn.execute(
            "SELECT eligibility, state FROM jobs WHERE url = ?",
            (job["url"],),
        ).fetchone()
        assert row["eligibility"] == eligibility_value
        assert row["state"] == "archived"

    def test_eligible_job_is_not_archived(self, tmp_db, seed_job):
        """Requirement 7 (DB-level contrast case): a genuinely eligible job
        must not be swept into the archiving branch by the broadened
        membership check."""
        from applypilot.scoring.scorer import _flush_score_batch

        conn = tmp_db()
        job = seed_job(conn, fit_score=None, full_description="x", state="enriched")

        batch = [
            {
                "url": job["url"],
                "score": 9,
                "keywords": "python, aws",
                "reasoning": "strong match",
                "eligibility": "eligible",
            }
        ]
        _flush_score_batch(conn, batch, _now())

        row = conn.execute(
            "SELECT fit_score, eligibility, state FROM jobs WHERE url = ?",
            (job["url"],),
        ).fetchone()
        assert row["fit_score"] == 9
        assert row["eligibility"] == "eligible"
        assert row["state"] == "scored"


class TestTransitionReasonUsesActualReason:
    """Requirement 6: transition history records the real descriptive
    rejection reason, not the old hardcoded "non_us_only employer/role"
    literal -- for every category, not just geography."""

    @pytest.mark.parametrize(
        "eligibility_value,reasoning",
        [
            ("non_us_only", "Ineligible: non-US location field: Sydney, Australia."),
            (
                "seniority_mismatch",
                "Ineligible: seniority title match: 'Senior' in 'Senior Software Engineer'.",
            ),
            (
                "title_excluded",
                "Ineligible: title excluded by search configuration: VP of Engineering.",
            ),
        ],
    )
    def test_transition_reason_matches_actual_reasoning(self, tmp_db, seed_job, eligibility_value, reasoning):
        from applypilot.scoring.scorer import _flush_score_batch

        conn = tmp_db()
        job = seed_job(conn, fit_score=None, full_description="x", state="enriched")

        batch = [
            {
                "url": job["url"],
                "score": 2,
                "keywords": "",
                "reasoning": reasoning,
                "eligibility": eligibility_value,
            }
        ]
        _flush_score_batch(conn, batch, _now())

        transitions = conn.execute(
            "SELECT to_state, reason FROM job_state_transitions WHERE job_url = ? ORDER BY id DESC",
            (job["url"],),
        ).fetchall()
        assert transitions[0]["to_state"] == "archived"
        assert transitions[0]["reason"] == reasoning[:200]
        assert transitions[0]["reason"] != "non_us_only employer/role"


class TestStageSelectionExcludesAllCategories:
    """Requirement 5: all deterministic ineligible categories remain
    excluded from pending_tailor/pending_cover/pending_apply -- exercised
    through database.py's real stage-selection predicates, not just the
    positive-check source read."""

    @pytest.mark.parametrize(
        "eligibility_value",
        ["non_us_only", "seniority_mismatch", "title_excluded", "ineligible_other"],
    )
    def test_ineligible_category_excluded_from_pending_tailor(self, tmp_db, seed_job, eligibility_value):
        from applypilot.database import get_jobs_by_stage

        conn = tmp_db()
        job = seed_job(
            conn,
            fit_score=2,
            full_description="x",
            state="scored",
            eligibility=eligibility_value,
            tailored_resume_path=None,
        )

        rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=0)
        urls = {r["url"] for r in rows}
        assert job["url"] not in urls

    def test_eligible_job_is_selected_for_pending_tailor(self, tmp_db, seed_job):
        """Contrast case for requirement 5/7: a genuinely eligible, scored
        job must still be selectable."""
        from applypilot.database import get_jobs_by_stage

        conn = tmp_db()
        job = seed_job(
            conn,
            fit_score=9,
            full_description="x",
            state="scored",
            eligibility="eligible",
            tailored_resume_path=None,
        )

        rows = get_jobs_by_stage(conn, stage="pending_tailor", min_score=0)
        urls = {r["url"] for r in rows}
        assert job["url"] in urls
