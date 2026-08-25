"""Tests for the consolidated seniority hard-disqualifier.

2026-08-21 incident: a PayPal "Sr Software Engineer" scored 8 under an LLM
call that predated the seniority guard, then sat untouched through tailor,
cover, and ready_to_apply -- nothing ever re-validated the stored score
against current rules, and the guard itself existed as two independent,
drifted regexes (scorer.py post-LLM cap, apply/launcher.py apply-time
reject). These tests cover: the single canonical predicate
(applypilot.eligibility), that it's checked BEFORE the LLM call so a
stored fit_score can never bypass it, that apply's defense-in-depth check
agrees with it exactly, and that the revalidation sweep correctly
archives stale matches (idempotently, without touching non-matches or
deleting anything) so a rule change can be applied retroactively.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import UTC

from applypilot.eligibility import seniority_disqualifier

# ── seniority_disqualifier(): the canonical predicate ────────────────────


@pytest.mark.parametrize(
    "title",
    [
        "Senior Software Engineer",
        "Sr. Software Engineer",
        "Sr Software Engineer",
        "Senior Systems Engineer",
        "Staff Engineer",
        "Staff Software Engineer",
        "Principal Engineer",
        "Principal Software Engineer",
        "Lead Engineer",
        "Lead Software Engineer",
        "Senior IT Support Specialist",
        "Staff IT Systems Administrator",
        "Solutions Architect",
        "Software Architect",
        "Engineering Director",
        "VP of Engineering",
        "Vice President, Platform",
        "Chief Technology Officer",
        "Engineering Manager",
        "Head of Engineering",
        "Distinguished Engineer",
        "Technical Fellow",
        "Software Engineer III",
        "Software Engineer IV",
        "Software Engineer V",
        "Software Engineer VI",
        "Software Developer III",
        "Software Developer IV",
        "Software Developer V",
        "Software Developer VI",
        "Software Engineer 3",
        "Software Engineer 4",
        "Software Engineer 5",
        "Software Engineer 6",
    ],
)
def test_senior_titles_disqualified(title):
    assert seniority_disqualifier(title) is not None, f"{title!r} should be disqualified"


@pytest.mark.parametrize(
    "title",
    [
        "Software Engineer",
        "IT Support Specialist",
        "Help Desk Technician",
        "Junior Software Engineer",
        "Entry Level Software Engineer",
        "New Grad Software Engineer",
        "Graduate Software Engineer",
        "Fresher Software Engineer",
        "Software Engineering Intern",
        "Systems Administrator",
        "Desktop Support Technician",
        "Technical Support Engineer",
    ],
)
def test_non_senior_titles_not_disqualified(title):
    assert seniority_disqualifier(title) is None, f"{title!r} should NOT be disqualified"


def test_disqualifier_reason_identifies_matched_keyword():
    """The predicate returns enough information to see WHY, not just True/False."""
    reason = seniority_disqualifier("Staff Software Engineer")
    assert reason is not None
    assert "staff" in reason.lower()


def test_engineer_developer_level_i_and_ii_are_not_disqualified():
    for title in [
        "Software Engineer I",
        "Software Engineer II",
        "Software Developer I",
        "Software Developer II",
        "Software Engineer 1",
        "Software Engineer 2",
        "Software Developer 1",
        "Software Developer 2",
    ]:
        assert seniority_disqualifier(title) is None, f"{title!r} should remain allowed"


def test_engineer_developer_level_three_and_above_are_disqualified():
    for title in [
        "Software Engineer III",
        "Software Engineer IV",
        "Software Engineer V",
        "Software Developer III",
        "Software Developer IV",
        "Software Developer V",
        "Software Engineer 3",
        "Software Engineer 4",
        "Software Developer 5",
    ]:
        assert seniority_disqualifier(title) is not None, f"{title!r} should be disqualified"


def test_empty_or_none_title_not_disqualified():
    assert seniority_disqualifier(None) is None
    assert seniority_disqualifier("") is None


# ── Score never compensates for a hard disqualifier ──────────────────────


class TestScoreCannotBypassSeniority:
    """A senior title must be rejected before the LLM ever runs -- so no
    LLM-assigned score, however high, can ever reach min_score."""

    def _job(self, title="Senior Software Engineer"):
        return {
            "title": title,
            "site": "TestCo",
            "location": "Remote (US)",
            "full_description": "We need an expert with 10 years of experience.",
        }

    def test_check_ineligible_rejects_senior_title_without_profile(self):
        """Works even when no profile is supplied (profile=None), matching
        how _check_ineligible is called in some code paths."""
        from applypilot.scoring.scorer import _check_ineligible

        assert _check_ineligible(self._job()) is not None

    def test_score_job_never_calls_llm_for_senior_title(self):
        """The strongest guarantee: the LLM client is never even
        constructed/invoked for a disqualified title, so there is no
        code path by which an LLM response could produce a high score."""
        import applypilot.scoring.scorer as scorer_mod

        with patch.object(scorer_mod, "get_stage_client") as mock_get_client:
            result = scorer_mod.score_job(
                resume_text="Resume text.",
                job=self._job("Senior Software Engineer"),
                profile={"education": []},
            )

        mock_get_client.assert_not_called()
        assert result["score"] == 2
        assert result["eligibility"] == "non_us_only"  # reused generic reject signal
        assert "seniority" in result["reasoning"].lower()

    @pytest.mark.parametrize(
        "title",
        [
            "Senior Software Engineer",
            "Sr. Software Engineer",
            "Staff Software Engineer",
            "Principal Engineer",
            "Senior Systems Engineer",
        ],
    )
    def test_a_stored_high_score_cannot_have_come_from_these_titles(self, title):
        """Directly answers 'a previously stored fit_score of 9 does not
        bypass the current seniority rule': for every required senior
        title, confirm _check_ineligible rejects it pre-LLM regardless of
        what score a caller might imagine attaching to it -- there is no
        code path where score_job would return a passing score for these."""
        import applypilot.scoring.scorer as scorer_mod

        with patch.object(scorer_mod, "get_stage_client") as mock_get_client:
            result = scorer_mod.score_job("Resume.", self._job(title), {"education": []})
        mock_get_client.assert_not_called()
        assert result["score"] < 8


# ── apply and scoring share one definition ────────────────────────────────


@pytest.mark.parametrize(
    "title",
    [
        "Senior Software Engineer",
        "Sr. Software Engineer",
        "Staff Software Engineer",
        "Principal Engineer",
        "Senior Systems Engineer",
        "Solutions Architect",
        "Engineering Manager",
        "Head of Engineering",
        "Software Engineer",
        "Junior Software Engineer",
        "IT Support Specialist",
    ],
)
def test_apply_and_scoring_agree(title):
    """apply/launcher.py's defense-in-depth check must never diverge from
    the canonical predicate -- that divergence (two independently
    maintained regexes) is what let the 2026-08-21 incident happen."""
    from applypilot.apply.launcher import _auto_reject_title

    scorer_says_disqualified = seniority_disqualifier(title) is not None
    apply_says_reject = _auto_reject_title(title)
    assert scorer_says_disqualified == apply_says_reject, (
        f"{title!r}: scorer={scorer_says_disqualified} apply={apply_says_reject}"
    )


# ── Stale-data revalidation sweep ─────────────────────────────────────────


class TestRevalidateSeniority:
    def test_ready_to_apply_senior_job_is_not_retroactively_archived(self, tmp_db, seed_job):
        """Once a job reaches the post-tailoring lifecycle, the canonical
        rejection helper must stop re-archiving it. This preserves the
        state machine instead of letting stale revalidation delete valid
        in-flight work."""
        from applypilot.eligibility import revalidate_seniority

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="stale-senior",
            title="Sr Software Engineer",
            fit_score=8,
            state="ready_to_apply",
            tailored_resume_path="/tmp/r.docx",
            cover_letter_path="/tmp/c.docx",
        )

        result = revalidate_seniority(conn)

        assert result["matched"] == 0
        assert result["updated"] == 0
        row = conn.execute(
            "SELECT state, fit_score, tailored_resume_path FROM jobs WHERE url LIKE '%stale-senior%'"
        ).fetchone()
        assert row["state"] == "ready_to_apply"
        assert row["fit_score"] == 8
        assert row["tailored_resume_path"] == "/tmp/r.docx"

    def test_non_senior_job_not_touched(self, tmp_db, seed_job):
        from applypilot.eligibility import revalidate_seniority

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="clean-title",
            title="Software Engineer",
            fit_score=9,
            state="ready_to_apply",
        )

        result = revalidate_seniority(conn)

        assert result["matched"] == 0
        row = conn.execute("SELECT state FROM jobs WHERE url LIKE '%clean-title%'").fetchone()
        assert row["state"] == "ready_to_apply"

    def test_entry_level_and_junior_titles_unaffected(self, tmp_db, seed_job):
        """Legitimate existing exceptions (junior/entry-level/new-grad)
        continue to work -- the sweep must not over-match."""
        from applypilot.eligibility import revalidate_seniority

        conn = tmp_db()
        for i, title in enumerate(
            [
                "Junior Software Engineer",
                "Entry Level Software Engineer",
                "New Grad Software Engineer",
                "IT Support Specialist",
            ]
        ):
            seed_job(conn, url_suffix=f"ok-{i}", title=title, fit_score=8, state="ready_to_apply")

        result = revalidate_seniority(conn)

        assert result["matched"] == 0
        rows = conn.execute("SELECT state FROM jobs").fetchall()
        assert all(r["state"] == "ready_to_apply" for r in rows)

    def test_idempotent(self, tmp_db, seed_job):
        from applypilot.eligibility import revalidate_seniority

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="stale-senior-2",
            title="Staff Software Engineer",
            fit_score=9,
            state="scored",
        )

        first = revalidate_seniority(conn)
        second = revalidate_seniority(conn)

        assert first["updated"] == 1
        assert second["matched"] == 0
        assert second["updated"] == 0

    def test_dry_run_reports_without_modifying(self, tmp_db, seed_job):
        from applypilot.eligibility import revalidate_seniority

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="dry-run-senior",
            title="Principal Engineer",
            fit_score=8,
            state="scored",
        )

        result = revalidate_seniority(conn, dry_run=True)

        assert result["matched"] == 1
        assert result["updated"] == 0
        row = conn.execute("SELECT state FROM jobs WHERE url LIKE '%dry-run-senior%'").fetchone()
        assert row["state"] == "scored"  # untouched while in pre-tailoring state

    def test_tailored_senior_job_is_archived(self, tmp_db, seed_job):
        """2026-08-25 real-data audit: 12 jobs (PayPal Sr Software Engineer,
        Twilio Sr Architect, 6 Pinterest Staff roles, etc.) sat in "tailored"
        with real resumes already generated, invisible to revalidate_seniority
        because it inherited reject_jobs_by_title_patterns' default
        protected-states set built for the unrelated --auto-reject-titles
        flag. "tailored" must not protect a canonical-predicate match --
        that's the entire point of retroactive revalidation."""
        from applypilot.eligibility import revalidate_seniority

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="tailored-senior",
            title="Sr Software Engineer",
            fit_score=8,
            state="tailored",
            tailored_resume_path="/tmp/r.docx",
        )

        result = revalidate_seniority(conn)

        assert result["matched"] == 1
        assert result["updated"] == 1
        row = conn.execute(
            "SELECT state, tailored_resume_path FROM jobs WHERE url LIKE '%tailored-senior%'"
        ).fetchone()
        assert row["state"] == "archived"
        # Sunk-cost artifact preserved for audit -- only the state changes.
        assert row["tailored_resume_path"] == "/tmp/r.docx"

    def test_cover_failed_senior_job_is_archived(self, tmp_db, seed_job):
        from applypilot.eligibility import revalidate_seniority

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="cover-failed-senior",
            title="Staff Software Engineer",
            fit_score=8,
            state="cover_failed",
            tailored_resume_path="/tmp/r.docx",
        )

        result = revalidate_seniority(conn)

        assert result["matched"] == 1
        assert result["updated"] == 1
        row = conn.execute("SELECT state FROM jobs WHERE url LIKE '%cover-failed-senior%'").fetchone()
        assert row["state"] == "archived"

    def test_already_applied_job_never_touched(self, tmp_db, seed_job):
        """Safety: never retroactively archive a job that was already
        submitted, even if its title matches."""
        from datetime import datetime

        from applypilot.eligibility import revalidate_seniority

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="already-applied-senior",
            title="Senior Software Engineer",
            fit_score=8,
            state="applied",
            apply_status="applied",
            applied_at=datetime.now(UTC).isoformat(),
        )

        result = revalidate_seniority(conn)

        assert result["matched"] == 0
        row = conn.execute("SELECT state FROM jobs WHERE url LIKE '%already-applied-senior%'").fetchone()
        assert row["state"] == "applied"


# ── Stale-score revalidation sweep (independent of title) ────────────────


class TestRevalidateStaleScores:
    """2026-08-25: 8 real jobs (Twilio 'Software Engineer (L2)', Axon 'Site
    Reliability Engineer II', etc.) were scored 8/10 on 2026-08-17 under the
    pre-a6f72a6 rubric, correctly archived by the 2026-08-18 requeue sweep,
    then resurrected back into tailor_failed/cover_failed/tailored by a
    since-fixed pending_tailor state-filter bug that re-selected already-
    archived rows for tailoring without re-scoring them. None of their
    titles match SENIORITY_TITLE_PATTERN ('L2'/'II' are allowed levels), so
    revalidate_seniority can never catch this class of staleness -- these
    tests cover the scored_at-based sweep that does."""

    CUTOFF = "2026-08-18T00:00:00+00:00"

    def test_job_scored_before_cutoff_is_archived(self, tmp_db, seed_job):
        from applypilot.eligibility import revalidate_stale_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="stale-swe-l2",
            title="Software Engineer (L2)",
            fit_score=8,
            state="tailored",
            scored_at="2026-08-17T01:56:11.346301+00:00",
            tailored_resume_path="/tmp/r.docx",
        )

        result = revalidate_stale_scores(conn, cutoff=self.CUTOFF)

        assert result["matched"] == 1
        assert result["updated"] == 1
        row = conn.execute(
            "SELECT state, fit_score, tailored_resume_path FROM jobs WHERE url LIKE '%stale-swe-l2%'"
        ).fetchone()
        assert row["state"] == "archived"
        # Sunk-cost artifacts preserved for audit -- only state changes.
        assert row["fit_score"] == 8
        assert row["tailored_resume_path"] == "/tmp/r.docx"

    def test_job_scored_after_cutoff_not_touched(self, tmp_db, seed_job):
        from applypilot.eligibility import revalidate_stale_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="fresh-score",
            title="Software Engineer (L2)",
            fit_score=8,
            state="tailored",
            scored_at="2026-08-19T00:00:00+00:00",
        )

        result = revalidate_stale_scores(conn, cutoff=self.CUTOFF)

        assert result["matched"] == 0
        row = conn.execute("SELECT state FROM jobs WHERE url LIKE '%fresh-score%'").fetchone()
        assert row["state"] == "tailored"

    def test_unscored_job_not_touched(self, tmp_db, seed_job):
        """scored_at IS NULL must never match -- there is no stale score to
        revalidate, only an unscored job."""
        from applypilot.eligibility import revalidate_stale_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="never-scored",
            title="Software Engineer",
            fit_score=None,
            state="enriched",
            scored_at=None,
        )

        result = revalidate_stale_scores(conn, cutoff=self.CUTOFF)

        assert result["matched"] == 0

    @pytest.mark.parametrize("protected_state", ["applied", "applying", "ready_to_apply", "manual_only", "archived"])
    def test_protected_states_never_touched(self, tmp_db, seed_job, protected_state):
        from applypilot.eligibility import revalidate_stale_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix=f"protected-{protected_state}",
            title="Software Engineer (L2)",
            fit_score=8,
            state=protected_state,
            scored_at="2026-08-17T00:00:00+00:00",
        )

        result = revalidate_stale_scores(conn, cutoff=self.CUTOFF)

        assert result["matched"] == 0
        row = conn.execute(f"SELECT state FROM jobs WHERE url LIKE '%protected-{protected_state}%'").fetchone()
        assert row["state"] == protected_state

    @pytest.mark.parametrize("revalidatable_state", ["scored", "tailored", "tailor_failed", "cover_failed"])
    def test_each_revalidatable_state_is_archived(self, tmp_db, seed_job, revalidatable_state):
        from applypilot.eligibility import revalidate_stale_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix=f"stale-{revalidatable_state}",
            title="Site Reliability Engineer II",
            fit_score=8,
            state=revalidatable_state,
            scored_at="2026-08-17T00:00:00+00:00",
        )

        result = revalidate_stale_scores(conn, cutoff=self.CUTOFF)

        assert result["matched"] == 1
        assert result["updated"] == 1
        row = conn.execute(f"SELECT state FROM jobs WHERE url LIKE '%stale-{revalidatable_state}%'").fetchone()
        assert row["state"] == "archived"

    def test_idempotent(self, tmp_db, seed_job):
        from applypilot.eligibility import revalidate_stale_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="idempotent-stale",
            title="Software Engineer I, Privacy",
            fit_score=8,
            state="cover_failed",
            scored_at="2026-08-17T00:00:00+00:00",
        )

        first = revalidate_stale_scores(conn, cutoff=self.CUTOFF)
        second = revalidate_stale_scores(conn, cutoff=self.CUTOFF)

        assert first["updated"] == 1
        assert second["matched"] == 0
        assert second["updated"] == 0

    def test_dry_run_reports_without_modifying(self, tmp_db, seed_job):
        from applypilot.eligibility import revalidate_stale_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="dry-run-stale",
            title="Software Engineer - Video",
            fit_score=8,
            state="tailor_failed",
            scored_at="2026-08-17T00:00:00+00:00",
        )

        result = revalidate_stale_scores(conn, cutoff=self.CUTOFF, dry_run=True)

        assert result["matched"] == 1
        assert result["updated"] == 0
        assert result["sample"][0]["fit_score"] == 8
        row = conn.execute("SELECT state FROM jobs WHERE url LIKE '%dry-run-stale%'").fetchone()
        assert row["state"] == "tailor_failed"  # untouched in dry-run

    def test_sample_carries_audit_fields(self, tmp_db, seed_job):
        """The CLI/report needs enough per-row detail (title/score/scored_at/
        state) to review matches before a live run -- unlike
        revalidate_seniority's sample (which only needs url/title/pattern),
        this sweep's whole point is showing WHEN a job was scored."""
        from applypilot.eligibility import revalidate_stale_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="audit-fields",
            title="Site Reliability Engineer II",
            fit_score=8,
            state="cover_failed",
            scored_at="2026-08-17T02:11:58.584795+00:00",
        )

        result = revalidate_stale_scores(conn, cutoff=self.CUTOFF, dry_run=True)

        sample = result["sample"][0]
        assert sample["title"] == "Site Reliability Engineer II"
        assert sample["fit_score"] == 8
        assert sample["scored_at"] == "2026-08-17T02:11:58.584795+00:00"
        assert sample["state"] == "cover_failed"
