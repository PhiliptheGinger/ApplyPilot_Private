"""Tests for applypilot.scoring.contamination -- the narrow, content-keyed
mechanism (distinct from eligibility.revalidate_stale_scores) that finds
and remediates historical scores contaminated by the pre-Phase-1 fabricated
resume source. See docs/audit-2026-08-27.md and the 2026-08-27 Phase 4b
investigation for the real-data evidence behind these marker patterns.
"""

import pytest


class TestFindContaminatedScores:
    def test_positive_seattle_location_marker_detected(self, tmp_db, seed_job):
        from applypilot.scoring.contamination import find_contaminated_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="contaminated-loc",
            title="Senior Software Engineer",
            fit_score=8,
            state="scored",
            score_reasoning=(
                "Python, APIs, automation. The candidate has strong backend skills. "
                "The candidate is located in Seattle, WA, making them eligible."
            ),
        )

        matches = find_contaminated_scores(conn, min_score=7)

        assert len(matches) == 1
        assert "candidate_seattle_location" in matches[0]["markers"]

    def test_positive_seniority_marker_detected(self, tmp_db, seed_job):
        from applypilot.scoring.contamination import find_contaminated_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="contaminated-sen",
            title="Staff Software Engineer",
            fit_score=8,
            state="scored",
            score_reasoning=(
                "The candidate has over 3 years of experience as a senior software engineer "
                "with a focus on backend systems."
            ),
        )

        matches = find_contaminated_scores(conn, min_score=7)

        assert len(matches) == 1
        assert "candidate_senior_identity" in matches[0]["markers"]

    def test_legitimate_geo_mismatch_reasoning_not_flagged(self, tmp_db, seed_job):
        """A job actually located in Seattle, correctly compared against the
        true NC-based profile as a mismatch, must NOT be flagged -- this is
        truthful reasoning, not fabricated-identity contamination. This is
        the exact false-positive class the Phase 4b investigation found and
        deliberately excluded from the marker set."""
        from applypilot.scoring.contamination import find_contaminated_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="clean-geo-mismatch",
            title="Software Engineer",
            fit_score=6,
            state="low_score",
            score_reasoning=(
                "This is capped at a 6 because it is an onsite role in Seattle, which is "
                "outside the candidate's stated relocation/commute preference of North Carolina."
            ),
        )

        matches = find_contaminated_scores(conn, min_score=1)

        assert matches == []

    def test_ordinary_job_description_seniority_not_flagged(self, tmp_db, seed_job):
        """A job description that itself requires seniority must not be
        flagged merely for containing 'senior' -- only reasoning that
        attributes the identity to the CANDIDATE is a marker."""
        from applypilot.scoring.contamination import find_contaminated_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="clean-job-senior",
            title="Senior Software Engineer",
            fit_score=2,
            state="archived",
            score_reasoning=(
                "This role requires 8+ years of professional software engineering experience "
                "at a senior level, which the candidate does not have."
            ),
        )

        matches = find_contaminated_scores(conn, min_score=1)

        assert matches == []

    def test_below_min_score_excluded(self, tmp_db, seed_job):
        from applypilot.scoring.contamination import find_contaminated_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="low-score-contaminated",
            title="Software Developer I",
            fit_score=6,
            state="low_score",
            score_reasoning="The candidate is located in Seattle, WA, making them eligible.",
        )

        matches = find_contaminated_scores(conn, min_score=7)

        assert matches == []

    def test_null_reasoning_and_null_score_excluded(self, tmp_db, seed_job):
        from applypilot.scoring.contamination import find_contaminated_scores

        conn = tmp_db()
        seed_job(
            conn,
            url_suffix="unscored",
            title="Software Engineer",
            fit_score=None,
            state="enriched",
            score_reasoning=None,
        )

        matches = find_contaminated_scores(conn, min_score=1)

        assert matches == []


class TestSummarizeContamination:
    def test_buckets_by_state_and_score(self):
        from applypilot.scoring.contamination import summarize_contamination

        matches = [
            {"url": "a", "title": "A", "state": "scored", "fit_score": 8, "markers": ["candidate_seattle_location"]},
            {"url": "b", "title": "B", "state": "archived", "fit_score": 8, "markers": ["candidate_senior_identity"]},
            {"url": "c", "title": "C", "state": "applied", "fit_score": 7, "markers": ["candidate_seattle_location"]},
        ]

        summary = summarize_contamination(matches)

        assert summary["matched"] == 3
        assert summary["by_state"] == {"scored": 1, "archived": 1, "applied": 1}
        assert summary["by_score"] == {8: 2, 7: 1}
        assert summary["auto_remediable_count"] == 1  # only "scored"
        assert summary["needs_review_count"] == 1  # only "applied"
        assert summary["already_terminal_count"] == 1  # "archived"


class TestRemediateContaminatedScoresSafety:
    """The safety-critical tests: NEVER_MUTATE_STATES must never be written,
    even in non-dry-run mode. These mirror the same rigor as
    tests/test_phase3_state_resurrection.py."""

    def _seed_contaminated(self, conn, seed_job, *, url_suffix, state, fit_score=8):
        return seed_job(
            conn,
            url_suffix=url_suffix,
            title="Senior Software Engineer",
            fit_score=fit_score,
            state=state,
            score_reasoning="The candidate is located in Seattle, WA, making them eligible.",
            tailored_resume_path="/tmp/resume.pdf" if state != "scored" else None,
        )

    def test_dry_run_never_mutates_any_state(self, tmp_db, seed_job):
        from applypilot.scoring.contamination import remediate_contaminated_scores

        conn = tmp_db()
        self._seed_contaminated(conn, seed_job, url_suffix="dry-scored", state="scored")
        self._seed_contaminated(conn, seed_job, url_suffix="dry-applied", state="applied")

        result = remediate_contaminated_scores(conn, min_score=7, dry_run=True)

        assert result["matched"] == 2
        assert result["would_archive"] == 1  # only the "scored" one is auto-remediable
        assert result["archived"] == 0
        rows = conn.execute("SELECT url, state FROM jobs ORDER BY url").fetchall()
        states = {r["url"]: r["state"] for r in rows}
        assert states["https://example.com/job/dry-scored"] == "scored"
        assert states["https://example.com/job/dry-applied"] == "applied"

    @pytest.mark.parametrize("state", ["scored", "tailored", "tailor_failed", "cover_failed"])
    def test_auto_remediable_states_archived_in_non_dry_run(self, tmp_db, seed_job, state):
        from applypilot.scoring.contamination import remediate_contaminated_scores

        conn = tmp_db()
        row = self._seed_contaminated(conn, seed_job, url_suffix=f"auto-{state}", state=state)

        result = remediate_contaminated_scores(conn, min_score=7, dry_run=False)

        assert result["archived"] == 1
        updated = conn.execute("SELECT state, fit_score FROM jobs WHERE url=?", (row["url"],)).fetchone()
        assert updated["state"] == "archived"
        # Score preserved for audit -- only state changes.
        assert updated["fit_score"] == 8

    @pytest.mark.parametrize(
        "state", ["applied", "ready_to_apply", "applying", "apply_failed", "manual_only", "needs_human"]
    )
    def test_never_mutate_states_untouched_in_non_dry_run(self, tmp_db, seed_job, state):
        """Safety-critical: these states must NEVER be archived by this
        mechanism, even when explicitly invoked with dry_run=False."""
        from applypilot.scoring.contamination import remediate_contaminated_scores

        conn = tmp_db()
        row = self._seed_contaminated(conn, seed_job, url_suffix=f"protected-{state}", state=state)

        result = remediate_contaminated_scores(conn, min_score=7, dry_run=False)

        assert result["archived"] == 0
        untouched = conn.execute("SELECT state FROM jobs WHERE url=?", (row["url"],)).fetchone()
        assert untouched["state"] == state

    def test_already_archived_and_low_score_untouched(self, tmp_db, seed_job):
        from applypilot.scoring.contamination import remediate_contaminated_scores

        conn = tmp_db()
        row_a = self._seed_contaminated(conn, seed_job, url_suffix="already-archived", state="archived")
        row_l = self._seed_contaminated(conn, seed_job, url_suffix="already-low", state="low_score")

        result = remediate_contaminated_scores(conn, min_score=7, dry_run=False)

        assert result["archived"] == 0
        assert conn.execute("SELECT state FROM jobs WHERE url=?", (row_a["url"],)).fetchone()["state"] == "archived"
        assert conn.execute("SELECT state FROM jobs WHERE url=?", (row_l["url"],)).fetchone()["state"] == "low_score"

    def test_uncontaminated_row_never_touched(self, tmp_db, seed_job):
        from applypilot.scoring.contamination import remediate_contaminated_scores

        conn = tmp_db()
        row = seed_job(
            conn,
            url_suffix="clean",
            title="Software Engineer",
            fit_score=8,
            state="scored",
            score_reasoning="Strong Python and automation background, grounded in real profile facts.",
        )

        result = remediate_contaminated_scores(conn, min_score=7, dry_run=False)

        assert result["matched"] == 0
        assert result["archived"] == 0
        assert conn.execute("SELECT state FROM jobs WHERE url=?", (row["url"],)).fetchone()["state"] == "scored"

    def test_audit_metadata_preserved_on_archival(self, tmp_db, seed_job):
        from applypilot.scoring.contamination import remediate_contaminated_scores

        conn = tmp_db()
        row = self._seed_contaminated(conn, seed_job, url_suffix="audit-check", state="tailored")

        remediate_contaminated_scores(conn, min_score=7, dry_run=False)

        updated = conn.execute(
            "SELECT fit_score, score_reasoning, tailored_resume_path FROM jobs WHERE url=?", (row["url"],)
        ).fetchone()
        assert updated["fit_score"] == 8
        assert "Seattle" in updated["score_reasoning"]
        assert updated["tailored_resume_path"] == "/tmp/resume.pdf"

    def test_transition_audit_row_written(self, tmp_db, seed_job):
        from applypilot.scoring.contamination import remediate_contaminated_scores

        conn = tmp_db()
        row = self._seed_contaminated(conn, seed_job, url_suffix="audit-trail", state="scored")

        remediate_contaminated_scores(conn, min_score=7, dry_run=False)

        audit = conn.execute(
            "SELECT to_state, reason FROM job_state_transitions WHERE job_url=? ORDER BY at DESC LIMIT 1",
            (row["url"],),
        ).fetchone()
        assert audit["to_state"] == "archived"
        assert "contamination_remediation" in audit["reason"]
