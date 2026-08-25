"""Authoritative, deterministic hard-disqualifier logic.

Single source of truth for eligibility checks that must hold regardless of
what any individual LLM scoring call decides -- a seniority mismatch (or
similar categorical disqualifier) must never be something the LLM can
compensate for by accumulating points elsewhere, and it must never depend
on a user-editable config file being correctly populated.

2026-08-21 incident: this candidate applies with no professional software
engineering experience. Senior/Staff/Principal-level roles were meant to be
permanently excluded, but the check existed in two independent, drifted
copies (scorer.py's _SENIOR_TITLE_PATTERN, applied only as a post-LLM score
cap, and apply/launcher.py's _AUTO_REJECT_TITLE_PATTERN, applied only at
apply-acquisition time) plus a third opt-in copy in cli.py used only by the
--auto-reject-titles flag. A PayPal "Sr Software Engineer" scored under an
LLM call that predated the guard even existing, and nothing ever
re-validated that stored score against current rules -- it sat at
fit_score=8, reached tailor, cover, and ready_to_apply, three days later.

This module fixes both problems: one canonical predicate every stage calls
(so the definition can't drift again), checked BEFORE the LLM call (so a
disqualified job never gets a chance to accumulate a compensating score),
plus a revalidation function that can re-run this same predicate against
already-scored rows whenever the rule changes, without needing an LLM call
or touching data that doesn't match.
"""

from __future__ import annotations

import re

# Union of every seniority-adjacent keyword previously spread across
# scorer.py's _SENIOR_TITLE_PATTERN (senior/sr/staff/principal/lead/
# architect/director/vp/chief) and apply/launcher.py's
# _AUTO_REJECT_TITLE_PATTERN (same, plus manager/head, minus architect).
# This union matches cli.py's DEFAULT_TITLE_REJECT_PATTERNS, the most
# complete of the three pre-existing lists -- adopting the widest
# already-vetted list rather than inventing a new one.
SENIORITY_TITLE_PATTERN = re.compile(
    r"\b(?:senior|sr\.?|staff|principal|lead|architect|director|"
    r"manager|head|vp|vice president|chief|distinguished|fellow|"
    # 2026-08-25: "cto" added -- a bare acronym found NOT to match any
    # existing alternative (discovered while consolidating tailor.py's
    # seniority gate onto this predicate; the removed duplicate had its own
    # "cto" case this canonical version lacked). Unlike director/chief/
    # manager/head, "CTO" has essentially no legitimate non-technical
    # meaning in a job title, so this is a narrow, low-risk addition rather
    # than the kind of breadth tradeoff those other words represent.
    r"cto)\b|"
    # Explicit level-number conventions: III/IV/V/VI and 3/4/5/6.
    # I/II and 1/2 remain allowed because they can represent entry-level
    # or early-career roles. "swe"/"sde" added 2026-08-25 alongside the
    # bare-"L"/"Level" branches below -- same real-data audit that found
    # the L-notation gap also found these abbreviated titles ("SWE III",
    # "SDE 3") using the identical adjacency structure.
    r"\b(?:engineer|developer|swe|sde)\s*[-,]?\s*(?:III|IV|V|VI|3|4|5|6)\b|"
    # 2026-08-25: equivalent "L3"/"Level 3"-style leveling notation (Twilio,
    # Meta-style "Software Engineer, Backend, Level 5", Amazon "SDE III").
    # Live-data audit found 19 real job titles using this convention with no
    # other senior keyword present, invisible to the branch above because
    # the numeral isn't directly adjacent to "engineer"/"developer" (e.g.
    # "Software Engineer (L4)", "Level 3 Software Engineer"). Deliberately
    # bare/domain-agnostic -- matches the existing keyword branch above,
    # which already disqualifies "Senior IT Support Specialist" regardless
    # of role domain, not just software engineering titles. L1/L2 and
    # Level 1/2 remain allowed, matching the I/II policy above -- only
    # 3 and up are disqualifying.
    r"\bL[3-6]\b|"
    r"\bLevel\s*[-,]?\s*(?:III|IV|V|VI|3|4|5|6)\b",
    re.IGNORECASE,
)


def seniority_disqualifier(title: str | None) -> str | None:
    """Return a disqualification reason if the title signals a seniority
    level this candidate has no professional experience to legitimately
    claim, else None.

    Deterministic, no LLM call, no profile lookup needed -- seniority-by-
    title is purely a function of the title string.
    """
    if not title:
        return None
    m = SENIORITY_TITLE_PATTERN.search(title)
    if m:
        return f"seniority title match: {m.group(0)!r} in {title!r}"
    return None


def revalidate_seniority(conn=None, *, dry_run: bool = False) -> dict:
    """Re-run the current seniority disqualifier against already-scored rows.

    For rule changes made after jobs were already scored (the exact failure
    mode above): finds any pre-submission job whose title now matches
    SENIORITY_TITLE_PATTERN and archives it, regardless of what fit_score it
    was previously assigned, INCLUDING jobs already sitting in "tailored" or
    "cover_failed" -- unlike database.reject_jobs_by_title_patterns' default
    protected-states set (built for the general-purpose --auto-reject-titles
    CLI flag, where discarding an arbitrary-pattern match against real
    generated work is the wrong default), the canonical seniority predicate
    is a hard categorical disqualifier: a stale pre-predicate score reaching
    "tailored" or even burning cover_attempts against "cover_failed" is
    exactly the failure this module exists to catch, not a sunk cost worth
    protecting. 2026-08-25 real-data audit found 12 such jobs live in the
    database (PayPal "Sr Software Engineer"/"Sr Machine Learning Engineer",
    Twilio "Sr Architect", Axon "Sr. Full Stack Member of Technical Staff",
    6 Pinterest Staff-level roles, etc.) with real .docx resumes already on
    disk and up to 5 cover_attempts already spent -- confirming
    revalidate-seniority --dry-run's prior "0/0 matched" result reflected
    this blind spot, not an actually-clean database. Only states genuinely
    at or past submission (applied / applying / cover_writing / manual_only
    / archived / ready_to_apply) remain protected here. Safe to call
    repeatedly -- reuses database.reject_jobs_by_title_patterns(), so a
    second run against already-archived matches is a no-op (idempotent),
    and rows that don't match are never touched. No job record is deleted;
    fit_score, score_reasoning, tailored_resume_path, and cover_letter_path
    are all preserved on the archived row for audit purposes.

    Returns the same {"matched", "updated", "sample"} dict as
    reject_jobs_by_title_patterns().
    """
    from applypilot.database import reject_jobs_by_title_patterns

    protected_states = frozenset(
        {"applied", "applying", "cover_writing", "manual_only", "archived", "ready_to_apply"}
    )
    return reject_jobs_by_title_patterns(
        [SENIORITY_TITLE_PATTERN.pattern],
        conn=conn,
        dry_run=dry_run,
        sample_limit=1000,
        protected_states=protected_states,
    )


# States a stale-score sweep is allowed to touch. Deliberately narrower than
# "everything scored before cutoff": once a job is at or past submission
# (applied/applying/ready_to_apply) or already archived/manual_only, a stale
# score is either moot or a sunk cost not worth re-litigating -- matches the
# same protected-submission philosophy revalidate_seniority uses above.
_STALE_SCORE_REVALIDATABLE_STATES = frozenset({"scored", "tailored", "tailor_failed", "cover_failed"})


def revalidate_stale_scores(conn=None, *, cutoff: str, dry_run: bool = False) -> dict:
    """Archive jobs whose fit_score was assigned by an LLM call that predates
    a scoring-rubric fix, regardless of what score they received.

    2026-08-25 real-data audit: commit a6f72a6 (2026-08-18) added the
    seniority/experience-check paragraph to SCORE_PROMPT_TEMPLATE, fixing the
    rubric itself. A one-time requeue swept most stale-scored rows into
    "archived" that same day, but a since-fixed bug in get_jobs_by_stage's
    "pending_tailor" query (missing a state filter until commit f2a3fab,
    2026-08-24 -- see decision on the pending_tailor archived/tailor_failed
    oscillation) let 63 of those already-archived rows get re-selected for
    tailoring on 2026-08-20, and tailor.py's _mark_tailor_result force-
    transitioned them straight from "archived" to "tailor_failed" on failure
    without ever re-scoring them -- resurrecting their stale, pre-fix
    fit_score back into the live funnel. 8 of those resurrected rows are
    still live (e.g. Twilio "Software Engineer (L2)" / "Site Reliability
    Engineer II" scored 8/10 under the pre-fix rubric despite explicitly
    requiring 2-5+ years of professional experience and, in one case, a
    Bachelor's CS degree). None of these titles match SENIORITY_TITLE_PATTERN
    (they're "L2"/"II"/"I" -- deliberately-allowed levels), so
    revalidate_seniority's title-based sweep can never catch this class of
    staleness; the failure mode here is the *scoring date*, not the title.

    Unlike revalidate_seniority, this can't reuse
    database.reject_jobs_by_title_patterns (title-regex specific) -- it
    queries scored_at directly and archives via transition_state, following
    the same "preserve fit_score/tailored_resume_path/cover_letter_path for
    audit, only change state" convention.

    This is deliberately NOT a general periodic re-scoring system -- just a
    reusable version of the one-time 2026-08-18 requeue sweep, so a future
    rubric fix doesn't leave another orphaned cohort the way this one did.

    Args:
        conn: Optional sqlite connection.
        cutoff: ISO timestamp (e.g. the rubric-fix commit's date). Jobs with
            scored_at < cutoff are matched.
        dry_run: When True, report matches only and make no DB writes.

    Returns:
        Dict with keys: matched (int), updated (int), sample (list[dict]),
        each sample row carrying url/title/fit_score/scored_at/state for
        audit review before a live run.
    """
    from applypilot.database import commit_with_retry, get_connection, transition_state

    if conn is None:
        conn = get_connection()

    placeholders = ", ".join("?" for _ in _STALE_SCORE_REVALIDATABLE_STATES)
    rows = conn.execute(
        f"""
        SELECT url, title, fit_score, scored_at, state
        FROM jobs
        WHERE scored_at IS NOT NULL
          AND scored_at < ?
          AND COALESCE(state, '') IN ({placeholders})
        """,
        (cutoff, *_STALE_SCORE_REVALIDATABLE_STATES),
    ).fetchall()

    matches = [
        {
            "url": row["url"],
            "title": row["title"] or "",
            "fit_score": row["fit_score"],
            "scored_at": row["scored_at"],
            "state": row["state"],
        }
        for row in rows
    ]

    if dry_run:
        return {"matched": len(matches), "updated": 0, "sample": matches}

    updated = 0
    for m in matches:
        ok = transition_state(
            conn,
            m["url"],
            "archived",
            reason=f"stale_score_before_{cutoff}",
            metadata={"prior_fit_score": m["fit_score"], "prior_scored_at": m["scored_at"], "title": m["title"][:120]},
            force=True,
        )
        if ok:
            updated += 1

    commit_with_retry(conn)
    return {"matched": len(matches), "updated": updated, "sample": matches}
