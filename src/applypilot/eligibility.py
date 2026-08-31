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


# 2026-08-25: narrow exception for four title families that are
# near-universally individual-contributor, customer-facing roles in
# industry convention despite containing the bare word "manager" --
# NOT a general "manager" exemption. Investigated after the sales/
# recruiting/solutions occupational-blacklist removal (commit 4b888af):
# a live-DB audit found 1,213 rows blocked by SENIORITY_TITLE_PATTERN's
# bare "manager" branch alone, and a real posting (Zillow/Aryeo,
# "Technical Customer Success Manager, API") explicitly reads "you will
# own a portfolio of... customers" with zero team-management language
# and only a 2+ year requirement -- exactly the kind of case-by-case
# judgment the LLM scorer exists to make, which a title-only block never
# lets it reach.
#
# Deliberately the EXACT four families, not a broader heuristic (per the
# investigation's own "prefer the exact four-title approach" guidance):
# "Engineering Manager", "Product Manager", "Project Manager", "Program
# Manager", "Operations Manager", and every other manager-titled role
# are intentionally NOT covered and remain fully disqualifying.
# Business-segment/region modifiers (Regional, Strategic, Enterprise,
# Channel, Commercial, Named, Key Accounts, etc.) do not imply seniority
# and are deliberately left unfiltered -- "Regional Sales Manager" and
# "Enterprise Account Manager" match this pattern the same as the bare
# family name. Genuine seniority modifiers (Senior, Lead, Director, Head,
# Principal, Staff, VP, Chief, ...) are NOT special-cased here at all --
# they don't need to be: seniority_disqualifier() below only consults
# this exemption when EVERY SENIORITY_TITLE_PATTERN match in the title is
# the bare "manager" token, so "Senior Account Manager" / "Lead Technical
# Account Manager" / "Director of Account Management" / "Head of Customer
# Success" all still disqualify via their own independent token exactly
# as before this change.
#
# Known, accepted residual risk: title-only matching cannot see body-level
# people-management language a stored title doesn't reveal. Live-data
# review found this cuts both ways -- "Manager, Software Technical Account
# Managers" (a real people-management posting) stays correctly blocked
# only because its title uses the PLURAL "Account Managers" (no \b match
# on the singular family pattern below, an incidental but verified-correct
# side effect), while a plain-titled "Technical Account Manager" (Brady
# Corp, Charlotte NC) that opens its body with "you will lead and manage a
# team" would now reach the scorer with nothing in the title to stop it --
# and a live sample of the "Sales Manager" family specifically found 3/14
# postings with explicit body-level team-lead language ("lead a team",
# "hire, coach and develop"), a higher rate than the other three families
# (0/32, 2/26, 0/19). This is not a new gap this exemption introduces --
# no title-only heuristic here has ever been able to see body text for ANY
# title -- it's the reason this exemption routes these four families to
# the real LLM scorer (which does read the full description) instead of
# leaving them permanently unreachable.
_IC_CUSTOMER_FACING_MANAGER_TITLES = re.compile(
    r"\b(?:account manager|customer success manager|technical account manager|sales manager)\b",
    re.IGNORECASE,
)


def seniority_disqualifier(title: str | None) -> str | None:
    """Return a disqualification reason if the title signals a seniority
    level this candidate has no professional experience to legitimately
    claim, else None.

    Deterministic, no LLM call, no profile lookup needed -- seniority-by-
    title is purely a function of the title string. See
    _IC_CUSTOMER_FACING_MANAGER_TITLES above for the one narrow exception:
    Account Manager / Customer Success Manager / Technical Account Manager
    / Sales Manager are not senior-disqualifying on the bare word "manager"
    alone, but any OTHER seniority token anywhere in the title (Senior,
    Lead, Director, Head, Staff, Principal, VP, Chief, ...) still
    disqualifies exactly as it did before this exception existed.
    """
    if not title:
        return None
    matches = list(SENIORITY_TITLE_PATTERN.finditer(title))
    if not matches:
        return None
    if all(m.group(0).lower() == "manager" for m in matches) and _IC_CUSTOMER_FACING_MANAGER_TITLES.search(title):
        return None
    m = matches[0]
    return f"seniority title match: {m.group(0)!r} in {title!r}"


def revalidate_seniority(conn=None, *, dry_run: bool = False) -> dict:
    """Re-run the current seniority disqualifier against already-scored rows.

    For rule changes made after jobs were already scored (the exact failure
    mode above): finds any pre-submission job whose title still triggers
    seniority_disqualifier() and archives it, regardless of what fit_score it
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
    repeatedly -- rows that don't match are never touched, and a second run
    against already-archived matches is a no-op (idempotent). No job record
    is deleted; fit_score, score_reasoning, tailored_resume_path, and
    cover_letter_path are all preserved on the archived row for audit
    purposes.

    2026-08-25 follow-up (IC-manager-title exception): this used to delegate
    to database.reject_jobs_by_title_patterns([SENIORITY_TITLE_PATTERN.pattern],
    ...) -- a second, independent regex evaluation of the raw pattern string
    that bypassed seniority_disqualifier()'s Python-level exemption logic
    entirely (see _IC_CUSTOMER_FACING_MANAGER_TITLES above). That would have
    made this sweep the fourth drifted copy the 2026-08-21 incident this
    module was built to prevent -- running `applypilot revalidate-seniority`
    would have silently re-archived every "Account Manager"/"Customer
    Success Manager"/"Technical Account Manager"/"Sales Manager" job the
    exemption had just unblocked. Now iterates rows directly and calls
    seniority_disqualifier(title) per row, so this can never drift from the
    canonical predicate again regardless of future changes to it. Mirrors
    revalidate_stale_scores' row-iteration shape below (also in this
    module) rather than reject_jobs_by_title_patterns' shared regex-list
    matcher, which stays untouched -- it's also used by the unrelated
    --auto-reject-titles CLI flag's arbitrary user-supplied patterns.
    Behavior difference from the old delegation: apply_category/apply_error
    are no longer set on the archived row (only state, via transition_state)
    -- the same, already-shipped precedent revalidate_stale_scores uses.

    Returns a {"matched", "updated", "sample"} dict; each sample row carries
    url/title/reason (reason replaces the old delegated-call's "pattern" key
    -- unused by any consumer, which only reads url/title).
    """
    from applypilot.database import commit_with_retry, get_connection, transition_state

    if conn is None:
        conn = get_connection()

    protected_states = frozenset(
        {"applied", "applying", "cover_writing", "manual_only", "archived", "ready_to_apply"}
    )
    placeholders = ", ".join("?" for _ in protected_states)
    rows = conn.execute(
        f"""
        SELECT url, title
        FROM jobs
        WHERE applied_at IS NULL
          AND COALESCE(state, '') NOT IN ({placeholders})
        """,
        tuple(protected_states),
    ).fetchall()

    matches = []
    for row in rows:
        title = row["title"] or ""
        reason = seniority_disqualifier(title)
        if reason:
            matches.append({"url": row["url"], "title": title, "reason": reason})

    if dry_run:
        return {"matched": len(matches), "updated": 0, "sample": matches[:1000]}

    updated = 0
    for m in matches:
        ok = transition_state(
            conn,
            m["url"],
            "archived",
            reason=f"revalidate_seniority: {m['reason']}",
            metadata={"title": m["title"][:120]},
            force=True,
        )
        if ok:
            updated += 1

    commit_with_retry(conn)
    return {"matched": len(matches), "updated": updated, "sample": matches[:1000]}


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
