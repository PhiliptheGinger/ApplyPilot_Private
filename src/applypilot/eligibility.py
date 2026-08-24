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
    # or early-career roles.
    r"\b(?:engineer|developer)\s*[-,]?\s*(?:III|IV|V|VI|3|4|5|6)\b",
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
    mode above): finds any still-pre-tailoring job whose title now matches
    SENIORITY_TITLE_PATTERN and archives it, regardless of what fit_score it
    was previously assigned. Jobs that have already reached the protected
    post-tailoring lifecycle states (tailored / cover_failed / ready_to_apply /
    applying / applied) are intentionally left alone so the state machine
    can continue to operate. Safe to call repeatedly -- reuses
    database.reject_jobs_by_title_patterns(), which already excludes the
    protected states and archived/manual-only rows, so a second run against
    already-archived matches is a no-op (idempotent), and rows that don't
    match are never touched. No job record is deleted; fit_score,
    score_reasoning, tailored_resume_path, and cover_letter_path are all
    preserved on the archived row for audit purposes.

    Returns the same {"matched", "updated", "sample"} dict as
    reject_jobs_by_title_patterns().
    """
    from applypilot.database import reject_jobs_by_title_patterns

    return reject_jobs_by_title_patterns(
        [SENIORITY_TITLE_PATTERN.pattern],
        conn=conn,
        dry_run=dry_run,
        sample_limit=1000,
    )
