"""Detect and narrowly remediate historical scores contaminated by the
pre-Phase-1 fabricated resume source (commit 035484a fixed the scorer to
read the truthful profile.json-derived reference instead of the stale,
hand-maintained ``~/.applypilot/resume.txt``, which had drifted into a
fabricated "Senior Software Engineer, Seattle WA" identity).

Deliberately separate from ``applypilot.eligibility.revalidate_stale_scores``:
that mechanism is date/state-keyed ("was this scored before a rubric fix
regardless of what it says"), this one is content-keyed ("does the stored
reasoning show direct evidence the fabricated identity was actually used").
Staleness and contamination are orthogonal signals -- a row can be stale
without being contaminated (most low_score rows: correctly disqualified for
reasons unrelated to the fake resume) and, in principle, a row scored after
the fix could still show a false positive here. Conflating the two by
widening ``_STALE_SCORE_REVALIDATABLE_STATES`` or ``revalidate_stale_scores``
itself would be the wrong fix (see the 2026-08-27 Phase 4b investigation).

This module has zero dependency on ``applypilot.eligibility`` by design.
"""

from __future__ import annotations

import re
import sqlite3

# High-confidence markers only. Validated against the live database during
# the Phase 4b investigation (2026-08-27): each pattern was manually
# spot-checked against real stored score_reasoning text to confirm it
# captures genuine candidate-identity fabrication (the LLM asserting the
# CANDIDATE is Seattle-based or a senior software engineer) and does NOT
# fire on legitimate job-description content -- e.g. a job actually located
# in Seattle, correctly compared against the true NC-based profile as a
# geography mismatch, is not a match here.
#
# Deliberately rejected: bare "Seattle", bare "senior", or technology-name
# keyword searches (Kubernetes, Docker, cloud, ...). Each of those either
# fires on truthful reasoning (a real geo mismatch mentioning Seattle) or
# on ordinary job-requirement text, producing false-positive/negative rates
# unsuitable for anything that writes to the database. A distinct, harder
# signal -- reasoning crediting technical skills/seniority the profile does
# not actually support -- was identified during the investigation (e.g. the
# federal "ML Ops" apply_failed job crediting Kubernetes/Docker/cloud
# experience) but is NOT included here: it requires comparing claims
# against the live profile content, not a static regex, and is out of
# scope for this narrow mechanism. Rows like that must currently be found
# by manual review, not this tool.
_CANDIDATE_LOCATION_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(based|located) in Seattle,\s*WA,\s*(making them eligible|which matches)", re.I),
    re.compile(r"perfectly matched to candidates? Seattle", re.I),
    re.compile(r"candidates? Seattle location", re.I),
    re.compile(r"US-based in Seattle,\s*WA,\s*they are eligible", re.I),
    re.compile(r"candidate is US-based \(Seattle", re.I),
    re.compile(r"candidate is (located|based) in Seattle", re.I),
)
_CANDIDATE_SENIORITY_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"candidates? seniority", re.I),
    re.compile(r"seniority level \(Senior\)", re.I),
    re.compile(r"candidate (is|as|has)[^.]{0,40}senior software engineer", re.I),
)

_ALL_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, rx)
    for label, group in (
        ("candidate_seattle_location", _CANDIDATE_LOCATION_MARKERS),
        ("candidate_senior_identity", _CANDIDATE_SENIORITY_MARKERS),
    )
    for rx in group
)

# States it is safe to auto-archive in non-dry-run mode: pure scoring/
# tailoring pipeline states with no submitted or in-flight application.
# Deliberately the same states as (but not imported from)
# eligibility._STALE_SCORE_REVALIDATABLE_STATES -- kept as an independent
# literal so this module stays decoupled from eligibility.py.
AUTO_REMEDIABLE_STATES = frozenset({"scored", "tailored", "tailor_failed", "cover_failed"})

# States this mechanism NEVER mutates, regardless of mode. Reported for
# explicit human review only. "archived"/"low_score" are omitted from this
# set deliberately: they're already terminal, so there's nothing to mutate
# either way, but a match there is still reported (it's genuine historical
# contamination worth knowing about, e.g. for a future targeted rescore).
NEVER_MUTATE_STATES = frozenset(
    {"applied", "ready_to_apply", "applying", "apply_failed", "manual_only", "needs_human"}
)


def _match_markers(text: str | None) -> list[str]:
    if not text:
        return []
    return [label for label, rx in _ALL_MARKERS if rx.search(text)]


def find_contaminated_scores(conn: sqlite3.Connection, *, min_score: int = 7) -> list[dict]:
    """Scan scored jobs for high-confidence fabricated-resume markers.

    Read-only -- never writes. Returns the full (untruncated) match list;
    use ``summarize_contamination`` to build a display-ready report.
    """
    rows = conn.execute(
        """
        SELECT url, title, state, fit_score, scored_at, score_reasoning
        FROM jobs
        WHERE score_reasoning IS NOT NULL
          AND fit_score IS NOT NULL
          AND fit_score >= ?
        """,
        (min_score,),
    ).fetchall()

    matches = []
    for row in rows:
        markers = _match_markers(row["score_reasoning"])
        if not markers:
            continue
        matches.append(
            {
                "url": row["url"],
                "title": row["title"] or "",
                "state": row["state"],
                "fit_score": row["fit_score"],
                "scored_at": row["scored_at"],
                "markers": markers,
                "reasoning_excerpt": (row["score_reasoning"] or "")[:600],
            }
        )
    return matches


def summarize_contamination(matches: list[dict], *, sample_limit: int = 20) -> dict:
    """Build a display-ready report dict from a match list."""
    by_state: dict[str, int] = {}
    by_score: dict[int, int] = {}
    for m in matches:
        by_state[m["state"]] = by_state.get(m["state"], 0) + 1
        by_score[m["fit_score"]] = by_score.get(m["fit_score"], 0) + 1

    auto_remediable = [m for m in matches if m["state"] in AUTO_REMEDIABLE_STATES]
    needs_review = [m for m in matches if m["state"] in NEVER_MUTATE_STATES]
    archived_already = [
        m for m in matches if m["state"] not in AUTO_REMEDIABLE_STATES and m["state"] not in NEVER_MUTATE_STATES
    ]

    return {
        "matched": len(matches),
        "by_state": by_state,
        "by_score": by_score,
        "auto_remediable_count": len(auto_remediable),
        "needs_review_count": len(needs_review),
        "already_terminal_count": len(archived_already),
        "sample": matches[:sample_limit],
        "needs_review_sample": needs_review[:sample_limit],
    }


def remediate_contaminated_scores(conn: sqlite3.Connection, *, min_score: int = 7, dry_run: bool = True) -> dict:
    """Archive contaminated rows -- but ONLY those in ``AUTO_REMEDIABLE_STATES``.

    Rows in ``NEVER_MUTATE_STATES`` (applied, ready_to_apply, applying,
    apply_failed, manual_only, needs_human) and already-terminal rows
    (archived, low_score) are never mutated by this function, regardless of
    ``dry_run`` -- they are always reported only, for a separate, explicit
    human decision. ``fit_score``/``score_reasoning``/``tailored_resume_path``/
    ``cover_letter_path`` are preserved on any archived row for audit,
    matching the existing convention in ``revalidate_stale_scores`` and
    ``revalidate_seniority``.
    """
    from applypilot.database import commit_with_retry, transition_state

    matches = find_contaminated_scores(conn, min_score=min_score)
    summary = summarize_contamination(matches)
    to_archive = [m for m in matches if m["state"] in AUTO_REMEDIABLE_STATES]

    if dry_run:
        return {**summary, "would_archive": len(to_archive), "archived": 0, "archive_sample": to_archive[:20]}

    archived = 0
    for m in to_archive:
        ok = transition_state(
            conn,
            m["url"],
            "archived",
            reason=f"contamination_remediation: markers={','.join(m['markers'])}",
            metadata={"fit_score": m["fit_score"], "title": m["title"][:120]},
            force=True,
        )
        if ok:
            archived += 1
    commit_with_retry(conn)
    return {**summary, "would_archive": len(to_archive), "archived": archived, "archive_sample": to_archive[:20]}
