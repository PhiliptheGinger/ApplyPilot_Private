"""Targeted, LLM-free correction for three real bugs found via a real
Gemini-vs-local comparison run (30 of the 362 jobs re-scored by both):
almost every "software_engineering" family job with years_required=None
locally had an explicit, real stated years requirement Gemini's own
reasoning quoted verbatim from the same posting text. Root-caused to
three compounding gaps in extract_years_required, all now fixed in
deterministic_fallback.py:

(a) a spelled-out number repeated parenthetically ("eight (8) years")
    never matched _YEARS_MENTION_RE at all (")" broke the digit-then-
    "years" adjacency the old regex required).
(b) "N+ years" with no "experience" word nearby ("5+ years working on
    complex systems") was invisible -- the old regex hard-required
    "...experience" in the trailing window regardless of a "+".
(c) real section headers other than "Minimum/Required/Basic
    Qualifications" ("Requirements:", "About You:", "You Have", bare
    "Qualifications") were never recognized at all.
(d) the 3000-char extraction window was too short for longer, more
    verbose postings -- one real case's requirements section started at
    character 3196 of a 6208-char description. Bumped to 6000 to match
    scorer.py's own real LLM-prompt window.

Only years_required extraction changed -- family classification and
cs_degree_required are both unaffected by these fixes -- so this re-
derives each affected row's score using the ALREADY-STORED family/
cs_degree values plus a freshly-recomputed years_required and
compensation adjustment, mirroring decision #82's own correction script
one bug class over. No new LLM/local-model calls.
"""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import database  # noqa: E402
from applypilot.scoring.compensation import classify_compensation, compensation_score_adjustment  # noqa: E402
from applypilot.scoring.deterministic_fallback import SCORE_METHOD, deterministic_combine, extract_years_required  # noqa: E402

DRY_RUN = "--apply" not in sys.argv


def parse_reasoning(reasoning: str) -> dict:
    fam = re.search(r"family=(\S+)", reasoning or "")
    yrs = re.search(r"years_required=(\S+)", reasoning or "")
    cs = re.search(r"cs_degree_required=(\S+)", reasoning or "")
    return {
        "family": None if not fam or fam.group(1) == "None" else fam.group(1),
        "years_required": None if not yrs or yrs.group(1) == "None" else int(yrs.group(1)),
        "cs_degree_required": bool(cs and cs.group(1) == "True"),
    }


def main() -> None:
    conn = database.get_connection()
    rows = conn.execute("SELECT * FROM jobs WHERE score_method = ?", (SCORE_METHOD,)).fetchall()
    print(f"total deterministic_fallback rows: {len(rows)}")

    changed = []
    skipped_ineligible = 0
    for row in rows:
        reasoning = row["score_reasoning"] or ""
        if "CORRECTED" in reasoning:
            continue  # already corrected by a prior pass (decision #82/#83)
        if reasoning.strip().startswith("Ineligible"):
            # score_job_deterministic's pre-filter short-circuit never
            # calls classify_family/extract_years_required/extract_cs_degree_required
            # for these -- parse_reasoning's regexes find nothing and
            # silently default to family=None, which deterministic_combine
            # maps to a neutral score=5 regardless of years, incorrectly
            # "correcting" a real ineligibility rejection (score=2) upward.
            # Caught by inspection before applying: real "Full Stack
            # Software Engineer" row (ethical exclusion: 'dod') would have
            # been wrongly bumped 2 -> 5.
            skipped_ineligible += 1
            continue

        old = parse_reasoning(reasoning)
        new_years = extract_years_required(row["full_description"] or "")
        # 2026-09-09 (decision #89): do NOT skip just because new_years ==
        # old_years -- a real bug in an earlier version of this script.
        # decision #89 also changed deterministic_combine's own lookup
        # table (software_engineering years>=1 used to flatten to a
        # single score regardless of the actual number; now tiered), so a
        # row whose EXTRACTED number is unchanged can still need a
        # different SCORE. Always recompute; the final
        # `new_score == row["fit_score"]` check below is what actually
        # decides whether there's anything to correct.
        new_score = deterministic_combine(old["family"], new_years, old["cs_degree_required"])
        comp = classify_compensation(dict(row), conn=conn)
        adjustment, note = compensation_score_adjustment(comp)
        if adjustment:
            new_score = max(1, new_score + adjustment)

        if new_score == row["fit_score"]:
            continue

        changed.append(
            {
                "url": row["url"],
                "title": row["title"],
                "old_score": row["fit_score"],
                "new_score": new_score,
                "family": old["family"],
                "cs_degree": old["cs_degree_required"],
                "old_years": old["years_required"],
                "new_years": new_years,
            }
        )

    print(f"skipped (pre-filter ineligible, not family-classified): {skipped_ineligible}")
    print(f"\nrows whose score actually changes: {len(changed)}")
    for c in changed:
        print(
            f"  {c['title']!r}: score {c['old_score']} -> {c['new_score']}  "
            f"(years {c['old_years']} -> {c['new_years']}, family={c['family']})"
        )

    if DRY_RUN:
        print("\nDRY RUN -- no DB writes. Re-run with --apply to commit these corrections.")
        return

    now = datetime.now(UTC).isoformat()
    for c in changed:
        new_reasoning = (
            f"\n[deterministic fallback, model=corrected] family={c['family']} "
            f"years_required={c['new_years']} cs_degree_required={c['cs_degree']} "
            f"-- CORRECTED 2026-09-09 (years-extraction fixes: parenthetical digits, "
            f"'+'-without-experience, new headers, 6000-char window), was score={c['old_score']} "
            f"years_required={c['old_years']}"
        )
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
            (c["new_score"], new_reasoning, now, c["url"]),
        )
        to_state = "scored" if c["new_score"] >= 8 else "low_score"
        database.transition_state(
            conn,
            c["url"],
            to_state,
            reason=f"corrected fallback score {c['new_score']}/10 (years-extraction fixes)",
            metadata={"score": c["new_score"]},
            force=True,
        )
    conn.commit()
    print(f"\nApplied {len(changed)} corrections and committed.")


if __name__ == "__main__":
    main()
