"""Targeted, LLM-free correction for decision #82's real accuracy fix:
extract_years_required now recognizes "Minimum/Required/Basic
Qualifications" section headers, not just inline "required" phrasing.

Only years_required extraction was buggy -- family classification and
cs_degree_required were both already correct and deterministic-combine's
own logic is unchanged. So this re-derives each affected row's final score
by re-running the SAME deterministic pipeline (extract_years_required with
the FIXED code -> deterministic_combine -> compensation adjustment)
without a single new LLM/local-model call, using the family/cs_degree
values already stored in that row's own score_reasoning.

Only touches deterministic_fallback-scored rows whose recomputed
years_required differs from what's embedded in score_reasoning AND whose
final score actually changes as a result -- every row that was already
correct is left untouched.
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
    rows = conn.execute(
        "SELECT * FROM jobs WHERE score_method = ?",
        (SCORE_METHOD,),
    ).fetchall()
    print(f"total deterministic_fallback rows: {len(rows)}")

    changed = []
    for row in rows:
        old = parse_reasoning(row["score_reasoning"] or "")
        new_years = extract_years_required(row["full_description"] or "")
        if new_years == old["years_required"]:
            continue  # no change in extraction -- nothing to correct

        new_score = deterministic_combine(old["family"], new_years, old["cs_degree_required"])
        comp = classify_compensation(dict(row), conn=conn)
        adjustment, note = compensation_score_adjustment(comp)
        if adjustment:
            new_score = max(1, new_score + adjustment)

        if new_score == row["fit_score"]:
            continue  # extraction changed but didn't move the final score

        changed.append(
            {
                "url": row["url"],
                "title": row["title"],
                "old_score": row["fit_score"],
                "new_score": new_score,
                "old_years": old["years_required"],
                "new_years": new_years,
                "family": old["family"],
                "note": note,
            }
        )

    print(f"\nrows whose score actually changes: {len(changed)}")
    for c in changed:
        print(f"  {c['title']!r}: score {c['old_score']} -> {c['new_score']}  (years {c['old_years']} -> {c['new_years']}, family={c['family']})")

    if DRY_RUN:
        print("\nDRY RUN -- no DB writes. Re-run with --apply to commit these corrections.")
        return

    now = datetime.now(UTC).isoformat()
    for c in changed:
        new_reasoning = (
            f"\n[deterministic fallback, model=corrected] family={c['family']} "
            f"years_required={c['new_years']} cs_degree_required=? "
            f"-- CORRECTED 2026-09-08 (decision #82 years-required header-context fix), "
            f"was score={c['old_score']} years_required={c['old_years']}"
        )
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
            (c["new_score"], new_reasoning, now, c["url"]),
        )
        # Re-run state transition consistent with the normal scorer's own logic
        from applypilot.database import transition_state

        to_state = "scored" if c["new_score"] >= 8 else "low_score"
        transition_state(
            conn,
            c["url"],
            to_state,
            reason=f"corrected fallback score {c['new_score']}/10 (decision #82 fix)",
            metadata={"score": c["new_score"]},
            force=True,
        )
    conn.commit()
    print(f"\nApplied {len(changed)} corrections and committed.")


if __name__ == "__main__":
    main()
