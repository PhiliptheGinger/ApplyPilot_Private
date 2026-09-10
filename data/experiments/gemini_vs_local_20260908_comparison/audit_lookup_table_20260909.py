"""Full re-derivation audit for decision #89's deterministic_combine
lookup-table fix (software_engineering years>=1 used to flatten to a
single score; now tiered 1/2/3+ per the rubric's own stated bands).

Unlike fix_years_extraction_20260909.py (which skips any row already
marked CORRECTED by a prior pass), this audit deliberately IGNORES that
guard -- the lookup-table change affects every already-corrected row too,
since those were corrected using whatever deterministic_combine looked
like AT THE TIME, which for years>=1 software_engineering rows was the
old flat rule. Re-parses each row's CURRENT reasoning (whichever pass
last wrote it) for family/years/cs_degree, recomputes via the CURRENT
deterministic_combine, and flags any real mismatch against the stored
fit_score. Still skips "Ineligible: ..." pre-filter rows (never
family-classified at all, so nothing to re-derive).
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
from applypilot.scoring.deterministic_fallback import SCORE_METHOD, deterministic_combine  # noqa: E402

DRY_RUN = "--apply" not in sys.argv


def parse_reasoning(reasoning: str) -> dict:
    fam = re.search(r"family=(\S+)", reasoning or "")
    yrs = re.search(r"years_required=(\S+)", reasoning or "")
    cs = re.search(r"cs_degree_required=(\S+)", reasoning or "")
    return {
        "family": None if not fam or fam.group(1) == "None" else fam.group(1),
        "years_required": None if not yrs or yrs.group(1) == "None" or yrs.group(1) == "?" else int(yrs.group(1)),
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
        if reasoning.strip().startswith("Ineligible"):
            skipped_ineligible += 1
            continue

        old = parse_reasoning(reasoning)
        new_score = deterministic_combine(old["family"], old["years_required"], old["cs_degree_required"])
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
                "years": old["years_required"],
                "cs_degree": old["cs_degree_required"],
            }
        )

    print(f"skipped (pre-filter ineligible): {skipped_ineligible}")
    print(f"\nrows whose score actually changes: {len(changed)}")
    for c in changed:
        print(f"  {c['title']!r}: score {c['old_score']} -> {c['new_score']}  (years={c['years']}, family={c['family']})")

    if DRY_RUN:
        print("\nDRY RUN -- no DB writes. Re-run with --apply to commit these corrections.")
        return

    now = datetime.now(UTC).isoformat()
    for c in changed:
        new_reasoning = (
            f"\n[deterministic fallback, model=corrected] family={c['family']} "
            f"years_required={c['years']} cs_degree_required={c['cs_degree']} "
            f"-- CORRECTED 2026-09-09 (decision #89 lookup-table re-tiering), was score={c['old_score']}"
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
            reason=f"corrected fallback score {c['new_score']}/10 (lookup-table re-tiering)",
            metadata={"score": c["new_score"]},
            force=True,
        )
    conn.commit()
    print(f"\nApplied {len(changed)} corrections and committed.")


if __name__ == "__main__":
    main()
