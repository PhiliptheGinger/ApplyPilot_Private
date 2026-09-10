"""Targeted, LLM-free correction for two real accuracy bugs found via a
manual spot-check of the 2026-09-08 backlog run (362 jobs scored by
deterministic_fallback):

1. Clinical-license titles (Physician, Medical Assistant/LPN, etc.) were
   getting FAMILY: customer_facing_or_sales from the small model despite
   _FAMILY_SYSTEM already telling it clinical work belongs in
   specialized_or_other -- fixed with a deterministic title override
   (_CLINICAL_LICENSE_TITLE_RE) that now bypasses the LLM call for these
   titles entirely.
2. extract_cs_degree_required never got decision #82's section-header
   fix (a degree stated under "Minimum Qualifications" with no inline
   "required" word was missed), and separately never matched a Unicode
   curly apostrophe ("Bachelor's" vs "Bachelor's") -- both fixed in
   extract_cs_degree_required/_CS_DEGREE_RE.

Neither bug touched years_required or the family classifier's LLM output
for non-clinical titles, so this re-derives each affected row's score
using ONLY the already-stored family/years plus a freshly-recomputed
cs_degree_required (and, for clinical titles, a forced family override) --
no new LLM/local-model calls, mirroring decision #82's own correction
script (fix_years_required_regression.py) one bug class over.
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
from applypilot.scoring.deterministic_fallback import (  # noqa: E402
    SCORE_METHOD,
    _CLINICAL_LICENSE_TITLE_RE,
    deterministic_combine,
    extract_cs_degree_required,
)

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
    for row in rows:
        # Already-corrected rows (decision #82) carry family=None in their
        # rewritten reasoning ("model=corrected") -- nothing to re-derive.
        if "CORRECTED" in (row["score_reasoning"] or ""):
            continue

        old = parse_reasoning(row["score_reasoning"] or "")
        new_family = old["family"]
        reason_bits = []

        if _CLINICAL_LICENSE_TITLE_RE.search(row["title"] or "") and old["family"] != "specialized_or_other":
            new_family = "specialized_or_other"
            reason_bits.append(f"family {old['family']}->specialized_or_other (clinical title override)")

        new_cs_degree = extract_cs_degree_required(row["full_description"] or "")
        if new_cs_degree != old["cs_degree_required"]:
            reason_bits.append(f"cs_degree_required {old['cs_degree_required']}->{new_cs_degree}")

        if not reason_bits:
            continue  # neither fix changes this row's inputs

        new_score = deterministic_combine(new_family, old["years_required"], new_cs_degree)
        comp = classify_compensation(dict(row), conn=conn)
        adjustment, note = compensation_score_adjustment(comp)
        if adjustment:
            new_score = max(1, new_score + adjustment)

        if new_score == row["fit_score"]:
            continue  # inputs changed but didn't move the final score

        changed.append(
            {
                "url": row["url"],
                "title": row["title"],
                "old_score": row["fit_score"],
                "new_score": new_score,
                "new_family": new_family,
                "new_cs_degree": new_cs_degree,
                "years_required": old["years_required"],
                "reason_bits": reason_bits,
            }
        )

    print(f"\nrows whose score actually changes: {len(changed)}")
    for c in changed:
        print(f"  {c['title']!r}: score {c['old_score']} -> {c['new_score']}  ({'; '.join(c['reason_bits'])})")

    if DRY_RUN:
        print("\nDRY RUN -- no DB writes. Re-run with --apply to commit these corrections.")
        return

    now = datetime.now(UTC).isoformat()
    for c in changed:
        new_reasoning = (
            f"\n[deterministic fallback, model=corrected] family={c['new_family']} "
            f"years_required={c['years_required']} cs_degree_required={c['new_cs_degree']} "
            f"-- CORRECTED 2026-09-09 (family/cs_degree fixes), was score={c['old_score']}"
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
            reason=f"corrected fallback score {c['new_score']}/10 (family/cs_degree fix)",
            metadata={"score": c["new_score"]},
            force=True,
        )
    conn.commit()
    print(f"\nApplied {len(changed)} corrections and committed.")


if __name__ == "__main__":
    main()
