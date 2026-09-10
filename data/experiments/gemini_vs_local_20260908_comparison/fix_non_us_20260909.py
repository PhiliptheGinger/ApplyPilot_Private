"""Correction pass for the non-US/foreign-language posting detection fix
(CLAUDE.md decision #90): four real Accenture/international postings
(Madrid, Buenos Aires, Brussels, Warsaw) scored 7-9 by the deterministic
fallback because `_check_ineligible`'s description-language patterns only
recognized specific ENGLISH phrasings ("based in Spain"), never the
posting's own text being substantially non-English. Fixed in
scorer._INELIGIBLE_DESC_PATTERNS (shared by both scoring paths).

This re-runs `_check_ineligible` fresh against every deterministic_fallback
row's already-stored full_description/title/location and flips any row
that NOW gets caught (and wasn't already an "Ineligible: ..." row) to the
same 2-point ineligible score `score_job_deterministic` itself would have
produced, tagging the reasoning so it's auditable and revalidation-eligible
like every other correction this session.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import database  # noqa: E402
from applypilot.scoring.deterministic_fallback import SCORE_METHOD  # noqa: E402
from applypilot.scoring.scorer import _check_ineligible, _classify_ineligibility  # noqa: E402

DRY_RUN = "--apply" not in sys.argv


def main() -> None:
    conn = database.get_connection()
    rows = conn.execute("SELECT * FROM jobs WHERE score_method = ?", (SCORE_METHOD,)).fetchall()
    print(f"total deterministic_fallback rows: {len(rows)}")

    changed = []
    for row in rows:
        reasoning = row["score_reasoning"] or ""
        if reasoning.strip().startswith("Ineligible"):
            continue  # already ineligible for some other reason

        job = dict(row)
        reason = _check_ineligible(job, profile=None)
        if not reason:
            continue
        if "non-US" not in reason and "non_us" not in reason:
            continue  # a different ineligibility reason newly firing isn't in scope here

        if row["fit_score"] == 2:
            continue  # already scored 2, nothing to correct

        changed.append({"url": row["url"], "title": row["title"], "old_score": row["fit_score"], "reason": reason})

    print(f"\nrows whose score actually changes: {len(changed)}")
    for c in changed:
        print(f"  {c['title']!r}: score {c['old_score']} -> 2  ({c['reason'][:80]})")

    if DRY_RUN:
        print("\nDRY RUN -- no DB writes. Re-run with --apply to commit these corrections.")
        return

    now = datetime.now(UTC).isoformat()
    for c in changed:
        new_reasoning = f"\nIneligible: {c['reason']}. -- CORRECTED 2026-09-09 (decision #90 non-US detection fix), was score={c['old_score']}"
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, eligibility = ?, scored_at = ? WHERE url = ?",
            (2, new_reasoning, _classify_ineligibility(c["reason"]), now, c["url"]),
        )
        database.transition_state(
            conn,
            c["url"],
            "low_score",
            reason=f"corrected fallback score 2/10 (non-US detection fix): {c['reason'][:80]}",
            metadata={"score": 2},
            force=True,
        )
    conn.commit()
    print(f"\nApplied {len(changed)} corrections and committed.")


if __name__ == "__main__":
    main()
