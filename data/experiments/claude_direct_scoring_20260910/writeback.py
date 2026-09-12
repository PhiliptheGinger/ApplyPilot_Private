"""Write Claude's direct scores (claude_scores.json, produced by hand after
reading to_score.json) to the real DB via the production `_flush_score_batch`
path -- reused rather than reimplemented so eligibility-driven archival,
scored/low_score state transitions, and the score_method audit tag all
behave identically to a real `applypilot run score` run (CLAUDE.md's
"one source of truth" convention, same precedent as the deterministic
fallback scorer, decision #76-77).

score_method is tagged "claude_direct" so these rows are visibly
distinguishable from both real Gemini scores and the local qwen-based
deterministic_fallback scores, and can be revalidated later the same way.

Also merges in prefiltered.json (the deterministic pre-filter results from
select_batch.py) -- those never needed manual scoring but still need to be
written the same way.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot.config import load_env  # noqa: E402

load_env()

from applypilot import database  # noqa: E402
from applypilot.scoring.scorer import _flush_score_batch  # noqa: E402

OUT_DIR = REPO_ROOT / "data/experiments/claude_direct_scoring_20260910"


def main() -> None:
    prefiltered = json.loads((OUT_DIR / "prefiltered.json").read_text(encoding="utf-8"))
    claude_scores = json.loads((OUT_DIR / "claude_scores.json").read_text(encoding="utf-8"))

    batch = []
    for r in prefiltered:
        batch.append(
            {
                "url": r["url"],
                "score": r["score"],
                "keywords": r["keywords"],
                "reasoning": r["reasoning"],
                "eligibility": r["eligibility"],
            }
        )
    for r in claude_scores:
        batch.append(
            {
                "url": r["url"],
                "score": r["score"],
                "keywords": r.get("keywords", ""),
                "reasoning": r["reasoning"],
                "eligibility": r.get("eligibility", "eligible"),
            }
        )

    print(f"writing {len(batch)} scores ({len(prefiltered)} prefiltered + {len(claude_scores)} claude-direct)")

    conn = database.get_connection()

    missing = []
    for r in batch:
        row = conn.execute("SELECT 1 FROM jobs WHERE url = ?", (r["url"],)).fetchone()
        if not row:
            missing.append(r["url"])
    if missing:
        print(f"ABORTING: {len(missing)} URL(s) not found in jobs table:")
        for u in missing:
            print(f"  {u}")
        return

    now = datetime.now(UTC).isoformat()
    _flush_score_batch(conn, batch, now, score_method="claude_direct")
    conn.commit()
    print("done, committed.")


if __name__ == "__main__":
    main()
