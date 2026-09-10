"""Real Gemini scoring run against the exact 362 jobs the local
deterministic-fallback scorer scored on 2026-09-08 (CLAUDE.md decisions
#82-84), requested directly: "let's see how it compares to what we got
from the local model yesterday." Serves three purposes at once (per the
same-session discussion before running this): (1) accuracy comparison
against the local scores, (2) fresh REASONING/KEYWORDS material for
Gemini-reasoning distillation at a much bigger n than decision #77's n=23,
(3) a fresh, independently-sampled batch to eventually revalidate the
escalation-trigger keyword list against (decision #81's blocker -- all 24
prior clean positives were already consumed by the original validation).

Read-only with respect to the `jobs` table: uses the real, unmodified
scorer.score_job() (which never writes to the DB itself, per its own
docstring) and stores results in a separate JSON file alongside the
existing deterministic_fallback scores already in the DB -- nothing here
overwrites or resets any row. Committing these as the jobs' real scores
(if wanted) is a separate, explicit follow-up step.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot.config import load_env  # noqa: E402

load_env()  # MUST run before llm.py (or anything importing it) is imported -- gotcha #3/#8

from applypilot import database  # noqa: E402
from applypilot.config import load_profile  # noqa: E402
from applypilot.scoring.deterministic_fallback import SCORE_METHOD  # noqa: E402
from applypilot.scoring.resume_router import render_profile_reference  # noqa: E402
from applypilot.scoring.scorer import score_job  # noqa: E402

OUT_PATH = REPO_ROOT / "data/experiments/gemini_vs_local_20260908_comparison/results.json"


def main() -> None:
    # 2026-09-09: autonomous-loop tick found the user's three-way question
    # (small bounded batch vs. full 362 vs. wait) unanswered. Real testing
    # that same session showed Gemini's flash tier hitting persistent,
    # unpredictable 503s on our heavy score prompt (33-262s/job observed) --
    # committing to the full 362 unattended risked many hours against a
    # visibly unstable service the user hadn't signed off on. A bounded,
    # reversible 30-job slice (no DB writes, just this JSON file) continues
    # the already-authorized "let's see how it compares"/"yep go ahead"
    # work without gambling on the costlier, still-open choice.
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    conn = database.get_connection()
    rows = conn.execute(
        "SELECT * FROM jobs WHERE score_method = ?",
        (SCORE_METHOD,),
    ).fetchall()
    jobs = [dict(r) for r in rows]
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} jobs to re-score with real Gemini")

    profile = load_profile()
    resume_text = render_profile_reference(profile)

    results = []
    # Resume progress if this was interrupted and re-run.
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        done_urls = {r["url"] for r in results}
        jobs = [j for j in jobs if j["url"] not in done_urls]
        print(f"resuming: {len(results)} already done, {len(jobs)} remaining")

    for i, job in enumerate(jobs):
        t0 = time.time()
        r = score_job(resume_text, job, profile=profile, conn=conn)
        elapsed = time.time() - t0
        results.append(
            {
                "url": job["url"],
                "title": job["title"],
                "local_fit_score": job["fit_score"],
                "local_reasoning": job["score_reasoning"],
                "gemini_score": r.get("score"),
                "gemini_eligibility": r.get("eligibility"),
                "gemini_keywords": r.get("keywords"),
                "gemini_reasoning": r.get("reasoning"),
                "gemini_model": r.get("model"),
                "gemini_escalated": r.get("escalated"),
                "gemini_error": r.get("error"),
                "elapsed": elapsed,
                "scored_at": datetime.now(UTC).isoformat(),
            }
        )
        if (i + 1) % 5 == 0 or r.get("error"):
            OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
            status = r.get("error") or f"score={r.get('score')} model={r.get('model')}"
            print(f"  [{i + 1}/{len(jobs)}] {job['title'][:50]:50s} {status} ({elapsed:.1f}s)")

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDone. {len(results)} results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
