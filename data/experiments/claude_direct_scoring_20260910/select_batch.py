"""Select a fresh batch of real pending jobs for Claude (this interactive
session) to score directly, per user request 2026-09-10: Gemini and
OpenAI are both quota-exhausted for hours (Gemini flash ~5-12h, OpenAI
~24 days per decision #67c) so "have Claude do the scoring instead" is a
genuine substitute, not just a comparison exercise like decisions #88-90.

Uses the REAL production selection query (get_jobs_by_stage("pending_score",
max_age_days=14)) so this batch is representative of what `applypilot run
score` would process next -- not a hand-picked convenience sample. Runs the
REAL deterministic pre-filter (_check_ineligible) first and shortcuts those
jobs exactly like score_job() does (no LLM/manual reasoning needed) so I
only spend manual-reasoning effort on jobs that genuinely need it.

Avoids jobs already consumed by prior experiment batches (gemini_vs_local
comparison batches 1-3) so this is genuinely fresh ground truth, useful for
the still-open escalation-trigger revalidation (Future Work item 2).

Writes two files:
  - prefiltered.json: jobs the deterministic pre-filter already resolved
    (score=2, ineligible reason) -- written straight to score_writeback.json
    format, no manual work needed.
  - to_score.json: remaining jobs with rendered prompt text, for me to
    read and score directly in-conversation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot.config import load_env  # noqa: E402

load_env()

from applypilot import database  # noqa: E402
from applypilot.config import load_profile  # noqa: E402
from applypilot.database import get_jobs_by_stage  # noqa: E402
from applypilot.scoring.resume_router import render_profile_reference  # noqa: E402
from applypilot.scoring.scorer import (  # noqa: E402
    _build_candidate_summary,
    _build_location_context,
    _check_ineligible,
    _classify_ineligibility,
)

OUT_DIR = REPO_ROOT / "data/experiments/claude_direct_scoring_20260910"

PRIOR_BATCH_FILES = [
    REPO_ROOT / "data/experiments/gemini_vs_local_20260908_comparison/job_texts_for_claude.json",
    REPO_ROOT / "data/experiments/gemini_vs_local_20260908_comparison/larger_batch_jobs.jsonl",
    REPO_ROOT / "data/experiments/gemini_vs_local_20260908_comparison/batch3_jobs.jsonl",
]


def _prior_urls() -> set[str]:
    urls: set[str] = set()
    for p in PRIOR_BATCH_FILES:
        if not p.exists():
            continue
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            urls.update(j["url"] for j in data)
        else:  # .jsonl
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    urls.add(json.loads(line)["url"])
    return urls


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    # 2026-09-11: the within-14-day pending_score backlog was fully cleared
    # by a real large-scale Gemini run (decision #117) -- select_batch.py's
    # original default returned 0 candidates. This audit's real purpose
    # (stress-testing the rubric/extraction logic against diverse real
    # posting text) doesn't depend on a job's age, only real production
    # discovery scoping does -- so an optional second CLI arg widens the
    # age window (0 = unbounded) without changing the funnel's own
    # max_age_days=14 policy anywhere else in the pipeline.
    max_age_days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    conn = database.get_connection()
    profile = load_profile()
    resume_text = render_profile_reference(profile)

    exclude = _prior_urls()
    print(f"excluding {len(exclude)} URLs already used in prior comparison batches")

    # Pull generously over-limit since we'll drop prior-batch URLs and want
    # `limit` genuinely fresh jobs remaining after that filter.
    jobs = get_jobs_by_stage(conn=conn, stage="pending_score", max_age_days=max_age_days, limit=limit * 3)
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]
    jobs = [j for j in jobs if j["url"] not in exclude]
    jobs = jobs[:limit]
    print(f"{len(jobs)} fresh jobs selected for this batch")

    candidate_summary = _build_candidate_summary(profile)
    location_context = _build_location_context(profile)

    prefiltered = []
    to_score = []
    for job in jobs:
        reason = _check_ineligible(job, profile)
        if reason:
            prefiltered.append(
                {
                    "url": job["url"],
                    "title": job["title"],
                    "score": 2,
                    "keywords": "",
                    "reasoning": f"Ineligible: {reason}.",
                    "eligibility": _classify_ineligibility(reason),
                }
            )
            continue
        job_text = (
            f"TITLE: {job['title']}\n"
            f"COMPANY: {job['site']}\n"
            f"LOCATION: {job.get('location', 'N/A')}\n\n"
            f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
        )
        to_score.append(
            {
                "url": job["url"],
                "title": job["title"],
                "site": job["site"],
                "location": job.get("location"),
                "job_text": job_text,
            }
        )

    print(f"  {len(prefiltered)} resolved by deterministic pre-filter (no manual scoring needed)")
    print(f"  {len(to_score)} need direct scoring")

    (OUT_DIR / "prefiltered.json").write_text(json.dumps(prefiltered, indent=2), encoding="utf-8")
    (OUT_DIR / "to_score.json").write_text(json.dumps(to_score, indent=2), encoding="utf-8")
    (OUT_DIR / "candidate_summary.txt").write_text(candidate_summary, encoding="utf-8")
    (OUT_DIR / "location_context.txt").write_text(location_context, encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'to_score.json'} -- ready for direct scoring.")


if __name__ == "__main__":
    main()
