"""Decomposed qualify-then-quantify local scorer, per real research (LLM-
as-judge best practice: decompose into narrow criteria, aggregate
deterministically -- see tonight's research summary) and the user's own
"qualify vs quantify" framing, directly motivated by the single real test
that showed qwen3:1.7b correctly IDENTIFYING a disqualifier in its own
reasoning ("does not meet the required seniority level") but failing to
APPLY the resulting rule (scored it 6 instead of 1-2).

Architecture:
  1. QUANTIFY-FIRST, zero LLM cost: reuse scorer.py's existing
     _check_ineligible (geography/seniority/degree/clearance/commission)
     -- already-proven, already-tested hard gates. If it fires: score=2,
     no LLM call needed at all.
  2. QUALIFY, one narrow LLM call: only for jobs that pass step 1. Reuses
     the EXACT tier language from the real production SCORE_PROMPT_TEMPLATE
     (the 9-10/7-8/5-6/3-4 descriptions are already proven-good text --
     not reinvented) but strips out everything the model previously had to
     compute itself (the disqualifying checks), since step 1 already
     confirmed those don't apply. The model's ONLY job is: given this is
     already an eligible, non-disqualified posting, which of these four
     tiers best fits?
  3. QUANTIFY-SECOND, zero LLM cost: map the returned tier label to a
     fixed numeric anchor (9, 7, 5, 3).

Tested against the same clean (scored_at >= 2026-08-28, non-garbage-
description) dataset used all night.
"""
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

os.environ["APPLYPILOT_LOCAL_OLLAMA_NATIVE"] = "1"

from applypilot import config  # noqa: E402

config.load_env()

from applypilot import database  # noqa: E402
from applypilot.config import load_profile  # noqa: E402
from applypilot.llm import LLMClient, ModelEntry, local_openai_base_url  # noqa: E402
from applypilot.scoring.scorer import (  # noqa: E402
    _build_candidate_summary,
    _build_location_context,
    _check_ineligible,
)

LOCAL_URL = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", "qwen3:1.7b")

# Reuses the exact tier language from scorer.py's real SCORE_PROMPT_TEMPLATE
# (9-10 / 7-8 / 5-6 / 3-4 bands) -- proven-good text, not reinvented. The
# 1-2 disqualifier band is deliberately omitted: step 1 (_check_ineligible)
# already confirmed none of those hard disqualifiers apply before this
# prompt is ever sent.
_QUALIFY_SYSTEM = """You are classifying how well a candidate fits a job posting. \
A separate, already-completed check has confirmed this posting does NOT have any \
of these disqualifying issues: non-US-only geography, a Senior/Staff/Principal/Lead/ \
Architect/Director/Manager/VP/Chief title or scope, a required advanced degree the \
candidate lacks, a required security clearance, or 3+ years of required professional \
software engineering experience. Your ONLY job is to pick which ONE of these four \
tiers best describes the fit, given the candidate's real background below.

THE CANDIDATE: {candidate_summary}

TIER A: Entry-level / no-experience-required IT support, help desk, desktop support, \
technical support, customer support, systems administration, or network engineering \
role. Matches the CompTIA A+ certification and hands-on troubleshooting / customer- \
facing background directly. Does not require a CS degree or an advanced cert beyond A+.
TIER B: Same IT-support family as TIER A but with 1-2 stretch requirements (e.g. \
"1-2 years preferred" rather than required, or a nice-to-have second cert), OR a \
genuinely entry-level / new-grad / junior software, backend, or Python role that \
explicitly does not require prior professional software engineering experience or a \
CS degree.
TIER C: IT support role requiring 2+ years of experience the candidate doesn't have, \
OR a junior software/backend role nominally wanting ~1 year of experience where the \
rest of the requirements are learnable and stack-agnostic.
TIER D: Software/backend/data engineering role requiring 2+ years of professional \
experience or a CS degree as a hard requirement, even if the candidate's personal \
Python projects touch some of the listed tech stack.

First write ONE sentence comparing this specific posting's actual requirements to \
the candidate's specific real background above (name at least one concrete \
requirement and one concrete candidate fact). Then, on its own final line, output \
exactly: TIER: A, TIER: B, TIER: C, or TIER: D."""

_TIER_TO_SCORE = {"A": 9, "B": 7, "C": 5, "D": 3}


def local_only_client() -> LLMClient:
    client = LLMClient(LOCAL_URL, LOCAL_MODEL, "", quality=True)
    client._fallback_chain = [ModelEntry(LOCAL_MODEL, "local", local_openai_base_url(LOCAL_URL), "")]
    return client


def qualify_tier(client, profile, job) -> str | None:
    candidate_summary = _build_candidate_summary(profile)
    system = _QUALIFY_SYSTEM.format(candidate_summary=candidate_summary)
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:4000]}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": job_text}]
    try:
        resp = client.chat(messages, max_tokens=700, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        return None
    m = re.search(r"TIER:\s*([ABCD])", resp or "", re.IGNORECASE)
    return m.group(1).upper() if m else None


def decomposed_score(client, profile, job) -> tuple[int, str]:
    reason = _check_ineligible(job, profile)
    if reason:
        return 2, f"gate:{reason[:40]}"
    tier = qualify_tier(client, profile, job)
    if tier is None:
        return None, "qualify_failed"
    return _TIER_TO_SCORE[tier], f"tier:{tier}"


def main():
    conn = database.get_connection()
    profile = load_profile()

    rows = conn.execute(
        """
        SELECT url, title, site, location, fit_score, full_description FROM jobs
        WHERE fit_score IS NOT NULL
          AND (score_reasoning IS NULL OR score_reasoning NOT LIKE '%Ineligible:%')
          AND full_description IS NOT NULL
          AND scored_at >= '2026-08-28'
          AND full_description NOT LIKE '%Replace All Your Work Tools%'
          AND full_description NOT LIKE '%official careers website%'
        """
    ).fetchall()
    rows = [dict(r) for r in rows]
    positives = [r for r in rows if r["fit_score"] >= 8]
    negatives = [r for r in rows if r["fit_score"] < 8]
    import random

    random.seed(20260906)
    random.shuffle(positives)
    random.shuffle(negatives)
    N = int(os.environ.get("N_SAMPLE", "10"))
    sample = positives[: N // 2] + negatives[: N - N // 2]
    random.shuffle(sample)
    print(f"clean pool: {len(rows)} (pos={len(positives)}, neg={len(negatives)}); testing n={len(sample)}")

    client = local_only_client()
    results = []
    for i, job in enumerate(sample):
        t0 = time.time()
        score, detail = decomposed_score(client, profile, job)
        dt = time.time() - t0
        results.append({**job, "local_score": score, "detail": detail, "seconds": round(dt, 1)})
        print(f"[{i+1}/{len(sample)}] real={job['fit_score']} local={score} ({dt:.1f}s, {detail}) {job['title'][:45]}")

    have = [r for r in results if r["local_score"] is not None]
    n = len(have)
    print(f"\nn={len(results)}, parseable={n} ({100*n/len(results):.0f}%)")
    if n == 0:
        return
    exact = sum(1 for r in have if r["local_score"] == r["fit_score"])
    within2 = sum(1 for r in have if abs(r["local_score"] - r["fit_score"]) <= 2)
    real_pos = [r for r in have if r["fit_score"] >= 8]
    real_neg = [r for r in have if r["fit_score"] < 8]
    tp = sum(1 for r in real_pos if r["local_score"] >= 8)
    fn = len(real_pos) - tp
    pred_pos = [r for r in have if r["local_score"] >= 8]
    fp = sum(1 for r in pred_pos if r["fit_score"] < 8)
    recall = tp / len(real_pos) if real_pos else float("nan")
    precision = tp / len(pred_pos) if pred_pos else float("nan")
    gate_agree = sum(1 for r in have if (r["local_score"] >= 8) == (r["fit_score"] >= 8)) / n
    avg_seconds = sum(r["seconds"] for r in have) / n
    print(f"exact match: {100*exact/n:.0f}%  within +-2: {100*within2/n:.0f}%  gate_agree: {100*gate_agree:.0f}%")
    print(f"real positives (>=8): n={len(real_pos)}  TP={tp} FN={fn}  recall={recall:.2f} precision={precision:.2f}")
    print(f"avg latency: {avg_seconds:.1f}s")

    import json

    with open(Path(__file__).parent / "decomposed_local_scorer_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
