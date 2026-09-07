"""Escalation architecture, per user design: ask qwen3:1.7b (fast, cheap)
the family question TWICE at nonzero temperature. If both answers agree,
trust it -- no escalation needed. If they disagree, that's genuine,
measured evidence of low confidence (self-consistency, a real technique --
not guessing at or trusting the model's own self-reported confidence), so
escalate that one job to qwen3:8b (slow, more reliable on ambiguous cases
per tonight's spot-check) for a tiebreak.

Goal: combine 1.7b's speed on the (probably-majority) easy cases with 8b's
accuracy on the (probably-minority) genuinely ambiguous ones, without
paying 8b's ~77s/call cost on every single job.

Reuses run_atomic_facts_scorer.py's proven prompt, regex extraction, and
deterministic_combine table unchanged -- only the family-classification
call path changes.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

os.environ["APPLYPILOT_LOCAL_OLLAMA_NATIVE"] = "1"

from applypilot import config  # noqa: E402

config.load_env()

from applypilot import database  # noqa: E402
from applypilot.config import load_profile  # noqa: E402
from applypilot.llm import LLMClient, ModelEntry, local_openai_base_url  # noqa: E402
from applypilot.scoring.scorer import _check_ineligible  # noqa: E402
from run_atomic_facts_scorer import (  # noqa: E402
    classify_family,
    deterministic_combine,
    extract_cs_degree_required,
    extract_years_required,
)

FAST_URL = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")
FAST_MODEL = "qwen3:1.7b"
SLOW_MODEL = "qwen3:8b"


def make_client(model: str) -> LLMClient:
    client = LLMClient(FAST_URL, model, "", quality=True)
    client._fallback_chain = [ModelEntry(model, "local", local_openai_base_url(FAST_URL), "")]
    return client


def escalating_classify(fast_client, slow_client, job) -> tuple[str | None, bool]:
    """Returns (family, escalated). Calls the fast model twice; if they
    agree, trusts it (escalated=False). If they disagree (including either
    call failing to parse), escalates to the slow model once
    (escalated=True) and trusts its answer."""
    a = classify_family(fast_client, job)
    b = classify_family(fast_client, job)
    if a is not None and a == b:
        return a, False
    escalated = classify_family(slow_client, job)
    return escalated, True


def decomposed_score_escalating(fast_client, slow_client, profile, job) -> tuple[int, str, bool]:
    reason = _check_ineligible(job, profile)
    if reason:
        return 2, f"gate:{reason[:40]}", False
    family, escalated = escalating_classify(fast_client, slow_client, job)
    years = extract_years_required(job.get("full_description") or "")
    cs_degree = extract_cs_degree_required(job.get("full_description") or "")
    score = deterministic_combine(family, years, cs_degree)
    return score, f"family:{family} years:{years} cs_degree:{cs_degree}", escalated


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

    fast_client = make_client(FAST_MODEL)
    slow_client = make_client(SLOW_MODEL)
    results = []
    n_escalated = 0
    for i, job in enumerate(sample):
        t0 = time.time()
        score, detail, escalated = decomposed_score_escalating(fast_client, slow_client, profile, job)
        dt = time.time() - t0
        n_escalated += int(escalated)
        results.append({**job, "local_score": score, "detail": detail, "escalated": escalated, "seconds": round(dt, 1)})
        esc_tag = "ESCALATED" if escalated else ""
        print(f"[{i+1}/{len(sample)}] real={job['fit_score']} local={score} ({dt:.1f}s{' '+esc_tag if esc_tag else ''}, {detail}) {job['title'][:35]}")

    have = [r for r in results if r["local_score"] is not None]
    n = len(have)
    print(f"\nn={len(results)}, parseable={n} ({100*n/len(results):.0f}%)")
    print(f"escalated to 8b: {n_escalated}/{len(results)} ({100*n_escalated/len(results):.0f}%)")
    exact = sum(1 for r in have if r["local_score"] == r["fit_score"])
    within2 = sum(1 for r in have if abs(r["local_score"] - r["fit_score"]) <= 2)
    real_pos = [r for r in have if r["fit_score"] >= 8]
    tp = sum(1 for r in real_pos if r["local_score"] >= 8)
    fn = len(real_pos) - tp
    pred_pos = [r for r in have if r["local_score"] >= 8]
    fp = sum(1 for r in pred_pos if r["fit_score"] < 8)
    recall = tp / len(real_pos) if real_pos else float("nan")
    precision = tp / len(pred_pos) if pred_pos else float("nan")
    gate_agree = sum(1 for r in have if (r["local_score"] >= 8) == (r["fit_score"] >= 8)) / n
    total_seconds = sum(r["seconds"] for r in have)
    avg_seconds = total_seconds / n
    print(f"exact match: {100*exact/n:.0f}%  within +-2: {100*within2/n:.0f}%  gate_agree: {100*gate_agree:.0f}%")
    print(f"real positives (>=8): n={len(real_pos)}  TP={tp} FN={fn}  recall={recall:.2f} precision={precision:.2f}")
    print(f"avg latency: {avg_seconds:.1f}s  total: {total_seconds:.0f}s ({total_seconds/60:.1f} min)")

    import json

    with open(Path(__file__).parent / "escalation_scorer_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
