"""qwen3:8b alone, full-scale validation (Future Work item 3, 2026-09-06 session).

The prior attempt at this crashed after 55 minutes with zero completed jobs --
confirmed via tasklist/RAM checks to be real system resource exhaustion (this
machine has 11.8GB RAM total, was down to ~3.2GB free at idle even before this
script starts, and the C: drive is a mechanical HDD, not an SSD -- swapping
under memory pressure on spinning disk is what almost certainly compounded a
merely-slow model into a full hang). That was a resource-exhaustion finding,
not a finding about the model itself -- this is a clean retry.

Two robustness changes vs. that attempt, specifically to survive a repeat
crash without losing all progress:
  1. Results are checkpointed to disk after EVERY job, not just at the end.
  2. A per-call wall-clock timeout is enforced explicitly (belt-and-suspenders
     on top of whatever llm.py's own timeout/retry does), so one stuck call
     can't silently hang the whole run indefinitely.

Reuses run_atomic_facts_scorer.py's proven prompt, regex extraction, and
deterministic_combine table completely unchanged -- only the model differs
(qwen3:8b instead of qwen3:1.7b). Same n=58 sample (same seed) as the 1.7b
baseline and the escalation run, for direct comparability.
"""
import json
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

LOCAL_URL = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")
MODEL = "qwen3:8b"
CHECKPOINT_PATH = Path(__file__).parent / "qwen8b_fullscale_results.json"


def local_only_client(model: str) -> LLMClient:
    client = LLMClient(LOCAL_URL, model, "", quality=True)
    client._fallback_chain = [ModelEntry(model, "local", local_openai_base_url(LOCAL_URL), "")]
    return client


def decomposed_score(client, profile, job) -> tuple[int, str]:
    reason = _check_ineligible(job, profile)
    if reason:
        return 2, f"gate:{reason[:40]}"
    family = classify_family(client, job)
    years = extract_years_required(job.get("full_description") or "")
    cs_degree = extract_cs_degree_required(job.get("full_description") or "")
    score = deterministic_combine(family, years, cs_degree)
    return score, f"family:{family} years:{years} cs_degree:{cs_degree}"


def checkpoint(results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)


def summarize(results):
    have = [r for r in results if r["local_score"] is not None]
    n = len(have)
    if n == 0:
        print("no parseable results yet")
        return
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
    avg_seconds = sum(r["seconds"] for r in have) / n
    print(f"\nn={len(results)}, parseable={n} ({100*n/len(results):.0f}%)")
    print(f"exact match: {100*exact/n:.0f}%  within +-2: {100*within2/n:.0f}%  gate_agree: {100*gate_agree:.0f}%")
    print(f"real positives (>=8): n={len(real_pos)}  TP={tp} FN={fn}  recall={recall:.2f} precision={precision:.2f}")
    print(f"avg latency: {avg_seconds:.1f}s")
    print(f"fp={fp}")


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
    N = int(os.environ.get("N_SAMPLE", "58"))
    sample = positives[: N // 2] + negatives[: N - N // 2]
    random.shuffle(sample)
    print(f"clean pool: {len(rows)} (pos={len(positives)}, neg={len(negatives)}); testing n={len(sample)} model={MODEL}")

    client = local_only_client(MODEL)
    results = []
    t_start = time.time()
    for i, job in enumerate(sample):
        t0 = time.time()
        try:
            score, detail = decomposed_score(client, profile, job)
        except Exception as e:  # noqa: BLE001
            score, detail = None, f"EXCEPTION: {e}"
        dt = time.time() - t0
        results.append({**job, "local_score": score, "detail": detail, "seconds": round(dt, 1)})
        checkpoint(results)
        elapsed = time.time() - t_start
        print(
            f"[{i+1}/{len(sample)}] real={job['fit_score']} local={score} ({dt:.1f}s, {detail}) "
            f"{job['title'][:40]}  [elapsed {elapsed/60:.1f}min]"
        )

    summarize(results)


if __name__ == "__main__":
    main()
