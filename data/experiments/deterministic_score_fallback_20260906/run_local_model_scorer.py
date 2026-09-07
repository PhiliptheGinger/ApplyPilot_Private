"""Test qwen3:1.7b as a SCORING judge (reading/classifying), not a writer
(generating) -- user's hypothesis: tailoring's fabrication failures may not
transfer to a narrower judgment task. Uses the REAL production
SCORE_PROMPT_TEMPLATE and reuses scorer.py's own prompt-building/parsing
functions for a fair, realistic comparison -- not a fresh reimplementation.

Runs against the properly CLEAN dataset only (scored_at >= 2026-08-28, past
the fabricated-identity bug found earlier tonight, excluding the WeWorkRemotely/
Intel garbage-description rows) -- lesson learned from tonight's contamination
finding applies here too: don't calibrate/validate against tainted ground
truth.

Uses tonight's own fix (native Ollama think:false fast path) via
APPLYPILOT_LOCAL_OLLAMA_NATIVE for fast, reliable local calls.
"""
import os
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
    SCORE_PROMPT_TEMPLATE,
    _build_candidate_summary,
    _build_location_context,
    _parse_score_response,
)

LOCAL_URL = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", "qwen3:1.7b")


def local_only_client() -> LLMClient:
    client = LLMClient(LOCAL_URL, LOCAL_MODEL, "", quality=True)
    client._fallback_chain = [ModelEntry(LOCAL_MODEL, "local", local_openai_base_url(LOCAL_URL), "")]
    return client


def score_with_local_model(client, profile, job) -> dict:
    candidate_summary = _build_candidate_summary(profile)
    location_context = _build_location_context(profile)
    score_prompt = SCORE_PROMPT_TEMPLATE.format(
        candidate_summary=candidate_summary,
        location_context=location_context,
    )
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )
    messages = [
        {"role": "system", "content": score_prompt},
        {"role": "user", "content": job_text},
    ]
    try:
        response = client.chat(messages, max_tokens=1500, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        return {"score": None, "error": f"{type(exc).__name__}: {exc}"}
    return _parse_score_response(response)


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
    # 169s/call measured live -- keep this small. Stratified: half positives
    # (the class that matters most / the class the single real test got
    # wrong), half negatives.
    N = int(os.environ.get("N_SAMPLE", "10"))
    sample = positives[: N // 2] + negatives[: N - N // 2]
    random.shuffle(sample)
    print(f"clean pool: {len(rows)} (pos={len(positives)}, neg={len(negatives)}); testing n={len(sample)}")

    client = local_only_client()
    results = []
    for i, job in enumerate(sample):
        t0 = time.time()
        r = score_with_local_model(client, profile, job)
        dt = time.time() - t0
        results.append({**job, "local_score": r.get("score"), "local_error": r.get("error"), "seconds": round(dt, 1)})
        print(
            f"[{i+1}/{len(sample)}] real={job['fit_score']} local={r.get('score')} "
            f"({dt:.1f}s) {job['title'][:50]}"
            + (f"  ERR: {r.get('error')}" if r.get("error") else "")
        )

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
    print(f"exact match: {100*exact/n:.0f}%  within +-2: {100*within2/n:.0f}%  gate_agree: {100*gate_agree:.0f}%")
    print(f"real positives (>=8): n={len(real_pos)}  TP={tp} FN={fn}  recall={recall:.2f} precision={precision:.2f}")

    import json

    with open(Path(__file__).parent / "local_model_scorer_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
