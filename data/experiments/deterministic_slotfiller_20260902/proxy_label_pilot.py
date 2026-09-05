"""Proxy-label pilot: get REAL survival labels (not just distribution
shape) for the category_tier rarity-weighting question, by actually
running the production degraded-mode local-realization path
(local_tailor.request_local_realization) on isolated single-keyword
("near_prototype") requirements spanning the weight distribution, and
checking whether the resulting bullet survives request_local_realization's
own built-in claim/agency/causal/metric/passive checks -- the same checks
that decide, in production, whether a realized bullet actually reaches a
real resume.

Each requirement is tested ISOLATED (a synthetic one-requirement job_schema
built by slicing a single entry out of a real job_schema, everything else
unchanged) rather than batched with a job's other requirements -- batched
realization keys bullets by EVIDENCE NAME, and one evidence item often
backs several requirements, which would blur which requirement's weighted
score actually predicted the outcome. Isolating gives a clean per-case
label.

This is a PILOT (small N) to check whether there's any survival variance
to correlate against at all before committing to a larger run -- historical
decision-log context (local_tailor.py's 2026-09-02 finding cited in
_build_realization_prompt's docstring) found survival at "single digits to
zero" for the full prototype/near_prototype population once checked
against an EXTERNAL keyword-injection check this script does not run --
so a near-all-failure result here would not be a bug, it would replicate
that known finding, and should be reported as such rather than treated as
a broken experiment.

Requires a locally reachable Ollama instance (confirmed running before
this script was written: qwen3:1.7b available). No cloud LLM calls.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ["LLM_URL"] = "http://localhost:11434/v1"
os.environ["LLM_MODEL_QUALITY"] = "qwen3:1.7b"
os.environ["APPLYPILOT_LOCAL_LLM_URL"] = "http://localhost:11434/v1"
os.environ["APPLYPILOT_LOCAL_LLM_MODEL"] = "qwen3:1.7b"

from applypilot import config, database  # noqa: E402
from applypilot import llm  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402
from applypilot.scoring.local_tailor import request_local_realization  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
IDF = json.loads((OUT_DIR / "idf_weights.json").read_text(encoding="utf-8"))


def weighted(kws: list[str]) -> float:
    return sum(IDF.get(k.lower(), 1.0) for k in kws)


def isolated_schema(job_schema: dict, requirement: dict) -> dict:
    return {
        "job_url": job_schema.get("job_url"),
        "requirements": [requirement],
        "viewpoint": job_schema.get("viewpoint"),
        "summary_schema": job_schema.get("summary_schema"),
        "evidence_considered": 1,
    }


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()

    random.seed(20260903)
    pool = [
        r[0]
        for r in conn.execute(
            "SELECT rowid FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 800"
        ).fetchall()
    ]
    job_ids = random.sample(pool, min(150, len(pool)))

    candidates = []
    for jid in job_ids:
        row = conn.execute("SELECT rowid, title, full_description FROM jobs WHERE rowid=?", (jid,)).fetchone()
        job = {"url": f"local:{jid}", "title": row[1], "full_description": row[2]}
        rep = build_job_schema_representation(job, profile)
        for r in rep["requirements"]:
            if r.get("category_tier") == "near_prototype" and r.get("exact_keywords"):
                w = weighted(r["exact_keywords"])
                candidates.append({"job": job, "job_schema": rep, "requirement": r, "weight": w})

    candidates.sort(key=lambda c: c["weight"])
    n = len(candidates)
    print(f"total near_prototype candidates: {n}")

    # Stratified pilot sample: 5 low, 5 mid, 5 high (by weight tercile)
    low = candidates[: n // 3]
    mid = candidates[n // 3 : 2 * n // 3]
    high = candidates[2 * n // 3 :]
    random.seed(1)
    sample = random.sample(low, min(5, len(low))) + random.sample(mid, min(5, len(mid))) + random.sample(high, min(5, len(high)))
    print(f"pilot sample size: {len(sample)}")

    client = llm.get_client(quality=True)

    results = []
    for i, c in enumerate(sample):
        req = c["requirement"]
        schema = isolated_schema(c["job_schema"], req)
        t0 = time.time()
        realization, meta = request_local_realization(client, c["job"], schema, profile)
        elapsed = time.time() - t0
        evidence_name = (req.get("resume_evidence") or [None])[0]
        survived = bool(realization and realization.get("bullets") and evidence_name in (realization.get("bullets") or {}))
        attempted = bool(meta.get("llm_called"))
        result = {
            "title": c["job"]["title"][:50],
            "requirement": req["requirement"][:80],
            "keywords": req["exact_keywords"],
            "weight": round(c["weight"], 3),
            "attempted": attempted,
            "survived": survived,
            "violations": meta.get("claim_strength_violations", 0),
            "elapsed_s": round(elapsed, 1),
        }
        results.append(result)
        print(f"[{i+1}/{len(sample)}] w={result['weight']:.2f} attempted={attempted} survived={survived} ({elapsed:.1f}s) | {result['title']}")

    (OUT_DIR / "proxy_label_pilot_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    for r in sorted(results, key=lambda x: x["weight"]):
        print(f"  w={r['weight']:5.2f}  attempted={r['attempted']!s:5}  survived={r['survived']!s:5}  {r['keywords']}")
    n_attempted = sum(1 for r in results if r["attempted"])
    n_survived = sum(1 for r in results if r["survived"])
    print(f"\nattempted: {n_attempted}/{len(results)}   survived: {n_survived}/{len(results)}")
    print("wrote proxy_label_pilot_results.json")


if __name__ == "__main__":
    main()
