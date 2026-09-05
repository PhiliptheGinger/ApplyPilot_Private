"""Small-batch comparison: does using the cross-encoder to rank pool
sentences against a requirement (instead of raw cosine) actually change
what gets selected, and does it look like a better fit? 2026-09-04, direct
follow-up to the "Set up home appliances from Lowe's" weak-match finding.

No ground truth for "which selection is the better fit" exists (unlike
the earlier same-claim/different-claim dedup bake-off, which had a real
labelable answer) -- this reports every case where the two methods
DISAGREE, so a human can eyeball fit quality directly, plus timing cost
for both methods across the batch.

Cost note: unlike the duplicate-detection tier (which restricts cross-
encoder calls to an ambiguous cosine band to avoid O(n^2) pairwise cost),
this only needs O(pool_size) cross-encoder calls per requirement -- the
pool is 8 sentences, so scoring ALL of them per requirement is cheap
regardless; no tiering needed here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402
from applypilot.scoring.semantic_match import cosine_similarity, cross_encoder_score, embed_texts  # noqa: E402

POOL_PATH = REPO_ROOT / "data/experiments/deterministic_slotfiller_20260902/inventory_expansion_pilot_v2_diversity_results.json"
TARGET_ENTRY = "Alex Prosperity Group / UST Logistics"


def select_via_cosine(requirement_text: str, pool: list[str], pool_embeddings: list[list[float]]) -> tuple[str, float]:
    req_emb = embed_texts([requirement_text])[0]
    scored = [(s, cosine_similarity(req_emb, e)) for s, e in zip(pool, pool_embeddings)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[0]


def select_via_cross_encoder(requirement_text: str, pool: list[str]) -> tuple[str, float]:
    scored = [(s, cross_encoder_score(requirement_text, s) or 0.0) for s in pool]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[0]


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    pool_data = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    pool = pool_data["final_accepted"]
    pool_embeddings = embed_texts(pool)

    rows = conn.execute(
        "SELECT url, title, fit_score, full_description FROM jobs "
        "WHERE fit_score >= 4 AND full_description IS NOT NULL "
        "ORDER BY discovered_at DESC LIMIT 1500"
    ).fetchall()

    matches = []
    for r in rows:
        job = dict(r)
        try:
            schema = build_job_schema_representation(job, profile)
        except Exception:
            continue
        for req in schema.get("requirements") or []:
            if req.get("supported") and TARGET_ENTRY in (req.get("resume_evidence") or []):
                matches.append((job, req.get("requirement")))
                break

    print(f"Found {len(matches)} jobs matching {TARGET_ENTRY} evidence out of {len(rows)} scanned.")
    batch = matches[:25]
    print(f"Running comparison on {len(batch)} jobs.\n")

    cosine_time = 0.0
    ce_time = 0.0
    agreements = 0
    disagreements = []

    for job, req_text in batch:
        t0 = time.time()
        cos_best, cos_score = select_via_cosine(req_text, pool, pool_embeddings)
        cosine_time += time.time() - t0

        t0 = time.time()
        ce_best, ce_score = select_via_cross_encoder(req_text, pool)
        ce_time += time.time() - t0

        if cos_best == ce_best:
            agreements += 1
        else:
            disagreements.append({
                "job_title": job["title"],
                "fit_score": job["fit_score"],
                "requirement": req_text,
                "cosine_pick": cos_best,
                "cosine_score": round(cos_score, 3),
                "cross_encoder_pick": ce_best,
                "cross_encoder_score": round(ce_score, 3),
            })

    print(f"=== RESULTS: {agreements}/{len(batch)} agreed, {len(disagreements)}/{len(batch)} disagreed ===\n")
    print(f"cosine total time: {cosine_time:.2f}s ({cosine_time/len(batch)*1000:.1f}ms/job)")
    print(f"cross-encoder total time: {ce_time:.2f}s ({ce_time/len(batch)*1000:.1f}ms/job)")

    print("\n=== DISAGREEMENTS (eyeball these for fit quality) ===")
    for d in disagreements:
        print(f"\nJOB: {d['job_title']} (fit_score={d['fit_score']})")
        print(f"  REQUIREMENT: {d['requirement']}")
        print(f"  cosine picked        (score={d['cosine_score']}): {d['cosine_pick']}")
        print(f"  cross-encoder picked (score={d['cross_encoder_score']}): {d['cross_encoder_pick']}")

    (Path(__file__).resolve().parent / "selector_cosine_vs_crossencoder_20260904_results.json").write_text(
        json.dumps({
            "n_jobs": len(batch),
            "agreements": agreements,
            "disagreements": disagreements,
            "cosine_total_time_s": round(cosine_time, 2),
            "cross_encoder_total_time_s": round(ce_time, 2),
        }, indent=2),
        encoding="utf-8",
    )
    print("\nwrote selector_cosine_vs_crossencoder_20260904_results.json")


if __name__ == "__main__":
    main()
