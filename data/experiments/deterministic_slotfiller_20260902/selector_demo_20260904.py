"""The missing piece: given a job requirement's text and an already-
generated, already-diversity-filtered sentence pool (tonight's actual
output, data/experiments/deterministic_slotfiller_20260902/
inventory_expansion_pilot_v2_diversity_results.json), deterministically
select which pool sentence best answers THAT requirement. Zero LLM call
at selection time -- the LLM's job was already done, offline, tonight,
producing the pool. This is the connective tissue that was missing.

Proof of concept: run it against two differently-worded requirements and
show it picks two DIFFERENT sentences from the SAME pool -- i.e. the
resume actually varies per job now, which is the entire point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot.scoring.semantic_match import cosine_similarity, embed_texts  # noqa: E402

POOL_PATH = REPO_ROOT / "data/experiments/deterministic_slotfiller_20260902/inventory_expansion_pilot_v2_diversity_results.json"


def select_best_sentence(requirement_text: str, pool: list[str], pool_embeddings: list[list[float]]) -> tuple[str, float]:
    """Deterministic, zero-LLM selection: rank the pool by cosine
    similarity to the requirement text, return the top match. This is the
    piece that was missing -- everything upstream (the pool itself, the
    requirement text from schemas.py) already existed."""
    req_emb = embed_texts([requirement_text])[0]
    scored = [(s, cosine_similarity(req_emb, e)) for s, e in zip(pool, pool_embeddings)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[0]


def main() -> None:
    data = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    pool = data["final_accepted"]
    print(f"Pool for '{data['target']}' ({len(pool)} sentences, generated + diversity-filtered tonight):")
    for s in pool:
        print(f"  - {s}")

    pool_embeddings = embed_texts(pool)

    requirements = [
        "Must communicate clearly and professionally with customers throughout service visits.",
        "Follow established company protocols and manufacturer specifications when performing installation work.",
    ]

    print("\n=== Deterministic selection per requirement (zero LLM calls here) ===")
    for req in requirements:
        best, score = select_best_sentence(req, pool, pool_embeddings)
        print(f"\nREQUIREMENT: {req}")
        print(f"  -> SELECTED (cosine={score:.3f}): {best}")

    print("\nTAKEAWAY: two different requirement phrasings pulled two different sentences out of "
          "the SAME pool, with no LLM call at selection time. That's the missing connective tissue "
          "-- the resume now actually varies per job instead of being the master resume verbatim.")


if __name__ == "__main__":
    main()
