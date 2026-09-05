"""Frequency/presence-penalty generation sweep -- the third technique from
the 2026-09-04 bake-off, tested last because it required threading new
optional params through llm.py's chat() first (done; see
tests/test_llm_cascade.py::TestFrequencyPresencePenaltyPassthrough).

Isolates the GENERATION-time effect specifically: does asking the local
model for less self-repetition change what it produces, BEFORE any
downstream fabrication check or diversity filter runs. One baseline call
(no penalty, matches every other run tonight) and one elevated-penalty
call (0.6/0.6, a moderate value commonly cited for reducing repetition
without destroying coherence) against the SAME entry, SAME prompt.

Single-sample per condition (temperature=0.6 means each call is already
stochastic) -- this is a directional A/B, not a statistically powered
study. Reported honestly as such.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ["LLM_URL"] = "http://localhost:11434/v1"
os.environ["LLM_MODEL_QUALITY"] = "qwen3:1.7b"
os.environ["APPLYPILOT_LOCAL_LLM_URL"] = "http://localhost:11434/v1"
os.environ["APPLYPILOT_LOCAL_LLM_MODEL"] = "qwen3:1.7b"

from applypilot import config, llm  # noqa: E402
from applypilot.scoring.semantic_match import cosine_similarity, embed_texts  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
TARGET_ENTRY_NAME = "Alex Prosperity Group / UST Logistics"
CANDIDATES_PER_ROUND = 15

_GENERATION_SYSTEM = """You expand a job history entry into more sentence variants for a resume-writing tool.

RULES (do not break these):
- Every sentence must restate a fact ALREADY present in the "EXISTING FACTS" list below, in different words.
- NEVER add a new tool, technology, metric, number, outcome, or responsibility not already stated.
- NEVER claim a result, improvement, or measurable outcome unless one is already stated in EXISTING FACTS.
- Vary sentence structure, word choice, and emphasis -- not the underlying facts.
- Each sentence must stand alone (no "additionally", no pronouns referring to other sentences).
- Output ONLY a JSON object: {"sentences": ["...", "...", ...]}. No other text, no markdown fences.
"""


def _generation_user_prompt(item: dict) -> str:
    facts = "\n".join(f"- {r}" for r in item.get("responsibilities") or [])
    constraints = "\n".join(f"- {c}" for c in item.get("constraints") or [])
    role = item.get("role_title") or item.get("name")
    lines = [
        f"ROLE: {role}",
        "",
        "EXISTING FACTS (the only things you may restate, nothing else):",
        facts,
    ]
    if constraints:
        lines += ["", "ADDITIONAL CONSTRAINTS:", constraints]
    lines += ["", f"Generate {CANDIDATES_PER_ROUND} alternate-phrasing sentences restating ONLY the facts above."]
    return "\n".join(lines)


def _run_condition(client, item, frequency_penalty, presence_penalty) -> dict:
    t0 = time.time()
    raw = client.chat(
        [
            {"role": "system", "content": _GENERATION_SYSTEM},
            {"role": "user", "content": _generation_user_prompt(item)},
        ],
        max_tokens=1500,
        temperature=0.6,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    elapsed = time.time() - t0
    try:
        parsed = json.loads(raw)
        candidates = [str(s).strip() for s in (parsed.get("sentences") or []) if str(s).strip()]
    except (json.JSONDecodeError, AttributeError):
        candidates = []

    stats = {
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "elapsed_s": round(elapsed, 1),
        "n_candidates": len(candidates),
        "candidates": candidates,
    }

    if len(candidates) >= 2:
        embeddings = embed_texts(candidates)
        if embeddings is not None:
            scores = []
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    scores.append(cosine_similarity(embeddings[i], embeddings[j]))
            stats["n_pairs"] = len(scores)
            stats["mean_pairwise_similarity"] = round(sum(scores) / len(scores), 3)
            stats["max_pairwise_similarity"] = round(max(scores), 3)
            stats["n_pairs_over_0.90"] = sum(1 for s in scores if s >= 0.90)
            stats["n_pairs_over_0.84"] = sum(1 for s in scores if s >= 0.84)

    return stats


def main() -> None:
    profile = config.load_profile()
    item = next(
        (it for it in profile.get("experience_inventory") or [] if it.get("name") == TARGET_ENTRY_NAME),
        None,
    )
    if item is None:
        raise SystemExit(f"entry not found: {TARGET_ENTRY_NAME}")

    client = llm.get_client(quality=True)

    print("=== condition A: baseline (no penalty) ===")
    baseline = _run_condition(client, item, frequency_penalty=None, presence_penalty=None)
    print(json.dumps({k: v for k, v in baseline.items() if k != "candidates"}, indent=2))
    for i, c in enumerate(baseline["candidates"], 1):
        print(f"  [{i}] {c}")

    print("\n=== condition B: elevated penalty (0.6 / 0.6) ===")
    elevated = _run_condition(client, item, frequency_penalty=0.6, presence_penalty=0.6)
    print(json.dumps({k: v for k, v in elevated.items() if k != "candidates"}, indent=2))
    for i, c in enumerate(elevated["candidates"], 1):
        print(f"  [{i}] {c}")

    print("\n=== comparison ===")
    if "mean_pairwise_similarity" in baseline and "mean_pairwise_similarity" in elevated:
        print(f"mean pairwise similarity: baseline={baseline['mean_pairwise_similarity']} "
              f"vs elevated={elevated['mean_pairwise_similarity']}")
        print(f"max pairwise similarity:  baseline={baseline['max_pairwise_similarity']} "
              f"vs elevated={elevated['max_pairwise_similarity']}")
        print(f"pairs >= 0.90 (near-verbatim): baseline={baseline['n_pairs_over_0.90']}/{baseline['n_pairs']} "
              f"vs elevated={elevated['n_pairs_over_0.90']}/{elevated['n_pairs']}")
        print(f"pairs >= 0.84 (bake-off best-fit threshold): "
              f"baseline={baseline['n_pairs_over_0.84']}/{baseline['n_pairs']} "
              f"vs elevated={elevated['n_pairs_over_0.84']}/{elevated['n_pairs']}")
    print(f"wall time: baseline={baseline['elapsed_s']}s vs elevated={elevated['elapsed_s']}s")

    (OUT_DIR / "bakeoff_penalty_sweep_20260904_results.json").write_text(
        json.dumps({"baseline": baseline, "elevated_penalty": elevated}, indent=2), encoding="utf-8"
    )
    print("\nwrote bakeoff_penalty_sweep_20260904_results.json")


if __name__ == "__main__":
    main()
