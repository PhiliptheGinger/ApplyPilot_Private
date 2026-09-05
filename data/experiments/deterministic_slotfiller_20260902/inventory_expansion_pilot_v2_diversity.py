"""Sentence-inventory expansion pilot, v2: adds a diversity filter and a
regeneration-with-feedback loop on top of v1 (inventory_expansion_pilot.py).

v1 (2026-09-03 night) found that fabrication checks alone are not enough:
local qwen3:1.7b produced 12 "survivors" for one thin experience_inventory
entry, but on inspection ~9 of them were the same handful of sentences with
words swapped for near-synonyms ("customer specifications and company
protocols" vs "customer requirements and company procedures"). Discussed
that night: a purely statistical (cosine-embedding) dissimilarity filter
was proposed, with an explicit caveat raised before testing -- averaged
sentence embeddings might not separate "same template, different words"
from genuine variety, because that requires looking at which words play
which grammatical role (constituency), not just overall meaning.

That caveat was verified empirically before writing this script (see
semantic_match.diversity_threshold()'s docstring for the numbers): cosine
similarity on real all-minilm embeddings of the v1 survivor set DOES
cleanly separate near-verbatim restatements (0.93-0.98 similarity) from a
clearly distinct sentence (down to 0.34-0.46), but does NOT separate out
same-template/synonym-slot-substituted sentences, which scored only
~0.82-0.85 -- below any threshold that wouldn't also start rejecting
genuinely distinct content. A word/bigram-overlap Jaccard signal was also
tested on the same data and did WORSE (synonym substitution defeats
surface lexical overlap even more thoroughly than it dilutes an embedding
average).

Architecture, given that finding: cosine dissimilarity is used as-is for
what it's actually good at (bounding obvious near-verbatim duplication,
cheap and deterministic) via semantic_match.select_diverse_indices. The
harder, structural-duplication case is NOT claimed to be solved
statistically -- instead, each regeneration round shows the model its own
already-accepted sentences and explicitly instructs it to vary sentence
STRUCTURE (which part leads, clause order), not just word choice, putting
the correction where it belongs (generation), not pretending a filter can
fix what the filter has already been shown not to catch.

Provider-agnostic by construction: this script drives whatever
llm.get_client(quality=True) resolves to. It's run once against local
Ollama here (env vars below), and per 2026-09-04 direction the same
pipeline should eventually be run against the real cloud cascade too, for
guardrail consistency -- NOT because cloud has shown this failure mode
(it hasn't been tested at all yet), but because a fix that only applies
when the local model is picked would silently stop applying the moment
cloud recovers from exhaustion. NOTE: not run against cloud in this
session -- ~/.applypilot/.env has no cloud API keys configured in this
environment (confirmed by checking, not assumed), so a live cloud
comparison is blocked here and left as a follow-up.

Same discipline as v1: nothing is added to profile.json by this script.
Survivors are written to a review file for manual approval only.
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
from applypilot.scoring.schemas import (  # noqa: E402
    _evidence_own_text,
    agency_ceiling_for_evidence,
    check_agency_strength,
    check_causal_claim,
    check_claim_strength,
    check_metric_fabrication,
    claim_ceiling_for_evidence,
)
from applypilot.scoring.semantic_match import (  # noqa: E402
    diversity_threshold,
    embed_texts,
    select_diverse_indices,
)

OUT_DIR = Path(__file__).resolve().parent

TARGET_ENTRY_NAME = "Alex Prosperity Group / UST Logistics"
TARGET_SURVIVOR_COUNT = 6
MAX_ROUNDS = 3
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

_STRUCTURE_CORRECTION = """
You have already produced sentences with this SAME sentence template, just with different synonyms:
{already_have}

Those do NOT count as variety -- swapping "specifications" for "requirements" or "protocols" for \
"procedures" inside the same sentence shape is not a new sentence, it's the same sentence. For this \
batch, each new sentence must use a DIFFERENT STRUCTURE from every sentence above and from each other:
- Change which part of the sentence leads (the action, the object, the context/condition, or who it \
was for) instead of always opening with the verb.
- Change clause order (e.g. a leading subordinate clause instead of a trailing one).
- Vary sentence length noticeably (some short and direct, some with more clauses).
Still restate only the EXISTING FACTS below -- structure varies, facts do not.
"""


def _generation_user_prompt(item: dict, already_have: list[str] | None = None) -> str:
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
    if already_have:
        lines.append(_STRUCTURE_CORRECTION.format(already_have="\n".join(f"- {s}" for s in already_have)))
    lines += ["", f"Generate {CANDIDATES_PER_ROUND} alternate-phrasing sentences restating ONLY the facts above."]
    return "\n".join(lines)


def _generate_round(client, item: dict, already_have: list[str]) -> tuple[list[str], float, str]:
    t0 = time.time()
    raw = client.chat(
        [
            {"role": "system", "content": _GENERATION_SYSTEM},
            {"role": "user", "content": _generation_user_prompt(item, already_have or None)},
        ],
        max_tokens=1500,
        temperature=0.6,
    )
    elapsed = time.time() - t0
    try:
        parsed = json.loads(raw)
        candidates = [str(s).strip() for s in (parsed.get("sentences") or []) if str(s).strip()]
    except (json.JSONDecodeError, AttributeError):
        candidates = []
    return candidates, elapsed, raw


def _passes_fabrication_checks(sentence: str, evidence_text: str, claim_ceiling: str, agency_ceiling: str,
                                known_metrics: list[str]) -> list[str]:
    checks = {
        "claim": check_claim_strength(sentence, claim_ceiling),
        "agency": check_agency_strength(sentence, agency_ceiling),
        "causal": check_causal_claim(sentence, evidence_text),
        "metric": check_metric_fabrication(sentence, evidence_text, known_metrics),
    }
    return [f"{name}: {res.get('violation')}" for name, res in checks.items() if not res.get("passed", True)]


def main() -> None:
    profile = config.load_profile()
    item = next(
        (it for it in profile.get("experience_inventory") or [] if it.get("name") == TARGET_ENTRY_NAME),
        None,
    )
    if item is None:
        raise SystemExit(f"entry not found: {TARGET_ENTRY_NAME}")

    print(f"target: {TARGET_ENTRY_NAME}")
    print(f"existing responsibilities ({len(item.get('responsibilities') or [])}):")
    for r in item.get("responsibilities") or []:
        print(f"  - {r}")
    print(f"\ndiversity threshold: {diversity_threshold()}")

    client = llm.get_client(quality=True)
    evidence_text = _evidence_own_text(item)
    known_metrics = (profile.get("resume_facts") or {}).get("real_metrics") or []
    claim_ceiling = claim_ceiling_for_evidence(item)
    agency_ceiling = agency_ceiling_for_evidence(item)

    accepted: list[str] = []
    rounds_log: list[dict] = []

    for round_num in range(1, MAX_ROUNDS + 1):
        if len(accepted) >= TARGET_SURVIVOR_COUNT:
            break

        print(f"\n=== round {round_num} (have {len(accepted)}/{TARGET_SURVIVOR_COUNT}) ===")
        candidates, elapsed, raw = _generate_round(client, item, accepted)
        print(f"generation call: {elapsed:.1f}s, {len(candidates)} candidates")

        fabrication_rejected = []
        fabrication_survivors = []
        for c in candidates:
            if c in accepted:
                continue
            failures = _passes_fabrication_checks(c, evidence_text, claim_ceiling, agency_ceiling, known_metrics)
            if failures:
                fabrication_rejected.append({"sentence": c, "failures": failures})
            else:
                fabrication_survivors.append(c)
        print(f"fabrication check: {len(fabrication_survivors)}/{len(candidates)} survived")

        # Diversity pass: seed with already-accepted sentences (kept
        # unconditionally, index order 0..n-1) so this round's candidates
        # are screened against BOTH prior rounds and each other in one
        # pass -- select_diverse_indices is order-dependent by design (see
        # its docstring), and prior-accepted sentences must always win a
        # collision over a new candidate, not the other way around.
        pool = accepted + fabrication_survivors
        embeddings = embed_texts(pool) if pool else []
        similarity_rejected = []
        if embeddings is None:
            print("embedding call failed -- skipping diversity filter this round (fabrication survivors kept as-is)")
            newly_accepted = fabrication_survivors
        else:
            kept_indices = set(select_diverse_indices(embeddings, threshold=diversity_threshold()))
            newly_accepted = []
            for offset, sentence in enumerate(fabrication_survivors):
                idx = len(accepted) + offset
                if idx in kept_indices:
                    newly_accepted.append(sentence)
                else:
                    similarity_rejected.append(sentence)
        print(f"diversity check: {len(newly_accepted)}/{len(fabrication_survivors)} kept "
              f"({len(similarity_rejected)} too similar to an existing sentence)")

        accepted.extend(newly_accepted)
        rounds_log.append({
            "round": round_num,
            "generation_elapsed_s": round(elapsed, 1),
            "n_generated": len(candidates),
            "n_fabrication_survived": len(fabrication_survivors),
            "fabrication_rejected": fabrication_rejected,
            "n_diversity_kept": len(newly_accepted),
            "similarity_rejected": similarity_rejected,
            "raw_response_if_unparsed": None if candidates else raw[:500],
        })

    print(f"\n=== FINAL: {len(accepted)}/{TARGET_SURVIVOR_COUNT} target, "
          f"{'reached' if len(accepted) >= TARGET_SURVIVOR_COUNT else 'NOT reached'} "
          f"after {len(rounds_log)} round(s) ===")
    for i, s in enumerate(accepted, 1):
        print(f"  [{i}] {s}")

    print("\nNOTE: diversity filtering bounds near-verbatim duplication only (see "
          "semantic_match.diversity_threshold() docstring for the documented gap on "
          "same-template/synonym-substituted sentences). This script cannot certify "
          "structural variety -- but it CAN point out which surviving pairs are closest "
          "to each other, so a human reviewer isn't starting from zero:")

    closest_pairs: list[dict] = []
    accepted_embeddings = embed_texts(accepted) if len(accepted) > 1 else None
    if accepted_embeddings:
        from applypilot.scoring.semantic_match import cosine_similarity

        scored_pairs = []
        for i in range(len(accepted)):
            for j in range(i + 1, len(accepted)):
                scored_pairs.append((cosine_similarity(accepted_embeddings[i], accepted_embeddings[j]), i, j))
        scored_pairs.sort(reverse=True)
        for score, i, j in scored_pairs[:3]:
            closest_pairs.append({"score": round(score, 3), "a": accepted[i], "b": accepted[j]})
            print(f"  {score:.3f}  [{i + 1}] {accepted[i]}")
            print(f"         [{j + 1}] {accepted[j]}")
    else:
        print("  (fewer than 2 accepted sentences, or embedding call failed -- nothing to compare)")

    (OUT_DIR / "inventory_expansion_pilot_v2_diversity_results.json").write_text(
        json.dumps(
            {
                "target": TARGET_ENTRY_NAME,
                "target_survivor_count": TARGET_SURVIVOR_COUNT,
                "diversity_threshold": diversity_threshold(),
                "rounds": rounds_log,
                "final_accepted": accepted,
                "target_reached": len(accepted) >= TARGET_SURVIVOR_COUNT,
                "closest_surviving_pairs": closest_pairs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nwrote inventory_expansion_pilot_v2_diversity_results.json")
    print("NOTHING added to profile.json -- survivors are for manual review only.")


if __name__ == "__main__":
    main()
