"""Cloud counterpart to inventory_expansion_pilot_v2_diversity.py.

Identical pipeline (generate -> fabrication checks -> diversity filter ->
regenerate-with-structure-correction if short of target) run against the
real Gemini/OpenAI cascade instead of local qwen3:1.7b, to answer a
concrete open question from 2026-09-04: does the near-duplicate,
same-template-different-synonyms problem observed on local-model output
also happen on cloud output, or is it specific to a small local model?

Unlike the local pilot scripts, this does NOT hardcode LLM_URL/
APPLYPILOT_LOCAL_LLM_* env vars -- it calls config.load_env() before
importing llm (per llm.py's own documented gotcha: it reads provider keys
from os.environ inside _build_fallback_chain, called from LLMClient.__init__,
so keys must be in the environment before get_client() constructs a
client -- config.load_env() loads GEMINI_API_KEY/OPENAI_API_KEY from the
repo-root .env, confirmed present 2026-09-04) so llm.get_client(quality=True)
resolves to the real quality-tier cascade (Gemini Pro first, OpenAI
fallback). No ANTHROPIC_API_KEY is configured in this environment, so
Claude cloud fallback is not reachable from this script.

Makes real (free-tier Gemini, and OpenAI only if Gemini is exhausted)
external API calls -- not a local/free Ollama call like the other
experiment scripts in this directory.

Same discipline as v1/v2: nothing is added to profile.json. Survivors are
for manual review only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot import llm  # noqa: E402
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
    print(f"resolved fallback chain: {[e.name for e in client._fallback_chain]}")

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
        print(f"generation call: {elapsed:.1f}s, model used: {client.last_model_used}, "
              f"{len(candidates)} candidates")

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
            "model_used": client.last_model_used,
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

    closest_pairs: list[dict] = []
    accepted_embeddings = embed_texts(accepted) if len(accepted) > 1 else None
    print("\nclosest surviving pairs (for manual review -- see semantic_match.diversity_threshold() "
          "docstring for why this filter alone can't certify structural variety):")
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

    (OUT_DIR / "inventory_expansion_pilot_v2_diversity_cloud_results.json").write_text(
        json.dumps(
            {
                "target": TARGET_ENTRY_NAME,
                "target_survivor_count": TARGET_SURVIVOR_COUNT,
                "diversity_threshold": diversity_threshold(),
                "fallback_chain": [e.name for e in client._fallback_chain],
                "rounds": rounds_log,
                "final_accepted": accepted,
                "target_reached": len(accepted) >= TARGET_SURVIVOR_COUNT,
                "closest_surviving_pairs": closest_pairs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nwrote inventory_expansion_pilot_v2_diversity_cloud_results.json")
    print("NOTHING added to profile.json -- survivors are for manual review only.")


if __name__ == "__main__":
    main()
