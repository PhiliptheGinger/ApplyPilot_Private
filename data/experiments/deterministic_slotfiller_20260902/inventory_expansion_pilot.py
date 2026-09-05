"""Sentence-inventory expansion pilot: generate MORE candidate sentences
for one thin experience_inventory entry, using a local model, then run
every candidate through the SAME deterministic fabrication checks already
built and shipped tonight (schemas.check_claim_strength/check_agency_
strength/check_causal_claim/check_metric_fabrication) before a human ever
sees them. Anything that fails is dropped silently. Survivors are written
to a numbered review file -- nothing is added to profile.json here; that
step stays manual, same discipline as every prior sentence review tonight.

This is the "safe half" of the two-stage architecture discussed: risky
generation happens ONCE, offline, in a small reviewable batch -- not
per-job, not required to be automatically trustworthy. The per-job
deterministic selector (schemas.py / local_tailor.py) is UNCHANGED by this
script; it just gets more real material to choose from once a human
approves some of this batch.

Target: "Alex Prosperity Group / UST Logistics" -- currently only 3
responsibilities, the thinnest entry in experience_inventory.
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
    claim_ceiling_for_evidence,
    agency_ceiling_for_evidence,
    check_claim_strength,
    check_agency_strength,
    check_causal_claim,
    check_metric_fabrication,
    _evidence_own_text,
)

OUT_DIR = Path(__file__).resolve().parent

TARGET_ENTRY_NAME = "Alex Prosperity Group / UST Logistics"

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
    lines += ["", "Generate 15 alternate-phrasing sentences restating ONLY the facts above."]
    return "\n".join(lines)


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

    client = llm.get_client(quality=True)
    t0 = time.time()
    raw = client.chat(
        [
            {"role": "system", "content": _GENERATION_SYSTEM},
            {"role": "user", "content": _generation_user_prompt(item)},
        ],
        max_tokens=1500,
        temperature=0.6,
    )
    elapsed = time.time() - t0
    print(f"\ngeneration call: {elapsed:.1f}s")
    print(f"raw response ({len(raw)} chars): {raw[:200]}...")

    try:
        parsed = json.loads(raw)
        candidates = [str(s).strip() for s in (parsed.get("sentences") or []) if str(s).strip()]
    except (json.JSONDecodeError, AttributeError) as exc:
        print(f"FAILED TO PARSE JSON: {exc}")
        (OUT_DIR / "inventory_expansion_pilot_raw_failure.txt").write_text(raw, encoding="utf-8")
        return

    print(f"\ngenerated {len(candidates)} candidate sentences")

    evidence_text = _evidence_own_text(item)
    known_metrics = (profile.get("resume_facts") or {}).get("real_metrics") or []
    claim_ceiling = claim_ceiling_for_evidence(item)
    agency_ceiling = agency_ceiling_for_evidence(item)

    survivors = []
    rejected = []
    for c in candidates:
        checks = {
            "claim": check_claim_strength(c, claim_ceiling),
            "agency": check_agency_strength(c, agency_ceiling),
            "causal": check_causal_claim(c, evidence_text),
            "metric": check_metric_fabrication(c, evidence_text, known_metrics),
        }
        failures = [f"{name}: {res.get('violation')}" for name, res in checks.items() if not res.get("passed", True)]
        if failures:
            rejected.append({"sentence": c, "failures": failures})
        else:
            survivors.append(c)

    print(f"\nsurvived all checks: {len(survivors)}/{len(candidates)}")
    print(f"rejected: {len(rejected)}/{len(candidates)}")

    print("\n=== SURVIVORS (candidates for your review) ===")
    for i, s in enumerate(survivors, 1):
        print(f"  [{i}] {s}")

    print("\n=== REJECTED (never shown for review, dropped automatically) ===")
    for r in rejected:
        print(f"  - {r['sentence']}")
        for f in r["failures"]:
            print(f"      x {f}")

    (OUT_DIR / "inventory_expansion_pilot_results.json").write_text(
        json.dumps(
            {
                "target": TARGET_ENTRY_NAME,
                "generation_elapsed_s": round(elapsed, 1),
                "n_generated": len(candidates),
                "n_survived": len(survivors),
                "survivors": survivors,
                "rejected": rejected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nwrote inventory_expansion_pilot_results.json")
    print("NOTHING added to profile.json -- survivors are for manual review only.")


if __name__ == "__main__":
    main()
