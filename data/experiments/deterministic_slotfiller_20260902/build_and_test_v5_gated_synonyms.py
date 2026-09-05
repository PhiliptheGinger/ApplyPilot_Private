"""Deterministic slot-filler v5: gate synonym substitution to requirements
whose evidence pairing is already independently well-corroborated.

v4 (build_and_test_v4_synonyms.py) found real yield was tiny (2/267) and
BOTH hits fired on the same already topically-wrong evidence pairing (a
car-repair sentence cited as "evidence" for two different white-collar
software/product-management requirements) -- the claim/agency safety
backstop can't catch this because it only checks whether realized text
overclaims strength/authority, not whether the evidence is topically the
right match at all.

v5's fix: gate substitution to `category_tier == "prototype"` -- i.e. the
evidence item already has >=2 exact_keywords independently matched against
this requirement's text (schemas.classify_category_tier), not just the one
keyword substitution would add. "near_prototype" (a single, possibly
coincidental keyword -- exactly the failure mode v4 hit) is excluded.
Substitution may only reinforce an already-multiply-corroborated match,
never be what makes a marginal match look legitimate.

This is a validation script, not a production change: the verbatim
sentence-selector itself (v2/v3/v4/v5) is still experimental and not wired
into tailor_resume's production degraded-mode path.

No LLM call anywhere in this file.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import (  # noqa: E402
    build_job_schema_representation,
    claim_ceiling_for_evidence,
    agency_ceiling_for_evidence,
    check_claim_strength,
    check_agency_strength,
)

OUT_DIR = Path(__file__).resolve().parent

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Same hand-verified table as v4 -- unchanged, this experiment only adds a
# gate on WHEN substitution is attempted, not which pairs are considered safe.
_SYNONYM_SUBSTITUTIONS: dict[str, tuple[re.Pattern, str]] = {
    "problems": (re.compile(r"\bissues\b", re.IGNORECASE), "problems"),
}


def _candidate_sentences(item: dict) -> list[str]:
    sentences: list[str] = []
    for resp in item.get("responsibilities") or []:
        if isinstance(resp, str) and resp.strip():
            sentences.append(resp.strip())
    desc = item.get("description")
    if isinstance(desc, str) and desc.strip():
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(desc.strip()) if s.strip())
    return sentences


def _find_item(name: str, profile: dict) -> dict | None:
    for key in ("experience_inventory", "project_inventory"):
        for it in profile.get(key) or []:
            if it.get("name") == name:
                return it
    return None


def _literal_score(sentence: str, exact_keywords: list[str]) -> int:
    sl = sentence.lower()
    return sum(1 for kw in exact_keywords if re.search(rf"\b{re.escape(kw.lower())}\b", sl))


def _apply_gated_synonym_substitution(
    sentence: str, exact_keywords: list[str], category_tier: str | None
) -> tuple[str, list[str]]:
    """Same substitution logic as v4, but only attempted at all when
    category_tier == 'prototype' -- the gate this experiment is testing."""
    if category_tier != "prototype":
        return sentence, []
    applied = []
    sl = sentence.lower()
    for kw in exact_keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", sl):
            continue
        sub = _SYNONYM_SUBSTITUTIONS.get(kw.lower())
        if sub is None:
            continue
        pattern, replacement = sub
        m = pattern.search(sentence)
        if not m:
            continue
        sentence = pattern.sub(replacement, sentence, count=1)
        sl = sentence.lower()
        applied.append(f"{m.group()} -> {replacement}")
    return sentence, applied


def _safety_check(new_sentence: str, evidence_item: dict) -> dict:
    claim_ceiling = claim_ceiling_for_evidence(evidence_item)
    agency_ceiling = agency_ceiling_for_evidence(evidence_item)
    cs = check_claim_strength(new_sentence, claim_ceiling)
    ags = check_agency_strength(new_sentence, agency_ceiling)
    return {"claim_ok": cs["passed"], "agency_ok": ags["passed"]}


def _would_ungated_fire(sentence: str, exact_keywords: list[str]) -> bool:
    """Would v4's ungated rule attempt a substitution on this sentence?"""
    sl = sentence.lower()
    for kw in exact_keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", sl):
            continue
        sub = _SYNONYM_SUBSTITUTIONS.get(kw.lower())
        if sub and sub[0].search(sentence):
            return True
    return False


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
    job_ids = random.sample(pool, min(100, len(pool)))

    n = 0
    substitution_examples: list[dict] = []
    blocked_by_gate: list[dict] = []
    would_have_fired_ungated = 0

    for job_id in job_ids:
        row = conn.execute(
            "SELECT rowid, url, title, company, site, location, full_description FROM jobs WHERE rowid=?", (job_id,)
        ).fetchone()
        if row is None:
            continue
        job = dict(row)
        schema = build_job_schema_representation(job, profile)
        for r in schema["requirements"]:
            if not (r.get("supported") and r.get("schema") and r.get("category_tier") in ("prototype", "near_prototype")):
                continue
            ev = r.get("resume_evidence") or []
            if not ev:
                continue
            item = _find_item(ev[0], profile)
            if item is None:
                continue
            kws = r.get("exact_keywords") or []

            best_s, best_sc = None, -1
            for s in _candidate_sentences(item):
                sc = _literal_score(s, kws)
                if sc > best_sc or (sc == best_sc and best_s is not None and len(s) > len(best_s)):
                    best_s, best_sc = s, sc
            if best_s is None:
                continue
            n += 1

            ungated_would_fire = _would_ungated_fire(best_s, kws)
            if ungated_would_fire:
                would_have_fired_ungated += 1

            new_sentence, applied = _apply_gated_synonym_substitution(best_s, kws, r.get("category_tier"))
            if applied:
                safety = _safety_check(new_sentence, item)
                if not (safety["claim_ok"] and safety["agency_ok"]):
                    new_sentence = best_s
                    applied = []
                else:
                    substitution_examples.append(
                        {
                            "title": job["title"][:50],
                            "requirement": r["requirement"][:90],
                            "category_tier": r.get("category_tier"),
                            "exact_keywords": kws,
                            "before": best_s,
                            "after": new_sentence,
                            "substitutions": applied,
                        }
                    )

            if ungated_would_fire and not applied:
                blocked_by_gate.append(
                    {
                        "title": job["title"][:50],
                        "requirement": r["requirement"][:90],
                        "category_tier": r.get("category_tier"),
                        "sentence": best_s,
                    }
                )

    print(f"bullets evaluated: {n}")
    print(f"substitutions that WOULD have fired under v4's ungated rule: {would_have_fired_ungated}")
    print(f"substitutions actually applied under the 'prototype'-tier gate: {len(substitution_examples)}")
    print(f"substitutions blocked by the gate: {len(blocked_by_gate)}")

    print(f"\n=== substitutions BLOCKED by the gate ({len(blocked_by_gate)}) ===")
    for ex in blocked_by_gate:
        print(f"[{ex['title']}] tier={ex['category_tier']}")
        print(f"  req: {ex['requirement']}")
        print(f"  sentence: {ex['sentence']}")

    print(f"\n=== substitutions actually applied under the gate ({len(substitution_examples)}) ===")
    for ex in substitution_examples:
        print(f"[{ex['title']}] tier={ex['category_tier']} keywords={ex['exact_keywords']}")
        print(f"  req: {ex['requirement']}")
        print(f"  before: {ex['before']}")
        print(f"  after:  {ex['after']}")
        print(f"  subs:   {ex['substitutions']}")

    (OUT_DIR / "prototype_v5_gated_synonym_results.json").write_text(
        json.dumps(
            {
                "n": n,
                "would_have_fired_ungated": would_have_fired_ungated,
                "applied_under_gate": len(substitution_examples),
                "blocked_by_gate": blocked_by_gate,
                "substitution_examples": substitution_examples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nwrote prototype_v5_gated_synonym_results.json")


if __name__ == "__main__":
    main()
