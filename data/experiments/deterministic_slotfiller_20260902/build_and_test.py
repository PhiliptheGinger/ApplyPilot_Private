"""Prototype: deterministic (no-LLM) bullet slot-filler for degraded mode,
gated to prototype/near_prototype (literal-match) requirements only -- the
same population local_tailor._build_realization_prompt now restricts to.

Not wired into production. This is a "start small, see if it's any good"
prototype per explicit user request, run against real jobs and printed for
direct comparison against real saved local-LLM output from the earlier
batch-25 experiment (same jobs where possible).

No LLM call anywhere in this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# Canonical present-tense-past verb per claim tier, first/most natural form
# from schemas._CLAIM_VERB_PATTERNS -- same lattice, just picking ONE form
# per tier instead of matching many.
_CANONICAL_VERB = {
    "participation": "Used",
    "execution": "Operated",
    "implementation": "Built",
    "design": "Designed",
    "authority": "Architected",
}


def _find_evidence_item(evidence_name: str, profile: dict) -> dict | None:
    for key in ("experience_inventory", "historical_experience_inventory", "qualifications", "project_inventory"):
        for item in profile.get(key) or []:
            if str(item.get("name") or "").strip() == evidence_name:
                return item
    return None


def _object_phrase(entry: dict, evidence_item: dict) -> str | None:
    """Build the object clause purely from evidence the item ALREADY has --
    factual_concepts first (most specific, human-authored), then
    exact_keywords (guaranteed literal overlap with the job posting, by
    construction of the prototype/near_prototype gate), then
    relevance_categories as a last resort (category tags only, weakest
    material -- flagged in the output so this is visible, not hidden)."""
    concepts = [str(c) for c in (evidence_item.get("factual_concepts") or [])]
    exact = list(entry.get("exact_keywords") or [])
    ordered = []
    for term in exact + concepts:
        if term not in ordered:
            ordered.append(term)
    if ordered:
        return _join_natural(ordered[:3]), "factual_concepts/exact_keywords"

    categories = [str(c) for c in (evidence_item.get("relevance_categories") or [])]
    if categories:
        return _join_natural(categories[:3]), "relevance_categories (weak -- category tags only)"

    return None, "no_evidence_text"


def _join_natural(items: list[str]) -> str:
    items = [i[0].lower() + i[1:] if i and i[0].isupper() and not i.isupper() else i for i in items]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def compose_deterministic_bullet(entry: dict, profile: dict) -> dict:
    evidence_names = entry.get("resume_evidence") or []
    if not evidence_names:
        return {"bullet": None, "reason": "no_evidence_name"}
    evidence_item = _find_evidence_item(evidence_names[0], profile)
    if evidence_item is None:
        return {"bullet": None, "reason": "evidence_not_found_in_profile"}

    verb = _CANONICAL_VERB.get(entry.get("claim_ceiling") or "participation", "Used")
    obj_result = _object_phrase(entry, evidence_item)
    obj_phrase, source = obj_result
    if obj_phrase is None:
        return {"bullet": None, "reason": source}

    bullet = f"{verb} {obj_phrase}."
    return {"bullet": bullet, "reason": "ok", "object_source": source, "evidence": evidence_names[0]}


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()

    import random

    random.seed(20260902)
    all_qualifying = conn.execute(
        "SELECT rowid FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 800"
    ).fetchall()
    pool = [r[0] for r in all_qualifying]
    job_ids = random.sample(pool, min(15, len(pool)))

    results = []
    for job_id in job_ids:
        row = conn.execute(
            "SELECT rowid, url, title, company, site, location, full_description FROM jobs WHERE rowid=?", (job_id,)
        ).fetchone()
        if row is None:
            continue
        job = dict(row)
        job_schema = build_job_schema_representation(job, profile)
        gated = [
            r
            for r in job_schema["requirements"]
            if r.get("supported") and r.get("schema") and r.get("category_tier") in ("prototype", "near_prototype")
        ]
        if not gated:
            print(f"job {job_id} ({job['title'][:50]}): no literal-match requirements, nothing to compose")
            continue
        print(f"\njob {job_id} ({job['title'][:60]}):")
        entry_out = {"job_id": job_id, "title": job["title"], "bullets": []}
        for r in gated:
            comp = compose_deterministic_bullet(r, profile)
            entry_out["bullets"].append({"requirement": r["requirement"][:80], **comp})
            if comp["bullet"]:
                print(f"  [{comp.get('object_source')}] {comp['bullet']}")
            else:
                print(f"  SKIPPED ({comp['reason']}) -- requirement: {r['requirement'][:80]}")
        results.append(entry_out)

    (OUT_DIR / "prototype_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    total_gated = sum(len(e["bullets"]) for e in results)
    total_composed = sum(1 for e in results for b in e["bullets"] if b["bullet"])
    weak_source = sum(
        1 for e in results for b in e["bullets"] if b["bullet"] and "weak" in (b.get("object_source") or "")
    )
    print(f"\n=== SUMMARY ===")
    print(f"gated requirements: {total_gated}")
    print(f"composed (non-empty bullet): {total_composed}/{total_gated}")
    print(f"composed from weak (category-tag-only) evidence: {weak_source}/{total_composed}")


if __name__ == "__main__":
    main()
