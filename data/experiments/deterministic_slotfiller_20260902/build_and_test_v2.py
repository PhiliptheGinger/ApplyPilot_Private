"""Deterministic slot-filler v2: SELECTION over GENERATION.

v1 (build_and_test.py) tried to reconstruct a sentence from matched
keyword fragments -- broke down badly once _item_terms started splitting
`responsibilities` sentences into individual words for matching purposes
("Used develop and environment." from a sentence about developing outreach
strategies). Now that experience_inventory has real, human-authored,
sentence-level `responsibilities` text (schemas.py/local_tailor.py wiring
fix, same session), the better primitive is SELECT the real sentence that
best overlaps the job's exact_keywords and use it near-verbatim -- zero
generation, zero fabrication risk by construction, and it reads naturally
because it already IS natural prose a human wrote and approved.

Only `project_inventory` entries (factual_concepts are short noun phrases,
not full sentences -- there's nothing to select) still need light
template construction, capped at ONE concept phrase per bullet (not
joined lists, which is what produced "Designed data, financial/news
data, and data collection." in v1).

No LLM call anywhere in this file.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

_CANONICAL_VERB = {
    "participation": "Used",
    "execution": "Operated",
    "implementation": "Built",
    "design": "Designed",
    "authority": "Architected",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _find_evidence_item(evidence_name: str, profile: dict) -> dict | None:
    for key in ("experience_inventory", "historical_experience_inventory", "qualifications", "project_inventory"):
        for item in profile.get(key) or []:
            if str(item.get("name") or "").strip() == evidence_name:
                return item
    return None


def _candidate_sentences(evidence_item: dict) -> list[str]:
    """Real, already-authored sentences this item's OWN text contains --
    responsibilities entries are already one sentence each; description
    may be multiple sentences, so it's split."""
    sentences: list[str] = []
    for resp in evidence_item.get("responsibilities") or []:
        if isinstance(resp, str) and resp.strip():
            sentences.append(resp.strip())
    desc = evidence_item.get("description")
    if isinstance(desc, str) and desc.strip():
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(desc.strip()) if s.strip())
    return sentences


def _best_sentence(sentences: list[str], exact_keywords: list[str]) -> tuple[str, int] | None:
    """Pick the sentence with the most word-boundary keyword hits. Ties
    (or near-ties -- a short fragment with 1 hit vs. a full sentence with
    1 hit is a real collision seen in practice: old bare noun fragments
    like "Vehicle alignment work" can tie a genuinely fuller, more
    complete real sentence on raw keyword count and get picked instead)
    broken toward the LONGER, more complete sentence -- a fuller real
    sentence is always at least as informative as a short fragment that
    scored the same."""
    if not sentences:
        return None
    best, best_score = None, -1
    for s in sentences:
        s_lower = s.lower()
        score = sum(1 for kw in exact_keywords if re.search(rf"\b{re.escape(kw.lower())}\b", s_lower))
        if score > best_score or (score == best_score and best is not None and len(s) > len(best)):
            best, best_score = s, score
    return (best, best_score) if best is not None else None


def compose_bullet(entry: dict, profile: dict) -> dict:
    evidence_names = entry.get("resume_evidence") or []
    if not evidence_names:
        return {"bullet": None, "reason": "no_evidence_name"}
    evidence_item = _find_evidence_item(evidence_names[0], profile)
    if evidence_item is None:
        return {"bullet": None, "reason": "evidence_not_found_in_profile"}

    exact_keywords = list(entry.get("exact_keywords") or [])

    # Path A: real sentence selection (experience_inventory / any item with
    # responsibilities or a prose description).
    sentences = _candidate_sentences(evidence_item)
    if sentences:
        picked = _best_sentence(sentences, exact_keywords)
        if picked:
            sentence, score = picked
            return {
                "bullet": sentence,
                "reason": "ok",
                "method": "sentence_selection",
                "match_score": score,
                "evidence": evidence_names[0],
            }

    # Path B: template construction from factual_concepts (project_inventory
    # -- short noun phrases, nothing sentence-shaped to select). Capped at
    # ONE concept phrase, prioritizing an exact_keyword hit if present.
    concepts = [str(c) for c in (evidence_item.get("factual_concepts") or [])]
    if concepts:
        chosen = next((c for c in concepts if any(kw.lower() in c.lower() for kw in exact_keywords)), concepts[0])
        verb = _CANONICAL_VERB.get(entry.get("claim_ceiling") or "participation", "Used")
        lowered = chosen[0].lower() + chosen[1:] if chosen[0].isupper() and not chosen.isupper() else chosen
        return {
            "bullet": f"{verb} {lowered}.",
            "reason": "ok",
            "method": "single_concept_template",
            "evidence": evidence_names[0],
        }

    return {"bullet": None, "reason": "no_evidence_text"}


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
            comp = compose_bullet(r, profile)
            entry_out["bullets"].append({"requirement": r["requirement"][:80], **comp})
            if comp["bullet"]:
                print(f"  [{comp.get('method')}] {comp['bullet']}")
            else:
                print(f"  SKIPPED ({comp['reason']}) -- requirement: {r['requirement'][:80]}")
        results.append(entry_out)

    (OUT_DIR / "prototype_v2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    total_gated = sum(len(e["bullets"]) for e in results)
    total_composed = sum(1 for e in results for b in e["bullets"] if b["bullet"])
    by_method: dict[str, int] = {}
    for e in results:
        for b in e["bullets"]:
            if b["bullet"]:
                by_method[b.get("method", "?")] = by_method.get(b.get("method", "?"), 0) + 1
    print("\n=== SUMMARY ===")
    print(f"gated requirements: {total_gated}")
    print(f"composed (non-empty bullet): {total_composed}/{total_gated}")
    print(f"by method: {by_method}")


if __name__ == "__main__":
    main()
