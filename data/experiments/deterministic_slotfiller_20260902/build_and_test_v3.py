"""Deterministic slot-filler v3: widen the sentence-selection candidate
pool beyond the single evidence item schemas.py assigned to a
requirement.

Finding from v2 (see report / decision discussion): the reuse of ~27
sentences across many jobs isn't "the best sentences winning
universally" -- it's the ONLY sentence in a tiny per-item pool (most
experience_inventory items have 3-10 responsibilities sentences)
winning by default, often with match_score==1 (weakest possible
nonzero keyword overlap). The real problem is coverage, not diversity:
a forced weak-fit bullet inside ONE application, not repetition across
applications no single reader ever compares.

v3 hypothesis: widen the candidate pool per requirement to every item
already in that job's own `ranked_evidence` (not just the one
schemas.py happened to assign as "the" supporting item), and see if
match quality (weak-match rate) improves. This is safe by construction
for the SAME reason v2 was safe: every candidate sentence is still
100% real, human-authored text, and every candidate SOURCE ITEM is
still independently, legitimately relevant to this job (it has >=1 of
its own matched terms against the job description -- that's the
definition of membership in ranked_evidence). We are not attributing a
sentence to a job it has zero relation to; we're just no longer
committing to only the #1-ranked item's own (possibly tiny) sentence
pool when a #2 or #3 ranked item -- also genuinely relevant to this
job -- happens to have a stronger sentence for THIS SPECIFIC
requirement's wording.

No LLM call anywhere in this file. No claim/agency ceiling recompute
needed: verbatim sentence selection carries no ceiling risk regardless
of which ranked item it comes from (the risk only exists for
GENERATED text, i.e. the Path B template branch, which is unchanged
from v2 and still scoped to the originally-assigned item only).
"""

from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.local_tailor import rank_profile_evidence  # noqa: E402
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


def _candidate_sentences(item: dict) -> list[str]:
    sentences: list[str] = []
    for resp in item.get("responsibilities") or []:
        if isinstance(resp, str) and resp.strip():
            sentences.append(resp.strip())
    desc = item.get("description")
    if isinstance(desc, str) and desc.strip():
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(desc.strip()) if s.strip())
    return sentences


def _score(sentence: str, exact_keywords: list[str]) -> int:
    s_lower = sentence.lower()
    return sum(1 for kw in exact_keywords if re.search(rf"\b{re.escape(kw.lower())}\b", s_lower))


def _best_sentence_single_item(item: dict, exact_keywords: list[str]) -> tuple[str, int] | None:
    """v2 behavior: search only inside the one assigned item."""
    best, best_score = None, -1
    for s in _candidate_sentences(item):
        score = _score(s, exact_keywords)
        if score > best_score or (score == best_score and best is not None and len(s) > len(best)):
            best, best_score = s, score
    return (best, best_score) if best is not None else None


def _best_sentence_pooled(
    ranked_evidence: list[dict], exact_keywords: list[str]
) -> tuple[str, int, str] | None:
    """v3 behavior: search every item already independently ranked as
    relevant to this job (experience/project types only -- skills and
    certifications aren't sentence-shaped), return (sentence, score,
    source_item_name)."""
    best, best_score, best_source = None, -1, None
    for r in ranked_evidence:
        if r["type"] not in ("experience", "project"):
            continue
        item = r["item"]
        for s in _candidate_sentences(item):
            score = _score(s, exact_keywords)
            if score > best_score or (score == best_score and best is not None and len(s) > len(best)):
                best, best_score, best_source = s, score, r["name"]
    return (best, best_score, best_source) if best is not None else None


def _find_item_by_name(name: str, ranked_evidence: list[dict]) -> dict | None:
    for r in ranked_evidence:
        if r["name"] == name:
            return r["item"]
    return None


def compose_v2(entry: dict, ranked_evidence: list[dict]) -> dict:
    evidence_names = entry.get("resume_evidence") or []
    if not evidence_names:
        return {"bullet": None, "reason": "no_evidence_name"}
    item = _find_item_by_name(evidence_names[0], ranked_evidence)
    if item is None:
        return {"bullet": None, "reason": "evidence_not_found"}
    exact_keywords = list(entry.get("exact_keywords") or [])
    picked = _best_sentence_single_item(item, exact_keywords)
    if picked:
        sentence, score = picked
        return {"bullet": sentence, "method": "sentence_selection", "match_score": score, "evidence": evidence_names[0]}
    concepts = [str(c) for c in (item.get("factual_concepts") or [])]
    if concepts:
        chosen = next((c for c in concepts if any(kw.lower() in c.lower() for kw in exact_keywords)), concepts[0])
        verb = _CANONICAL_VERB.get(entry.get("claim_ceiling") or "participation", "Used")
        lowered = chosen[0].lower() + chosen[1:] if chosen[0].isupper() and not chosen.isupper() else chosen
        return {"bullet": f"{verb} {lowered}.", "method": "single_concept_template", "evidence": evidence_names[0]}
    return {"bullet": None, "reason": "no_evidence_text"}


def compose_v3(entry: dict, ranked_evidence: list[dict]) -> dict:
    evidence_names = entry.get("resume_evidence") or []
    if not evidence_names:
        return {"bullet": None, "reason": "no_evidence_name"}
    exact_keywords = list(entry.get("exact_keywords") or [])
    picked = _best_sentence_pooled(ranked_evidence, exact_keywords)
    if picked:
        sentence, score, source = picked
        return {
            "bullet": sentence,
            "method": "sentence_selection_pooled",
            "match_score": score,
            "evidence": source,
            "originally_assigned": evidence_names[0],
            "source_changed": source != evidence_names[0],
        }
    # fall back to v2's template path against the originally-assigned item
    item = _find_item_by_name(evidence_names[0], ranked_evidence)
    if item is not None:
        concepts = [str(c) for c in (item.get("factual_concepts") or [])]
        if concepts:
            chosen = next((c for c in concepts if any(kw.lower() in c.lower() for kw in exact_keywords)), concepts[0])
            verb = _CANONICAL_VERB.get(entry.get("claim_ceiling") or "participation", "Used")
            lowered = chosen[0].lower() + chosen[1:] if chosen[0].isupper() and not chosen.isupper() else chosen
            return {"bullet": f"{verb} {lowered}.", "method": "single_concept_template", "evidence": evidence_names[0]}
    return {"bullet": None, "reason": "no_evidence_text"}


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()

    random.seed(20260903)
    all_qualifying = conn.execute(
        "SELECT rowid FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 800"
    ).fetchall()
    pool = [r[0] for r in all_qualifying]
    sample_n = min(100, len(pool))
    job_ids = random.sample(pool, sample_n)

    v2_bullets: list[dict] = []
    v3_bullets: list[dict] = []
    jobs_considered = 0
    jobs_with_gated = 0

    for job_id in job_ids:
        row = conn.execute(
            "SELECT rowid, url, title, company, site, location, full_description FROM jobs WHERE rowid=?", (job_id,)
        ).fetchone()
        if row is None:
            continue
        job = dict(row)
        jobs_considered += 1
        job_schema = build_job_schema_representation(job, profile)
        ranked_evidence = rank_profile_evidence(job, profile, top_n=6)
        gated = [
            r
            for r in job_schema["requirements"]
            if r.get("supported") and r.get("schema") and r.get("category_tier") in ("prototype", "near_prototype")
        ]
        if not gated:
            continue
        jobs_with_gated += 1
        for r in gated:
            c2 = compose_v2(r, ranked_evidence)
            c3 = compose_v3(r, ranked_evidence)
            if c2.get("bullet"):
                v2_bullets.append({"job_id": job_id, "title": job["title"], "requirement": r["requirement"][:80], **c2})
            if c3.get("bullet"):
                v3_bullets.append({"job_id": job_id, "title": job["title"], "requirement": r["requirement"][:80], **c3})

    def summarize(bullets: list[dict], label: str) -> dict:
        sel = [b for b in bullets if b.get("method", "").startswith("sentence_selection")]
        scores = [b["match_score"] for b in sel]
        score_dist = collections.Counter(scores)
        distinct = len(set(b["bullet"] for b in sel))
        weak = sum(1 for s in scores if s <= 1)
        print(f"\n=== {label} ===")
        print(f"total composed bullets: {len(bullets)}")
        print(f"sentence_selection bullets: {len(sel)}")
        print(f"distinct sentences used: {distinct}")
        print(f"match_score distribution: {dict(sorted(score_dist.items()))}")
        print(f"weak matches (score<=1): {weak}/{len(sel)} ({100*weak/len(sel):.1f}%)" if sel else "weak matches: n/a")
        return {
            "total_composed": len(bullets),
            "sentence_selection_count": len(sel),
            "distinct_sentences": distinct,
            "score_distribution": dict(sorted(score_dist.items())),
            "weak_match_count": weak,
            "weak_match_pct": round(100 * weak / len(sel), 1) if sel else None,
        }

    print(f"jobs sampled: {sample_n}, jobs considered: {jobs_considered}, jobs with gated requirements: {jobs_with_gated}")
    summary_v2 = summarize(v2_bullets, "v2 (single assigned item)")
    summary_v3 = summarize(v3_bullets, "v3 (pooled across ranked_evidence)")

    source_changed = sum(1 for b in v3_bullets if b.get("source_changed"))
    print(f"\nv3 bullets where the selected source item differs from the originally-assigned item: "
          f"{source_changed}/{len([b for b in v3_bullets if 'source_changed' in b])}")

    (OUT_DIR / "prototype_v3_results.json").write_text(
        json.dumps({"v2_bullets": v2_bullets, "v3_bullets": v3_bullets}, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "prototype_v3_summary.json").write_text(
        json.dumps(
            {
                "sample_n": sample_n,
                "jobs_considered": jobs_considered,
                "jobs_with_gated": jobs_with_gated,
                "v2": summary_v2,
                "v3": summary_v3,
                "source_changed": source_changed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nwrote prototype_v3_results.json and prototype_v3_summary.json")


if __name__ == "__main__":
    main()
