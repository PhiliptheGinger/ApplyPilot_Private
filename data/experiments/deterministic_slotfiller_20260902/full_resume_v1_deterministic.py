"""First real test: can we produce a complete, plausible tailored resume
entirely deterministically -- zero LLM calls, cloud or local -- using only
verbatim sentence selection from the candidate's own evidence, layered
onto the SAME production merge machinery degraded mode already uses
(build_base_resume_model -> [[realization]] -> merge_realization)?

This reuses:
  - build_job_schema_representation (schemas.py, production, already has
    tonight's fixes: inflection-tolerant matching, ambiguous-term context
    check, constraints wiring)
  - build_base_resume_model / merge_realization (local_tailor.py,
    production, already used by degraded mode -- just swapping the
    "realize a bullet" step for sentence-selection instead of an LLM call)
  - validate_json_fields (validator.py, production) -- the SAME safety
    check a real cloud- or local-generated resume has to pass, run here
    against a deterministic one, so this isn't graded on a curve

Bullet composition: for every evidence item backing at least one
schema-supported, category_tier in (prototype, near_prototype) requirement
(the SAME eligibility gate _build_realization_prompt already uses), gather
the UNION of exact_keywords across all requirements it backs, then pick
the real sentence (from that item's own responsibilities/description) with
the highest keyword overlap -- same selection logic validated across the
v2-v6 experiments, one bullet per evidence item (matching the granularity
merge_realization already expects).

Summary is left untouched (falls back to the original resume's own
summary via merge_realization) -- deliberately not attempting deterministic
summary generation in this first pass; scoping discipline, not an
oversight.

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
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402
from applypilot.scoring.local_tailor import build_base_resume_model, merge_realization  # noqa: E402
from applypilot.scoring.resume_router import load_resume_text_for_job  # noqa: E402
from applypilot.scoring.validator import validate_json_fields  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
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


def _find_item(name: str, profile: dict) -> dict | None:
    for key in ("experience_inventory", "project_inventory"):
        for it in profile.get(key) or []:
            if it.get("name") == name:
                return it
    return None


def _score(sentence: str, keywords: set[str]) -> int:
    sl = sentence.lower()
    return sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw.lower())}\b", sl))


def compose_deterministic_realization(job_schema: dict, profile: dict) -> dict:
    evidence_keywords: dict[str, set[str]] = {}
    for r in job_schema.get("requirements") or []:
        if not (r.get("supported") and r.get("schema") and r.get("category_tier") in ("prototype", "near_prototype")):
            continue
        ev = r.get("resume_evidence") or []
        if not ev:
            continue
        evidence_keywords.setdefault(ev[0], set()).update(r.get("exact_keywords") or [])

    bullets: dict[str, str] = {}
    for name, kws in evidence_keywords.items():
        item = _find_item(name, profile)
        if item is None:
            continue
        best_s, best_sc = None, -1
        for s in _candidate_sentences(item):
            sc = _score(s, kws)
            if sc > best_sc or (sc == best_sc and best_s is not None and len(s) > len(best_s)):
                best_s, best_sc = s, sc
        if best_s is not None:
            bullets[name] = best_s

    return {"summary": None, "bullets": bullets}


def run_one(job: dict, resume_text: str, profile: dict) -> dict:
    job_schema = build_job_schema_representation(job, profile)
    base_resume = build_base_resume_model(resume_text, profile)
    realization = compose_deterministic_realization(job_schema, profile)
    merged = merge_realization(base_resume, realization, job)
    validation = validate_json_fields(merged, profile, job=job)
    return {
        "job_url": job.get("url"),
        "title": job.get("title"),
        "n_evidence_realized": len(realization["bullets"]),
        "n_requirements_supported_eligible": sum(
            1
            for r in job_schema.get("requirements") or []
            if r.get("supported") and r.get("schema") and r.get("category_tier") in ("prototype", "near_prototype")
        ),
        "validation_passed": validation["passed"],
        "validation_errors": validation["errors"],
        "validation_warnings": validation["warnings"],
        "resume": merged,
    }


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()

    # Start with exactly ONE real, high-scoring job.
    row = conn.execute(
        "SELECT rowid, url, title, full_description FROM jobs "
        "WHERE full_description IS NOT NULL AND length(full_description) > 800 "
        "AND fit_score >= 7 ORDER BY rowid LIMIT 1"
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT rowid, url, title, full_description FROM jobs "
            "WHERE full_description IS NOT NULL AND length(full_description) > 800 LIMIT 1"
        ).fetchone()

    job = {"url": row[1], "title": row[2], "full_description": row[3]}
    print(f"job: {job['title']}  ({job['url']})\n")

    # SAME resume_text source the real production tailor_resume() call
    # uses (resume_router.load_resume_text_for_job) -- not RESUME_PATH
    # directly, which is what v1 of this script did before the base-
    # resume-parsing bug was found and fixed.
    resume_text, resume_source_path = load_resume_text_for_job(job)
    print(f"resume_text source: {resume_source_path}\n")

    result = run_one(job, resume_text, profile)
    print(f"requirements eligible for realization: {result['n_requirements_supported_eligible']}")
    print(f"evidence items realized: {result['n_evidence_realized']}")
    print(f"validation passed: {result['validation_passed']}")
    if result["validation_errors"]:
        print("errors:")
        for e in result["validation_errors"]:
            print("  -", e)
    if result["validation_warnings"]:
        print("warnings:")
        for w in result["validation_warnings"]:
            print("  -", w)

    print("\n=== RESUME ===")
    r = result["resume"]
    print(f"TITLE: {r['title']}")
    print(f"SUMMARY: {r['summary']}")
    print(f"SKILLS: {r['skills']}")
    print("\nEXPERIENCE:")
    for e in r["experience"]:
        print(f"  {e['header']} | {e['subtitle']}")
        for b in e["bullets"]:
            print(f"    - {b}")
    print("\nPROJECTS:")
    for e in r["projects"]:
        print(f"  {e['header']} | {e['subtitle']}")
        for b in e["bullets"]:
            print(f"    - {b}")
    print(f"\nEDUCATION: {r['education']}")

    (OUT_DIR / "full_resume_v1_one_job.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nwrote full_resume_v1_one_job.json")


if __name__ == "__main__":
    main()
