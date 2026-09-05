"""Same corrected re-score as rescore.py (keyword-injection check +
seniority-gate exclusion + github/non_github split), but pointed at the
_postfix batches -- collected AFTER wiring `responsibilities` into
schemas.py/local_tailor.py's evidence-text extraction and enriching
profile.json's experience_inventory. Direct before/after comparison
against rescore.py's original numbers (github 3/40=7.5%, non_github
1/29=3.4% for "old" prompt) on the identical 25 jobs.

Processes whichever _postfix files exist yet -- tolerant of apex/qwen25
not having finished if this is run while they're still in flight.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import (  # noqa: E402
    check_agency_strength,
    check_causal_claim,
    check_claim_strength,
    check_metric_fabrication,
    get_or_build_job_schema,
)
from applypilot.scoring.local_tailor import _parse_plan  # noqa: E402
from applypilot.scoring.tailor import classify_seniority_mismatch  # noqa: E402
from applypilot.scoring.validator import _distinctive_words, _entry_evidence_text  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PROJECT_NAMES = {"Standup-OCR", "CAP Predictor", "Express_Engine", "You Power You", "I_hate_social_media", "Sunburn", "ApplyPilot"}

TAGS = (("qwen3", "qwen3_postfix"), ("apex", "apex_postfix"), ("qwen25", "qwen25_postfix"))


def build_entry_evidence(profile: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in ("experience_inventory", "historical_experience_inventory", "qualifications", "project_inventory"):
        for item in profile.get(key) or []:
            name = str(item.get("name") or "").strip()
            if name:
                out[name] = _entry_evidence_text(item).lower()
    return out


def rescore_bullet(evidence_name: str, text: str, job_vocab: set[str], entry_evidence_words: dict[str, set[str]]) -> dict:
    ev_words = entry_evidence_words.get(evidence_name, set())
    bullet_words = _distinctive_words(text)
    injected = (bullet_words & job_vocab) - ev_words
    return {"keyword_injected": len(injected) >= 2, "injected_terms": sorted(injected)[:5]}


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    entry_evidence = build_entry_evidence(profile)
    entry_evidence_words = {name: _distinctive_words(text) for name, text in entry_evidence.items()}

    overall = {"github": {"old": [0, 0], "new": [0, 0]}, "non_github": {"old": [0, 0], "new": [0, 0]}}
    detail_examples = []

    for model_name, tag in TAGS:
        path = OUT_DIR / f"batch25_results_{tag}.json"
        if not path.exists():
            print(f"skipping {model_name} -- {path.name} not present yet")
            continue
        d = json.load(open(path, encoding="utf-8"))
        model_stats = {"github": {"old": [0, 0], "new": [0, 0]}, "non_github": {"old": [0, 0], "new": [0, 0]}}
        for entry in d:
            job_id = entry["job_id"]
            row = conn.execute("SELECT rowid, title, full_description FROM jobs WHERE rowid=?", (job_id,)).fetchone()
            if row is None:
                continue
            job = dict(row)
            if classify_seniority_mismatch(job, profile):
                continue

            job_schema = get_or_build_job_schema(job, profile)
            job_vocab = _distinctive_words((job.get("title") or "") + " " + (job.get("full_description") or ""))

            ceiling_by_evidence: dict[str, str] = {}
            agency_ceiling_by_evidence: dict[str, str] = {}
            provenance_by_evidence: dict[str, str] = {}
            for r in job_schema.get("requirements") or []:
                if not r.get("supported"):
                    continue
                for name in r.get("resume_evidence") or []:
                    ceiling_by_evidence[name] = r.get("claim_ceiling") or "participation"
                    agency_ceiling_by_evidence[name] = r.get("agency_ceiling") or "individual_contributor"
                    prov = r.get("provenance") or []
                    if prov:
                        first = prov[0]
                        provenance_by_evidence[name] = first.get("text", "") if isinstance(first, dict) else str(first)

            for label in ("old", "new"):
                cond = entry.get(label, {})
                raw = cond.get("raw_output")
                if not raw:
                    continue
                try:
                    parsed = _parse_plan(raw)
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                raw_bullets = parsed.get("bullets")
                if isinstance(raw_bullets, dict):
                    raw_bullets = [{"evidence": k, "text": v} for k, v in raw_bullets.items()]
                for b in raw_bullets or []:
                    if not (isinstance(b, dict) and b.get("evidence") and b.get("text")):
                        continue
                    name = str(b["evidence"]).strip()
                    text = str(b["text"]).strip()
                    category = "github" if name in PROJECT_NAMES else "non_github"
                    ceiling = ceiling_by_evidence.get(name, "participation")
                    agency_ceiling = agency_ceiling_by_evidence.get(name, "individual_contributor")
                    evidence_text = provenance_by_evidence.get(name, "")
                    checks = [
                        check_claim_strength(text, ceiling),
                        check_agency_strength(text, agency_ceiling),
                        check_causal_claim(text, evidence_text),
                        check_metric_fabrication(text, evidence_text, (profile.get("resume_facts") or {}).get("real_metrics") or []),
                    ]
                    passed_narrow = all(c["passed"] for c in checks)
                    kw = rescore_bullet(name, text, job_vocab, entry_evidence_words)

                    model_stats[category][label][0] += 1
                    if passed_narrow and not kw["keyword_injected"]:
                        model_stats[category][label][1] += 1

                    if passed_narrow and kw["keyword_injected"] and len(detail_examples) < 15:
                        detail_examples.append(
                            {"model": model_name, "prompt": label, "job": job["title"], "evidence": name, "text": text, "injected_terms": kw["injected_terms"]}
                        )

        print(f"\n=== {model_name} (POST-FIX: corrected checks, seniority-gated jobs excluded) ===")
        for category in ("github", "non_github"):
            for label in ("old", "new"):
                offered, survived = model_stats[category][label]
                rate = survived / offered if offered else 0.0
                print(f"  {category:10s} / {label}: {survived}/{offered} = {rate:.1%}")
                overall[category][label][0] += offered
                overall[category][label][1] += survived

    print("\n\n=== TOTALS ACROSS AVAILABLE MODELS (POST-FIX) ===")
    for category in ("github", "non_github"):
        for label in ("old", "new"):
            offered, survived = overall[category][label]
            rate = survived / offered if offered else 0.0
            print(f"  {category:10s} / {label}: {survived}/{offered} = {rate:.1%}")

    print("\n=== Examples caught ONLY by the keyword-injection check ===")
    for ex in detail_examples:
        print(f"  [{ex['model']}/{ex['prompt']}] job='{ex['job'][:40]}' evidence='{ex['evidence']}'")
        print(f"    text: {ex['text'][:200]}")
        print(f"    injected job-vocab terms: {ex['injected_terms']}")

    (OUT_DIR / "rescore_postfix_summary.json").write_text(
        json.dumps({"overall": overall, "flagged_examples": detail_examples}, indent=2, default=str), encoding="utf-8"
    )
    print("\nDONE")


if __name__ == "__main__":
    main()
