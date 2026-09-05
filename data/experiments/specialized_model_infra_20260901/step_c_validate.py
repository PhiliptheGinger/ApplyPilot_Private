"""STEP C (runs in the PRODUCTION venv): parse each specialized model's raw
output through the EXACT same pipeline the corrected qwen3 bake-off used
(_parse_plan -> claim/agency/causal safety checks -> merge_realization ->
validate_json_fields -> assemble_resume_text -> judge_tailored_resume if
validation passes), so all three local models (qwen3, T5, TinyLlama) are
scored by identical, unmodified production logic.

No production code modified. No DB writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config  # noqa: E402
from applypilot.llm import get_stage_client  # noqa: E402
from applypilot.scoring.local_tailor import (  # noqa: E402
    _parse_plan,
    build_base_resume_model,
    merge_realization,
)
from applypilot.scoring.schemas import (  # noqa: E402
    AGENCY_TIERS,
    CLAIM_TIERS,
    check_agency_strength,
    check_causal_claim,
    check_claim_strength,
    check_metric_fabrication,
    check_passive_voice,
)
from applypilot.scoring.tailor import assemble_resume_text, judge_tailored_resume  # noqa: E402
from applypilot.scoring.validator import validate_json_fields  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
PROMPTS = json.loads((OUT_DIR / "shared_prompts.json").read_text(encoding="utf-8"))
profile = config.load_profile()
resume_text = (OUT_DIR / "shared_resume_text.txt").read_text(encoding="utf-8")
base_resume = build_base_resume_model(resume_text, profile)


def realize_from_raw(raw_text: str, job_schema: dict) -> tuple[dict | None, dict]:
    """Mirrors request_local_realization's post-processing (claim/agency/
    causal/metric safety checks) but starting from an already-obtained raw
    model response instead of calling client.chat() again -- lets us score
    T5/TinyLlama's actual output through the identical safety net qwen3's
    output went through."""
    meta = {"violations": 0, "violation_reasons": []}
    try:
        parsed = _parse_plan(raw_text)
    except Exception as exc:  # noqa: BLE001 -- mirrors request_local_realization's own try/except around this call
        meta["parse_failed"] = True
        meta["parse_error"] = f"{type(exc).__name__}: {exc}"
        return None, meta
    if not isinstance(parsed, dict):
        meta["parse_failed"] = True
        return None, meta
    meta["parse_failed"] = False

    known_metrics = ((profile or {}).get("resume_facts") or {}).get("real_metrics") or []
    ceiling_by_evidence: dict[str, str] = {}
    agency_ceiling_by_evidence: dict[str, str] = {}
    provenance_by_evidence: dict[str, str] = {}
    for r in job_schema.get("requirements") or []:
        if not r.get("supported"):
            continue
        for name in r.get("resume_evidence") or []:
            ceiling_by_evidence[name] = r.get("claim_ceiling") or "participation"
            agency_ceiling_by_evidence[name] = r.get("agency_ceiling") or "individual_contributor"
            provenance = r.get("provenance") or []
            if provenance:
                first = provenance[0]
                provenance_by_evidence[name] = first.get("text", "") if isinstance(first, dict) else str(first)

    bullets: dict[str, str] = {}
    raw_bullets = parsed.get("bullets")
    if isinstance(raw_bullets, dict):
        raw_bullets = [{"evidence": k, "text": v} for k, v in raw_bullets.items()]
    for b in raw_bullets or []:
        if not (isinstance(b, dict) and b.get("evidence") and b.get("text")):
            continue
        evidence_name = str(b["evidence"]).strip()
        text = str(b["text"]).strip()
        evidence_text = provenance_by_evidence.get(evidence_name, "")
        ceiling = ceiling_by_evidence.get(evidence_name, "participation")
        agency_ceiling = agency_ceiling_by_evidence.get(evidence_name, "individual_contributor")

        checks = [
            check_claim_strength(text, ceiling),
            check_agency_strength(text, agency_ceiling),
            check_causal_claim(text, evidence_text),
            check_metric_fabrication(text, evidence_text, known_metrics),
        ]
        failed = next((c for c in checks if not c["passed"]), None)
        if failed:
            meta["violations"] += 1
            meta["violation_reasons"].append(f"{evidence_name}: {failed.get('violation')}")
            continue
        bullets[evidence_name] = text

    summary = parsed.get("summary")
    if summary:
        summary_ceiling = max(ceiling_by_evidence.values(), default="participation", key=lambda c: CLAIM_TIERS.index(c) if c in CLAIM_TIERS else 0)
        s_check = check_claim_strength(str(summary), summary_ceiling)
        if not s_check["passed"]:
            meta["violations"] += 1
            meta["violation_reasons"].append(f"summary: {s_check.get('violation')}")
            summary = None

    if not bullets and not summary:
        return None, meta
    return {"summary": summary, "bullets": bullets}, meta


def main() -> None:
    all_results = {}
    for model_key in ("t5", "tinyllama"):
        raw_path = OUT_DIR / f"raw_output_{model_key}.json"
        if not raw_path.exists():
            continue
        raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
        model_results = {}
        for job_id, entry in raw_data["jobs"].items():
            job_prompt_entry = PROMPTS["jobs"][job_id]
            job = job_prompt_entry["job"]
            job_schema = job_prompt_entry["job_schema"]
            standup_decision = job_prompt_entry["standup_decision"]

            if entry.get("skipped"):
                model_results[job_id] = {"status": "no_supported_evidence", "note": entry["skipped"]}
                continue
            if not entry.get("succeeded"):
                model_results[job_id] = {"status": "inference_failed", "error": entry.get("error")}
                continue

            realization, safety_meta = realize_from_raw(entry["raw_output"], job_schema)
            result = {"latency_s": entry["latency_s"], "safety_check_meta": safety_meta, "raw_output_preview": entry["raw_output"][:300]}

            if realization is None:
                result["status"] = "local_realization_failed" if safety_meta.get("parse_failed") else "all_content_safety_rejected"
                model_results[job_id] = result
                continue

            data = merge_realization(base_resume, realization, job)
            validation = validate_json_fields(data, profile, standup_decision=standup_decision)
            result["validator"] = validation
            assembled = assemble_resume_text(data, profile)
            result["resume_text_out_preview"] = assembled[:400]

            if not validation["passed"]:
                result["status"] = "failed_validation"
                model_results[job_id] = result
                continue

            result["status"] = "validated"
            try:
                client = get_stage_client("tailor", quality=True)
                judge = judge_tailored_resume(resume_text, assembled, job["title"], profile)
                result["judge"] = judge
            except Exception as exc:  # noqa: BLE001
                result["judge"] = {"error": str(exc)}
            model_results[job_id] = result

        all_results[model_key] = model_results
        print(f"=== {model_key} ===")
        for jid, r in model_results.items():
            print(f"  job {jid}: {r.get('status')}")

    (OUT_DIR / "step_c_validated_results.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
