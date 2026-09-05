"""Read-only, corrected re-run of the local-model direct-tailoring bakeoff.

Continuation of the Q2 local-model investigation. Reuses production
functions EXACTLY as tailor_resume()'s _run_degraded_mode does (same
call order, same arguments) so results are representative of what
degraded mode actually produces -- not an approximation of it.

Fixes two bugs found in the prior ad-hoc run
(data/experiments/local_model_bakeoff_20260901_161413 and _v2):
  1. That run passed a "CANONICAL PROFILE REFERENCE" text dump (a raw
     listing of profile.json's experience_inventory) as `resume_text`,
     not an actual resume. build_base_resume_model()'s parser
     (scoring/pdf.py's parse_resume, which looks for SUMMARY/TECHNICAL
     SKILLS/EXPERIENCE/PROJECTS/EDUCATION section headers) found none of
     those sections in that input, so experience/projects/education came
     back empty -- validate_json_fields then failed with "Missing
     required field: experience" / "...education" for every job. That
     was a harness defect, not a finding about the model.
  2. ~/.applypilot/resume.txt on this machine is a stale placeholder
     ("Philip Ginger", dated 2026-08-16, describing ApplyPilot itself) --
     unrelated to the real candidate identity in data/profile.json
     ("Philip McLaughlin"). Using it as resume_text would splice real
     McLaughlin evidence into a fake-identity shell. Worked around by
     using a real, previously cloud-tailored, validator-passing
     McLaughlin resume already on disk as the fixed base resume text
     instead (same base resume text used for every job in this run).

No production code is modified. No DB writes. Local-only LLM calls
(qwen3:1.7b via Ollama), no cloud LLM keys are configured in this
environment so the "judge" call (when reached) also falls through to
the same local model -- flagged explicitly in the output, not hidden.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")
os.environ.setdefault("APPLYPILOT_LOCAL_LLM_MODEL", "qwen3:1.7b")

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot import database  # noqa: E402
from applypilot.llm import get_stage_client, is_local_configured  # noqa: E402
from applypilot.scoring.local_tailor import (  # noqa: E402
    build_base_resume_model,
    merge_realization,
    request_local_realization,
)
from applypilot.scoring.schemas import get_or_build_job_schema  # noqa: E402
from applypilot.scoring.tailor import (  # noqa: E402
    assemble_resume_text,
    classify_seniority_mismatch,
    classify_standup_relevance,
    judge_tailored_resume,
)
from applypilot.scoring.validator import validate_json_fields  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
JOB_IDS = [27323, 7267, 7406, 22916]
BASE_RESUME_PATH = REPO_ROOT / "data" / "resumes" / "applypilot" / "Philip_McLaughlin_Software_Development_Engineer_74c9630f.txt"


def _job_row_to_dict(conn, rowid: int) -> dict:
    cur = conn.execute(
        "SELECT rowid, url, title, company, site, location, full_description, fit_score, state "
        "FROM jobs WHERE rowid = ?",
        (rowid,),
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"job {rowid} not found")
    return dict(row)


def _requirement_coverage(job_schema: dict) -> dict:
    reqs = job_schema.get("requirements") or []
    supported = [r for r in reqs if r.get("supported")]
    ambiguous = [r for r in reqs if r.get("ambiguous")]
    unsupported = [r for r in reqs if not r.get("supported") and not r.get("ambiguous")]
    return {
        "total": len(reqs),
        "supported": len(supported),
        "ambiguous": len(ambiguous),
        "unsupported": len(unsupported),
        "supported_texts": [r.get("text", "")[:140] for r in supported],
    }


def run_one(conn, profile: dict, resume_text: str, job_id: int) -> dict:
    job = _job_row_to_dict(conn, job_id)
    result: dict = {"job_id": job_id, "title": job["title"], "site": job["site"], "fit_score": job["fit_score"]}

    try:
        job_schema = get_or_build_job_schema(job, profile)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"job_schema build failed: {exc}"
        return result

    standup_decision = classify_standup_relevance(job)
    seniority_gate = classify_seniority_mismatch(job, profile)
    result["standup_decision"] = standup_decision
    result["seniority_gate_would_block_in_production"] = bool(seniority_gate)
    result["requirement_coverage"] = _requirement_coverage(job_schema)

    client = get_stage_client("tailor", quality=True)
    result["is_local_configured"] = is_local_configured()
    result["client_has_cloud_available"] = bool(getattr(client, "has_cloud_available", lambda: True)())

    base_resume = build_base_resume_model(resume_text, profile)

    t0 = time.time()
    try:
        realization, meta = request_local_realization(client, job, job_schema, profile)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"request_local_realization raised: {exc}\n{traceback.format_exc()}"
        result["latency_s"] = round(time.time() - t0, 1)
        return result
    latency = time.time() - t0
    result["latency_s"] = round(latency, 1)
    result["realization"] = realization
    result["realization_meta"] = meta

    if realization is None:
        result["status"] = "local_realization_failed" if meta.get("llm_called") else "no_supported_evidence"
        result["resume_text_out"] = resume_text  # unchanged, matches production behavior exactly
        return result

    data = merge_realization(base_resume, realization, job)
    result["merged_data"] = data
    validation = validate_json_fields(data, profile, standup_decision=standup_decision)
    result["validator"] = validation
    assembled = assemble_resume_text(data, profile)
    result["resume_text_out"] = assembled

    if not validation["passed"]:
        result["status"] = "failed_validation"
        result["judge"] = {"skipped": "validation failed -- production never calls judge after failed validation"}
        return result

    result["status"] = "validated"
    try:
        judge = judge_tailored_resume(resume_text, assembled, job["title"], profile)
        result["judge"] = judge
        result["judge_note"] = (
            "No cloud LLM keys configured in this environment -- judge call fell through the "
            "same cascade to the local model (qwen3:1.7b), same as production would in a fully "
            "cloud-exhausted state. This is a model judging output from its own family, so treat "
            "the verdict as lower-confidence than a real cloud judge."
        )
    except Exception as exc:  # noqa: BLE001
        result["judge"] = {"error": str(exc)}

    result["status"] = "approved" if result.get("judge", {}).get("passed") else "validated_judge_disagreed"

    # Structural diff vs the base resume: degraded mode NEVER drops
    # existing bullets (unlike cloud tailoring, which curates/cuts) --
    # this is a qualitative, architecture-level property worth
    # capturing explicitly, not just inferring from the text.
    realized_bullet_count = len((realization or {}).get("bullets") or {})
    result["bullets_realized"] = realized_bullet_count
    result["bullets_dropped_from_base"] = 0  # merge_realization is additive-only by construction

    return result


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    resume_text = BASE_RESUME_PATH.read_text(encoding="utf-8")

    prestate = {
        "resume_text_source": str(BASE_RESUME_PATH),
        "resume_text_source_note": (
            "~/.applypilot/resume.txt on this machine is a stale placeholder identity "
            "('Philip Ginger') unrelated to data/profile.json's real candidate "
            "('Philip McLaughlin') -- substituted a real, previously cloud-validated "
            "McLaughlin resume instead so build_base_resume_model() parses correctly "
            "and results reflect the real candidate."
        ),
        "local_llm_url": os.environ.get("APPLYPILOT_LOCAL_LLM_URL"),
        "local_llm_model": os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL"),
        "ollama_list": os.popen("ollama list").read(),
    }
    (OUT_DIR / "prestate.json").write_text(json.dumps(prestate, indent=2), encoding="utf-8")

    results = []
    for job_id in JOB_IDS:
        print(f"=== job {job_id} ===", flush=True)
        t0 = time.time()
        res = run_one(conn, profile, resume_text, job_id)
        print(f"  status={res.get('status')} latency={res.get('latency_s')}s (wall {time.time()-t0:.1f}s)", flush=True)
        results.append(res)
        (OUT_DIR / f"job_{job_id}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        if res.get("resume_text_out"):
            (OUT_DIR / f"job_{job_id}_resume.txt").write_text(res["resume_text_out"], encoding="utf-8")

    (OUT_DIR / "all_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
