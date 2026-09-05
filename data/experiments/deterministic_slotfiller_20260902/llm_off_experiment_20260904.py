"""Turn off the tailoring LLM entirely and see what the deterministic
layer alone produces, for a real job, right now -- no speculation, just
run it. 2026-09-04, in direct response to: "let's turn it off and see
what happens."

Uses ONLY the pieces that are actually deterministic today:
  - schemas.build_job_schema_representation(job, profile) -- zero LLM,
    tells us which requirements are SUPPORTED and by which evidence.
  - local_tailor.build_base_resume_model(resume_text, profile) -- zero
    LLM, deterministic parse of the existing resume into a structured
    model.
  - local_tailor.merge_realization(base, realization, job) -- called with
    realization=None, i.e. the LLM step is skipped entirely, not mocked.

This is NOT the selector/assembler (that doesn't exist yet) -- it's the
honest floor: what do you get if you strip the LLM out of the CURRENT
architecture, with nothing built yet to replace its job of choosing/
varying content per posting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.local_tailor import build_base_resume_model, merge_realization  # noqa: E402
from applypilot.scoring.resume_router import load_resume_text_for_job  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
JOB_URL = "https://www.linkedin.com/jobs/view/4457705031"  # Coffee and Tea Equipment Tech, fit_score=10, cleanly bulleted description


def main() -> None:
    conn = database.get_connection()
    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (JOB_URL,)).fetchone()
    if row is None:
        raise SystemExit(f"job not found: {JOB_URL}")
    job = dict(row)
    profile = config.load_profile()

    print(f"JOB: {job['title']} (fit_score={job['fit_score']})")
    print(f"Real LLM-tailored resume already on file at: {job.get('tailored_resume_path')}")

    # --- Step 1: what does the deterministic layer know? (zero LLM) ---
    schema = build_job_schema_representation(job, profile)
    requirements = schema.get("requirements") or []
    supported = [r for r in requirements if r.get("supported")]
    print(f"\n=== DETERMINISTIC LAYER: {len(supported)}/{len(requirements)} requirements supported ===")
    for r in supported:
        print(f"  REQUIREMENT: {r.get('text', '')[:90]}")
        print(f"    evidence: {r.get('resume_evidence')}  frame={r.get('frame')}  "
              f"category_tier={r.get('category_tier')}  cognitive_schema={r.get('cognitive_schema')}")

    # --- Step 2: deterministic base resume model (zero LLM) ---
    resume_text, _ = load_resume_text_for_job(job)
    base = build_base_resume_model(resume_text, profile)

    # --- Step 3: THE LLM IS OFF. realization=None -- no generative call
    # of any kind happens between here and the final merged resume. ---
    merged = merge_realization(base, None, job)

    print("\n=== RESULT: deterministic-only resume (LLM fully off) ===")
    print(json.dumps(merged, indent=2)[:3000])

    (OUT_DIR / "llm_off_experiment_20260904_result.json").write_text(
        json.dumps({"job_title": job["title"], "schema_supported_count": len(supported),
                    "schema_total_requirements": len(requirements), "deterministic_result": merged},
                   indent=2),
        encoding="utf-8",
    )
    print("\nwrote llm_off_experiment_20260904_result.json")
    print("\nHONEST TAKEAWAY: compare this to the real tailored_resume_path above. If they're "
          "identical (or nearly so), that confirms the deterministic layer currently has NO "
          "mechanism to vary content per job -- it can tell you WHAT'S supported, but nothing "
          "exists yet that ACTS on that to select/reorder/emphasize content without an LLM call. "
          "That's the selector/assembler gap, made concrete instead of argued about.")


if __name__ == "__main__":
    main()
