"""STEP A (runs in the PRODUCTION venv): build the exact same realization
prompts used by the qwen3 corrected bake-off (local_model_bakeoff_v3),
for the SAME 4 real jobs, SAME base resume, SAME profile/evidence catalog.
Dumps (system, user) prompt text per job so STEP B (isolated venv, no
DB/production deps) can feed them to the specialized models without
needing the full ApplyPilot dependency tree installed there.

No production code modified. No DB writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config  # noqa: E402
from applypilot import database  # noqa: E402
from applypilot.scoring.local_tailor import _build_realization_prompt  # noqa: E402
from applypilot.scoring.schemas import get_or_build_job_schema  # noqa: E402
from applypilot.scoring.tailor import classify_seniority_mismatch, classify_standup_relevance  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
JOB_IDS = [27323, 7267, 7406, 22916]
BASE_RESUME_PATH = REPO_ROOT / "data" / "resumes" / "applypilot" / "Philip_McLaughlin_Software_Development_Engineer_74c9630f.txt"


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    resume_text = BASE_RESUME_PATH.read_text(encoding="utf-8")

    out = {"base_resume_source": str(BASE_RESUME_PATH), "jobs": {}}
    for job_id in JOB_IDS:
        row = conn.execute(
            "SELECT rowid, url, title, company, site, location, full_description, fit_score, state "
            "FROM jobs WHERE rowid = ?",
            (job_id,),
        ).fetchone()
        job = dict(row)
        job_schema = get_or_build_job_schema(job, profile)
        prompt = _build_realization_prompt(job_schema)
        out["jobs"][str(job_id)] = {
            "job": job,
            "standup_decision": classify_standup_relevance(job),
            "seniority_gate_would_block_in_production": bool(classify_seniority_mismatch(job, profile)),
            "job_schema": job_schema,
            "prompt": None if prompt is None else {"system": prompt[0], "user": prompt[1]},
        }
        print(f"job {job_id}: prompt={'built' if prompt else 'SKIPPED (no supported requirements)'}")

    (OUT_DIR / "shared_prompts.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "shared_resume_text.txt").write_text(resume_text, encoding="utf-8")
    (OUT_DIR / "shared_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
