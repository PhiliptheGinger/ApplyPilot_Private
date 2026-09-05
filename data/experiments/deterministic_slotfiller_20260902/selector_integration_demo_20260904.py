"""Full end-to-end demo of the wired-in selector: schema matching ->
pool-based deterministic realization -> merge -> the SAME assemble_
resume_text renderer the real pipeline uses -- zero LLM calls anywhere in
this path. Run against two real, different DB jobs that both match
Alex Prosperity Group / UST Logistics evidence, to show the resume
actually varies per job and to let a human judge whether the selected
content is a good fit.

sentence_pools is loaded from tonight's already-generated, already-
diversity-filtered experiment output -- NOT from profile.json (that pool
hasn't been human-reviewed/approved yet, and build_pool_realization
deliberately never reads profile.json for this on its own).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.local_tailor import build_base_resume_model, build_pool_realization, merge_realization  # noqa: E402
from applypilot.scoring.resume_router import load_resume_text_for_job  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402
from applypilot.scoring.tailor import assemble_resume_text  # noqa: E402

POOL_PATH = REPO_ROOT / "data/experiments/deterministic_slotfiller_20260902/inventory_expansion_pilot_v2_diversity_results.json"

JOB_URLS = [
    "https://www.linkedin.com/jobs/view/4457669576",  # Field Technician, fit_score=9
    "https://www.linkedin.com/jobs/view/4431716923",  # Early Career Field Service Technician, fit_score=9
]


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()

    pool_data = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    sentence_pools = {pool_data["target"]: pool_data["final_accepted"]}
    print(f"Pool source: {POOL_PATH.name}")
    print(f"Pools available for: {list(sentence_pools.keys())}\n")

    for url in JOB_URLS:
        row = conn.execute("SELECT * FROM jobs WHERE url = ?", (url,)).fetchone()
        if row is None:
            print(f"SKIP (not found): {url}")
            continue
        job = dict(row)

        print("=" * 90)
        print(f"JOB: {job['title']} (fit_score={job['fit_score']})")
        print(f"URL: {url}")
        print("=" * 90)

        schema = build_job_schema_representation(job, profile)
        matched = [
            r for r in (schema.get("requirements") or [])
            if r.get("supported") and "Alex Prosperity Group / UST Logistics" in (r.get("resume_evidence") or [])
        ]
        print(f"\nRequirements matched to Alex Prosperity Group evidence: {len(matched)}")
        for r in matched:
            print(f"  - {r.get('requirement', '')[:110]}")

        resume_text, _ = load_resume_text_for_job(job)
        base = build_base_resume_model(resume_text, profile)
        realization = build_pool_realization(schema, sentence_pools)
        merged = merge_realization(base, realization, job)

        selected = (realization or {}).get("bullets", {}).get("Alex Prosperity Group / UST Logistics")
        print(f"\nSELECTED sentence for this job: {selected!r}")

        rendered = assemble_resume_text(merged, profile)
        print("\n--- Rendered resume (assemble_resume_text, same renderer the real pipeline uses) ---")
        print(rendered)
        print()


if __name__ == "__main__":
    main()
