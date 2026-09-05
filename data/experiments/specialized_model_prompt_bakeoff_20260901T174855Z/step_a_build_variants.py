"""STEP A (production venv): build the 3 prompt-variant configs (native,
bullet-rewrite, plan) for the primary jobs (7267, 22916) and a Config-1-only
prompt for the spot-check job (27323). Reuses production evidence/schema
code (get_or_build_job_schema, build_base_resume_model, _fuzzy_evidence_match)
so requirement/evidence selection is identical to what ApplyPilot's real
deterministic layer would select -- only the PROMPT SHAPE handed to the
model differs from production's own prompt.

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
from applypilot.scoring.local_tailor import _fuzzy_evidence_match, build_base_resume_model  # noqa: E402
from applypilot.scoring.schemas import get_or_build_job_schema  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
BASE_RESUME_PATH = REPO_ROOT / "data" / "resumes" / "applypilot" / "Philip_McLaughlin_Software_Development_Engineer_74c9630f.txt"

PRIMARY_JOBS = [7267, 22916]
SPOTCHECK_JOBS = [27323]

NATIVE_SYSTEM = (
    "You are a career assistant. Rewrite resumes to better fit a target job. "
    "Never invent facts, employers, titles, dates, or skills that are not "
    "already present in the resume. If something isn't supported by the "
    "resume, leave it out."
)

REWRITE_SYSTEM = (
    "You rewrite individual resume bullet points to better match a specific "
    "job requirement. Rewrite ONLY using facts already present in the "
    "bullet(s) given to you. Do not add any new fact, tool, employer, "
    "metric, or outcome that is not already stated."
)

PLAN_SYSTEM = (
    "You are a career assistant tailoring a resume to a job. You are given "
    "the job's requirements and, for each one, the ONLY resume evidence you "
    "are allowed to draw on. Never invent facts, employers, titles, dates, "
    "metrics, or skills beyond what's given. If a requirement has no "
    "evidence listed, do not address it."
)


def get_job(conn, job_id: int) -> dict:
    row = conn.execute(
        "SELECT rowid, url, title, company, site, location, full_description, fit_score, state "
        "FROM jobs WHERE rowid = ?",
        (job_id,),
    ).fetchone()
    return dict(row)


def build_native_prompt(job: dict, resume_text: str) -> dict:
    desc = (job.get("full_description") or "")[:3000]
    user = (
        f"JOB DESCRIPTION:\n{job['title']}\n{desc}\n\n"
        f"RESUME:\n{resume_text}\n\n"
        "TASK: Tailor this resume to the job above. Do not invent any information."
    )
    return {"system": NATIVE_SYSTEM, "user": user}


def find_bullets_for_evidence(base_resume: dict, evidence_name: str) -> list[str]:
    for section in ("experience", "projects"):
        for entry in base_resume.get(section) or []:
            header = entry.get("title") or ""
            if _fuzzy_evidence_match(header, evidence_name):
                return list(entry.get("bullets") or [])
    return []


def build_rewrite_prompt(job_schema: dict, base_resume: dict) -> dict | None:
    """Config 4: pick ONE supported requirement + its matching existing bullets."""
    for r in job_schema.get("requirements") or []:
        if not r.get("supported"):
            continue
        evidence_names = r.get("resume_evidence") or []
        bullets: list[str] = []
        for name in evidence_names:
            bullets.extend(find_bullets_for_evidence(base_resume, name))
        if bullets:
            bullet_block = "\n".join(f"- {b}" for b in bullets[:3])
            user = (
                f"JOB REQUIREMENT: {r.get('requirement', '')}\n\n"
                f"EXISTING RESUME BULLET(S) (this is the only information you may draw on):\n{bullet_block}\n\n"
                "TASK: Rewrite the bullet(s) above to more directly speak to the job "
                "requirement, using only what's already stated. Output only the "
                "rewritten bullet(s), one per line."
            )
            return {"system": REWRITE_SYSTEM, "user": user, "requirement_used": r.get("requirement", ""), "evidence_used": evidence_names, "original_bullets": bullets}
    return None


def build_plan_prompt(job: dict, job_schema: dict, resume_text: str) -> dict | None:
    """Config 'plan': ALL supported requirements, plain-language evidence blocks, whole EXPERIENCE section rewrite."""
    blocks = []
    any_supported = False
    for r in job_schema.get("requirements") or []:
        if not r.get("supported"):
            continue
        any_supported = True
        evidence_lines = []
        for prov in r.get("provenance") or []:
            text = prov.get("text", "") if isinstance(prov, dict) else str(prov)
            src = prov.get("source_type", "evidence") if isinstance(prov, dict) else "evidence"
            evidence_lines.append(f"- {src}: {text[:200]}")
        block = (
            f"REQUIREMENT: {r.get('requirement', '')}\n"
            f"SUPPORTED BY:\n" + "\n".join(evidence_lines) + "\n"
            f"TAILORING DIRECTION: emphasize this requirement using only the evidence above."
        )
        blocks.append(block)
    if not any_supported:
        return None

    # Extract just the EXPERIENCE section from the base resume text for a smaller, focused prompt.
    exp_start = resume_text.find("EXPERIENCE")
    exp_end = resume_text.find("PROJECTS")
    experience_section = resume_text[exp_start:exp_end].strip() if exp_start != -1 and exp_end != -1 else resume_text

    user = (
        f"JOB: {job['title']}\n\n" + "\n\n".join(blocks) + "\n\n"
        f"EXISTING EXPERIENCE SECTION (for reference and reuse of anything not touched above):\n{experience_section}\n\n"
        "TASK: Rewrite the EXPERIENCE section to better address the requirements "
        "above, using only the evidence given. Keep every employer/date/title "
        "exactly as in the original. Output only the rewritten EXPERIENCE section."
    )
    return {"system": PLAN_SYSTEM, "user": user}


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    resume_text = BASE_RESUME_PATH.read_text(encoding="utf-8")
    base_resume = build_base_resume_model(resume_text, profile)

    out = {"base_resume_source": str(BASE_RESUME_PATH), "jobs": {}}

    for job_id in PRIMARY_JOBS + SPOTCHECK_JOBS:
        job = get_job(conn, job_id)
        job_schema = get_or_build_job_schema(job, profile)
        entry = {
            "job": job,
            "requirement_coverage": {
                "total": len(job_schema.get("requirements") or []),
                "supported": sum(1 for r in job_schema.get("requirements") or [] if r.get("supported")),
            },
            "configs": {},
        }
        entry["configs"]["native"] = build_native_prompt(job, resume_text)
        if job_id in PRIMARY_JOBS:
            entry["configs"]["rewrite"] = build_rewrite_prompt(job_schema, base_resume)
            entry["configs"]["plan"] = build_plan_prompt(job, job_schema, resume_text)
        out["jobs"][str(job_id)] = entry
        print(f"job {job_id}: native={'ok'}, rewrite={'ok' if entry['configs'].get('rewrite') else 'n/a'}, plan={'ok' if entry['configs'].get('plan') else 'n/a'}")

    (OUT_DIR / "shared_variants.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
