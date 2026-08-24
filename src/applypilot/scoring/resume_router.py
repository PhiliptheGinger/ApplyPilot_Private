"""Resume variant routing by role communication intensity.

Supports two optional user-provided resume variants:
- technical/non-communication resume
- communication-forward resume

If variants are missing, falls back to RESUME_PATH.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from applypilot.config import APP_DIR, PROJECT_PROFILE_PATH, RESUME_PATH, load_profile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_RESUMES_DIR = PROJECT_ROOT / "data" / "resumes"

# Optional variant files users can provide in-repo data dir or ~/.applypilot
PROJECT_TECHNICAL_RESUME_PATH = PROJECT_RESUMES_DIR / "resume_technical.txt"
PROJECT_COMMUNICATION_RESUME_PATH = PROJECT_RESUMES_DIR / "resume_communication.txt"
DEFAULT_TECHNICAL_RESUME_PATH = APP_DIR / "resume_technical.txt"
DEFAULT_COMMUNICATION_RESUME_PATH = APP_DIR / "resume_communication.txt"

COMMUNICATION_ROLE_KEYWORDS: tuple[str, ...] = (
    "help desk",
    "helpdesk",
    "service desk",
    "technical support",
    "customer support",
    "customer success",
    "call center",
    "phone support",
    "sales",
    "account executive",
    "sdr",
    "bdr",
    "inside sales",
    "client-facing",
    "customer-facing",
    "stakeholder-facing",
    "public speaking",
    "presentation",
    "presentations",
)


def is_communication_role(job: dict) -> bool:
    """Return True when the role likely requires high verbal communication."""
    title = (job.get("title") or "").lower()
    desc = (job.get("full_description") or "").lower()
    haystack = f"{title}\n{desc}"
    return any(re.search(rf"\b{re.escape(k)}\b", haystack) for k in COMMUNICATION_ROLE_KEYWORDS)


def _env_path(name: str) -> Path | None:
    v = (os.environ.get(name) or "").strip()
    if not v:
        return None
    return Path(v)


def choose_resume_path_for_job(job: dict) -> Path:
    """Choose resume file path for a job.

    Priority:
      1) communication role -> APPLYPILOT_RESUME_COMM_PATH -> project data -> APP_DIR default
      2) technical role     -> APPLYPILOT_RESUME_TECH_PATH -> project data -> APP_DIR default
      3) fallback           -> RESUME_PATH
    """
    if is_communication_role(job):
        for p in (
            _env_path("APPLYPILOT_RESUME_COMM_PATH"),
            PROJECT_COMMUNICATION_RESUME_PATH,
            DEFAULT_COMMUNICATION_RESUME_PATH,
        ):
            if p and p.exists():
                return p
    else:
        for p in (
            _env_path("APPLYPILOT_RESUME_TECH_PATH"),
            PROJECT_TECHNICAL_RESUME_PATH,
            DEFAULT_TECHNICAL_RESUME_PATH,
        ):
            if p and p.exists():
                return p
    return RESUME_PATH


def load_resume_text_for_job(job: dict) -> tuple[str, Path]:
    """Load canonical profile reference, with legacy resume fallback.

    The job is retained for backward-compatible callers, but it no longer
    selects the candidate's primary factual source. Relevance selection is
    performed by tailoring against the canonical profile.
    """
    if PROJECT_PROFILE_PATH.exists():
        return _render_profile_reference(load_profile()), PROJECT_PROFILE_PATH

    path = choose_resume_path_for_job(job)
    return path.read_text(encoding="utf-8"), path


def _render_profile_reference(profile: dict) -> str:
    """Render a compact factual reference from the canonical profile."""
    lines: list[str] = ["CANONICAL PROFILE REFERENCE"]
    for key in (
        "education",
        "experience_inventory",
        "historical_experience_inventory",
        "qualifications",
        "project_inventory",
    ):
        for item in profile.get(key, []):
            if not isinstance(item, dict) or item.get("private"):
                continue
            name = item.get("name") or item.get("institution") or item.get("official_degree")
            if name:
                lines.append(f"{key}: {name}")
            for field in ("official_degree", "role_title", "responsibilities", "factual_concepts", "evidence"):
                value = item.get(field)
                if value:
                    values = value if isinstance(value, list) else [value]
                    lines.extend(f"- {value}" for value in values)
    return "\n".join(lines)
