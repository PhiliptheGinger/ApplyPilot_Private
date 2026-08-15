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

from applypilot.config import APP_DIR, RESUME_PATH

# Optional variant files users can provide in ~/.applypilot (or APPLYPILOT_DIR)
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
      1) communication role -> APPLYPILOT_RESUME_COMM_PATH -> default comm file
      2) technical role     -> APPLYPILOT_RESUME_TECH_PATH -> default tech file
      3) fallback           -> RESUME_PATH
    """
    if is_communication_role(job):
        for p in (_env_path("APPLYPILOT_RESUME_COMM_PATH"), DEFAULT_COMMUNICATION_RESUME_PATH):
            if p and p.exists():
                return p
    else:
        for p in (_env_path("APPLYPILOT_RESUME_TECH_PATH"), DEFAULT_TECHNICAL_RESUME_PATH):
            if p and p.exists():
                return p
    return RESUME_PATH


def load_resume_text_for_job(job: dict) -> tuple[str, Path]:
    """Load selected resume text and return (text, source_path)."""
    path = choose_resume_path_for_job(job)
    return path.read_text(encoding="utf-8"), path
