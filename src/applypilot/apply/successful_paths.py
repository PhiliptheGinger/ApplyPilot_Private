"""Per-ATS memoization of the tool-call sequence for a successful apply.

When the agent finishes a job with RESULT:APPLIED (or inferred), we record
the sequence of MCP tool calls (browser_navigate, browser_click,
browser_fill_form, …) to a JSON file keyed by the ATS slug
(``greenhouse``, ``workday``, ``ashby``, …). The next time we apply on
that ATS, ``build_prompt`` looks up the file and prepends a "PRIOR
SUCCESSFUL PATH ON THIS ATS" hint to the agent prompt.

Why: Sonnet/Haiku spend non-trivial tokens figuring out a Greenhouse form's
shape on the first encounter (where to click "Apply", how to handle the
phone-country dropdown, the auto-detected disability/veteran questions,
where the resume Attach button is). A prior path that worked end-to-end
collapses that exploration into a few lines of context.

Storage: ``~/.applypilot/successful_paths/{ats_slug}.json``. One file per
ATS, latest-wins (overwrite on each success). Format::

    {
      "ats_slug":   "greenhouse",
      "captured_at": "2026-04-26T11:09:14+00:00",
      "job_url":     "https://job-boards.greenhouse.io/databricks/jobs/...",
      "duration_ms": 400_000,
      "steps": [
        {"tool": "browser_navigate", "summary": "browser_navigate https://..."},
        {"tool": "browser_snapshot", "summary": "browser_snapshot"},
        {"tool": "browser_click",    "summary": "browser_click Apply button"},
        ...
      ]
    }
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from applypilot import config

logger = logging.getLogger(__name__)

# Steps come out of the agent's stream-json with one entry per tool_use
# block. Real applies emit ~40-80 of these; we cap at the last 60 so the
# prompt-side hint stays compact (most of the value is in the form-fill
# phase, which is the tail).
MAX_STEPS = 60

PATHS_DIR = config.APP_DIR / "successful_paths"


def save_path(ats_slug: str, steps: list[dict],
              job_url: str | None = None,
              duration_ms: int | None = None) -> Path | None:
    """Persist a successful tool-call sequence for ``ats_slug``.

    Returns the file path written, or None if the slug was empty / no
    steps were recorded / write failed.
    """
    if not ats_slug or not steps:
        return None
    try:
        PATHS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PATHS_DIR / f"{ats_slug}.json"
        payload = {
            "ats_slug":    ats_slug,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "job_url":     job_url,
            "duration_ms": duration_ms,
            "steps":       steps[-MAX_STEPS:],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved successful path for %s (%d steps)",
                    ats_slug, len(payload["steps"]))
        return out_path
    except Exception:
        logger.debug("Could not save successful path", exc_info=True)
        return None


def load_path(ats_slug: str) -> dict | None:
    """Read the cached successful path for ``ats_slug``, or return None."""
    if not ats_slug:
        return None
    path = PATHS_DIR / f"{ats_slug}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Could not load successful path for %s", ats_slug, exc_info=True)
        return None


def format_path_for_prompt(payload: dict | None) -> str | None:
    """Render the saved path as a prompt section, or return None.

    Output is hint-shaped — "this is what worked LAST time, may need
    adjustment" — so the agent doesn't blindly replay it on a different
    employer's form.
    """
    if not payload:
        return None
    steps = payload.get("steps") or []
    if not steps:
        return None
    captured = (payload.get("captured_at") or "")[:19]
    dur_s = (payload.get("duration_ms") or 0) // 1000
    lines = [
        f"== PRIOR SUCCESSFUL PATH ({payload.get('ats_slug')}) ==",
        f"Captured {captured}, completed in {dur_s}s. This is the sequence of",
        "tool calls that worked last time on this ATS — use it as a guide,",
        "not a script. The form fields, screening questions, or order may",
        "differ for this employer. Verify each step against the actual page.",
        "",
    ]
    for i, step in enumerate(steps, 1):
        summary = step.get("summary") or step.get("tool") or "?"
        lines.append(f"  {i:2d}. {summary}")
    return "\n".join(lines)
