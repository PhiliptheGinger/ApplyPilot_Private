"""Fact-subset approval cache for tailored resumes.

A resume can be auto-approved when its extracted fact set is a subset of
facts already approved in previous resumes. This supports efficient reuse
without requiring identical wording.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from applypilot.config import APP_DIR

UTC = timezone.utc
_FACT_LOCK = threading.Lock()
_FACT_CACHE_PATH = APP_DIR / "approved_resume_facts.json"


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def _split_values(raw: str) -> list[str]:
    parts = re.split(r"[,;|]", raw)
    return [p.strip() for p in parts if p.strip()]


def _extract_metrics(text: str) -> set[str]:
    metrics = set()
    for m in re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)", text):
        metrics.add(f"metric:{m}")
    return metrics


def extract_facts_from_resume_json(data: dict, profile: dict) -> set[str]:
    """Extract stable factual atoms from a structured resume JSON payload.

    Excludes stylistic narrative text (summary/title phrasing) and focuses on
    verifiable entities and evidence markers.
    """
    facts: set[str] = set()

    # Allowed skills from profile are used as stable anchors.
    allowed_skills: set[str] = set()
    boundary = profile.get("skills_boundary", {})
    for values in boundary.values():
        if isinstance(values, list):
            allowed_skills.update(_normalize(v) for v in values if _normalize(v))

    skills = data.get("skills")
    if isinstance(skills, dict):
        for _, raw in skills.items():
            for value in _split_values(str(raw)):
                nv = _normalize(value)
                if not nv:
                    continue
                if nv in allowed_skills:
                    facts.add(f"skill:{nv}")

    for section in ("experience", "projects"):
        entries = data.get(section) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            header = _normalize(entry.get("header"))
            subtitle = _normalize(entry.get("subtitle"))
            if header:
                facts.add(f"{section}:header:{header}")
            if subtitle:
                facts.add(f"{section}:subtitle:{subtitle}")
            for bullet in entry.get("bullets", []) or []:
                nb = _normalize(str(bullet))
                if not nb:
                    continue
                facts |= _extract_metrics(nb)

    education = data.get("education") or []
    if isinstance(education, list):
        for row in education:
            if isinstance(row, dict):
                inst = _normalize(row.get("institution"))
                degree = _normalize(row.get("degree"))
                dates = _normalize(row.get("dates"))
                if inst:
                    facts.add(f"edu:institution:{inst}")
                if degree:
                    facts.add(f"edu:degree:{degree}")
                if dates:
                    facts.add(f"edu:dates:{dates}")
            else:
                text = _normalize(str(row))
                if text:
                    facts.add(f"edu:text:{text}")

    return facts


def _read_cache() -> dict:
    if not _FACT_CACHE_PATH.exists():
        return {"entries": []}
    try:
        return json.loads(_FACT_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": []}


def _write_cache(payload: dict) -> None:
    _FACT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FACT_CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def approved_fact_union() -> set[str]:
    with _FACT_LOCK:
        payload = _read_cache()
    union: set[str] = set()
    for entry in payload.get("entries", []):
        facts = entry.get("facts", []) if isinstance(entry, dict) else []
        for fact in facts:
            if isinstance(fact, str):
                union.add(fact)
    return union


def is_auto_approvable(candidate_facts: set[str]) -> bool:
    """Return True when all candidate facts are already approved."""
    if not candidate_facts:
        return False
    union = approved_fact_union()
    if not union:
        return False
    return candidate_facts.issubset(union)


def record_approved_facts(facts: set[str], *, source: str) -> None:
    if not facts:
        return

    with _FACT_LOCK:
        payload = _read_cache()
        entries = payload.setdefault("entries", [])
        entries.append(
            {
                "approved_at": datetime.now(UTC).isoformat(),
                "source": source,
                "facts": sorted(facts),
            }
        )
        # Keep file bounded while preserving recent approvals.
        if len(entries) > 500:
            payload["entries"] = entries[-500:]
        _write_cache(payload)
