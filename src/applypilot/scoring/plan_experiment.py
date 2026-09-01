"""Read-only Q2 experiment helpers for structured tailoring plans.

This module is intentionally isolated from production runners. It reuses
existing deterministic extraction/ranking/schema machinery to build a
machine-validatable plan artifact for explicitly selected jobs.
"""

from __future__ import annotations

import os
from collections import Counter


def build_structured_plan(job: dict, profile: dict) -> dict:
    """Build a deterministic structured plan for one job.

    No LLM calls, no DB writes, no resume generation.
    """
    from applypilot.scoring.local_tailor import (
        _auto_resolve_requirements,
        _split_requirement_lines,
        rank_profile_evidence,
    )
    from applypilot.scoring.schemas import build_job_schema_representation

    top_n = int(os.environ.get("APPLYPILOT_LOCAL_EVIDENCE_TOPN", "6"))
    ranked = rank_profile_evidence(job, profile, top_n=top_n)
    requirement_lines, dropped_benefits = _split_requirement_lines(job.get("full_description") or "")
    resolved, candidates = _auto_resolve_requirements(requirement_lines, ranked)
    schema_rep = build_job_schema_representation(job, profile)

    evidence_catalog = []
    for idx, entry in enumerate(ranked, start=1):
        item = entry.get("item") or {}
        evidence_catalog.append(
            {
                "id": idx,
                "type": entry.get("type"),
                "name": entry.get("name"),
                "score": int(entry.get("score") or 0),
                "matched_terms": list(entry.get("matched_terms") or []),
                "permitted": item.get("resume_allowed") is not False,
            }
        )

    requirements = []
    schema_requirements = schema_rep.get("requirements") or []
    for ridx, line in enumerate(requirement_lines, start=1):
        schema_entry = schema_requirements[ridx - 1] if ridx - 1 < len(schema_requirements) else {}
        supported = bool(schema_entry.get("supported"))
        ambiguous = bool(schema_entry.get("ambiguous"))
        if supported:
            status = "supported"
            evidence_ids = list(resolved.get(ridx) or [])
        elif ambiguous:
            status = "ambiguous"
            evidence_ids = list(candidates.get(ridx) or [])
        else:
            status = "unsupported"
            evidence_ids = []

        schema_obj = schema_entry.get("schema") or {}
        requirements.append(
            {
                "id": ridx,
                "text": line.get("text", ""),
                "importance": line.get("importance", "unspecified"),
                "status": status,
                "evidence_ids": evidence_ids,
                "directives": {
                    "cognitive_schema": schema_obj.get("cognitive_schema"),
                    "bullet_schema": schema_obj.get("bullet_schema"),
                    "claim_ceiling": schema_entry.get("claim_ceiling"),
                    "agency_ceiling": schema_entry.get("agency_ceiling"),
                    "force_relation": schema_entry.get("force_relation"),
                    "salience_order": list(schema_entry.get("salience_order") or []),
                    "exact_keywords": list(schema_entry.get("exact_keywords") or []),
                    "synonym_concepts": list(schema_entry.get("synonym_concepts") or []),
                    "vary_phrasing": bool(schema_entry.get("vary_phrasing")),
                },
            }
        )

    counts = Counter(r["status"] for r in requirements)
    return {
        "schema_version": "q2.plan.v1",
        "mode": "deterministic",
        "job": {
            "id": job.get("rowid"),
            "url": job.get("url"),
            "title": job.get("title"),
            "company": job.get("company") or "",
            "site": job.get("site") or "",
        },
        "viewpoint": schema_rep.get("viewpoint"),
        "summary_schema": schema_rep.get("summary_schema"),
        "requirement_count": len(requirements),
        "counts": {
            "supported": int(counts.get("supported", 0)),
            "ambiguous": int(counts.get("ambiguous", 0)),
            "unsupported": int(counts.get("unsupported", 0)),
        },
        "dropped_benefit_lines": dropped_benefits,
        "evidence_catalog": evidence_catalog,
        "requirements": requirements,
    }


def validate_structured_plan(plan: dict) -> dict:
    """Validate a structured plan with deterministic checks only."""
    from applypilot.scoring.schemas import AGENCY_TIERS, CLAIM_TIERS

    errors: list[str] = []
    required_top = {
        "schema_version",
        "mode",
        "job",
        "viewpoint",
        "summary_schema",
        "requirement_count",
        "counts",
        "evidence_catalog",
        "requirements",
    }
    missing_top = sorted(required_top - set(plan.keys()))
    if missing_top:
        errors.append(f"Missing top-level fields: {', '.join(missing_top)}")
        return {"passed": False, "errors": errors}

    if plan.get("mode") != "deterministic":
        errors.append("mode must be 'deterministic'")

    evidence_catalog = plan.get("evidence_catalog") or []
    if not isinstance(evidence_catalog, list):
        errors.append("evidence_catalog must be a list")
        return {"passed": False, "errors": errors}

    evidence_by_id: dict[int, dict] = {}
    for e in evidence_catalog:
        if not isinstance(e, dict):
            errors.append("evidence_catalog contains a non-object entry")
            continue
        eid = e.get("id")
        if not isinstance(eid, int):
            errors.append("each evidence entry must have integer id")
            continue
        if eid in evidence_by_id:
            errors.append(f"duplicate evidence id: {eid}")
            continue
        evidence_by_id[eid] = e

    requirements = plan.get("requirements") or []
    if not isinstance(requirements, list):
        errors.append("requirements must be a list")
        return {"passed": False, "errors": errors}

    if plan.get("requirement_count") != len(requirements):
        errors.append("requirement_count does not match requirements length")

    expected_counts = Counter(r.get("status") for r in requirements if isinstance(r, dict))
    counts = plan.get("counts") or {}
    for status in ("supported", "ambiguous", "unsupported"):
        if int(counts.get(status, 0)) != int(expected_counts.get(status, 0)):
            errors.append(f"counts.{status} does not match computed requirement statuses")

    allowed_directive_keys = {
        "cognitive_schema",
        "bullet_schema",
        "claim_ceiling",
        "agency_ceiling",
        "force_relation",
        "salience_order",
        "exact_keywords",
        "synonym_concepts",
        "vary_phrasing",
    }
    for r in requirements:
        if not isinstance(r, dict):
            errors.append("requirements contains a non-object entry")
            continue
        rid = r.get("id")
        text = r.get("text")
        status = r.get("status")
        evidence_ids = r.get("evidence_ids")
        directives = r.get("directives")

        if not isinstance(rid, int):
            errors.append("requirement.id must be integer")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"requirement {rid}: text must be non-empty string")
        if status not in ("supported", "ambiguous", "unsupported"):
            errors.append(f"requirement {rid}: invalid status '{status}'")
        if not isinstance(evidence_ids, list) or any(not isinstance(v, int) for v in evidence_ids):
            errors.append(f"requirement {rid}: evidence_ids must be a list[int]")
            evidence_ids = []

        if status == "supported" and not evidence_ids:
            errors.append(f"requirement {rid}: supported requirements must reference evidence_ids")
        if status == "unsupported" and evidence_ids:
            errors.append(f"requirement {rid}: unsupported requirements must not reference evidence_ids")

        for eid in evidence_ids:
            e = evidence_by_id.get(eid)
            if e is None:
                errors.append(f"requirement {rid}: references unknown evidence id {eid}")
                continue
            if e.get("permitted") is False:
                errors.append(f"requirement {rid}: references non-permitted evidence id {eid}")

        if not isinstance(directives, dict):
            errors.append(f"requirement {rid}: directives must be an object")
            continue
        unknown_keys = sorted(set(directives.keys()) - allowed_directive_keys)
        if unknown_keys:
            errors.append(f"requirement {rid}: unknown directive keys: {', '.join(unknown_keys)}")

        claim_ceiling = directives.get("claim_ceiling")
        if claim_ceiling is not None and claim_ceiling not in CLAIM_TIERS:
            errors.append(f"requirement {rid}: invalid claim_ceiling '{claim_ceiling}'")

        agency_ceiling = directives.get("agency_ceiling")
        if agency_ceiling is not None and agency_ceiling not in AGENCY_TIERS:
            errors.append(f"requirement {rid}: invalid agency_ceiling '{agency_ceiling}'")

        for list_key in ("salience_order", "exact_keywords", "synonym_concepts"):
            value = directives.get(list_key)
            if value is None:
                continue
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                errors.append(f"requirement {rid}: directives.{list_key} must be list[str]")

        if status == "unsupported" and (directives.get("cognitive_schema") or directives.get("bullet_schema")):
            errors.append(f"requirement {rid}: unsupported requirement cannot declare schemas")

    return {"passed": not errors, "errors": errors}
