"""THROWAWAY test -- does the EXISTING Qwen3 arbitration path correctly
reject the semantic false positive found in the semantic-retrieval
experiment (You Power You vs. CompTIA A+ for a desktop-hardware
requirement)?

Reuses the real, unmodified local_tailor.py functions that build/parse
the arbitration prompt (_PLAN_SYSTEM, format_evidence_for_prompt,
_format_requirement_lines, _parse_plan, _merge_model_matches_with_resolved,
validate_local_plan) -- no new prompt is invented. The only thing
hand-constructed here is the two-item candidate pool itself, standing in
for what a semantic-retrieval admission step would eventually produce
(rank_profile_evidence naturally never puts these two together for this
job, since You Power You has zero literal overlap with job 19447 at all --
that's the exact gap being tested).

Not part of the production pipeline. Does not modify any production file,
the database, or the semantic architecture. Safe to delete after review.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot.scoring.local_tailor import (  # noqa: E402
    _PLAN_SYSTEM,
    _format_requirement_lines,
    _merge_model_matches_with_resolved,
    _parse_plan,
    format_evidence_for_prompt,
    validate_local_plan,
)

DB_PATH = Path.home() / ".applypilot" / "applypilot.db"
PROFILE_PATH = REPO_ROOT / "data" / "profile.json"
OLLAMA_URL = "http://localhost:11434"
MODEL = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", "qwen3:1.7b")

JOB_ID = 19447
REQUIREMENT_TEXT = "Install, configure, maintain, and support desktop hardware, software, and related technology solutions."


def main() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    conn = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT rowid, title, site FROM jobs WHERE rowid = ?", (JOB_ID,)).fetchone()
    conn.close()
    print(f"Job {JOB_ID}: {row['title']} @ {row['site']}")

    comptia = next(i for i in profile["certifications"] if i["name"] == "CompTIA A+")
    ypy = next(i for i in profile["project_inventory"] if i["name"] == "You Power You")

    # Real evidence items, shaped exactly like rank_profile_evidence's
    # output -- score/matched_terms are cosmetic here (format_evidence_for_prompt
    # only reads item/type/name), so they're left empty rather than faked.
    ranked_evidence = [
        {"type": "certification", "name": "CompTIA A+", "score": 0, "matched_terms": [], "item": comptia},
        {"type": "project", "name": "You Power You", "score": 0, "matched_terms": [], "item": ypy},
    ]
    requirement_lines = [{"text": REQUIREMENT_TEXT, "importance": "unspecified"}]
    candidates = {1: [1, 2]}  # both tied as top-tier -- the exact ambiguous shape arbitration is built for
    ambiguous_ids = {1}

    job_text = f"TITLE: {row['title']}\nCOMPANY: unknown"
    evidence_text = format_evidence_for_prompt(ranked_evidence)
    requirements_text = _format_requirement_lines(requirement_lines, candidates=candidates, only_ids=ambiguous_ids)
    user_msg = (
        f"JOB:\n{job_text}\n\n"
        f"REQUIREMENTS:\n{requirements_text}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "Return the JSON match list now:"
    )

    print("\n" + "=" * 100)
    print("EXACT PROMPT SENT (system + user) -- unmodified _PLAN_SYSTEM / format_evidence_for_prompt / _format_requirement_lines")
    print("=" * 100)
    print("--- SYSTEM ---")
    print(_PLAN_SYSTEM)
    print("\n--- USER ---")
    print(user_msg)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "think": False,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 300},
    }

    print("\n" + "=" * 100)
    print(f"CALLING {OLLAMA_URL}/api/chat  model={MODEL}")
    print("=" * 100)
    resp = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    raw_text = (data.get("message") or {}).get("content", "")
    print("RAW MODEL RESPONSE (message.content):")
    print(repr(raw_text))

    raw_plan = _parse_plan(raw_text)
    print("\nPARSED (via _parse_plan):", raw_plan)

    combined, warnings = _merge_model_matches_with_resolved(raw_plan, resolved={}, candidates=candidates, ambiguous_ids=ambiguous_ids)
    print("MERGED (via _merge_model_matches_with_resolved):", combined)
    print("WARNINGS:", warnings)

    sanitized = validate_local_plan(combined, requirement_lines, ranked_evidence)
    print("\n" + "=" * 100)
    print("FINAL VALIDATED PLAN (via validate_local_plan -- same function get_local_tailoring_plan returns)")
    print("=" * 100)
    print(json.dumps(sanitized, indent=2, default=str))

    picked_ids = combined["matches"][0]["e"] if combined["matches"] else []
    picked_names = [ranked_evidence[i - 1]["name"] for i in picked_ids]
    print("\n" + "=" * 100)
    print("DECISION SUMMARY")
    print("=" * 100)
    print(f"Selected evidence: {picked_names or '(none)'}")
    print(f"CompTIA A+ selected: {'CompTIA A+' in picked_names}")
    print(f"You Power You selected: {'You Power You' in picked_names}")


if __name__ == "__main__":
    main()
