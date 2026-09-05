"""Old (jargon-labeled) vs new (plain-language + few-shot) degraded-mode
realization prompt, on real jobs, with the actual local model (qwen3:1.7b)
production currently falls back to. Old prompt logic hand-copied from the
pre-redesign version (data/experiments/local_prompt_redesign_20260902/
old_local_tailor.py) rather than dynamically imported, to avoid any risk
of import side effects from the rest of that large module.

No production code modified by this script. No DB writes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import (  # noqa: E402
    BULLET_SCHEMAS,
    check_agency_strength,
    check_causal_claim,
    check_claim_strength,
    check_metric_fabrication,
)
from applypilot.scoring.local_tailor import (  # noqa: E402
    _build_realization_prompt as new_build_realization_prompt,
    _parse_plan,
)
from applypilot.scoring.schemas import get_or_build_job_schema  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
OLLAMA_MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:1.7b"
OUTPUT_TAG = sys.argv[2] if len(sys.argv) > 2 else OLLAMA_MODEL.replace(":", "_").replace("/", "_")

# ---------------------------------------------------------------------------
# OLD prompt logic, hand-copied verbatim from git HEAD (pre-redesign) --
# see old_local_tailor.py in this directory for the source of truth.
# ---------------------------------------------------------------------------

_OLD_REALIZATION_SYSTEM = (
    "You write SHORT resume content from pre-verified facts. You are given "
    "a list of bullet SLOTS to fill and a summary schema. For each slot you "
    "are given: which evidence it's based on, the rhetorical shape to "
    "follow (a sequence of slot names, e.g. action -> object -> outcome), "
    "the underlying fact, a CLAIM CEILING, an AGENCY CEILING, and -- when "
    "given -- exact terms to use verbatim or a note that this is "
    "transferable (not identical-domain) experience. Write exactly ONE "
    "sentence per bullet slot, realizing that shape. Never invent a fact, "
    "employer, tool, or number beyond what you're given -- never change or "
    "add a number not already in the fact text. When a slot is marked "
    "transferable, do not claim the source experience IS the target domain "
    "-- state the transferable capability. Use ACTIVE voice, candidate as "
    "subject ('Diagnosed X'), never passive ('X was diagnosed').\n\n"
    "CLAIM CEILING limits TECHNICAL DEPTH: participation < execution < "
    "implementation < design < authority (architected). AGENCY CEILING is "
    "a SEPARATE limit on PEOPLE authority: individual_contributor < owner "
    "< team_lead (led/managed) < director (directed/spearheaded). Never "
    "use a verb stronger than either given ceiling.\n\n"
    "Also write one short (2-3 sentence) professional summary following "
    "the given summary schema's shape, from the given viewpoint where it "
    "fits naturally -- the viewpoint changes emphasis, never facts.\n\n"
    "Output ONLY this JSON, nothing else -- no markdown fences, no prose:\n"
    '{"summary": "...", "bullets": [{"evidence": "<evidence name exactly '
    'as given>", "text": "<one sentence>"}]}\n'
    "Include exactly one bullet entry per slot you were given, using the "
    "evidence name verbatim so it can be matched back."
)


def old_build_realization_prompt(job_schema: dict, max_items: int = 5) -> tuple[str, str] | None:
    supported = [r for r in (job_schema.get("requirements") or []) if r.get("supported") and r.get("schema")][
        :max_items
    ]
    if not supported:
        return None
    lines = [
        f"VIEWPOINT: {job_schema.get('viewpoint', 'general')}",
        f"SUMMARY SCHEMA: {job_schema.get('summary_schema', '')}",
        "",
        "BULLET SLOTS TO FILL (one sentence each):",
    ]
    for r in supported:
        bullet = BULLET_SCHEMAS.get(r["schema"]["bullet_schema"], {})
        if r.get("exact_keywords"):
            anchor = f" | use these exact terms: {', '.join(r['exact_keywords'])}"
        elif r.get("synonym_concepts"):
            anchor = " | TRANSFERABLE experience -- do not claim identical domain"
        else:
            anchor = ""
        force_note = (
            " | this is a PREVENTION fact -- name the risk and the outcome it avoided, not just the action taken"
            if r.get("force_relation") == "prevention"
            else ""
        )
        lines.append(
            f"- evidence: {', '.join(r['resume_evidence'])} | "
            f"shape: {' -> '.join(bullet.get('slots', []))} | "
            f"claim ceiling: {r.get('claim_ceiling', 'participation')} | "
            f"agency ceiling: {r.get('agency_ceiling', 'individual_contributor')} | "
            f"fact: {r['requirement']}{anchor}{force_note}"
        )
    user = "\n".join(lines) + "\n\nReturn the JSON now:"
    return _OLD_REALIZATION_SYSTEM, user


# ---------------------------------------------------------------------------
# Shared evaluation: same safety checks production's request_local_
# realization applies, so both prompt versions are scored identically.
# ---------------------------------------------------------------------------


def call_ollama(system: str, user: str, max_tokens: int = 600) -> tuple[str, float]:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_predict": max_tokens},
    }
    t0 = time.time()
    resp = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=300.0)
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("message") or {}).get("content", "")
    return content, time.time() - t0


def evaluate(raw_text: str, job_schema: dict, profile: dict) -> dict:
    """Mirrors request_local_realization's own scoring: parseable? echoes
    prompt scaffolding? how many bullets survive the real safety checks?"""
    result = {"parseable": False, "echoes_scaffolding": False, "bullets_offered": 0, "bullets_survived": 0, "violations": []}

    try:
        parsed = _parse_plan(raw_text)
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        return result
    result["parseable"] = True

    # Heuristic for the exact Apex failure mode: prompt-scaffolding phrases
    # leaking into bullet text verbatim.
    scaffolding_markers = ("claim ceiling:", "agency ceiling:", "worked example", "bullet slots to fill", "shape:")
    haystack = json.dumps(parsed).lower()
    result["echoes_scaffolding"] = any(m in haystack for m in scaffolding_markers)

    known_metrics = ((profile or {}).get("resume_facts") or {}).get("real_metrics") or []
    ceiling_by_evidence: dict[str, str] = {}
    agency_ceiling_by_evidence: dict[str, str] = {}
    provenance_by_evidence: dict[str, str] = {}
    for r in job_schema.get("requirements") or []:
        if not r.get("supported"):
            continue
        for name in r.get("resume_evidence") or []:
            ceiling_by_evidence[name] = r.get("claim_ceiling") or "participation"
            agency_ceiling_by_evidence[name] = r.get("agency_ceiling") or "individual_contributor"
            prov = r.get("provenance") or []
            if prov:
                first = prov[0]
                provenance_by_evidence[name] = first.get("text", "") if isinstance(first, dict) else str(first)

    raw_bullets = parsed.get("bullets")
    if isinstance(raw_bullets, dict):
        raw_bullets = [{"evidence": k, "text": v} for k, v in raw_bullets.items()]
    for b in raw_bullets or []:
        if not (isinstance(b, dict) and b.get("evidence") and b.get("text")):
            continue
        result["bullets_offered"] += 1
        name = str(b["evidence"]).strip()
        text = str(b["text"]).strip()
        evidence_text = provenance_by_evidence.get(name, "")
        ceiling = ceiling_by_evidence.get(name, "participation")
        agency_ceiling = agency_ceiling_by_evidence.get(name, "individual_contributor")
        checks = [
            check_claim_strength(text, ceiling),
            check_agency_strength(text, agency_ceiling),
            check_causal_claim(text, evidence_text),
            check_metric_fabrication(text, evidence_text, known_metrics),
        ]
        failed = next((c for c in checks if not c["passed"]), None)
        if failed:
            result["violations"].append(f"{name}: {failed.get('violation')}")
        else:
            result["bullets_survived"] += 1

    return result


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()

    # Small batch first: reuse the 3 primary jobs from earlier investigation
    # (varied support levels) plus 2 fresh ones for a bit more range.
    import random
    random.seed(20260902)
    all_qualifying = conn.execute(
        "SELECT rowid FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 800"
    ).fetchall()
    pool = [r[0] for r in all_qualifying]
    job_ids = random.sample(pool, min(25, len(pool)))

    results = []
    for job_id in job_ids:
        row = conn.execute(
            "SELECT rowid, url, title, company, site, location, full_description FROM jobs WHERE rowid=?", (job_id,)
        ).fetchone()
        if row is None:
            continue
        job = dict(row)
        job_schema = get_or_build_job_schema(job, profile)

        old_prompt = old_build_realization_prompt(job_schema)
        new_prompt = new_build_realization_prompt(job_schema)

        entry = {"job_id": job_id, "title": job["title"]}
        for label, prompt in (("old", old_prompt), ("new", new_prompt)):
            if prompt is None:
                entry[label] = {"skipped": "no_supported_requirements"}
                print(f"job {job_id} / {label}: skipped (no supported requirements)")
                continue
            system, user = prompt
            raw, latency = call_ollama(system, user)
            evald = evaluate(raw, job_schema, profile)
            entry[label] = {"latency_s": round(latency, 1), "raw_output": raw, **evald}
            print(
                f"job {job_id} / {label}: {latency:.0f}s parseable={evald['parseable']} "
                f"scaffolding_leak={evald['echoes_scaffolding']} "
                f"survived={evald['bullets_survived']}/{evald['bullets_offered']}"
            )
        results.append(entry)

    (OUT_DIR / f"batch25_results_{OUTPUT_TAG}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Summary: bullets-survived rate per prompt version, with Wilson CIs.
    import math
    def wilson_ci(successes, n, confidence):
        if n == 0:
            return (0.0, 0.0)
        z = {0.95: 1.959963985, 0.99: 2.575829304}[confidence]
        p = successes / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = (z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    for label in ("old", "new"):
        offered = sum(e.get(label, {}).get("bullets_offered", 0) for e in results)
        survived = sum(e.get(label, {}).get("bullets_survived", 0) for e in results)
        leaks = sum(1 for e in results if e.get(label, {}).get("echoes_scaffolding"))
        ci95 = wilson_ci(survived, offered, 0.95)
        ci99 = wilson_ci(survived, offered, 0.99)
        rate = (survived / offered) if offered else 0.0
        print(f"\n{OLLAMA_MODEL} / {label}: bullets {survived}/{offered} survived ({rate:.1%})")
        print(f"  95% CI: [{ci95[0]:.1%}, {ci95[1]:.1%}]  99% CI: [{ci99[0]:.1%}, {ci99[1]:.1%}]")
        print(f"  scaffolding leaks: {leaks}/{len(results)} jobs")

    print("DONE")


if __name__ == "__main__":
    main()
