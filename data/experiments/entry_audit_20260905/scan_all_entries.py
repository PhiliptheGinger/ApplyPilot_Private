"""Generalize decision #68's Alex Prosperity Group false-positive audit to
the other 5 experience_inventory entries. Reuses build_job_schema_
representation verbatim (the same production function that both
local_tailor's degraded-mode realization and schemas.format_schema_
guidance's live cloud-tailoring-prompt injection call) -- not a bespoke
reimplementation of the matching logic, so the counts reflect exactly
what the real pipeline would do today, post-#68's fixes.

Isolation technique (same as #68): build a copy of profile.json with
ONLY the one experience_inventory entry under test, other evidence-
bearing lists (project_inventory, skills_inventory, certifications,
historical_experience_inventory) emptied -- so a "supported" hit can only
be attributed to that one entry, not diluted/confused by the other five.

No LLM call. embed_texts degrades to None (connection refused, near-
instant) since no local Ollama is running -- semantic expansion is
skipped, matching is literal/root-family/synonym-table only, same
conditions the #68 audit ran under.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
EMPTY_KEYS = ("project_inventory", "skills_inventory", "certifications", "historical_experience_inventory")


def isolated_profile(full_profile: dict, entry: dict) -> dict:
    p = dict(full_profile)
    p["experience_inventory"] = [entry]
    for key in EMPTY_KEYS:
        p[key] = []
    return p


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    entries = [e for e in (profile.get("experience_inventory") or []) if e.get("resume_allowed") is not False]

    rows = conn.execute(
        "SELECT url, title, full_description FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 200 ORDER BY RANDOM()"
    ).fetchall()
    print(f"scanning {len(rows)} postings across {len(entries)} entries...\n")

    results = {}
    for entry in entries:
        name = entry.get("name")
        prof = isolated_profile(profile, entry)
        hits = []
        for i, (job_url, title, desc) in enumerate(rows):
            job = {"title": title or "", "full_description": desc or ""}
            try:
                rep = build_job_schema_representation(job, prof)
            except Exception as exc:  # noqa: BLE001 -- audit script, one bad row shouldn't kill the whole scan
                print(f"  [{name}] job {job_url} raised {type(exc).__name__}: {exc}")
                continue
            supported_reqs = [r for r in rep.get("requirements") or [] if r.get("supported")]
            if supported_reqs:
                hits.append(
                    {
                        "job_url": job_url,
                        "title": title,
                        "n_supported": len(supported_reqs),
                        "sample_requirement": supported_reqs[0]["requirement"][:160],
                        "matched_via": supported_reqs[0].get("exact_keywords") or supported_reqs[0].get("synonym_concepts"),
                    }
                )
            if (i + 1) % 5000 == 0:
                print(f"  [{name}] ...{i+1}/{len(rows)} ({len(hits)} hits so far)")
        results[name] = {"n_jobs_scanned": len(rows), "n_hits": len(hits), "hits": hits[:60]}
        print(f"[{name}] {len(hits)} / {len(rows)} jobs supported\n")

    (OUT_DIR / "scan_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote scan_results.json")
    print("\n=== SUMMARY ===")
    for name, r in results.items():
        print(f"  {name}: {r['n_hits']} / {r['n_jobs_scanned']}")


if __name__ == "__main__":
    main()
