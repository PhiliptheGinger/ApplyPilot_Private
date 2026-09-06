"""Round 2: what's still driving each entry's residual false-positive rate
after the 2026-09-05 buzzword-list + supported-flag fixes? Small escalating
sample (matching scan_escalating.py's method -- fast, statistically
sufficient), but this time capturing full matched_via term frequency
across ALL hits, not just a 60-example cap, so we get a real candidate
list for round 2 curation instead of guessing.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
EMPTY_KEYS = ("project_inventory", "skills_inventory", "certifications", "historical_experience_inventory")
SAMPLE_N = 2000  # fixed batch, escalation already proven unnecessary at this scale for these rates


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
        "SELECT url, title, full_description FROM jobs WHERE full_description IS NOT NULL "
        f"AND length(full_description) > 200 ORDER BY RANDOM() LIMIT {SAMPLE_N}"
    ).fetchall()
    print(f"sample size: {len(rows)}\n")

    results = {}
    for entry in entries:
        name = entry.get("name")
        prof = isolated_profile(profile, entry)
        term_counts: Counter = Counter()
        hits = 0
        for job_url, title, desc in rows:
            job = {"title": title or "", "full_description": desc or ""}
            try:
                rep = build_job_schema_representation(job, prof)
            except Exception:
                continue
            supported = [r for r in rep.get("requirements") or [] if r.get("supported")]
            if supported:
                hits += 1
                for r in supported:
                    for t in (r.get("exact_keywords") or []) + (r.get("synonym_concepts") or []):
                        term_counts[t] += 1
        rate = hits / len(rows)
        print(f"=== {name}: {hits}/{len(rows)} = {rate:.1%} ===")
        top = term_counts.most_common(15)
        for t, c in top:
            print(f"    {c:3d}  {t!r}")
        print()
        results[name] = {"n": len(rows), "hits": hits, "rate": rate, "top_terms": top}

    (OUT_DIR / "scan_residual_terms_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote scan_residual_terms_results.json")


if __name__ == "__main__":
    main()
