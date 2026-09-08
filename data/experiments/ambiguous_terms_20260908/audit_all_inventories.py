"""Extends audit_identity_term_trust.py's scope-note follow-up: that audit
only checked experience_inventory (found the "communications" bug, decision
#79). This generalizes the SAME escalating-batch trusted_alone audit across
ALL FOUR evidence-bearing inventories (experience_inventory,
project_inventory, skills_inventory, certifications) to check for the same
bug class anywhere it hasn't been looked for yet.

Read-only. No DB writes, no LLM calls.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
ALL_KEYS = ("experience_inventory", "project_inventory", "skills_inventory", "certifications")
N = 2000


def wilson_interval(hits: int, n: int, z: float = 2.576) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def isolated_profile(full_profile: dict, inv_key: str, entry: dict) -> dict:
    p = dict(full_profile)
    for key in ALL_KEYS:
        p[key] = [entry] if key == inv_key else []
    return p


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()

    rows = conn.execute(
        "SELECT url, title, full_description FROM jobs WHERE full_description IS NOT NULL "
        "AND length(full_description) > 200 ORDER BY RANDOM() LIMIT ?",
        (N,),
    ).fetchall()
    print(f"sample size: {len(rows)}\n")

    all_results = {}
    for inv_key in ALL_KEYS:
        items = [e for e in (profile.get(inv_key) or []) if e.get("resume_allowed") is not False]
        for entry in items:
            name = entry.get("name")
            prof = isolated_profile(profile, inv_key, entry)
            trusted_alone_hits = 0
            driver_words = Counter()
            samples = []
            total_supported = 0

            for url, title, desc in rows:
                job = {"title": title or "", "full_description": desc or ""}
                try:
                    rep = build_job_schema_representation(job, prof)
                except Exception:  # noqa: BLE001
                    continue
                job_supported = False
                for req in rep.get("requirements") or []:
                    if not req.get("supported"):
                        continue
                    job_supported = True
                    if req.get("category_tier") == "near_prototype":
                        trusted_alone_hits += 1
                        kws = req.get("exact_keywords") or []
                        for kw in kws:
                            driver_words[kw] += 1
                        if len(samples) < 5:
                            samples.append(
                                {"title": title, "requirement": req.get("requirement", "")[:150], "keywords": kws}
                            )
                if job_supported:
                    total_supported += 1

            rate = trusted_alone_hits / len(rows)
            lo, hi = wilson_interval(trusted_alone_hits, len(rows))
            print(f"=== [{inv_key}] {name} ===")
            print(
                f"  trusted_alone hits: {trusted_alone_hits}/{len(rows)} = {rate:.2%}  99% CI=[{lo:.2%},{hi:.2%}]  "
                f"total_supported={total_supported} ({total_supported/len(rows):.1%})"
            )
            if driver_words:
                print(f"  driver words: {driver_words.most_common(10)}")
                for s in samples:
                    print(f"    [{s['keywords']}] title={s['title']!r}  req={s['requirement']!r}")
            print()

            all_results[f"{inv_key}::{name}"] = {
                "inventory": inv_key,
                "n": len(rows),
                "trusted_alone_hits": trusted_alone_hits,
                "rate": rate,
                "ci99": [lo, hi],
                "driver_words": dict(driver_words),
                "samples": samples,
            }

    (OUT_DIR / "all_inventories_audit.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("wrote all_inventories_audit.json")


if __name__ == "__main__":
    main()
