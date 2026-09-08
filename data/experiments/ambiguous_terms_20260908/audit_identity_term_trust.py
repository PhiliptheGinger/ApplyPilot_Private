"""Future Work item 11 audit: decision #71 trusts a `near_prototype`
(single-keyword) match alone only when that keyword is part of an evidence
item's own deliberately-curated identity (name/relevance_categories/
factual_concepts, see local_tailor._item_identity_terms). That fixed the
specific cases found at the time (a skills_inventory entry literally named
"Database" needing single-word trust) but left an open question: is any
single IDENTITY word -- by construction never filtered by
_GENERIC_EVIDENCE_TERMS or _AMBIGUOUS_TERMS -- itself generic/ambiguous
enough to be driving disproportionate false-positive volume once trusted
alone, the same shape of bug as decision #71's original discovery, just
one level deeper (identity words were assumed safe BECAUSE curated, but
"curated" and "domain-specific" aren't the same guarantee)?

Escalating-batch scan (500 tier, 99% CI) across all 6 experience_inventory
entries, isolated one at a time (same technique as scan_escalating.py),
specifically isolating `trusted_alone` hits (category_tier == "near_prototype"
AND supported == True) and recording which single keyword drove each one,
to see if any one word accounts for a disproportionate share.

Read-only. No DB writes, no LLM calls -- safe to run alongside a local-model
job without resource contention.
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
EMPTY_KEYS = ("project_inventory", "skills_inventory", "certifications", "historical_experience_inventory")
N = 2000


def wilson_interval(hits: int, n: int, z: float = 2.576) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


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
        "AND length(full_description) > 200 ORDER BY RANDOM() LIMIT ?",
        (N,),
    ).fetchall()
    print(f"sample size: {len(rows)}\n")

    all_results = {}
    for entry in entries:
        name = entry.get("name")
        prof = isolated_profile(profile, entry)
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
        print(f"=== {name} ===")
        print(f"  trusted_alone (near_prototype+supported) hits: {trusted_alone_hits}/{len(rows)} = {rate:.2%}  99% CI=[{lo:.2%},{hi:.2%}]")
        print(f"  total supported jobs: {total_supported}/{len(rows)} ({total_supported/len(rows):.1%})")
        print(f"  driver word breakdown: {driver_words.most_common(10)}")
        for s in samples:
            print(f"    [{s['keywords']}] title={s['title']!r}  req={s['requirement']!r}")
        print()

        all_results[name] = {
            "n": len(rows),
            "trusted_alone_hits": trusted_alone_hits,
            "rate": rate,
            "ci99": [lo, hi],
            "driver_words": dict(driver_words),
            "samples": samples,
        }

    (OUT_DIR / "identity_term_trust_audit.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("wrote identity_term_trust_audit.json")


if __name__ == "__main__":
    main()
