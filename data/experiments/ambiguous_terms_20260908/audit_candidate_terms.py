"""Audit whether "troubleshooting"/"hands-on"/"equipment" (Mavis) and
"servers" (Waffle House) are driving real cross-domain false-positive
`supported` matches against the live job corpus -- the exact follow-up
CLAUDE.md's Future Work item 8 flagged after decision #71 deliberately
left these four out of _AMBIGUOUS_TERMS (documented in local_tailor.py's
_GENERIC_EVIDENCE_TERMS comment, ~line 946).

Escalating-batch methodology (500 -> 2,000 -> 8,000 -> full corpus), same
technique as entry_audit_20260905/scan_escalating.py: report a 95% CI at
each tier, escalate while the interval is still too wide to act on
confidently (half-width > PRECISION_TARGET), and report a 99% CI at
whichever tier we stop on for the final, more conservative confirmation.

Two things measured per tier, same isolation technique as
scan_escalating.py (isolate ONE experience_inventory entry at a time so no
other entry's terms can also drive a hit):

1. For every job where the entry is `supported`, record which exact_keywords
   drove it. Bucket into "candidate-term-only" (the requirement's
   exact_keywords are exactly one of the 4 candidate terms and nothing
   else) vs "other". This tells us how much real volume these 4 words are
   responsible for on their own.

2. Print a real sample of the candidate-term-only hits (job title +
   requirement text + the evidence's own source text) for direct manual
   inspection -- don't trust a rate number alone, read real examples.

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
from applypilot.scoring.local_tailor import rank_profile_evidence  # noqa: E402
from applypilot.scoring.schemas import _evidence_own_text, build_job_schema_representation  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
EMPTY_KEYS = ("project_inventory", "skills_inventory", "certifications", "historical_experience_inventory")
CANDIDATE_TERMS = {"troubleshooting", "hands-on", "equipment", "servers"}
ENTRY_NAMES = {"National Tire and Battery / Mavis", "Waffle House"}
SAMPLE_SIZE = 10
TIERS = [500, 2000, 8000, None]  # None = full corpus
PRECISION_TARGET = 0.01  # stop escalating once the 95% CI half-width is <=1pp


def wilson_interval(hits: int, n: int, z: float) -> tuple[float, float]:
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


def scan_tier(entry_name: str, prof: dict, rows: list) -> dict:
    evidence_text_cache: dict[str, str] = {}
    counts = Counter()
    candidate_only_samples = []
    other_samples = []
    total_supported = 0

    for url, title, desc in rows:
        job = {"title": title or "", "full_description": desc or ""}
        try:
            rep = build_job_schema_representation(job, prof)
        except Exception:  # noqa: BLE001 -- one bad row shouldn't kill the scan
            continue
        job_supported = False
        for req in rep.get("requirements") or []:
            if not req.get("supported"):
                continue
            job_supported = True
            keywords = set(req.get("exact_keywords") or [])
            is_candidate_only = bool(keywords) and keywords <= CANDIDATE_TERMS
            is_synonym_only = not keywords and req.get("category_tier") == "peripheral"
            if is_candidate_only:
                counts["candidate_term_only"] += 1
                if len(candidate_only_samples) < SAMPLE_SIZE:
                    ev_name = (req.get("resume_evidence") or [None])[0]
                    ev_text = evidence_text_cache.get(ev_name, "")
                    if ev_name and ev_name not in evidence_text_cache:
                        ranked = rank_profile_evidence(job, prof, top_n=6)
                        for r in ranked:
                            if r["name"] == ev_name:
                                ev_text = _evidence_own_text(r.get("item") or {})
                                evidence_text_cache[ev_name] = ev_text
                                break
                    candidate_only_samples.append(
                        {
                            "url": url,
                            "title": title,
                            "requirement": req.get("requirement"),
                            "exact_keywords": sorted(keywords),
                            "evidence_text": ev_text[:300],
                        }
                    )
            elif not is_synonym_only:
                counts["other_literal"] += 1
                if len(other_samples) < 3:
                    other_samples.append(
                        {"title": title, "requirement": req.get("requirement"), "exact_keywords": sorted(keywords)}
                    )
            else:
                counts["synonym_only"] += 1
        if job_supported:
            total_supported += 1

    return {
        "n": len(rows),
        "total_supported": total_supported,
        "counts": dict(counts),
        "candidate_only_samples": candidate_only_samples,
        "other_samples": other_samples,
    }


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    entries = {
        e.get("name"): e
        for e in (profile.get("experience_inventory") or [])
        if e.get("name") in ENTRY_NAMES and e.get("resume_allowed") is not False
    }

    all_rows = conn.execute(
        "SELECT url, title, full_description FROM jobs WHERE full_description IS NOT NULL "
        "AND length(full_description) > 200 ORDER BY RANDOM()"
    ).fetchall()
    print(f"corpus size: {len(all_rows)}\n")

    final_results = {}
    for name, entry in entries.items():
        prof = isolated_profile(profile, entry)
        print(f"=== {name} ===")
        last_tier_result = None
        for tier in TIERS:
            n = len(all_rows) if tier is None else min(tier, len(all_rows))
            rows = all_rows[:n]
            result = scan_tier(name, prof, rows)
            hits = result["counts"].get("candidate_term_only", 0)
            rate = hits / result["n"] if result["n"] else 0.0
            lo95, hi95 = wilson_interval(hits, result["n"], z=1.960)
            half_width_95 = (hi95 - lo95) / 2
            print(
                f"  tier n={result['n']}: candidate-term-only rate = {hits}/{result['n']} = {rate:.2%}  "
                f"95% CI=[{lo95:.2%}, {hi95:.2%}] (half-width={half_width_95:.2%})"
            )
            last_tier_result = (tier, result, rate, half_width_95)
            if tier is None or half_width_95 <= PRECISION_TARGET:
                break
            print(f"    -> CI too wide (target half-width <= {PRECISION_TARGET:.1%}), escalating...")

        tier, result, rate, half_width_95 = last_tier_result
        hits = result["counts"].get("candidate_term_only", 0)
        lo99, hi99 = wilson_interval(hits, result["n"], z=2.576)
        print(
            f"  FINAL (n={result['n']}): candidate-term-only rate = {rate:.2%}  "
            f"99% CI=[{lo99:.2%}, {hi99:.2%}]"
        )
        print(f"  requirement-level breakdown: {result['counts']}")
        print(f"  --- sample candidate-term-only hits (up to {SAMPLE_SIZE}) ---")
        for s in result["candidate_only_samples"]:
            print(f"    [{s['exact_keywords']}] title={s['title']!r}")
            print(f"      requirement: {s['requirement'][:150]!r}")
            print(f"      evidence_text: {s['evidence_text']!r}")
        print()

        final_results[name] = {
            "final_n": result["n"],
            "rate": rate,
            "ci95": [lo95, hi95],
            "ci99": [lo99, hi99],
            "counts": result["counts"],
            "candidate_only_samples": result["candidate_only_samples"],
            "other_samples": result["other_samples"],
        }

    (OUT_DIR / "audit_results_before.json").write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    print("wrote audit_results_before.json")


if __name__ == "__main__":
    main()
