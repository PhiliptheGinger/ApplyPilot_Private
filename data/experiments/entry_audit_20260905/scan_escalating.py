"""Escalating-batch version of scan_all_entries.py. Instead of always
scanning the full ~31,644-job corpus (~1hr), start with a small random
sample per entity, compute a Wilson-score confidence interval on the hit
rate, and only escalate to a bigger sample if the result is still
ambiguous -- i.e. if the "before" (pre-fix) rate could still plausibly be
the true rate given the current sample's uncertainty.

Escalation tiers: 500 -> 2,000 -> 8,000 -> full corpus. Stops as soon as a
tier's 99%-confidence interval clearly excludes the pre-fix rate (a full
population count, not itself an estimate, so no CI needed on that side).

Same isolation technique and same production code path
(build_job_schema_representation) as scan_all_entries.py.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
EMPTY_KEYS = ("project_inventory", "skills_inventory", "certifications", "historical_experience_inventory")
TIERS = [500, 2000, 8000, None]  # None = full corpus

# Pre-fix full-corpus rates from data/experiments/entry_audit_20260905/run.log
BEFORE_RATES = {
    "Freelance Photography / Videography": 5029 / 31644,
    "AMP Smart": 17575 / 31644,
    "UPS": 16237 / 31644,
    "National Tire and Battery / Mavis": 16806 / 31644,
    "Alex Prosperity Group / UST Logistics": 6055 / 31644,
    "Waffle House": 18257 / 31644,
}


def wilson_interval(hits: int, n: int, z: float = 2.576) -> tuple[float, float]:
    """z=2.576 -> 99% CI. Wilson score interval -- more reliable than the
    normal approximation for proportions that aren't close to 0.5 or for
    smaller n, both of which apply here."""
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


def scan(entry: dict, prof: dict, rows: list) -> tuple[int, int]:
    hits = 0
    for job_url, title, desc in rows:
        job = {"title": title or "", "full_description": desc or ""}
        try:
            rep = build_job_schema_representation(job, prof)
        except Exception:  # noqa: BLE001 -- one bad row shouldn't kill the scan
            continue
        if any(r.get("supported") for r in rep.get("requirements") or []):
            hits += 1
    return hits, len(rows)


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    entries = [e for e in (profile.get("experience_inventory") or []) if e.get("resume_allowed") is not False]

    all_rows = conn.execute(
        "SELECT url, title, full_description FROM jobs WHERE full_description IS NOT NULL "
        "AND length(full_description) > 200 ORDER BY RANDOM()"
    ).fetchall()
    print(f"corpus size: {len(all_rows)}\n")

    results = {}
    pending = list(entries)
    for tier in TIERS:
        if not pending:
            break
        n = len(all_rows) if tier is None else min(tier, len(all_rows))
        rows = all_rows[:n]
        print(f"=== tier n={n} ({len(pending)} entries still unresolved) ===")
        still_pending = []
        for entry in pending:
            name = entry.get("name")
            prof = isolated_profile(profile, entry)
            hits, scanned = scan(entry, prof, rows)
            rate = hits / scanned
            lo, hi = wilson_interval(hits, scanned)
            before = BEFORE_RATES.get(name)
            resolved = tier is None or before is None or not (lo <= before <= hi)
            status = "RESOLVED" if resolved else "ambiguous, escalating"
            print(
                f"  [{name}] {hits}/{scanned} = {rate:.1%}  99% CI=[{lo:.1%}, {hi:.1%}]  "
                f"before={before:.1%}  -> {status}"
            )
            results[name] = {
                "n": scanned,
                "hits": hits,
                "rate": rate,
                "ci99": [lo, hi],
                "before_rate": before,
                "resolved": resolved,
            }
            if not resolved:
                still_pending.append(entry)
        pending = still_pending
        print()

    (OUT_DIR / "scan_escalating_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("=== SUMMARY ===")
    for name, r in results.items():
        before = r["before_rate"]
        delta = (r["rate"] - before) if before is not None else None
        print(f"  {name}: before={before:.1%}  after={r['rate']:.1%} (n={r['n']})  delta={delta:+.1%}")
    print("\nwrote scan_escalating_results.json")


if __name__ == "__main__":
    main()
