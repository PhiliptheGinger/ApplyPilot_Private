"""Spot-check: can the existing deterministic evidence-matching layer
(schemas.py's build_job_schema_representation) approximate the LLM fit
score, as a quota-outage fallback for scorer.py's LLM-only judgment step?

Context: scorer.py already has a deterministic pre-filter (_check_ineligible
-- geography/seniority/ethics/degree/clearance/commission-only) that catches
clear disqualifications for free. What it does NOT have is any fallback for
the POSITIVE judgment (how well does this ELIGIBLE job match, 1-10) -- that
step is LLM-only today, and a Gemini daily-quota exhaustion blocks it
entirely for hours.

Method: pull a stratified real sample from jobs the LLM actually judged
(excludes _check_ineligible's own "Ineligible:" rejections -- those are
already deterministic), compute build_job_schema_representation's supported/
total requirement ratio for each, map it to a 1-10 score via a naive linear
formula, and compare against the REAL, already-recorded LLM score. This is
a real spot-check against real historical ground truth, not a synthetic
test.

No production code is touched by this script.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot import database  # noqa: E402
from applypilot.config import load_profile  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402

HERE = Path(__file__).parent


def deterministic_score(job: dict, profile: dict) -> dict:
    """Naive evidence-coverage-ratio approximation. Returns a dict with the
    score plus the raw signal, so disagreements can be inspected."""
    schema = build_job_schema_representation(job, profile)
    reqs = schema.get("requirements") or []
    total = len(reqs)
    supported = sum(1 for r in reqs if r.get("supported"))
    if total == 0:
        return {"score": None, "supported": 0, "total": 0, "ratio": None}
    ratio = supported / total
    # Naive linear map: 0% supported -> 2, 100% supported -> 10.
    score = round(2 + ratio * 8)
    return {"score": score, "supported": supported, "total": total, "ratio": round(ratio, 3)}


def main():
    conn = database.get_connection()
    profile = load_profile()

    with open(HERE / "sample_urls.json") as f:
        urls = json.load(f)

    results = []
    for url in urls:
        row = conn.execute("SELECT * FROM jobs WHERE url = ?", (url,)).fetchone()
        if row is None:
            continue
        job = dict(row)
        det = deterministic_score(job, profile)
        results.append(
            {
                "url": url,
                "title": job.get("title"),
                "real_score": job.get("fit_score"),
                "det_score": det["score"],
                "supported": det["supported"],
                "total": det["total"],
                "ratio": det["ratio"],
                "diff": (det["score"] - job.get("fit_score")) if det["score"] is not None else None,
            }
        )

    results.sort(key=lambda r: (r["real_score"] is None, r["real_score"]))
    print(f"{'real':>4} {'det':>4} {'diff':>5} {'sup/total':>10}  title")
    for r in results:
        diff_str = f"{r['diff']:+d}" if r["diff"] is not None else "  n/a"
        sup_total = f"{r['supported']}/{r['total']}" if r["total"] else "0/0"
        print(f"{r['real_score']:>4} {str(r['det_score']):>4} {diff_str:>5} {sup_total:>10}  {r['title'][:60]}")

    have_both = [r for r in results if r["det_score"] is not None]
    n = len(have_both)
    exact = sum(1 for r in have_both if r["diff"] == 0)
    within2 = sum(1 for r in have_both if abs(r["diff"]) <= 2)
    no_signal = sum(1 for r in results if r["total"] == 0)
    print()
    print(f"n={len(results)}, no requirement lines extracted (0/0): {no_signal}")
    print(f"of {n} with a signal: exact match {exact} ({100*exact/n:.0f}%), within +-2 {within2} ({100*within2/n:.0f}%)")

    # Same-side-of-the-tailoring-gate agreement: does det_score's >=8/<8
    # split agree with the real score's >=8/<8 split? This is the actual
    # decision the pipeline makes downstream (min_score=8 gate).
    gate_agree = sum(1 for r in have_both if (r["det_score"] >= 8) == (r["real_score"] >= 8))
    print(f"agrees with real score on the >=8 tailoring gate: {gate_agree}/{n} ({100*gate_agree/n:.0f}%)")

    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
