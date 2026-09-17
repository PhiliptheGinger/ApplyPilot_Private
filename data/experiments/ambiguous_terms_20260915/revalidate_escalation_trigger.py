"""2026-09-15: real, fresh out-of-sample revalidation of the escalation
trigger (decision #77's opt-in `_AMBIGUOUS_TITLE_RE`, decision #81's
bootstrap CI on the SAME n=52 sample it was built from). Since then this
session's own multi-day scoring push produced 11,904 real cloud-scored
jobs plus 1,000 Claude-direct-scored jobs -- a real, independent pool the
original validation never touched (that used a different, much smaller
local-model bake-off, not real cloud/Claude ground truth at all).

Method (mirrors decisions #77/#81's own design):
  - ground truth = real fit_score (score_method IS NULL -> real Gemini, or
    'claude_direct') from jobs NOT in the original 58-URL validation set.
  - stratified sample: ambiguous-titled jobs are rare (~2% of the real
    pool), so oversample that slice deliberately -- but analyze it as its
    own slice, not blended into an unweighted "gate agreement" number that
    would hide the very thing being tested.
  - for the ambiguous-title slice: run BOTH qwen3:1.7b and qwen3:8b via the
    real, unmodified score_job_deterministic (same ineligibility
    pre-filter, same deterministic_combine table -- not a reimplementation)
    so hybrid's decision for this slice is exactly the 8b result, and
    1.7b-alone's decision is exactly the 1.7b result.
  - for the non-ambiguous slice: only qwen3:1.7b is run (hybrid and
    1.7b-alone are identical here by construction -- no need to also pay
    for an 8b call that would never be used).
  - gate = score >= 8 (the real funnel threshold, decision #29).
  - bootstrap 95% CI on the ambiguous-slice comparison (1.7b vs 8b vs
    ground truth), same methodology as decision #81, so the result can be
    judged for statistical significance, not just point estimates.

Per-job progress is printed as it runs (long real local-model calls) and
partial results are written to a JSON file after every job so an
interrupted run doesn't lose completed work.
"""

import json
import os
import random
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from applypilot import config  # noqa: E402

config.load_env()  # must run before llm.py is imported (module-level env read)

from applypilot.config import load_profile  # noqa: E402
from applypilot.scoring.deterministic_fallback import (  # noqa: E402
    _AMBIGUOUS_TITLE_RE,
    score_job_deterministic,
)

random.seed(20260915)

HERE = os.path.dirname(__file__)
ORIGINAL_URLS_PATH = os.path.join(HERE, "original_validation_urls.json")
OUT_PATH = os.path.join(HERE, "revalidation_results.json")

N_AMBIGUOUS = 50
N_NON_AMBIGUOUS = 30


def main():
    original_urls = set(json.load(open(ORIGINAL_URLS_PATH, encoding="utf-8")))
    db = os.path.expanduser("~/.applypilot/applypilot.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT url, title, site, full_description, fit_score, score_method FROM jobs "
        "WHERE fit_score IS NOT NULL AND full_description IS NOT NULL "
        "AND (score_method IS NULL OR score_method = 'claude_direct')"
    ).fetchall()
    rows = [r for r in rows if r["url"] not in original_urls]
    print(f"Eligible ground-truth pool (fresh, held-out): {len(rows)}")

    ambiguous_pool = [r for r in rows if _AMBIGUOUS_TITLE_RE.search(r["title"] or "")]
    non_ambiguous_pool = [r for r in rows if not _AMBIGUOUS_TITLE_RE.search(r["title"] or "")]
    print(f"  ambiguous-titled: {len(ambiguous_pool)}  non-ambiguous: {len(non_ambiguous_pool)}")

    ambiguous_sample = random.sample(ambiguous_pool, min(N_AMBIGUOUS, len(ambiguous_pool)))
    non_ambiguous_sample = random.sample(non_ambiguous_pool, min(N_NON_AMBIGUOUS, len(non_ambiguous_pool)))
    print(f"Sampling {len(ambiguous_sample)} ambiguous + {len(non_ambiguous_sample)} non-ambiguous jobs.\n")

    profile = load_profile()
    results = {"ambiguous": [], "non_ambiguous": []}

    def job_dict(row):
        return {"title": row["title"] or "", "site": row["site"] or "", "full_description": row["full_description"] or ""}

    total = len(ambiguous_sample) + len(non_ambiguous_sample)
    done = 0
    t_start = time.time()

    for row in ambiguous_sample:
        job = job_dict(row)
        t0 = time.time()
        r17 = score_job_deterministic(job, profile, conn=None, model="qwen3:1.7b")
        t17 = time.time() - t0
        t0 = time.time()
        r8b = score_job_deterministic(job, profile, conn=None, model="qwen3:8b")
        t8b = time.time() - t0
        done += 1
        print(
            f"[{done}/{total}] AMBIGUOUS {row['title']!r} truth={row['fit_score']} "
            f"1.7b={r17['score']} ({t17:.0f}s) 8b={r8b['score']} ({t8b:.0f}s)"
        )
        results["ambiguous"].append(
            {
                "url": row["url"],
                "title": row["title"],
                "truth": row["fit_score"],
                "score_17b": r17["score"],
                "score_8b": r8b["score"],
            }
        )
        json.dump(results, open(OUT_PATH, "w", encoding="utf-8"), indent=2)

    for row in non_ambiguous_sample:
        job = job_dict(row)
        t0 = time.time()
        r17 = score_job_deterministic(job, profile, conn=None, model="qwen3:1.7b")
        t17 = time.time() - t0
        done += 1
        print(f"[{done}/{total}] non-ambig {row['title']!r} truth={row['fit_score']} 1.7b={r17['score']} ({t17:.0f}s)")
        results["non_ambiguous"].append(
            {
                "url": row["url"],
                "title": row["title"],
                "truth": row["fit_score"],
                "score_17b": r17["score"],
            }
        )
        json.dump(results, open(OUT_PATH, "w", encoding="utf-8"), indent=2)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")

    # --- Analysis ---
    def gate(score):
        return score >= 8

    amb = results["ambiguous"]
    non_amb = results["non_ambiguous"]

    def agreement_rate(records, pred_key):
        if not records:
            return None
        correct = sum(1 for r in records if gate(r[pred_key]) == gate(r["truth"]))
        return correct / len(records)

    print("\n=== Ambiguous-title slice (n={}) ===".format(len(amb)))
    print("1.7b-alone gate agreement:", agreement_rate(amb, "score_17b"))
    print("8b-alone gate agreement:  ", agreement_rate(amb, "score_8b"))
    print("(hybrid == 8b-alone on this slice by construction)")

    print("\n=== Non-ambiguous slice (n={}) ===".format(len(non_amb)))
    print("1.7b-alone gate agreement:", agreement_rate(non_amb, "score_17b"))
    print("(hybrid == 1.7b-alone on this slice by construction)")

    all_1_7b = [(gate(r["score_17b"]) == gate(r["truth"])) for r in amb + non_amb]
    all_hybrid = [(gate(r["score_17b"]) == gate(r["truth"])) for r in non_amb] + [
        (gate(r["score_8b"]) == gate(r["truth"])) for r in amb
    ]
    print("\n=== Overall (blended, n={}) ===".format(len(amb) + len(non_amb)))
    print("1.7b-alone overall gate agreement:", sum(all_1_7b) / len(all_1_7b) if all_1_7b else None)
    print("hybrid overall gate agreement:    ", sum(all_hybrid) / len(all_hybrid) if all_hybrid else None)

    # Bootstrap 95% CI on the ambiguous slice (where the real question is)
    def bootstrap_ci(records, pred_key, n_resamples=10000):
        if not records:
            return None
        rng = random.Random(20260915)
        n = len(records)
        rates = []
        for _ in range(n_resamples):
            sample = [records[rng.randrange(n)] for _ in range(n)]
            correct = sum(1 for r in sample if gate(r[pred_key]) == gate(r["truth"]))
            rates.append(correct / n)
        rates.sort()
        lo = rates[int(0.025 * n_resamples)]
        hi = rates[int(0.975 * n_resamples)]
        return lo, hi

    print("\n=== Bootstrap 95% CI, ambiguous-title slice only ===")
    ci17 = bootstrap_ci(amb, "score_17b")
    ci8b = bootstrap_ci(amb, "score_8b")
    print(f"1.7b-alone: {agreement_rate(amb, 'score_17b'):.3f} CI={ci17}")
    print(f"8b-alone (=hybrid on this slice): {agreement_rate(amb, 'score_8b'):.3f} CI={ci8b}")

    print(f"\nFull results written to {OUT_PATH}")


if __name__ == "__main__":
    main()
