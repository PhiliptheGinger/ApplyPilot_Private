"""Combine Tier A (deterministic hard gate) with Tier B's best embedding
signal (snippet_mean_raw, AUC=0.615 alone), per real research confirming
this is the standard effective pattern for this problem class (hard-
constraint enforcement + semantic similarity, not either alone).

Reuses embeddings already computed and saved by
run_tierb_embedding_signal.py -- no re-embedding needed.

Gate = eligibility.seniority_disqualifier(title) [already-canonical,
title-only] OR a years-required-in-software/engineering regex over the
description body [new, narrow, modeled on tailor.py's existing
_SENIOR_YEARS_RE but with the candidate's REAL disqualification threshold
per the scoring rubric (2-3+ years professional SWE experience, not just
8+/"senior"-tier -- the profile has no years_of_experience_total to compare
against, so this mirrors the rubric's own hardcoded domain-experience gate
instead of a generic number comparison].

Evaluation: of jobs the gate does NOT reject, does embedding similarity
(snippet_mean_raw) meaningfully separate real >=8 from real <8? Report
AUC among survivors, plus overall recall/precision on the real >=8 class
for the full two-stage pipeline (gate rejects -> predict <8; gate passes
-> predict >=8 iff embedding score is above a train-fit threshold).
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot.eligibility import seniority_disqualifier  # noqa: E402

HERE = Path(__file__).parent

# Modeled on tailor.py's _SENIOR_YEARS_RE, but at the candidate's real
# disqualification threshold (2+) per the scoring rubric's own tiers
# ("5-6: role requiring 2+ years... 3-4: role requiring 2+ years of
# professional experience... 1-2: any role requiring 3+ years"), not just
# the 8+/"senior" tier that regex targets.
_YEARS_SWE_RE = re.compile(
    r"\b([2-9]|1[0-9])\+?\s*years?\b[^.\n]{0,60}\b"
    r"(?:software engineering|software development|programming|coding|"
    r"engineering experience)\b",
    re.IGNORECASE,
)


def gate_reject(title: str, description: str) -> bool:
    if seniority_disqualifier(title):
        return True
    if _YEARS_SWE_RE.search(description or ""):
        return True
    return False


def auc_stat(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def main():
    with open(HERE / "tierb_embedding_results.json") as f:
        data = json.load(f)
    rows = data["all"]
    train_urls = set(data["train_urls"])
    held_urls = set(data["held_urls"])

    for r in rows:
        r["gate_reject"] = gate_reject(r["title"], r["full_description"])

    n_rejected = sum(1 for r in rows if r["gate_reject"])
    print(f"total jobs: {len(rows)}, gate-rejected: {n_rejected} ({100*n_rejected/len(rows):.1f}%)")

    # How accurate is the gate ALONE (ignore embeddings) -- rejected -> predict <8?
    rejected = [r for r in rows if r["gate_reject"]]
    survivors = [r for r in rows if not r["gate_reject"]]
    gate_false_negatives = sum(1 for r in rejected if r["fit_score"] >= 8)
    print(
        f"gate false negatives (rejected but real fit_score>=8): {gate_false_negatives}/{len(rejected)} "
        f"({100*gate_false_negatives/len(rejected):.2f}%)"
    )
    print(f"survivors (gate did not reject): {len(survivors)}")

    # Does embedding similarity separate real >=8 from real <8 AMONG SURVIVORS?
    surv_pos = [r["snippet_mean_raw"] for r in survivors if r["fit_score"] >= 8]
    surv_neg = [r["snippet_mean_raw"] for r in survivors if r["fit_score"] < 8]
    print(f"\namong survivors: real>=8 n={len(surv_pos)}, real<8 n={len(surv_neg)}")
    print(f"  real>=8 mean sim: {statistics.mean(surv_pos):.3f}" if surv_pos else "  (none)")
    print(f"  real<8  mean sim: {statistics.mean(surv_neg):.3f}" if surv_neg else "  (none)")
    print(f"  AUC among survivors: {auc_stat(surv_pos, surv_neg):.3f}")

    # Full two-stage pipeline, train-fit threshold on survivors' embedding score.
    train_survivors = [r for r in survivors if r["url"] in train_urls]
    held_survivors = [r for r in survivors if r["url"] in held_urls]

    train_pos = [r["snippet_mean_raw"] for r in train_survivors if r["fit_score"] >= 8]
    train_neg = [r["snippet_mean_raw"] for r in train_survivors if r["fit_score"] < 8]
    # Threshold: midpoint between the two train means (simple, honest -- not
    # over-fit to held-out).
    if train_pos and train_neg:
        threshold = (statistics.mean(train_pos) + statistics.mean(train_neg)) / 2
    else:
        threshold = 0.15
    print(f"\ntrain-fit threshold (midpoint of class means): {threshold:.3f}")

    def predict(r):
        if r["gate_reject"]:
            return 2
        return 9 if r["snippet_mean_raw"] >= threshold else 5

    for split_name, split_rows in [("TRAIN", [r for r in rows if r["url"] in train_urls]), ("HELD-OUT", [r for r in rows if r["url"] in held_urls])]:
        preds = [predict(r) for r in split_rows]
        real = [r["fit_score"] for r in split_rows]
        real_pos_idx = [i for i, s in enumerate(real) if s >= 8]
        tp = sum(1 for i in real_pos_idx if preds[i] >= 8)
        fn = len(real_pos_idx) - tp
        pred_pos_idx = [i for i, p in enumerate(preds) if p >= 8]
        fp = sum(1 for i in pred_pos_idx if real[i] < 8)
        recall = tp / len(real_pos_idx) if real_pos_idx else float("nan")
        precision = tp / len(pred_pos_idx) if pred_pos_idx else float("nan")
        gate_agree = sum(1 for p, s in zip(preds, real) if (p >= 8) == (s >= 8)) / len(real)
        print(
            f"{split_name}: n={len(real)} real_pos={len(real_pos_idx)} ({100*len(real_pos_idx)/len(real):.1f}%) "
            f"TP={tp} FN={fn} FP={fp} recall={recall:.2f} precision={precision:.2f} gate_agree={100*gate_agree:.1f}%"
        )


if __name__ == "__main__":
    main()
