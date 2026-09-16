"""2026-09-16: compute precision/recall at candidate threshold values
against the manually-labeled real (requirement, evidence) pairs from both
samples, to decide whether semantic_match.semantic_match_threshold()'s
guessed 0.30 default should change."""

import json
import os

HERE = os.path.dirname(__file__)


def load(sample_file, label_file):
    with open(os.path.join(HERE, sample_file), encoding="utf-8") as f:
        pairs = json.load(f)
    with open(os.path.join(HERE, label_file), encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    for i, p in enumerate(pairs):
        p["label"] = labels[str(i)]
    return pairs


def stats_at(pairs, threshold):
    tp = sum(1 for p in pairs if p["label"] == 1 and p["score"] >= threshold)
    fn = sum(1 for p in pairs if p["label"] == 1 and p["score"] < threshold)
    fp = sum(1 for p in pairs if p["label"] == 0 and p["score"] >= threshold)
    tn = sum(1 for p in pairs if p["label"] == 0 and p["score"] < threshold)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return tp, fn, fp, tn, precision, recall


def main():
    s1 = load("sampled_pairs.json", "labels_sample1.json")
    s2 = load("sampled_pairs_2.json", "labels_sample2.json")
    combined = s1 + s2

    for name, pairs in [("Sample 1 (Axon-only)", s1), ("Sample 2 (12 companies)", s2), ("Combined", combined)]:
        n_pos = sum(p["label"] for p in pairs)
        print(f"\n=== {name} (n={len(pairs)}, {n_pos} genuine) ===")
        for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
            tp, fn, fp, tn, prec, rec = stats_at(pairs, t)
            print(
                f"  threshold={t:.2f}: TP={tp} FN={fn} FP={fp} TN={tn} "
                f"precision={prec:.2f} recall={rec:.2f}"
            )

    print("\n=== Genuine (label=1) pairs, sorted by score, combined ===")
    for p in sorted([p for p in combined if p["label"] == 1], key=lambda p: p["score"]):
        print(f"  {p['score']:.4f}  {p['job_title'][:30]:30s}  {p['requirement'][:50]:50s}  {p['evidence_name']}")


if __name__ == "__main__":
    main()
