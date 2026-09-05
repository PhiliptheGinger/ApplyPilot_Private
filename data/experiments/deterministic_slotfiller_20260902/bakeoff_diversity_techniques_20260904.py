"""Bake-off: score every candidate near-duplicate-detection technique
against a small, hand-labeled ground-truth pair set (diversity_groundtruth_
pairs.json), reporting BOTH accuracy and real cost (wall time, call counts,
approximate tokens, disk footprint) per technique -- so "which combination
works" and "which one is draining" are answered from the same run, per
2026-09-04 direction.

Techniques scored:
  1. raw_cosine       -- existing semantic_match.cosine_similarity on
                         all-minilm embeddings (today's shipped behavior).
  2. centered_cosine  -- same embeddings, but semantic_match.center_
                         embeddings() (mean-subtraction within this eval
                         pool) applied first. Cheap, corpus-free partial
                         anisotropy fix (see that function's docstring).
  3. llm_judge        -- local qwen3:1.7b asked directly, per pair,
                         "same claim or different claim?" Zero new
                         dependency, reuses existing llm.py. Included
                         per 2026-09-04 direction ("try it, if it works it
                         works") despite this codebase's own documented
                         precedent that LLM arbitration is unreliable for
                         a related judgment (semantic_match.py's module
                         docstring) -- held to the same quantified bar as
                         everything else here, not given a pass for running.
  4. cross_encoder    -- sentence-transformers CrossEncoder
                         (cross-encoder/quora-distilroberta-base, trained
                         on duplicate-question detection -- closest
                         off-the-shelf analog to "same claim, different
                         words"). New dependency (sentence-transformers +
                         torch), installed in the venv for this experiment
                         only -- not yet added to pyproject.toml. Whether
                         it's worth keeping is exactly what this bake-off
                         answers.

For raw_cosine/centered_cosine (continuous scores), reports precision/
recall/accuracy at BOTH the currently-shipped default threshold (0.90, see
semantic_match.diversity_threshold()) and the best-in-sample threshold
(the split point that maximizes accuracy on just this small set) -- the
first shows "as configured today," the second shows the honest ceiling for
that signal type on this data, since a 12-pair sample is too small to trust
as a generalizable optimal threshold on its own.

Nothing here modifies profile.json, the production diversity filter, or
any pipeline code -- this only decides what deserves to be wired in next.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ["LLM_URL"] = "http://localhost:11434/v1"
os.environ["LLM_MODEL_QUALITY"] = "qwen3:1.7b"
os.environ["APPLYPILOT_LOCAL_LLM_URL"] = "http://localhost:11434/v1"
os.environ["APPLYPILOT_LOCAL_LLM_MODEL"] = "qwen3:1.7b"

from applypilot import llm  # noqa: E402
from applypilot.scoring.semantic_match import (  # noqa: E402
    center_embeddings,
    cosine_similarity,
    embed_texts,
)

OUT_DIR = Path(__file__).resolve().parent
GROUNDTRUTH_PATH = OUT_DIR / "diversity_groundtruth_pairs.json"
DEFAULT_COSINE_THRESHOLD = 0.90

_LLM_JUDGE_SYSTEM = """You compare two resume-bullet sentences and decide if they restate the SAME \
underlying factual claim (even with different wording), or DIFFERENT claims (different facts, \
different responsibilities, or different jobs).

Answer with ONLY one word: SAME or DIFFERENT. No other text."""


def _load_pairs() -> list[dict]:
    return json.loads(GROUNDTRUTH_PATH.read_text(encoding="utf-8"))["pairs"]


def _confusion(predictions: list[str], labels: list[str]) -> dict:
    tp = sum(1 for p, y in zip(predictions, labels) if p == "SAME" and y == "SAME")
    fp = sum(1 for p, y in zip(predictions, labels) if p == "SAME" and y == "DIFFERENT")
    fn = sum(1 for p, y in zip(predictions, labels) if p == "DIFFERENT" and y == "SAME")
    tn = sum(1 for p, y in zip(predictions, labels) if p == "DIFFERENT" and y == "DIFFERENT")
    n = len(labels)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3) if precision == precision else None,
        "recall": round(recall, 3) if recall == recall else None,
    }


def _best_threshold(scores: list[float], labels: list[str]) -> tuple[float, dict]:
    """Sweep every observed score as a candidate split point (predict SAME
    if score >= threshold) and return the threshold maximizing accuracy on
    THIS sample. In-sample by construction -- reported as a ceiling
    estimate for the signal, not a recommended production value."""
    candidates = sorted(set(scores)) + [max(scores) + 0.001]
    best = (0.0, {"accuracy": -1.0})
    for t in candidates:
        preds = ["SAME" if s >= t else "DIFFERENT" for s in scores]
        result = _confusion(preds, labels)
        if result["accuracy"] > best[1]["accuracy"]:
            best = (t, result)
    return best


def eval_cosine_techniques(pairs: list[dict]) -> dict:
    texts = []
    text_index: dict[str, int] = {}
    for p in pairs:
        for key in ("a", "b"):
            if p[key] not in text_index:
                text_index[p[key]] = len(texts)
                texts.append(p[key])

    t0 = time.time()
    embeddings = embed_texts(texts)
    embed_elapsed = time.time() - t0
    if embeddings is None:
        return {"error": "embedding call failed -- is Ollama running with all-minilm pulled?"}

    labels = [p["label"] for p in pairs]

    raw_scores = [cosine_similarity(embeddings[text_index[p["a"]]], embeddings[text_index[p["b"]]]) for p in pairs]
    raw_default_preds = ["SAME" if s >= DEFAULT_COSINE_THRESHOLD else "DIFFERENT" for s in raw_scores]
    raw_best_t, raw_best_result = _best_threshold(raw_scores, labels)

    centered = center_embeddings(embeddings)
    centered_scores = [cosine_similarity(centered[text_index[p["a"]]], centered[text_index[p["b"]]]) for p in pairs]
    centered_default_preds = ["SAME" if s >= DEFAULT_COSINE_THRESHOLD else "DIFFERENT" for s in centered_scores]
    centered_best_t, centered_best_result = _best_threshold(centered_scores, labels)

    return {
        "cost": {
            "embedding_calls": 1,
            "n_texts_embedded": len(texts),
            "wall_time_s": round(embed_elapsed, 3),
        },
        "raw_cosine": {
            "scores": [round(s, 3) for s in raw_scores],
            "at_default_threshold_0.90": _confusion(raw_default_preds, labels),
            "at_best_in_sample_threshold": {"threshold": round(raw_best_t, 3), **raw_best_result},
        },
        "centered_cosine": {
            "scores": [round(s, 3) for s in centered_scores],
            "at_default_threshold_0.90": _confusion(centered_default_preds, labels),
            "at_best_in_sample_threshold": {"threshold": round(centered_best_t, 3), **centered_best_result},
        },
    }


def eval_llm_judge(pairs: list[dict]) -> dict:
    client = llm.get_client(quality=True)
    predictions = []
    labels = [p["label"] for p in pairs]
    total_elapsed = 0.0
    total_chars_in = 0
    total_chars_out = 0
    raw_outputs = []

    for p in pairs:
        user_msg = f'SENTENCE A: "{p["a"]}"\nSENTENCE B: "{p["b"]}"'
        t0 = time.time()
        try:
            raw = client.chat(
                [
                    {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 -- bake-off must keep going even if one pair errors
            raw = f"ERROR: {exc}"
        elapsed = time.time() - t0
        total_elapsed += elapsed
        total_chars_in += len(_LLM_JUDGE_SYSTEM) + len(user_msg)
        total_chars_out += len(raw)
        raw_outputs.append(raw.strip()[:80])

        upper = raw.upper()
        if "SAME" in upper and "DIFFERENT" not in upper:
            predictions.append("SAME")
        elif "DIFFERENT" in upper:
            predictions.append("DIFFERENT")
        else:
            predictions.append("UNPARSEABLE")

    # Score UNPARSEABLE as always-wrong (neither confusion bucket credits it)
    # by mapping it to whichever ground-truth label it did NOT match.
    scored_predictions = [
        (pred if pred in ("SAME", "DIFFERENT") else ("DIFFERENT" if y == "SAME" else "SAME"))
        for pred, y in zip(predictions, labels)
    ]

    return {
        "cost": {
            "llm_calls": len(pairs),
            "model_used": client.last_model_used,
            "wall_time_s": round(total_elapsed, 3),
            "approx_prompt_chars_total": total_chars_in,
            "approx_completion_chars_total": total_chars_out,
            "note": "char counts are a rough token proxy (~4 chars/token); "
                    "real usage not exposed by chat()'s current return contract",
        },
        "n_unparseable": predictions.count("UNPARSEABLE"),
        "raw_outputs": raw_outputs,
        "result": _confusion(scored_predictions, labels),
    }


def eval_cross_encoder(pairs: list[dict]) -> dict:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return {"error": "sentence-transformers not installed"}

    model_name = "cross-encoder/quora-distilroberta-base"
    t0 = time.time()
    model = CrossEncoder(model_name)
    load_elapsed = time.time() - t0

    labels = [p["label"] for p in pairs]
    t0 = time.time()
    raw_scores = model.predict([(p["a"], p["b"]) for p in pairs])
    inference_elapsed = time.time() - t0
    scores = [float(s) for s in raw_scores]

    best_t, best_result = _best_threshold(scores, labels)
    default_preds = ["SAME" if s >= 0.5 else "DIFFERENT" for s in scores]  # 0.5 = natural midpoint for this model

    return {
        "cost": {
            "model": model_name,
            "model_load_time_s": round(load_elapsed, 3),
            "inference_time_s_for_all_pairs": round(inference_elapsed, 3),
            "inference_time_s_per_pair": round(inference_elapsed / len(pairs), 4),
            "n_pairs": len(pairs),
        },
        "scores": [round(s, 3) for s in scores],
        "at_natural_threshold_0.5": _confusion(default_preds, labels),
        "at_best_in_sample_threshold": {"threshold": round(best_t, 3), **best_result},
    }


def main() -> None:
    pairs = _load_pairs()
    print(f"loaded {len(pairs)} ground-truth pairs "
          f"({sum(1 for p in pairs if p['label'] == 'SAME')} SAME / "
          f"{sum(1 for p in pairs if p['label'] == 'DIFFERENT')} DIFFERENT)")

    results = {}

    print("\n=== cosine techniques (raw + centered) ===")
    results["cosine"] = eval_cosine_techniques(pairs)
    print(json.dumps(results["cosine"], indent=2))

    print("\n=== LLM-as-judge (local qwen3:1.7b) ===")
    results["llm_judge"] = eval_llm_judge(pairs)
    print(json.dumps(results["llm_judge"], indent=2))

    print("\n=== cross-encoder (sentence-transformers) ===")
    results["cross_encoder"] = eval_cross_encoder(pairs)
    print(json.dumps(results["cross_encoder"], indent=2))

    (OUT_DIR / "bakeoff_diversity_techniques_20260904_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\nwrote bakeoff_diversity_techniques_20260904_results.json")


if __name__ == "__main__":
    main()
