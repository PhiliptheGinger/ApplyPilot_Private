"""Follow-up to run_tierb_frame_signal.py's clean negative result: test
embeddings as a STANDALONE primary signal (not frame's tie-breaker, per
user direction) for the same deterministic-scoring-fallback question.

Tests title-only vs title+description-snippet text variants, and raw vs
mean-centered cosine (semantic_match.center_embeddings -- a real, once-
tested-elsewhere de-anisotropy correction, decision #66), against the same
real 6,225-job labeled dataset and train/held-out split used for the frame
test. Lesson applied from that test: report distribution separation AND a
rank-based statistic (AUC-style: fraction of (positive, negative) pairs
correctly ordered) BEFORE any threshold-fitting, since naive accuracy
against a 93.7%-imbalanced base rate is misleading (see the frame test's
93% "gate agreement" that turned out to be 0% recall).

No production code touched.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot import database  # noqa: E402
from applypilot.config import load_profile  # noqa: E402
from applypilot.scoring.schemas import _evidence_own_text  # noqa: E402
from applypilot.scoring.semantic_match import (  # noqa: E402
    center_embeddings,
    cosine_similarity,
    embed_texts,
)

HERE = Path(__file__).parent
SEED = 20260906
BATCH_SIZE = 200


def _profile_evidence_items(profile: dict) -> list[dict]:
    items: list[dict] = []
    for key in ("experience_inventory", "project_inventory", "skills_inventory", "certifications"):
        for item in profile.get(key) or []:
            if isinstance(item, dict) and item.get("resume_allowed") is not False and item.get("name"):
                items.append(item)
    return items


def embed_in_batches(texts: list[str]) -> list[list[float]] | None:
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i : i + BATCH_SIZE]
        vecs = embed_texts(chunk, timeout=60.0)
        if vecs is None:
            print(f"  embedding batch {i}-{i+len(chunk)} FAILED (both Ollama and Gemini unavailable)")
            return None
        out.extend(vecs)
        print(f"  embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def stratified_split(rows: list[dict], seed: int, train_frac: float = 0.7):
    by_band: dict[str, list[dict]] = {}
    for r in rows:
        s = r["fit_score"]
        band = "1-4" if s <= 4 else ("5-7" if s <= 7 else "8-10")
        by_band.setdefault(band, []).append(r)
    rng = random.Random(seed)
    train, held = [], []
    for band, items in by_band.items():
        rng.shuffle(items)
        cut = int(len(items) * train_frac)
        train.extend(items[:cut])
        held.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(held)
    return train, held


def auc_stat(pos: list[float], neg: list[float]) -> float:
    """Fraction of (positive, negative) pairs where positive's score >
    negative's score (0.5 tie credit) -- the Mann-Whitney U / AUC
    statistic. 0.5 = no separation, 1.0 = perfect separation, <0.5 means
    the signal is INVERTED (negatively correlated with real fit)."""
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


def analyze(name: str, rows: list[dict], score_key: str) -> None:
    pos = [r[score_key] for r in rows if r["fit_score"] >= 8 and r[score_key] is not None]
    neg = [r[score_key] for r in rows if r["fit_score"] < 8 and r[score_key] is not None]
    if not pos or not neg:
        print(f"  {name}: insufficient data (pos={len(pos)} neg={len(neg)})")
        return
    import statistics

    auc = auc_stat(pos, neg)
    print(
        f"  {name}: pos(n={len(pos)}) mean={statistics.mean(pos):.3f} median={statistics.median(pos):.3f} | "
        f"neg(n={len(neg)}) mean={statistics.mean(neg):.3f} median={statistics.median(neg):.3f} | AUC={auc:.3f}"
    )


def main():
    conn = database.get_connection()
    profile = load_profile()
    ev_items = _profile_evidence_items(profile)
    ev_texts = [_evidence_own_text(item) for item in ev_items]
    print(f"profile evidence items: {len(ev_items)}")

    rows = conn.execute(
        """
        SELECT url, title, fit_score, full_description FROM jobs
        WHERE fit_score IS NOT NULL
          AND (score_reasoning IS NULL OR score_reasoning NOT LIKE '%Ineligible:%')
          AND full_description IS NOT NULL
        """
    ).fetchall()
    rows = [dict(r) for r in rows]
    print(f"total labeled (non-prefilter) jobs: {len(rows)}")

    print("\nEmbedding profile evidence (title text variant)...")
    ev_vecs = embed_in_batches(ev_texts)
    if ev_vecs is None:
        print("FATAL: could not embed profile evidence. Aborting.")
        return

    title_texts = [r["title"] or "" for r in rows]
    print(f"\nEmbedding {len(title_texts)} job titles...")
    title_vecs = embed_in_batches(title_texts)
    if title_vecs is None:
        print("FATAL: could not embed job titles. Aborting.")
        return

    snippet_texts = [f"{r['title'] or ''}. {(r['full_description'] or '')[:500]}" for r in rows]
    print(f"\nEmbedding {len(snippet_texts)} title+description snippets...")
    snippet_vecs = embed_in_batches(snippet_texts)
    if snippet_vecs is None:
        print("FATAL: could not embed snippets. Aborting.")
        return

    # Raw cosine, both text variants.
    ev_vecs_centered = center_embeddings(ev_vecs)
    title_vecs_centered = center_embeddings(title_vecs)
    snippet_vecs_centered = center_embeddings(snippet_vecs)

    for r, tv, tvc, sv, svc in zip(rows, title_vecs, title_vecs_centered, snippet_vecs, snippet_vecs_centered):
        r["title_max_raw"] = max(cosine_similarity(tv, ev) for ev in ev_vecs)
        r["title_max_centered"] = max(cosine_similarity(tvc, ev) for ev in ev_vecs_centered)
        r["snippet_max_raw"] = max(cosine_similarity(sv, ev) for ev in ev_vecs)
        r["snippet_max_centered"] = max(cosine_similarity(svc, ev) for ev in ev_vecs_centered)
        r["title_mean_raw"] = sum(cosine_similarity(tv, ev) for ev in ev_vecs) / len(ev_vecs)
        r["snippet_mean_raw"] = sum(cosine_similarity(sv, ev) for ev in ev_vecs) / len(ev_vecs)

    print("\n=== Distribution separation + AUC (positive=fit_score>=8 vs negative=<8), ALL DATA ===")
    for key in [
        "title_max_raw",
        "title_max_centered",
        "snippet_max_raw",
        "snippet_max_centered",
        "title_mean_raw",
        "snippet_mean_raw",
    ]:
        analyze(key, rows, key)

    train_jobs, held_jobs = stratified_split(rows, seed=SEED)
    print(f"\ntrain: {len(train_jobs)} held-out: {len(held_jobs)}")

    with open(HERE / "tierb_embedding_results.json", "w") as f:
        json.dump(
            {"all": rows, "train_urls": [r["url"] for r in train_jobs], "held_urls": [r["url"] for r in held_jobs]},
            f,
            default=str,
        )
    print("\nSaved full results to tierb_embedding_results.json")


if __name__ == "__main__":
    main()
