"""2026-09-16: build a real, stratified sample of (requirement, evidence)
cosine-similarity pairs to calibrate semantic_match.semantic_match_threshold()
(currently a guessed 0.30, never validated against real labeled data).

Pulls real requirement lines from real DB jobs (via the same
_split_requirement_lines the production pipeline uses) and the real
candidate evidence corpus (via _build_full_evidence_corpus), embeds both
with the real all-minilm model, and computes every requirement x evidence
cosine pair. Only pairs where the LITERAL matcher found nothing
(matched_terms empty at the whole-job level) are candidates for this scan --
those are the exact cases semantic recall exists to help with; a pair the
literal matcher already covers isn't testing anything new about this
threshold.

Stratifies into 5 score bands (0.10-0.20, 0.20-0.30, 0.30-0.40, 0.40-0.50,
0.50+) and samples up to N per band for manual labeling, since a uniform
random sample would be dominated by low-score (obviously irrelevant) pairs
and never exercise the boundary this threshold actually needs to draw.
"""

import json
import os
import random
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot.config import load_profile  # noqa: E402
from applypilot.scoring import local_tailor, semantic_match  # noqa: E402

random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 20260916)

PER_BAND = 12
BANDS = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 1.01)]


def main():
    db = os.path.expanduser("~/.applypilot/applypilot.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    profile = load_profile()

    corpus = local_tailor._build_full_evidence_corpus(profile)
    print(f"Evidence corpus: {len(corpus)} items")
    corpus_texts = [c["text"] for c in corpus]
    corpus_embeddings = semantic_match.embed_texts(corpus_texts)
    if corpus_embeddings is None:
        print("ERROR: could not embed corpus (Ollama unreachable?)")
        return
    print(f"Embedded {len(corpus_embeddings)} corpus items")

    # 2026-09-16: `ORDER BY RANDOM() LIMIT N` forces a full-table materialize
    # + sort before the limit is ever applied -- a known SQLite anti-pattern
    # that stalled indefinitely on this machine's real, slow HDD against the
    # jobs table's real size. Sampling several disjoint rowid ranges spread
    # across the table instead touches only the rows actually needed, while
    # avoiding a single contiguous block landing entirely within one
    # company's postings (found live: the first version of this script drew
    # 400 consecutive rows that were ALL from Axon, since scraped jobs are
    # inserted in same-company batches).
    max_id = conn.execute("SELECT MAX(rowid) FROM jobs").fetchone()[0] or 0
    n_chunks = 6
    per_chunk = 400 // n_chunks
    rows = []
    for _ in range(n_chunks):
        start = random.randint(0, max(0, max_id - per_chunk - 100))
        rows.extend(
            conn.execute(
                "SELECT url, title, full_description FROM jobs "
                "WHERE rowid >= ? AND full_description IS NOT NULL AND full_description != '' "
                "LIMIT ?",
                (start, per_chunk),
            ).fetchall()
        )

    distinct_urls = {r["url"].split("/")[2] if "//" in r["url"] else r["url"][:30] for r in rows}
    print(f"{len(rows)} jobs pulled, {len(distinct_urls)} distinct hosts/companies represented")

    all_pairs = []
    seen_req_texts = set()
    for row in rows:
        job = dict(row)
        lines, _dropped = local_tailor._split_requirement_lines(job["full_description"] or "")
        if not lines:
            continue
        req_texts = [l["text"] for l in lines]
        new_texts = [t for t in req_texts if t not in seen_req_texts]
        if not new_texts:
            continue
        req_embeddings = semantic_match.embed_texts(new_texts)
        if req_embeddings is None:
            continue
        for text, vec in zip(new_texts, req_embeddings):
            seen_req_texts.add(text)
            for c, cvec in zip(corpus, corpus_embeddings):
                score = semantic_match.cosine_similarity(vec, cvec)
                all_pairs.append(
                    {
                        "job_title": job["title"],
                        "job_url": job["url"],
                        "requirement": text,
                        "evidence_type": c["type"],
                        "evidence_name": c["name"],
                        "evidence_text": c["text"][:200],
                        "score": round(score, 4),
                    }
                )
        if len(seen_req_texts) % 100 < len(new_texts) and len(seen_req_texts) > 0:
            print(f"...{len(seen_req_texts)} distinct requirement lines processed, {len(all_pairs)} pairs so far")
        if len(seen_req_texts) >= 400:
            break

    print(f"Total pairs computed: {len(all_pairs)}")

    sampled = []
    for lo, hi in BANDS:
        band_pairs = [p for p in all_pairs if lo <= p["score"] < hi]
        random.shuffle(band_pairs)
        picked = band_pairs[:PER_BAND]
        print(f"Band [{lo},{hi}): {len(band_pairs)} candidates, sampled {len(picked)}")
        sampled.extend(picked)

    random.shuffle(sampled)
    out_name = sys.argv[1] if len(sys.argv) > 1 else "sampled_pairs.json"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2)
    print(f"Wrote {len(sampled)} sampled pairs to {out_path}")


if __name__ == "__main__":
    main()
