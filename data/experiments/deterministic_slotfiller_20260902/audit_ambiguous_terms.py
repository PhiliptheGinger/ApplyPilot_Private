"""Proactive polysemy audit: instead of waiting for another real-job bug
report to find the next 'alignment', scan the WHOLE evidence vocabulary
(every single-word term any profile.json entry could match on) against the
real ~30K-job corpus, and flag terms whose real-world usage contexts
disagree with each other -- the same signal that caught 'alignment'
(corporate 'strategic alignment' vs. automotive 'wheel alignment' share no
other local context) -- as candidates for _AMBIGUOUS_TERMS.

Single pass over the corpus (not one query per term): for each job
posting, split into sentences, and for each sentence, intersect its word
set against the candidate-term set. Any hit records that sentence's OTHER
words as one "context sample" for that term (same extraction shape as
schemas._local_context_words). After the full pass, terms with enough
samples get scored by average pairwise Jaccard overlap across their
samples -- low overlap means the term shows up in very different
surrounding vocabulary from one posting to the next, i.e. likely used in
different senses. High overlap means it's consistently used the same way.

This does not by itself add anything to _AMBIGUOUS_TERMS -- it produces a
ranked candidate list for a human (you) to review, same as how "alignment"
itself was found by inspection, not automatically trusted.

No LLM call anywhere in this file.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.local_tailor import _item_terms, _NAME_TOKEN_STOPWORDS, _TERM_WORD_RE  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

MAX_SAMPLES_PER_TERM = 15
MIN_SAMPLES_TO_SCORE = 5


def _candidate_terms(profile: dict) -> set[str]:
    all_terms: set[str] = set()
    for key in ("experience_inventory", "project_inventory", "skills_inventory", "certifications"):
        for item in profile.get(key) or []:
            if isinstance(item, dict) and item.get("resume_allowed") is not False:
                all_terms |= _item_terms(item)
    return {t for t in all_terms if " " not in t and "-" not in t and t.isalpha() and len(t) > 2}


def _context_words(sentence: str, term: str) -> frozenset[str]:
    words = set()
    for w in _TERM_WORD_RE.findall(sentence):
        wl = w.lower()
        if wl != term and wl not in _NAME_TOKEN_STOPWORDS and len(wl) > 2:
            words.add(wl[:-1] if wl.endswith("s") and len(wl) > 3 else wl)
    return frozenset(words)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    terms = _candidate_terms(profile)
    print(f"candidate single-word terms: {len(terms)}")

    samples: dict[str, list[frozenset]] = {t: [] for t in terms}

    rows = conn.execute(
        "SELECT title, full_description FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 200"
    ).fetchall()
    print(f"scanning {len(rows)} postings...")

    for i, row in enumerate(rows):
        text = f"{row[0] or ''}\n{row[1] or ''}"
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            sentence_words = {m.group(0).lower() for m in _TERM_WORD_RE.finditer(sentence)}
            hits = sentence_words & terms
            if not hits:
                continue
            for term in hits:
                bucket = samples[term]
                if len(bucket) < MAX_SAMPLES_PER_TERM:
                    ctx = _context_words(sentence, term)
                    if ctx:
                        bucket.append(ctx)
        if (i + 1) % 10000 == 0:
            print(f"  ...{i+1}/{len(rows)}")

    scored = []
    for term, bucket in samples.items():
        if len(bucket) < MIN_SAMPLES_TO_SCORE:
            continue
        pairs = list(itertools.combinations(bucket, 2))
        avg = sum(jaccard(a, b) for a, b in pairs) / len(pairs)
        scored.append((term, avg, len(bucket)))

    scored.sort(key=lambda x: x[1])
    print(f"\nterms with enough samples to score: {len(scored)}")
    print(f"\n=== 25 lowest average context-overlap (highest polysemy risk) ===")
    for term, avg, n in scored[:25]:
        print(f"  {avg:.3f}  n={n:2d}  {term!r}")
    print(f"\n=== 15 highest average context-overlap (consistent usage, low risk) ===")
    for term, avg, n in scored[-15:]:
        print(f"  {avg:.3f}  n={n:2d}  {term!r}")

    (OUT_DIR / "ambiguous_term_audit.json").write_text(
        json.dumps([{"term": t, "avg_overlap": round(a, 4), "n_samples": n} for t, a, n in scored], indent=2),
        encoding="utf-8",
    )
    print("\nwrote ambiguous_term_audit.json")


if __name__ == "__main__":
    main()
