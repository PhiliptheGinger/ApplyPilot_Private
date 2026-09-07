"""Gemini-reasoning distillation (Future Work item 1, 2026-09-06 session close).

The prior pass (mentioned in decision #76) was a quick single-word frequency
differential over n=23 clean HIGH-scoring rows only, done ad hoc and not
saved as a script -- flagged in CLAUDE.md as too small a sample to trust as a
negative result ("is there a factor we're missing").

This version, deliberately built as a standalone rerunnable tool (not a
one-off notebook cell) so it can be re-run as more clean data accumulates --
that rerunnability is the actual answer to "could this be automated": the
mining logic itself doesn't need a human in the loop, only running it on a
schedule / after each scoring batch would need real pipeline wiring, which
this does NOT attempt (deliberately out of scope tonight, noted at the end).

Three improvements over the prior pass:
  1. Pulls BOTH sides -- real >=8 (good fit) AND real <=4 (poor fit) reasoning
     text, not just the high side. The low side is >10x more data (302+45+32+14
     = 393 rows at fit_score<=4 vs 23 at >=8) and was never looked at.
  2. Mines multi-word phrases (bigrams/trigrams), not just single words --
     real disqualifying/qualifying language is often multi-word ("does not
     meet", "far exceeding", "direct occupation match", "entry level"),
     exactly as flagged as a known gap in CLAUDE.md's Future Work item 1.
  3. Strips the rubric's own template boilerplate (the SCORE_PROMPT_TEMPLATE's
     own fixed phrasing, e.g. "per the scoring criteria", "results in a score
     of") before mining -- otherwise the differential is dominated by
     sentence-template noise common to ALL reasoning, not signal.

Uses log-odds-ratio with add-k smoothing (a standard, real technique for
comparing word/phrase frequency between two corpora -- not a home-grown
formula) rather than raw frequency, so results aren't just "common words
appear a lot in the bigger corpus."
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot import database  # noqa: E402

OUT_PATH = Path(__file__).parent / "reasoning_distillation_results.json"

# The rubric's own template scaffolding (SCORE_PROMPT_TEMPLATE and its
# variants) -- fixed phrasing that appears in nearly every reasoning string
# regardless of the actual job, so it would otherwise swamp the differential
# with non-informative "signal."
_BOILERPLATE_PHRASES = [
    "per the scoring criteria",
    "per the scoring rubric",
    "results in a score of",
    "result in a score of",
    "the candidate is a",
    "the candidate has",
    "the candidate is",
    "scoring criteria",
    "the candidate",
]

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "to", "of", "in", "on", "for", "with",
    "and", "or", "but", "as", "at", "by", "from", "it", "its", "has", "have",
    "had", "not", "no", "role", "job", "posting", "position",
}


def _strip_boilerplate(text: str) -> str:
    lowered = text.lower()
    for phrase in _BOILERPLATE_PHRASES:
        lowered = lowered.replace(phrase, " ")
    return lowered


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z\-]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def extract_terms(reasoning_texts: list[str]) -> Counter:
    """Word + bigram + trigram counts, boilerplate-stripped, stopword-only
    single words dropped (bigrams/trigrams keep stopwords since phrase
    context matters, e.g. "does not meet")."""
    counts: Counter = Counter()
    for text in reasoning_texts:
        clean = _strip_boilerplate(text)
        tokens = [t for t in _tokenize(clean) if t not in _STOPWORDS]
        counts.update(tokens)
        raw_tokens = _tokenize(clean)
        counts.update(t for t in _ngrams(raw_tokens, 2) if len(t) > 6)
        counts.update(t for t in _ngrams(raw_tokens, 3) if len(t) > 10)
    return counts


def log_odds_ratio(counts_a: Counter, counts_b: Counter, k: float = 0.5) -> dict[str, float]:
    """Informative Dirichlet-prior log-odds-ratio (Monroe et al. 2008 style,
    simplified): compares term frequency in corpus A vs corpus B, smoothed so
    rare terms don't dominate with noisy small-count ratios. Positive score =
    over-represented in A; negative = over-represented in B."""
    vocab = set(counts_a) | set(counts_b)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    scores = {}
    for term in vocab:
        a = counts_a.get(term, 0)
        b = counts_b.get(term, 0)
        if a + b < 3:
            continue  # too rare to trust either way
        log_odds_a = math.log((a + k) / (total_a - a + k))
        log_odds_b = math.log((b + k) / (total_b - b + k))
        delta = log_odds_a - log_odds_b
        variance = 1 / (a + k) + 1 / (b + k)
        z = delta / math.sqrt(variance)
        scores[term] = z
    return scores


def fetch_reasoning(min_score: int | None, max_score: int | None) -> list[str]:
    conn = database.get_connection()
    clauses = [
        "fit_score IS NOT NULL",
        "(score_reasoning IS NULL OR score_reasoning NOT LIKE '%Ineligible:%')",
        "score_reasoning IS NOT NULL AND score_reasoning != ''",
        "scored_at >= '2026-08-28'",
        "full_description NOT LIKE '%Replace All Your Work Tools%'",
        "full_description NOT LIKE '%official careers website%'",
    ]
    params: list = []
    if min_score is not None:
        clauses.append("fit_score >= ?")
        params.append(min_score)
    if max_score is not None:
        clauses.append("fit_score <= ?")
        params.append(max_score)
    sql = f"SELECT score_reasoning FROM jobs WHERE {' AND '.join(clauses)}"
    rows = conn.execute(sql, params).fetchall()
    return [r["score_reasoning"] for r in rows]


def main():
    high = fetch_reasoning(min_score=8, max_score=None)
    low = fetch_reasoning(min_score=None, max_score=4)
    print(f"high (>=8) reasoning rows: {len(high)}")
    print(f"low  (<=4) reasoning rows: {len(low)}")

    if len(high) < 15:
        print(
            "\nWARNING: high-score sample is still small (<15) -- treat results as "
            "directional, not conclusive. Rerun this script after a fresh scoring "
            "batch to get a larger, more trustworthy sample."
        )

    counts_high = extract_terms(high)
    counts_low = extract_terms(low)
    scores = log_odds_ratio(counts_high, counts_low)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_high = [(t, round(z, 2)) for t, z in ranked if z > 0][:30]
    # Bug fix, 2026-09-07: `ranked` is sorted descending (most positive
    # first), so filtering z<0 and taking the first 30 grabbed the LEAST
    # negative values (closest to zero, weakest signal) instead of the most
    # negative (strongest poor-fit signal) -- caught because the first
    # real run's "poor-fit" list was suspiciously clustered right at -0.07
    # for every term, which is what you'd expect from the near-zero tail,
    # not genuine outliers. Sort ascending and take the most negative.
    top_low = sorted(((t, round(z, 2)) for t, z in ranked if z < 0), key=lambda kv: kv[1])[:30]

    print("\n=== Terms over-represented in GOOD-FIT (>=8) reasoning ===")
    for term, z in top_high:
        print(f"  {z:+.2f}  {term}")

    print("\n=== Terms over-represented in POOR-FIT (<=4) reasoning ===")
    for term, z in top_low:
        print(f"  {z:+.2f}  {term}")

    with open(OUT_PATH, "w") as f:
        json.dump(
            {
                "n_high": len(high),
                "n_low": len(low),
                "top_high": top_high,
                "top_low": top_low,
            },
            f,
            indent=2,
        )
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
