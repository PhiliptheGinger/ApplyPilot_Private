"""Compute per-term inverse-document-frequency weights from this project's
own ~29K-row local jobs database, to fix the v5 gate's failure mode: raw
keyword COUNT (>=2) can be satisfied by two generic connector words
("using", "issues") just as easily as two genuinely specific ones
("alignment", "troubleshooting"), which is exactly why v5's gate blocked
0/3 known-bad hits.

No new dependency: pure word-counting + a log formula, over the job corpus
already sitting in ~/.applypilot/applypilot.db. This is a plain empirical
corpus statistic, not a claim/fact-verification tool -- it says nothing
about whether a match is TRUE, only how common the matched WORD is across
real job postings (a proxy for "how much does this specific hit actually
corroborate topical relevance").

Output: idf_weights.json -- {term: idf_weight} for every word appearing in
>=1 job posting, using smoothed IDF: idf = ln((N + 1) / (df + 1)) + 1
(sklearn's smooth-idf formula -- avoids div-by-zero, and the "+1" floor
means even a term in EVERY document still gets a small positive weight
rather than exactly zero).
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import database  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

_TERM_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#/_-]*")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TERM_WORD_RE.finditer(text)}


def main() -> None:
    conn = database.get_connection()
    rows = conn.execute(
        "SELECT title, full_description FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 200"
    ).fetchall()

    n_docs = 0
    df: Counter[str] = Counter()
    for row in rows:
        text = f"{row[0] or ''}\n{row[1] or ''}"
        terms = _tokenize(text)
        if not terms:
            continue
        n_docs += 1
        df.update(terms)

    print(f"corpus size: {n_docs} job postings")
    print(f"distinct terms: {len(df)}")

    idf_weights = {term: math.log((n_docs + 1) / (count + 1)) + 1.0 for term, count in df.items()}

    # Sanity spot-check: print idf for the exact terms from the v5 known-bad
    # hits, to confirm the intuition before this gets used for anything.
    print("\n=== spot-check idf weights ===")
    for term in ["using", "issues", "problems", "alignment", "customer", "troubleshooting", "python", "the", "and"]:
        w = idf_weights.get(term)
        print(f"  {term!r}: idf={w:.3f}" if w is not None else f"  {term!r}: not in corpus")

    (OUT_DIR / "idf_weights.json").write_text(json.dumps(idf_weights, sort_keys=True), encoding="utf-8")
    print(f"\nwrote idf_weights.json ({len(idf_weights)} terms)")


if __name__ == "__main__":
    main()
