"""Ambiguous-term audit v3: fix v1's and v2's confounds with one shared
mechanism, per the review of v1/v2's results.

v1 (raw context-overlap) was dominated by genericness, not polysemy:
function words and generic business-speak have inherently varied context
just from being common, not from having two real senses.

v2 (filtered to idf>2.0 first) fixed that, but surfaced a different
confound: past-tense narrative verbs ("explained", "adapted", "diagnosed")
scored as artificially RARE, because their document-frequency count was
computed against the EXACT surface form only -- job POSTINGS are written
imperative/present-tense ("explain complex concepts"), while resume
EVIDENCE is written past-tense narrative, so "explained" looks rare in the
job corpus for a grammatical-style reason, not a topical one.

Both confounds share a root cause: the underlying document-frequency count
(and the context-word sets used for Jaccard overlap) are keyed on exact
surface form, so inflected variants of the same word/concept are silently
undercounted. A single, bounded, deterministic verb-inflection normalizer
-- fix both: aggregate frequency across a word's own inflected forms
(fixes v2's tense-mismatch), AND normalize context words the same way
before computing overlap (fixes v1's inflection-driven noise -- a
sentence saying "explains" and one saying "explained" should count as
sharing that context word, not as disagreeing).

Deliberately NOT a general stemmer applied to arbitrary text for matching
purposes (that's exactly the "us\\w* matches user" failure mode
_CLAIM_VERB_PATTERNS/_AGENCY_VERB_PATTERNS were burned by and moved away
from) -- this is used only inside this offline audit tool, to improve a
ranking signal a human still reviews before anything gets added to
_AMBIGUOUS_TERMS, not to gate production matching. Validated below against
the exact known collision risk before being trusted for anything.

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

_VOWELS = "aeiou"


def normalize_inflection(word: str) -> str:
    """Bounded de-inflection: strips regular -ing/-ed/-s(/-es) suffixes
    with standard silent-e and y->i restoration. NOT a general stemmer --
    no consonant-doubling undo, no Porter-style recursive rule chains, and
    a minimum-length guard so short words are never touched (this is
    exactly the guard that would have stopped "ups" from ever being
    treated as a stem of "up" -- "ups" is length 3, at or below the floor,
    so it's returned unchanged here even though _term_in_text's separate,
    already-shipped pluralization tolerance has its own guard for that
    specific case)."""
    w = word.lower()
    if len(w) <= 4:
        return w
    if w.endswith("ies") and len(w) > 5:
        stem = w[:-3] + "y"
    elif w.endswith("ing") and len(w) > 6:
        stem = w[:-3]
    elif w.endswith("ed") and len(w) > 5:
        stem = w[:-2]
        if stem.endswith("i"):
            stem = stem[:-1] + "y"
    elif (w.endswith("es") and len(w) > 5 and w[-3] in "sxz") or w.endswith(("ches", "shes")):
        stem = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 4:
        stem = w[:-1]
    else:
        stem = w
    # Uniform trailing-silent-e normalization, applied AFTER suffix
    # stripping (and to untouched base forms) -- so "diagnose" (base,
    # untouched by any branch above), "diagnosed" (-ed-stripped to
    # "diagnos"), and "diagnosing" (-ing-stripped to "diagnos") all
    # converge on one bucket, instead of needing correct silent-e
    # RESTORATION after stripping, which would need a real dictionary to
    # do properly. Collapsing everything toward the e-less form is
    # sufficient since this is used for consistent BUCKETING, not for
    # producing a linguistically correct base form.
    if stem.endswith("e") and len(stem) > 4:
        stem = stem[:-1]
    return stem


def _regression_check() -> None:
    """The exact known risk this codebase has been burned by before --
    fail loudly before this script does anything else if the normalizer
    collapses two genuinely unrelated words together."""
    assert normalize_inflection("user") == "user", "must not collapse 'user' toward 'us'"
    assert normalize_inflection("ups") == "ups", "must not touch words at/under the length floor"
    assert normalize_inflection("explained") == normalize_inflection("explains"), "explain-family must unify"
    assert normalize_inflection("diagnose") == normalize_inflection("diagnosed") == normalize_inflection("diagnosing"), (
        "diagnose-family must unify across base/past/-ing despite the silent-e"
    )
    assert normalize_inflection("identify") == normalize_inflection("identified") == normalize_inflection("identifying"), (
        "identify-family must unify across base/past/-ing despite the y->i spelling change"
    )
    print("regression check passed:", {
        "user": normalize_inflection("user"),
        "ups": normalize_inflection("ups"),
        "explained/explains": (normalize_inflection("explained"), normalize_inflection("explains")),
        "diagnosed/diagnosing": (normalize_inflection("diagnosed"), normalize_inflection("diagnosing")),
        "identified/identifying": (normalize_inflection("identified"), normalize_inflection("identifying")),
        "adapted/adapting": (normalize_inflection("adapted"), normalize_inflection("adapting")),
        "alignment": normalize_inflection("alignment"),
        "customers/customer": (normalize_inflection("customers"), normalize_inflection("customer")),
    })


def _candidate_terms(profile: dict) -> set[str]:
    all_terms: set[str] = set()
    for key in ("experience_inventory", "project_inventory", "skills_inventory", "certifications"):
        for item in profile.get(key) or []:
            if isinstance(item, dict) and item.get("resume_allowed") is not False:
                all_terms |= _item_terms(item)
    return {t for t in all_terms if " " not in t and "-" not in t and t.isalpha() and len(t) > 2}


def _context_words(sentence: str, term_stem: str) -> frozenset[str]:
    words = set()
    for w in _TERM_WORD_RE.findall(sentence):
        wl = w.lower()
        stem = normalize_inflection(wl)
        if stem != term_stem and wl not in _NAME_TOKEN_STOPWORDS and len(wl) > 2:
            words.add(stem)
    return frozenset(words)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    _regression_check()

    conn = database.get_connection()
    profile = config.load_profile()
    terms = _candidate_terms(profile)
    term_stems = {t: normalize_inflection(t) for t in terms}
    stem_to_terms: dict[str, list[str]] = {}
    for t, s in term_stems.items():
        stem_to_terms.setdefault(s, []).append(t)
    print(f"candidate terms: {len(terms)}  distinct stems: {len(stem_to_terms)}")

    df: dict[str, int] = {s: 0 for s in stem_to_terms}
    samples: dict[str, list[frozenset]] = {s: [] for s in stem_to_terms}

    rows = conn.execute(
        "SELECT title, full_description FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 200"
    ).fetchall()
    n_docs = len(rows)
    print(f"scanning {n_docs} postings...")

    for i, row in enumerate(rows):
        text = f"{row[0] or ''}\n{row[1] or ''}"
        doc_stems_hit: set[str] = set()
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            sentence_stems = {normalize_inflection(m.group(0).lower()) for m in _TERM_WORD_RE.finditer(sentence)}
            hits = sentence_stems & stem_to_terms.keys()
            if not hits:
                continue
            doc_stems_hit |= hits
            for stem in hits:
                bucket = samples[stem]
                if len(bucket) < MAX_SAMPLES_PER_TERM:
                    ctx = _context_words(sentence, stem)
                    if ctx:
                        bucket.append(ctx)
        for s in doc_stems_hit:
            df[s] += 1
        if (i + 1) % 10000 == 0:
            print(f"  ...{i+1}/{n_docs}")

    import math

    idf = {s: math.log((n_docs + 1) / (c + 1)) + 1.0 for s, c in df.items()}

    scored = []
    for stem, bucket in samples.items():
        if len(bucket) < MIN_SAMPLES_TO_SCORE:
            continue
        pairs = list(itertools.combinations(bucket, 2))
        avg = sum(jaccard(a, b) for a, b in pairs) / len(pairs)
        scored.append({"stem": stem, "terms": stem_to_terms[stem], "idf": round(idf[stem], 3), "overlap": round(avg, 4), "n": len(bucket)})

    # sanity check: where does 'alignment' rank now?
    scored_by_overlap = sorted(scored, key=lambda c: c["overlap"])
    for i, c in enumerate(scored_by_overlap):
        if "alignment" in c["terms"]:
            print(f"\n'alignment' rank by raw overlap: {i+1}/{len(scored_by_overlap)}  {c}")
            break

    specific = [c for c in scored if c["idf"] > 2.0]
    specific.sort(key=lambda c: c["overlap"])
    print(f"\nspecific-enough (idf>2.0) stems: {len(specific)}")
    print(f"\n=== 25 lowest overlap (highest polysemy risk), stem-normalized ===")
    for c in specific[:25]:
        print(f"  idf={c['idf']:.2f}  overlap={c['overlap']:.3f}  n={c['n']:2d}  {c['terms']}")

    (OUT_DIR / "ambiguous_term_audit_v3.json").write_text(json.dumps(specific, indent=2), encoding="utf-8")
    print("\nwrote ambiguous_term_audit_v3.json")


if __name__ == "__main__":
    main()
