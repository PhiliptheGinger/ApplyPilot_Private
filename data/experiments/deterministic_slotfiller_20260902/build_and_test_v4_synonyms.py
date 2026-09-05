"""Deterministic slot-filler v4: inflection-tolerant + narrow-synonym
keyword matching, layered on top of v2's sentence-selection.

Built after mining real near-misses across the 100-job v3 sample rather
than guessing a synonym list up front. Finding: of 75 bullets with >=1
missed exact_keyword, ~13/15 hand-inspected cases were the SAME WORD in
a different inflection (customer/customers, operational/operations) --
zero meaning-change risk, just a stricter-than-necessary literal match.
Only ONE case was a genuinely distinct word pair with the same meaning
("issues" in the candidate's sentence for "problems" in the posting).
Several apparent "misses" (e.g. sentence about tires/brakes for a
"solutions" requirement) were genuinely unrelated evidence -- no
synonym system should paper over those, so this deliberately does not
try.

Per explicit instruction: watch for connotation drift even between
technically-synonymous words (example given: "missing children" vs.
"absent children" -- same denotation, very different register/implied
meaning). So the synonym table stays a hand-verified, tiny, closed set
-- same discipline local_tailor.py's _CONCEPT_SYNONYM_PATTERNS already
uses ("add an entry only when a real miss is observed").

Two independent mechanisms:
  1. Inflection-tolerant matching -- does NOT rewrite the sentence, only
     changes whether a keyword counts as matched. A keyword "customers"
     is considered present if the sentence contains "customer" or
     "customers" (simple optional-s pluralization) or, for the small
     curated ROOT_FAMILIES table, another same-root form. No fabrication
     risk whatsoever since no text changes.
  2. Synonym substitution -- DOES rewrite the sentence (swaps in the
     job's own word for the candidate's synonymous word), so every
     substitution is re-checked against the SAME claim_ceiling /
     agency_ceiling the sentence's source evidence item already has,
     using the existing schemas.py checks. A substitution that would
     cross a claim/agency tier is rejected and the original word is
     kept -- this reuses safety machinery rather than inventing new
     rules, and catches the case where a "synonym" happens to also be a
     stronger claim verb (not true of issues/problems, but this is the
     backstop for whatever gets added to the table later).

No LLM call anywhere in this file.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config, database  # noqa: E402
from applypilot.scoring.schemas import (  # noqa: E402
    build_job_schema_representation,
    claim_ceiling_for_evidence,
    agency_ceiling_for_evidence,
    check_claim_strength,
    check_agency_strength,
)

OUT_DIR = Path(__file__).resolve().parent

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# ---------------------------------------------------------------------------
# Mechanism 1: inflection-tolerant matching (no text rewrite).
# ---------------------------------------------------------------------------

# Same-root word families that aren't simple suffix variants (so a plain
# "try adding/stripping 's'" rule wouldn't catch them). Deliberately tiny,
# grown only from the real misses found in the 100-job sample -- NOT a
# general stemmer (decision history: loose stems like "us\w*" matching
# "user" have caused real false positives before; this table is explicit
# word forms, same discipline as _CLAIM_VERB_PATTERNS).
_ROOT_FAMILIES: list[frozenset[str]] = [
    frozenset({"operation", "operations", "operational"}),
]


def _keyword_present(keyword: str, text_lower: str) -> bool:
    """True if `keyword` (or a same-root/plural form) appears in
    `text_lower`. Narrower than stemming: only tries appending/stripping
    a single trailing 's' (ordinary pluralization) plus the small
    explicit _ROOT_FAMILIES table -- never a generic prefix/stem match."""
    kw = keyword.lower()
    variants = {kw}
    if kw.endswith("s"):
        variants.add(kw[:-1])
    else:
        variants.add(kw + "s")
        variants.add(kw + "es")
    for family in _ROOT_FAMILIES:
        if kw in family:
            variants |= family
    return any(re.search(rf"\b{re.escape(v)}\b", text_lower) for v in variants)


def _inflected_score(sentence: str, exact_keywords: list[str]) -> int:
    sl = sentence.lower()
    return sum(1 for kw in exact_keywords if _keyword_present(kw, sl))


# ---------------------------------------------------------------------------
# Mechanism 2: narrow synonym substitution (DOES rewrite text).
# ---------------------------------------------------------------------------

# Hand-verified, same-register, same-claim-tier pairs only. Each entry:
# job_word -> (candidate_word_pattern, candidate_word_replacement).
# Rejected during mining (documented, not added): "solutions"/"product"
# (categorically different things -- a solution isn't a product),
# "solutions"/"troubleshooting" (noun-of-result vs. noun-of-process, not
# interchangeable), "market"/anything (no clean pair found). Per explicit
# review instruction: even a technically-synonymous pair can carry
# different connotation (e.g. "missing children" vs. "absent children")
# -- every entry here was read in context, not just checked against a
# thesaurus.
_SYNONYM_SUBSTITUTIONS: dict[str, tuple[re.Pattern, str]] = {
    "problems": (re.compile(r"\bissues\b", re.IGNORECASE), "problems"),
}


def _apply_synonym_substitution(sentence: str, exact_keywords: list[str]) -> tuple[str, list[str]]:
    """Try substituting one candidate word for the job's own word, for
    each exact_keyword not already present. Returns (possibly-modified
    sentence, list of substitutions applied as 'candidate_word -> job_word')."""
    applied = []
    sl = sentence.lower()
    for kw in exact_keywords:
        if _keyword_present(kw, sl):
            continue
        sub = _SYNONYM_SUBSTITUTIONS.get(kw.lower())
        if sub is None:
            continue
        pattern, replacement = sub
        m = pattern.search(sentence)
        if not m:
            continue
        sentence = pattern.sub(replacement, sentence, count=1)
        sl = sentence.lower()
        applied.append(f"{m.group()} -> {replacement}")
    return sentence, applied


def _safety_check(original_sentence: str, new_sentence: str, evidence_item: dict) -> dict:
    """Re-run the same claim/agency ceiling checks the pipeline already
    uses. A substitution that pushes the sentence's detected tier above
    what the SOURCE ITEM's own text supports is rejected."""
    claim_ceiling = claim_ceiling_for_evidence(evidence_item)
    agency_ceiling = agency_ceiling_for_evidence(evidence_item)
    cs = check_claim_strength(new_sentence, claim_ceiling)
    ags = check_agency_strength(new_sentence, agency_ceiling)
    return {"claim_ok": cs["passed"], "agency_ok": ags["passed"]}


# ---------------------------------------------------------------------------


def _candidate_sentences(item: dict) -> list[str]:
    sentences: list[str] = []
    for resp in item.get("responsibilities") or []:
        if isinstance(resp, str) and resp.strip():
            sentences.append(resp.strip())
    desc = item.get("description")
    if isinstance(desc, str) and desc.strip():
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(desc.strip()) if s.strip())
    return sentences


def _find_item(name: str, profile: dict) -> dict | None:
    for key in ("experience_inventory", "project_inventory"):
        for it in profile.get(key) or []:
            if it.get("name") == name:
                return it
    return None


def _literal_score(sentence: str, exact_keywords: list[str]) -> int:
    sl = sentence.lower()
    return sum(1 for kw in exact_keywords if re.search(rf"\b{re.escape(kw.lower())}\b", sl))


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()

    random.seed(20260903)
    pool = [
        r[0]
        for r in conn.execute(
            "SELECT rowid FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 800"
        ).fetchall()
    ]
    job_ids = random.sample(pool, min(100, len(pool)))

    baseline_scores: list[int] = []
    inflected_scores: list[int] = []
    final_scores: list[int] = []
    substitution_examples: list[dict] = []
    rejected_by_safety_check = 0

    for job_id in job_ids:
        row = conn.execute(
            "SELECT rowid, url, title, company, site, location, full_description FROM jobs WHERE rowid=?", (job_id,)
        ).fetchone()
        if row is None:
            continue
        job = dict(row)
        schema = build_job_schema_representation(job, profile)
        for r in schema["requirements"]:
            if not (r.get("supported") and r.get("schema") and r.get("category_tier") in ("prototype", "near_prototype")):
                continue
            ev = r.get("resume_evidence") or []
            if not ev:
                continue
            item = _find_item(ev[0], profile)
            if item is None:
                continue
            kws = r.get("exact_keywords") or []

            # pick best sentence using the ORIGINAL literal scorer (unchanged
            # selection step -- this experiment only changes scoring/wording
            # AFTER a sentence is already chosen, not which one gets chosen)
            best_s, best_sc = None, -1
            for s in _candidate_sentences(item):
                sc = _literal_score(s, kws)
                if sc > best_sc or (sc == best_sc and best_s is not None and len(s) > len(best_s)):
                    best_s, best_sc = s, sc
            if best_s is None:
                continue

            baseline_scores.append(_literal_score(best_s, kws))
            inflected_scores.append(_inflected_score(best_s, kws))

            new_sentence, applied = _apply_synonym_substitution(best_s, kws)
            if applied:
                safety = _safety_check(best_s, new_sentence, item)
                if not (safety["claim_ok"] and safety["agency_ok"]):
                    rejected_by_safety_check += 1
                    new_sentence = best_s  # revert
                    applied = []
                else:
                    substitution_examples.append(
                        {
                            "title": job["title"][:50],
                            "requirement": r["requirement"][:90],
                            "before": best_s,
                            "after": new_sentence,
                            "substitutions": applied,
                        }
                    )
            final_scores.append(_inflected_score(new_sentence, kws))

    n = len(baseline_scores)
    print(f"bullets evaluated: {n}")
    print(f"mean literal match_score (baseline, v2 behavior): {sum(baseline_scores)/n:.3f}")
    print(f"mean score with inflection-tolerant matching only: {sum(inflected_scores)/n:.3f}")
    print(f"mean score with inflection + synonym substitution: {sum(final_scores)/n:.3f}")
    improved_by_inflection = sum(1 for b, i in zip(baseline_scores, inflected_scores) if i > b)
    improved_by_synonym = sum(1 for i, f in zip(inflected_scores, final_scores) if f > i)
    print(f"\nbullets improved by inflection tolerance alone: {improved_by_inflection}/{n}")
    print(f"bullets further improved by synonym substitution: {improved_by_synonym}/{n}")
    print(f"substitutions rejected by claim/agency safety check: {rejected_by_safety_check}")

    print(f"\n=== synonym substitutions actually applied ({len(substitution_examples)}) ===")
    for ex in substitution_examples:
        print(f"[{ex['title']}] req: {ex['requirement']}")
        print(f"  before: {ex['before']}")
        print(f"  after:  {ex['after']}")
        print(f"  subs:   {ex['substitutions']}")

    (OUT_DIR / "prototype_v4_synonym_results.json").write_text(
        json.dumps(
            {
                "n": n,
                "mean_baseline": sum(baseline_scores) / n,
                "mean_inflected": sum(inflected_scores) / n,
                "mean_final": sum(final_scores) / n,
                "improved_by_inflection": improved_by_inflection,
                "improved_by_synonym": improved_by_synonym,
                "rejected_by_safety_check": rejected_by_safety_check,
                "substitution_examples": substitution_examples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nwrote prototype_v4_synonym_results.json")


if __name__ == "__main__":
    main()
