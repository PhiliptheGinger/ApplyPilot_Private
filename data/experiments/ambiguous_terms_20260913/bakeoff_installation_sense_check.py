"""Bake-off for CLAUDE.md Future Work item 25 / decision #143: three candidate
fixes for _context_senses_agree's false negative on "installation" (Machine
Installation Technician vs. Alex Prosperity Group's own thin evidence
sentence), tested against a real sample of jobs mentioning "install".

Candidates:
  (a) widen_evidence_context -- _local_context_words scoped to the WHOLE
      evidence item's own text (not just the sentence containing the term),
      evidence side only. Requirement side stays sentence-scoped.
  (b) DATA fix (enrich Alex Prosperity Group's own sentence) -- not testable
      as a code bake-off; noted as a manual real candidate, not run here.
  (c) physical_signal_agree -- a narrower, deterministic check: agree if
      EITHER text's local context around the term contains a "physical
      on-site installation" signal word, regardless of literal overlap.

For each candidate, measures on a real random sample:
  - recovery: real jobs where baseline=unsupported but the candidate says
    supported, manually spot-checked for correctness.
  - false-positive risk: real jobs using "installation" in a clearly
    SOFTWARE/SYSTEM sense that the candidate should NOT flip to supported.
"""

import random
import sqlite3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from applypilot.scoring import local_tailor, schemas  # noqa: E402
from applypilot.config import load_profile  # noqa: E402

random.seed(20260913)

_PHYSICAL_SIGNAL_WORDS = {
    "equipment", "appliance", "appliances", "machine", "machines", "unit", "units",
    "device", "devices", "onsite", "site", "premises", "vehicle", "truck", "vending",
    "hvac", "furniture", "hardware",
}


def widen_evidence_context(evidence_text: str, term: str) -> set[str]:
    """Fix (a): the SAME extraction as schemas._local_context_words, but
    scanning the whole text for content words rather than just the sentence
    containing the term's occurrence -- only sensible on the EVIDENCE side,
    where the source text is short and hand-curated, not a full job posting."""
    import re

    words: set[str] = set()
    for w in schemas._TERM_WORD_RE.findall(evidence_text or ""):
        wl = w.lower()
        if wl == term.lower() or wl in schemas._NAME_TOKEN_STOPWORDS or len(wl) <= 2:
            continue
        normalized = wl[:-1] if wl.endswith("s") and len(wl) > 3 else wl
        if schemas._is_generic_evidence_term(normalized):
            continue
        if wl in schemas._AMBIGUOUS_TERMS or normalized in schemas._AMBIGUOUS_TERMS:
            continue
        words.add(normalized)
    return words


def context_senses_agree_fix_a(requirement_text: str, evidence_text: str, term: str) -> bool:
    req_ctx = schemas._local_context_words(requirement_text, term)
    ev_ctx = widen_evidence_context(evidence_text, term)
    if not req_ctx or not ev_ctx:
        return False
    return bool(req_ctx & ev_ctx)


def context_senses_agree_fix_c(requirement_text: str, evidence_text: str, term: str) -> bool:
    # Original check first (never LOSES a legitimate match the original had).
    if schemas._context_senses_agree(requirement_text, evidence_text, term):
        return True
    req_ctx = schemas._local_context_words(requirement_text, term)
    ev_ctx = schemas._local_context_words(evidence_text, term)
    if not req_ctx or not ev_ctx:
        return False
    return bool((req_ctx & _PHYSICAL_SIGNAL_WORDS) and (ev_ctx & _PHYSICAL_SIGNAL_WORDS))


# Fix (d): fix (c), but with a negative IT/computing counter-signal --
# real bake-off data showed bare "hardware"/"equipment"/"device" overlap
# alone can't tell "installed home appliances" apart from "installed
# computer hardware/software", a genuinely different specific domain this
# candidate hasn't done. A computing-specific word anywhere in the
# REQUIREMENT's own local context vetoes the physical-signal agreement,
# even if a physical-signal word is also present.
_IT_COUNTER_SIGNAL_WORDS = {
    "software", "os", "driver", "drivers", "workstation", "workstations",
    "desktop", "laptop", "ram", "server", "servers", "network", "computing",
    "printer", "printers", "notebook", "notebooks", "webcam", "webcams",
    "helpdesk",
}


def _raw_local_sentence_text(text: str, term: str) -> str:
    """Unlike schemas._local_context_words (which filters generic/ambiguous
    words BEFORE returning), this returns the raw sentence(s) containing the
    term, unfiltered -- needed because "software" is itself classified as a
    generic evidence term (real IDF-based finding, decision #68) and so
    never survives into _local_context_words' output at all, even though
    it's exactly the word an IT-counter-signal check needs to see."""
    return " ".join(
        s for s in schemas._SENTENCE_SPLIT_RE.split(text or "") if schemas._term_in_text(term, s.lower())
    ).lower()


def context_senses_agree_fix_d(requirement_text: str, evidence_text: str, term: str) -> bool:
    if schemas._context_senses_agree(requirement_text, evidence_text, term):
        return True
    req_ctx = schemas._local_context_words(requirement_text, term)
    ev_ctx = schemas._local_context_words(evidence_text, term)
    if not req_ctx or not ev_ctx:
        return False
    raw_req = _raw_local_sentence_text(requirement_text, term)
    if "operating system" in raw_req or any(schemas._term_in_text(w, raw_req) for w in _IT_COUNTER_SIGNAL_WORDS):
        return False
    return bool((req_ctx & _PHYSICAL_SIGNAL_WORDS) and (ev_ctx & _PHYSICAL_SIGNAL_WORDS))


def main():
    db = os.path.expanduser("~/.applypilot/applypilot.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT url, title, full_description FROM jobs WHERE full_description LIKE '%install%' "
        "AND full_description IS NOT NULL"
    ).fetchall()
    sample = random.sample(rows, min(500, len(rows)))

    profile = load_profile()

    alex = next(
        i for i in profile["experience_inventory"] if "Alex Prosperity" in i["name"]
    )
    alex_evidence_text = schemas._evidence_own_text(alex)

    baseline_supported = []
    fix_a_recovered = []
    fix_c_recovered = []
    fix_d_recovered = []

    for row in sample:
        job = {"title": row["title"], "full_description": row["full_description"]}
        schemas.clear_schema_cache()
        lines, _dropped = local_tailor._split_requirement_lines(row["full_description"] or "")
        ranked = local_tailor.rank_profile_evidence(
            {"title": row["title"] or "", "full_description": row["full_description"] or ""}, profile
        )
        alex_idx = next((i for i, w in enumerate(ranked, start=1) if "Alex Prosperity" in w["name"]), None)
        if alex_idx is None:
            continue  # Alex isn't even a whole-job candidate here

        for line in lines:
            text = line["text"] if isinstance(line, dict) else line
            if "install" not in text.lower():
                continue
            resolved = local_tailor._pair_candidate_evidence(text, ranked)
            if alex_idx not in resolved:
                continue  # Alex isn't the literal top match for this specific line

            baseline = schemas._context_senses_agree(text, alex_evidence_text, "installation")
            fix_a = context_senses_agree_fix_a(text, alex_evidence_text, "installation")
            fix_c = context_senses_agree_fix_c(text, alex_evidence_text, "installation")
            fix_d = context_senses_agree_fix_d(text, alex_evidence_text, "installation")

            record = (row["title"], text[:120])
            if baseline:
                baseline_supported.append(record)
            if fix_a and not baseline:
                fix_a_recovered.append(record)
            if fix_c and not baseline:
                fix_c_recovered.append(record)
            if fix_d and not baseline:
                fix_d_recovered.append(record)

    print(f"Sample size: {len(sample)}")
    print(f"Baseline already-supported ('installation' + Alex, sense agrees): {len(baseline_supported)}")
    print(f"\nFix (a) [widen evidence context] newly recovers: {len(fix_a_recovered)}")
    for t, txt in fix_a_recovered[:20]:
        print("  -", t, "|", txt)
    print(f"\nFix (c) [physical-signal-word agreement] newly recovers: {len(fix_c_recovered)}")
    for t, txt in fix_c_recovered[:30]:
        print("  -", t, "|", txt)
    print(f"\nFix (d) [fix c + IT counter-signal veto] newly recovers: {len(fix_d_recovered)}")
    for t, txt in fix_d_recovered[:30]:
        print("  -", t, "|", txt)


if __name__ == "__main__":
    main()
