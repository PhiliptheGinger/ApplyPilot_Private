"""Deterministic slot-filler v6: replace v5's raw-keyword-COUNT gate with
an IDF-WEIGHTED-score gate, using idf_weights.json (computed by
compute_idf_weights.py from this project's own ~29K-row local jobs corpus).

v5's gate (category_tier == "prototype", i.e. >=2 matched exact_keywords)
failed its own validation: it blocked 0/3 hits, including both known-bad
ones, because a generic connector word ("using") counts the same as a
specific one ("alignment") under raw counting. The fix scoped afterward:
weight each matched keyword by its corpus rarity (IDF) instead of counting
it as 1, and gate on a weighted-score threshold instead of a raw count.

This script tests that fix honestly, including checking whether it
actually resolves the two known-bad hits from v5 -- not assumed, verified.
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
IDF_WEIGHTS: dict[str, float] = json.loads((OUT_DIR / "idf_weights.json").read_text(encoding="utf-8"))
_DEFAULT_IDF = 1.0  # a term never seen in the corpus gets the lowest possible weight, not the highest

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_SYNONYM_SUBSTITUTIONS: dict[str, tuple[re.Pattern, str]] = {
    "problems": (re.compile(r"\bissues\b", re.IGNORECASE), "problems"),
}


def _idf(term: str) -> float:
    return IDF_WEIGHTS.get(term.lower(), _DEFAULT_IDF)


def _weighted_score(exact_keywords: list[str]) -> float:
    return sum(_idf(kw) for kw in exact_keywords)


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


def _apply_idf_gated_substitution(
    sentence: str, exact_keywords: list[str], threshold: float
) -> tuple[str, list[str], float]:
    weighted = _weighted_score(exact_keywords)
    if weighted < threshold:
        return sentence, [], weighted
    applied = []
    sl = sentence.lower()
    for kw in exact_keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", sl):
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
    return sentence, applied, weighted


def _safety_check(new_sentence: str, evidence_item: dict) -> dict:
    claim_ceiling = claim_ceiling_for_evidence(evidence_item)
    agency_ceiling = agency_ceiling_for_evidence(evidence_item)
    cs = check_claim_strength(new_sentence, claim_ceiling)
    ags = check_agency_strength(new_sentence, agency_ceiling)
    return {"claim_ok": cs["passed"], "agency_ok": ags["passed"]}


def _would_ungated_fire(sentence: str, exact_keywords: list[str]) -> bool:
    sl = sentence.lower()
    for kw in exact_keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", sl):
            continue
        sub = _SYNONYM_SUBSTITUTIONS.get(kw.lower())
        if sub and sub[0].search(sentence):
            return True
    return False


# The 3 hits v5's raw-count gate let through, for the explicit check below.
_KNOWN_BAD = {
    ("Forward Deployed Software Engineer, New Grad - US", "problems", "using"),
    ("Principal Product Manager, Enterprise Applications", "alignment", "problems"),
    ("Product Support Specialist", "customer", "issues", "problems"),
}


def collect(threshold: float, job_rows: list[dict], profile: dict) -> dict:
    n = 0
    applied_examples: list[dict] = []
    blocked: list[dict] = []
    would_fire = 0

    for job in job_rows:
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

            best_s, best_sc = None, -1
            for s in _candidate_sentences(item):
                sc = _literal_score(s, kws)
                if sc > best_sc or (sc == best_sc and best_s is not None and len(s) > len(best_s)):
                    best_s, best_sc = s, sc
            if best_s is None:
                continue
            n += 1

            ungated = _would_ungated_fire(best_s, kws)
            if ungated:
                would_fire += 1

            new_sentence, applied, weighted = _apply_idf_gated_substitution(best_s, kws, threshold)
            if applied:
                safety = _safety_check(new_sentence, item)
                if not (safety["claim_ok"] and safety["agency_ok"]):
                    applied = []
                else:
                    applied_examples.append(
                        {
                            "title": job["title"][:55],
                            "requirement": r["requirement"][:90],
                            "exact_keywords": kws,
                            "weighted_score": round(weighted, 3),
                            "before": best_s,
                            "after": new_sentence,
                        }
                    )
            if ungated and not applied:
                blocked.append(
                    {
                        "title": job["title"][:55],
                        "exact_keywords": kws,
                        "weighted_score": round(weighted, 3),
                    }
                )

    return {"n": n, "would_fire": would_fire, "applied": applied_examples, "blocked": blocked}


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
    job_rows = []
    for job_id in job_ids:
        row = conn.execute(
            "SELECT rowid, url, title, company, site, location, full_description FROM jobs WHERE rowid=?", (job_id,)
        ).fetchone()
        if row is not None:
            job_rows.append(dict(row))

    # Calibrate: try a few thresholds and report how each does against the
    # known-bad set vs. overall yield, rather than pick one blindly.
    for threshold in (3.0, 4.0, 5.0, 6.0, 7.0):
        result = collect(threshold, job_rows, profile)
        bad_still_through = [
            ex
            for ex in result["applied"]
            if any(set(ex["exact_keywords"]) == set(bad[1:]) and ex["title"].startswith(bad[0][:20]) for bad in _KNOWN_BAD)
        ]
        print(
            f"threshold={threshold:.1f}  would_fire_ungated={result['would_fire']}  "
            f"applied={len(result['applied'])}  blocked={len(result['blocked'])}  "
            f"known_bad_still_through={len(bad_still_through)}"
        )

    print("\n=== detail at threshold=5.0 ===")
    result = collect(5.0, job_rows, profile)
    for ex in result["applied"]:
        print(f"[{ex['title']}] weighted_score={ex['weighted_score']} keywords={ex['exact_keywords']}")
        print(f"  before: {ex['before']}")
        print(f"  after:  {ex['after']}")
    print(f"\nblocked ({len(result['blocked'])}):")
    for ex in result["blocked"]:
        print(f"  [{ex['title']}] weighted_score={ex['weighted_score']} keywords={ex['exact_keywords']}")

    (OUT_DIR / "prototype_v6_idf_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nwrote prototype_v6_idf_results.json")


if __name__ == "__main__":
    main()
