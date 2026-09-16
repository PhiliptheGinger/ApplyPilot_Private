"""2026-09-16: real audit of the no_supported_evidence bucket, requested
after a real "Coffee and Tea Equipment Tech" case surfaced a genuine
requirement-extraction quality bug (garbled/truncated lines, company
boilerplate mistaken for requirements) rather than a semantic-threshold
issue specifically. This script categorizes a real sample of recent
no_supported_evidence jobs by likely root cause, surfacing diagnostic
signals (extraction quality, candidate nomination, near-threshold scores)
for manual review rather than trying to fully automate the judgment call.
"""

import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from applypilot import config  # noqa: E402

config.load_env()

from applypilot.config import load_profile  # noqa: E402
from applypilot.scoring import local_tailor, schemas  # noqa: E402


def looks_garbled(text: str) -> bool:
    """Heuristic: no ending punctuation and doesn't start with a capital
    letter mid-sentence-style, or ends mid-word/mid-clause with a comma."""
    t = text.strip()
    if not t:
        return False
    if t.endswith((",", "-", "and", "or", "the", "to", "of", "a", "an")):
        return True
    if re.match(r"^(our company|about us|who we are|company overview)\b", t, re.IGNORECASE):
        return True
    return False


def main():
    urls = json.load(open(os.path.join(os.path.dirname(__file__), "..", "no_evidence_audit_20260916_urls.json")))
    db = os.path.expanduser("~/.applypilot/applypilot.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    profile = load_profile()

    for url in urls:
        row = conn.execute("SELECT * FROM jobs WHERE url = ?", (url,)).fetchone()
        if row is None:
            continue
        job = dict(row)
        schemas.clear_schema_cache()
        try:
            rep = schemas.build_job_schema_representation(job, profile)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR building schema for {job['title']!r}: {exc}")
            continue

        reqs = rep["requirements"]
        garbled_count = sum(1 for r in reqs if looks_garbled(r["requirement"]))
        any_candidate = any(r.get("resume_evidence") for r in reqs)
        ranked = local_tailor.rank_profile_evidence(
            {"title": job["title"] or "", "full_description": job["full_description"] or ""}, profile
        )
        top_candidates = ", ".join(f"{w['name']}({w['score']})" for w in ranked[:3]) if ranked else "NONE"

        flag = "?"
        if garbled_count >= 2:
            flag = "EXTRACTION_QUALITY"
        elif not ranked:
            flag = "NO_CANDIDATE_AT_ALL"
        elif any_candidate:
            flag = "CANDIDATE_FOUND_BUT_UNSUPPORTED"
        else:
            flag = "CLEAN_NO_MATCH"

        print(f"[{flag}] {job['title'][:55]:55s} | reqs={len(reqs)} garbled={garbled_count} | top={top_candidates}")


if __name__ == "__main__":
    main()
