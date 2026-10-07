"""Validate the FW28 requirement-framing classifier against the Claude-direct
audit set (decision #236). Read-only: opens the DB in read-only mode.

For every job scored by Claude directly (score_method='claude_direct',
decisions #94-#128), compare what the local scorer's years extraction does
today with what the classifier would add:

- A job is CHANGED when today's extractor finds no years requirement but the
  classifier finds one (or, for software postings, implied seniority).
- A change AGREES with Claude when Claude's audited score is at or below the
  cap the change would impose (a requirement of 1 year caps at 7; 2+ years at
  5; implied seniority at 3). If Claude scored the job higher, the classifier
  would have lowered a score Claude judged good: a disagreement.

Pass = at least 98% of changed jobs agree. Then set
APPLYPILOT_FRAMING_CLASSIFIER=on in ~/.applypilot/.env.

Also lists jobs where today's extractor finds years but the classifier reads
every mention as preferred, about the company, or not about experience: candidate false positives in
the old extractor, for a human to read (the classifier never removes them).

Usage: python scripts/validate_requirement_framing.py [--db PATH] [--show N]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _cap_for(years: int) -> int:
    return 7 if years <= 1 else 5


def main() -> int:
    from applypilot import config
    from applypilot.scoring.deterministic_fallback import extract_years_required
    from applypilot.scoring.requirement_framing import (
        PREFERRED,
        UNRELATED,
        classify_years_mentions,
        framed_years_required,
        implicit_seniority,
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--show", type=int, default=25, help="how many examples to print per list")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{Path(args.db).as_posix()}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT url, title, fit_score, score_reasoning, full_description FROM jobs "
        "WHERE score_method = 'claude_direct' AND fit_score IS NOT NULL AND full_description IS NOT NULL"
    ).fetchall()
    if not rows:
        print("No claude_direct jobs found in", args.db)
        return 2

    changed, disagree, old_suspect = [], [], []
    for url, title, claude_score, reasoning, desc in rows:
        old = extract_years_required(desc)
        if old is None:
            framed = framed_years_required(desc)
            senior = implicit_seniority(desc)
            if framed is not None:
                cap, why = _cap_for(framed), f"years_required={framed}"
            elif len(senior) >= 2:
                cap, why = 3, f"implied seniority ({', '.join(senior)})"
            else:
                continue
            item = (url, title, claude_score, cap, why, (reasoning or "")[:200])
            changed.append(item)
            if claude_score > cap:
                disagree.append(item)
        else:
            mentions = classify_years_mentions(desc)
            contradicted = all(m.frame in (PREFERRED, UNRELATED) or "company" in m.signals for m in mentions)
            if mentions and contradicted:
                old_suspect.append((url, title, claude_score, old, [(m.years, m.frame) for m in mentions][:5]))

    agree = len(changed) - len(disagree)
    rate = agree / len(changed) if changed else 1.0
    print(f"claude_direct jobs: {len(rows)}")
    print(f"changed by classifier: {len(changed)} ({len(changed) / len(rows):.1%})")
    print(f"agree with Claude: {agree}/{len(changed)} = {rate:.1%}  (pass needs >= 98%)")
    print()
    print(f"Disagreements (Claude scored above the cap), first {args.show}:")
    for url, title, score, cap, why, reasoning in disagree[: args.show]:
        print(f"  claude={score} cap={cap} {why} | {title} | {url}\n      {reasoning}")
    print()
    print(f"Changes that agree, first {args.show}:")
    for url, title, score, cap, why, _r in [c for c in changed if c not in disagree][: args.show]:
        print(f"  claude={score} cap={cap} {why} | {title}")
    print()
    print(f"Old extractor finds years but classifier contradicts it: {len(old_suspect)}; first {args.show}:")
    for url, title, score, old, ms in old_suspect[: args.show]:
        print(f"  old={old} claude={score} mentions={ms} | {title} | {url}")
    return 0 if rate >= 0.98 else 1


if __name__ == "__main__":
    sys.exit(main())
