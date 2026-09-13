"""One-off correction: re-score PNC/Tempus 'Software Engineer Associate' against
current scoring standards (deterministic_combine: family=software_engineering,
years=None, cs_degree=False -> 5), since it predates decisions #82-131's scorer
fixes and no longer qualifies for the funnel (min_score=8).

Same pattern as data/experiments/gemini_vs_local_20260906.../fix_years_extraction_20260909.py.
"""

import datetime
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from applypilot import database  # noqa: E402

URL = "https://builtin.com/job/software-engineer-associate-tempus/10748275"
NEW_SCORE = 5


def main():
    db_path = os.path.expanduser("~/.applypilot/applypilot.db")
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT fit_score, score_reasoning, state FROM jobs WHERE url=?", (URL,)
    ).fetchone()
    if row is None:
        print("Job not found.")
        return
    old_score, old_reasoning, old_state = row
    print(f"Before: fit_score={old_score} state={old_state}")

    new_reasoning = (old_reasoning or "") + (
        "\n\n-- CORRECTED 2026-09-13: re-applied current scoring standards "
        "(deterministic_combine: family=software_engineering, years=None, "
        "cs_degree=False -> 5). No explicit years/degree requirement stated, "
        "but candidate has no professional SWE background beyond personal "
        "projects; per decisions #82-131, software_engineering-family "
        f"postings without a strong domain-fit signal score below the "
        f"funnel threshold under current standards. Old score: {old_score}."
    )
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "UPDATE jobs SET fit_score=?, score_reasoning=?, scored_at=? WHERE url=?",
        (NEW_SCORE, new_reasoning, now, URL),
    )
    database.transition_state(
        conn,
        URL,
        "low_score",
        reason="corrected_score_reapplied_current_standards: 8 -> 5 (software_engineering, no explicit requirement, weak domain fit)",
        metadata={"score": NEW_SCORE},
        force=True,
    )
    conn.commit()

    row2 = conn.execute("SELECT fit_score, state FROM jobs WHERE url=?", (URL,)).fetchone()
    print(f"After: fit_score={row2[0]} state={row2[1]}")


if __name__ == "__main__":
    main()
