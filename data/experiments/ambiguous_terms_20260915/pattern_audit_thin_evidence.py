"""2026-09-15: user asked, after decision #143/#144's single confirmed
false negative (Alex Prosperity Group / "installation" on a real Machine
Installation Technician job -- fixed same day via schemas.py's new
physical-signal fallback, see the module docstring there), whether this is
a one-off or a real recurring PATTERN worth understanding before chasing
more individual fixes.

Method: _context_senses_agree's conservative "no shared context word = not
agreeing" rule is most likely to misfire when an evidence item's own text
is THIN (few words to draw context from at all) -- Alex Prosperity Group
was the thinnest experience_inventory item BEFORE this session's data
cleanup. A direct scan of every profile item's evidence text for
_AMBIGUOUS_TERMS membership, ranked by text length, finds three more
candidates just as thin or thinner:
  - Freelance Photography / Videography (106 chars) -- "equipment"
  - You Power You (130 chars) -- "servers"
  - I_hate_social_media (161 chars) -- "media"

This script generalizes the original Alex-Prosperity-Group-specific
bake-off (data/experiments/ambiguous_terms_20260913/) to these three, using
the SAME "is this item the literal top candidate for a real job's
requirement line containing the term" methodology, and reports the
True/False split plus real sample text for manual judgment.
"""

import random
import re
import sqlite3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from applypilot.scoring import local_tailor, schemas  # noqa: E402
from applypilot.config import load_profile  # noqa: E402

random.seed(20260915)

CASES = [
    ("Freelance Photography / Videography", "equipment"),
    ("You Power You", "servers"),
    ("I_hate_social_media", "media"),
]


def find_item(profile, name):
    for section in ("experience_inventory", "project_inventory", "skills_inventory", "certifications"):
        for item in profile.get(section, []):
            if item.get("name") == name:
                return item
    raise KeyError(name)


def main():
    db = os.path.expanduser("~/.applypilot/applypilot.db")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    profile = load_profile()

    for item_name, term in CASES:
        item = find_item(profile, item_name)
        evidence_text = schemas._evidence_own_text(item)
        print("=" * 90)
        print(f"ITEM: {item_name!r}  TERM: {term!r}")
        print(f"evidence_text ({len(evidence_text)} chars): {evidence_text!r}")

        rows = conn.execute(
            "SELECT url, title, full_description FROM jobs WHERE full_description LIKE ? "
            "AND full_description IS NOT NULL",
            (f"%{term}%",),
        ).fetchall()
        sample = random.sample(rows, min(500, len(rows)))

        agree_true = []
        agree_false = []

        for row in sample:
            schemas.clear_schema_cache()
            lines, _dropped = local_tailor._split_requirement_lines(row["full_description"] or "")
            ranked = local_tailor.rank_profile_evidence(
                {"title": row["title"] or "", "full_description": row["full_description"] or ""}, profile
            )
            item_idx = next((i for i, w in enumerate(ranked, start=1) if w["name"] == item_name), None)
            if item_idx is None:
                continue

            for line in lines:
                text = line["text"] if isinstance(line, dict) else line
                if not re.search(re.escape(term), text, re.IGNORECASE):
                    continue
                resolved = local_tailor._pair_candidate_evidence(text, ranked)
                if item_idx not in resolved:
                    continue

                agrees = schemas._context_senses_agree(text, evidence_text, term)
                record = (row["title"], text[:140])
                (agree_true if agrees else agree_false).append(record)

        total = len(agree_true) + len(agree_false)
        print(f"sample size: {len(sample)} | requirement-line matches where item is top candidate: {total}")
        print(f"  agree=True (kept as supported): {len(agree_true)}")
        print(f"  agree=False (dropped):          {len(agree_false)}")
        print("  -- dropped (False) samples for manual review --")
        for t, txt in agree_false[:15]:
            print("    -", t, "|", txt)
        print("  -- kept (True) samples for manual review --")
        for t, txt in agree_true[:8]:
            print("    -", t, "|", txt)
        print()


if __name__ == "__main__":
    main()
