import re
import sys

sys.path.insert(0, "src")
from applypilot import database

OLD = re.compile(r"\bTS[/\s-]?SCI\b", re.IGNORECASE)
NEW = re.compile(
    r"\bTS[/\s-]?SCI\b" r"|\btop\s+secret\s*/?\s*sci\b" r"|\btop\s+secret\s+clearance\s+with\s+sci\b",
    re.IGNORECASE,
)

conn = database.get_connection()
rows = conn.execute("SELECT url, full_description, fit_score, state FROM jobs WHERE full_description IS NOT NULL").fetchall()

old_hits = 0
new_hits = 0
newly_caught_examples = []
for r in rows:
    desc = r["full_description"] or ""
    o = bool(OLD.search(desc))
    n = bool(NEW.search(desc))
    if o:
        old_hits += 1
    if n:
        new_hits += 1
    if n and not o:
        newly_caught_examples.append((r["url"], r["fit_score"], r["state"]))

print(f"old pattern matches: {old_hits}")
print(f"new pattern matches: {new_hits}")
print(f"newly caught (real gap closed): {len(newly_caught_examples)}")
print()
high_score_newly_caught = [x for x in newly_caught_examples if (x[1] or 0) >= 7]
print(f"of those, currently scored >=7 (real risk -- would proceed toward tailoring): {len(high_score_newly_caught)}")
for u, s, st in high_score_newly_caught[:10]:
    print(f"  score={s} state={st} {u[:80]}")
