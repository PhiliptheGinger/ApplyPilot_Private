import sys

sys.path.insert(0, "src")
from applypilot import database

conn = database.get_connection()

# How many rows mention "Top Secret" spelled out near "SCI" already?
rows = conn.execute(
    "SELECT url, title, full_description FROM jobs "
    "WHERE full_description LIKE '%Top Secret%SCI%' OR full_description LIKE '%Top Secret/SCI%' LIMIT 20"
).fetchall()
print("Top Secret ... SCI matches:", len(rows))
for r in rows:
    idx = r["full_description"].lower().find("top secret")
    snippet = r["full_description"][max(0, idx - 40) : idx + 120]
    print((r["title"] + " | " + snippet).encode("ascii", "replace").decode())
    print("---")

# Bare "Top Secret" without SCI -- check for false-positive risk (e.g. "Secret" tier without SCI, which
# the existing rubric explicitly treats as a softer "ability to obtain" case, not a bare disqualifier)
rows2 = conn.execute("SELECT COUNT(*) c FROM jobs WHERE full_description LIKE '%Top Secret%'").fetchone()
print("\ntotal rows mentioning 'Top Secret' at all:", rows2["c"])
