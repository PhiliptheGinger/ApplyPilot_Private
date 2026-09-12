import sys

sys.path.insert(0, "src")
from applypilot import database

conn = database.get_connection()
rows = conn.execute(
    "SELECT url, title, full_description FROM jobs "
    "WHERE full_description LIKE '%LATAM%' LIMIT 25"
).fetchall()
print("bare 'LATAM' matches:", len(rows))
for r in rows:
    idx = r["full_description"].find("LATAM")
    snippet = r["full_description"][max(0, idx - 100) : idx + 100]
    print((r["title"] + " | " + snippet).encode("ascii", "replace").decode())
    print("---")
