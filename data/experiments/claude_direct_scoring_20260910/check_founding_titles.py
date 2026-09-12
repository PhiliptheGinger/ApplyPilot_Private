import sys

sys.path.insert(0, "src")
from applypilot import database

conn = database.get_connection()
rows = conn.execute(
    "SELECT DISTINCT title FROM jobs WHERE title LIKE '%found%' COLLATE NOCASE LIMIT 100"
).fetchall()
for r in rows:
    print(r["title"])
print("---")
print("total distinct:", len(rows))
