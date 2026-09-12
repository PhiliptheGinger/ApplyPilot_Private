import sys
sys.path.insert(0, "src")
from applypilot import database

conn = database.get_connection()
rows = conn.execute(
    "SELECT fit_score, state, COUNT(*) c FROM jobs WHERE score_method = 'claude_direct' "
    "GROUP BY fit_score, state ORDER BY fit_score"
).fetchall()
for r in rows:
    print(dict(r))
total = conn.execute("SELECT COUNT(*) c FROM jobs WHERE score_method = 'claude_direct'").fetchone()
print("total:", total["c"])
