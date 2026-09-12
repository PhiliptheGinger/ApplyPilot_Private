import sys
sys.path.insert(0, "src")
from applypilot import database

conn = database.get_connection()
row = conn.execute(
    "SELECT COUNT(*) c FROM jobs WHERE state IN ('enriched') AND fit_score IS NULL"
).fetchone()
print("pending scoring (enriched, no score):", row["c"])
row2 = conn.execute(
    "SELECT COUNT(*) c FROM jobs WHERE score_method = 'claude_direct'"
).fetchone()
print("already claude-scored:", row2["c"])
