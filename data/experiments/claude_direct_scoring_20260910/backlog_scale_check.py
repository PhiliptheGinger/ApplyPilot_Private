import sys
sys.path.insert(0, "src")
from applypilot import database

conn = database.get_connection()

row = conn.execute("SELECT COUNT(*) c FROM jobs").fetchone()
print("total jobs ever discovered:", row["c"])

row = conn.execute("SELECT MIN(discovered_at) a, MAX(discovered_at) b FROM jobs").fetchone()
print("discovery date range:", row["a"], "to", row["b"])

row = conn.execute("SELECT COUNT(*) c FROM jobs WHERE full_description IS NOT NULL").fetchone()
print("jobs with a real description (ever enrichable):", row["c"])

# how many claude_direct-scored jobs so far this session
row = conn.execute("SELECT COUNT(*) c FROM jobs WHERE score_method='claude_direct'").fetchone()
print("claude_direct scored so far:", row["c"])

row = conn.execute(
    "SELECT COUNT(*) c FROM jobs WHERE full_description IS NOT NULL AND score_method != 'claude_direct' OR score_method IS NULL"
).fetchone()
print("remaining real-description jobs not yet claude-direct-scored (rough):", row["c"])
