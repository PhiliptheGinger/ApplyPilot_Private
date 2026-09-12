import sys
sys.path.insert(0, "src")
from applypilot import database

conn = database.get_connection()

print("=== State distribution (all jobs) ===")
for r in conn.execute("SELECT state, COUNT(*) c FROM jobs GROUP BY state ORDER BY c DESC"):
    print(f"  {r['state']:20s} {r['c']}")

print("\n=== Score method breakdown (fit_score >= 8, non-archived) ===")
for r in conn.execute(
    "SELECT score_method, COUNT(*) c FROM jobs WHERE fit_score >= 8 AND state != 'archived' "
    "GROUP BY score_method ORDER BY c DESC"
):
    print(f"  {r['score_method']}: {r['c']}")

print("\n=== Age of pending-score backlog ===")
row = conn.execute(
    "SELECT COUNT(*) c FROM jobs WHERE state='discovered' OR (state='enriched' AND fit_score IS NULL)"
).fetchone()
print("  total unscored (any age):", row["c"])
row = conn.execute(
    "SELECT COUNT(*) c FROM jobs WHERE (state='discovered' OR (state='enriched' AND fit_score IS NULL)) "
    "AND discovered_at >= datetime('now', '-14 days')"
).fetchone()
print("  unscored within 14-day funnel window:", row["c"])

print("\n=== Recent discovery rate (last 7 days) ===")
row = conn.execute(
    "SELECT COUNT(*) c FROM jobs WHERE discovered_at >= datetime('now', '-7 days')"
).fetchone()
print("  new jobs discovered in last 7 days:", row["c"])

print("\n=== Apply-stage activity ===")
for r in conn.execute(
    "SELECT apply_status, COUNT(*) c FROM jobs WHERE apply_status IS NOT NULL GROUP BY apply_status ORDER BY c DESC"
):
    print(f"  apply_status={r['apply_status']}: {r['c']}")
row = conn.execute("SELECT MAX(last_attempted_at) m FROM jobs WHERE last_attempted_at IS NOT NULL").fetchone()
print("  most recent apply attempt timestamp:", row["m"])

print("\n=== ready_to_apply / cover_ready queue ===")
for r in conn.execute("SELECT state, COUNT(*) c FROM jobs WHERE state IN ('ready_to_apply','cover_ready','tailored') GROUP BY state"):
    print(f"  {r['state']}: {r['c']}")
