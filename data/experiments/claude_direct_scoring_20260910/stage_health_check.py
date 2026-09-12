import sys
sys.path.insert(0, "src")
from applypilot import database

conn = database.get_connection()

print("=== Cumulative tailoring/cover output (ever produced) ===")
row = conn.execute("SELECT COUNT(*) c FROM jobs WHERE tailored_resume_path IS NOT NULL").fetchone()
print("  jobs with a tailored resume ever generated:", row["c"])
row = conn.execute("SELECT COUNT(*) c FROM jobs WHERE cover_letter_path IS NOT NULL").fetchone()
print("  jobs with a cover letter ever generated:", row["c"])

print("\n=== Tailor outcome breakdown (jobs with fit_score >= 8, tailor_attempts > 0) ===")
for r in conn.execute(
    "SELECT state, COUNT(*) c FROM jobs WHERE tailor_attempts > 0 GROUP BY state ORDER BY c DESC"
):
    print(f"  {r['state']}: {r['c']}")

print("\n=== Recent tailor/cover activity timestamps ===")
row = conn.execute("SELECT MAX(scored_at) m FROM jobs WHERE score_method IS NULL OR score_method NOT IN ('claude_direct','deterministic_fallback')").fetchone()
print("  most recent real-LLM score:", row["m"])
row = conn.execute("SELECT tailored_resume_path, scored_at FROM jobs WHERE tailored_resume_path IS NOT NULL ORDER BY rowid DESC LIMIT 3").fetchall()
for r in row:
    print("  recent tailored job scored_at:", r["scored_at"])

print("\n=== score_error / stuck-job check ===")
row = conn.execute("SELECT COUNT(*) c FROM jobs WHERE score_attempts >= 5 AND fit_score IS NULL").fetchone()
print("  jobs permanently score_failed (maxed retries):", row["c"])
