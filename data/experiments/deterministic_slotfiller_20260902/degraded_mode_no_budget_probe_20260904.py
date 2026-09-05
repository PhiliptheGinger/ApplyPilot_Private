"""Diagnostic: reproduce tonight's REAL degraded-mode realization failure
("Greeter / Counter Desk Attendant", max_tokens=600 -> empty content) with
a much larger token budget, and actually LOOK at the raw output -- not just
success/failure -- to tell "it needed more room" apart from "it loops/
repeats and never converges." Uses the exact same prompt-building function
production uses (_build_realization_prompt), just calls client.chat()
directly instead of going through request_local_realization, so the raw
text is visible instead of being parsed-and-discarded.

2026-09-04, direct follow-up to the real log trace: max_tokens=600, 2452
char prompt, 3 requirements, 57s elapsed, then "Null/empty content from
local/qwen3:1.7b". Starting at max_tokens=4096 (~7x) rather than truly
unbounded -- qwen3 runs at ~5-6 tok/s on this CPU-only machine per earlier
logged timings, so an actually-unbounded budget risks a call running the
better part of an hour if the model genuinely never converges.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ["LLM_URL"] = "http://localhost:11434/v1"
os.environ["LLM_MODEL_QUALITY"] = "qwen3:1.7b"
os.environ["APPLYPILOT_LOCAL_LLM_URL"] = "http://localhost:11434/v1"
os.environ["APPLYPILOT_LOCAL_LLM_MODEL"] = "qwen3:1.7b"

PROBE_MAX_TOKENS = 4096

from applypilot import config, database, llm  # noqa: E402
from applypilot.scoring.local_tailor import _build_realization_prompt  # noqa: E402
from applypilot.scoring.schemas import build_job_schema_representation  # noqa: E402

JOB_TITLE_LIKE = "Greeter / Counter Desk Attendant"


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    row = conn.execute("SELECT * FROM jobs WHERE title = ? ORDER BY discovered_at DESC LIMIT 1", (JOB_TITLE_LIKE,)).fetchone()
    if row is None:
        raise SystemExit(f"job not found: {JOB_TITLE_LIKE}")
    job = dict(row)

    schema = build_job_schema_representation(job, profile)
    prompt = _build_realization_prompt(schema)
    if prompt is None:
        raise SystemExit("no schema-supported requirements -- nothing to realize (different failure than tonight's)")
    system, user = prompt
    print(f"prompt: {len(system) + len(user)} chars (~{(len(system) + len(user)) // 4} tokens)")
    print(f"max_tokens for this probe: {PROBE_MAX_TOKENS} (production used 600 and got empty content)")

    client = llm.get_client(quality=True)

    t0 = time.time()
    try:
        raw = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=PROBE_MAX_TOKENS,
            temperature=0.3,
        )
        elapsed = time.time() - t0
        print(f"\ncall finished in {elapsed:.1f}s, model used: {client.last_model_used}")
        print(f"raw response length: {len(raw)} chars")
        print("\n=== FULL RAW RESPONSE ===")
        print(raw)
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\ncall FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
