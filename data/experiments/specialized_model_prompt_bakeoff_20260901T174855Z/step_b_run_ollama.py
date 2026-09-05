"""STEP B: feed the shared prompt variants (step_a) to a single Ollama
model, one model per process invocation. All four candidates this round
are Ollama/GGUF, so this talks to the local Ollama HTTP API directly --
no torch/transformers needed for this experiment.

Usage: python step_b_run_ollama.py <ollama_model_name> <output_key>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

OUT_DIR = Path(__file__).resolve().parent
VARIANTS = json.loads((OUT_DIR / "shared_variants.json").read_text(encoding="utf-8"))

OLLAMA_MODEL = sys.argv[1]
OUTPUT_KEY = sys.argv[2]
TEMPERATURE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3


def call_ollama(system: str, user: str, max_tokens: int = 700) -> tuple[str, dict]:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
    }
    t0 = time.time()
    resp = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=600.0)
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.time() - t0
    content = (data.get("message") or {}).get("content", "")
    meta = {
        "latency_s": round(elapsed, 2),
        "eval_count": data.get("eval_count"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "total_duration_ns": data.get("total_duration"),
    }
    return content, meta


def main() -> None:
    results: dict = {"ollama_model": OLLAMA_MODEL, "temperature": TEMPERATURE, "jobs": {}}
    for job_id, entry in VARIANTS["jobs"].items():
        job_results = {}
        for config_name, prompt in entry["configs"].items():
            if prompt is None:
                job_results[config_name] = {"skipped": "no prompt built (e.g. no supported requirements)"}
                print(f"job {job_id} / {config_name}: skipped")
                continue
            try:
                content, meta = call_ollama(prompt["system"], prompt["user"])
                job_results[config_name] = {"raw_output": content, **meta, "succeeded": True}
                print(f"job {job_id} / {config_name}: ok, {meta['latency_s']}s, {len(content)} chars")
            except Exception as exc:  # noqa: BLE001
                job_results[config_name] = {"succeeded": False, "error": f"{type(exc).__name__}: {exc}"}
                print(f"job {job_id} / {config_name}: FAILED: {exc}")
        results["jobs"][job_id] = job_results

    (OUT_DIR / f"raw_output_{OUTPUT_KEY}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
