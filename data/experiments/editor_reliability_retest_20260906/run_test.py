"""Retest local-model editor reliability (decision #70/#73 follow-up).

Hypothesis: production's edit_sentence_for_requirement (llm.py's OpenAI-compat
path + a `/no_think` text-prefix) is what's unreliable, not the bounded-rewrite
TASK itself -- the 2026-09-01 bake-off proved bounded rewrite is safe and
functional for qwen3:1.7b when called via Ollama's NATIVE /api/chat with the
real `think: false` API flag and num_predict=700.

This script runs the SAME real bank sentence + a real job requirement through
three configurations and reports success/failure + latency for each:
  A) production's actual edit_sentence_for_requirement() unmodified (default
     max_tokens=300) -- reproduces last night's failure if still broken.
  B) production's function but with max_tokens bumped to 700 (still goes
     through the OpenAI-compat /no_think path) -- isolates "is it just budget."
  C) Ollama's native /api/chat with think:false, num_predict=700 -- isolates
     "is it the API surface/thinking-suppression mechanism."

No production code is modified. Real client, real model, real network calls.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from applypilot import config  # noqa: E402

config.load_env()

import httpx  # noqa: E402

from applypilot import llm  # noqa: E402
from applypilot.llm import LLMClient, ModelEntry, local_openai_base_url  # noqa: E402
from applypilot.scoring import local_tailor  # noqa: E402

LOCAL_URL = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", "qwen3:1.7b")


def _local_only_client() -> LLMClient:
    """Pin to ONLY the local entry -- mirrors cli.py's test-local command --
    so a live cloud key can't mask what local actually does."""
    client = LLMClient(LOCAL_URL, LOCAL_MODEL, "", quality=True)
    client._fallback_chain = [ModelEntry(LOCAL_MODEL, "local", local_openai_base_url(LOCAL_URL), "")]
    return client

ORIGINAL_SENTENCE = (
    "Handled manual servicing and work on tires, brakes, shocks, struts, "
    "fluids, and connected vehicle components."
)
REQUIREMENT_TEXT = (
    "Perform routine maintenance and repair on vehicle brake and suspension "
    "systems, ensuring safety and quality standards are met."
)

N_TRIALS = 5


def config_a():
    """Production edit_sentence_for_requirement(), unmodified, default max_tokens=300."""
    client = _local_only_client()
    results = []
    for i in range(N_TRIALS):
        t0 = time.time()
        out = local_tailor.edit_sentence_for_requirement(
            client, ORIGINAL_SENTENCE, REQUIREMENT_TEXT
        )
        dt = time.time() - t0
        results.append({"trial": i + 1, "output": out, "seconds": round(dt, 1)})
        print(f"  [A trial {i+1}] {dt:.1f}s -> {out!r}")
    return results


def config_b():
    """Same production function, max_tokens bumped to 700."""
    client = _local_only_client()
    results = []
    for i in range(N_TRIALS):
        t0 = time.time()
        out = local_tailor.edit_sentence_for_requirement(
            client, ORIGINAL_SENTENCE, REQUIREMENT_TEXT, max_tokens=700
        )
        dt = time.time() - t0
        results.append({"trial": i + 1, "output": out, "seconds": round(dt, 1)})
        print(f"  [B trial {i+1}] {dt:.1f}s -> {out!r}")
    return results


def config_c():
    """Ollama native /api/chat, think:false, num_predict=700 -- bypasses llm.py entirely."""
    system = local_tailor._EDITOR_SYSTEM
    user = (
        f'REQUIREMENT: "{REQUIREMENT_TEXT}"\n\n'
        f'ORIGINAL SENTENCE (reword this only -- do not add anything new):\n"{ORIGINAL_SENTENCE}"'
    )
    results = []
    for i in range(N_TRIALS):
        payload = {
            "model": "qwen3:1.7b",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.3, "num_predict": 700},
        }
        t0 = time.time()
        try:
            resp = httpx.post(
                "http://127.0.0.1:11434/api/chat", json=payload, timeout=180.0
            )
            resp.raise_for_status()
            data = resp.json()
            out = (data.get("message") or {}).get("content", "").strip()
        except Exception as exc:  # noqa: BLE001
            out = f"ERROR: {type(exc).__name__}: {exc}"
        dt = time.time() - t0
        results.append({"trial": i + 1, "output": out, "seconds": round(dt, 1)})
        print(f"  [C trial {i+1}] {dt:.1f}s -> {out!r}")
    return results


if __name__ == "__main__":
    print("=== Config A: production edit_sentence_for_requirement, max_tokens=300 (default) ===")
    a = config_a()
    print("\n=== Config B: production function, max_tokens=700 ===")
    b = config_b()
    print("\n=== Config C: Ollama native /api/chat, think:false, num_predict=700 ===")
    c = config_c()

    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(
        json.dumps({"A_default_300": a, "B_bumped_700": b, "C_native_think_false": c}, indent=2)
    )
    print(f"\nWrote {out_path}")
