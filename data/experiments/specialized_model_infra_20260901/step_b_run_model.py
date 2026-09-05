"""STEP B (runs in the ISOLATED experiment venv): feed the SAME prompts
built by step_a to a specialized model, one model per process run so only
one model is ever resident in memory at a time. Loads once, runs all
jobs, then explicitly releases.

Usage: python step_b_run_model.py t5|tinyllama
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

OUT_DIR = Path(__file__).resolve().parent
PROMPTS = json.loads((OUT_DIR / "shared_prompts.json").read_text(encoding="utf-8"))

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else ""
if MODEL_KEY not in ("t5", "tinyllama"):
    print("Usage: step_b_run_model.py t5|tinyllama")
    sys.exit(1)


def run_t5(system: str, user: str) -> str:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "Abhishek9998/t5-base-finetuned-resumes_t2json_large"
    tok = getattr(run_t5, "_tok", None) or AutoTokenizer.from_pretrained(model_id)
    mdl = getattr(run_t5, "_mdl", None)
    if mdl is None:
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch.float32, low_cpu_mem_usage=True)
        mdl.eval()
        run_t5._mdl = mdl
        run_t5._tok = tok
    combined = f"{system}\n\n{user}"
    inputs = tok(combined, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out_ids = mdl.generate(**inputs, max_new_tokens=256, num_beams=1, do_sample=False)
    return tok.decode(out_ids[0], skip_special_tokens=True)


def run_tinyllama(system: str, user: str) -> str:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_id = "akshayrinku/tinyllama-resume-tailor-lora"
    tok = getattr(run_tinyllama, "_tok", None)
    mdl = getattr(run_tinyllama, "_mdl", None)
    if mdl is None:
        tok = AutoTokenizer.from_pretrained(base_id)
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        mdl = PeftModel.from_pretrained(base, adapter_id)
        mdl.eval()
        run_tinyllama._mdl = mdl
        run_tinyllama._tok = tok
    chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 -- some chat templates reject a system role
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}], tokenize=False, add_generation_prompt=True
        )
    inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs, max_new_tokens=600, do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id
        )
    new_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True)


RUNNER = {"t5": run_t5, "tinyllama": run_tinyllama}[MODEL_KEY]

results: dict = {"model_key": MODEL_KEY, "jobs": {}}
t_load0 = time.time()
for job_id, entry in PROMPTS["jobs"].items():
    prompt = entry.get("prompt")
    if prompt is None:
        results["jobs"][job_id] = {"skipped": "no_supported_evidence (matches production gate -- model not invoked)"}
        print(f"job {job_id}: skipped (no supported requirements)")
        continue
    t0 = time.time()
    try:
        raw = RUNNER(prompt["system"], prompt["user"])
        latency = time.time() - t0
        results["jobs"][job_id] = {"raw_output": raw, "latency_s": round(latency, 2), "succeeded": True}
        print(f"job {job_id}: ok, {latency:.1f}s, {len(raw)} chars")
    except Exception as exc:  # noqa: BLE001
        results["jobs"][job_id] = {"succeeded": False, "error": f"{type(exc).__name__}: {exc}", "latency_s": round(time.time() - t0, 2)}
        print(f"job {job_id}: FAILED: {exc}")

results["total_wall_s"] = round(time.time() - t_load0, 2)

(OUT_DIR / f"raw_output_{MODEL_KEY}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

gc.collect()
print("DONE")
