"""Single minimal inference test for the verified T5-base resume candidate.

Exact HF identifier (from the original discovery, re-verified via HF API):
  Abhishek9998/t5-base-finetuned-resumes_t2json_large

Goal: prove the model loads and produces SOME output. Not a quality
assessment yet -- that's the bake-off step, gated on this succeeding.
Runs once, then explicitly releases the model before exiting.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "Abhishek9998/t5-base-finetuned-resumes_t2json_large"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "t5_single_inference_result.json"

# A short, real snippet (one experience entry) from the candidate's actual
# resume -- enough to see what the model does with real input, without
# spending the full bake-off budget yet.
SAMPLE_INPUT = (
    "Alignment Technician at National Tire and Battery / Mavis. "
    "Performed vehicle alignment and maintenance tasks including tires, "
    "shocks, struts, brakes, and fluids. Provided hands-on technical "
    "troubleshooting and customer service to ensure vehicle safety and "
    "satisfaction."
)


def main() -> None:
    result: dict = {"model_id": MODEL_ID, "sample_input": SAMPLE_INPUT}

    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        load_tok_s = time.time() - t0

        t1 = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        load_model_s = time.time() - t1
    except Exception as exc:  # noqa: BLE001
        result["load_succeeded"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("LOAD_FAILED:", result["error"])
        return

    result["load_succeeded"] = True
    result["load_tokenizer_s"] = round(load_tok_s, 2)
    result["load_model_s"] = round(load_model_s, 2)
    result["param_count"] = sum(p.numel() for p in model.parameters())

    t2 = time.time()
    try:
        inputs = tokenizer(SAMPLE_INPUT, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, num_beams=1, do_sample=False)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        gen_s = time.time() - t2
        result["inference_succeeded"] = True
        result["inference_latency_s"] = round(gen_s, 2)
        result["output_text"] = output_text
        result["output_looks_like_json"] = output_text.strip().startswith("{")
    except Exception as exc:  # noqa: BLE001
        result["inference_succeeded"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["total_wall_s"] = round(time.time() - t0, 2)

    # Explicit resource release before process exit, per instructions.
    del model
    del tokenizer
    gc.collect()

    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
