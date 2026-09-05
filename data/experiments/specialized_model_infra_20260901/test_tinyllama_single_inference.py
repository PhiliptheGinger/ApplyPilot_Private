"""Single minimal inference test for the verified TinyLlama resume-tailor LoRA.

Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Adapter (exact HF identifier, re-verified via HF API):
  akshayrinku/tinyllama-resume-tailor-lora

RAM is tight on this machine (baseline ~3.15GB free out of 11.79GB total).
1.1B params in fp32 (~4.4GB) will NOT fit. Loading in bfloat16 (~2.2GB)
instead, with low_cpu_mem_usage=True. Adapter is kept SEPARATE from the
base model (PeftModel wrapping, no merge_and_unload()), per instructions.
Not converting to GGUF.

Runs once, then explicitly releases the model before exiting.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_ID = "akshayrinku/tinyllama-resume-tailor-lora"
OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "tinyllama_single_inference_result.json"

SAMPLE_INPUT = (
    "Rewrite this resume bullet to be more concise, using only facts already "
    "present -- do not invent anything:\n"
    "Alignment Technician at National Tire and Battery / Mavis. Performed "
    "vehicle alignment and maintenance tasks including tires, shocks, struts, "
    "brakes, and fluids. Provided hands-on technical troubleshooting and "
    "customer service to ensure vehicle safety and satisfaction."
)


def main() -> None:
    result: dict = {"base_model_id": BASE_MODEL_ID, "adapter_id": ADAPTER_ID, "sample_input": SAMPLE_INPUT}
    dtype = torch.bfloat16
    result["dtype"] = "bfloat16 (fp32 would need ~4.4GB, not available on this machine)"

    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        load_tok_s = time.time() - t0

        t1 = time.time()
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        load_base_s = time.time() - t1

        t2 = time.time()
        # Adapter kept separate from the base model (PeftModel wraps rather
        # than merges) -- per instructions, no merge_and_unload() here.
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        model.eval()
        load_adapter_s = time.time() - t2
    except Exception as exc:  # noqa: BLE001
        result["load_succeeded"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("LOAD_FAILED:", result["error"])
        return

    result["load_succeeded"] = True
    result["load_tokenizer_s"] = round(load_tok_s, 2)
    result["load_base_model_s"] = round(load_base_s, 2)
    result["load_adapter_s"] = round(load_adapter_s, 2)
    result["base_param_count"] = sum(p.numel() for p in base_model.parameters())
    result["adapter_kept_separate"] = True

    t3 = time.time()
    try:
        chat = [{"role": "user", "content": SAMPLE_INPUT}]
        prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        gen_s = time.time() - t3
        result["inference_succeeded"] = True
        result["inference_latency_s"] = round(gen_s, 2)
        result["output_text"] = output_text
    except Exception as exc:  # noqa: BLE001
        result["inference_succeeded"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["total_wall_s"] = round(time.time() - t0, 2)

    del model
    del base_model
    del tokenizer
    gc.collect()

    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
