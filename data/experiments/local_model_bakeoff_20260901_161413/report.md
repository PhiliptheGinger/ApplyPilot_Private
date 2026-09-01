# Local Model Bake-Off (Read-Only)

- Timestamp (UTC): 2026-09-01T16:14:21.275924+00:00
- Jobs tested: 3
- Runnable candidates tested: 1

## Summary Table

| model | exact identifier | size | jobs tested | successful | avg latency | validator pass | judge pass | grounding issues | useful tailoring | omitted evidence | overall availability |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| TinyLlama Resume Tailor (1.1B LoRA) | akshayrinku/tinyllama-resume-tailor-lora | 1.1B base + LoRA adapter | 0 | 0 | N/A ms | 0/0 | 0/0 (judge unavailable) | 0 | 0 | 0 | unavailable/impractical |
| T5-Base resume candidate (~220M) | Abhishek9998/t5-base-finetuned-resumes_t2json_large | ~220M | 0 | 0 | N/A ms | 0/0 | 0/0 (judge unavailable) | 0 | 0 | 0 | unavailable/impractical |
| GPT-Neo resume assistant candidate (~1.3B) | N/A | ~1.3B target | 0 | 0 | N/A ms | 0/0 | 0/0 (judge unavailable) | 0 | 0 | 0 | unavailable/impractical |
| Qwen2.5 Resume LoRA (~7B) | Ella0506/Qwen2.5-7B-Instruct-Resume-LoRA | ~7B base + LoRA adapter | 0 | 0 | N/A ms | 0/0 | 0/0 (judge unavailable) | 0 | 0 | 0 | unavailable/impractical |
| qwen3 baseline | qwen3:1.7b | 2.0B (Q4_K_M, 1.4GB) | 3 | 2 | 55976.3 ms | 0/3 | 0/3 | 0 | 0 | 0 | runnable |

## Recommendations
- TinyLlama Resume Tailor (1.1B LoRA) (akshayrinku/tinyllama-resume-tailor-lora): reject
- T5-Base resume candidate (~220M) (Abhishek9998/t5-base-finetuned-resumes_t2json_large): reject
- GPT-Neo resume assistant candidate (~1.3B) (N/A): reject
- Qwen2.5 Resume LoRA (~7B) (Ella0506/Qwen2.5-7B-Instruct-Resume-LoRA): impractical_on_current_hardware
- qwen3 baseline (qwen3:1.7b): best_local_candidate