# Controlled Local Model Bake-Off v2 (Read-Only)

- Timestamp UTC: 2026-09-01T16:19:55.576083+00:00
- Jobs tested: 3 real jobs
- Runnable models tested: 1

| model | exact identifier | size | jobs tested | successful | avg latency | validator pass | judge pass | grounding issues | useful tailoring | omitted evidence | overall |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| TinyLlama Resume Tailor (1.1B LoRA) | akshayrinku/tinyllama-resume-tailor-lora | 1.1B base + LoRA adapter | 0 | 0 | N/A ms | 0/0 | 0/0 | 0 | 0 | 0 | unavailable |
| T5-Base resume candidate (~220M) | Abhishek9998/t5-base-finetuned-resumes_t2json_large | ~220M | 0 | 0 | N/A ms | 0/0 | 0/0 | 0 | 0 | 0 | unavailable |
| GPT-Neo resume assistant candidate (~1.3B) | N/A | ~1.3B target | 0 | 0 | N/A ms | 0/0 | 0/0 | 0 | 0 | 0 | unavailable |
| Qwen2.5 Resume LoRA (~7B) | Ella0506/Qwen2.5-7B-Instruct-Resume-LoRA | ~7B base + LoRA adapter | 0 | 0 | N/A ms | 0/0 | 0/0 | 0 | 0 | 0 | unavailable |
| qwen3 local baseline | qwen3:1.7b | 2.0B Q4_K_M (1.4GB) | 3 | 0 | 369166.0 ms | 0/3 | 0/3 | 0 | 0 | 1 | tested |

## Recommendations
- TinyLlama Resume Tailor (1.1B LoRA) (akshayrinku/tinyllama-resume-tailor-lora): reject
- T5-Base resume candidate (~220M) (Abhishek9998/t5-base-finetuned-resumes_t2json_large): reject
- GPT-Neo resume assistant candidate (~1.3B) (N/A): reject
- Qwen2.5 Resume LoRA (~7B) (Ella0506/Qwen2.5-7B-Instruct-Resume-LoRA): impractical on current hardware
- qwen3 local baseline (qwen3:1.7b): best local candidate