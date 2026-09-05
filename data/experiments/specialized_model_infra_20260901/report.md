# Specialized Small-Model Infrastructure Experiment (Controlled)

Goal: make the two realistically small specialized resume-tailoring candidates (T5-base, TinyLlama-1.1B+LoRA) actually runnable, in a fully isolated environment, and test them fairly against the same jobs/harness as qwen3:1.7b. Qwen2.5-7B intentionally excluded (too large for this hardware, per instructions).

**Bottom line: infrastructure succeeded (both models install, load, and run without destabilizing the machine). Neither model is usable for the tailoring task. Both perform worse than the already-rejected qwen3:1.7b generic baseline.**

## Baseline (before installing anything)

| | |
|---|---|
| Total RAM | 11.79 GB |
| Free RAM at start | 3.15 GB (tight -- flagged before proceeding) |
| Disk free (C:) | 557.95 GB (ample) |
| Python | 3.11.9 |
| Production venv ML packages | none (torch/transformers/peft/accelerate/sentencepiece all absent -- confirmed clean) |
| Ollama | 0.33.2, models: all-minilm, qwen3:1.7b, qwen3:8b |
| Git status | clean relative to session start |

## Isolation

New venv created at `data/experiments/specialized_model_infra_20260901/venv` from a fresh `py -3.11` interpreter (independent of the production venv). Named literally `venv` so the repo's existing `.gitignore` rule (`venv/`, matches at any depth) excludes it automatically -- verified, zero git noise from thousands of installed package files. **Production venv was never touched; no pinned dependency changed.**

Installed (CPU-only, confirmed no CUDA wheel pulled): `torch==2.14.0+cpu`, `transformers==5.16.1`, `peft==0.20.0`, `accelerate==1.14.0`, `sentencepiece==0.2.2`, `numpy==2.4.6`. Both installs completed without incident; RAM returned to baseline after each (no lasting footprint from install itself).

## Stability proof (single inference per model, per the guardrails)

| | T5-base | TinyLlama+LoRA |
|---|---|---|
| Exact identifier | `Abhishek9998/t5-base-finetuned-resumes_t2json_large` | base `TinyLlama/TinyLlama-1.1B-Chat-v1.0` + adapter `akshayrinku/tinyllama-resume-tailor-lora` (kept separate, not merged) |
| Params | 222,903,552 (confirms "~220M") | 1,103,112,192 base (confirms "~1.1B") |
| dtype | float32 (fits comfortably) | bfloat16 (fp32 would need ~4.4GB, not available) |
| First load (incl. download) | 43.5s (tokenizer) + model | 218.3s base + 3.7s adapter (one-time download cost) |
| Single inference | 4.5s, succeeded | 28.8s, succeeded, output coherent (rephrased input, added no false facts) |
| RAM after process exit | back to baseline exactly | back to baseline exactly |
| **Verdict** | **stable** | **stable** |

No memory pressure, thrashing, or instability observed at any point. Both cleared the stability gate, so the bake-off proceeded per your instructions.

## Bake-off (production-identical harness)

To keep this rigorous, the harness reused **actual production code**, not a re-implementation:
- Prompts built with `local_tailor._build_realization_prompt(job_schema)` -- the exact same function/prompt qwen3 received in the earlier corrected bake-off.
- Same 4 real jobs, same base resume, same profile/evidence catalog as the qwen3 run (job 7406 is correctly gated with zero supported requirements in all three models -- the deterministic evidence layer, not the LLM, makes that call).
- Model output was parsed and safety-checked with the *same* claim-strength/agency/causal/metric-fabrication checks (`schemas.check_claim_strength` etc.) that guard qwen3's output in production, then run through the same `merge_realization` → `validate_json_fields` → `assemble_resume_text` pipeline.
- One model resident in memory at a time; each released before the next loaded.

### Results

| Job | qwen3:1.7b (generic, prior run) | T5-base (specialized) | TinyLlama+LoRA (specialized) |
|---|---|---|---|
| 27323 (Software Engineer Graduate) | Content generated, both bullet+summary rejected by safety checks | Output was the literal string `"True"` -- unparseable | Output parsed as JSON, but both bullet and summary rejected by safety checks (overclaimed relative to evidence) |
| 7267 (Software Development Engineer) | Malformed JSON | `"True"` -- unparseable | Parsed and passed safety checks, but **failed validation**: title-inflation ("engineer" not in candidate's authoritative titles) + base-resume-artifact issues |
| 7406 (Cloud Platform Engineer) | Gated -- zero supported requirements | Gated (same) | Gated (same) |
| 22916 (Hotel Partner Solutions Specialist) | Malformed JSON | `"True"` -- unparseable | Parsed and passed safety checks, but **failed validation** (base-resume-artifact issues); bullet *content* included literal fragments like `"Design"` and `"Individual_contributor"` -- the model emitted the prompt's internal field-label vocabulary as if it were bullet text, not real sentences |

**Successful, validated, usable output: 0/3 testable jobs, for all three local models.** (Job 7406 excluded from the denominator -- correctly gated before any model runs, not a model failure.)

### What actually went wrong, per model

- **T5-base**: total task collapse. Given the real structured realization prompt (not just plain resume text), it returned the single word `"True"` for every job -- not JSON, not prose, nothing resembling the task. This matches the model card's own published evaluation (ROUGE1 4.3, ROUGE-L 3.6 out of 100) -- by the author's own numbers, this model does not perform its stated task well. **Verdict: not fit for purpose, independent of prompting.**
- **TinyLlama+LoRA**: produced syntactically-plausible JSON, which is a real step up from qwen3's malformed responses -- but the *content* is the problem. It copied the prompt's own internal scaffolding (`"claim ceiling: participation | agency ceiling: individual_contributor | fact: ..."`) verbatim into bullet fields instead of writing a bullet, and in one case emitted single schema-label words (`"Design"`, `"Individual_contributor"`) as standalone "bullets." This reads as an out-of-distribution prompt for whatever format the LoRA adapter was actually fine-tuned on -- it doesn't know these are instructions, not content to echo. **Verdict: better JSON hygiene than qwen3, but the actual writing quality is worse -- not usable.**
- **qwen3:1.7b (recap, generic baseline)**: 2 malformed-JSON responses, 1 fully safety-rejected. Included as the control -- **the generic, non-specialized model was not meaningfully worse than either "specialized" candidate**, and arguably better than TinyLlama in one respect (never echoed prompt scaffolding as content).

### Independent structural factor (affects all three)

As found in the qwen3 report, the shared base-resume artifact used in this experiment contains pre-2026-08-25 formatting ("Dates not specified" placeholders, one title upgrade) that now hard-fails `validate_json_fields` regardless of model output, and `merge_realization` copies the raw job title verbatim, tripping the title-inflation guard for any "engineer"-titled job. **This means even TinyLlama's technically-parseable, safety-passing content on jobs 7267/22916 still couldn't have shipped** -- both were structurally doomed independent of specialization. Not fixed here (out of scope).

### Resource cost

| | T5-base | TinyLlama+LoRA |
|---|---|---|
| Per-job latency (warm, cached) | ~1.3s | 96-152s |
| Total wall time, 3 jobs (2nd+ run, cached) | 88.8s | 372.5s |
| Peak observed free-RAM dip | to ~2.4GB (from 3.15GB baseline) | to ~2.4GB (from 3.15GB baseline) |
| RAM after process exit | baseline, exactly | baseline, exactly |
| Disk footprint (approx, from known download sizes) | ~900MB (fp32 weights + tokenizer) | ~2.3GB (bf16 base weights) + adapter (tens of MB) |

No thrashing or instability at any point across either model's full run.

## Answer to the research question

**"Does a genuinely resume-tailoring-specialized small model provide better truthful tailoring than qwen3:1.7b or the existing deterministic + cloud pipeline, while being practical on this hardware?"**

**No, on both counts it was tested on:**
1. **Practical to run**: yes, now that they're installed in an isolated environment -- both load and run without destabilizing the machine. This part of the question is answered affirmatively.
2. **Better truthful tailoring than qwen3:1.7b**: no. Both specialized candidates performed at or below qwen3's already-poor level. T5 is fundamentally broken for this task (confirmed by its own training metrics). TinyLlama produces cleaner JSON syntax but worse content -- it doesn't understand the instruction-style prompt and echoes scaffolding instead of writing bullets.
3. **Better than the existing deterministic + cloud pipeline**: no. Cloud-tailored baselines for these same jobs addressed 7-8/8 requirements each; all three local models (generic and specialized) addressed 0/3 testable jobs.

## Recommendations

- **T5-base (`Abhishek9998/t5-base-finetuned-resumes_t2json_large`): reject.** Confirmed practical to run, confirmed not fit for purpose (task collapse + author's own poor metrics). No further investigation warranted.
- **TinyLlama Resume Tailor (`akshayrinku/tinyllama-resume-tailor-lora`): reject.** Confirmed practical to run, produces worse usable content than the generic qwen3:1.7b baseline despite being "specialized." Its LoRA fine-tuning does not appear to generalize to ApplyPilot's structured, schema-guided prompt format.
- **Qwen2.5-7B Resume LoRA: still not attempted, per instructions** -- nothing in this experiment changes the earlier "impractical on current hardware" assessment (this machine's 3.15GB free RAM baseline can't accommodate a 7B model even quantized, and the two smaller candidates already answer the "does specialization help" question negatively).
- **No production architecture change is warranted.** This closes the specialized-small-model question: neither generic nor specialized small local models produce usable tailored output on this candidate's real job mix. The deterministic-extraction + cloud-realization path remains the only reliable option.
- **Cleanup**: the isolated venv and downloaded model weights (~3.2GB combined) can be deleted at any time (`data/experiments/specialized_model_infra_20260901/venv` and the HF cache entries for these two model IDs) -- they're git-ignored and have no further use now that the question is answered. Left in place for now in case you want to re-inspect anything.
