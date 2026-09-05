# Small Specialized Resume-Tailoring Model Bake-Off (Corrected Re-Run)

Continuation of the Q2 local-model investigation. Read-only: no production code changed, no DB writes, no commits.

## What this fixes vs. the prior partial run

The prior run (`local_model_bakeoff_20260901_161413` / `_v2`) reported qwen3:1.7b as validator-failing on every job with "Missing required field: experience/education". That was a **harness bug**: it fed a raw profile-data dump (not an actual resume) into the resume parser, so those sections legitimately had nothing to parse. It also used `~/.applypilot/resume.txt`, which on this machine is a stale placeholder identity ("Philip Ginger") unrelated to the real candidate. This run fixes both by calling the exact production degraded-mode sequence (`build_base_resume_model` → `request_local_realization` → `merge_realization` → `validate_json_fields` → `judge_tailored_resume`) against a real, correctly-formatted McLaughlin resume.

## Candidate models

| Model | Exact identifier | Actually resume-specialized? | Runnable here? | Recommendation |
|---|---|---|---|---|
| TinyLlama Resume Tailor (1.1B LoRA) | `akshayrinku/tinyllama-resume-tailor-lora` | Yes (tagged `resume`, fine-tuned on TinyLlama-1.1B-Chat) | No — PEFT adapter, needs `transformers`+`peft` (not installed) | **reject** |
| T5-Base resume candidate (~220M) | `Abhishek9998/t5-base-finetuned-resumes_t2json_large` | Unclear — dataset name suggests resume→JSON extraction, not tailoring | No — same missing stack | **reject** |
| GPT-Neo resume assistant (~1.3B) | *(none)* | N/A | N/A | **reject — does not exist** (HF search returns 0 results) |
| Qwen2.5 Resume LoRA (~7B) | `Ella0506/Qwen2.5-7B-Instruct-Resume-LoRA` | Yes (LoRA over Qwen2.5-7B-Instruct, resume-optimization model card) | No — needs `transformers`+`peft`, and 7B merged is ~14GB+ RAM/VRAM | **impractical on current hardware** |
| qwen3:1.7b | `qwen3:1.7b` | **No** — generic instruction model, not resume-specialized | Yes (local Ollama) | **reject** (see below) |

All four HF identifiers above were independently re-verified in this session via the HuggingFace models API (not just trusted from the prior agent's report) — every field (pipeline_tag, library_name, base_model, tags) matched exactly.

## qwen3:1.7b direct-tailoring results (4 real jobs)

| Job | Title | Requirement support | Result | Latency |
|---|---|---|---|---|
| 27323 | Software Engineer Graduate (HPE) | 1 supported / 3 ambiguous / 4 unsupported | Model responded, but **both** its bullet and its summary were rejected by the claim-strength safety net (overclaimed relative to evidence) | 62.8s |
| 7267 | Software Development Engineer (Adobe) | 2 / 1 / 5 | Malformed JSON — response unparseable, discarded | 61.5s |
| 7406 | Cloud Platform Engineer (Accenture) | 0 / 1 / 7 | Correctly gated — zero schema-supported requirements, model never invoked | 0.0s |
| 22916 | Hotel Partner Solutions Specialist (Expedia) | 3 / 0 / 5 | Malformed JSON — response unparseable, discarded | 62.8s |

**0 of 4 jobs produced any usable tailored content.** Where the model was invoked (3/4), it failed 100% of the time — 2 malformed-JSON responses, 1 fully safety-rejected. The base resume was returned byte-identical in every case (the safety design is working: nothing false ever reached output), but that also means zero tailoring value.

## A second, independent, model-agnostic finding

A direct probe (`title_inflation_probe.json`) tested what would happen with a **hypothetically perfect, clean** local realization instead of qwen3's actual output. Result: **still 0/4 pass.**

Two causes, both structural, not model-quality issues:

1. **Title inflation (real, reproducible, affects any local model):** `local_tailor.merge_realization()` copies the job posting's raw title verbatim into the resume header with no grounding check. `validator.check_title_inflation` (added 2026-08-25) then hard-rejects any elevated word ("engineer") not present in the candidate's authoritative titles. This candidate's profile has no authoritative "engineer" title (service/retail background), so **any software-engineering job title reaching degraded mode is structurally guaranteed to fail validation**, regardless of which model realizes the bullets. Confirmed on 3/4 jobs; the 4th (non-engineering title) doesn't trigger it, consistent with the mechanism. The cloud path avoids this because its prompt explicitly instructs the LLM to derive a non-inflated title — degraded mode has no equivalent step.
2. **Stale base-resume artifact (likely specific to this experiment's substitute file, flagged as a caveat):** the real base resume text used here predates the 2026-08-25 validator hardening and contains "Dates not specified" placeholders and one pre-existing title upgrade, both of which now hard-fail validation on their own.

Net effect: **the 0% success rate above is a lower bound, not just a qwen3 problem.** Not fixed here (out of scope — no production changes), but worth a dedicated follow-up: either `merge_realization` needs to derive/validate a grounded title the way the cloud prompt does, or degraded mode needs to skip title-bearing jobs it structurally cannot pass.

## Comparison to existing cloud-tailored baselines (already on disk, real production output)

| Job | Cloud requirements addressed | Local (qwen3) requirements addressed |
|---|---|---|
| 27323 | 7/8 | 0/8 |
| 7267 | 7/8 | 0/8 |
| 7406 | 8/8 | 0/8 |
| 22916 | n/a (never reached tailoring in production) | 0/8 |

## Answer to the core question

**No.** No genuinely small resume-tailoring-specialized model could be tested directly — every named candidate is either non-runnable in this environment without installing new ML infrastructure, doesn't exist, or is impractical on this hardware. The one runnable local model, qwen3:1.7b (explicitly *not* a specialized model — included only as the generic local baseline), produced zero usable output across 4 real, characteristically-different jobs, and a structural gap independent of model choice means even a perfect local model would still fail most of them today.

This extends, rather than contradicts, the original Q2 finding: local-model *planning* (feeding qwen3 output into the cloud path) was previously found to add no coverage benefit and cost ~2x latency. This run shows local-model *direct* tailoring (no cloud involvement) is materially worse still — the existing deterministic-extraction + cloud-realization architecture remains the only path in this system producing validated, truthful, requirement-covering resumes.

## Recommendation

- **Do not pursue any of the 4 named specialized candidates** — none are practically available.
- **Do not pursue qwen3:1.7b (or any similarly-sized generic local model) for direct tailoring** — 0% success rate, ~60s/job cost, and a structural blocker independent of the model.
- **No production architecture change is warranted** from this investigation. The one actionable follow-up worth a human decision: whether to invest in fixing `merge_realization`'s title-grounding gap, since it currently makes degraded mode nearly unusable for this candidate's profile shape on engineering-titled jobs specifically (a real, narrow bug, not a "small models don't work" conclusion).
