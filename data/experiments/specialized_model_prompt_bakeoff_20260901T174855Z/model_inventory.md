# Model Inventory & Native Format (Phase 1)

Continuation of the specialized small-model tailoring investigation. Prior experiment (`specialized_model_infra_20260901`) tested T5-base and TinyLlama+LoRA under ApplyPilot's OWN structured prompt/schema and found both unusable. This experiment tests different, larger, more recently-published candidates under prompts closer to how they were actually documented/trained, to separate "model can't do the task" from "our prompt format is wrong for this model."

## Candidates and verified availability

| # | Candidate (as named in the request) | Verified exact identifier | Runnable? | Format |
|---|---|---|---|---|
| 1 | Apex Resume Qwen2.5-3B | `ahmadd46/apex-resume-qwen-3b` (GGUF `apex-resume-qwen-3b.Q4_K_M.gguf`, base `Qwen/Qwen2.5-3B-Instruct`) | **Yes** — already GGUF, tagged `ollama`, official run command documented: `ollama run hf.co/ahmadd46/apex-resume-qwen-3b:Q4_K_M` | Ollama/GGUF, ~1.93GB |
| 2 | IterateCV / Llama 3.2 3B resume-tailoring | `abhaykanjoor/iteratecv-llama-3.2-3b-gguf` (GGUF `llama-3.2-3b-instruct.Q4_K_M.gguf`, base Llama-3.2-3B-Instruct, converted via Unsloth). A separate PEFT-LoRA sibling repo (`abhaykanjoor/iteratecv-llama-3.2-3b-lora`) also exists but was NOT used, since the GGUF release is directly runnable. | **Yes** — GGUF release with an included Ollama Modelfile | Ollama/GGUF, ~1.9GB (Q4_K_M) |
| 3 | Ordinary Qwen2.5-3B (non-specialized control) | `qwen2.5:3b` — official Ollama library model | **Yes** | Ollama/GGUF, ~1.9GB |
| 4 | qwen3:1.7b (existing local baseline) | `qwen3:1.7b` — already installed | **Yes** (already used in two prior experiments) | Ollama/GGUF, 1.4GB |
| — | Qwen2.5-7B (mentioned in earlier round) | not attempted | Deliberately not downloaded — prior experiment already established 7B is impractical on this machine's ~3GB free RAM baseline, and this experiment's own instructions say not to unless first determined practical. Not re-litigated. |  |

All identifiers independently verified via the HuggingFace models API in this session (not assumed from the request text) — file lists, quantization, base models, and tags all confirmed directly.

## Native prompt/representation research (Phase 1, before any testing)

### Apex Resume Qwen2.5-3B

- **Method**: "LoRA fine-tuning on resume exemplars, then merged and quantized to GGUF (Q4_K_M)" — per the model's own README.
- **Stated purpose**: "rewriting and prioritising your resume content for that specific role" as part of a larger tool ("Apex Hunter, a local-first career assistant"). Explicitly documents a "honesty constraint" — does not fabricate skills/titles/employers/dates/metrics, and frames transferable experience as adjacency rather than mastery. This is a directly relevant, encouraging signal (the fine-tuning was reportedly truthfulness-aware).
- **Documented prompt format**: **none.** No example prompts, no chat template shown, no sample input/output pair. Only CLI run commands (`ollama run ...`, `llama-cli -m ... -p "Your prompt here"`).
- **Training dataset**: not disclosed, no dataset: tag, no linked HF dataset repo.
- **HF widget examples**: present but generic/default ("What is 84 * 3 / 2?", "Tell me an interesting fact about the universe!") — these are the standard HF widget placeholder questions, NOT resume-specific examples. This strongly suggests the widget was never customized, i.e. there is no vetted example of this model's actual resume-tailoring behavior anywhere in its public listing.
- **Conclusion**: the base model is Qwen2.5-3B-**Instruct**, so it almost certainly expects the standard Qwen2.5 ChatML-style template (`<|im_start|>system ... <|im_end|><|im_start|>user ... <|im_end|>`), which Ollama applies automatically. No special delimiters, no JSON schema, no resume-specific markup documented anywhere. "Native/simple" for this model = a plain system+user chat turn with the task stated in ordinary language, since that's all the documentation supports.

### IterateCV (Llama 3.2 3B GGUF)

- **Method**: fine-tuned and converted to GGUF via Unsloth (a common efficient-LoRA-training toolkit); a separate raw LoRA adapter repo also exists, confirming this is genuinely a fine-tune, not just a re-upload of stock Llama 3.2.
- **Documented prompt format**: the repo ships an actual Ollama **Modelfile**, which is more concrete evidence than Apex has -- but it turns out to be the **stock Llama-3.2-Instruct chat template** (`<|start_header_id|>...<|end_header_id|>...<|eot_id|>`), with generation params `temperature=1.5, min_p=0.1` (notably high temperature for a factual-accuracy-sensitive task -- flagged as a risk: this invites more creative/less literal output than tailoring should want. Overridden to a lower temperature in this experiment's own inference calls, noted per-run).
- **Training dataset / example conversations**: none published. README is minimal -- confirms only the base model, the Unsloth conversion, and CLI run commands.
- **Conclusion**: like Apex, there is no documented resume-specific prompt shape. "Native/simple" for this model = a plain system+user chat turn via the standard Llama 3.2 Instruct template.

### Ordinary Qwen2.5-3B (control)

- Stock instruct model, official Ollama library release, standard ChatML template. Used exactly as a non-specialized same-size-class control against Apex, and as a mid-size point of comparison against qwen3:1.7b.

## Phase-1 headline finding

**Neither specialized candidate documents an actual native prompt format, example input/output pair, or training dataset.** Both are LoRA fine-tunes of stock instruct models with no evidence beyond their name and a one-paragraph description of what they're "for." This matters for interpreting results: there is no ground-truth "correct" prompt to defer to, so "native/simple" in this experiment means the most reasonable plain-language prompt consistent with each base model's standard chat template and the one-paragraph stated purpose -- not a verified training-matched format. This constrains how strong a claim we can make from a single sweep (documented in `report.md`, Q10).
