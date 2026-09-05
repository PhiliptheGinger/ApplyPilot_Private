# Specialized Small-Model Prompt/Representation Bake-Off

Continuation of the local small-model tailoring investigation. The prior experiment (`specialized_model_infra_20260901`) tested T5-base and TinyLlama+LoRA under ApplyPilot's own JSON schema and prompt, and found both unusable. This experiment asks a different question: **is that a model-capability problem, or a representation-mismatch problem?** Four models (two specialized, two generic) were tested under three different prompt shapes — none of them ApplyPilot's production schema — on the same 2-3 real jobs.

**Headline result: representation was the dominant variable, not model selection.** Every model tested — specialized or generic, 1.7B or 3B — was safe and truthful when restricted to rewriting a small, pre-selected span of text, and every model fabricated something at least once when asked to plan or generate a whole resume on its own. Neither specialized candidate was more truthful *and* more useful than a same-size generic control.

Full per-output judgments: `structured_evaluation.json`. Raw text for every model/config/job: `raw_output_*.json` and the consolidated `all_raw_outputs_consolidated.txt`. Prompt construction: `shared_variants.json`, `prompt_variants.md`. Model research: `model_inventory.md`.

---

## 1. Which models were actually runnable?

All four. Two verification surprises worth noting: both "specialized" candidates turned out to already be quantized GGUF releases (not raw PEFT adapters needing a merge, unlike the prior experiment's TinyLlama), so this round needed **no torch/transformers install at all** — everything ran through Ollama.

| Model | Verified identifier | Format |
|---|---|---|
| Apex Resume Qwen2.5-3B | `ahmadd46/apex-resume-qwen-3b` | GGUF Q4_K_M, ~1.9GB |
| IterateCV Llama-3.2-3B | `abhaykanjoor/iteratecv-llama-3.2-3b-gguf` | GGUF Q4_K_M, ~2.0GB |
| Qwen2.5-3B (control) | `qwen2.5:3b` (official Ollama library) | GGUF, ~1.9GB |
| qwen3:1.7b (baseline) | `qwen3:1.7b` (already installed) | GGUF, 1.4GB |

Qwen2.5-7B was **not** downloaded — nothing in this round changed the prior experiment's "impractical on this hardware" conclusion (this machine's RAM baseline is ~3GB free at rest), and the two smaller candidates already answer the core question.

## 2. What representation was each model actually trained for?

**Neither specialized model documents one.** This was the most surprising Phase-1 finding: Apex's README has no example prompt, no chat template, no training dataset, and its HuggingFace widget examples are the generic *default* placeholder questions ("What is 84 * 3 / 2?") — never customized, meaning there is no vetted example of the model's actual resume behavior anywhere in its public listing. IterateCV ships an Ollama Modelfile (more concrete than Apex has), but it's just the **stock Llama-3.2-Instruct chat template** with no resume-specific system prompt, and a notably high default temperature (1.5) that its own Modelfile sets — a red flag for a fact-sensitive task, which this experiment overrode to 0.3. Both models are LoRA fine-tunes of stock instruct models with no evidence beyond a one-paragraph description of intended use. "Native/simple" in this experiment therefore means the most reasonable plain-language prompt using each base model's standard chat template — not a verified training-matched format, because no such format is published. This constrains how strong a claim any single sweep like this one can make (see Q10).

## 3. Which representation worked best for each model?

**The rewrite-only config, for every model, without exception.** Across all 4 models × 2 primary jobs = 8 rewrite-config outputs, **zero contained an invented fact.** All 8 were grounded, on-topic, truthful rewrites of the exact evidence handed to the model. This is the one config where every model succeeded.

The plan and native configs were far more mixed and are where every model's failures concentrated (detailed per-model below).

## 4. Did changing the representation materially improve output quality?

**Yes, dramatically — but the improvement was in truthfulness, not in "does it work at all."** All three configs were parseable and produced *some* output for every model (once the qwen3 thinking-mode bug was fixed — see the Incident section below). The representation change that mattered wasn't native vs. plan vs. schema-JSON; it was **rewrite-only vs. anything-whole-resume**. This directly confirms the experiment's premise: the prior experiment's ApplyPilot-schema results (T5 collapsing to "True", TinyLlama echoing prompt scaffolding) were at least partly a representation problem — but it turns out the deeper, more general problem isn't JSON-schema-shaped prompts specifically, it's **asking a small model to do open-ended resume planning at all**, in any prompt shape.

## 5. Did specialized models beat qwen3:1.7b?

**No, and the picture is more interesting than a simple win/loss.**

- **Apex** was the *safest* model in the whole experiment — zero fabrications across all 7 of its outputs, including native and plan modes where every other model (including qwen3) fabricated something. But it paid for that safety with near-total inertia: in native and plan modes it reproduced the original resume almost byte-for-byte, doing essentially no tailoring at all. Its one clear defect was echoing the prompt's own instruction text (`"Tailor: emphasize this requirement using only the evidence above."`) verbatim into the output twice in the plan config — the same "prompt-scaffolding-as-content" failure mode found in TinyLlama in the prior experiment, now confirmed on a different, larger, more recent model.
- **IterateCV** was the *worst* model tested — it fabricated a fictional prior job ("ALIGNMENT TECHNIQUES") plus a false "5+ years of software engineering experience" claim in native mode, and showed a distinct, systematic **keyword-injection pattern** in plan mode: appending job-posting phrases like "utilizing data to inform comedic timing" onto bullets (including the Stand-up Comedian entry!) that have nothing to do with data at all. This reads like the model learned "insert relevant-sounding keywords everywhere" as a shortcut for tailoring, with no grounding check.
- **qwen2.5:3b (generic control)** sat in between: safe in rewrite mode, but produced the single most severe fabrication of the entire experiment — twice inventing an entire fictional prior job, once literally titled with the *exact target company and job title* ("Hotel Partner Solutions Specialist at Expedia Group") and populated with duties lifted from the job posting rather than the candidate's real history.
- **qwen3:1.7b**, once the harness bug was fixed, performed comparably to the 3B control: safe rewrites, but fabricated an extensive skills list in 2/3 native-mode outputs and, in the worst single case in the experiment, **invented specific employment date ranges** for every job on the spot-check resume, where the original explicitly said "Dates not specified."

None of the three larger (3B) models were more truthful than qwen3:1.7b. Being 3B instead of 1.7B bought more complete, more fluent native-mode output and (for two of the three) real document-level curation ability (dropping irrelevant sections) — but not better grounding.

## 6. Did they work better at bullet/section rewriting than whole-resume generation?

**Yes, unambiguously, for every model.** This is Phase 4's central question and the experiment's clearest result. See the table:

| Config | Fabrication rate (outputs with ≥1 invented fact) |
|---|---|
| Rewrite (isolated bullet(s), pre-selected evidence) | **0 / 8** (4 models × 2 jobs) |
| Native or Plan (any whole-resume mode) | **9 / 16** (4 models × 4 whole-resume outputs each, incl. spot-check) |

Every model that fabricated, fabricated in native or plan mode. Not one model fabricated in rewrite mode. This held regardless of specialization, base model family (Qwen vs. Llama), or size (1.7B vs. 3B).

## 7. Did they produce truthful, evidence-grounded improvements?

In rewrite mode: yes, consistently, across all 4 models. The rewrites were genuinely better-targeted than the originals — they picked up on the requirement's specific framing (e.g. "data consistency and validation") and re-emphasized language already present in the evidence, without inventing anything. This is a real, positive, replicable result.

In native/plan mode: rarely, and never without risk. The two models that did show real document-level judgment (qwen2.5:3b and Apex, in plan mode) only ever *selected which existing entries to keep* — none of the four models successfully rewrote bullet text to better match a requirement while working from a whole-resume-scale prompt. Either they fabricated (qwen2.5:3b, IterateCV, qwen3) or they simply reproduced everything verbatim without engaging (qwen3's plan outputs, Apex's plan-mode bullets).

## 8. Were failures caused by the model or by our representation/integration?

Distinguished explicitly per output in `structured_evaluation.json` (an `attribution` field on every entry: `model`, `integration`, or `none`). Summary:

- **Genuine model-quality failures** (fabricated fact/employer/title/date/skill, or systematically inventing content): the large majority of the failures — 9 whole-resume outputs across all 4 models, plus IterateCV's two systematic keyword-injection cases.
- **Integration/representation-adjacent failures**: Apex echoing the plan-config's own instruction text into the output (`"Tailor: emphasize..."`) — the model wasn't lying, it was confused about what was instruction vs. content, which is at least partly a prompt-design problem (the literal instruction text was visually adjacent to content it was meant to modify). One genuine harness bug was caught and corrected: qwen3's Ollama "thinking" mode silently consumed the entire token budget on 4 of 7 first-pass calls, producing empty output that had nothing to do with qwen3's actual capability — confirmed by a targeted retest and then fixed for the full run (see Incident below). This is exactly the class of error this experiment was designed to catch and NOT misattribute to the model.
- **Task non-compliance (not fabrication, not clearly a bug either)**: several "safe" outputs that simply ignored the tailoring instructions and reproduced the resume verbatim (qwen3's plan-mode outputs, Apex's and IterateCV's native-mode outputs). These aren't lies, but they're not useful either — a third category the pass/fail framing of the prior experiment couldn't represent.

## 9. Is there a promising architecture for using one of these models inside ApplyPilot?

**A narrow, bounded rewrite step — not a planner — is the only pattern that showed zero fabrication across every model tested.** Concretely: ApplyPilot's existing deterministic evidence-selection layer (`rank_profile_evidence`, `_auto_resolve_requirements`, already production code) already does exactly the narrowing this needs — pick ONE supported requirement, hand the model ONLY its matching existing bullet(s), ask for a rewrite, nothing else. This experiment's rewrite config is a close preview of that architecture and it worked, consistently, for all 4 models including the smallest and least "specialized" one (qwen3:1.7b). That's actually good news independent of which specific model gets used — it suggests the fix isn't "find a better model," it's "never let a small local model see the whole resume-planning problem at once." Whether this is *worth* building (a bullet-at-a-time rewrite loop is architecturally different from the current single-shot JSON generation, and still doesn't rewrite the summary or make cross-requirement tradeoffs) is a real design question, not something this experiment resolves — see Q10.

## 10. What is the smallest next experiment that would decisively answer the remaining question?

This experiment answered "does representation matter" (yes) and "is rewrite-only safer than whole-resume" (yes, decisively). It did **not** answer whether a bullet-at-a-time rewrite architecture is *good enough* to replace degraded mode, because:
- Only one requirement per job was rewritten (Config 4 picked the *first* supported requirement, not all of them) — an architecture using this pattern would need to rewrite every supported requirement's bullets in sequence, and this experiment has no data on whether quality holds up over many sequential rewrite calls on the same resume, or whether a model starts drifting/repeating after several rounds.
- The rewrite config's evidence was hand-picked by production code in this experiment's harness, not actually wired into a live pipeline — there's no data yet on latency at realistic scale (a real job might have 3-5 supported requirements, meaning 3-5 sequential ~10-90s calls instead of one).
- No specialized/generic model was tested on the summary line at all (only bullets) — a real degraded-mode replacement needs the summary rewritten too, and this experiment has zero data on whether summary-rewriting is as safe as bullet-rewriting.

**The smallest next experiment**: take ONE real job with 3+ supported requirements, run the rewrite-only config sequentially across ALL of that job's supported requirements (not just the first) plus one summary-rewrite call, on qwen3:1.7b only (cheapest, and this experiment found no accuracy advantage to the 3B models), and check (a) does fabrication risk stay at zero across a multi-call sequence on the same resume, and (b) is total latency (likely 5-8 sequential local calls) acceptable versus just waiting for cloud to un-exhaust. That single-job, single-model, multi-call test would directly answer "is this architecture viable," which is the one thing this broader bake-off didn't have the scope to test.

---

## Incident: RAM pressure during the qwen3 sanity run

Free RAM dropped to 0.42GB partway through the initial qwen3 sweep — the lowest point in this entire investigation. Root cause: Ollama keeps a model resident in memory for a keep-alive period after the last API call by default, independent of the calling script exiting, and the harness's 7 sequential calls stacked on top of that. Disk activity (21% disk time, 65 pages/sec) was elevated but not full thrashing. Resolved immediately via `ollama stop <model>` (RAM recovered to 3.35GB within seconds) rather than waiting out the keep-alive timer — and from that point on, every subsequent model was explicitly force-unloaded and RAM-confirmed before the next one loaded, rather than relying on the default timeout. Reported to the user in real time per the resource guardrails rather than pushed through silently.

## Recommendation

- **Neither Apex nor IterateCV should be adopted as-is.** Apex is safe but nearly useless without a narrower task. IterateCV is actively worse than the generic control on truthfulness.
- **Do not pursue "find a better specialized model" as the next step.** This experiment's clearest signal is that the *task shape* (whole-resume planning vs. bounded rewriting), not model selection, is what determines fabrication risk across every model tried, specialized or not.
- **The one architecture worth prototyping further**: a bounded, evidence-scoped rewrite loop reusing ApplyPilot's existing deterministic evidence selection, per Q9/Q10. This is a real, if modest, positive finding — worth a small follow-up experiment before any production change, not worth a production change on the strength of this bake-off alone.
- No production code was modified in this experiment.
