# Prompt/Representation Variants (Phase 2 design)

Per the user's explicit note "we don't necessarily need every representation... if that becomes expensive" and the deliverable's own request for "the smallest experiment that would decisively answer the remaining question," this sweep is scoped down from the full 5-configuration matrix to the 3 configurations that most directly answer Phase 4's core hypothesis (rewriting vs. planning) plus one native-format sanity check. This is a deliberate reduction, documented here rather than silently applied.

## Configurations actually run

### Config 1 — Native/simple
Closest to each model's own (essentially undocumented — see model_inventory.md) base template: a plain system+user chat turn, no ApplyPilot schema, no pre-selected evidence. The model must find and select relevant content itself.

```
SYSTEM: You are a career assistant. Rewrite resumes to better fit a target job.
Never invent facts, employers, titles, dates, or skills that are not already
present in the resume. If something isn't supported by the resume, leave it out.

USER: JOB DESCRIPTION:
<job title + full description, truncated to 3000 chars>

RESUME:
<full base resume text, verbatim>

TASK: Tailor this resume to the job above. Do not invent any information.
```

### Config 4 — Bullet/section rewriting only ("especially important" per instructions)
Isolates language-generation quality from resume structure/JSON/omission decisions. For each job, ApplyPilot's OWN deterministic evidence system (`rank_profile_evidence` + `_auto_resolve_requirements`, unchanged production code) selects ONE supported requirement and its matching existing resume bullet(s) — the model only rewrites, never plans.

```
SYSTEM: You rewrite individual resume bullet points to better match a specific
job requirement. Rewrite ONLY using facts already present in the bullet(s)
given to you. Do not add any new fact, tool, employer, metric, or outcome
that is not already stated.

USER: JOB REQUIREMENT: <one requirement line, plain text>

EXISTING RESUME BULLET(S) (this is the only information you may draw on):
- <bullet 1>
- <bullet 2 if present>

TASK: Rewrite the bullet(s) above to more directly speak to the job
requirement, using only what's already stated. Output only the rewritten
bullet(s), one per line.
```

### Config "plan" — Whole-resume generation from a plain-language deterministic plan (Phase 4's other arm)
Same deterministic evidence selection as Config 4, but ALL supported requirements for the job are handed over (not just one), in the plain human-readable format the user specified (not ApplyPilot's internal JSON schema), and the model is asked to produce full tailored EXPERIENCE section text, not just isolated bullets.

```
SYSTEM: You are a career assistant tailoring a resume to a job. You are
given the job's requirements and, for each one, the ONLY resume evidence
you are allowed to draw on. Never invent facts, employers, titles, dates,
metrics, or skills beyond what's given. If a requirement has no evidence
listed, do not address it.

USER: JOB: <title>

REQUIREMENT: <requirement 1 text>
SUPPORTED BY:
- <evidence source> — <evidence description>
TAILORING DIRECTION: emphasize this requirement using only the evidence above.

REQUIREMENT: <requirement 2 text>
SUPPORTED BY:
- ...
TAILORING DIRECTION: ...

[... one block per supported requirement ...]

EXISTING EXPERIENCE SECTION (for reference and reuse of anything not
touched above):
<verbatim EXPERIENCE section from the base resume>

TASK: Rewrite the EXPERIENCE section to better address the requirements
above, using only the evidence given. Keep every employer/date/title
exactly as in the original. Output only the rewritten EXPERIENCE section.
```

## Configurations explicitly NOT run this round (and why)

- **Config 2 — Structured evidence (separate from Config "plan")**: substantially overlaps with Config "plan" (both hand over pre-selected evidence + explicit direction); running both would mostly duplicate signal for 2x the cost. Config "plan" was kept because it also answers Phase 4's whole-resume-vs-rewrite question directly; a separate narrower Config 2 was dropped.
- **Config 5 — Minimal instruction**: with no documented native example to defer to for either specialized model (see model_inventory.md), Config 1 already IS the minimal-instruction case in practice — there's no more-minimal documented alternative to test against.
- **ApplyPilot production JSON-schema config**: already fully covered by the prior `specialized_model_infra_20260901` and `local_model_bakeoff_v3` experiments (T5, TinyLlama, qwen3 all tested against production's real prompt/parser). Not repeated here — this experiment exists specifically to test OTHER representations.

## Jobs

Primary (all 3 configs): **7267** (Software Development Engineer, Adobe — 2/8 requirements supported) and **22916** (Hotel Partner Solutions Specialist, Expedia — 3/8 supported, the strongest evidence coverage of the 4 jobs used across this investigation so far). Chosen per instructions to "prefer jobs where our deterministic evidence system has reasonably strong support."

Spot-check (Config 1 only, for breadth across a lower-evidence job): **27323** (Software Engineer Graduate, HPE — 1/8 supported).

Same base resume (the real, previously cloud-validated McLaughlin resume used throughout this investigation) and same `data/profile.json` for every model/config, per the "identical base resume, identical grounding constraints" requirement. No model's output is ever fed into another model.

## Models

`ahmadd46/apex-resume-qwen-3b` (Apex), `abhaykanjoor/iteratecv-llama-3.2-3b-gguf` (IterateCV), `qwen2.5:3b` (non-specialized control, same size class as both specialized candidates), `qwen3:1.7b` (existing local baseline — smaller, already used throughout this investigation).

All four run via Ollama (no transformers/torch needed this round — unlike the T5/TinyLlama experiment, both specialized candidates here ship as ready GGUF releases). One model resident at a time; each model's full job×config sweep runs to completion before the next model loads.
