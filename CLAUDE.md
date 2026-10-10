# ApplyPilot — Claude Code Operating Manual

## Mission

ApplyPilot is an autonomous job application pipeline.
Claude's job is to **operate, monitor, and fix the pipeline** — not to manually do what the pipeline automates.

**The goal**: Discover jobs → Score them → Tailor resumes → Generate cover letters → Auto-apply. All automated.

---

## Claude's Role (READ THIS FIRST)

**Claude is the pipeline engineer and operator. Claude does NOT manually apply to jobs.**

### What Claude Does
1. Run pipeline commands (`applypilot run ...`, `applypilot apply`, `applypilot status`)
2. Monitor logs and output for errors
3. Diagnose root causes when things break
4. Fix the source code (`src/applypilot/`)
5. Re-run and verify fixes work
6. Keep this CLAUDE.md updated with decisions and learnings (see "Keeping this file small" below)

### What Claude Does NOT Do
- Open a browser via Playwright and manually fill out application forms
- Act as the "apply agent" — that's what `applypilot apply` spawns Claude Code subprocesses for
- Skip the automation and do things by hand "just this once"

### Daily Operating Loop
```
1. applypilot status              # Where are we? What's the funnel?
2. applypilot run discover        # Find new jobs
3. applypilot run enrich          # Fetch full descriptions
4. applypilot run score           # AI scoring
5. applypilot run tailor          # Tailor resumes for 7+ scores
6. applypilot run cover           # Generate cover letters
7. applypilot apply               # Auto-apply (Tier 3 — uses Claude Code credits)
8. applypilot dashboard           # Generate HTML dashboard for review
```

When a stage fails: stop, read logs, find root cause, fix code, re-run.

---

## Architecture

### Three Tiers
- **Tier 1** (Discovery): No API key. Scrapes job boards.
- **Tier 2** (AI Processing): Gemini/OpenAI API. Score, tailor, cover letters.
- **Tier 3** (Auto-Apply): Claude Code CLI as subprocess. Fills forms via Playwright.

### Two Credit Systems (IMPORTANT)
- **Tier 2**: Gemini API (free tier) + OpenAI fallback. Keys in `~/.applypilot/.env`
- **Tier 3**: Claude Code CLI with a Pro plan (2026-09-09 correction: prior docs/decisions in this file say "Max plan" throughout -- the account is actually Pro, meaningfully lower usage headroom than Max; don't assume Max-plan-scale budget when reasoning about `claude_cli`/interactive-Claude usage). IMPORTANT: `ANTHROPIC_API_KEY` must be stripped from subprocess env (launcher.py does this) or it overrides subscription auth with API billing. No Gemini browser agent exists — the Gemini/OpenAI cascade is Tier 2 only. `--strict-mcp-config` is required to prevent Docker MCP's Playwright (which can't access host files) from interfering with resume uploads.

### LLM Client (`src/applypilot/llm.py`)

Multi-provider fallback with two-tier model strategy:
- **Fast** (scoring, HN extraction): Gemini Flash → OpenAI → Anthropic Haiku
- **Quality** (tailoring, cover letters): Gemini Pro → OpenAI → Anthropic Sonnet

Key behaviors:
- `get_client(quality=False)` for fast, `get_client(quality=True)` for quality
- On 429: marks model exhausted for 5 min, falls to next in chain
- `config.load_env()` MUST be called before importing `llm` (env vars read at module import)
- Gemini 2.5+ thinking tokens consume max_tokens budget — set much higher than visible output needs

### Database (`src/applypilot/database.py`)

SQLite with WAL mode. Thread-local connections.
- `ensure_columns()` auto-adds missing columns via ALTER TABLE
- URL normalization at insert time (resolves relative URLs via `sites.yaml` base_urls)
- `company` column extracted from `application_url` domain (Workday, Greenhouse, Lever, iCIMS patterns)
- `acquire_job()` uses company-aware prioritization to spread applications across employers

### Pipeline Stages

| Stage | Condition | Tab |
|-------|-----------|-----|
| `discovered` | no description, no error | active |
| `enrich_error` | has `detail_error` | archive |
| `enriched` | has description, no score | active |
| `scored` | score < 7 | archive |
| `scored_high` | score >= 7, not tailored | active |
| `tailor_failed` | attempts >= 5, no result | archive |
| `tailored` | has resume, no cover letter | active |
| `cover_ready` | has cover letter, not applied | active |
| `applied` | `apply_status = 'applied'` | applied |
| `apply_failed` | permanent apply error | archive |
| `apply_retry` | retryable apply error | active |

---

## File Locations

| What | Path |
|------|------|
| Source code | `src/applypilot/` (editable install) |
| Venv | `venv/` |
| Resume (txt) | `~/.applypilot/resume.txt` |
| Resume (PDF) | `~/.applypilot/resume.pdf` |
| API keys | `~/.applypilot/.env` (NEVER commit.) |
| Profile | `data/profile.json` (project canonical profile) |
| Search config | `~/.applypilot/searches.yaml` |
| Database | `~/.applypilot/applypilot.db` |
| Tailored resumes (new source files) | `data/resumes/applypilot/{name}_{title}_{hash}.txt` (+`.pdf`) |
| Tailored resumes (submitted copies) | `data/resumes/applypilot/submitted/` |
| Legacy tailored exports | `~/Downloads/Resumes/` (existing database rows may reference these historical exports) |
| Cover letters | `~/.applypilot/cover_letters/{site}_{title}_{hash}_CL.txt` (+`.pdf`) |
| Apply logs | `~/.applypilot/logs/claude_{YYYYMMDD_HHMMSS}_w{N}_{site}.txt` |
| Dashboard | `~/.applypilot/dashboard.html` |

---

## Candidate Profile

Loaded from `~/.applypilot/profile.json` at runtime. See `applypilot init` to create one.

---

## Key Commands

```bash
# Tier 2 pipeline (safe, uses Gemini/OpenAI)
applypilot run discover                        # Find new jobs
applypilot run enrich                          # Fetch full descriptions
applypilot run score --limit 100               # AI scoring
applypilot run tailor --limit 50               # Tailor resumes (score >= 7)
applypilot run cover                           # Generate cover letters
applypilot run score tailor cover --stream     # All stages concurrently
applypilot status                              # Pipeline funnel stats
applypilot dashboard                           # Generate HTML dashboard

# Tier 3 apply (uses Claude Code credits)
applypilot apply --dry-run --url URL           # Test one job (no submit)
applypilot apply                               # Auto-apply to cover_ready jobs
applypilot apply --human-first                 # LinkedIn human-first flow (decision #187) -- click Apply yourself, automation takes over on hand-off
applypilot sms --setup                         # Verify Twilio SMS relay connectivity (decision #197, relay only -- not wired into apply flow)
applypilot notify --test                       # Email alerts for needs_human pauses / Claude usage limits (FW53; set APPLYPILOT_NOTIFY_EMAIL=self in .env)
```

---

## Orchestration Strategy

When running the pipeline:
1. **Throughput** — use `--stream` for concurrent stages
2. **Quality** — highest scores get tailored first, company diversity in applications
3. **Error handling** — if > 30% failure rate, stop and fix before continuing
4. **Bottleneck focus** — priority is building the apply-ready queue

Error patterns:
- Gemini 429: automatic fallback, no intervention needed
- Tailor validation failures > 30%: investigate validator settings
- Apply credit exhaustion: alert user, cannot auto-fix
- `hn://` URLs or malformed data: check hackernews.py sanitization

---

## Security Decisions

Full rationale/detail for every decision below has been moved to
`docs/decisions_archive.md` to keep this auto-loaded file small -- this
table is a compact index only (number + one-line gist). Look up a
decision by number in the archive file for the complete real-data
verification, root cause, and fix detail.

**Keeping this file small (rule, 2026-10-07).** This file loads into every
session, so its size is paid on every turn. When recording a new decision:
add ONE line here (number + gist, ~150 characters max) and put the full
write-up as a row in `docs/decisions_archive.md`. New ideas and follow-ups go
in `docs/future_work.md` (full text, keep item numbering) and get a row in
`docs/backlog.md` (priority, dependencies, where it can run). Never paste
a full write-up into this file.

**Archive, never delete (rule, 2026-10-07).** Decisions, Future Work items,
notes and stray files are never deleted. A closed Future Work item stays in
`docs/future_work.md` struck through; a superseded file moves to
`docs/archive/` (see its README). Only provably empty files may be removed.

| # | Decision |
|---|----------|
| 0 | Never paste API keys in chat |
| 1 | Display name from profile.json |
| 3 | No real password in profile.json |
| 4 | Tier 2 only until pipeline is stable |
| 5 | Review tailored resumes before using |
| 6 | Gemini free tier + OpenAI fallback |
| 7 | Location from searches.yaml |
| 12 | Two-tier model strategy |
| 13 | High max_tokens for thinking models |
| 17 | Skip Gmail MCP / CapSolver |
| 18 | URL normalization at discovery |
| 19 | Banned words = warnings not errors |
| 20 | Jobs without application_url = manual |
| 23 | Company-aware apply prioritization |
| 25 | Apply uses Claude Code CLI, not Gemini |
| 26 | HN URL sanitization |
| 27 | Basic prompt injection defense |
| 28 | `--strict-mcp-config` for apply subprocess |
| 29 | Funnel config: min_score=8, age=14d, cap=3/30d |
| 30 | Explicit state machine replaces implicit status derivation |
| 31 | P0 state-machine rollout shipped |
| 32 | Per-worker resume dir |
| 33 | Apply uses Chrome for Testing |
| 34 | HITL paths collapsed |
| 35 | Standalone human-review deleted |
| 36 | patchright launch flags + stdin Done fallback |
| 37 | P0.5 status path leaks closed (audit #10) |
| 38 | Per-install random extension key |
| 39 | Extension per-job tab tracking + action log |
| 40 | Pause-cycle action-log wiring |
| 41 | HITL toggle indicator in popup |
| 42 | Stealth init scripts via chrome.scripting (spec §2.2) |
| 43 | launcher.py decompose |
| 44 | Settings options page (extension) |
| 45 | Embedded-ATS URL canonicalization |
| 46 | Per-ATS successful-path memoization |
| 47 | Default the apply web driver to sonnet |
| 48 | Extension ID computed at runtime, not hardcoded |
| 49 | Active no-hitl toggle in popup |
| 50 | Worker-log buffering = line-buffered |
| 51 | Cover-letter quality overhaul |
| 52 | Vast.ai GPU backend evaluated, then reverted (superseded by #53) |
| 53 | Hosted-API-only tailoring; local/Vast model server dropped |
| 54 | Cognitive-linguistic schema layer for tailoring/cover letters |
| 55 | DEGRADED MODE realizes schema-bounded content, not a full resume |
| 56 | Degraded-mode stale-exhaustion-check fix + explicit base/realization/merge split |
| 57 | Cloud-only discovery attempt eliminates the expensive degraded-mode discovery call |
| 58 | Degraded mode distinguishes no-supported-evidence / realization-failed / genuinely-invalid |
| 59 | Cognitive-linguistic architecture expansion: frame/event/claim-strength/viewpoint/salience layers |
| 60 | Round-2 expansion: agency axis split, frame widening, prevention/force_relation, metric-fabrication + passive-voice checks |
| 61 | Markerless-paragraph requirement extraction (Direction 1 of the extraction audit) |
| 62 | Profile-authority hardening: title/skill/seniority/standup fabrication fixes |
| 63 | Adversarial review of #62 found and fixed 5 real gaps -- 3 in #62's own new code, 2 pre-existing but newly-exposed |
| 64 | `classify_seniority_mismatch` consolidated onto the pre-existing canonical `applypilot.eligibility` predicate |
| 65 | `pending_tailor` selection query gained a real `state` filter -- fixes a live archived/tailor_failed infinite oscillation |
| 66 | Sentence-diversity bake-off: cross-encoder wins, wired in as an optional two-tier tier in semantic_match.py |
| 67 | Real production bugs found and fixed live, plus a test-infra leak that was corrupting real pipeline state |
| 68 | Evidence-matching precision hardening: 388 → 39 false-positive-inflated matches for one profile entry, plus a discovered gap in the ambiguous-term safety net |
| 69 | Deterministic selector built and proven end-to-end on real jobs -- not wired into the live pipeline, and only has real pool material for one entry |
| 70 | Editor mode built: reword one already-true sentence instead of writing a new one from evidence -- safety net proven live twice, local-model reliability still open |
| 71 | Evidence-matching round 2: the `supported`-flag gap closed (narrowly), a real corroboration attempt reverted after breaking real evidence, and the actual fix -- an identity-vs-incidental term distinction |
| 72 | Phrase bank ("Stage 0") built and wired into both tailoring paths -- generation moved from "fresh every job" to "generate once, select+edit per job" |
| 73 | Phrase bank proven end-to-end on a real job; four real bugs found and fixed closing the gap between "#72 is tested" and "#72 actually works" |
| 74 | Editor-as-writer hypothesis confirmed: `/no_think` replaced with the real `think` API field, editor reliability and speed both fixed with real testing, native-Ollama fast path added opt-in |
| 75 | Real data-integrity audit: a fabricated-identity scoring bug (fixed 2026-08-27) left stale scores live in production, including one real submitted application; a separate, unrelated enrichment bug found (WeWorkRemotely/Intel garbage descriptions) |
| 76 | Deterministic/local scoring fallback: extensive same-night exploration for Gemini-quota-outage resilience -- three failed approaches, then a real, validated architecture (81% gate agreement / 0.83 recall / 0.73 precision at n=58), still NOT wired into production |
| 77 | Decision #76's five-item Future Work list closed out: qwen3:8b validated (87% gate agreement), a real escalation trigger found and shipped opt-in, the deterministic scorer promoted to production with a real near-miss caught along the way, two real enrichment bugs fixed, and the reasoning-distillation redo confirmed no missing signal |
| 78 | Future Work item 8 closed: "troubleshooting"/"equipment"/"hands-on"/"servers" added to `_AMBIGUOUS_TERMS`, plus a real second bug found and fixed in the mechanism itself -- ambiguous terms were silently vouching for each other |
| 79 | Future Work item 11 (systematic identity-term audit) found and closed a much bigger version of decision #78's bug: "communications"/"sales"/"customer-facing"/"content"/"creative"/"media" driving 19.75%-26.55% false-positive rates for two entire experience_inventory entries, plus a real occurrence-detection bug in the ambiguous-term mechanism itself |
| 80 | Decision #79's scope note closed: extended the same audit to project_inventory/skills_inventory/certifications, found and fixed 3 more real name-collision/polysemy bugs, and confirmed the remaining drivers (python/automation/web/customer service/logistics/technical support) are genuinely correct signal, not bugs |
| 81 | Automatic quota-reset scoring resume scheduled; escalation trigger bootstrap-revalidated using only existing data, finding wide overlapping CIs and one real redundant keyword ("technician") |
| 82 | Real accuracy spot-check of a live 3-hour local scoring run found a confirmed false-positive (years-requirement stated as a section HEADER, not inline "required" text) affecting ~3% of top-scoring output; fixed and 10 already-scored jobs corrected with zero new LLM calls |
| 83 | Two more real scoring bugs found by extending #82's spot-check across the full 2026-09-08 batch: clinical-license titles (Physician, Medical Assistant/LPN) mis-family'd as customer_facing_or_sales despite the prompt already forbidding it; CS-degree section-header gap (the sibling of #82's years-required fix) plus a Unicode curly-apostrophe miss |
| 84 | Fixed real model-switch thrashing in the hybrid escalation scorer (grouped into two passes instead of interleaved); GitHub-pull correction (Money Machine = CAP Predictor, not a new repo); pinned a reputational-review requirement for future GitHub ingestion; Seattle roadmap flagged for future genericization |
| 85 | GitHub-import gate built: deterministic + LLM reputational flagging, sparse-repo filter, interactive per-repo review, wired into both `applypilot init` (opt-in step) and a standalone `applypilot import-github <username>` command |
| 86 | Real Gemini-vs-local comparison run (30 of the 362 jobs) found a persistent-503 exhaustion gap (fixed) and three compounding real bugs in `extract_years_required` affecting EVERY occupation family, not just software engineering -- 37 more already-scored rows corrected, all moving down |
| 87 | Follow-up to #86, same day: generalized `extract_years_required` away from a verb-inventory dependency per explicit user request, added typo/format tolerance, and fixed a real range-extraction bug (upper bound taken instead of lower) plus a real false-negative the generalization itself introduced |
| 88 | Claude (this session, interactive -- not the `claude_cli` subprocess tier) scored the same 27 jobs Gemini scored in decision #86, using the identical rubric, as a distillation exercise; corrected two "Max plan" doc references to Pro plan and flagged the auto-apply budget-reservation policy for future reconsideration |
| 89 | Patched decision #88's two fixable findings (lookup-table coarseness, decimal-years gap) same day; a real bug in the correction script's own re-run guard silently skipped most of the affected rows until caught and fixed with a full re-derivation audit |
| 90 | Larger, more diverse (5-family) batch scored directly by Claude found the most severe bug of this whole multi-day effort -- non-US/foreign-language postings scoring 7-9 -- plus 8 more real extraction gaps, one of them a real false positive introduced while fixing another; ~38 more rows corrected, fallback `fit_score>=8` pool dropped 40->27 |
| 91 | Cross-section bleed-through in `extract_years_required`'s inline-context window: a "Required Equipment" header on the NEXT line wrongly vouched for an unrelated years-mention one line above it -- fixed by bounding the trailing side of the window to the mention's own line |
| 92 | Continued batch-3 review found a Markdown-bold-wrapped header gap (a real, previously-invisible header shape) plus two NEW false positives it exposed on the very jobs it was meant to fix -- an "or older" age phrasing and an unrelated "About [Company]" history blurb both newly swept into the required-context zone |
| 93 | Gemini confirmed genuinely back online (real successful scoring calls, not just an expired local exhaustion timer), then re-exhausted at real scale a few minutes later -- session's exit condition met; full pytest suite run and one pre-existing, unrelated test-boundary bug fixed along the way |
| 94 | Claude (this session, interactive) scored a fresh 35-job batch directly against the real SCORE_PROMPT_TEMPLATE rubric as a genuine Gemini substitute (not just a comparison exercise) while both Gemini (~5-12h) and OpenAI (~24 days) were quota-exhausted; wrote results to the DB via the real `_flush_score_batch` path under `score_method='claude_direct'`; surfaced a new class of rubric-coverage finding distinct from decisions #82-93's extraction-mechanics bugs |
| 95 | Triaged decision #94's five findings into easy-fix-now vs. document-as-hard, per explicit user instruction; fixed two, documented three, then continued scoring |
| 96 | Scored a second, fresh 30-job batch directly (mostly formal BuiltIn enterprise "Software Engineer II"-style postings, a cleaner source than decision #94's terse HN batch) as continued Claude-direct production scoring; found and fixed one more real, narrow, verified-safe deterministic gap -- "based in Latin America" hidden in body text despite a "USA" location field |
| 97 | Scored a third 35-job batch directly, reaching 100 total `claude_direct` jobs this session; confirmed the decision #95 "founding" fix and the new "commercial experience" prompt clarification both fire correctly on real postings in this batch, and found one more real occupation-carve-out edge case (Marketing Operations Specialist) worth noting for a future audit, not fixed |
| 98 | Scored a fourth 35-job batch directly, reaching 135 total; confirmed the decision #95 "founding" fix twice more on real postings, reconfirmed the title-vs-body seniority gap a third time, and surfaced two more law-enforcement/public-safety-contractor cases plus one minor, not-yet-actioned title-abbreviation finding (AVP) |
| 99 | Found and fixed a real, significant deterministic-gate gap while scoring a fifth 35-job batch: `_TS_SCI_PATTERN` only ever matched the abbreviated "TS/SCI" token, never the spelled-out "Top Secret SCI"/"Top Secret/SCI" phrasing real postings overwhelmingly use -- 191 real live rows newly caught after the fix, verified safe before shipping |
| 100 | Scored a sixth (30-job) batch directly, reaching 200 total; no new code bugs found -- explicit-signal scoring continues to hold cleanly, one real near-miss on a strong CompTIA-A+-matching IT Technician posting capped by an implied onsite location, and a minor LinkedIn-scraper location-field gap noted (not fixed) |
| 101 | Scored a seventh (30-job) batch directly, reaching 230 total; confirmed decision #95's "currently-enrolled student" prompt clarification fires correctly on a real Mastercard posting, and found several more real defense-contractor/non-US cases plus two more genuine IT-support/customer-service near-matches (both capped by onsite locations) -- no new code bugs |
| 102 | Scored an eighth (25-job) batch; found the session's first real job to cross the funnel's fit_score>=8 threshold -- a genuine match, not a scoring artifact |
| 103 | Session's stop condition met: Gemini confirmed genuinely back online via a real probe, closing out the Claude-direct scoring stretch (decisions #94-102) |
| 104 | Resumed real Gemini scoring at scale per explicit user instruction; genuine progress made, quota re-exhausted again after ~12 real LLM calls, no data loss |
| 105 | Explicit user methodology correction: stop benchmarking the local scorer against Gemini and instead treat rubric-grounded, evidence-cited judgment (what Claude-direct scoring already does) as ground truth; user directed continued large-scale auditing "until errors are insignificant," given the stakes (this could help far more than one candidate); a ninth 35-job batch run immediately after found zero new bug classes |
| 106 | Batches 10-11 (70 more jobs) found and fixed two more real deterministic-gate gaps, per the explicit "fix before continuing" instruction -- a location-field Canada/UK omission (with a real US-place-name false-positive caught before shipping), and a "Supervisor" title gap in the canonical seniority pattern |
| 107 | Batch 12 (35 more jobs) found a third real location-pattern gap in the same cluster as #106 -- bare "AUS" country-code abbreviation for Australia missing from `_INELIGIBLE_LOCATION_PATTERNS` |
| 108 | The flagged systematic location-pattern audit from #107 done immediately: found and fixed a fourth, more architecturally significant gap -- multi-location strings listing the US ALONGSIDE a non-US country were wrongly rejected outright, a pre-existing bug affecting every country in the list, not just ones added this session, and the opposite failure direction from every other fix this session |
| 109 | Batch 13 (35 more jobs) run with no new bug classes found; one deliberate non-fix investigated and rejected -- CACI's "required to obtain a Top Secret clearance" phrasing was checked against `_CLEARANCE_REQUIRED_PATTERN` and found to be a real gap, but NOT added, since it would contradict the pattern's own documented, deliberate leniency policy for Secret/Top Secret-tier "obtain"-framed requirements |
| 110 | Batch 14 (35 more jobs) found a fifth real location-pattern gap -- a bare non-US city name ("São Paulo") with no country word in the same string -- fixed narrowly, not as a general city-name expansion |
| 111 | Batch 15 (35 more jobs) found two more real bare-city-name location gaps ("Sydney") plus, far more significantly, a 190-real-row clearance-phrasing gap and a real description-window truncation bug in `_check_ineligible` -- the largest single fix of this session's location/eligibility cluster |
| 112 | Batch 16 (35 more jobs) found the session's second job to cross the funnel's fit_score>=8 threshold; investigated a bare Canadian-province-name location gap ("Ontario - Remote") and deliberately deferred it given a real US-place-name collision risk |
| 113 | Batch 17 (35 more jobs) found zero new bug classes; no jobs crossed the funnel threshold, all explicit years/degree/seniority disqualifiers or genuine near-misses capped by out-of-area location |
| 114 | Batch 18 (35 more jobs) found zero new bug classes; no jobs crossed the funnel threshold, three more genuine near-misses (Property Damage Tech, Maintenance Technician/Trainee, Process Operator) scored 7 on real domain fit and low formal bars |
| 115 | Batch 19 (35 more jobs) found the session's third job to cross the funnel's fit_score>=8 threshold; a real Toronto-based OpenAI posting confirmed the bare-"Ontario"-location gap (Future Work item 17) is a live, active issue, not just a historical curiosity |
| 116 | Batch 20 (35 more jobs) found zero new bug classes, closing out this session's Claude-direct scoring stretch at 674 total; Gemini then confirmed genuinely back online via a real probe (per decision #93's methodology, not trusting the expired-timer alone) and a real production `run score` batch launched at scale |
| 117 | The real `applypilot run score --limit 5000` production run (launched decision #116) completed: 3,429 jobs attempted, 214 real Gemini scores landed, three more real jobs crossed the funnel's fit_score>=8 threshold, before quota re-exhausted at scale -- Gemini exhausted again (14.4-24h), resuming Claude-direct batches |
| 118 | Batch 21 selection found ZERO fresh jobs -- the real scoring backlog within the funnel's 14-day age window is now fully exhausted, meeting this session's own stop condition in its strongest form (Gemini confirmed back online AND no more problems/backlog to work on); 34 real `state='scored'` jobs are sitting idle awaiting tailoring, reconfirming the standing scoring-vs-apply-bottleneck concern |
| 119 | Real production bug found and fixed by the user directly: jobs that exhaust all 5 cloud scoring retries were being marked `score_failed` FOREVER with no recovery path, even though a working local scorer (decision #76-77) already exists -- `_flush_score_batch` now tries it once as a genuine last resort before giving up |
| 120 | User clarified the REAL purpose of the Claude-direct push: it's an audit/QA sampling exercise (confidence that the deterministic rubric layer has no more extraction gaps), not a backlog-clearing exercise -- resumed toward 1000 by widening `select_batch.py` to unbounded job age, since the within-14-day pool is genuinely empty; batch 21 (35 jobs) found zero new bugs |
| 121 | Batch 22 (35 more jobs, via widened `select_batch.py 35 0`) found zero new bug classes; no jobs crossed the funnel threshold, several genuine near-misses capped by out-of-area Seattle/Scottsdale/Boston onsite locations |
| 122 | Batch 23 (35 more jobs, via widened `select_batch.py 35 0`) found zero new bug classes; no jobs crossed the funnel threshold, three duplicate Axon "Field Inside Sales Representative" postings (Atlanta GA / Florida-Remote, both fully remote) scored consistently with the earlier Charlotte NC posting from batch 22 |
| 123 | Batch 24 (35 more jobs, via widened `select_batch.py 35 0`) found zero new bug classes; no jobs crossed the funnel threshold, including a real Cisco hardware-design posting genuinely located in the candidate's own RTP, NC area that was still correctly disqualified on explicit years/degree grounds |
| 124 | Batch 25 (35 more jobs, via widened `select_batch.py 35 0`) found zero new bug classes; no jobs crossed the funnel threshold, two genuine near-misses (Intel TEM Technician / Ocotillo Failure Analysis Technician, both satisfiable via a bare technical certification or 1-year hands-on bar) scored 6, capped by onsite Oregon/Arizona locations |
| 125 | Batch 26 (35 more jobs, via widened `select_batch.py 35 0`) found zero new bug classes; no jobs crossed the funnel threshold; a real live confirmation of Future Work item 17 (bare-Canadian-location gap) caught and correctly scored non_us_only by reading the full body text, not the pre-filter |
| 126 | Batch 27 (35 more jobs, via widened `select_batch.py 35 0`) found zero new bug classes; no jobs crossed the funnel threshold; two more real Intel Module Equipment Technician postings (Phoenix AZ) scored 6, same qualifying shape as decision #124-125's finds |
| 127 | Batch 28 (35 more jobs, via widened `select_batch.py 35 0`) found zero new bug classes; no jobs crossed the funnel threshold; a real Ashby posting explicitly self-declared an exclusion matching the candidate's IT-support background verbatim |
| 128 | Batch 29 (46 jobs, via widened `select_batch.py 46 0`) closed out the session's 1000-job Claude-direct target exactly; a real fourth job crossed the funnel threshold (Motorola Solutions 'Customer Support Technician I,' fully remote, genuine low-bar IT-support match) |
| 129 | User explicitly authorized moving the REST of the real scoring backlog to the local deterministic-fallback scorer, per decision #105's own stated trigger condition ("once we have evidence there aren't more extraction gaps") -- widened `run_deterministic_fallback_scoring`'s scope beyond decision #76's original quota-cooldown-only design, and launched the real backlog run in the background |
| 130 | Real launch of decision #129's backlog run required a redo: the first launch attempt used `nohup ... &` inside a backgrounded shell call, which detached the actual scoring process from the harness's own task tracker (the tracked "task" was just the launcher line exiting immediately, not the multi-day scoring work) -- killed and relaunched as a directly-tracked foreground background command instead |
| 131 | User pushed back on the raw system impact of decision #130's run and asked to throttle it -- stopped, real machine specs confirmed (12GB RAM with only ~460MB free even at idle, mechanical HDD, no GPU -- `ollama ps` showed `size_vram: 0` for the loaded model, confirming pure CPU/RAM inference), then relaunched with both the Python client and the Ollama inference server set to Below Normal OS priority |
| 132 | User explicitly redirected scope: 1000+ Claude-direct scores is enough real material -- stop growing the scoring backlog and get back to testing tailor/cover (per decision #118's own standing bottleneck memory). Killed the still-running local scoring backlog run; ran a real `applypilot run tailor --limit 3` against the waiting `state='scored'` pool -- 1 job (Smartsheet SDR) succeeded end-to-end with a real approved resume, 1 hit a real, confirmed bug: a persistent cloud 503/429 on the ONLY remaining fallback-chain entry never marked itself exhausted before raising, so `client.has_cloud_available()` wrongly reported cloud as still up and tailor.py never redirected to the degraded/local path -- the exact "stuck on a dead cloud provider, no real fallback" failure class decision #119 fixed for the scorer, now confirmed to exist in tailor.py's own cascade too, per the user's direct suspicion |
| 133 | Local tailor tested for real for the first time at more than n=1: phrase banks generated for 9 of 11 remaining entries via forced-local-only `expand-bank`, degraded-mode tailoring re-verified against a known-good job, a real fabricated-employer bug found and fixed (a "Stand-Up Comedian" entry invented from nothing, since it existed nowhere in profile.json), and an honest quality-ceiling finding: degraded mode is safe (never fabricates) but visibly rougher than cloud |
| 134 | Real, previously-unknown `applypilot status` crash found and fixed while checking pipeline state during wrap-up: any Unicode character in rich-rendered CLI output crashes on this Windows console (cp1252, not UTF-8) |
| 135 | Session close: local scorer run for exactly one final hour, per explicit user instruction, before shutting the machine down for the night -- 136 more real jobs scored, all below the funnel threshold, no crashes or notable events |
| 136 | Standup grounding re-verified per user follow-up: decision #133's root cause confirmed accurate (not a miscommunication), a second, genuinely separate piece of pre-schema-era hardcoded categorization found and explained, a real latent validator whitelist bug fixed, and spelling corrected per explicit instruction |
| 137 | Cover-letter degraded mode built per Future Work item 22's outline: closes the "zero fallback on cloud exhaustion" gap, reusing the resume phrase-bank pipeline for the evidence paragraph and template-filled paragraphs for hook/company-fit/close; found and fixed 3 real bugs during testing, including a pre-existing validator defect independent of this change |
| 138 | Real forced-local-only cover-letter batch run (4 real jobs) found a significant, previously-undiscovered requirement-extraction gap affecting BOTH resume tailoring and cover letters -- fixed and verified against the full 31,710-job corpus, zero regressions, 553 jobs (later 432 after a companion fix) recover real content that previously extracted zero requirement lines |
| 139 | Benefit-line filter gap (from decision #138) fixed and verified against the full corpus; the recurring banned-phrase-from-editor-output bug fixed with a real, opt-in check threaded through the shared editor -- confirmed live, twice, that it now catches what it used to miss |
| 140 | Word-count shortfall traced to a concrete mechanism (per direct user request) and partially closed: FIT paragraph was silently discarding real, already-vetted evidence that EVIDENCE's own cap never used -- fixed, verified against real embeddings before shipping, and confirmed live across 5-6 real jobs in a fourth forced-local-only batch |
| 141 | The "Requirement:" bleed fixed, revealing (and then closing) a much bigger real pattern: 13,805 lines across the corpus are shaped "Label: content," splitting into three genuinely different real categories, two of which are now handled |
| 142 | Filler-phrase variety built (deterministic pools + an optional bounded local-model polish) per explicit request; two real regressions found and fixed via real forced-local-only batches before the feature could be trusted -- variant pools with unequal lengths, and a too-permissive length-safety check |
| 143 | Retried the 2 stale real cover_failed jobs against current scoring standards (1 confirmed still qualifies and got a real cover letter shipped, 1 corrected down and excluded); root-caused decision #142's Future Work #25 regression (not a stale-cache issue after all); built and shipped `judge_cover_letter`, a resume-generator-parity advisory LLM pass for cover letters; confirmed the standup-goes-with-communication-roles hypothesis (Future Work #21) with a real test; explained the Freelance Photography "no survivors" case as a data-thinness issue, not a recognition failure |
| 144 | Real n=500 bake-off for decision #143's item-2 fix found a working, verified option (not yet shipped); full-name cover-letter sign-off + "Your team" wording fixed; local model can now be primary via `APPLYPILOT_LOCAL_LLM_URL` alone; the standup "confirmation" from decision #143 was retracted after a larger, real-job sample exposed the same class of false positive decisions #79-80 already found and couldn't fix here (empty own-text) |
| 145 | Follow-up housekeeping session: profile data-quality cleanup, the escalation trigger's long-open revalidation finally closed decisively, a new evidence-matching false-positive found and fixed, and two real, non-forced production runs of tailor/cover surfaced one severe, previously-underestimated gap |
| 146 | A missing location-awareness gap closed in the deterministic scorer; the GitHub importer run for real (found and removed a real reputational-risk project); a cross-pollination map written up; real stand-up profile facts added; and, most significantly, a REAL FABRICATION FOUND SHIPPING IN LIVE COVER LETTERS during a real backlog run -- three distinct root causes traced and fixed, plus a new general safety net (judge-verdict-blocks-on-fabrication) that already proved itself catching a fourth case in production |
| 147 | Ran the real backlog-clearing batch and found the actual bottleneck: 91% of the "backlog" had no application_url at all (structural, not staleness) -- reversed the pre-tailor URL gate's now-outdated cost rationale so those jobs get real materials anyway; traced a real occupation-misclassification bug behind an OpenAI job scoring 8 despite no seniority-title match; confirmed LinkedIn's own Easy Apply exclusion was never a deliberate policy, just a missing URL |
| 148 | Root-caused and fixed the dominant (53%) real cause of the "no matching evidence" bucket for Talent.com/SimplyHired jobs -- a single spurious bullet character crowding out a much richer, unmarked real requirements list -- then found and fixed two real regressions the fix itself introduced, both caught by the existing test suite before shipping |
| 149 | Re-verified decision #148's fix at both the sample and corpus level; confirmed the remaining low-extraction jobs are the already-documented, already-deferred run-on-paragraph case (decision #61), not a new gap |
| 150 | Semantic-match-threshold regression built and shipped, per the user's explicit authorization ("The semantic-match threshold scope seems pretty above board to me if you want to go ahead and build it") -- real n=98 calibration found 0.30 was a genuinely poor guess (80% of admitted pairs spurious with zero recall benefit), raised the default to 0.35, a strictly cost-free improvement over 0.30 |
| 151 | Decision #131's flagged "known limitation" recurred for real: Ollama's server process drifted back to Normal OS priority mid-run, found live and fixed a second time; a new, not-yet-investigated observation logged for a future session (intermittent qwen3:8b native-API timeouts under this run's real load) |
| 152 | The overnight local backlog-scoring run finished cleanly on its own: 2,000/6,877 unscored jobs processed, hybrid qwen3:1.7b/8b model, per-job DB flush confirmed safe throughout -- real final numbers pulled, one real concern flagged (not investigated) for a future session |
| 153 | A real, live tailor-batch run found and fixed a genuine staleness bug in the cloud-exhaustion-to-degraded-mode handoff |
| 154 | Decision #153's fix verified live: a fresh, unbounded tailor batch hit zero provider_unavailable failures, tailored count rose 341->369 |
| 155 | Live cover-letter batch found the judge correctly flagging self-defeating filler phrasing ('fake my way through it', 'not from the sidelines', 'I do not have one specific example') that shipped anyway since it wasn't tagged FABRICATION -- reworded to remove the self-deprecation while staying fact-free |
| 156 | Live apply-stage testing found and fixed a real resume-format-fallback bug (all 13 jobs hard-failed pre-fix); 2 real applications submitted after the fix; expired-posting waste identified, not yet fixed |
| 157 | Pre-apply expired-posting check: fast httpx pre-check (404/410/451, redirect-to-root) before Chrome/Claude spawn |
| 158 | `testpaths = ["tests"]` so a bare `pytest` stops recursing into a nested experiments venv |
| 159 | Sparse-marker paragraph fallback pulled an unmarked header in as a fake requirement -- fixed; benefit-drop warning preserved |
| 160 | Two `test-local` CLI tests leaked the real `~/.applypilot/.env`; now mock `load_env` + fake cloud key |
| 161 | Degraded-mode cover batch: 17/34 generated, 3 real fabrication catches; Field Sales Rep word-count shortfall found |
| 162 | Policy: cliche phrasing is fine in cover letters -- `CL_BANNED_PATTERNS` emptied; `_stretch_to_min_words` adds opinion-only filler (never factual) |
| 163 | #162 verified live: 17/17 cover letters generated, 309-343 words, zero fabrication rejects |
| 164 | `apply` hung forever when a worker's `--limit` split was 0 (silently continuous); `continuous` now passed explicitly. Use py-spy for stuck threads |
| 165 | Playwright MCP failed to connect in apply subprocesses (`browser_tool_unavailable`) -- pre-existing, intermittent |
| 166 | Unattended apply spun ~4h in session-limit backoff, burning the shared Pro pool. Lesson: stop unproductive loops early; `TaskStop` unreliable, use `Stop-Process` |
| 167 | MCP-connect race is intermittent, not reproducible; shipped a real CDP-readiness poll + one MCP-connect retry |
| 168 | Retry verified live; fallback scorer `ORDER BY discovered_at DESC`; capped transient-apply-failure requeue (3); dashboard `file://` document links |
| 169 | Haiku escalation when a fresh `successful_paths` memo exists; deterministic engine `_FIELD_SELECTORS` map + Workday personal-info slice (unverified live); `qa_knowledge` answers |
| 170 | Skyvern spike: Gemini `safety_settings` bug, qwen3:8b timeouts, qwen3:1.7b non-JSON -- no working run; Claude OAuth session deliberately not reused |
| 171 | MCP-connect retries 2->3; stale CLI hints fixed (`apply --reset-category`, removed `human-review`) |
| 172 | No-`application_url` jobs were stranded in `ready_to_apply`; `acquire_job` now sweeps them to `manual_only` |
| 173 | Mid-run Chrome death traced to Windows `ScheduledDefrag`; HITL retry loop checks CDP liveness and relaunches once |
| 174 | Gmail OAuth set up; `store_account` upserts on (domain, email); local Gmail token expiry is unreliable; Forestar scam email caught |
| 175 | `test_case` column: deliberately low-score jobs reuse an approved resume/CL pair to exercise apply mechanics safely |
| 176 | MCP-race defense layers 1-2: WebSocket CDP readiness + 2s staggered worker startup; layers 3-4 (diagnostics, persistent MCP) unbuilt |
| 177 | Quota-cooldown scoring errors fall back to the local hybrid scorer immediately instead of after 5 retries (~27h) |
| 178 | Backlog cleanup: deleted 28,059 pre-gap archived/low_score rows (backup taken); 36,316 -> 8,257 jobs |
| 179 | `run_tailoring` claim loop wrapped in `write_with_retry` (database-locked crashes) |
| 180 | `manual://` stub jobs no longer re-selected for enrichment (first branch requires `detail_error_category IS NULL`) |
| 181 | `_mark_enrich_result` missing lock retry found live (logged; fixed in #182) |
| 182 | `_mark_enrich_result` success + error branches wrapped in `write_with_retry` |
| 183 | OPEN: `write_with_retry` still exhausts 8 retries under sustained `--stream` load (tailor/cover flush) -- capacity question, not a missing wrapper; llama-server priority drift |
| 184 | LinkedIn `application_url` is null because jobspy never populates `job_url_direct`; test-case email forwarding; `applypilot mark-test-case` |
| 185 | Session-limit loop: orchestrator waits on the durable `claude_status` exhaustion signal (~30 min probe) instead of a blind 300s retry |
| 186 | SEVERE: apply agent wrote its own ungated Playwright script when MCP failed. Fixed with an explicit `--allowedTools` whitelist (playwright + 3 Gmail tools) |
| 187 | `apply --human-first`: human clicks Apply/Hand Off on LinkedIn, automation takes over; new `manual_only -> applying` edge |
| 188 | Human-first live-tested: banner injection retried and re-injected every poll; hand-off URL persisted; MCP race + usage limit recurred |
| 189 | `extract_with_llm` propagates errors (`LLM error: ...`) so quota cooldowns classify retriable, not permanent; stale-enrichment-completion collisions noted |
| 190 | SEVERE: HITL stdin fallback treated EOF as "done", auto-dismissing every pause in non-interactive runs -- fixed |
| 191 | Human-first Hand Off hard-blocked on linkedin/indeed/ziprecruiter/glassdoor + `sites.yaml` blocked list |
| 192 | #189 fix holds for 503 errors too; jobspy insert path also hits #183's lock exhaustion |
| 193 | `recover_stale_claims` wrapped in `write_with_retry`; llama-server priority drift recurred (watchdog needed) |
| 194 | First human-first batch: 1 applied, 1 `browser_unreachable`; flag glyph removed from the older HITL banner |
| 195 | First normal apply batch: 5 applied over ~7h, mostly waiting out the shared usage limit; #185 backoff worked as designed |
| 196 | Midas application found only in transcripts (`company` NULL for CareerPlug); Lumen SMS escalation was valid; speed comes from haiku swap, not deterministic engine |
| 197 | `tracking/sms_client.py` Twilio relay (read-only) + `applypilot sms --setup`; NOT wired into apply |
| 198 | Wizard Step 6: SMS relay setup + email verification; credentials only in `.env` |
| 199 | Google Voice relay via Gmail forwarding (free) becomes the default; Twilio kept as an option |
| 200 | ADB relay (`adb_sms_client.py`, reads phone SMS over USB); `sms --setup` provider choice |
| 201 | ADB auto-detect in common install paths; stalled apply batch cleaned up |
| 202 | ADB relay working live; missing `Confirm` import fixed; `_GENERIC_FIELD_SELECTORS` autocomplete fallback (schema Stage 1) |
| 203 | Full-day Gemini exhaustion exposed smartextract Phase 2 local-model JSON parse failures (safe, wasteful) |
| 204 | `--stream` exited while stale tailor/cover claims were the only work; `count_stale_claims` added to the pending count |
| 205 | Schema Stage 3: deterministic engine dispatches any ATS to `_run_generic` (fills, never submits) |
| 206 | Okta posting checked (legit); data-broker exclusion keywords; email scam detector; `labor_signals.py` positive bonus (+2 cap) |
| 207 | Non-blocking HITL (`--non-blocking-hitl`): a paused session finishes on a background thread under a synthetic worker id |
| 208 | Non-blocking HITL live-tested; `_dispatch_needs_human` unifies 3 call sites; `credits_exhausted` replicated in background |
| 209 | `backfill_states` wrapped in `write_with_retry` (was aborting 9 scrapers) |
| 210 | Score stage stalled ~9h; `pending_score` cutoff `< 5` -> `<= 5` so the exhaustion fallback is reachable |
| 211 | Root cause of #210: an unguarded `_count_pending` exception killed the `--stream` stage thread; now logged and retried |
| 212 | Global `threading.excepthook`; `dispatch_hitl` background worker and dashboard health loop guarded |
| 213 | #212 validated live; launcher's 300s DB retry still exhausted under dual-process load; 2 jobs left in `applying` |
| 214 | Smartextract JSON rescue, flaky test fixed, `backfill_categories` retry, `backfill_companies` wired into `init_db` (14,634 rows) |
| 215 | CLAUDE.md 353KB->50KB (full text to archive/future_work); clean-checkout tests/lint/CI fixed; POSIX scheduler lock; archive-never-delete rule |
| 216 | write_with_retry + HTTP handlers/store_qa now roll back on any error: leaked open transactions held the DB write lock (candidate root cause of #183) |
| 217 | Discovery location filter consolidated (4 drifted copies); whole-word accept matching in discovery + scorer; wizard writes location filter (E9) |
| 218 | One shared LLM-JSON parser (llm_json.parse_llm_json); phrase-bank generation no longer loses rounds to fenced/prose/<think> replies |
| 219 | FW53: opt-in email alerts (APPLYPILOT_NOTIFY_EMAIL) for needs_human pauses and Claude exhaustion/recovery; `applypilot notify --test` |
| 220 | FW48: Hand Off confirmation leads with the site name in large text; full URL folded under 'Show full link' |
| 221 | FW36 answered by experiment: pinned playwright-mcp 0.0.75 `--port` serves sequential MCP sessions on one Chrome; design in docs/scoping/session_architecture.md |
| 222 | E6: ruff now lints tests/ in CI; dead test code fixed and one test strengthened. E5 needed no change (copies are one-line delegates) |
| 223 | FW55: extract_company returns the tenant for multi-tenant ATS hosts and None for opaque ones (was the vendor name); backfill corrects old rows |
| 224 | FW17: bare Canadian province locations rejected, skipping US namesakes (Ontario CA/OR/NY, New Brunswick NJ, Alberta VA, Ontario County); corpus check pending |
| 225 | FW27: GitHub-import review shows the text that triggered each flag; model quotes shown as quotes only if found verbatim |
| 226 | FW49/FW45: Hand Off button counts up with live worker status; pause banner always says 'Do the step, then click Done' |
| 227 | FW51: extension content script declines cookie banners (known CMPs, or reject-only labels inside 'cookie' containers); never clicks accept/EEO buttons |
| 228 | FW60: wizard choice prompts name the default once via wizard.init.ask_choice |
| 229 | FW26: expand-bank asks four factual follow-up questions when an entry yields no phrase-bank survivors; saves to profile.json only after confirmation |
| 230 | E8 root cause: smartextract inserted 'enriched' jobs with no full_description (never scoreable, re-scraped and discarded every pass); fixed + init_db repair |
| 231 | FW65: deterministic engine pauses 0.6-1.8s between fields (APPLYPILOT_DETERMINISTIC_PACING=0 disables); prompt already feeds known facts to Claude |
| 232 | E1 investigated: markdownify>=0.14.1 is a security pin (GHSA-7mpr-5m44-h73r) and every newer jobspy caps markdownify<0.14 -- stay on Python 3.11 |
| 233 | docs/scoping/ added: staged plans + decisions needed for session architecture, login/2FA, budget/escalation, scoring, LinkedIn/locality, engineering |
| 234 | User decisions: no login/2FA automation (FW43/44 declined); Claude stays reserved for apply (FW13); locality data US-first, GitHub-hosted (FW71); FW72 security review added |
| 235 | Persistent Playwright MCP HTTP server per apply worker is the default; failed connect restarts it once, then falls back to per-job stdio |
| 236 | FW28 requirement-framing classifier in the local scorer: additive, shadow mode until scripts/validate_requirement_framing.py passes live |
| 237 | FW39 and FW52 declined by the user; E3 lock-error baseline recorded (~2-3 give-ups/hour, only in full discover+4more runs) |
| 238 | `APPLYPILOT_LOCAL_FIRST=1` reorders the fallback chain to try local before cloud (decision #144 only fixed the no-cloud-key case); Best Buy test job marked `applied_manually` |
| 239 | FW78: human-first banner Skip button + `unavailable` outcome built; a real `textContent`-vs-`innerHTML` HTML-entity bug fixed live along the way |
| 240 | FW41 (live queue UI), FW47 (flag-job + bug-report icons), FW50 (document dedup, storage half only) built; a real cross-filename dedup bug caught by the new test suite before shipping |
| 241 | FW79 root-caused (per-poll fresh-Node-spawn cost + Defender/RAM pressure) and mitigated (15s->30s); 9 real CDP-unreachable events found across today's logs, already self-healing via #173 |

---

## Known Technical Gotchas

1. **Gemini thinking tokens**: 2.5+ models use thinking tokens that consume max_tokens budget. A simple response needs 30 tokens, a bullet rewrite needs 1200+.
2. **Agent log timezone**: Log filenames use local time, DB `last_attempted_at` is UTC. Dashboard matcher converts UTC→local.
3. **Singleton LLM client**: `llm.py` reads env vars at module import. Call `config.load_env()` BEFORE importing.
4. **Editable install**: `pip install -e .` means source edits take effect immediately.
5. **gemini-2.0-flash deprecated**: Use `gemini-2.5-flash` or newer for new API users.
6. **Docker MCP Toolkit interference**: If Docker Desktop is installed with MCP Toolkit, it exposes `mcp__MCP_DOCKER__browser_*` tools that shadow the local Playwright MCP. These Docker tools can't access the host filesystem, breaking resume/cover letter uploads. Fix: `--strict-mcp-config` in the claude subprocess command.
7. **qwen3 local calls need more `max_tokens` headroom than the visible output suggests**: the local model burns part of its budget on hidden internal reasoning before writing anything. Confirmed live twice on 2026-09-04: `max_tokens=200` and `max_tokens=600` both produced "Null/empty content" on real, previously-working-looking prompts; `max_tokens=1024`/`4096` fixed it in some but not all cases (one case timed out instead at 4096 -- see decision #70). Don't assume a small `max_tokens` is safe for a local qwen3 call just because the expected visible output is short. **2026-09-06 (decision #74)**: the older `/no_think` text-prefix convention referenced in earlier notes has been REMOVED and replaced with the real `"think": false` API field (sent as a top-level JSON payload field, not a message-content prefix) -- real A/B testing found the text-prefix version doesn't reliably suppress qwen3's reasoning even when correctly applied, while the API field does. This headroom gotcha still applies even with the field set correctly (300 tokens still failed 0/6 in testing; 700 was the smallest value that worked reliably) -- thinking-suppression and token-budget are separate, both-required fixes, not substitutes for each other. See also `APPLYPILOT_LOCAL_OLLAMA_NATIVE` (opt-in, `llm.py`) for Ollama's *native* `/api/chat`, which was both faster and more reliable than the OpenAI-compat shim even with the field fix in place.
8. **`.env` keys can be in two different places**: `config.load_env()` loads `~/.applypilot/.env` explicitly, THEN a bare `load_dotenv()` that searches from cwd upward -- a repo-root `.env` (e.g. `C:\Users\phili\Projects\resume-agent\.env`) is a real, separate, valid source of `GEMINI_API_KEY`/`OPENAI_API_KEY` and is easy to miss if you only check `~/.applypilot/.env` (see decision #67's correction to #63).
9. **`adb`'s background server can block safely ejecting the phone**: any `adb` command (`devices`, `content query`, etc. -- everything `adb_sms_client.py` runs) auto-starts a persistent background `adb.exe` server process if one isn't already running, and it stays running afterward, holding the USB connection open. If Windows refuses to safely eject the phone with "a program is still using it," check for a lingering `adb` process and run `adb kill-server` (or `Stop-Process`) to release it -- confirmed live 2026-09-25, the exact ADB relay work this session shipped caused this on the first real use.

---

## Where Things Live

| What | File |
|------|------|
| Full text of every decision | `docs/decisions_archive.md` |
| Future Work items (full text, numbered) | `docs/future_work.md` |
| Prioritized backlog (Eisenhower quadrant, dependencies, stage, cloud-ok vs needs-live-machine) | `docs/backlog.md` |
| Pattern audit findings and checklists | `docs/audit_2026-10.md` |
| Archived files and the archive policy | `docs/archive/README.md` |
| Scoping docs for large or gated backlog items | `docs/scoping/README.md` |
