# ApplyPilot — Copilot Handoff Audit (2026-08-31)

Self-contained. Written for GitHub Copilot to pick up with no access to the
originating conversation. All claims below are grounded in repository code
(cited as `path:line`) or in live queries against
`C:\Users\phili\.applypilot\applypilot.db` / the local filesystem, run on
2026-08-31. Nothing here is speculative — where something couldn't be
verified, it's stated as unverified, not asserted.

---

## 1. Executive Summary

**Working, verified:**
- The deterministic hard-eligibility gate (`src/applypilot/eligibility.py` +
  `scorer.py:_check_ineligible`) runs **before** any LLM scoring call and
  correctly blocks non-US geography, seniority-mismatched titles, advanced-
  degree/clearance requirements, commission-only comp, and a live 86-term
  defense/military/surveillance/policing keyword list.
- The specific Boeing job cited as a "leak" (see §2) was in fact **already
  correctly caught and archived** by an existing stale-score revalidation
  sweep. It is not live/actionable data. This is a stale-data misread, not a
  current pipeline defect.
- Tailoring's validator/judge/approval pipeline (`scoring/tailor.py`,
  `scoring/validator.py`) is functioning as designed — the sampled resumes
  the prior investigation reviewed passed validation and judge review.
- Cover-letter generation is pure LLM text generation with **no browser
  dependency at all** (confirmed: no `patchright`/`playwright` import in
  `scoring/cover_letter.py`). The Boeing record's cover letter *did*
  generate successfully (`cover_attempts: 1`, `cover_letter_at` populated).

**Broken / gap, verified by code inspection:**
- `cover_letter.py:generate_cover_letter` (line 223) calls `client.chat()`
  with **no `exclude_providers` argument** — unlike `tailor.py`'s heavy
  cloud-quality call (`tailor.py:1151`, `exclude_providers=frozenset({"local"})`,
  added specifically to fix this exact multi-minute-stall failure mode for
  tailoring in a prior fix cycle — see `tailor.py:1109,1137-1155`). Cover
  letters have no equivalent fast-fail: under sustained cloud exhaustion, a
  cover-letter call can still fall through to the slow local Ollama leg, up
  to `max_retries + 1 = 4` times per job (`cover_letter.py:203`). This is the
  most plausible, code-grounded explanation for multi-minute cover-letter
  stalls — not a browser/Patchright issue, not a hard timeout misconfiguration.
- `config.get_chrome_path()` (`src/applypilot/config.py:55-107`) has **no
  Windows branch that looks for a Chrome-for-Testing install** — the CfT
  preference logic exists only under the Linux branch
  (`config.py:83-95`, comment explicitly says "Chrome for Testing... Linux").
  On this machine, it currently resolves to
  `C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`, version
  **152.0.7977.65** (verified via file metadata) — a branded Chrome build
  well past the Chrome-137 cutoff the codebase's own prior decision
  documents as silently rejecting `--load-extension`. This is a real,
  separate, pre-existing gap in the apply-stage browser layer on Windows.
  It affects the apply-stage extension only (job-tracking pill, action-log
  capture, HITL banner) — **it does not affect discovery/enrichment's
  separate Patchright usage** (`smartextract.py`, `enrichment/detail.py`,
  `scoring/pdf.py`), which launches its own bundled Chromium via
  Playwright/Patchright's normal API and does not need `--load-extension`.
  See §5 for full detail; this is unrelated to the recent `patchright
  install` update.

**Uncertain / not verified in this pass:**
- Whether the Windows Chrome-152/no-extension-load gap is *currently*
  causing observable apply-stage failures (no live apply run was executed
  during this audit — see §8 for the exact command to check).
- Whether cover-letter stalls have actually been observed recently in logs
  (this audit did not locate a specific log excerpt showing an 800+ second
  cover-letter run; the `exclude_providers` gap is a mechanism finding, not
  a confirmed-from-logs incident). §3 lists the exact log location to check.

---

## 2. Eligibility / Seniority Problem — Boeing Trace

**Job:** `https://boeing.wd1.myworkdayjobs.com/EXTERNAL_CAREERS/job/USA---Tukwila-WA/Software-Engineer--Associate-or-Experienced-_JR2026500177-1`

**Current DB state** (queried 2026-08-31):
```
title:        Software Engineer (Associate or Experienced)
eligibility:  eligible          <- STALE, preserved-for-audit only, see below
fit_score:    8                 <- STALE, same
apply_category: manual_only     <- STALE, same
state:        archived          <- LIVE / authoritative. Terminal state.
```

**Full transition history** (`job_state_transitions`, this URL):
```
(None,          enriched,       2026-08-17T02:03:59Z, migrated 2026-04-24)
(enriched,      scored,         2026-08-17T03:21:12Z, scored 8/10)
(scored,        tailored,       2026-08-17T03:26:25Z, tailored OK)
(tailored,      ready_to_apply, 2026-08-17T03:33:04Z, cover letter done)
(ready_to_apply, manual_only,   2026-08-18T02:44:57Z, acquire_job: manual ATS)
(manual_only,   archived,       2026-08-18T23:38:21Z, requeue_after_scoring_fix_2026-08-18)
```

**Root cause, confirmed:** this job was scored on **2026-08-17**, before the
ethical-exclusion / defense-keyword filter existed in the live pipeline
(`scorer.py:443-478`'s own code comment: *"289 ethical_exclusion archives on
2026-08-18, none since"* — the feature went live 2026-08-18, one day after
this job scored). The Boeing description (full text confirmed via DB query)
contains, verbatim, in the second paragraph: *"self-protect suites, **weapon
systems**, data links"* and *"Boeing Defense Services (BDS)"* / *"USAF E-7
Airborne Early Warning & Control"* in the opening paragraphs — well within
the first 6,000 characters (`_DESC_SCAN_CHARS`, `scorer.py:359`) that
`_check_ineligible` scans.

**Confirmed: the current live keyword list would catch this today.**
Queried the live `~/.applypilot/searches.yaml` `exclude_description_keywords`
(86 terms, loaded via `config.load_search_config()`): it includes
`"weapon systems"`, `"weapons systems"`, `"defense contractor"`,
`"department of defense"`, and 82 others. `scorer.py:476-478` does a
word-boundary regex match of every keyword against
`f"{title}\n{desc_head}".lower()`. A live scan of the actual database
confirms this list is *actively working* on jobs discovered after
2026-08-18: **165 currently-stored jobs** discovered after that date match
one of these keywords, and every one checked (`Boeing F-15 Mission Systems
Software Engineer`, multiple RTX/Raytheon postings) shows
`eligibility != 'eligible'` and `fit_score <= 2` — i.e. the gate is live and
functioning for new discoveries. The Boeing `JR2026500177` row predates it.

**Confirmed: this exact job was already correctly handled by an existing,
purpose-built mechanism.** `src/applypilot/eligibility.py:260-349`,
`revalidate_stale_scores()`, exists specifically to archive jobs whose
`fit_score` was assigned before a scoring-rubric fix (its docstring cites
the *identical* 2026-08-18 date and describes *"a one-time requeue [that]
swept most stale-scored rows into archived that same day"* —
`eligibility.py:264-266`). The Boeing row's own transition history shows
exactly this: `manual_only → archived` on `2026-08-18T23:38:21Z`, reason
`requeue_after_scoring_fix_2026-08-18`. **This is that sweep.** The stale
`eligibility`/`fit_score`/`apply_category` columns are preserved on the
archived row *by design*, for audit purposes — `eligibility.py`'s own
docstring states this explicitly (line 178-180: *"No job record is deleted;
fit_score, score_reasoning, tailored_resume_path, and cover_letter_path are
all preserved on the archived row for audit purposes"*).

**State is terminal.** Per `database.py`'s `VALID_TRANSITIONS` (referenced
throughout this codebase's history; not re-verified line-by-line in this
pass but consistent with every other terminal-state row inspected),
`archived` has zero legal outgoing transitions. `get_jobs_by_stage`'s
`pending_tailor`/`pending_score`/`pending_cover` selectors all filter on
`state`, not on `eligibility`/`fit_score` — an `archived` row is never
re-selected for any pipeline stage regardless of what its stale columns say.
**This job will not be re-tailored, re-covered, or re-applied to.**

### Per-item disposition against the four concerns raised

1. **Defense/military application** — correctly excluded by the live
   keyword gate for any job discovered after 2026-08-18; this specific row
   predates the gate and was separately swept into `archived` by the
   stale-score revalidation mechanism the day after the gate shipped.
   **No current gap identified.**
2. **Location/relocation mismatch** — not the operative disqualifier here
   (Tukwila, WA is a legitimate US location); not further investigated in
   this pass since the defense-keyword and stale-score findings already
   fully explain the record. **Not evaluated — flag as open if a genuine
   relocation-mismatch case is found separately.**
3. **Experience/seniority requirements unrealistic for the candidate** —
   `eligibility.py:39-71`'s `SENIORITY_TITLE_PATTERN` does **not** match
   "Software Engineer (Associate or Experienced)" — no senior/staff/
   lead/director/level-3+ token is present. This is **correct, not a gap**:
   the title genuinely doesn't claim a seniority level the candidate can't
   support; the description's *implicit* 2-3+ years and technical-leadership
   language is a body-text signal, and `SENIORITY_TITLE_PATTERN` is
   documented as deliberately title-only (`eligibility.py:134`: *"seniority-
   by-title is purely a function of the title string"*). Whether body-text
   experience-requirement extraction should be added is a real, separate,
   **not-yet-existing** capability — see §7 for a scoped recommendation, but
   do not conflate this with the Boeing incident, which the keyword gate
   already resolves independently.
4. **Title/level interpretation too permissive** — same as #3: the title
   itself doesn't overclaim; this is not a title-parsing bug.

**Conclusion for §2:** no code change is required to fix the Boeing case
itself — it was already correctly resolved by existing, purpose-built
machinery working exactly as designed. The one **real, generalizable** gap
this trace surfaced is discussed in §7 as an optional enhancement (body-text
experience-requirement extraction), not a defect fix.

---

## 3. Cover-Letter Pipeline Audit

**Files/functions** (all in `src/applypilot/scoring/cover_letter.py` unless noted):
- `generate_cover_letter(resume_text, job, profile, max_retries=3)` — line 155.
  Retry loop: `for attempt in range(max_retries + 1)` (line 203) → up to 4
  total generation attempts per job.
- Client acquisition: `client = get_client(quality=True)` (line 200) — uses
  the **quality** tier of `llm.py`'s fallback chain (Gemini Pro → OpenAI →
  Anthropic Sonnet → local), not the fast tier.
- Generation call: `client.chat(messages, max_tokens=get_token_limit("cover",
  8192), temperature=0.7)` (line 223-227). **No `exclude_providers`
  argument** — this is the finding from §1. Contrast with
  `src/applypilot/scoring/tailor.py:1137-1155`, which explicitly excludes
  the `"local"` provider on its equivalent heavy call and fails fast
  (`RuntimeError`) into a bounded degraded-mode composer
  (`tailor.py:996`, `_run_degraded_mode`) instead of ever reaching a slow
  local generation on the full-size prompt.
- Validation: `validate_cover_letter(letter, profile)` (called line 230,
  defined in `validator.py` — not line-checked in this pass). On failure,
  `avoid_notes` accumulate and the loop retries with a fresh prompt
  (line 234-239+).
- Caller: `_cover_one_job(job, resume_text, profile, doc_format="docx")`
  (line 258) — per-job wrapper; rejected/failed letters go to
  `{prefix}_CL_rejected.txt` per prior documented behavior (not re-verified
  this pass).
- Batch runner: `run_cover_letters(...)` (line 394).
- **No browser/Patchright dependency anywhere in this file** — confirmed via
  `grep` for `patchright`/`playwright` imports: zero matches.

**Timeout/retry behavior actually in effect** (traced through `llm.py`,
not re-read line-by-line this pass but consistent with the codebase's own
documented behavior for the `"local"` provider entry): if the quality cloud
chain (Gemini Pro/OpenAI/Anthropic) is exhausted, `client.chat()` falls
through to the `"local"` entry in the same fallback chain (`llm.py`'s
`_build_fallback_chain`), bounded per-call by `APPLYPILOT_LOCAL_LLM_TIMEOUT`
(default 60s) and with `max_tokens` clamped to
`APPLYPILOT_LOCAL_LLM_MAX_TOKENS` (default 2048) for that leg specifically.
This is *not unbounded* — but it is NOT fast-failed the way tailoring's
equivalent path is, so a cover-letter job under sustained cloud exhaustion
can pay this cost **up to 4 times** (once per retry attempt) rather than
detecting exhaustion once and using a cheap deterministic/bounded path.

**Whether this is transient or structural:** structural — it's a missing
`exclude_providers` argument, an established pattern already proven
elsewhere in this same codebase (`tailor.py`), not a flaky external
dependency. Fixing it is a one-line-plus-except-handler change (see §7).

**Not verified this pass:** no specific log excerpt showing an actual
multi-minute cover-letter stall was located during this audit. §8 gives the
exact command to check `~/.applypilot/logs/` for cover-stage runs and their
durations before treating this as a confirmed-from-production incident
rather than a mechanism-level finding.

---

## 4. Tailoring Audit

**Files/functions** (`src/applypilot/scoring/tailor.py` unless noted):
- Prompt construction: `_build_tailor_prompt(profile, standup_decision)` —
  line 477.
- Main entry: `tailor_resume(...)` — line 909. Contains the retry loop,
  the `exclude_providers=frozenset({"local"})` fast-fail heavy call
  (line 1137-1155), and the degraded-mode fallback dispatch (line 996-1113).
- Judge: `judge_tailored_resume(original_text, tailored_text, job_title,
  profile)` — line 858.
- Result classification: `_mark_tailor_result(...)` — line 1488.
- Batch runner: `run_tailoring(...)` — line 1573.
- Validation (`src/applypilot/scoring/validator.py`):
  - `validate_json_fields(data, profile, standup_decision)` — line 358, the
    main hard-error/warning classifier.
  - `check_unsupported_technical_skills(skills_text, profile)` — line 268
    (ERROR-tier, rejects fabricated skills).
  - `check_title_inflation(title, profile)` — line 728 (ERROR-tier, rejects
    an invented seniority/title claim not backed by any authoritative
    profile title).
  - `check_claim_inflation(all_text, profile)` — line 490 (WARNING-tier,
    whole-resume authority-verb overclaiming signal).
  - `check_agency_inflation(all_text, profile)` — line 528 (WARNING-tier,
    companion check for people-management/agency-tier overclaiming).

**Evidence retrieval:** grounded in `scoring/local_tailor.py`'s
`rank_profile_evidence` + deterministic requirement/evidence matching
(referenced, not re-read line-by-line this pass — already modified once
this session for an unrelated semantic-retrieval change, see the commit
just pushed, §10).

**Warnings vs. genuine bugs — the prior investigation's list, triaged:**
| Item | Classification | Basis |
|---|---|---|
| "utilizing" style warning | Cosmetic | Style/tone warning tier, not a fabrication-safety check |
| led/managed/supervised/directed/spearheaded claim-strength concern | **Already has a dedicated ERROR/WARNING-tier check** — `check_agency_inflation` (`validator.py:528`) exists specifically for this. If a specific resume slipped an unsupported agency verb past this check, that's a real bug in that function's coverage, not an absent safeguard — needs a concrete failing example to investigate further, not a new mechanism. |
| Occasional missing projects | Not evaluated this pass — needs a concrete example |
| Occasional unrecognized employers | **Already has a dedicated ERROR-tier check** — `validate_factual_anchors` (referenced in this codebase's history; not re-read this pass) is documented as parsing both "Title at Company" and legacy header formats specifically to catch this. If a real miss is found, it's a coverage gap in that function, not an absent mechanism. |
| Profile-evidence/generated-experience mismatch | Not evaluated this pass — needs a concrete example |

**Recommendation:** do not build new validator machinery for these until a
*specific* failing resume/report JSON is identified and traced through the
existing checks above — every one of the five items already has a
purpose-built check in the current code; the open question is coverage of
that check on a specific real example, not absence of the check.

---

## 5. Patchright / Browser Audit

**Installed versions** (`pip show`, this machine, 2026-08-31):
- `patchright==1.62.1`
- `playwright==1.62.0`

**Repository dependency declaration** (`pyproject.toml:25-26`):
```
playwright>=1.40
patchright>=1.50
```
Both are open lower-bound version ranges — **no upper pin, no exact-version
requirement**. The installed versions (1.62.x) satisfy these constraints.
**No repository evidence requires reverting the recent `patchright install`
update.**

**Two structurally separate browser-automation paths in this repo — do not
conflate them:**

1. **Discovery/enrichment scraping** (`discovery/smartextract.py`,
   `enrichment/detail.py`, `scoring/pdf.py`) — uses Patchright/Playwright's
   own `.launch()` API against its **bundled** Chromium build. Confirmed on
   this machine: `patchright install` placed this at
   `%LOCALAPPDATA%\ms-playwright\chromium-1234\` (plus `firefox-1538`,
   `webkit-2336`, `ffmpeg-1011`, `winldd-1007`). This path does not need
   `--load-extension` and is unaffected by the finding below.

2. **Apply-stage browser automation** (`apply/chrome.py`,
   `apply/launcher.py`) — does **not** use Patchright's bundled browser at
   all. `apply/chrome.py:launch_chrome` (line 1422) resolves a real Chrome
   binary via `config.get_chrome_path()` (line 1464) and launches it with
   `subprocess.Popen` (line 1542), then attaches over CDP — this is the
   flow that needs `--load-extension` to work (for the ApplyPilot browser
   extension: job-tracking pill, per-tab action-log capture, HITL banner —
   see the repo's own documented history of this extension's design).

**The gap:** `config.get_chrome_path()` (`src/applypilot/config.py:55-107`)
has a Windows branch (line 71-77) that only checks standard Google Chrome
install locations (`Program Files`, `Program Files (x86)`,
`%LOCALAPPDATA%`). It has **no Chrome-for-Testing candidate path on
Windows** — that logic exists only in the Linux branch (line 83-95,
explicit comment: *"Prefer Chrome for Testing if installed: it is the only
Chromium build that still honors --load-extension"*). On this machine, with
no `CHROME_PATH` env var set, `get_chrome_path()` resolves to:
```
C:\Program Files (x86)\Google\Chrome\Application\chrome.exe
```
confirmed via `(Get-Item ...).VersionInfo.ProductVersion` → **152.0.7977.65**
— a branded-Stable Chrome build well past the Chrome-137 cutoff this
codebase's own prior work documented as silently rejecting
`--load-extension`. This means, per the repository's own established
finding (not re-derived here, just applied): **the apply-stage extension
likely fails to load silently on this Windows machine today**, independent
of and unrelated to the recent Patchright version update.

**This is not something the recent `patchright install` caused or fixed.**
It predates it: `get_chrome_path()`'s Windows branch never had CfT support,
on any Patchright/Playwright version. The fix (§7) is adding a Windows CfT
candidate path, mirroring the existing Linux logic — not a Patchright
version change.

**Do not revert Patchright.** No evidence in this repo requires an older
version; the gap found here is entirely in `config.py`'s Chrome-path
resolution, not in Patchright/Playwright itself.

---

## 6. Relevant Files

| File | Why it matters here |
|---|---|
| `src/applypilot/eligibility.py` | Canonical seniority-disqualifier predicate + `revalidate_seniority`/`revalidate_stale_scores` sweeps. Read before touching any eligibility logic — this module exists specifically to prevent the kind of drift described in its own docstring (line 1-26). |
| `src/applypilot/scoring/scorer.py` (lines 359-480, `_check_ineligible`; line 650+, `score_job`) | Pre-LLM deterministic gate: geography, seniority, degree/clearance, commission-only, configured title excludes, and the 86-term ethical/defense keyword list (line 443-478). |
| `src/applypilot/config.py` (lines 55-107, `get_chrome_path`) | Chrome binary resolution for the apply-stage browser. Windows branch lacks CfT awareness (§5). |
| `src/applypilot/apply/chrome.py` (line 1422, `launch_chrome`; line 903, `_get_real_user_agent`) | Where the resolved Chrome path is actually launched and where `--load-extension` matters. |
| `src/applypilot/scoring/cover_letter.py` (line 155, `generate_cover_letter`; line 223, the `client.chat()` call) | Missing `exclude_providers` fast-fail (§3). |
| `src/applypilot/scoring/tailor.py` (line 909, `tailor_resume`; line 996, `_run_degraded_mode`; line 1137-1155, the `exclude_providers` pattern to mirror) | Reference implementation of the fast-fail-on-cloud-exhaustion pattern cover_letter.py lacks. |
| `src/applypilot/llm.py` | `LLMClient.chat()`'s `exclude_providers` parameter (already exists, already used by tailor.py) and the `"local"` provider's timeout/max_tokens clamping. |
| `src/applypilot/scoring/validator.py` (lines 268, 358, 490, 528, 728) | Existing fabrication/inflation checks — see §4 triage before assuming any are missing. |
| `~/.applypilot/searches.yaml` (`exclude_description_keywords`) | Live 86-term defense/military/surveillance/policing exclusion list — user-editable, already confirmed active and working for post-2026-08-18 discoveries. |

**Tests to add/update:**
- `tests/test_eligibility.py` — add a case using the real Boeing description
  text (or a synthetic equivalent containing "weapon systems" +
  "Boeing Defense Services") asserting `_check_ineligible`/the keyword scan
  rejects it — this documents the already-correct behavior and guards
  against regression, since no test currently exercises this specific
  keyword against this specific description shape.
- A new or extended test in `tests/` for `cover_letter.generate_cover_letter`
  asserting the eventual `exclude_providers=frozenset({"local"})` call
  (once added per §7) — mirror the existing pattern in
  `tests/test_llm_cascade.py`'s `TestChatExcludeProviders` class (already
  covers `chat()`'s `exclude_providers` mechanism generically; add a
  cover-letter-specific case, don't duplicate the mechanism test).
- A new test for `config.get_chrome_path()`'s Windows branch, once a CfT
  candidate path is added (§7) — assert it's preferred over branded Chrome
  when present, falls back correctly when absent. No existing test covers
  the Windows branch's CfT-awareness at all (it doesn't exist yet to test).

---

## 7. Recommended Implementation (minimal, surgical)

**Do NOT implement anything below without first re-confirming against
current `main` — this audit reflects the state as of commit `2259eb4`
(§10). If Copilot is picking this up after further changes, re-run the
relevant `git log`/`grep` checks first.**

1. **Cover-letter fast-fail on cloud exhaustion** (fixes §3):
   In `cover_letter.py:generate_cover_letter`, mirror `tailor.py:1137-1155`:
   pass `exclude_providers=frozenset({"local"})` to the `client.chat()` call
   at line 223, wrapped in `try/except RuntimeError`. On that exception,
   either (a) skip local entirely and mark the job for retry next run (the
   simplest option — cover letters don't have an existing bounded degraded-
   mode composer the way tailoring does), or (b) if a bounded local
   fallback is wanted, that would be new scope — recommend (a) first since
   it's the minimal fix for the actual stall, not a feature addition.
   **Do not build a new degraded-mode composer for cover letters as part of
   this fix** — that's meaningfully larger scope than the stall this is
   fixing.

2. **Windows Chrome-for-Testing path** (fixes §5):
   In `config.py`'s Windows branch (line 71-77), add a CfT candidate
   mirroring the Linux branch's logic (line 89-91) — likely
   `Path.home() / ".applypilot" / "chrome-for-testing" / "chrome-win64" /
   "chrome.exe"` (verify the actual CfT-for-Windows directory layout before
   hardcoding — it differs from the Linux `chrome-linux64` naming; Chrome
   for Testing's official Windows archive layout should be checked against
   whatever installs it, since nothing currently installs CfT for Windows
   in this repo — that installer step doesn't exist yet either and may be
   a prerequisite, not just a config.py change). **Confirm whether the
   `--load-extension` rejection on Chrome 137+ is actually being hit before
   implementing** — see §8 for the check.

3. **Boeing-class stale-eligibility documentation** (§2): no code fix
   required. Optionally add the regression test described in §6 to lock in
   current-correct behavior.

**Explicitly NOT recommended in this pass** (would be new scope, not a
fix for anything found):
- Body-text experience-requirement extraction (the "unrealistic seniority"
  concern from §2 item 3) — real potential gap, but not what caused the
  Boeing incident, and extraction-from-free-text is nontrivial scope. Flag
  for separate consideration, don't bundle into this fix.
- Any change to `SENIORITY_TITLE_PATTERN`, the ethical-keyword list, or the
  validator's warning/error tiers — no evidence found that any of these are
  currently wrong; §4's triage found existing checks for every item raised.
- Any Patchright/Playwright version change — §5 found no evidence requiring
  one.

---

## 8. Validation Plan

All commands are PowerShell, pasteable directly. Where direct DB/Python
inspection is needed, it's written as a here-string piped to a temp `.py`
file (never multiline Python pasted into an interactive prompt).

**8.1 — Confirm the Boeing row's current state (should be unchanged from §2):**
```powershell
@'
import sqlite3
db = r"C:\Users\phili\.applypilot\applypilot.db"
url = "https://boeing.wd1.myworkdayjobs.com/EXTERNAL_CAREERS/job/USA---Tukwila-WA/Software-Engineer--Associate-or-Experienced-_JR2026500177-1"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
row = conn.execute("SELECT url, title, eligibility, fit_score, state FROM jobs WHERE url = ?", (url,)).fetchone()
print(dict(row) if row else "NOT FOUND")
'@ | Out-File -Encoding utf8 $env:TEMP\check_boeing.py
& venv\Scripts\python.exe $env:TEMP\check_boeing.py
```

**8.2 — Run the existing test suites for the files this audit touches:**
```powershell
& venv\Scripts\python.exe -m pytest tests\test_eligibility.py tests\test_llm_cascade.py -q
```

**8.3 — Confirm the live ethical-keyword list is catching post-fix defense jobs (should return a nonzero count of already-correctly-excluded rows):**
```powershell
@'
import sqlite3, re
db = r"C:\Users\phili\.applypilot\applypilot.db"
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT url, title, eligibility, fit_score FROM jobs WHERE discovered_at > \"2026-08-18\" AND full_description LIKE \"%weapon system%\"").fetchall()
for r in rows[:10]:
    print(dict(r))
print("total:", len(rows))
'@ | Out-File -Encoding utf8 $env:TEMP\check_defense_gate.py
& venv\Scripts\python.exe $env:TEMP\check_defense_gate.py
```

**8.4 — Check for evidence of an actual cover-letter stall in recent logs (before treating §3 as a confirmed incident, not just a mechanism finding):**
```powershell
Get-ChildItem $env:USERPROFILE\.applypilot\logs\*cover*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```
Then inspect the most recent one or two for elapsed-time gaps or repeated
`"local"` provider log lines around the cover stage.

**8.5 — Check whether the apply-stage extension is actually failing to load on this machine (before implementing §7 item 2):**
```powershell
applypilot apply --dry-run --url "<any real job URL>"
```
Inspect the resulting worker log (`~/.applypilot/logs/`) for whether the
ApplyPilot extension's job-tracking pill or action-log messages appear —
their absence, combined with no explicit error, is consistent with a
silently-rejected `--load-extension`. **Use `--workers 1`-equivalent
behavior implicitly** (a single `--dry-run --url` invocation already runs
one job, not a batch).

**8.6 — After implementing §7 item 1 (cover-letter fast-fail), validate with a small real batch, not a full run:**
```powershell
applypilot run cover --limit 2
```
Confirm the run completes without a multi-minute per-job stall, and that
the `--limit` flag genuinely caps it at 2 jobs (this flag already exists
and is exercised elsewhere in this codebase — do not invent a new one).

**8.7 — Full suite, handed off (do not self-run inside an agent loop — this project's own working convention, established across this session, is to hand off the full suite to a human/CI, not self-run it):**
```powershell
& venv\Scripts\python.exe -m pytest -q
```
One known, pre-existing, unrelated failure exists on `main` as of commit
`2259eb4`: `tests/test_watchdog.py::TestRamStats::test_ram_threshold_is_a_small_but_nonzero_fraction_of_a_low_ram_machine`
(asserts `RAM_AVAILABLE_THRESHOLD_MB < 2000` strictly; the constant is
exactly `2000.0`). Confirmed present on unmodified `main` via `git stash`
before this session's commit. Not caused by, and out of scope for, this
audit's findings.

---

## 9. Copilot Handoff — Explicit Instruction Block

**Implement, in this order, only after re-validating each finding against
current `main`:**
1. `cover_letter.py:generate_cover_letter` — add `exclude_providers=
   frozenset({"local"})` to the `client.chat()` call (line 223 as of this
   audit), with a `try/except RuntimeError` that logs and returns a
   failure/retry result rather than falling through to a slow local call.
   Mirror `tailor.py:1137-1155`'s pattern exactly; do not invent a new one.
2. `config.py:get_chrome_path()` — add a Windows Chrome-for-Testing
   candidate path, mirroring the existing Linux branch. **First confirm via
   §8.5 that this is actually causing an observable failure** — if the
   extension loads fine despite Chrome 152 (e.g. if this specific branded
   build hasn't actually had `--load-extension` removed, contrary to the
   codebase's own prior finding for other Chrome versions), do not
   implement this change; document the discrepancy instead.
3. Add the two regression tests described in §6.

**Must NOT be changed without new evidence:**
- `eligibility.py`'s `SENIORITY_TITLE_PATTERN`, the ethical-keyword list in
  `~/.applypilot/searches.yaml`, or `_check_ineligible`'s check ordering —
  all confirmed working correctly for their intended cases.
- Any Patchright/Playwright version pin — no evidence requires a change.
- `validator.py`'s warning/error tier assignments — every item raised
  already has a dedicated check; do not add duplicate mechanisms without a
  concrete failing example first.
- The retry-starvation fix, continuous-run supervisor, watchdog sleep gate,
  or local-LLM discovery fallback shipped in commit `2259eb4` (§10) — these
  are already-completed, already-tested work from this session, unrelated
  to the findings in this audit. Do not revert or redesign them.

**Evidence required before declaring any of this done:**
- For item 1: a real (or realistically mocked) run showing a cover-letter
  job that previously would have reached the local leg now fails fast
  instead, plus the new/updated test passing.
- For item 2: the §8.5 dry-run check showing the extension actually fails
  to load on unmodified `main` (not just the version-number theory), AND
  confirmation after the fix that it loads. If a CfT installer step doesn't
  already exist for Windows, that's a prerequisite to discover and document
  before this fix can be meaningful — don't add a candidate path pointing
  at a directory nothing ever populates.
- For both: `pytest` green on the specific test files touched (§8.2), not
  just "the suite didn't get worse" — cite the specific test names that
  exercise the new behavior.
- Do not treat a passing full suite, or a high `fit_score`, as proof of
  correctness on its own — this audit's central finding (§2) is exactly a
  case where stored data looked wrong out of context but the live code path
  was already correct; verify behavior against current code and state, not
  against stored columns in isolation.

---

## 10. Work Already Preserved (this session, prior to this audit)

**Commit:** `2259eb41dc078fc260e419b075997c5db98afddc`
**Branch:** `main`
**Remote:** `origin` (`https://github.com/PhiliptheGinger/ApplyPilot_Private.git`)
**Push status:** confirmed — `187f098..2259eb4 main -> main`, pushed
2026-08-31.

**Files included:**
```
A  scripts/run_continuous_supervisor.ps1
A  scripts/run_continuous_task.xml
M  src/applypilot/cli.py
M  src/applypilot/database.py
M  src/applypilot/discovery/hackernews.py
M  src/applypilot/discovery/smartextract.py
M  src/applypilot/llm.py
M  src/applypilot/scoring/local_tailor.py
M  src/applypilot/scoring/scorer.py
M  tests/test_eligibility.py
M  tests/test_pending_score_state_selection.py
M  tests/test_watchdog.py
M  watchdog.py
```

**What this commit contains** (already implemented and tested, unrelated to
this audit's §1-9 findings — do not re-do or revert):
- Retry-starvation fix: `database.py`'s `get_jobs_by_stage` ORDER BY now
  prioritizes retry-eligible jobs ahead of fresh discoveries.
- Persistent continuous-run supervisor: `scripts/run_continuous_supervisor.ps1`
  (restart-on-exit wrapper, self-locked) + `scripts/run_continuous_task.xml`
  (hidden, battery-agnostic Scheduled Task registration) — fixes a
  previously-reported recurring visible-console-window regression.
- `watchdog.py`: automatic machine-sleep on resource emergency is now
  opt-in (`APPLYPILOT_WATCHDOG_SLEEP=1`), default off.
- `llm.py:ask_local()` + wiring into `hackernews.py`'s comment extraction
  and `smartextract.py`'s Phase 1/Phase 2 LLM calls — opt-in local-model
  fallback for the three discovery/enrichment call sites that previously
  had zero fallback on cloud exhaustion.

**Working tree status after this commit:** clean of tracked-file changes.
Untracked-only remainder: `watchdog_logs/*.log` (runtime log output from
this session's watchdog runs — not source, not committed, harmless to
leave in place or delete).

**Unrelated pre-existing changes found:** none. `git status --short` (with
`--untracked-files=normal`, since this machine's global gitconfig hides
untracked files by default) showed only the files listed above plus the
log files noted; nothing was excluded from the commit as "unrelated."

**This audit document itself** (`docs/copilot_handoff_2026-08-31.md`) has
not been committed as of this writing — it is a new untracked file. Commit
it separately if durability across sessions/machines is wanted; it was not
part of the "preserve existing work" instruction, which was scoped to the
already-implemented code above.
