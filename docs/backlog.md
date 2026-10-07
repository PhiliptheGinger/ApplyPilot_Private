# ApplyPilot Backlog

Prioritized view of every open item in `docs/future_work.md` (FW-n, numbers unchanged) plus
engineering items from `docs/audit_2026-10.md` (E-n) and discovery item D1.

Columns:
- **Quadrant**: Eisenhower. Q1 do now, Q2 schedule, Q3 quick wins, Q4 later/maybe.
- **Size**: S (under a session), M (one or two sessions), L (multi-session, needs a design pass first).
- **Where**: `cloud-ok` can be built and tested in a cloud session; `needs live machine` needs your DB, Chrome, phone or logs; `build in cloud, verify live` is both.
- **Depends on**: items that must land first.

Regenerate this file when items change; keep full write-ups in `docs/future_work.md`.

## Suggested order

**Stage 0 — this week, small, unblocks everything:** E3, FW24, FW23, FW49, FW51

**Stage 1 — keep the machine and budget healthy:** FW32, FW13, E1, FW17, FW55

**Track A — reach LinkedIn jobs without automating LinkedIn:** FW64 → FW38

**Track H — locality data:** FW71

**Track B — cheaper, faster applies:** FW65 → FW62 → FW30

**Track C — session/worker architecture:** FW36 → FW69 → FW46 → FW41

**Track D — login & 2FA (research-gated):** FW43 → FW44 → FW66

**Track E — scoring accuracy:** FW28 → FW15

**Track F — profile richness:** FW26 → FW9 → FW21

**Track G — code health (runs alongside anything):** E4, E5, E6

## Q1 — Do now (urgent + important)

| ID | Item | Track | Size | Where | Depends on | Why |
|---|---|---|---|---|---|---|
| E3 | Confirm the transaction-leak fix on the live machine, then close or redesign decision #183's DB-lock question | Reliability & operations | S | needs live machine | — | If leaked write transactions were the real cause, the 'capacity ceiling' work disappears. A one-night log comparison answers it. |
| FW38 | LinkedIn jobs have no application_url (6,108 jobs, 100% of manual_only) | Apply throughput & cost | L | needs live machine | FW64 | Biggest single source of unreachable jobs. Safest route is FW64's title+company lookup, not scraping LinkedIn. |
| FW18 | Keep tailor -> cover -> apply moving (operational, ongoing) | Profile & tailoring quality | S | needs live machine | — | Standing bottleneck; this is running the pipeline, not new code. |

## Q2 — Schedule (important, not urgent)

| ID | Item | Track | Size | Where | Depends on | Why |
|---|---|---|---|---|---|---|
| FW32 | Machine-health watchdog: keep llama-server at BelowNormal, avoid ScheduledDefrag collisions, memory-pressure guard | Reliability & operations | M | needs live machine | — | llama-server priority drift has recurred 5+ times. A watchdog.py already exists to extend. |
| FW13 | Smarter Claude budget policy than 'always reserve everything for apply' (Pro plan) | Reliability & operations | M | cloud-ok | — | Shared Pro pool is the dominant apply bottleneck (decisions #166/#185/#195). |
| FW64 | Cross-source duplicate detection + find a LinkedIn job's real ATS posting by title/company | Apply throughput & cost | L | build in cloud, verify live | — | Prevents double-applying and gives FW38 a path that never touches linkedin.com automation. |
| FW62 | Apply-page schema layer: Stage 2 verified vendor selectors (needs real DOM captures), Stages 4-5 | Apply throughput & cost | L | build in cloud, verify live | FW65 | Stage 1 and 3 shipped. Stage 2 needs live page captures to avoid fabricated selectors. |
| FW65 | Deterministic layer feeds Claude (known fields, Q&A, path hints) instead of replacing it; human-like pacing | Apply throughput & cost | M | cloud-ok | — | Cuts cost/latency without the bot-like uniform timing of a pure script. |
| FW30 | Workday personal-info page + qa_knowledge reuse for screening questions | Apply throughput & cost | M | needs live machine | FW65 | Workday/ADP dominate real runs; selectors written in #169 are still unverified live. |
| FW36 | Research: does @playwright/mcp support a persistent HTTP/SSE server mode? | Session / worker architecture | S | cloud-ok | — | Gating question for the whole session/worker redesign. Cheap to answer. |
| FW69 | Decouple browser sessions from worker threads (general version of non-blocking HITL) | Session / worker architecture | L | needs live machine | FW36 | Same root gap behind FW36, FW46 and FW69: worker id == Chrome profile == CDP port. |
| FW43 | Research bot-detection risk of automated login/2FA (gate for FW44) | Login, 2FA & SMS | M | cloud-ok | — | Explicit user gate: no login/2FA automation until researched. |
| FW44 | Wire the SMS relay into the apply agent's verification step | Login, 2FA & SMS | M | needs live machine | FW43 | Relay clients exist (Twilio, Google Voice, ADB) but are unused by apply. |
| FW28 | Requirement-framing classifier ('seeking N years', 'commercial experience', founding titles) instead of growing regex lists | Scoring accuracy | L | build in cloud, verify live | — | Covers FW14 and FW15(b)(d). Needs real-corpus false-positive checks. |
| FW15 | Remaining rubric gaps: implicit-seniority postings, defense/law-enforcement contractor backstop | Scoring accuracy | M | build in cloud, verify live | FW28 | Scoring false positives cost real apply budget. |
| FW26 | Ask follow-up questions when an entry is too thin to bank (in expand-bank first) | Profile & tailoring quality | M | cloud-ok | — | Real fix for 'no survivors' without fabricating. |
| FW9 | Build experience/project inventory from resume + GitHub in the wizard (then expand banks) | Profile & tailoring quality | L | cloud-ok | FW26 | Wizard never builds the inventory phrase banks need; every entry is hand-authored today. |
| FW23 | Line-by-line review of resume degraded-mode output (grammar/pronoun bugs like #137 found in cover letters) | Profile & tailoring quality | S | needs live machine | — | Never reviewed at the same rigor as cover letters. |
| FW71 | Locality data pack: nearby places (auto-fill the location filter) and local employers + their ATS, delivered by the wizard | Discovery coverage | L | cloud-ok | — | Turns E9's hand-edited accept list into an automatic one and generalizes the NC/Seattle employer lists to any locality. |
| E1 | Move off the python-jobspy pin that forces numpy==1.24.2 (unlocks Python 3.12+) | Engineering health | M | build in cloud, verify live | — | Install fails on 3.12/3.13; the pin exists for a markdownify conflict. |
| E4 | Split database.py, then local_tailor.py / launcher.py, one cohesive group per PR with patch targets updated | Engineering health | L | cloud-ok | — | 4.4k/3.3k/3k-line files; must not silently break module-path patches in tests. |

## Q3 — Quick wins (lower impact, cheap)

| ID | Item | Track | Size | Where | Depends on | Why |
|---|---|---|---|---|---|---|
| E8 | Enrichment double-processing: stale-completion guard firing on ~100% of a site batch (decisions #189/#203) | Reliability & operations | M | build in cloud, verify live | — | Safe but wasteful HTTP/LLM spend; likely the --stream enrich loop re-selecting in-flight jobs. |
| FW51 | Auto-reject cookie banners via the extension's content-script hook | Apply throughput & cost | S | build in cloud, verify live | — | Saves an agent tool-call (and confusion) on many pages; low risk. |
| FW17 | Bare Canadian province names in location (needs Ontario, CA disambiguation) | Scoring accuracy | S | build in cloud, verify live | — | Only 3-4 live rows; recurs occasionally. |
| FW24 | Verify the cover-letter word-count shortfall is closed (decisions #162/#163 suggest yes) | Profile & tailoring quality | S | needs live machine | — | Probably done; confirm and close. |
| FW55 | Company name for non-pattern ATSes (CareerPlug, Eightfold) + transcript fallback | Discovery coverage | S | cloud-ok | — | #214 wired the backfill; unknown ATS domains still get NULL company. |
| FW49 | 'Handing off...' button shows live progress instead of looking frozen | HITL & wizard UX | M | build in cloud, verify live | — | Needs a small status channel back from run_job. |

## Q4 — Later / maybe

| ID | Item | Track | Size | Where | Depends on | Why |
|---|---|---|---|---|---|---|
| FW35 | One parameterized escalation primitive (cost/latency/capability ordering) to replace hand-rolled cascades | Reliability & operations | L | cloud-ok | — | Makes later escalation ideas (haiku fast path, editor 1.7b->8b) cheap to add. |
| FW50 | Per-job download of application documents + dedup of recurring notices | Apply throughput & cost | M | build in cloud, verify live | — | Nice-to-have record keeping. |
| FW56 | Why the Lumen phone-2FA wall later disappeared | Apply throughput & cost | S | needs live machine | — | One data point; investigate if it recurs. |
| FW52 | Deliberately imperfect answers on subjective assessments | Apply throughput & cost | S | cloud-ok | — | Needs an explicit ethics/detection discussion before any code. Recommendation: don't build. |
| FW39 | Proxy/fingerprint hardening for LinkedIn scraping | Apply throughput & cost | L | needs live machine | — | Needs a defined scope ('spoofing' is undefined) and an ethics conversation first. |
| FW46 | Separate human-interaction coordinator from the automation worker pool | Session / worker architecture | L | needs live machine | FW69 | Falls out of FW69. Avoid the name 'server'. |
| FW41 | Live queue UI for human-first manual tasks | Session / worker architecture | M | build in cloud, verify live | FW69 | v1 already auto-advances; this adds visibility from a second window. |
| FW66 | Minimal SMS-gateway Android app forked from open-source prior art | Login, 2FA & SMS | L | needs live machine | FW44 | Replaces the USB-tethered ADB relay; security model must be designed first. |
| FW21 | Retest whether stand-up's hardcoded title regex is still needed | Scoring accuracy | M | build in cloud, verify live | FW9 | Needs qualifications wired into evidence matching first. |
| FW67 | Company-reputation research to ground the ethical filter | Scoring accuracy | L | build in cloud, verify live | — | Real cost and bias risks; needs its own scoping pass. |
| FW27 | GitHub-import review shows the flagged text, not just the category | Profile & tailoring quality | S | cloud-ok | — | Show the evidence, not just the verdict. |
| D1 | Custom-ATS Playwright scrapers (Microsoft, Google, Apple, Meta) and Workday tenant registry | Discovery coverage | L | needs live machine | — | Seattle-specific lists are a fork leftover; generalize per locality first. |
| FW60 | Wizard wording/tone pass ('bedside manner') | HITL & wizard UX | S | cloud-ok | — | Polish. |
| FW33 | Wizard walkthroughs for external setup (Gmail OAuth, etc.) with known-failure decision tree | HITL & wizard UX | M | cloud-ok | FW60 | Gmail OAuth setup had real snags. |
| FW45 | First-run onboarding for the HITL banner | HITL & wizard UX | S | cloud-ok | — | Discoverability. |
| FW47 | Bug-report icon (logs -> GitHub issue) and 'flag this job' icon in the banner | HITL & wizard UX | M | build in cloud, verify live | — | Feedback loop from real use. |
| E5 | Remove identical helper copies (_strip_html, _get_ua, _fetch_page, _fetch_json) | Engineering health | S | cloud-ok | — | Cheap; prevents the next drift. |
| E6 | Lint tests/ and widen ruff rules one family at a time (B, UP, SIM) | Engineering health | S | cloud-ok | — | 56 findings in tests/ today. |
| FW58 | Companion mobile app / remote monitoring (Obtainium-distributed) | Long-horizon ideas | L | needs live machine | — | FW53 email alerts (done) now cover most of the need. |
| FW68 | Filter keyword store, preference questionnaire, wizard scope, multi-user accounts | Long-horizon ideas | L | cloud-ok | — | Each needs its own scoping conversation. |
| FW59 | Research ideas (RSI search, DreamCoder, embodied-cognition sim) | Long-horizon ideas | L | cloud-ok | — | Thought experiments; possibly a side project. |

## Closed or merged (full text still kept in docs/future_work.md / docs/decisions_archive.md)

| ID | Item | Status |
|---|---|---|
| FW48 | Readable hand-off confirmation | built 2026-10-07 (#220) |
| FW53 | Email alerts for needs_human / Claude limits | built 2026-10-07 (#219); verify a real send |
| E7 | Make CI run | it already runs on PRs/pushes to main; first runs after PR #16 exposed 2 real failures, fixed 2026-10-07 |
| E9 | Wizard writes a location filter | built 2026-10-07; automatic nearby-places fill is FW71 |
| FW16 | GB/CAN/AU/MX country codes | already shipped (#214) |
| FW22 | Cover-letter degraded mode | built #137-#142; remaining parts folded into FW24 |
| FW31 | Pre-apply expired-posting check | built #157 |
| FW34 | Scam detection for tracked emails | built #206 |
| FW37 | Flaky deterministic-fallback ordering test | fixed #214 |
| FW40 | LinkedIn human-first flow | built #187 |
| FW57 | Wizard SMS/email verification | built #198 |
| FW61 | ADB not found | resolved #202 |
| FW63 | smartextract local-model JSON | fixed #214; shared parser in this pass |
| FW70 | Silent thread death audit | done #212 |
| FW10 | GitHub import gate | built #85; remaining work is FW9 |
| FW12 | OpenAI $0 credits | not code; deprioritized by user |
| FW14 | 'Seeking N years' phrasing | merged into FW28 |
| FW29 | Cross-pollination map | used as the source for FW28/FW35/FW65; kept as reference |
| FW42 | Repurposable apply-page schemas | scoped as FW62 |
| FW54 | Apply cost pointer | umbrella for FW30/FW35/FW36/FW65 |
| Local-1..6 | Seattle employer lists, Lever/Ashby scrapers | Lever/Ashby built; rest folded into D1 |
