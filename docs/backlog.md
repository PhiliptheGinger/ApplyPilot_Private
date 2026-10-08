# ApplyPilot Backlog

Prioritized view of every open item in `docs/future_work.md` (FW-n, numbers unchanged) plus
engineering items from `docs/audit_2026-10.md` (E-n) and discovery item D1.

Columns:
- **Quadrant**: Eisenhower. Q1 do now, Q2 schedule, Q3 quick wins, Q4 later/maybe.
- **Size**: S (under a session), M (one or two sessions), L (multi-session, needs a design pass first).
- **Where**: `cloud-ok` can be built and tested in a cloud session; `needs live machine` needs your DB, Chrome, phone or logs; `build in cloud, verify live` is both.
- **Depends on**: items that must land first.

Regenerate this file when items change; keep full write-ups in `docs/future_work.md`. Staged plans and open decisions for the larger items are in `docs/scoping/`.

## Suggested order

**Stage 0 — this week, small, unblocks everything:** E3, FW24, FW23

**Stage 1 — keep the machine healthy:** FW32

**Track A — reach LinkedIn jobs without automating LinkedIn:** FW64 → FW38

**Track H — locality data:** FW71

**Track B — cheaper, faster applies:** FW62 → FW65 → FW30

**Track C — session/worker architecture:** Stage A done (#235) → FW69 → FW46 → FW41

**Track E — scoring accuracy:** FW28 live validation → FW15(c)

**Track F — profile richness:** FW9 → FW21

**Track G — code health and security (runs alongside anything):** E4, FW72

## Q1 — Do now (urgent + important)

| ID | Item | Track | Size | Where | Depends on | Why |
|---|---|---|---|---|---|---|
| E3 | Confirm the transaction-leak fix on the live machine, then close or redesign decision #183's DB-lock question | Reliability & operations | S | needs live machine | — | Baseline recorded 2026-10-07: ~2-3 give-ups/hour, only in full discover+4more runs. One overnight run on the fix answers it. |
| FW38 | LinkedIn jobs have no application_url (6,108 jobs, 100% of manual_only) | Apply throughput & cost | L | needs live machine | FW64 | Biggest single source of unreachable jobs. Safest route is FW64's title+company lookup, not scraping LinkedIn. |
| FW18 | Keep tailor -> cover -> apply moving (operational, ongoing) | Profile & tailoring quality | S | needs live machine | — | Standing bottleneck; this is running the pipeline, not new code. |

## Q2 — Schedule (important, not urgent)

| ID | Item | Track | Size | Where | Depends on | Why |
|---|---|---|---|---|---|---|
| FW72 | Security hardening review: secrets, local listener, prompt injection, untrusted input, data at rest | Code health | M | build in cloud, verify live | — | User wants program security; replaces the proxy idea from FW39 as the actual security work. |
| FW32 | Machine-health watchdog: keep llama-server at BelowNormal, avoid ScheduledDefrag collisions, memory-pressure guard | Reliability & operations | M | needs live machine | — | llama-server priority drift has recurred 5+ times. A watchdog.py already exists to extend. |
| FW64 | Cross-source duplicate detection + find a LinkedIn job's real ATS posting by title/company | Apply throughput & cost | L | build in cloud, verify live | — | Prevents double-applying and gives FW38 a path that never touches linkedin.com automation. |
| FW62 | Apply-page schema layer: Stage 2 verified vendor selectors (needs real DOM captures), Stages 4-5 | Apply throughput & cost | L | build in cloud, verify live | — | Stage 1 and 3 shipped. Stage 2 needs live page captures to avoid fabricated selectors; first step is a capture hook (docs/scoping/smaller_items.md). |
| FW65 | Remaining: verified per-ATS field hints in the Claude prompt (pacing and known-facts feeding are done, #231) | Apply throughput & cost | S | cloud-ok | FW62 | Fewer exploration steps per application; only for selectors verified by FW62 Stage 2. |
| FW30 | Workday personal-info page + qa_knowledge reuse for screening questions | Apply throughput & cost | M | needs live machine | FW62 | Workday/ADP dominate real runs; selectors written in #169 are still unverified live. |
| FW69 | Decouple browser sessions from worker threads (general version of non-blocking HITL) | Session / worker architecture | L | needs live machine | — | Same root gap behind FW36, FW46 and FW69: worker id == Chrome profile == CDP port. |
| FW15 | Remaining rubric gap: defense/law-enforcement contractor backstop (implicit seniority shipped in FW28) | Scoring accuracy | M | build in cloud, verify live | FW28 | Scoring false positives cost real apply budget. |
| FW9 | Build experience/project inventory from resume + GitHub in the wizard (then expand banks) | Profile & tailoring quality | L | cloud-ok | — | Wizard never builds the inventory phrase banks need; every entry is hand-authored today. |
| FW23 | Line-by-line review of resume degraded-mode output (grammar/pronoun bugs like #137 found in cover letters) | Profile & tailoring quality | S | needs live machine | — | Never reviewed at the same rigor as cover letters. |
| FW71 | Locality data pack: nearby places (auto-fill the location filter) and local employers + their ATS, delivered by the wizard | Discovery coverage | L | cloud-ok | — | US first, worldwide later; hosted on GitHub, wizard downloads only the user's region (decision #234). |
| E4 | Split database.py, then local_tailor.py / launcher.py, one cohesive group per PR with patch targets updated | Engineering health | L | cloud-ok | — | 4.4k/3.3k/3k-line files; must not silently break module-path patches in tests. |

## Q3 — Quick wins (lower impact, cheap)

| ID | Item | Track | Size | Where | Depends on | Why |
|---|---|---|---|---|---|---|
| FW24 | Verify the cover-letter word-count shortfall is closed (decisions #162/#163 suggest yes) | Profile & tailoring quality | S | needs live machine | — | Probably done; confirm and close. |

## Q4 — Later / maybe

| ID | Item | Track | Size | Where | Depends on | Why |
|---|---|---|---|---|---|---|
| FW35 | One parameterized escalation primitive (cost/latency/capability ordering) to replace hand-rolled cascades | Reliability & operations | L | cloud-ok | — | Makes later escalation ideas (haiku fast path, editor 1.7b->8b) cheap to add. |
| FW50 | Per-job download of application documents + dedup of recurring notices | Apply throughput & cost | M | build in cloud, verify live | — | Nice-to-have record keeping. |
| FW56 | Why the Lumen phone-2FA wall later disappeared | Apply throughput & cost | S | needs live machine | — | One data point; investigate if it recurs. |
| FW46 | Separate human-interaction coordinator from the automation worker pool | Session / worker architecture | L | needs live machine | FW69 | Falls out of FW69. Avoid the name 'server'. |
| FW41 | Live queue UI for human-first manual tasks | Session / worker architecture | M | build in cloud, verify live | FW69 | v1 already auto-advances; this adds visibility from a second window. |
| FW21 | Retest whether stand-up's hardcoded title regex is still needed | Scoring accuracy | M | build in cloud, verify live | FW9 | Needs qualifications wired into evidence matching first. |
| FW67 | Company-reputation research to ground the ethical filter | Scoring accuracy | L | build in cloud, verify live | — | Real cost and bias risks; needs its own scoping pass. |
| D1 | Custom-ATS Playwright scrapers (Microsoft, Google, Apple, Meta) and Workday tenant registry | Discovery coverage | L | needs live machine | — | Seattle-specific lists are a fork leftover; generalize per locality first. |
| FW33 | Wizard walkthroughs for external setup (Gmail OAuth, etc.) with known-failure decision tree | HITL & wizard UX | M | cloud-ok | — | Gmail OAuth setup had real snags. |
| FW47 | Bug-report icon (logs -> GitHub issue) and 'flag this job' icon in the banner | HITL & wizard UX | M | build in cloud, verify live | — | Feedback loop from real use. |
| FW58 | Companion mobile app / remote monitoring (Obtainium-distributed) | Long-horizon ideas | L | needs live machine | — | FW53 email alerts (done) now cover most of the need. |
| FW68 | Filter keyword store, preference questionnaire, wizard scope, multi-user accounts | Long-horizon ideas | L | cloud-ok | — | Each needs its own scoping conversation. |
| FW59 | Research ideas (RSI search, DreamCoder, embodied-cognition sim) | Long-horizon ideas | L | cloud-ok | — | Thought experiments; possibly a side project. |
| FW73 | LinkedIn employee-referral awareness ('recommended by someone who works here') | Discovery coverage | L | needs live machine | FW38/FW39 | Blocked on the same LinkedIn-scrape-risk caution as FW38/39. |
| FW74 | Temp/staffing-platform discovery (e.g. IT Worx) | Discovery coverage | M | build in cloud, verify live | — | Unknown API/bot-detection posture; research before building. |
| FW75 | Wizard asks candidate's risk/breadth preference (broad-net vs. selective) up front | HITL & wizard UX | S | cloud-ok | — | Real family-pressure example this session; default broad for unemployed candidates. |
| FW76 | Explicit human "I finished this myself" hand-off signal for human-first/needs_human pauses | HITL & wizard UX | M | build in cloud, verify live | FW40 | Real gap found live: no way to tell the agent a human already finished the page. |
| FW77 | Ask the candidate to disambiguate screening questions that don't reduce cleanly from the profile (e.g. "When can you start?") | Apply throughput & cost | M | needs live machine | FW30 | Needs a live channel back to the candidate mid-apply; none exists today. |

## Closed or merged (full text still kept in docs/future_work.md / docs/decisions_archive.md)

| ID | Item | Status |
|---|---|---|
| FW17 | Bare Canadian provinces | built #224; check live corpus for false positives |
| FW26 | Follow-up questions for thin entries | built #229 |
| FW27 | GitHub import shows flagged text | built #225 |
| FW36 | Persistent Playwright MCP research | answered by experiment #221 |
| FW45 | HITL banner guidance | built #226 |
| FW49 | Hand Off progress | built #226 |
| FW51 | Cookie-banner auto-decline | built #227 |
| FW55 | Company names for multi-tenant ATS | built #223 |
| FW60 | Wizard choice prompts | concrete fix built #228 |
| E1 | numpy pin / Python 3.12+ | investigated #232: blocked upstream by a security pin; stay on 3.11 |
| E5 | Duplicate helpers | no change needed #222 |
| E6 | Lint tests/ | built #222 |
| E8 | Enrichment double-processing | root-caused and fixed #230 |
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
| FW13 | Claude budget policy | closed #234: keep reserve-for-apply |
| FW43 | Login/2FA research gate | declined #234: botting-detection risk; gate stays closed |
| FW44 | SMS relay wired into apply | declined #234 with FW43; relay stays a standalone tool |
| FW66 | SMS-gateway Android app | parked with FW44 (#234) |
| FW28 | Requirement-framing classifier | built #236 (shadow); run scripts/validate_requirement_framing.py live, then turn on |
| FW36/Stage A | Persistent Playwright MCP per worker | built #235, default; watch the first live apply run |
| FW39 | Proxy/fingerprint hardening | declined #237; security work is FW72 |
| FW52 | Imperfect assessment answers | declined #237 |
