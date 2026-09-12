"""Deterministic/local scoring fallback for Gemini quota outages.

CLAUDE.md decision #76 (2026-09-06/07): a real production incident showed
`applypilot run score` freezes entirely once every cloud LLM provider hits
its quota cooldown (`llm.py` raises "All LLM providers are on quota
cooldown") -- nothing downstream (tailor/cover/apply) can proceed for a job
until it's scored, and there was no deterministic fallback at all. A full
investigation that same night tried and rejected three other approaches
(naive keyword coverage, frame semantics, embeddings -- all showed no real
signal on properly-cleaned labeled data) before landing on this one:
decompose "is this job a good fit" into ONE atomic, genuinely single-fact
LLM question (occupation family) plus two deterministic regex extractions
(years-of-experience required, CS-degree required), then combine via a
plain Python lookup table -- mirrors `scorer._check_ineligible`'s own
"regex for what's mechanical, model only for what's genuinely semantic"
split, applied one level deeper. Validated on real, clean (post-2026-08-27
fabricated-identity-bug-fix) labeled data at n=52-58: qwen3:1.7b gets ~81%
gate agreement (recall 0.83, precision 0.73, ~26s/job); qwen3:8b gets ~87%
(recall 0.83, precision 0.86, ~76s/job, slower but meaningfully more
precise) -- both real, both options here via the `model` parameter.

Deliberately NOT wired into the automatic score_job() fallback chain for
the general case, and NOT run automatically as part of the normal
pipeline's happy path. This is real, imperfect data (81-87% gate
agreement, not 100%) -- an honest score_error/pending-retry is normally
recoverable once quota resets; a bad deterministic score silently reaching
`tailored`/`ready_to_apply` is not (see the real fabricated-identity-score
incident, decision #75, for exactly what that class of mistake costs). So
this module is still only ever invoked explicitly for BULK/manual use
(`applypilot revalidate-deterministic-fallback-scores` covers the cleanup
half; the scoring half is `run_deterministic_fallback_scoring`, wired to a
dedicated CLI flag) against jobs already stuck specifically on a
quota-cooldown score_error -- never against jobs that simply haven't been
scored yet for other reasons.

2026-09-11 (decision #119) narrowed, not reversed, the "no automatic path"
rule above: `scorer._flush_score_batch` now calls `score_job_deterministic`
directly, but ONLY at the exact moment a job has exhausted all
`MAX_SCORE_RETRIES` cloud attempts and would otherwise be marked
`score_failed` permanently -- a real production run found 62 such jobs,
stuck purely on repeated quota-outage bad luck with no recovery path at
all. "Permanently failed" is strictly worse than "imperfect but real, and
still revalidation-eligible" -- the original caution's own logic (prefer
recoverable-via-retry over an unrevalidated bad guess) doesn't apply once
retry has genuinely been exhausted, since there's nothing left to recover.
This one narrow caller is the ONLY automatic invocation; the general "score
whatever's currently pending" path is still 100% manual.

Every fallback-scored row (whether reached via the manual bulk path or this
one narrow automatic rescue) is tagged `score_method = 'deterministic_fallback'`
(see database.py's `_ALL_COLUMNS`) so it stays visibly distinguishable from
a real LLM score and is revalidation-eligible once quota returns.
"""

from __future__ import annotations

import math
import os
import re
from datetime import UTC, datetime

from applypilot.llm import LLMClient, ModelEntry, local_openai_base_url
from applypilot.scoring.compensation import classify_compensation, compensation_score_adjustment
from applypilot.scoring.scorer import _check_ineligible, _classify_ineligibility, _flush_score_batch

SCORE_METHOD = "deterministic_fallback"

# 2026-09-09: found via the same real Gemini-comparison batch as the
# years-mention/header fixes above -- a real BuiltIn "Software Engineer,
# Front-End" posting's requirements section (a bare "Qualifications"
# header, "7 years of experience...") starts at character 3196 of a
# 6208-char description, past the old 3000-char cutoff every extraction
# function in this module used -- longer, more verbose postings routinely
# put a lengthy company-intro/responsibilities section before
# requirements. scorer.py's own real LLM-scoring prompt already uses a
# 6000-char window for the full description; this module's deterministic
# extraction functions had never been aligned to it.
DESCRIPTION_WINDOW = 6000

# 8b is the recommended default: this path is only ever invoked explicitly
# by a human already choosing to accept slower scoring during a quota
# outage (rather than nothing progressing at all), so the ~3x latency cost
# vs. 1.7b is worth the real precision gain (0.86 vs 0.73 at n=52-58) --
# fewer false "good fit" scores means fewer wasted tailor/cover/apply
# credits on a bad match before the revalidation sweep catches it. Override
# with the "fast" model name directly if you'd rather trade precision for
# throughput on a big backlog.
DEFAULT_MODEL = os.environ.get("APPLYPILOT_DETERMINISTIC_FALLBACK_MODEL", "qwen3:8b")
LOCAL_URL = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")

_FAMILY_SYSTEM = """Classify a job posting's CORE day-to-day work into exactly ONE of \
five families. Ignore the company/industry -- classify the actual work described. Be \
STRICT: only use the first three families if the posting is a clear, direct match --  \
default to "specialized_or_other" for anything that merely sounds hands-on/technical \
but actually requires formal engineering education, specialized manufacturing/lab \
expertise, or deep domain-specific technical depth.

- it_or_tech_support: IT support / help desk / desktop support / technical support / \
systems administration / network engineering -- fixing/maintaining computers, \
software, or IT systems for end users.
- hands_on_repair_or_trade: hands-on REPAIR, MAINTENANCE, or INSTALLATION of physical \
equipment, vehicles, appliances, or machinery, using tools and manual troubleshooting \
(e.g. automotive technician, equipment technician, maintenance mechanic, installation \
tech). NOT manufacturing/assembly-line production work, NOT laboratory/quality-control \
work, NOT formal engineering design/analysis work -- those go to specialized_or_other \
even though they can also involve "hands-on" tasks.
- customer_facing_or_sales: direct customer-facing sales, customer service, or account \
management, where the core job is talking to/serving customers (not engineering or \
technical support as the primary function).
- software_engineering: software / backend / frontend / data / ML / DevOps engineering, \
where programming is the core, primary skill.
- specialized_or_other: everything else, INCLUDING: manufacturing/production/assembly- \
line work, laboratory or quality-control/quality-assurance technician work, formal \
engineering roles (structural, mechanical design, electrical/IC design, embedded \
systems, process/manufacturing engineering), enterprise IT service management/ \
delivery roles, finance/legal/clinical work, or anything requiring specialized \
technical/engineering depth beyond basic hands-on troubleshooting.

First write ONE short sentence naming the core day-to-day work described in the \
posting. Then, on its own final line, output exactly one of: FAMILY: it_or_tech_support, \
FAMILY: hands_on_repair_or_trade, FAMILY: customer_facing_or_sales, \
FAMILY: software_engineering, or FAMILY: specialized_or_other."""

# 2026-09-09: found via a real accuracy spot-check of the 2026-09-08
# backlog run -- a Sana "Physician" posting and an Eagle Family Medicine
# "Medical Assistant or LPN" posting both got FAMILY: customer_facing_or_sales
# (score 9) from qwen3:1.7b, despite _FAMILY_SYSTEM already explicitly
# listing "clinical work" under specialized_or_other. Root cause: these
# postings are full of patient/compassion/service language ("compassionate
# and patient-oriented", "serving customers") that reads to the small model
# like customer-service vocabulary, overriding its own explicit
# instruction. Rather than trust the LLM to reliably honor a carve-out it
# already ignored twice, this is a narrow, high-confidence deterministic
# override -- same design philosophy as scorer.py's _TS_SCI_PATTERN/
# _CLEARANCE_REQUIRED_PATTERN: any clearly-licensed clinical title bypasses
# the LLM call entirely and goes straight to specialized_or_other (also
# saves a local-model call on these unambiguous cases).
_CLINICAL_LICENSE_TITLE_RE = re.compile(
    r"\bphysician\b|\bnurse\s*practitioner\b|\bregistered\s+nurse\b|\b(?:rn|lpn|lvn|pa-c)\b"
    r"|\bphysician\s+assistant\b|\bmedical\s+assistant\b|\bdentist\b|\bpharmacist\b"
    r"|\bveterinarian\b|\b(?:physical|occupational)\s+therapist\b|\bpsychiatrist\b",
    re.IGNORECASE,
)

# 2026-09-09: found via a real Gemini-vs-local comparison run (30 real
# jobs re-scored by both) -- nearly every "software_engineering" family
# job in the sample had years_required=None locally despite Gemini's own
# reasoning quoting an explicit, real stated requirement from the SAME
# posting text. Real phrasing gaps, each confirmed against real posting
# text before fixing (not guessed):
# (a) a spelled-out number repeated parenthetically -- "at least eight (8)
#     years of professional experience" (Clear Street) -- the digit isn't
#     directly followed by whitespace+"years"; ")" sits in between, so the
#     old \s*years? never matched at all. Fixed with an optional
#     "word (" prefix before the captured digit and an optional ")" after.
# (b) "N+ years" or bare "N years" with NO "experience" word anywhere
#     nearby -- "5+ years working on complex systems..." (Cash App). The
#     old regex hard-required "...experience" appear in a trailing window
#     regardless of phrasing, which means catching every real verb
#     phrasing ("working on"/"building"/"leading"/...) would need an
#     ever-growing, never-complete word inventory. Fixed by DECOUPLING
#     "is this worth considering as a candidate mention" (now maximally
#     permissive -- any "N(+)? years/yrs", no trailing-context
#     requirement at all) from "does it count as a hard requirement"
#     (unchanged: only a recognized required-section header or an inline
#     required/must-have/minimum-of phrase qualifies -- see
#     extract_years_required). The requirement-detection logic never
#     needs a verb list at all this way.
# (c) "yrs" abbreviation and a hyphenated "5-years" separator, both
#     common in real postings, added to the unit-word/separator classes.
# (d) an explicit range -- "Entry Level / 1 - 3 years in the role"
#     (real Truist "Wealth Support Specialist I" posting, en dash "-"
#     between the numbers) -- previously extracted the UPPER bound (3),
#     since the single-number alternative matches starting at the second
#     digit once the first digit's own attempt fails (the "-3" after "1"
#     isn't "years"). Taking the upper bound systematically
#     under-credits a candidate who meets the range's actual floor,
#     inconsistent with this function's own stated intent of taking the
#     SMALLEST qualifying number across the whole posting. A dedicated
#     range alternative, tried first, now captures the LOWER bound
#     instead. Handles hyphen, en dash, and em dash separators (real
#     postings use all three depending on source formatting).
# (e) a decimal years mention -- "1.5+ years of experience as a software
#     engineer" (real Affirm posting) -- the old \d{1,2}-only pattern
#     matched nothing at this position at all (found via decision #88's
#     real Claude-vs-Gemini comparison). Both number slots now accept an
#     optional decimal component; extract_years_required rounds UP
#     (ceil) when converting to the int deterministic_combine expects --
#     "1.5+ years" is closer in spirit to "you need almost 2 years" than
#     "you need just 1", so rounding down would understate the real bar.
# (f) "N or more years" (real Boeing postings, "9 or more years of
#     related work experience") -- a natural-English equivalent of "N+"
#     that the old pattern, which only recognized a literal "+", never
#     matched. Added as an alternative to "+" wherever it appeared.
# (g) a SPELLED-OUT number word -- "Two years of teller or cash handling
#     experience" (real, recurring Truist template across multiple branch
#     postings) -- no digit at all, so no numeric pattern could ever have
#     matched. A dedicated word-number alternative (one-twelve) with its
#     own capture group; the caller looks the word up in _NUMBER_WORDS.
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_YEARS_MENTION_RE = re.compile(
    r"\b(\d{1,2}(?:\.\d+)?)\s*[-–—]\s*\d{1,2}(?:\.\d+)?(?:\+|\s+or\s+more)?[\s-]*(?:years?|yrs?)\b"
    r"|\b(?:[a-z]+\s*\()?(\d{1,2}(?:\.\d+)?)\)?(?:\+|\s+or\s+more)?[\s-]*(?:years?|yrs?)\b"
    r"|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_REQUIRED_CONTEXT_RE = re.compile(r"\b(?:required|must have|minimum of)\b", re.IGNORECASE)
# 2026-09-09: "minimum N years" (no "of") -- real RTX "Supplier Performance
# Specialist" posting, "minimum 5 years prior relevant experience" -- the
# _REQUIRED_CONTEXT_RE phrase above only recognized "minimum of", missing
# this extremely common variant entirely. Deliberately a SEPARATE, tightly
# proximity-scoped check (immediately before the mention, not the wide
# symmetric 60-char window _REQUIRED_CONTEXT_RE uses) rather than just
# adding bare "minimum" to that shared pattern -- a first attempt doing
# exactly that caused a real false positive caught before shipping: a
# Sherwin-Williams posting's "Minimum Requirements:" SECTION HEADER (an
# unrelated line, describing a completely different bullet) sat within
# the wide window of an unrelated "eighteen (18) years of age" mention
# nearby and wrongly vouched for it as an experience requirement.
_MINIMUM_PREFIX_RE = re.compile(r"\bminimum\s*$", re.IGNORECASE)
# 2026-09-09: broadening (b) above means a bare "N years" mention no
# longer needs the word "experience" nearby to be considered a candidate
# -- opening a real false-positive class the old "experience" requirement
# used to block structurally: a number-of-years mention that has nothing
# to do with professional experience at all (license/certification
# RENEWAL CADENCE, warranty periods, tenure/anniversary benefits) sitting
# inside an otherwise-recognized required-qualifications section. Rather
# than an open-ended positive list of experience-indicating verbs, this is
# a short, closed negative list -- checked against just the mention's OWN
# LINE (not a flat character radius, which can wrongly bleed context from
# an adjacent, unrelated bullet point in a tightly-packed list -- confirmed
# this exact cross-bullet contamination with a real-shaped two-bullet test
# before scoping it to the line). Deliberately keyed on the
# RENEWAL/VALIDITY-PERIOD framing (renew/valid for/expire/warranty/tenure/
# anniversary/PTO), NOT on the bare credential nouns "license"/
# "certificate" themselves -- a real false-negative caught before shipping:
# a genuine Boeing "Machine Repair Mechanic" requirement, "1+ years of
# related experience OR A COMPLETED TECHNICAL CERTIFICATE / post-secondary
# degree," uses "certificate" as an ALTERNATIVE-credential noun in the same
# bullet as the real years-of-experience requirement -- the original
# broader list (bare "licen[cs]\w*"/"certificat\w*") wrongly excluded this
# entirely valid mention just because that word appeared later in the same
# sentence for an unrelated reason.
# 2026-09-09: added "of age" -- real Sherwin-Williams "Bilingual Customer
# Service Specialist" posting, "Must be at least eighteen (18) years of
# age" -- a minimum-AGE requirement (universal, harmless, present on
# nearly every job posting), not an experience requirement. Caught while
# verifying the "minimum N years" fix below: "Minimum Requirements:"
# (the section's own header, unrelated to this bullet) sat within the
# qualifying window and wrongly vouched for the age mention.
# 2026-09-09: added "full[- ]time education" -- Accenture's own recurring
# real template across its India/Philippines postings, "15 years full
# time education" (India's convention for "equivalent of a bachelor's
# degree," counting years of schooling, not professional tenure), often
# phrased "A 15 years full time education is required" -- the literal
# word "required" sits right next to it, letting an EDUCATION-duration
# marker qualify as an experience-years mention. Confirmed dormant rather
# than yet-consequential in the live 362-job batch (a real, smaller
# professional-experience mention always won the min() comparison in
# every case checked), but a real defect nonetheless: 9/13 rows with this
# marker have "required" close enough to trigger it, and this exact
# phrasing is Accenture's own reused template, not a one-off.
# 2026-09-09 (decision #92): added "or older" -- a real Avionics
# Technician posting phrases the same universal minimum-age requirement
# as "Must be 18 years or older" rather than "18 years of age" -- the
# existing "of age" exclusion didn't cover this equally common phrasing.
_NON_EXPERIENCE_YEARS_CONTEXT_RE = re.compile(
    r"\b(?:renew\w*|valid\s+for|expir\w*|warrant\w*|tenure|anniversary|"
    r"\bpto\b|paid\s+time\s+off|of\s+age|or\s+older|full.?time\s+education)\b",
    re.IGNORECASE,
)

# 2026-09-08 (decision #82): found via a real manual accuracy spot-check of
# a live scoring batch -- a real Sourcegraph "Security Engineer" posting
# ("MINIMUM QUALIFICATIONS\n\nBachelor's degree with 8+ years of hands-on
# experience with Tenable.io...") scored a false-positive 9/10 because its
# years requirement is established by a SECTION HEADER, not an inline
# "required"/"must have"/"minimum of" phrase next to the number --
# _REQUIRED_CONTEXT_RE's 60-char window never sees any of those words. A
# live check of the same batch found this wasn't a one-off: 4/125 jobs
# scored >=7 had a qualifications header + a years-mention that
# extract_years_required missed entirely. "Preferred"/"desired"/"nice to
# have" headers are deliberately excluded from this -- a years-mention
# under a preferred-only section is genuinely optional, not a hard
# requirement, and must still return None.
# 2026-09-09: extended with four more real header conventions found in the
# same Gemini-comparison batch, each on its OWN line followed directly by
# a plain bullet list of hard requirements: "Requirements:" (Clear
# Street), "About You:" (Vercel), "You Have" (Cash App -- no colon at
# all), "Qualifications" bare, no Minimum/Required/Basic prefix (a
# front-end role at an unnamed employer -- also no colon). The colon is
# inconsistent across real postings (some HTML-to-text conversions keep
# it, some don't), so this is now anchored to the header being ALONE on
# its own line (^...$ with MULTILINE) rather than requiring a trailing
# colon -- bare single words like "requirements"/"qualifications" are
# extremely common in ordinary prose ("this role has strict requirements
# around location", "system requirements: 8GB RAM"), but never as the
# ENTIRE content of their own line the way a real section header is;
# verified this discriminates correctly against exactly those two
# realistic false-positive shapes before shipping.
# 2026-09-09: extended further via a real larger-batch scoring comparison.
# (h) Boeing's own recurring template, "Basic Qualifications (Required
#     Skills and Experience):" / "...(Required Skills/Experience):" --
#     appeared in multiple real Boeing postings in the same batch and
#     never matched, since the old pattern required the header to be
#     JUST "Basic Qualifications" with nothing else on the line. Now
#     allows an optional parenthetical annotation before the colon.
# (i) three more real header conventions: "What We're Looking For" (a
#     real Accenture posting; distinct from the already-handled bare
#     "What We Look For"), "What You'll Bring", "What We Require", and
#     "Typically requires:" (a real RTX posting). Apostrophes matched via
#     "." (any char) rather than a literal quote, mirroring the earlier
#     curly-vs-straight-apostrophe lesson from _CS_DEGREE_RE.
# 2026-09-09 (decision #92): a real SunTech Medical "Technical Support
# Repair Technician" posting renders its header as
# "**Minimum Qualifications  \n  \n**" -- a Markdown-to-text conversion
# artifact (bold markers + hard-line-break trailing spaces) that split
# the header's own "**" wrapper across two lines. The whole-line match
# never allowed for leading/trailing "**" at all, so this recognized
# header phrase was invisible and its "3+ years working in electronics
# manufacturing facility" requirement went uncredited. Added optional
# `\*{0,2}` on both sides of the phrase -- still a strict whole-line
# match otherwise, so this can't match a header phrase merely mentioned
# mid-sentence for emphasis.
_REQUIRED_SECTION_HEADER_RE = re.compile(
    r"^\s*\*{0,2}\s*(?:(?:minimum|required|basic)\s+qualifications(?:\s*\([^)]{0,80}\))?|requirements|about\s+you|"
    r"you\s+have|qualifications|what\s+we\s+look\s+for|what\s+we.re\s+looking\s+for|"
    r"what\s+you.ll\s+bring|what\s+we\s+require|typically\s+requires)\s*:?\s*\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NEXT_SECTION_HEADER_RE = re.compile(
    r"\b(?:preferred|desired|nice.to.have|bonus)\s+qualifications\b|\bpreferred\s+skills\b|\bnice.to.haves?\b",
    re.IGNORECASE,
)
# 2026-09-09 (decision #92): a real Rooms To Go "Furniture Service Tech"
# posting has no "Preferred Qualifications"-style closing header, so
# _required_section_span's 1500-char fallback window swept in an
# unrelated "About Rooms To Go" company-history blurb ("Founded in 1991
# ... More than 30 years later...") sitting right after the requirements
# bullet list -- "30 years" wrongly counted as an experience requirement.
# Whole-line anchored like _REQUIRED_SECTION_HEADER_RE (so it can't match
# "about" merely appearing mid-sentence); deliberately excludes "about
# you", which is itself a REQUIRED-section start header above, not an
# end-of-section boundary.
_ABOUT_COMPANY_HEADER_RE = re.compile(
    r"^\s*\*{0,2}\s*about\s+(?!you\b)\S.*?\s*:?\s*\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# 2026-09-09: found via the same real spot-check that surfaced the
# clinical-title bug above -- a Cisco "Cloud Engineer" posting states
# "Minimum Qualifications \n\n Bachelor's degree in computer science,
# Computer Engineering, or a related technical field" and scored a
# false-positive 9. This is the exact same section-HEADER-not-inline-word
# gap decision #82 already found and fixed for extract_years_required, just
# never ported to this sibling function -- the degree phrase has no literal
# "required" anywhere near it, so the old regex (which demanded "required"
# within 30 chars) never matched. Fixed the same way: a degree+field match
# now also qualifies if it falls inside a "Minimum/Required/Basic
# Qualifications" section, not just next to the literal word "required".
_CS_DEGREE_RE = re.compile(
    r"\b(?:bachelor(?:['’]s)?|b\.?s\.?)\s+degree\b[^.\n]{0,60}\b"
    r"(?:computer science|computer engineering|software engineering)\b",
    re.IGNORECASE,
)


def _line_span(text: str, pos: int) -> tuple[int, int]:
    """Start/end of the line containing ``pos`` -- used to scope the
    _NON_EXPERIENCE_YEARS_CONTEXT_RE check to the mention's own bullet
    point rather than a flat character radius that can bleed context from
    an adjacent, unrelated line."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, end if end != -1 else len(text)


def _required_section_span(text: str) -> tuple[int, int] | None:
    """Find a "Minimum/Required/Basic Qualifications"-style section header
    and return (start, end) of the text it covers -- from just after the
    header to the next section header (a "Preferred Qualifications"-style
    heading) or a bounded 1500-char window, whichever comes first. Returns
    None if no such header is found. See extract_years_required's
    2026-09-08 decision #82 note for why this exists alongside the
    inline-phrase check."""
    m = _REQUIRED_SECTION_HEADER_RE.search(text)
    if not m:
        return None
    start = m.end()
    candidates = [
        stop.start()
        for stop in (
            _NEXT_SECTION_HEADER_RE.search(text, start),
            _ABOUT_COMPANY_HEADER_RE.search(text, start),
        )
        if stop is not None
    ]
    end = min(candidates) if candidates else min(len(text), start + 1500)
    return start, end


def extract_years_required(description: str) -> int | None:
    """Minimum years-of-experience explicitly stated as REQUIRED (not just
    "preferred"), or None if no such hard requirement is found. A
    years-mention qualifies if EITHER "required"/"must have"/"minimum of"
    appears within a nearby inline window, OR it falls inside a "Minimum/
    Required/Basic Qualifications" section (2026-09-08, decision #82 --
    real postings very commonly state requirements under a section HEADER
    rather than repeating "required" next to every number). A bare
    years-mention with neither signal, or one that only appears under a
    "Preferred Qualifications"-style section, is treated as not-a-hard-
    requirement. A mention whose own line reads as clearly non-experience
    (license/certification renewal, warranty, tenure -- see
    _NON_EXPERIENCE_YEARS_CONTEXT_RE) is excluded even if it otherwise
    qualifies. No length limit -- this is a pure regex scan, not an LLM
    call, so there's no cost reason to truncate (2026-09-09: found via a
    real BuiltIn posting whose requirements section started past the old
    3000-char cutoff -- unlike classify_family's LLM prompt, which stays
    bounded at DESCRIPTION_WINDOW for real token-cost reasons, these
    regex-only extractors now scan the full description)."""
    text = description or ""
    qualifying: list[int] = []
    required_span = _required_section_span(text)
    for m in _YEARS_MENTION_RE.finditer(text):
        ls, le = _line_span(text, m.start())
        if _NON_EXPERIENCE_YEARS_CONTEXT_RE.search(text[ls:le]):
            continue
        # 2026-09-09: the trailing side of this window is bounded to the
        # mention's OWN LINE (`le`), not a flat +60 chars -- a real cross-
        # section bleed found via a live "IT and Telecom Field Technicians"
        # posting: "...5+ years verifiable field experience in I.T./
        # Telecom\nRequired Equipment & Qualifications\n..." -- the word
        # "Required" starting an entirely different, unrelated section
        # (equipment, not experience) on the NEXT line wrongly vouched for
        # this mention. A same-line (or earlier-line, via the leading
        # -60 side, unchanged) "required"/"must have" still counts --
        # e.g. "5 years of experience required." (same line, trailing) and
        # "Qualifications You Must Have\n...\n1+ years..." (must-have on
        # an earlier line) both still work, verified directly against
        # both real cases before shipping this narrower bound.
        window = text[max(0, m.start() - 60) : min(m.end() + 60, le)]
        in_required_section = required_span is not None and required_span[0] <= m.start() < required_span[1]
        minimum_prefix = _MINIMUM_PREFIX_RE.search(text[max(0, m.start() - 15) : m.start()])
        if in_required_section or _REQUIRED_CONTEXT_RE.search(window) or minimum_prefix:
            numeric = m.group(1) or m.group(2)
            if numeric is not None:
                qualifying.append(math.ceil(float(numeric)))
            else:
                word_years = _NUMBER_WORDS.get((m.group(3) or "").lower())
                if word_years is not None:
                    qualifying.append(word_years)
    if not qualifying:
        return None
    return min(qualifying)


def extract_cs_degree_required(description: str) -> bool:
    """A Bachelor's-in-CS/CE/SWE degree stated as a hard requirement --
    either inline next to "required", or under a "Minimum/Required/Basic
    Qualifications" section header (see _CS_DEGREE_RE's 2026-09-09 note).
    No length limit -- see extract_years_required's note on why these
    regex-only extractors don't share classify_family's LLM-prompt-cost
    reason to truncate."""
    text = description or ""
    required_span = _required_section_span(text)
    for m in _CS_DEGREE_RE.finditer(text):
        in_required_section = required_span is not None and required_span[0] <= m.start() < required_span[1]
        window = text[max(0, m.start() - 30) : m.end() + 30]
        if in_required_section or _REQUIRED_CONTEXT_RE.search(window):
            return True
    return False


def local_only_client(model: str) -> LLMClient:
    client = LLMClient(LOCAL_URL, model, "", quality=True)
    client._fallback_chain = [ModelEntry(model, "local", local_openai_base_url(LOCAL_URL), "")]
    return client


def classify_family(client: LLMClient, job: dict) -> str | None:
    if _CLINICAL_LICENSE_TITLE_RE.search(job.get("title") or ""):
        return "specialized_or_other"
    job_text = (
        f"TITLE: {job['title']}\nCOMPANY: {job['site']}\n\nDESCRIPTION:\n{(job.get('full_description') or '')[:DESCRIPTION_WINDOW]}"
    )
    messages = [{"role": "system", "content": _FAMILY_SYSTEM}, {"role": "user", "content": job_text}]
    try:
        resp = client.chat(messages, max_tokens=500, temperature=0.2)
    except Exception:  # noqa: BLE001
        return None
    m = re.search(
        r"FAMILY:\s*(it_or_tech_support|hands_on_repair_or_trade|customer_facing_or_sales"
        r"|software_engineering|specialized_or_other)",
        resp or "",
        re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


def deterministic_combine(family: str | None, years: int | None, cs_degree: bool) -> int:
    """Pure Python, zero LLM. Real, validated table (see module docstring
    for the n=52-58 validation results) over the three checkbox facts."""
    if family is None:
        return 5  # couldn't classify -- neutral, not a guess in either direction
    if family == "specialized_or_other":
        return 3  # deliberately conservative: catches plausible-but-unrelated occupations
    if family in ("it_or_tech_support", "hands_on_repair_or_trade", "customer_facing_or_sales"):
        if cs_degree:
            return 5  # unusual for these families, but a hard-ish requirement caps it
        if years is None or years == 0:
            return 9
        if years == 1:
            return 7
        return 5  # years >= 2
    if family == "software_engineering":
        if cs_degree:
            return 3
        if years == 0:
            return 7  # explicitly entry-level
        if years is None:
            return 5  # uncertain, not optimistic -- ownership/scope language can imply seniority regex can't see
        # 2026-09-09 (decision #89): found via a real Claude-vs-Gemini
        # comparison run (27 real jobs, decision #88) -- this branch used
        # to map EVERY years>=1 to a flat 3, never distinguishing "1 year,
        # rest learnable" from "10+ years, deep specialist" the way the
        # sibling families above already do (years==1 -> 7, years>=2 ->
        # 5). Both Claude and Gemini independently scored every real 5+/
        # 6+/7/8/10+-year software posting a 1, while a real 1-3-year-
        # range posting (floor=1) landed at 5 for both -- and the written
        # SCORE_PROMPT_TEMPLATE rubric itself already describes exactly
        # this ladder in prose ("5-6: ...nominally ~1 year... learnable",
        # "3-4: ...2+ years...", "1-2: ...3+ years...") that this lookup
        # table had never actually implemented. Calibrated directly
        # against both the rubric's own stated bands and the real n=27
        # convergence, not guessed.
        if years == 1:
            return 5
        if years == 2:
            return 3
        return 1  # years >= 3
    return 5


def is_quota_cooldown_error(score_error: str | None) -> bool:
    return bool(score_error) and "quota cooldown" in score_error.lower()


# 2026-09-07 (Future Work item 2): self-consistency escalation (ask the
# fast model the same question twice, escalate on disagreement) was tried
# and rejected -- the fast model's errors are systematic, not randomly
# uncertain, so asking twice just gets the same wrong answer twice (only
# 1/58 jobs ever escalated). Real signal found instead by directly
# comparing qwen3:1.7b's and qwen3:8b's actual family classifications on
# the SAME 52 real jobs (both already had full result sets from separate
# validation runs -- no new model calls needed to find this): family
# disagreements cluster tightly around a specific, nameable occupational
# zone -- manufacturing/hands-on-adjacent titles the 1.7b model
# systematically confuses with the candidate's real trade/repair
# background. A title-keyword trigger built directly from those observed
# disagreement titles catches 5/6 of the disagreements that actually flip
# the >=8 gate decision, while only escalating 33% of jobs to the slower
# model -- gate agreement rises from 79% (1.7b alone, this same n=52
# sample) to 85% (hybrid), close to pure-8b's 87%, at roughly 1/3 the
# latency cost. HONEST CAVEAT: this list was built FROM the exact n=52
# sample it's validated against -- the same small-n-overfitting trap this
# session already got bitten by once (decision #76's sales-rep
# overcorrection). Ship as opt-in, not the default, until revalidated on a
# fresh, independently-sampled batch.
#
# 2026-09-08 (decision #81): re-validated using ONLY data already on hand
# (no new model calls, no Gemini quota needed -- see
# data/experiments/ambiguous_terms_20260908/validate_escalation_trigger.py).
# Two findings: (1) a bootstrap 95% CI on this same n=52 sample is WIDE and
# OVERLAPPING across all three configurations (1.7b-alone 78.8%
# [67.3%,88.5%], 8b-alone 86.5% [76.9%,94.2%], hybrid 84.6% [75.0%,94.2%])
# -- the point estimates above are accurate, but at this sample size the
# hybrid's apparent edge over 1.7b-alone is NOT statistically distinguishable
# from noise; keeping this opt-in rather than default remains the right
# call, now for a quantified reason rather than a vague "small n" caveat.
# (2) "technician" -- the single most frequently-firing alternative (11/52
# titles) -- contributes ZERO unique catches: every real fast/slow
# disagreement its pattern matches is ALSO independently matched by a
# strictly more specific alternative already in this list (composites,
# field service, maintenance, assembler, embedded each catch their own
# case even with "technician" removed). Verified directly: removing it
# drops escalation volume 17/52 -> 13/52 (-23.5%) with IDENTICAL hybrid
# gate agreement (84.6%, unchanged). This is a structural redundancy
# elimination, not a re-fit to the same sample's noise -- safe to make
# without fresh data because it doesn't rely on any NEW claim about what
# generalizes, only on the fact that "technician" duplicates coverage
# already provided by more specific patterns. NOTED CONCERN, not yet
# acted on: "field service"'s one unique disagreement catch ("Early
# Career Field Service Technician") is actually a case where escalating
# HURTS -- 1.7b was correct (fast=9, real=9) and 8b was wrong (slow=3),
# one of 8b's own documented specialized_or_other blind spots (decision
# #77). Left in place pending more data (n=1 for this specific pattern is
# too little to act on either way), flagged for the next real revalidation
# pass alongside a fresh recall check once Gemini quota returns and more
# real positives accumulate (see decision #81's auto-resume scheduling).
_AMBIGUOUS_TITLE_RE = re.compile(
    r"maintenance|assembler|composites|field service|embedded|infotainment",
    re.IGNORECASE,
)


def score_job_deterministic(
    job: dict,
    profile: dict,
    conn=None,
    model: str | None = None,
    escalate_model: str | None = None,
) -> dict:
    """Mirrors scorer.score_job's return contract ({"score", "keywords",
    "reasoning", "eligibility", "compensation"}), but produces the score via
    the deterministic/local-model fallback instead of a cloud LLM call.

    Reuses scorer._check_ineligible and scoring.compensation unchanged --
    same deterministic pre-filter and pay-uncertainty adjustment the real
    LLM path already applies, so a fallback-scored ineligible job is exactly
    as trustworthy as an LLM-scored one (identical deterministic function),
    and the fallback score isn't missing the free compensation signal
    real-world reasoning text shows the LLM path actually uses.

    ``escalate_model``: when given, jobs whose title matches
    _AMBIGUOUS_TITLE_RE are classified with this (presumably slower, more
    accurate) model INSTEAD of ``model`` -- a direct substitution, not a
    tiebreak, matching what was actually validated (see the module-level
    note above). Opt-in / off by default.
    """
    ineligible_reason = _check_ineligible(job, profile)
    if ineligible_reason:
        result = {
            "score": 2,
            "keywords": "",
            "reasoning": f"Ineligible: {ineligible_reason}.",
            "eligibility": _classify_ineligibility(ineligible_reason),
        }
        if ineligible_reason.startswith("commission-only compensation:"):
            result["compensation"] = {"status": "explicitly_absent"}
        return result

    effective_model = model or DEFAULT_MODEL
    escalated = False
    if escalate_model and _AMBIGUOUS_TITLE_RE.search(job.get("title") or ""):
        effective_model = escalate_model
        escalated = True

    client = local_only_client(effective_model)
    family = classify_family(client, job)
    years = extract_years_required(job.get("full_description") or "")
    cs_degree = extract_cs_degree_required(job.get("full_description") or "")
    score = deterministic_combine(family, years, cs_degree)

    result = {
        "score": score,
        "keywords": "",
        "reasoning": (
            f"[deterministic fallback, model={effective_model}{' (escalated)' if escalated else ''}] "
            f"family={family} years_required={years} cs_degree_required={cs_degree}"
        ),
        "eligibility": "eligible",
    }

    comp = classify_compensation(job, conn=conn)
    result["compensation"] = comp
    adjustment, note = compensation_score_adjustment(comp)
    if adjustment:
        result["score"] = max(1, result["score"] + adjustment)
    if note:
        result["reasoning"] = f"{result['reasoning']} {note}".strip()

    return result


# 2026-09-07 near-miss, documented rather than quietly patched: a "smoke
# test" invocation of this function with no --limit against the real
# production DB found 7,041 real jobs genuinely stuck on quota cooldown --
# at qwen3:8b's real ~76s/call, an unbounded run would have taken multiple
# DAYS of continuous local-model inference on this machine, kicked off by
# what looked like routine verification. `limit` now defaults to a small,
# safe number rather than 0/unlimited -- pass an explicit limit (or 0
# deliberately) to process more. This mirrors the module's own core safety
# principle (explicit invocation only, never silently large-scale) one
# level deeper: even an explicit invocation still needs an explicit *scope*.
DEFAULT_LIMIT = 10


def run_deterministic_fallback_scoring(
    conn=None,
    limit: int = DEFAULT_LIMIT,
    model: str | None = None,
    escalate_model: str | None = None,
    scope: str = "quota_cooldown",
) -> dict:
    """Score jobs using the local deterministic fallback scorer.

    ``scope="quota_cooldown"`` (default): ONLY jobs currently stuck on a
    quota-cooldown score_error -- the original decision #76 design, never
    touching a job that hasn't been scored for any other reason.

    ``scope="all_unscored"`` (2026-09-11, decision #129): every job sitting
    in `state='enriched'` with no `fit_score` yet, regardless of
    score_error/score_attempts -- covers jobs never yet attempted
    (score_error IS NULL), jobs that failed for a related-but-differently-
    worded reason (e.g. "All models exhausted after trying: [...]", the
    live-retry sibling of the fast-fail "All LLM providers are on quota
    cooldown" message -- see llm.py's two exhaustion-message call sites),
    and jobs already at MAX_SCORE_RETRIES. This is an explicit, deliberate
    widening of decision #76's original scope, authorized after the
    multi-day Claude-direct scoring audit (CLAUDE.md decisions #94-128,
    1000 real jobs) found the deterministic-gate bug-hunting cluster
    (location patterns, seniority-title regex, clearance phrasing --
    shared by both the real LLM path and this module via `_check_ineligible`)
    flatten to zero new findings across its final several batches -- exactly
    the "evidence there aren't more extraction gaps" condition the user set
    as the trigger for moving the rest of the backlog to local scoring
    (decision #105). Still explicit-invocation only, never wired into the
    automatic pipeline -- this widens WHICH jobs are eligible for a manual
    invocation, not WHEN one happens automatically.

    ``limit`` defaults to DEFAULT_LIMIT (NOT unlimited -- see the 2026-09-07
    near-miss note above); pass ``limit=0`` explicitly to process every
    candidate.

    ``escalate_model``: opt-in title-keyword escalation (see
    score_job_deterministic / _AMBIGUOUS_TITLE_RE) -- off by default.

    Writes are flushed after EVERY job, not batched until the end -- a
    long run (qwen3:8b: ~76s/job) must be safely interruptible without
    losing already-scored progress, and should show real-time DB progress
    to anyone watching, not go silent until the whole run finishes.

    Jobs are processed in TWO GROUPED PASSES (all ``model`` jobs, then all
    ``escalate_model`` jobs) rather than in whatever order the DB query
    returns, so the local model only ever switches once per run instead of
    interleaving. 2026-09-09: real log timing from the 2026-09-08 backlog
    run showed escalated (qwen3:8b) calls costing 139-420s EACH, recurring
    throughout the run rather than warming up once and settling into the
    ~76s/call steady state a dedicated qwen3:8b run shows (decision #77) --
    consistent with Ollama evicting/reloading a model every time the
    requested model differs from what's currently loaded on this machine's
    constrained RAM, not a one-time cold-start cost. Grouping avoids paying
    that reload cost once per escalated title scattered through the run.
    """
    from applypilot.config import load_profile
    from applypilot.database import get_connection

    if conn is None:
        conn = get_connection()
    profile = load_profile()

    if scope == "all_unscored":
        rows = conn.execute("SELECT * FROM jobs WHERE state = 'enriched' AND fit_score IS NULL").fetchall()
    elif scope == "quota_cooldown":
        rows = conn.execute(
            "SELECT * FROM jobs WHERE fit_score IS NULL AND score_error IS NOT NULL "
            "AND score_error LIKE '%quota cooldown%'"
        ).fetchall()
    else:
        raise ValueError(f"Unknown scope {scope!r}: expected 'quota_cooldown' or 'all_unscored'")
    jobs = [dict(r) for r in rows]
    if limit:
        jobs = jobs[:limit]

    if escalate_model and escalate_model != model:
        fast_jobs = [j for j in jobs if not _AMBIGUOUS_TITLE_RE.search(j.get("title") or "")]
        escalate_jobs = [j for j in jobs if _AMBIGUOUS_TITLE_RE.search(j.get("title") or "")]
        jobs = fast_jobs + escalate_jobs

    scored = 0
    for job in jobs:
        result = score_job_deterministic(job, profile, conn=conn, model=model, escalate_model=escalate_model)
        now = datetime.now(UTC).isoformat()
        _flush_score_batch(conn, [{"url": job["url"], **result}], now, score_method=SCORE_METHOD)
        conn.commit()
        scored += 1

    return {
        "candidates": len(rows),
        "scored": scored,
        "model": model or DEFAULT_MODEL,
    }


def revalidate_deterministic_fallback_scores(conn=None, dry_run: bool = False) -> dict:
    """Reset every deterministic-fallback-scored row back to pending so the
    real LLM re-scores it once quota returns. Mirrors
    eligibility.revalidate_stale_scores's "preserve for audit, only change
    what's necessary" convention, adapted here to reset rather than archive
    (a fallback score is a temporary placeholder, not a judgment to
    preserve).

    Ineligible (`state='archived'`) fallback rows are deliberately NOT
    touched: those came from `_check_ineligible`, the exact same
    deterministic function the real LLM path already runs first -- an
    ineligible fallback verdict is exactly as trustworthy as an ineligible
    LLM verdict, so there's nothing to revalidate.
    """
    from applypilot.database import commit_with_retry, get_connection, transition_state

    if conn is None:
        conn = get_connection()

    rows = conn.execute(
        "SELECT url, title, fit_score, state, scored_at FROM jobs "
        "WHERE score_method = ? AND state IN ('scored', 'low_score')",
        (SCORE_METHOD,),
    ).fetchall()
    sample = [dict(r) for r in rows[:20]]

    if dry_run:
        return {"matched": len(rows), "updated": 0, "sample": sample}

    updated = 0
    for row in rows:
        conn.execute(
            "UPDATE jobs SET fit_score = NULL, score_reasoning = NULL, scored_at = NULL, "
            "score_method = NULL, score_error = NULL, score_attempts = 0, "
            "score_next_retry_at = NULL, eligibility = NULL WHERE url = ?",
            (row["url"],),
        )
        transition_state(
            conn,
            row["url"],
            "enriched",
            reason="deterministic_fallback_revalidation: resetting for real LLM re-score",
            metadata={"prior_fallback_score": row["fit_score"]},
            force=True,
        )
        updated += 1

    commit_with_retry(conn)
    return {"matched": len(rows), "updated": updated, "sample": sample}
