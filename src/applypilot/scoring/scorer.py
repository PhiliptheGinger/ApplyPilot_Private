"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta

from applypilot.config import RESUME_PATH, load_profile, load_search_config
from applypilot.database import get_connection, get_jobs_by_stage, write_with_retry
from applypilot.eligibility import seniority_disqualifier
from applypilot.llm import get_stage_client, get_token_limit

log = logging.getLogger(__name__)


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT_TEMPLATE = """You are a job fit evaluator. Given a candidate's resume and a job description, score how well the candidate fits the role.

THE CANDIDATE: {candidate_summary}

LOCATION: The candidate is {location_context}.

⚠️ GEOGRAPHY CHECK — DO THIS FIRST, BEFORE ANYTHING ELSE:
The candidate is US-based. Any role restricted to non-US geography is INELIGIBLE.
Read the FULL description for buried sentences like "This role will be remote and based in the UK"
or "Remote — Ontario, BC or Alberta". A perfect skills match on a non-US role is still INELIGIBLE.

Output ELIGIBILITY: non_us_only when the role's hiring location is restricted to a non-US country
even if the role is remote. Output ELIGIBILITY: eligible when the role is open to US workers
(US-only, US-remote, global-remote, or no restriction).

Common signals for non_us_only:
- "based in (UK|Canada|Ireland|Germany|...)"
- "Remote — (UK|Canada|Europe|EMEA|APAC)"
- "(m/f/d)" / "(m/w/d)" German title suffix
- UK/Canada right-to-work questions in the form
- CET / GMT+N / IST timezone requirement

If non_us_only, you MAY still produce a SCORE based on general fit (for audit), but the
eligibility tag is what determines whether the application proceeds.

⚠️ SENIORITY / EXPERIENCE CHECK — DO THIS SECOND, BEFORE SCORING TECH-STACK OVERLAP:
This candidate has NO professional software engineering employment history. Any role titled or
scoped as Senior / Staff / Principal / Lead / Architect / Director / Manager / VP / Chief, OR any
role that requires 3+ years of professional software engineering experience, OR a completed
CS/engineering degree as a hard requirement, is a POOR fit (score 1-2) no matter how many tech-stack
keywords in the description overlap with the candidate's personal projects. A job description
mentioning "Python" or "automation" does NOT by itself qualify this candidate for an experienced-IC
or senior role — do not let keyword overlap override this check.

SCORING CRITERIA — the candidate's real, documented qualifications are: a CompTIA A+
certification, a Media Studies degree (no CS/engineering degree), hands-on customer-facing and
technical-troubleshooting work history (automotive alignment technician, warehouse operations,
installation, sales), and personal-project-level Python (self-taught, not professional employment):

- 9-10: Entry-level / no-experience-required IT support, help desk, desktop support, technical
  support, customer support, systems administration, or network engineering role. Matches the
  CompTIA A+ certification and hands-on troubleshooting / customer-facing background directly.
  Does not require a CS degree or an advanced cert beyond A+.
- 7-8: Same IT-support family as above but with 1-2 stretch requirements (e.g. "1-2 years
  preferred" rather than required, or a nice-to-have second cert like Network+/Security+), OR a
  genuinely entry-level / new-grad / junior software, backend, or Python role that explicitly does
  not require prior professional software engineering experience or a CS degree.
- 5-6: IT support role requiring 2+ years of experience the candidate doesn't have, OR a junior
  software/backend role nominally wanting ~1 year of experience where the rest of the requirements
  are learnable and stack-agnostic.
- 3-4: Software/backend/data engineering role requiring 2+ years of professional experience or a
  CS degree as a hard requirement, even if the candidate's personal Python projects touch some of
  the listed tech stack.
- 1-2: Any Senior/Staff/Principal/Lead/Architect/Director/Manager/VP/Chief-scoped role, any role
  requiring 3+ years of professional experience or a completed CS/engineering degree, non-technical
  roles that don't match the candidate's actual background (recruiting, design, marketing, product
  management, sales, executive), OR non-US geographic restriction.

ADDITIONAL RULES:
- Distinguish REQUIRED skills from NICE-TO-HAVE. Only penalize for missing required skills, but do
  NOT let a long list of matching keyword buzzwords override the seniority/experience check above.
- LOCATION: if the description implies onsite in a specific city outside the candidate's stated
  location/relocation area above, and is not remote, cap the score at 6.
- Roles requiring a security clearance, or roles at defense, weapons, military, or law-enforcement
  contractors, are OUT OF SCOPE regardless of technical fit — score 1-2 and say why in REASONING.

Output ONLY the four lines below -- no walkthrough, no markdown headers, no
preamble before them. You MUST include all four lines. Do not skip REASONING.

ELIGIBILITY: [eligible|non_us_only]
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-3 sentences explaining the score, what matched well, and any gaps. If non_us_only, name the country/region.]"""


# ── Rule-based pre-filter (catches obvious ineligible before LLM call) ─────
#
# Patterns validated 2026-04-23 against 5,938 historical scored jobs.
# Any pattern that would reject more than 1-2 jobs scored >=8 is omitted
# (those rare false positives tend to be LLM mis-scores anyway — jobs
# titled "Junior" / "Intern" don't belong in a Senior/Staff queue).

# Patterns that make a job ineligible regardless of tech stack.
# Checked against title + location field only (not full description, to avoid
# false positives from US companies mentioning global offices).
_INELIGIBLE_TITLE_PATTERNS = re.compile(
    # Explicit non-US regions in title
    r"\bEMEA\b"
    r"|\bAPAC\b"
    r"|\bLATAM\b"
    r"|\bMENA\b"
    r"|\bTOLA\b"  # sales region: Texas/Oklahoma/Louisiana/Arkansas
    r"|\bANZ\b"  # Australia/New Zealand
    r"|\bNordics\b"
    r"|\bEU[- ]only\b"
    r"|\bUK[- ]only\b"
    r"|\bEurope[- ]only\b"
    r"|\(m/[fw]/d\)"  # German job title suffix (m/f/d) or (m/w/d)
    r"|\bm/[fw]/d\b"
    r"|\bOnly hiring in\b"
    # NOTE: Junior/Intern/Entry-Level/New-Grad/Trainee/Apprentice/Graduate titles are
    # intentionally NOT excluded here — this candidate has no professional software
    # engineering experience, so those levels are the actual target, not noise.
    # Internships/co-ops are still excluded via searches.yaml exclude_titles below.
    # Sales-adjacency (not IC engineering)
    r"|\bSales Engineer\b"
    r"|\bSolutions Engineer\b"
    r"|\bPre[- ]?[Ss]ales\b"
    r"|\bCustomer Success Engineer\b"
    # Retail / warehouse / service roles (filter out Costco + similar noise)
    r"|\bCashier\b|\bBaker\b|\bCake Decorator\b|\bButcher\b|\bMeat Cutter\b"
    r"|\bGas Station Attendant\b|\bPharmacy Technician\b|\bHearing Aid Dispenser\b"
    r"|\bStocker\b|\bForklift\b|\bWarehouse Associate\b|\bTruck Driver\b"
    r"|\bBakery Clerk\b|\bDeli Clerk\b|\bProduce Clerk\b|\bMember Service\b"
    r"|\bOptician\b|\bOptical\b"
    # Non-engineering roles
    r"|\bRecruiter\b"
    r"|\bTalent Acquisition\b|\bTalent Scout\b|\bTalent Sourcer\b"
    r"|\bAccount Manager\b|\bAccount Executive\b"
    r"|\bUX Designer\b|\bUI Designer\b|\bProduct Designer\b|\bGraphic Designer\b"
    # Specialist IC roles outside the target stack (mobile, legacy enterprise)
    r"|\bAndroid Engineer\b|\biOS Engineer\b|\bMobile Engineer\b"
    r"|\bSalesforce Developer\b|\bApex Developer\b"
    r"|\bMainframe Engineer\b|\bCOBOL Developer\b|\bTIBCO\b",
    re.IGNORECASE,
)

# Patterns checked against the location field specifically.
# Location field explicitly lists a non-US country name → ineligible.
_INELIGIBLE_LOCATION_PATTERNS = re.compile(
    # Regions
    r"\bEMEA\b"
    r"|\bAPAC\b"
    r"|\bEurope\b"
    # Europe
    r"|\bGermany\b|\bNetherlands\b|\bFrance\b|\bSpain\b|\bItaly\b"
    r"|\bPoland\b|\bUkraine\b|\bCzech\b|\bPortugal\b|\bIreland\b"
    r"|\bDenmark\b|\bSweden\b|\bNorway\b|\bFinland\b|\bBelgium\b"
    r"|\bSwitzerland\b|\bAustria\b|\bRomania\b|\bHungary\b|\bCroatia\b"
    r"|\bGreece\b|\bBulgaria\b|\bSerbia\b|\bSlovakia\b|\bSlovenia\b"
    r"|\bEstonia\b|\bLatvia\b|\bLithuania\b"
    # Asia
    r"|\bIndia\b|\bSingapore\b|\bJapan\b|\bVietnam\b|\bThailand\b"
    r"|\bPhilippines\b|\bIndonesia\b|\bKorea\b|\bTaiwan\b|\bHong Kong\b"
    r"|\bChina\b|\bPakistan\b|\bBangladesh\b|\bMalaysia\b"
    # Latin America
    r"|\bBrazil\b|\bBrasil\b|\bMexico\b|\bMéxico\b|\bArgentina\b"
    r"|\bChile\b|\bColombia\b|\bPeru\b|\bUruguay\b"
    # Middle East / Africa
    r"|\bEgypt\b|\bNigeria\b|\bKenya\b|\bSouth Africa\b|\bIsrael\b"
    r"|\bTurkey\b|\bTürkiye\b|\bUAE\b|\bSaudi Arabia\b"
    # Oceania
    r"|\bAustralia\b|\bNew Zealand\b",
    re.IGNORECASE,
)

# Catches postings that state an advanced degree as a hard minimum -- common
# Workday/PERM boilerplate ("Minimum Requirements: Master's degree, or foreign
# equivalent, in Computer Science...") that a generic "Software Engineer" title
# gives no hint of. Found via a live example (PayPal "Software Engineer",
# 2026-08-19): scored 8 under the pre-a6f72a6 rubric and sat in ready_to_apply
# with a tailored resume/cover letter already generated. Deliberately narrow --
# only fires on "degree...required"-style minimums, not "Master's preferred"
# or "Bachelor's or Master's" postings the candidate could plausibly clear.
# Scraped descriptions use whatever apostrophe the source site's HTML used --
# straight ('), curly right single quote (U+2019), or curly left (U+2018,
# occasionally misused as an apostrophe) all show up in the wild.
_APOS = "['‘’]?"

_ADVANCED_DEGREE_REQUIRED_PATTERN = re.compile(
    r"(?:master"
    + _APOS
    + r"s|ph\.?d\.?|doctoral|doctorate)\s+degree,?\s+(?:or\s+(?:foreign\s+)?equivalent\s+)?(?:is\s+)?required"
    r"|(?:minimum\s+requirements?|required\s+qualifications?|basic\s+qualifications?)\s*:?\s*(?:[^.\n]{0,40})?"
    r"(?:master" + _APOS + r"s|ph\.?d\.?|doctoral|doctorate)\s+degree"
    r"|master" + _APOS + r"s\s+degree,?\s+or\s+foreign\s+equivalent",
    re.IGNORECASE,
)


def _candidate_has_advanced_degree(profile: dict) -> bool:
    """True if any profile education entry is a Master's/PhD or higher."""
    for item in profile.get("education") or []:
        degree = str((item or {}).get("official_degree") or "").lower()
        if any(term in degree for term in ("master", "m.s.", "phd", "ph.d", "doctor")):
            return True
    return False


# Deterministic clearance/defense hard gate. Found via a live example
# (Workday "Software Development Engineer - ML Ops (US Federal)", 2026-08-25):
# scored 8 -- the LLM prompt's own "clearance roles are OUT OF SCOPE, score
# 1-2" instruction was never applied -- and reached apply_failed (a real
# application attempt) against a role requiring TS/SCI w/CI Poly. Unlike
# geography and seniority, this candidate-eligibility check had no
# deterministic backstop at all -- it depended entirely on LLM adherence.
#
# TWO DELIBERATELY DIFFERENT TIERS, not one blanket "clearance" keyword:
#
# 1. TS_SCI: a bare mention of "TS/SCI" (with or without "CI Poly") is
#    disqualifying on its own, regardless of "required" / "preferred" /
#    "ability to obtain" framing. TS/SCI is a specific, rare clearance tier
#    that in practice requires an existing government sponsor and a
#    multi-year background investigation -- a posting naming it at all
#    signals a role this candidate (no defense/federal/IC history) is not
#    realistically competitive for. This is also the exact pattern that
#    catches the live regression case above: its own wording is "may
#    require... at the TS/SCI w/CI Poly level... ability to obtain and
#    maintain... An active TS/SCI w/CI Poly is preferred" -- soft framing
#    throughout, so a pattern keyed on "required" language alone would NOT
#    have caught it. Only the bare TS/SCI token reliably does.
#
# 2. CLEARANCE_REQUIRED: generic "clearance" / "Top Secret clearance" /
#    "Secret clearance" mentions are NOT disqualifying on their own -- those
#    tiers are commonly described as "must be able to obtain," which many
#    employers extend to candidates with no prior clearance. Only an
#    EXPLICIT statement that an active/current clearance is already
#    required (not merely obtainable) is disqualifying at this tier.
#    "Ability to obtain a clearance" phrasing is deliberately left alone --
#    see the module-level ambiguity note below.
#
# AMBIGUITY NOT RESOLVED HERE (reported, not silently decided): whether a
# BARE "Top Secret clearance" / "Secret clearance" mention (no "required"
# qualifier) should be treated the same as bare TS/SCI is genuinely
# ambiguous from the existing prompt policy, which only says "roles
# requiring a security clearance... are OUT OF SCOPE" without addressing
# obtain-vs-hold framing. Secret-tier clearance is common enough, and often
# phrased as "ability to obtain," that auto-rejecting every bare mention
# risked meaningfully diverging from that "ability to obtain is a different
# case" carve-out -- so this gate stays conservative for Secret/Top Secret
# and requires an explicit hard-requirement qualifier for those two tiers.
_TS_SCI_PATTERN = re.compile(r"\bTS[/\s-]?SCI\b", re.IGNORECASE)

_CLEARANCE_REQUIRED_PATTERN = re.compile(
    r"(?:top\s+secret|secret)\s+clearance\s+(?:is\s+)?required"
    r"|(?:active|current)\s+security\s+clearance\s+(?:is\s+)?required"
    r"|must\s+(?:currently\s+)?(?:possess|hold|have)\s+(?:an?\s+)?(?:active|current)\s+(?:security\s+)?clearance"
    r"|requires?\s+(?:an?\s+)?(?:active|current)\s+(?:security\s+)?clearance",
    re.IGNORECASE,
)


# Description-level non-US patterns. Scans the full description (capped at
# 6000 chars by the caller). Patterns intentionally narrow — must explicitly
# RESTRICT to a non-US country, not merely mention global offices.
# Tightened 2026-04-30 after Twilio UK/Canada slipped through the 800-char head.
_INELIGIBLE_DESC_PATTERNS = re.compile(
    r"Remote\s*[\(\-—–]\s*(EMEA|Europe|EU|UK|United\s+Kingdom|Germany|India|Canada|Ireland|Netherlands|Brazil|Mexico|Argentina|Colombia|Australia|New\s+Zealand)"
    r"|EMEA\s*(only|region|remote|based)"
    r"|(Europe|European)\s*(only|Time\s*Zone|timezone|based|remote)"
    # Belt-and-suspenders: catches "based in (the) UK", "based in Europe", etc.
    r"|based\s+in\s+(the\s+)?(Europe|EU|UK|United\s+Kingdom|Germany|India|Netherlands|Canada|Ireland|France|Spain|Italy|Brazil|Mexico|Australia|New\s+Zealand|Singapore|Japan|Israel|South\s+Africa|Portugal|Poland|Romania)"
    r"|will\s+be\s+remote\s+and\s+based\s+in\s+(the\s+)?(UK|United\s+Kingdom|Canada|Ireland|Germany|Europe|EMEA|India)"
    # Canadian-province patterns (Twilio L3 example)
    r"|Remote\s+(in|from|—|-)\s*(Ontario|British\s+Columbia|Alberta|Quebec|Manitoba|Nova\s+Scotia|Saskatchewan)"
    r"|Ontario,\s*British\s+Columbia"
    # UK/Canada right-to-work questions on the form (often mirrored in JD)
    r"|right\s+to\s+work\s+in\s+(the\s+)?(UK|United\s+Kingdom|Canada|Ireland|EU|European\s+Union)"
    r"|requires?\s+(the\s+)?right\s+to\s+work\s+in\s+(the\s+)?(UK|United\s+Kingdom|Canada|Ireland|EU)"
    # Timezone restrictions
    r"|CET\s+timezone"
    r"|GMT[+\-]\d+\s+timezone"
    r"|IST\s+timezone",
    re.IGNORECASE,
)

# Window scanned for description-level patterns. Bumped from 800 to 6000 chars
# 2026-04-30 — Twilio buried the UK restriction in a paragraph below the
# requirements list, past the 800-char head.
_DESC_SCAN_CHARS = 6000


def _check_ineligible(job: dict, profile: dict | None = None) -> str | None:
    """Return an ineligibility reason if the job is obviously non-US or the
    candidate categorically can't meet a stated hard requirement, else None.

    Checked before the LLM call to save tokens and ensure consistency.
    Scans title, location field, and the first ``_DESC_SCAN_CHARS`` of the
    description.
    """
    title = job.get("title") or ""
    location = job.get("location") or ""
    desc_head = (job.get("full_description") or "")[:_DESC_SCAN_CHARS]

    if _INELIGIBLE_TITLE_PATTERNS.search(title):
        return f"non-US geography in title: {title}"
    if location and _INELIGIBLE_LOCATION_PATTERNS.search(location):
        return f"non-US location field: {location}"
    m = _INELIGIBLE_DESC_PATTERNS.search(desc_head)
    if m:
        return f"non-US geography in description: {m.group(0)[:80]}"

    # Hard disqualifier, checked before the LLM call so a stale/rescored
    # title can never accumulate enough other points to reach min_score --
    # this candidate has no professional software engineering experience,
    # so a Senior/Staff/Principal/Lead/Architect/Director/Manager/Head/VP/
    # Chief/Distinguished/Fellow title is always disqualifying regardless
    # of keyword overlap elsewhere in the description. See applypilot.
    # eligibility for the single canonical definition (also used by
    # apply/launcher.py's acquisition-time defense-in-depth check).
    seniority_reason = seniority_disqualifier(title)
    if seniority_reason:
        return seniority_reason

    if profile is not None and not _candidate_has_advanced_degree(profile):
        m = _ADVANCED_DEGREE_REQUIRED_PATTERN.search(desc_head)
        if m:
            return f"advanced degree required, candidate has none: {m.group(0)[:80]}"

    m = _TS_SCI_PATTERN.search(desc_head)
    if m:
        return f"security clearance required: {m.group(0)[:80]}"
    m = _CLEARANCE_REQUIRED_PATTERN.search(desc_head)
    if m:
        return f"security clearance required: {m.group(0)[:80]}"

    search_cfg = load_search_config() or {}
    excluded_titles = search_cfg.get("exclude_titles") or []
    configured_title_patterns = search_cfg.get("title_reject_patterns") or []
    if isinstance(search_cfg.get("filters"), dict):
        configured_title_patterns = configured_title_patterns or search_cfg["filters"].get("title_reject_patterns", [])

    # exclude_titles are plain words/phrases (e.g. "VP ", "head", "lead"),
    # not regex -- matching them with a raw re.search let "VP " match
    # inside "MVP", "head" match inside "Overhead Crane Technician", and
    # "lead" match inside "Lead Abatement Technician", wrongly excluding
    # legitimate non-senior titles. Word-boundary match instead (also
    # strips any manual trailing-space hack in the config, e.g. "VP ").
    # title_reject_patterns are genuine user-supplied regex (same
    # convention as cli.py's DEFAULT_TITLE_REJECT_PATTERNS) and are left
    # untouched.
    for raw_term in excluded_titles:
        term = str(raw_term).strip()
        if term and re.search(rf"\b{re.escape(term)}\b", title, re.IGNORECASE):
            return f"title excluded by search configuration: {title}"
    for pattern in configured_title_patterns:
        pattern = str(pattern)
        if pattern.strip() and re.search(pattern, title, re.IGNORECASE):
            return f"title excluded by search configuration: {title}"

    # Ethical exclusions (military/weapons/surveillance/policing/defense contractors,
    # see searches.yaml exclude_description_keywords). Checked against title + full
    # description head, not just the location field, since these are about the
    # employer's line of business, not geography.
    ethical_keywords = [
        str(k).strip().lower() for k in (search_cfg.get("exclude_description_keywords") or []) if str(k).strip()
    ]
    if ethical_keywords:
        haystack = f"{title}\n{desc_head}".lower()
        for kw in ethical_keywords:
            if kw in haystack:
                return f"ethical exclusion keyword matched: {kw!r}"

    return None


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int | None, "keywords": str, "reasoning": str, "eligibility": str}
        eligibility is one of: 'eligible' | 'non_us_only'.
        Older models that omit ELIGIBILITY default to 'eligible' (preserves
        prior behavior; new prompt requires the line so absence is rare).

        score is None when no "SCORE: <digits>" line could be found at all
        (empty/blank response, a refusal, a truncated/non-conforming
        response, ...) -- distinct from a real parsed score, which is always
        clamped to 1-10 (never 0: the prompt's own contract is "SCORE:
        [1-10]"). 2026-08-25 fix: this used to default to 0 on a parse
        failure, indistinguishable from a legitimate low score, so
        score_job()/_flush_score_batch's `score is not None` success check
        treated an unparseable response as a successful fit_score=0. See
        score_job()'s docstring for how the None case is surfaced as an
        error to callers.

    Uses regex search (not line.startswith()) because the claude_cli fallback
    tier answers in markdown -- "**SCORE: 1**" -- which a plain startswith()
    check misses entirely, silently defaulting every field. The \\**\\[?
    padding here absorbs bold markers and/or brackets around each label so
    both plain-text and markdown-formatted responses parse the same.
    """
    score = None
    keywords = ""
    reasoning = response
    eligibility = "eligible"

    elig_match = re.search(r"ELIGIBILITY:\s*\**\[?([a-z_\s]+?)\]?\**\s*(?:\n|$)", response, re.IGNORECASE)
    if elig_match:
        value = re.sub(r"[\[\]\s]+", "_", elig_match.group(1).strip().lower()).strip("_")
        eligibility = "non_us_only" if ("non" in value and "us" in value) else "eligible"

    score_match = re.search(r"SCORE:\s*\**\[?\s*(\d+)", response, re.IGNORECASE)
    if score_match:
        score = max(1, min(10, int(score_match.group(1))))

    kw_match = re.search(r"KEYWORDS:\s*\**\[?(.+?)\]?\**\s*(?:\n|$)", response, re.IGNORECASE)
    if kw_match:
        keywords = kw_match.group(1).strip()

    # DOTALL + greedy: claude_cli often wraps reasoning across several
    # sentences/paragraphs rather than one line, unlike the terser API models.
    reasoning_match = re.search(r"REASONING:\s*\**\[?(.+)", response, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip().rstrip("*").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning, "eligibility": eligibility}


def _build_candidate_summary(profile: dict) -> str:
    """Build a candidate summary string from profile for the scoring prompt.

    Pulls only real, documented facts (education, certifications, prior job
    titles, project-level skills) -- no inferred seniority or stack.
    """
    exp = profile.get("experience", {})
    education = profile.get("education") or []
    certs = profile.get("certifications") or []
    skills = profile.get("skills_inventory") or []
    experience_inventory = profile.get("experience_inventory") or []

    parts: list[str] = []

    degree_bits = [
        f"{item.get('official_degree')} ({item.get('institution', '')})"
        for item in education
        if isinstance(item, dict) and item.get("official_degree")
    ]
    if degree_bits:
        parts.append(f"Education: {'; '.join(degree_bits)}. No CS/engineering degree.")

    cert_names = [c.get("name") for c in certs if isinstance(c, dict) and c.get("name")]
    if cert_names:
        parts.append(f"Certifications: {', '.join(cert_names)}.")

    work_roles = [
        item.get("role_title") or item.get("role_type")
        for item in experience_inventory
        if isinstance(item, dict) and (item.get("role_title") or item.get("role_type"))
    ]
    if work_roles:
        parts.append(f"Work history (non-software roles): {', '.join(work_roles)}.")

    demonstrated = [
        s.get("name")
        for s in skills
        if isinstance(s, dict)
        and s.get("resume_allowed")
        and str(s.get("proficiency", "")).lower() not in ("", "learning")
    ]
    if demonstrated:
        parts.append(f"Demonstrated project-level skills: {', '.join(demonstrated)}.")

    current_title = exp.get("current_job_title") or "none"
    years = exp.get("years_of_experience_total") or "0 / not documented"
    parts.append(
        f"Current job title: {current_title}. Documented years of professional software engineering experience: {years}."
    )

    parts.append(
        "CRITICAL: this candidate has NO professional software engineering employment history. "
        "All programming experience is self-taught, personal-project-level Python only. Do not "
        "infer prior IC engineering experience, a CS background, or industry seniority merely "
        "because the candidate is applying to technical roles."
    )
    return " ".join(parts)


def _build_location_context(profile: dict) -> str:
    """Build a location/relocation summary string for the scoring prompt."""
    loc = (profile.get("application_profile") or {}).get("location") or {}
    bits: list[str] = []
    city = loc.get("current_city") or ""
    state = loc.get("current_state") or ""
    if city or state:
        bits.append(f"based in {', '.join(x for x in (city, state) if x)}")
    arrangement = loc.get("preferred_work_arrangement")
    if arrangement:
        bits.append(f"prefers {arrangement} work")
    if loc.get("willing_to_relocate") and loc.get("relocation_preference"):
        bits.append(f"open to relocating within {loc['relocation_preference']}")
    if loc.get("willing_onsite") or loc.get("willing_hybrid"):
        bits.append("open to onsite/hybrid within that area or a reasonable commute")
    return "; ".join(bits) or "US-based; location preferences not specified in profile"


def score_job(resume_text: str, job: dict, profile: dict | None = None) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    if profile is None:
        profile = load_profile()

    # Rule-based pre-filter: catch obvious non-US ineligible jobs before LLM call.
    # Tags eligibility=non_us_only so downstream stages skip the job; the score
    # is still recorded so audit views can see the original LLM-style severity.
    ineligible_reason = _check_ineligible(job, profile)
    if ineligible_reason:
        log.info("Pre-filter INELIGIBLE: %s — %s", (job.get("title") or "?")[:60], ineligible_reason)
        # eligibility="non_us_only" is reused here as the generic "force archive,
        # never tailor/apply" signal for ALL pre-filter rejections (geography,
        # title-pattern, ethical exclusion) -- see _flush_score_batch's archive routing.
        return {
            "score": 2,
            "keywords": "",
            "reasoning": f"Ineligible: {ineligible_reason}.",
            "eligibility": "non_us_only",
        }

    try:
        candidate_summary = _build_candidate_summary(profile)
        location_context = _build_location_context(profile)
        score_prompt = SCORE_PROMPT_TEMPLATE.format(
            candidate_summary=candidate_summary,
            location_context=location_context,
        )

        job_text = (
            f"TITLE: {job['title']}\n"
            f"COMPANY: {job['site']}\n"
            f"LOCATION: {job.get('location', 'N/A')}\n\n"
            f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
        )

        messages = [
            {"role": "system", "content": score_prompt},
            {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
        ]

        client = get_stage_client("score", quality=False)
        response = client.chat(
            messages,
            max_tokens=get_token_limit("score", 8192),
            temperature=0.2,
        )
        result = _parse_score_response(response)
        if result["score"] is None:
            # 2026-08-25 fix: _parse_score_response returns score=None when
            # no "SCORE: <digits>" line could be found (empty/blank/refusal/
            # truncated/non-conforming response) -- this is a parse FAILURE,
            # not a low-confidence low score, and must be routed through the
            # same failure path as an exception below (_flush_score_batch's
            # `if r["score"] is not None` branch reads r["error"], which only
            # the except branch used to populate).
            result["error"] = (
                f"LLM error: response did not contain a parseable SCORE line (first 200 chars: {response[:200]!r})"
            )
        return result
    except Exception as exc:
        log.exception("LLM error scoring job '%s'", (job or {}).get("title") or "?")
        return {"score": None, "keywords": "", "reasoning": "", "eligibility": None, "error": f"LLM error: {exc}"}


MAX_SCORE_RETRIES = 5


def _score_backoff_minutes(retry_count: int) -> int:
    """Exponential backoff for scoring retries: 5, 20, 80, ~5h, ~21h."""
    return min(5 * (4**retry_count), 24 * 60)


def _flush_score_batch(conn, batch: list[dict], now: str) -> None:
    """Write a batch of scoring results to the DB.

    On success (score is not None): writes fit_score, clears score_error.
    On failure (score is None): leaves fit_score NULL, writes score_error + backoff.
    Jobs that have already hit MAX_SCORE_RETRIES stay unscored indefinitely (manual rescue needed).
    """
    from applypilot.config import DEFAULTS as _cfg_DEFAULTS
    from applypilot.database import transition_state

    _min_score = _cfg_DEFAULTS["min_score"]

    for r in batch:
        # 2026-08-25 fix: a job that reached this batch despite already being
        # archived (e.g. via `--rescore`, which selects on full_description
        # alone with no state filter, or a race with a concurrent archival)
        # must never have its fit_score/score_reasoning/eligibility
        # overwritten -- see "pending_score"'s 2026-08-25 comment in
        # database.py for the paired query-side fix and the audit finding
        # (2,080 archived rows were wrongly reachable before that fix, all of
        # which would have had their archived fit_score/reasoning silently
        # clobbered here). Checking transition_state's return value alone
        # isn't enough: the non_us_only branch below calls it with
        # force=True (bypasses validation), and an archived->archived
        # self-transition is ALSO trivially "allowed" (to_state ==
        # from_state) -- neither would reject an already-archived row. Look
        # up the real current state first and skip entirely when it's the
        # state machine's one true terminal state (VALID_TRANSITIONS
        # ["archived"] == frozenset()).
        current_row = conn.execute("SELECT state FROM jobs WHERE url = ?", (r["url"],)).fetchone()
        if current_row and current_row[0] == "archived":
            log.warning(
                "Skipping score write for already-archived job (preserving "
                "its archived fit_score/reasoning for audit): %s",
                r["url"][:80],
            )
            continue

        if r["score"] is not None:
            eligibility = r.get("eligibility") or "eligible"
            conn.execute(
                "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ?, "
                "eligibility = ?, "
                "score_error = NULL, score_attempts = 0, score_next_retry_at = NULL "
                "WHERE url = ?",
                (r["score"], f"{r['keywords']}\n{r['reasoning']}", now, eligibility, r["url"]),
            )
            # Eligibility-driven state transition. Non-US roles go straight to
            # `archived` (terminal) so tailor/cover/apply never pick them up,
            # bypassing the scored→tailored→ready_to_apply chain entirely.
            if eligibility == "non_us_only":
                transition_state(
                    conn,
                    r["url"],
                    "archived",
                    reason="non_us_only employer/role",
                    metadata={"score": r["score"], "eligibility": eligibility},
                    force=True,
                )
            else:
                to_state = "scored" if r["score"] >= _min_score else "low_score"
                transition_state(
                    conn,
                    r["url"],
                    to_state,
                    reason=f"scored {r['score']}/10",
                    metadata={"score": r["score"], "eligibility": eligibility},
                )
        else:
            # LLM failure — keep fit_score NULL so it stays in pending_score
            row = conn.execute("SELECT COALESCE(score_attempts, 0) FROM jobs WHERE url = ?", (r["url"],)).fetchone()
            retry_count = row[0] if row else 0
            if retry_count >= MAX_SCORE_RETRIES:
                # Give up — write score_error but don't schedule another retry
                conn.execute(
                    "UPDATE jobs SET score_error = ?, score_attempts = ?, "
                    "score_next_retry_at = NULL, scored_at = ? WHERE url = ?",
                    (r["error"], retry_count + 1, now, r["url"]),
                )
                transition_state(
                    conn,
                    r["url"],
                    "score_failed",
                    reason=f"LLM failed after {retry_count + 1} attempts",
                    metadata={"error": r["error"]},
                )
            else:
                delay = _score_backoff_minutes(retry_count)
                next_retry = (datetime.now(UTC) + timedelta(minutes=delay)).isoformat()
                conn.execute(
                    "UPDATE jobs SET score_error = ?, score_attempts = ?, "
                    "score_next_retry_at = ?, scored_at = ? WHERE url = ?",
                    (r["error"], retry_count + 1, next_retry, now, r["url"]),
                )
                log.info(
                    "  score retry %d/%d scheduled in %d min for %s",
                    retry_count + 1,
                    MAX_SCORE_RETRIES,
                    delay,
                    r["url"][:60],
                )


def run_scoring(limit: int = 0, rescore: bool = False, workers: int = 1, max_age_days: int | None = None) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).
        workers: Parallel LLM threads (default 1 = sequential).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list,
         "job_urls": list[str]} -- job_urls is every URL this run selected
        for scoring (whether or not it succeeded), in selection order. A
        sequential pipeline run can hand this to run_tailoring(job_ids=...)
        so the following stage examines exactly this batch instead of
        independently re-querying and picking up unrelated already-eligible
        jobs from earlier runs.
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        # Note: get_jobs_by_stage now applies a 14-day discovered_at filter by
        # default (config.DEFAULTS["max_job_age_days"]). Pass max_age_days=0
        # to disable.
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", max_age_days=max_age_days, limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": [], "job_urls": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    # Captured now (batch is fixed from here on) so callers can carry this
    # exact batch's identity into a following stage -- see the job_urls
    # note in the docstring above.
    job_urls = [j["url"] for j in jobs if j.get("url")]

    log.info("Scoring %d jobs (workers=%d)...", len(jobs), workers)
    t0 = time.time()
    completed = 0
    errors = 0
    batch_size = 25  # Commit every N jobs so downstream stages see results sooner
    batch: list[dict] = []

    def _score_one(job: dict) -> dict:
        try:
            result = score_job(resume_text, job)
            result["url"] = job["url"]
        except Exception as exc:
            log.exception("Unexpected error scoring '%s'", (job or {}).get("title") or "?")
            result = {
                "score": None,
                "keywords": "",
                "reasoning": "",
                "error": f"Unexpected: {exc}",
                "url": (job or {}).get("url", ""),
            }
        return result

    def _flush_and_log(batch: list[dict], completed: int) -> list[dict]:
        now = datetime.now(UTC).isoformat()
        try:
            write_with_retry(conn, _flush_score_batch, conn, batch, now)
        except Exception:
            log.exception("Batch flush failed (batch of %d)", len(batch))
        log.info("Committed batch of %d scores to DB (%d/%d total)", len(batch), completed, len(jobs))
        return []

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                result = future.result()
                completed += 1
                if result["score"] is None:
                    errors += 1
                batch.append(result)
                log.info(
                    "[%d/%d] score=%s  %s",
                    completed,
                    len(jobs),
                    result["score"] if result["score"] is not None else "ERR",
                    (job.get("title") or "")[:60],
                )
                if len(batch) >= batch_size:
                    batch = _flush_and_log(batch, completed)
    else:
        for job in jobs:
            result = _score_one(job)
            completed += 1
            if result["score"] is None:
                errors += 1
            batch.append(result)
            log.info(
                "[%d/%d] score=%s  %s",
                completed,
                len(jobs),
                result["score"] if result["score"] is not None else "ERR",
                (job.get("title") or "")[:60],
            )
            if len(batch) >= batch_size:
                batch = _flush_and_log(batch, completed)

    # Flush remaining
    if batch:
        now = datetime.now(UTC).isoformat()
        try:
            write_with_retry(conn, _flush_score_batch, conn, batch, now)
        except Exception:
            log.exception("Final batch flush failed (batch of %d)", len(batch))

    elapsed = time.time() - t0
    log.info("Done: %d scored in %.1fs (%.1f jobs/sec)", completed, elapsed, completed / elapsed if elapsed > 0 else 0)

    # Score distribution
    try:
        dist = conn.execute("""
            SELECT fit_score, COUNT(*) FROM jobs
            WHERE fit_score IS NOT NULL
            GROUP BY fit_score ORDER BY fit_score DESC
        """).fetchall()
        distribution = [(row[0], row[1]) for row in dist]
    except Exception:
        log.exception("Distribution query failed")
        distribution = []

    return {
        "scored": completed,
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
        "job_urls": job_urls,
    }
