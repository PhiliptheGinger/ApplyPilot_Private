"""Resume tailoring: LLM-powered ATS-optimized resume generation per job.

THIS IS THE HEAVIEST REFACTOR. Every piece of personal data -- name, email, phone,
skills, companies, projects, school -- is loaded at runtime from the user's profile.
Zero hardcoded personal information.

The LLM returns structured JSON, code assembles the final text. Header (name, contact)
is always code-injected, never LLM-generated. Each retry starts a fresh conversation
to avoid apologetic spirals.
"""

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from applypilot.config import TAILORED_DIR, load_profile
from applypilot.llm import get_stage_client, get_token_limit, is_local_configured
from applypilot.scoring.fact_approval import (
    extract_facts_from_resume_json,
    is_auto_approvable,
    record_approved_facts,
)
from applypilot.scoring.resume_router import load_resume_text_for_job
from applypilot.scoring.validator import (
    sanitize_text,
    validate_json_fields,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up

STANDUP_INCLUDE = "INCLUDE"
STANDUP_OPTIONAL = "OPTIONAL"
STANDUP_EXCLUDE = "EXCLUDE"


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify_standup_relevance(job: dict) -> str:
    """Deterministically classify stand-up relevance from a job posting.

    Returns exactly one of:
      - INCLUDE
      - OPTIONAL
      - EXCLUDE

    The classifier is heuristic and non-LLM by design. It prioritizes the
    job description's responsibilities and uses title as a secondary signal.
    """
    title = (job.get("title") or "").lower()
    desc = (job.get("full_description") or "").lower()

    if not desc.strip():
        return STANDUP_EXCLUDE

    text = f"{title}\n{desc}"

    # Positive signals (deduplicated by category)
    public_speaking_core = _matches_any(text, (
        r"\bpublic speaking\b",
        r"\bspeaking engagements?\b",
        r"\bpresent (to|in front of) (customers|clients|users|audiences?)\b",
    ))
    customer_facing_substantial = _matches_any(text, (
        r"\bcustomer[- ]facing\b",
        r"\bclient[- ]facing\b",
        r"\bend[- ]user (support|interaction|facing)\b",
        r"\bhandle (customer|client|user) (issues|questions|inquiries|escalations)\b",
    ))
    presentations_training = _matches_any(text, (
        r"\bpresentations?\b",
        r"\bdemonstrations?\b",
        r"\bproduct demos?\b",
        r"\btrain(?:ing|er)?\b",
        r"\bworkshops?\b",
    ))
    sales_recruiting_biz = _matches_any(text, (
        r"\bsales\b",
        r"\brecruit(?:er|ing)\b",
        r"\bbusiness development\b",
        r"\baccount management\b",
        r"\baccount executive\b",
    ))
    communication_substantial = _matches_any(text, (
        r"\blead (customer|client|stakeholder) communications?\b",
        r"\bprimary point of contact\b",
        r"\bregularly communicat(?:e|ing) with (customers|clients|stakeholders|executives)\b",
        r"\baudience[- ]facing\b",
    ))
    explain_technical = _matches_any(text, (
        r"\bexplain (technical|complex) (concepts?|issues?|systems?) to (non[- ]technical|customers|clients|users|stakeholders)\b",
        r"\btranslate technical.*non[- ]technical\b",
    ))
    stakeholder_mgmt = _matches_any(text, (
        r"\bstakeholder management\b",
        r"\bmanage stakeholders?\b",
        r"\bstakeholder communication\b",
    ))
    community_media = _matches_any(text, (
        r"\bcommunity\b",
        r"\boutreach\b",
        r"\bcontent\b",
        r"\bmedia\b",
        r"\bcommunications?\b",
        r"\bmarketing\b",
        r"\beducation\b",
        r"\binterview(?:ing)?\b",
    ))
    communication_centric_title = _matches_any(title, (
        r"\bhelp desk\b",
        r"\btechnical support\b",
        r"\bcustomer support\b",
        r"\bsales engineer\b",
        r"\brecruit(?:er|ing)\b",
        r"\btrainer\b",
        r"\bcommunications?\b",
        r"\bmarketing\b",
        r"\bmedia\b",
    ))
    generic_comm = _matches_any(text, (
        r"\bexcellent communication skills\b",
        r"\bstrong communication skills\b",
        r"\bwritten and verbal communication\b",
        r"\binterpersonal skills\b",
        r"\bteam player\b",
    ))

    # Negative / technical-density signals
    technical_ic_title = _matches_any(title, (
        r"\bsoftware engineer\b",
        r"\bbackend (engineer|developer)\b",
        r"\bdevops engineer\b",
        r"\bsystems? administrat(?:or|ion)\b",
        r"\bnetwork engineer\b",
        r"\bcyber(?:security)? engineer\b",
        r"\bdata engineer\b",
        r"\bplatform engineer\b",
        r"\bsite reliability engineer\b",
    ))
    primarily_technical_work = _matches_any(text, (
        r"\bdesign and implement (software|services|systems)\b",
        r"\bbuild (microservices|distributed systems|infrastructure)\b",
        r"\binfrastructure automation\b",
        r"\bincident response\b",
        r"\bthreat detection\b",
        r"\bnetwork architecture\b",
        r"\bkubernetes\b",
        r"\bterraform\b",
        r"\bci/cd\b",
    ))

    strong_comm_indicator = any((
        public_speaking_core,
        customer_facing_substantial,
        presentations_training,
        sales_recruiting_biz,
        communication_substantial,
        explain_technical,
        stakeholder_mgmt,
        community_media,
    ))

    score = 0
    if public_speaking_core:
        score += 5
    if customer_facing_substantial:
        score += 5
    if presentations_training:
        score += 4
    if sales_recruiting_biz:
        score += 4
    if communication_substantial:
        score += 3
    if explain_technical:
        score += 3
    if stakeholder_mgmt:
        score += 2
    if community_media:
        score += 2
    if communication_centric_title:
        score += 3
    if generic_comm:
        score += 1

    if technical_ic_title and not strong_comm_indicator:
        score -= 5
    if primarily_technical_work and not strong_comm_indicator:
        score -= 3
    if generic_comm and not strong_comm_indicator:
        score -= 4

    if score >= 7:
        return STANDUP_INCLUDE
    if score >= 3:
        return STANDUP_OPTIONAL
    return STANDUP_EXCLUDE


# ── Prompt Builders (profile-driven) ──────────────────────────────────────

def _build_canonical_inventory_block(profile: dict) -> str:
    """Render factual profile inventory for relevance-based selection."""
    lines: list[str] = []
    education = profile.get("education", [])
    for item in education:
        if isinstance(item, dict):
            lines.append(
                f"EDUCATION: {item.get('official_degree', '')}; "
                f"{item.get('institution', '')}; "
                f"{item.get('start_year', '')}-{item.get('end_year', '')}"
            )

    for key, label in (
        ("experience_inventory", "PROFESSIONAL EXPERIENCE"),
        ("historical_experience_inventory", "SELECTABLE HISTORICAL EXPERIENCE"),
        ("qualifications", "SELECTABLE QUALIFICATIONS"),
        ("project_inventory", "SELECTABLE PROJECTS"),
    ):
        entries = profile.get(key, [])
        if not entries:
            continue
        lines.append(f"{label}:")
        for item in entries:
            if not isinstance(item, dict) or item.get("private") or item.get("resume_allowed") is False:
                continue
            name = item.get("name", "")
            status = item.get("status", "")
            categories = ", ".join(item.get("relevance_categories", []))
            lines.append(f"- {name} [{status}] ({categories})")
            evidence = item.get("factual_concepts") or item.get("evidence") or item.get("responsibilities")
            if evidence:
                lines.append(f"  Evidence: {'; '.join(str(x) for x in evidence)}")

    lines.append("SKILL EVIDENCE:")
    for item in profile.get("skills_inventory", []):
        if isinstance(item, dict):
            lines.append(
                f"- {item.get('name', '')}: evidence={item.get('evidence_level', '')}; "
                f"proficiency={item.get('proficiency', '')}; "
                f"resume_allowed={item.get('resume_allowed', True)}"
            )
    return "\n".join(lines) or "No structured inventory supplied; use only original resume facts."

def _build_tailor_prompt(profile: dict, standup_decision: str = STANDUP_EXCLUDE) -> str:
    """Build the resume tailoring system prompt from the user's profile.

    All skills boundaries, preserved entities, and formatting rules are
    derived from the profile -- nothing is hardcoded.
    """
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Format skills boundary for the prompt
    skills_lines = []
    for category, items in boundary.items():
        if isinstance(items, list) and items:
            label = category.replace("_", " ").title()
            if "learning" in category.lower() or "exposure" in category.lower():
                label += " (LEARNING/EXPOSURE, NOT EXPERTISE)"
            skills_lines.append(f"{label}: {', '.join(items)}")
    skills_block = "\n".join(skills_lines)

    # Preserved entities
    companies = resume_facts.get("preserved_companies", [])
    projects = resume_facts.get("preserved_projects", [])
    school = resume_facts.get("preserved_school", "")
    real_metrics = resume_facts.get("real_metrics", [])

    companies_str = ", ".join(companies) if companies else "N/A"
    _projects_str = ", ".join(projects) if projects else "N/A"
    metrics_str = ", ".join(real_metrics) if real_metrics else "N/A"

    education = profile.get("experience", {})
    education_level = education.get("education_level", "")
    inventory_block = _build_canonical_inventory_block(profile)
    official_education_text = ""
    if profile.get("education") and isinstance(profile["education"][0], dict):
        item = profile["education"][0]
        official_education_text = (
            f"Official degree fact: {item.get('official_degree', '')}; "
            f"institution: {item.get('institution', '')}; "
            f"dates: {item.get('start_year', '')}-{item.get('end_year', '')}."
        )

    if standup_decision not in {STANDUP_INCLUDE, STANDUP_OPTIONAL, STANDUP_EXCLUDE}:
        standup_decision = STANDUP_EXCLUDE

    standup_block = {
        STANDUP_INCLUDE: (
            "STAND-UP EXPERIENCE DECISION: INCLUDE\n\n"
            "The candidate's stand-up comedy experience is relevant to this job. "
            "Include it when appropriate as professional experience. Frame it in terms "
            "of public speaking, audience awareness, verbal communication, presentation, "
            "improvisation, or adapting communication to different audiences, but do not "
            "invent accomplishments or metrics."
        ),
        STANDUP_OPTIONAL: (
            "STAND-UP EXPERIENCE DECISION: OPTIONAL\n\n"
            "Stand-up comedy may be included only if space permits and it adds meaningful "
            "value for this particular job. It must not displace more directly relevant "
            "technical experience. Do not force it into the resume."
        ),
        STANDUP_EXCLUDE: (
            "STAND-UP EXPERIENCE DECISION: EXCLUDE\n\n"
            "Do not include, mention, or reference the candidate's stand-up comedy "
            "experience anywhere in the tailored resume."
        ),
    }[standup_decision]

    return f"""You are a senior technical recruiter rewriting a resume to get this person an interview.

Take the base resume and job description. Return a tailored resume as a JSON object.

## CANONICAL PROFILE SOURCE OF TRUTH:
{inventory_block}

The profile inventory is comprehensive and factual. Select only evidence that is
relevant and defensible for this job. Omission is preferable to irrelevant
inclusion. Never invent qualifications, responsibilities, metrics, project
outcomes, or proficiency. Never include private projects.

## RECRUITER SCAN (6 seconds):
1. Title -- matches what they're hiring?
2. Summary -- 2 sentences proving you've done this work
3. First 3 bullets of most recent role -- verbs and outcomes match?
4. Skills -- must-haves visible immediately?

## SKILLS BOUNDARY (real skills only):
{skills_block}

You MAY add 2-3 closely related tools (Kubernetes if Docker, Terraform if AWS, Redis if PostgreSQL). No unrelated languages/frameworks.

## TAILORING RULES:

{standup_block}

TITLE: Use the job title verbatim from the posting (Jobscan: 10.6x interview-rate lift). Only strip internal-req prefixes like "JREQ197053 -" and trailing team tags like "(Payments Team)". Keep the full role name including specialty — "Staff Software Engineer, AI Platform" stays as-is; "Senior Backend Engineer - Python" stays as-is.

SUMMARY: Rewrite from scratch. Lead with the 1-2 skills that matter most for THIS role. Sound like someone who's done this job.

SKILLS: Reorder each category so the job's must-haves appear first.
Do not convert learning/exposure into expertise. Only list those items when the
profile evidence supports a careful learning/exposure description.

Reframe EVERY bullet for this role. Same real work, different angle. Every bullet must be reworded. Never copy verbatim.

PROJECTS: Reorder by relevance. Drop irrelevant projects entirely.
Never include ApplyPilot or any project marked private/resume_allowed=false.
Sunburn is unfinished/experimental and must not be described as deployed or successful.

BULLETS: Strong verb + what you built + quantified impact. Vary verbs (Built, Designed, Implemented, Reduced, Automated, Deployed, Operated, Optimized). Most relevant first. Max 4 per section.

EDUCATION: Copy every school from the ORIGINAL RESUME's education section — one object per school. Preserve the institution name, degree, field of study, and years EXACTLY as written in the original. Never invent a degree, field, or date. If a school lists no degree, omit the "degree" key; if it lists no years, omit the "dates" key. Do not reorder, summarize, or merge schools — education is factual, not tailored. Do NOT output the education level ("{education_level}") as if it were a school.
{official_education_text}

## VOICE:
- Write like a real engineer. Short, direct.
- GOOD: "Automated financial reporting with Python + API integrations, cut processing time from 10 hours to 2"
- BAD: "Leveraged cutting-edge AI technologies to drive transformative operational efficiencies"
- NEVER use: passionate, dedicated, leveraging, spearheaded, robust, cutting-edge, proven track record, strong track record, eager, stakeholders, synergy, seamless, streamlined, end-to-end, detail-oriented, results-driven, I am confident, I believe, I am excited
- No em dashes. Use commas, periods, or hyphens.

## HARD RULES:
- Do NOT invent work, companies, degrees, or certifications
- Do NOT change real numbers ({metrics_str})
- Preserved companies: {companies_str} -- names stay as-is
- Preserved school: {school}
- Must fit 1 page.

## OUTPUT: Return ONLY valid JSON. No markdown fences. No commentary. No "here is" preamble.

{{"title":"Role Title","summary":"2-3 tailored sentences.","skills":{{"Languages":"...","Frameworks":"...","DevOps & Infra":"...","Databases":"...","Tools":"..."}},"experience":[{{"header":"Title at Company","subtitle":"Tech | Dates","bullets":["bullet 1","bullet 2","bullet 3","bullet 4"]}}],"projects":[{{"header":"Project Name - Description","subtitle":"Tech | Dates","bullets":["bullet 1","bullet 2"]}}],"education":[{{"institution":"School Name","degree":"Degree, Field of study (omit key if not in original)","dates":"YYYY - YYYY (omit key if not in original)"}}]}}"""


def _build_judge_prompt(profile: dict) -> str:
    """Build the LLM judge prompt from the user's profile."""
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Flatten allowed skills for the judge
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "N/A"

    real_metrics = resume_facts.get("real_metrics", [])
    metrics_str = ", ".join(real_metrics) if real_metrics else "N/A"
    private_projects = resume_facts.get("private_projects", [])
    unfinished_projects = resume_facts.get("unfinished_projects", [])

    return f"""You are a resume quality judge. A tailoring engine rewrote a resume to target a specific job. Your job is to catch LIES, not style changes.

You must answer with EXACTLY this format:
VERDICT: PASS or FAIL
ISSUES: (list any problems, or "none")

## CONTEXT -- what the tailoring engine was instructed to do (all of this is ALLOWED):
- Change the title to match the target role
- Rewrite the summary from scratch for the target job
- Reorder bullets and projects to put the most relevant first
- Reframe bullets to use the job's language
- Drop low-relevance bullets and replace with more relevant ones from other sections
- Reorder the skills section to put job-relevant skills first
- Change tone and wording extensively

## WHAT IS FABRICATION (FAIL for these):
1. Adding tools, languages, or frameworks to TECHNICAL SKILLS that aren't in the original. The allowed skills are ONLY: {skills_str}
2. Inventing NEW metrics or numbers not in the original. The real metrics are: {metrics_str}
3. Inventing work that has no basis in any original bullet (completely new achievements).
4. Adding companies, roles, or degrees that don't exist.
5. Changing real numbers (inflating 80% to 95%, 500 nodes to 1000 nodes).
6. Including private projects ({', '.join(private_projects) or 'none'}).
7. Describing unfinished projects ({', '.join(unfinished_projects) or 'none'}) as deployed or successful.

## WHAT IS NOT FABRICATION (do NOT fail for these):
- Rewording any bullet, even heavily, as long as the underlying work is real
- Combining two original bullets into one
- Splitting one original bullet into two
- Describing the same work with different emphasis
- Dropping bullets entirely
- Reordering anything
- Changing the title or summary completely

## TOLERANCE RULE:
The goal is to get interviews, not to be a perfect fact-checker. Allow up to 3 minor stretches per resume:
- Adding a closely related tool the candidate could realistically know is a MINOR STRETCH, not fabrication.
- Reframing a metric with slightly different wording is a MINOR STRETCH.
- Adding any LEARNABLE skill given their existing stack is a MINOR STRETCH.
- Only FAIL if there are MAJOR lies: completely invented projects, fake companies, fake degrees, wildly inflated numbers, or skills from a completely different domain.

Be strict about major lies. Be lenient about minor stretches and learnable skills. Do not fail for style, tone, or restructuring."""


# ── JSON Extraction ───────────────────────────────────────────────────────

def extract_json(raw: str) -> dict:
    """Robustly extract JSON from LLM response (handles fences, preamble).

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON found.
    """
    raw = raw.strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Markdown fences
    if "```" in raw:
        for part in raw.split("```")[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # Find outermost { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in LLM response")


# ── Resume Assembly (profile-driven header) ──────────────────────────────

def assemble_resume_text(data: dict, profile: dict) -> str:
    """Convert JSON resume data to formatted plain text.

    Header (name, location, contact) is ALWAYS code-injected from the profile,
    never LLM-generated. All text fields are sanitized.

    Args:
        data: Parsed JSON resume from the LLM.
        profile: User profile dict from load_profile().

    Returns:
        Formatted resume text.
    """
    personal = profile.get("personal", {})
    lines: list[str] = []

    # Header -- always code-injected from profile
    lines.append(personal.get("full_name", ""))
    lines.append(sanitize_text(data.get("title", "Software Engineer")))

    # Location from search config or profile -- leave blank if not available
    # The location line is optional; the original used a hardcoded city.
    # We omit it here; the LLM prompt can include it if the user sets it.

    # Contact line
    contact_parts: list[str] = []
    if personal.get("email"):
        contact_parts.append(personal["email"])
    if personal.get("phone"):
        contact_parts.append(personal["phone"])
    if personal.get("github_url"):
        contact_parts.append(personal["github_url"])
    if personal.get("linkedin_url"):
        contact_parts.append(personal["linkedin_url"])
    if contact_parts:
        lines.append(" | ".join(contact_parts))
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append(sanitize_text(data["summary"]))
    lines.append("")

    # Technical Skills
    lines.append("TECHNICAL SKILLS")
    if isinstance(data["skills"], dict):
        for cat, val in data["skills"].items():
            lines.append(f"{cat}: {sanitize_text(str(val))}")
    lines.append("")

    # Experience
    lines.append("EXPERIENCE")
    for entry in data.get("experience", []):
        lines.append(sanitize_text(entry.get("header", "")))
        if entry.get("subtitle"):
            lines.append(sanitize_text(entry["subtitle"]))
        for b in entry.get("bullets", []):
            lines.append(f"- {sanitize_text(b)}")
        lines.append("")

    # Projects
    lines.append("PROJECTS")
    for entry in data.get("projects", []):
        lines.append(sanitize_text(entry.get("header", "")))
        if entry.get("subtitle"):
            lines.append(sanitize_text(entry["subtitle"]))
        for b in entry.get("bullets", []):
            lines.append(f"- {sanitize_text(b)}")
        lines.append("")

    # Education — render structured entries as blank-line-separated 1-3 line
    # blocks (Institution / "Degree, Field" / "YYYY - YYYY"), which the PDF/DOCX
    # renderer's parse_education_entries() turns into rich per-school blocks.
    # Falls back to a single line for the legacy string format.
    lines.append("EDUCATION")
    edu = data.get("education", "")
    if isinstance(edu, list):
        blocks: list[str] = []
        for item in edu:
            if not isinstance(item, dict):
                text = sanitize_text(str(item)).strip()
                if text:
                    blocks.append(text)
                continue
            block: list[str] = []
            institution = sanitize_text(str(item.get("institution", ""))).strip()
            if institution:
                block.append(institution)
            # Degree + field on one line ("Degree, Field" or just one of them).
            degree_bits = [
                sanitize_text(str(item.get(k, ""))).strip()
                for k in ("degree", "area", "field", "studyType")
            ]
            degree_line = ", ".join(b for b in degree_bits if b)
            if degree_line:
                block.append(degree_line)
            dates = sanitize_text(str(item.get("dates", ""))).strip()
            if dates:
                block.append(dates)
            if block:
                blocks.append("\n".join(block))
        # Blank line between entries so the parser treats each as its own block.
        lines.append("\n\n".join(blocks))
    else:
        lines.append(sanitize_text(str(edu)))

    return "\n".join(lines)


# ── LLM Judge ────────────────────────────────────────────────────────────

def judge_tailored_resume(
    original_text: str, tailored_text: str, job_title: str, profile: dict
) -> dict:
    """LLM judge layer: catches subtle fabrication that programmatic checks miss.

    Args:
        original_text: Base resume text.
        tailored_text: Tailored resume text.
        job_title: Target job title.
        profile: User profile for building the judge prompt.

    Returns:
        {"passed": bool, "verdict": str, "issues": str, "raw": str}
    """
    judge_prompt = _build_judge_prompt(profile)

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": (
            f"JOB TITLE: {job_title}\n\n"
            f"ORIGINAL RESUME:\n{original_text}\n\n---\n\n"
            f"TAILORED RESUME:\n{tailored_text}\n\n"
            "Judge this tailored resume:"
        )},
    ]

    client = get_stage_client("judge", quality=False)  # judge uses fast model (binary evaluation)
    response = client.chat(
        messages,
        max_tokens=get_token_limit("judge", 4096),
        temperature=0.1,
    )

    passed = "VERDICT: PASS" in response.upper()
    issues = "none"
    if "ISSUES:" in response.upper():
        issues_idx = response.upper().index("ISSUES:")
        issues = response[issues_idx + 7:].strip()

    return {
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "issues": issues,
        "raw": response,
    }


# ── Core Tailoring ───────────────────────────────────────────────────────

def tailor_resume(
    resume_text: str, job: dict, profile: dict, max_retries: int = 2,
    local_plan: str | None = None,
) -> tuple[str, dict]:
    """Generate a tailored resume via JSON output + fresh context on each retry.

    Key design choices:
    - LLM returns structured JSON, code assembles the text (no header leaks)
    - Each retry starts a FRESH conversation (no apologetic spiral)
    - Issues from previous attempts are noted in the system prompt
    - Em dashes and smart quotes are auto-fixed, not rejected
    - First attempt skips LLM judge if programmatic validation passes clean
    - Final attempt accepts validation-passing resumes even if judge disagrees
    - local_plan: optional JSON string from local model (see local_tailor.py);
      prepended to the user message to guide the cloud model's focus.

    Args:
        resume_text: Base resume text.
        job: Job dict with title, site, location, full_description.
        profile: User profile dict.
        max_retries: Maximum retry attempts.
        local_plan: Optional tailoring plan JSON from a local first-pass.

    Returns:
        (tailored_text, report) where report contains validation details.
    """
    company = display_company(job)
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {company or 'unknown (aggregator listing; employer may be named in the description)'}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    standup_decision = classify_standup_relevance(job)
    report: dict = {
        "attempts": 0,
        "validator": None,
        "judge": None,
        "status": "pending",
        "standup_decision": standup_decision,
        "approved_facts": [],
    }
    avoid_notes: list[str] = []
    tailored = ""
    client = get_stage_client("tailor", quality=True)
    # When only the local fallback is available (all cloud models exhausted), use
    # a shorter compact prompt that fits within a 7B-13B model's context budget.
    # This is an emergency DEGRADED MODE, not equivalent to normal cloud
    # finalization -- see local_tailor.LOCAL_FULL_GENERATION_IS_DEGRADED.
    # has_cloud_available() is an LLMClient-specific introspection method, not
    # part of the minimal "has a .chat()" interface callers otherwise depend
    # on (get_stage_client() always returns a real LLMClient in production,
    # but test doubles that only fake .chat() are common and shouldn't need
    # to grow this too) -- so it's queried defensively, defaulting to "cloud
    # is available" (the safe assumption: don't degrade prompt quality unless
    # we can actually confirm cloud is exhausted).
    client_has_cloud_available = getattr(client, "has_cloud_available", lambda: True)
    if is_local_configured() and not client_has_cloud_available():
        from applypilot.scoring.local_tailor import _build_compact_local_prompt
        log.warning(
            "DEGRADED MODE: all cloud tailoring models are exhausted; falling back to "
            "local full-resume generation for %s. Quality is lower than cloud "
            "finalization -- this is an emergency fallback, not normal operation.",
            job.get("title", "")[:40],
        )
        tailor_prompt_base = _build_compact_local_prompt(profile)
    else:
        tailor_prompt_base = _build_tailor_prompt(profile, standup_decision=standup_decision)

    for attempt in range(max_retries + 1):
        report["attempts"] = attempt + 1

        # Fresh conversation every attempt
        prompt = tailor_prompt_base
        if avoid_notes:
            prompt += "\n\n## AVOID THESE ISSUES (from previous attempt):\n" + "\n".join(
                f"- {n}" for n in avoid_notes[-5:]
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                (f"TAILORING GUIDANCE (from local pre-analysis):\n{local_plan}\n\n---\n\n"
                 if local_plan else "")
                + f"ORIGINAL RESUME:\n{resume_text}\n\n---\n\nTARGET JOB:\n{job_text}\n\nReturn the JSON:"
            )},
        ]

        raw = client.chat(
            messages,
            max_tokens=get_token_limit("tailor", 16384),
            temperature=0.4,
        )

        # Parse JSON from response
        try:
            data = extract_json(raw)
        except ValueError:
            avoid_notes.append("Output was not valid JSON. Return ONLY a JSON object, nothing else.")
            continue

        # Layer 1: Validate JSON fields
        validation = validate_json_fields(data, profile, standup_decision=standup_decision)
        report["validator"] = validation

        if not validation["passed"]:
            avoid_notes.extend(validation["errors"])
            if attempt < max_retries:
                continue
            # Last attempt -- assemble whatever we got
            tailored = assemble_resume_text(data, profile)
            report["status"] = "failed_validation"
            return tailored, report

        # Assemble text (header injected by code, em dashes auto-fixed)
        tailored = assemble_resume_text(data, profile)
        candidate_facts = extract_facts_from_resume_json(data, profile)
        report["approved_facts"] = sorted(candidate_facts)

        if is_auto_approvable(candidate_facts):
            report["judge"] = {
                "passed": True,
                "verdict": "SKIP",
                "issues": "none",
                "raw": "auto-approved via fact-subset",
            }
            report["status"] = "approved"
            report["auto_approved_by_facts"] = True
            return tailored, report

        # Layer 2: LLM judge — skip on first clean pass to save an LLM call
        is_clean = not validation.get("warnings")
        is_last = attempt >= max_retries

        if attempt == 0 and is_clean:
            # First attempt with zero warnings — trust programmatic validation
            log.debug("Skipping judge on clean first attempt for %s", job.get("title", "")[:40])
            report["judge"] = {"passed": True, "verdict": "SKIP", "issues": "none", "raw": "skipped (clean first pass)"}
            report["status"] = "approved"
            return tailored, report

        judge = judge_tailored_resume(resume_text, tailored, job.get("title", ""), profile)
        report["judge"] = judge

        if not judge["passed"]:
            avoid_notes.append(f"Judge rejected: {judge['issues']}")
            if is_last:
                # Final attempt: accept if validation passed (judge is advisory)
                log.warning("Judge failed on final attempt for %s, accepting anyway (validation passed)",
                           job.get("title", "")[:40])
                report["status"] = "approved"
                return tailored, report
            continue

        # Both passed
        report["status"] = "approved"
        return tailored, report

    report["status"] = "exhausted_retries"
    return tailored, report


# ── Batch Entry Point ────────────────────────────────────────────────────

# Strategies where `site` column holds the actual employer (not an aggregator).
# When a job has NULL/empty `company`, these strategies let us fall back to
# `site` as the company-key for caps.
DIRECT_EMPLOYER_STRATEGIES = frozenset({
    "greenhouse_api", "workday_api", "lever_api", "ashby_api",
    "amazon_jobs", "microsoft_careers", "apple_jobs", "google_careers",
    "meta_careers", "twilio_greenhouse",
})


def resolve_company_key(job: dict) -> str | None:
    """Return a lowercase company key for cap logic, or None if exempt.

    Resolution order:
      1. The extracted ``company`` column when populated.
      2. The ATS-tenant slug embedded in ``application_url`` for any
         job that points at a known direct-employer ATS (Greenhouse,
         Lever, Ashby, Workday). This catches the LinkedIn-aggregator
         case where ``company`` is NULL but the apply URL clearly
         identifies the employer (e.g. a LinkedIn listing whose Apply
         button resolves to job-boards.greenhouse.io/{slug}/...).
      3. Direct-employer ``site`` (the existing fallback for the API
         scrapers).
    """
    co = (job.get("company") or "").strip().lower()
    if co:
        return co

    # 2) ATS-tenant extraction from application_url. Works regardless
    # of whether the discovery strategy was an aggregator or direct.
    apply_url = (job.get("application_url") or "").lower()
    if apply_url:
        # Greenhouse: boards.greenhouse.io/{slug}/jobs/N or
        #             job-boards.greenhouse.io/{slug}/jobs/N (eu/us)
        m = re.search(r"(?:job-)?boards(?:\.eu)?\.greenhouse\.io/([a-z0-9-]+)/", apply_url)
        if m:
            return m.group(1)
        # Lever: jobs.lever.co/{slug}/...
        m = re.search(r"jobs\.lever\.co/([a-z0-9-]+)/", apply_url)
        if m:
            return m.group(1)
        # Ashby: jobs.ashbyhq.com/{slug}/...
        m = re.search(r"jobs\.ashbyhq\.com/([a-z0-9-]+)/", apply_url)
        if m:
            return m.group(1)
        # Workday: {tenant}.wd*.myworkdayjobs.com/...
        m = re.search(r"https?://([a-z0-9-]+)\.wd[0-9]+\.myworkdayjobs\.com/", apply_url)
        if m:
            return m.group(1)

    # 3) Direct-employer site fallback.
    strategy = (job.get("strategy") or "").lower()
    if strategy in DIRECT_EMPLOYER_STRATEGIES:
        site = (job.get("site") or "").strip().lower()
        if site:
            return site
    return None


def display_company(job: dict) -> str:
    """Best-known employer name for LLM prompts, or '' if unknown.

    ``site`` is the discovery source — for aggregator listings (LinkedIn,
    Indeed, ...) it is NOT the employer and must never be presented as one.
    Prefers the ``company`` column's original casing; falls back to the
    lowercase ATS-tenant slug / direct-employer site from
    ``resolve_company_key``.
    """
    company = (job.get("company") or "").strip()
    if company:
        return company
    return resolve_company_key(job) or ""


def _name_parts(profile: dict) -> tuple[str, str]:
    """Return (first, last) name parts from profile, sanitized for filenames.

    Prefers preferred_name for the given-name half when available (e.g. "Jordan"
    from "Jordan Alexander Lee") so generated filenames match the
    candidate's public-facing name.
    """
    personal = profile.get("personal", {})
    preferred = (personal.get("preferred_name") or "").strip()
    full = (personal.get("full_name") or "").strip()
    parts = [p for p in re.split(r"\s+", full) if p] if full else []

    first_raw = preferred or (parts[0] if parts else "")
    last_raw = parts[-1] if len(parts) > 1 else ""
    first = re.sub(r"[^\w]", "", first_raw)
    last = re.sub(r"[^\w]", "", last_raw)
    return (first, last)


def _extract_keywords(job: dict, profile: dict, limit: int = 12) -> list[str]:
    """Extract ATS-relevant keywords from the job description.

    Returns the job title followed by any candidate boundary skills that
    appear in the title or description. Populated into DOCX
    core_properties.keywords so recruiters / ATS scanning document
    properties see a relevant keyword cloud.
    """
    result: list[str] = []
    title = (job.get("title") or "").strip()
    if title:
        result.append(title)

    boundary = profile.get("skills_boundary", {})
    skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            skills.extend(s for s in items if isinstance(s, str) and s)

    desc_head = (job.get("full_description") or "")[:4000].lower()
    combined = f"{title.lower()} {desc_head}"

    for skill in skills:
        if len(result) >= limit:
            break
        if skill.lower() in combined and skill not in result:
            result.append(skill)

    return result


def _tailor_one_job(job: dict, resume_text: str | None, profile: dict, doc_format: str = "docx") -> dict:
    """Tailor resume for a single job. Safe to call from multiple threads."""
    if resume_text is None:
        resume_text, _ = load_resume_text_for_job(job)
    # Optional local first-pass: ask the local model for a cheap, structured
    # tailoring PLAN before the cloud model produces the final resume. The
    # plan is grounding-checked (validate_local_plan) and rendered to compact
    # text (format_local_plan_for_cloud) before being handed to the cloud
    # model -- never the raw local-model output. Enabled via env var so the
    # batch runner doesn't need a per-job flag.
    local_plan_text: str | None = None
    if os.environ.get("APPLYPILOT_LOCAL_PLAN", "").lower() in ("1", "true", "yes"):
        from applypilot.scoring.local_tailor import format_local_plan_for_cloud, get_local_tailoring_plan
        try:
            plan = get_local_tailoring_plan(resume_text, job, profile)
            local_plan_text = format_local_plan_for_cloud(plan) or None
        except Exception:
            log.debug("Local tailoring plan failed for %s",
                      (job.get("title") or "")[:40], exc_info=True)
    tailored, report = tailor_resume(resume_text, job, profile, local_plan=local_plan_text)
    approved_facts = set(report.get("approved_facts") or [])
    if report.get("status") == "approved" and approved_facts:
        source = f"{job.get('url', '')}:{job.get('title', '')}"
        record_approved_facts(approved_facts, source=source)

    # Filename: FirstName_LastName_JobTitle_hash.docx per Jobscan §3.
    # Hash retains uniqueness since the same title may recur across employers.
    first, last = _name_parts(profile)
    safe_title = re.sub(r"[^\w\s-]", "", job.get("title") or "untitled")[:50].strip().replace(" ", "_")
    url_hash = hashlib.md5(job["url"].encode()).hexdigest()[:8]
    if first and last:
        prefix = f"{first}_{last}_{safe_title}_{url_hash}"
    elif first:
        prefix = f"{first}_{safe_title}_{url_hash}"
    else:
        safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
        prefix = f"{safe_site}_{safe_title}_{url_hash}"

    txt_path = TAILORED_DIR / f"{prefix}.txt"
    txt_path.write_text(tailored, encoding="utf-8")

    job_path = TAILORED_DIR / f"{prefix}_JOB.txt"
    job_desc = (
        f"Title: {job['title']}\n"
        f"Company: {display_company(job) or 'unknown'}\n"
        f"Source: {job['site']}\n"
        f"Location: {job.get('location', 'N/A')}\n"
        f"Score: {job.get('fit_score', 'N/A')}\n"
        f"URL: {job['url']}\n\n"
        f"{job.get('full_description', '')}"
    )
    job_path.write_text(job_desc, encoding="utf-8")

    report_path = TAILORED_DIR / f"{prefix}_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    doc_path = None
    if report["status"] == "approved":
        try:
            from applypilot.scoring.pdf import convert_to_pdf
            personal = profile.get("personal", {})
            full_name = personal.get("full_name") or personal.get("preferred_name") or ""
            job_title = (job.get("title") or "").strip()[:150]
            site = (job.get("site") or "").strip()[:80]
            metadata = {
                "title": f"Resume — {full_name} for {job_title}" if full_name else f"Resume — {job_title}",
                "subject": job_title,
                "author": full_name,
                "category": "Resume",
                "keywords": _extract_keywords(job, profile),
                "comments": (
                    f"Customized for: {job_title}\n"
                    f"Source: {site}\n"
                    f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
                ),
            }
            doc_path = str(convert_to_pdf(txt_path, doc_format=doc_format, metadata=metadata))
        except Exception:
            log.debug("Document generation failed for %s", txt_path, exc_info=True)

    return {
        "url": job["url"],
        "path": str(txt_path),
        "pdf_path": doc_path,
        "title": job["title"],
        "site": job["site"],
        "status": report["status"],
        "attempts": report["attempts"],
    }


def _mark_tailor_result(
    conn,
    url: str,
    status: str,
    path: str | None,
    *,
    attempts: int | None = None,
    auto_approved_by_facts: bool = False,
    now: str | None = None,
) -> None:
    """Persist one tailor result to the DB and emit a state transition.

    Extracted from the inner ``_flush_tailor_results`` so that tests can
    call it directly without running the full LLM pipeline.

    ``status`` should be one of: ``"approved"``, ``"failed_validation"``,
    ``"failed_judge"``, ``"error"``, ``"exhausted_retries"``.
    """
    if now is None:
        now = datetime.now(timezone.utc).isoformat()

    from applypilot.database import transition_state

    if status == "approved":
        # Atomic: write path AND increment counter in one UPDATE.
        # transition_state fires after — if it raises (rare), path+count are
        # already written so state drift is recoverable via backfill.
        conn.execute(
            "UPDATE jobs SET tailored_resume_path=?, tailored_at=?, "
            "tailor_attempts=COALESCE(tailor_attempts,0)+1, "
            "tailor_auto_approved=? WHERE url=?",
            (path, now, 1 if auto_approved_by_facts else 0, url),
        )
        transition_state(
            conn, url, "tailored",
            reason="tailored OK",
            metadata={"attempts": attempts, "filename": os.path.basename(path) if path else None},
            force=True,
        )
    else:
        conn.execute(
            "UPDATE jobs SET tailor_attempts = COALESCE(tailor_attempts, 0) + 1 "
            "WHERE url = ?",
            (url,),
        )
        transition_state(
            conn, url, "tailor_failed",
            reason=status,
            metadata={"attempts": attempts},
            force=True,
        )


def run_tailoring(min_score: int | None = None, limit: int = 20, workers: int = 1,
                  doc_format: str = "docx", max_age_days: int | None = None) -> dict:
    """Generate tailored resumes for high-scoring jobs.

    Args:
        min_score: Minimum fit_score to tailor for (default from config).
        limit: Maximum jobs to process.
        workers: Parallel LLM threads (default 1 = sequential).
        doc_format: Output document format — "docx" (default) or "pdf".
        max_age_days: Skip jobs older than this (default from config).

    Returns:
        {"approved": int, "failed": int, "errors": int, "elapsed": float}
    """
    from applypilot.config import DEFAULTS
    if min_score is None:
        min_score = DEFAULTS["min_score"]

    from applypilot.database import get_connection, get_jobs_by_stage, write_with_retry

    profile = load_profile()
    conn = get_connection()

    # Note: get_jobs_by_stage now applies a 14-day discovered_at filter by
    # default (config.DEFAULTS["max_job_age_days"]). Pass max_age_days=0
    # to disable.
    jobs = get_jobs_by_stage(conn=conn, stage="pending_tailor",
                             min_score=min_score, max_age_days=max_age_days,
                             limit=limit)

    # Per-company tailor cap: don't tailor more than N resumes per company
    # in the fresh window. Existing tailored docs count against the cap.
    #
    # Company key resolution:
    #   1. Prefer `company` column (extracted from application_url domain).
    #   2. If NULL, fall back to `site` ONLY for direct-employer scrapers
    #      (greenhouse_api, workday_api) where `site` is the actual employer.
    #   3. For aggregator sites (linkedin, indeed, simplyhired, etc.) the
    #      `site` is not the employer — exempt those from the cap since we
    #      can't accurately bucket them.
    cap = DEFAULTS["max_tailored_per_company"]

    # (resolve_company_key is defined at module level for reuse)

    existing_rows = conn.execute("""
        SELECT LOWER(company) AS key, COUNT(*) AS n
        FROM jobs
        WHERE tailored_resume_path IS NOT NULL
          AND state != 'archived'
          AND company IS NOT NULL AND TRIM(company) != ''
          AND discovered_at > datetime('now', ?)
        GROUP BY key
        UNION ALL
        SELECT LOWER(site) AS key, COUNT(*) AS n
        FROM jobs
        WHERE tailored_resume_path IS NOT NULL
          AND state != 'archived'
          AND (company IS NULL OR TRIM(company) = '')
          AND strategy IN ('greenhouse_api', 'workday_api', 'lever_api',
                           'ashby_api', 'amazon_jobs', 'microsoft_careers',
                           'apple_jobs', 'google_careers')
          AND site IS NOT NULL AND TRIM(site) != ''
          AND discovered_at > datetime('now', ?)
        GROUP BY key
    """, (f"-{max_age_days or DEFAULTS['max_job_age_days']} days",
          f"-{max_age_days or DEFAULTS['max_job_age_days']} days")).fetchall()
    existing: dict[str, int] = {}
    for r in existing_rows:
        existing[r["key"]] = existing.get(r["key"], 0) + r["n"]

    added_per_company: dict[str, int] = {}
    capped_jobs: list[dict] = []
    skipped_by_cap = 0
    for job in jobs:
        key = resolve_company_key(job)
        if key is None:
            capped_jobs.append(job)  # exempt (aggregator with no company)
            continue
        already = existing.get(key, 0) + added_per_company.get(key, 0)
        if already >= cap:
            skipped_by_cap += 1
            continue
        capped_jobs.append(job)
        added_per_company[key] = added_per_company.get(key, 0) + 1
    if skipped_by_cap:
        log.info("Tailor cap: skipped %d job(s) where company is at/over %d tailored.",
                 skipped_by_cap, cap)
    jobs = capped_jobs

    conn.commit()  # Close read transaction before long LLM phase

    if not jobs:
        log.info("No untailored jobs with score >= %d (after per-company cap).", min_score)
        return {"approved": 0, "failed": 0, "errors": 0, "elapsed": 0.0}

    TAILORED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Tailoring resumes for %d jobs (score >= %d, workers=%d)...", len(jobs), min_score, workers)
    t0 = time.time()
    completed = 0
    results: list[dict] = []
    stats: dict[str, int] = {"approved": 0, "failed_validation": 0, "failed_judge": 0, "error": 0}

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_tailor_one_job, job, None, profile, doc_format): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception:
                    result = {
                        "url": job["url"], "title": job["title"], "site": job["site"],
                        "status": "error", "attempts": 0, "path": None, "pdf_path": None,
                    }
                    log.exception("[ERROR] %s -- exception from future.result()", (job.get("title") or "")[:40])

                results.append(result)
                stats[result.get("status", "error")] = stats.get(result.get("status", "error"), 0) + 1
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                log.info(
                    "%d/%d [%s] attempts=%s | %.1f jobs/min | %s",
                    completed, len(jobs), result["status"].upper(),
                    result.get("attempts", "?"), rate * 60,
                    (result.get("title") or "")[:40],
                )
    else:
        for job in jobs:
            completed += 1
            try:
                result = _tailor_one_job(job, None, profile, doc_format)
            except Exception:
                result = {
                    "url": job["url"], "title": job.get("title") or "", "site": job["site"],
                    "status": "error", "attempts": 0, "path": None, "pdf_path": None,
                }
                log.exception("%d/%d [ERROR] %s -- exception during tailoring", completed, len(jobs), (job.get("title") or "")[:40])

            results.append(result)
            stats[result.get("status", "error")] = stats.get(result.get("status", "error"), 0) + 1
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            log.info(
                "%d/%d [%s] attempts=%s | %.1f jobs/min | %s",
                completed, len(jobs), result["status"].upper(),
                result.get("attempts", "?"), rate * 60,
                (result.get("title") or "")[:40],
            )

    # Persist to DB: increment attempt counter for ALL, save path only for approved
    now = datetime.now(timezone.utc).isoformat()

    def _flush_tailor_results(conn, results, now):
        for r in results:
            # Prefer the generated DOCX/PDF path. Fall back to the text
            # path only if conversion silently failed (e.g., missing
            # python-docx); the apply layer will flag that as invalid.
            stored_path = r.get("pdf_path") or r.get("path")
            _mark_tailor_result(
                conn, r["url"], r["status"], stored_path,
                attempts=r.get("attempts"),
                auto_approved_by_facts=bool(r.get("auto_approved_by_facts")),
                now=now,
            )

    try:
        write_with_retry(conn, _flush_tailor_results, conn, results, now)
    except Exception:
        log.exception("DB flush failed for tailor batch")

    elapsed = time.time() - t0
    log.info(
        "Tailoring done in %.1fs: %d approved, %d failed_validation, %d failed_judge, %d errors",
        elapsed,
        stats.get("approved", 0),
        stats.get("failed_validation", 0),
        stats.get("failed_judge", 0),
        stats.get("error", 0),
    )

    return {
        "approved": stats.get("approved", 0),
        "failed": stats.get("failed_validation", 0) + stats.get("failed_judge", 0),
        "errors": stats.get("error", 0),
        "elapsed": elapsed,
    }
