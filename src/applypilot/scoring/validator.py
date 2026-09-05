"""Resume and cover letter validation: banned words, fabrication detection, structural checks.

All validation is profile-driven -- no hardcoded personal data. The validator receives
a profile dict (from applypilot.config.load_profile()) and validates against the user's
actual skills, companies, projects, and school.
"""

import logging
import os
import re

log = logging.getLogger(__name__)


# ── Universal Constants (not personal data) ───────────────────────────────

BANNED_WORDS: list[str] = [
    "passionate",
    "committed to",
    "utilizing",
    "utilize",
    "harnessing",
    "spearheaded",
    "spearhead",
    "orchestrated",
    "championed",
    "pioneered",
    "robust",
    "scalable solutions",
    "cutting-edge",
    "state-of-the-art",
    "best-in-class",
    "proven track record",
    "track record of success",
    "demonstrated ability",
    "strong communicator",
    "team player",
    "fast learner",
    "self-starter",
    "go-getter",
    "synergy",
    "cross-functional collaboration",
    "holistic",
    "transformative",
    "innovative solutions",
    "paradigm",
    "ecosystem",
    "proactive",
    "detail-oriented",
    "highly motivated",
    "seamless",
    "full lifecycle",
    "deep understanding",
    "extensive experience",
    "comprehensive knowledge",
    "thrives in",
    "excels at",
    "adept at",
    "well-versed in",
    "i am confident",
    "i am excited",
    "plays a critical role",
    "instrumental in",
    "integral part of",
    "strong track record",
    "eager to",
    "eager",
    # Cover-letter-specific additions
    "this demonstrates",
    "this reflects",
    "i have experience with",
    "furthermore",
    "additionally",
    "moreover",
]

# Cover-letter hard-reject patterns (ERROR tier — triggers a regeneration
# retry, unlike BANNED_WORDS which only warns). Stem-based regexes so suffix
# variants ("aligns with", "These experiences demonstrate", "resonates") can't
# slip past plain-substring matching the way they did in ~28% of generated
# letters before this existed.
CL_BANNED_PATTERNS: list[tuple[str, str]] = [
    ("align with", r"\balign(s|ed|ing)?\s+with\b"),
    ("demonstrate", r"\bdemonstrat\w*\b"),
    ("happy to walk through", r"\bhappy to walk (you\s+)?through\b"),
    ("resonate", r"\bresonat\w*\b"),
]

LLM_LEAK_PHRASES: list[str] = [
    "i am sorry",
    "i apologize",
    "i will try",
    "let me try",
    "i am at a loss",
    "i am truly sorry",
    "apologies for",
    "i keep fabricating",
    "i will have to admit",
    "one final attempt",
    "one last time",
    "if it fails again",
    "persistent errors",
    "i am having difficulty",
    "i made an error",
    "my mistake",
    "here is the corrected",
    "here is the revised",
    "here is the updated",
    "here is my",
    "below is the",
    "as requested",
    "note:",
    "disclaimer:",
    "important:",
    "i have rewritten",
    "i have removed",
    "i have fixed",
    "i have replaced",
    "i have updated",
    "i have corrected",
    "per your feedback",
    "based on your feedback",
    "as per the instructions",
    "the following resume",
    "the resume below",
    "the following cover letter",
    "the letter below",
]

# Known fabrication markers.
#
# 2026-08-25 policy reversal (profile-authority hardening): this watchlist's
# ORIGINAL design (2026-06-ish) deliberately excluded "reasonable stretches"
# like K8s/Terraform/Redis/Kafka, on the theory that a closely-related tool
# the candidate could plausibly pick up was a minor, acceptable liberty --
# matching the tailoring prompt's old (now-removed) "you MAY add 2-3
# closely related tools" instruction and the judge prompt's old "closely
# related tool... is a MINOR STRETCH, not fabrication" tolerance rule.
# Real-world verification against an ACTUAL pre-fix generated resume found
# this produced literal, confirmed fabrication: "Kubernetes (Exposure)"
# appeared on a resume where Kubernetes is grounded NOWHERE in the profile
# (not skills_inventory, not any project's factual_concepts) -- the direct
# result of the old prompt's own worked example, "Kubernetes if Docker."
# Per the current, explicit, more conservative policy (profile.json is
# authoritative for every resume fact; omit rather than stretch), common
# cloud/infra/framework terms with zero grounding anywhere in this
# candidate's profile are now watchlisted like any other unrelated
# technology -- "reasonable stretch" is no longer an exemption. The
# cross-reference against `allowed_skills` (skip if the profile actually
# supports it) still applies, so this can never block a real skill.
FABRICATION_WATCHLIST: set[str] = {
    # Languages with zero relation to the candidate's stack
    # NOTE: "golang" removed — synonym for Go (in profile). "c#" skipped by len<=2 guard.
    "c#",
    "c++",
    "rust",
    "ruby",
    "swift",
    "scala",
    "matlab",
    # Frameworks for wrong languages
    # NOTE: kotlin, django, spring, angular, vue removed — all in candidate's skills_boundary.
    # The skip logic cross-references against profile, but keeping them out avoids edge cases.
    "rails",
    "svelte",
    # Hard lies: certifications not in profile (real certs are checked via skills_boundary)
    "pmp",
    "scrum master",
    # Cloud/infra/DevOps -- common hallucination targets, verified absent
    # from the current profile.json (skills_inventory, skills_boundary,
    # and every project's factual_concepts).
    "kubernetes",
    "terraform",
    "ansible",
    "jenkins",
    "docker compose",
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "nginx",
    # Databases/data infra
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "kafka",
    "spark",
    "hadoop",
    # Frontend/backend frameworks and languages beyond the profile's actual Python
    "react",
    "vue",
    "node.js",
    "typescript",
    "graphql",
    "flask",
    "spring",
    # ML/AI frameworks -- verified absent (profile's ML-adjacent work is
    # limited to project-level "machine-learning experimentation", never a
    # named framework)
    "tensorflow",
    "pytorch",
}

REQUIRED_SECTIONS: set[str] = {"SUMMARY", "TECHNICAL SKILLS", "EXPERIENCE", "PROJECTS", "EDUCATION"}


# Cover-letter validation defaults; all are optionally overridable via env.
_CL_MIN_WORDS = int(os.environ.get("APPLYPILOT_CL_MIN_WORDS", "260"))
_CL_TARGET_MIN_WORDS = int(os.environ.get("APPLYPILOT_CL_TARGET_MIN_WORDS", "300"))
_CL_TARGET_MAX_WORDS = int(os.environ.get("APPLYPILOT_CL_TARGET_MAX_WORDS", "400"))
_CL_HARD_MAX_WORDS = int(os.environ.get("APPLYPILOT_CL_HARD_MAX_WORDS", "1200"))
_CL_REQUIRED_PARAGRAPHS = int(os.environ.get("APPLYPILOT_CL_REQUIRED_PARAGRAPHS", "4"))


# ── Helpers ───────────────────────────────────────────────────────────────


def _build_skills_set(profile: dict) -> set[str]:
    """Build the set of allowed skills from the profile's skills_boundary,
    plus any skills_inventory item not explicitly marked resume_allowed=
    false (the more granular, authoritative source -- skills_boundary is
    typically a mirror of it, but this makes the allowed-set robust even if
    the two drift)."""
    boundary = profile.get("skills_boundary", {})
    allowed: set[str] = set()
    for category in boundary.values():
        if isinstance(category, (list, set)):
            allowed.update(s.lower().strip() for s in category)
    for item in profile.get("skills_inventory") or []:
        if isinstance(item, dict) and item.get("resume_allowed") is not False:
            name = item.get("name")
            if name:
                allowed.add(str(name).lower().strip())
    return allowed


def _disallowed_skill_names(profile: dict) -> list[str]:
    """skills_inventory entries explicitly marked resume_allowed=false
    (e.g. APIs/REST/JSON/Scripting/Docker: 'learning/exposure, not
    resume-ready') -- these must never appear in the SKILLS section."""
    names = []
    for item in profile.get("skills_inventory") or []:
        if isinstance(item, dict) and item.get("resume_allowed") is False and item.get("name"):
            names.append(str(item["name"]))
    return names


def _project_scoped_technology_terms(profile: dict) -> set[str]:
    """Technology terms that appear ONLY inside project_inventory's
    factual_concepts (e.g. "You Power You"'s "HTML/CSS/JavaScript") and are
    NOT in the allowed-skills set -- project-level evidence, never a
    general resume skill, per each project's own evidence_level. See
    check_unsupported_technical_skills."""
    allowed = _build_skills_set(profile)
    terms: set[str] = set()
    for item in profile.get("project_inventory") or []:
        if not isinstance(item, dict):
            continue
        for concept in item.get("factual_concepts") or []:
            for token in re.split(r"[/,;]+", str(concept)):
                token = token.strip().lower()
                if token and len(token) > 1 and token not in allowed:
                    terms.add(token)
    return terms


def check_unsupported_technical_skills(skills_text: str, profile: dict) -> list[str]:
    """Return the sorted list of skill/technology names found in
    `skills_text` (the tailored resume's SKILLS section, flattened to
    text) that the profile does NOT support as a general resume skill --
    either an explicit skills_inventory resume_allowed=false item, or a
    term that only ever appears as PROJECT-scoped evidence
    (project_inventory's factual_concepts) and was never separately vetted
    as a skill. Empty list when nothing to flag."""
    if not skills_text:
        return []
    skills_lower = skills_text.lower()
    violations: set[str] = set()
    for name in _disallowed_skill_names(profile):
        if re.search(rf"\b{re.escape(name.lower())}\b", skills_lower):
            violations.add(name)
    for term in _project_scoped_technology_terms(profile):
        if re.search(rf"\b{re.escape(term)}\b", skills_lower):
            violations.add(term)
    return sorted(violations)


# 2026-09-01: FABRICATION_WATCHLIST and check_unsupported_technical_skills
# above only catch fabrications someone has already observed and hand-
# enumerated (a specific denylist of tech terms, or skills_inventory items
# already known to the profile but marked non-resume-appropriate). Neither
# can catch a technology invented from nothing that happens not to be on
# the watchlist yet -- and the watchlist has to be manually re-curated
# every time a new fabrication pattern is discovered, which doesn't
# generalize across different candidates/profiles or a profile that
# changes over time. This is a POSITIVE check instead: every individual
# skill claim in the SKILLS section must trace back to something the
# loaded profile actually declares (skills_boundary / skills_inventory),
# regardless of which specific term it is. The stopword list below is
# deliberately generic ENGLISH GRAMMAR/structure (connectors, category
# labels, proficiency qualifiers) -- never a technology name -- so it
# never needs updating as new tools get invented or fabricated, unlike
# FABRICATION_WATCHLIST.
_SKILL_CLAIM_STOPWORDS: set[str] = {
    "and", "or", "with", "using", "etc", "other", "others", "various",
    "including", "such", "as", "of", "for", "in", "to", "the", "a", "an",
    "tools", "languages", "language", "frameworks", "framework",
    "technologies", "technology", "skills", "software", "systems",
    "system", "platforms", "platform", "methods", "methodologies",
    "methodology", "services", "solutions", "concepts", "practices",
    "libraries", "library", "databases", "infrastructure", "general",
    "exposure", "familiarity", "familiar", "basic", "basics", "beginner",
    "intermediate", "advanced", "proficient", "proficiency", "working",
    "knowledge", "hands-on", "hands", "on", "experience", "learning",
    "developing", "practical",
}


def _split_skill_claims(skills_text: str) -> list[str]:
    """Split a SKILLS section's text into individual discrete skill claims,
    independent of any hardcoded technology list. Flattens parenthetical
    sub-lists (e.g. "Web Development (HTML, CSS, JavaScript)") into
    separate claims, splits on the usual list separators, and drops
    generic connector/category-label/qualifier noise via
    _SKILL_CLAIM_STOPWORDS."""
    flattened = skills_text.replace("(", ",").replace(")", ",")
    claims: list[str] = []
    for chunk in re.split(r"[,;/|]+", flattened):
        chunk = chunk.strip().strip(":").strip()
        if not chunk:
            continue
        # Drop the chunk only if EVERY word in it is a stopword (e.g.
        # "Basic Familiarity") -- a chunk with at least one non-stopword
        # word (e.g. "Data Analysis") is a real candidate skill claim.
        words = chunk.lower().split()
        if words and all(w in _SKILL_CLAIM_STOPWORDS for w in words):
            continue
        claims.append(chunk)
    return claims


def check_unsupported_skill_claims(skills_text: str, profile: dict) -> list[str]:
    """General, profile-driven fabrication check for the SKILLS section:
    every individual skill claim must be traceable to the profile's own
    declared skills (skills_boundary / skills_inventory), via exact or
    substring/word-level match. Unlike FABRICATION_WATCHLIST (a fixed
    denylist of previously-observed hallucinations), this is a POSITIVE
    check derived entirely from whichever profile is loaded -- it
    generalizes to any candidate's resume and any future technology a
    model might invent, without the watchlist needing to be hand-updated
    every time a new fabrication is discovered. Kept alongside the
    watchlist and check_unsupported_technical_skills (defense in depth),
    not as a replacement. Empty list when there's nothing to flag, or when
    the profile declares no skills at all to ground a judgment on."""
    if not skills_text:
        return []
    allowed = _build_skills_set(profile)
    if not allowed:
        return []
    allowed_words: set[str] = set()
    for skill in allowed:
        allowed_words.update(w for w in re.findall(r"[a-z0-9+#.]+", skill) if len(w) > 2)

    violations: list[str] = []
    for claim in _split_skill_claims(skills_text):
        claim_lower = claim.lower()
        if len(claim_lower) <= 2:
            continue
        if any(claim_lower in a or a in claim_lower for a in allowed):
            continue
        claim_words = [w for w in re.findall(r"[a-z0-9+#.]+", claim_lower) if len(w) > 2]
        if claim_words and any(w in allowed_words for w in claim_words):
            continue
        if claim_words:
            violations.append(claim)
    return violations


def _private_project_names(profile: dict) -> list[str]:
    """Return profile-marked private/non-resume project names."""
    names = set(profile.get("resume_facts", {}).get("private_projects", []))
    for item in profile.get("project_inventory", []):
        if isinstance(item, dict) and (item.get("private") or item.get("resume_allowed") is False) and item.get("name"):
            names.add(item["name"])
    return sorted(names)


def _unfinished_project_names(profile: dict) -> list[str]:
    """Return projects whose profile status forbids success claims."""
    names = set(profile.get("resume_facts", {}).get("unfinished_projects", []))
    for item in profile.get("project_inventory", []):
        if isinstance(item, dict) and "unfinished" in str(item.get("status", "")).lower() and item.get("name"):
            names.add(item["name"])
    return sorted(names)


def _append_profile_integrity_errors(text: str, profile: dict, errors: list[str]) -> None:
    """Apply profile-driven privacy/status/education integrity checks."""
    text_lower = text.lower()
    for name in _private_project_names(profile):
        if name.lower() in text_lower:
            errors.append(f"Private project leaked into resume: '{name}'")

    success_terms = re.compile(
        r"\b(deployed|production|users?|revenue|profit(?:able)?|conversion|engagement|"
        r"accuracy|successful|generated income)\b"
    )
    for name in _unfinished_project_names(profile):
        if name.lower() in text_lower and success_terms.search(text_lower):
            errors.append(f"Unfinished project has unsupported success claim: '{name}'")

    for item in profile.get("education", []):
        if not isinstance(item, dict):
            continue
        official = str(item.get("official_degree", "")).lower()
        field = str(item.get("field_of_study", "")).lower()
        if official and field and field not in text_lower:
            errors.append(f"Official education field missing or changed: '{item.get('field_of_study')}'")
        if "bachelor of science" in text_lower and "bachelor of science" not in official:
            errors.append("Official education incorrectly changed to Bachelor of Science")


def _missing_schools(preserved_school: str, haystack: str) -> list[str]:
    """Return preserved schools that are absent from ``haystack``.

    ``preserved_school`` may list several schools separated by ``;`` (e.g.
    "Riverside Community College; Lakewood College; Central High School").
    Education now renders as structured per-school entries, so the exact joined
    string no longer appears verbatim — check each school individually instead.
    """
    haystack_lower = haystack.lower()
    schools = [s.strip() for s in preserved_school.split(";") if s.strip()]
    return [s for s in schools if s.lower() not in haystack_lower]


def sanitize_text(text: str) -> str:
    """Auto-fix common LLM output issues instead of rejecting."""
    text = text.replace(" \u2014 ", ", ").replace("\u2014", ", ")  # em dash -> comma
    text = text.replace("\u2013", "-")  # en dash -> hyphen
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # smart double quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # smart single quotes
    return text.strip()


# ── JSON Field Validation ─────────────────────────────────────────────────


def validate_json_fields(
    data: dict, profile: dict, standup_decision: str | None = None, job: dict | None = None
) -> dict:
    """Validate individual JSON fields from an LLM-generated tailored resume.

    Args:
        data: Parsed JSON from the LLM (title, summary, skills, experience, projects, education).
        profile: User profile dict from load_profile().
        job: Optional job dict (title/full_description) -- when given, also
            runs check_bullet_vocabulary_injection (advisory-tier only;
            omitting job simply skips that one check, everything else is
            unaffected).

    Returns:
        {"passed": bool, "errors": list[str], "warnings": list[str]}
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Required keys (projects is optional — 1-page resumes may omit them)
    for key in ("title", "summary", "skills", "experience", "education"):
        if key not in data or not data[key]:
            errors.append(f"Missing required field: {key}")
    if "projects" not in data or not data.get("projects"):
        warnings.append("Missing field: projects (optional, LLM may have folded into experience)")
    if errors:
        return {"passed": False, "errors": errors, "warnings": warnings}

    # Collect all text for bulk checks
    all_text_parts: list[str] = [data["summary"]]

    # Skills: check for fabrication (exclude items that are in user's actual profile)
    allowed_skills = _build_skills_set(profile)
    if isinstance(data["skills"], dict):
        # 2026-09-01: joining with a bare space let two adjacent SKILLS
        # categories merge into one claim when neither value's own text
        # ended in a delimiter (e.g. {"Other": "Outreach", "Fabricated":
        # "Salesforce"} -> "outreach salesforce") -- the merged claim then
        # falsely substring-matched the real "outreach" skill, silently
        # exempting the fabricated content riding along with it. Found by
        # the large-sample statistical fabrication-detection test
        # (data/experiments/validator_fabrication_detection_stats_20260901),
        # not by any hand-picked example -- every hand-picked test case so
        # far happened to keep multi-item claims within a single,
        # comma-separated category value. Joining with ", " instead
        # guarantees a category boundary is always also a claim boundary.
        skills_text = ", ".join(str(v) for v in data["skills"].values()).lower()
        for fake in FABRICATION_WATCHLIST:
            if len(fake) <= 2:
                continue
            # Skip if this "fabrication" is actually a real skill in the profile
            if any(fake in skill for skill in allowed_skills):
                continue
            if fake in skills_text:
                errors.append(f"Fabricated skill: '{fake}'")

        unsupported_skills = check_unsupported_technical_skills(skills_text, profile)
        flagged_skill_names = {n.lower() for n in unsupported_skills} | {
            f.lower() for f in FABRICATION_WATCHLIST if f in skills_text and len(f) > 2
        }
        for name in unsupported_skills:
            errors.append(
                f"Unsupported technical skill in SKILLS section: '{name}' "
                f"(not resume_allowed, or only ever project-scoped evidence)"
            )

        for claim in check_unsupported_skill_claims(skills_text, profile):
            if claim.lower() in flagged_skill_names:
                continue
            errors.append(
                f"Skill claim not grounded in profile: '{claim}' "
                f"(no matching entry in skills_boundary/skills_inventory)"
            )

    # Experience: check preserved companies (warn for missing, don't hard-fail
    # since 1-page resumes may legitimately omit early-career roles)
    resume_facts = profile.get("resume_facts", {})
    preserved_companies = resume_facts.get("preserved_companies", [])

    if isinstance(data["experience"], list):
        exp_and_proj_text = " ".join(str(e.get("header", "")) for e in data["experience"])
        if isinstance(data.get("projects"), list):
            exp_and_proj_text += " " + " ".join(str(e.get("header", "")) for e in data["projects"])
        for company in preserved_companies:
            if company.lower() not in exp_and_proj_text.lower():
                warnings.append(f"Company '{company}' not in experience or projects")
        for entry in data["experience"]:
            all_text_parts.extend(entry.get("bullets", []))

    # Projects: collect bullets
    if isinstance(data.get("projects"), list):
        for entry in data["projects"]:
            all_text_parts.extend(entry.get("bullets", []))

    # Education: each preserved school must be present (education may be a
    # structured list of per-school entries or a legacy single string).
    preserved_school = resume_facts.get("preserved_school", "")
    if preserved_school:
        edu = str(data.get("education", ""))
        missing = _missing_schools(preserved_school, edu)
        if missing:
            errors.append(f"Education missing school(s): {', '.join(missing)}")

    # Bulk checks on all text (word-boundary matching)
    all_text = " ".join(all_text_parts).lower()

    found_banned = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", all_text)]
    if found_banned:
        warnings.append(f"Banned words (style): {', '.join(found_banned[:3])}")

    inflation = check_claim_inflation(all_text, profile)
    if inflation:
        warnings.append(inflation)

    agency_inflation = check_agency_inflation(all_text, profile)
    if agency_inflation:
        warnings.append(agency_inflation)

    title_inflation = check_title_inflation(str(data.get("title") or ""), profile)
    if title_inflation:
        errors.append(title_inflation)

    errors.extend(check_date_placeholder_fabrication(data))
    errors.extend(check_date_fabrication(data, profile))

    found_leaks = [p for p in LLM_LEAK_PHRASES if p in all_text]
    if found_leaks:
        errors.append(f"LLM self-talk: '{found_leaks[0]}'")

    if standup_decision == "EXCLUDE" and re.search(r"\b(stand[- ]?up|comed(y|ian)|open mic|improv)\b", all_text):
        errors.append("Stand-up content present while decision is EXCLUDE")

    _append_profile_integrity_errors(str(data), profile, errors)

    # Factual anchor check: experience headers must use known employer names.
    # Catches local-model hallucinations (inventing new employers) that the
    # existing preserved-company warning above doesn't cover.
    _anchor = validate_factual_anchors(data, profile)
    errors.extend(_anchor["errors"])
    warnings.extend(_anchor["warnings"])

    warnings.extend(check_bullet_vocabulary_injection(data, profile, job=job))

    return {"passed": len(errors) == 0, "errors": errors, "warnings": warnings}


# ── Claim-strength inflation check (scoring/schemas.py's claim lattice) ──
#
# 2026-08-24: the degraded-mode local-realization path enforces claim-
# strength per bullet, against the specific evidence item each bullet
# realizes (see local_tailor.request_local_realization) -- it has clean
# per-bullet provenance to check against. The CLOUD tailoring path doesn't:
# the cloud model returns a whole reworded resume with no per-bullet
# evidence tag, so there's no way to know exactly which evidence a given
# bullet is claiming to realize. This is therefore a coarser, WARNING-tier
# (not error-tier -- see decision #19, "banned words = warnings not
# errors") global check: does the generated resume use an AUTHORITY-tier
# claim verb (architected/led/managed/owned/directed/spearheaded) anywhere,
# when NOTHING in the candidate's own profile evidence supports that tier
# at all? A false positive here just adds an advisory warning a human
# reviews, not a rejected draft -- deliberately conservative given it
# can't pinpoint which specific bullet is the problem.


def check_claim_inflation(all_text: str, profile: dict) -> str:
    """Return a warning string if `all_text` (the tailored resume's
    combined summary+bullets) uses an authority-tier claim verb while the
    candidate's own profile evidence never earns that tier anywhere --
    empty string when there's nothing to flag."""
    from applypilot.scoring.schemas import detect_claim_tier

    detected = detect_claim_tier(all_text)
    if detected != "authority":
        return ""

    for key in ("experience_inventory", "project_inventory", "historical_experience_inventory"):
        for item in profile.get(key) or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("description") or "")
            if detect_claim_tier(text) == "authority":
                return ""  # at least one real evidence item earns this tier -- not inflation

    return (
        "Possible claim-strength inflation: resume uses an authority-tier "
        "technical-depth verb (architected) but no profile evidence item's "
        "own description supports that tier"
    )


# ── Agency inflation check (schemas.py's separate agency axis) ───────────
#
# 2026-08-23: authority-sounding people verbs (led/managed/supervised/
# directed/spearheaded/drove) were split off the technical-depth lattice
# above into their own AGENCY_TIERS axis (schemas.AGENCY_TIERS) -- a
# candidate who "architected" a system doesn't thereby earn "led the team
# that architected it". This is the same coarse, WARNING-tier, whole-resume
# check as check_claim_inflation above, just against the agency axis
# instead: does the generated resume use a team_lead/director-tier agency
# verb anywhere the candidate's own profile evidence never earns?


def check_agency_inflation(all_text: str, profile: dict) -> str:
    """Return a warning string if `all_text` uses a team_lead- or
    director-tier agency verb while no profile evidence item's own
    description earns an agency tier at least that strong -- empty string
    when there's nothing to flag."""
    from applypilot.scoring.schemas import (
        AGENCY_TIERS,
        agency_ceiling_for_evidence,
        detect_agency_tier,
    )

    detected = detect_agency_tier(all_text)
    if detected not in ("team_lead", "director"):
        return ""
    detected_rank = AGENCY_TIERS.index(detected)

    for key in ("experience_inventory", "project_inventory", "historical_experience_inventory"):
        for item in profile.get(key) or []:
            if not isinstance(item, dict):
                continue
            earned = agency_ceiling_for_evidence(item)
            if earned in AGENCY_TIERS and AGENCY_TIERS.index(earned) >= detected_rank:
                return ""  # at least one real evidence item earns this tier -- not inflation

    return (
        "Possible claim-strength inflation: resume uses a "
        f"{detected.replace('_', ' ')}-tier agency verb (led/managed/supervised/"
        "directed/spearheaded) but no profile evidence item's own description "
        "supports that level of people/organizational authority"
    )


# ── Factual Anchor Validation ────────────────────────────────────────────

# 2026-08-25: header parsing previously only handled "Role | Company |
# Dates" (pipe-separated), but tailor.py's actual JSON contract asks the
# LLM for "header":"Title at Company" (no pipe -- see _build_tailor_prompt's
# OUTPUT schema) -- so this check's company/title extraction never actually
# matched any real cloud-generated header and was silently inert in
# production. Both shapes are now handled: pipe-separated first (unchanged,
# still what local_tailor.py's degraded-mode base-resume parser produces),
# falling back to splitting on the first standalone " at " when no pipe is
# present.
_HEADER_AT_SPLIT_RE = re.compile(r"\s+at\s+", re.IGNORECASE)


def _split_experience_header(header: str) -> tuple[str, str] | None:
    """Best-effort (title, company) split of one experience header, or None
    if the shape isn't recognized. See the module comment above."""
    parts = [p.strip() for p in header.split("|")]
    if len(parts) >= 2:
        return parts[0], parts[1]
    match = _HEADER_AT_SPLIT_RE.search(header)
    if match:
        return header[: match.start()].strip(), header[match.end() :].strip()
    return None


# Role words that imply a step up in seniority, scope, or professional
# identity beyond what a bare job title states. Used only to flag a
# candidate title that ADDS one of these words relative to the authoritative
# title -- a tailored title that's a subset/rewording of the authoritative
# words (e.g. "Warehouse Associate" from "Packaging Associate / Warehouse
# Associate") is never flagged; this only catches an upgrade.
_ELEVATED_TITLE_WORDS_RE = re.compile(
    r"\b("
    r"senior|sr|staff|principal|lead|director|manager|head|chief|vp|"
    r"specialist|coordinator|supervisor|architect|engineer|developer|"
    r"programmer|administrator|analyst|consultant"
    r")\b",
    re.IGNORECASE,
)


def _title_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}


def validate_factual_anchors(tailored_data: dict, profile: dict) -> dict:
    """Check experience headers for unrecognized employers AND invented/
    upgraded job titles.

    Employer check: guards against local models (and occasionally cloud
    models) inventing new employers in experience section headers -- only
    active when the profile has a non-empty ``preserved_companies`` list;
    issues a WARNING (not a hard error) since a single ambiguous header
    shouldn't block an otherwise valid draft on its own.

    Title check (2026-08-25): for any header whose company matches a
    profile employer with a known authoritative title (experience_
    inventory's role_title/role_type, or historical_experience_inventory's
    role_title), the header's title must be traceable to that authoritative
    title -- either sharing its words, or being a plain subset/rewording of
    them. A title that ADDS an elevated role word (see
    _ELEVATED_TITLE_WORDS_RE) not present in the authoritative title is a
    hard ERROR, not a warning: this is exactly the deterministic,
    high-confidence "invented job title" case (e.g. UPS's authoritative
    title "Packaging Associate / Warehouse Associate" becoming "High-Volume
    Operations Specialist") that must be rejected outright, not merely
    flagged for human review.

    Unparseable-header check (2026-09-01): a header that doesn't even
    contain a recognizable "Title at Company"/"Title | Company" shape is
    MORE suspicious than an ordinary unrecognized-employer case, not less
    -- the prompt bake-off's worst single fabrication (IterateCV inventing
    a standalone "ALIGNMENT TECHNIQUES" experience entry with no employer
    name at all) evaded every check that only fires once a header DOES
    split, because it never reached that branch. Any such header is now
    checked against every known entry name and title/qualification word in
    the profile (experience_inventory, historical_experience_inventory,
    AND qualifications -- some real, legitimate entries like "Stand-up
    comedy" live only in qualifications, not experience_inventory, and
    must not be false-flagged); no match at all is a WARNING (same tier as
    the unrecognized-employer case below, for the same reason: a single
    ambiguous header shouldn't block an otherwise valid draft alone).

    Returns {"errors": list, "warnings": list}.
    """
    errors: list[str] = []
    warnings: list[str] = []
    resume_facts = profile.get("resume_facts", {})
    preserved = [c.lower().strip() for c in resume_facts.get("preserved_companies", []) if c.strip()]

    # employer name -> authoritative title, from every inventory that
    # carries one. Keyed on the lowercased employer name for substring
    # matching against a header's (possibly abbreviated) company text.
    authoritative_titles: dict[str, str] = {}
    for key in ("experience_inventory", "historical_experience_inventory"):
        for item in profile.get(key) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            title = item.get("role_title") or item.get("role_type")
            if name and title:
                authoritative_titles[name.lower()] = str(title)

    # Broader "does this bare header correspond to ANYTHING real" set --
    # employer names, their authoritative TITLES (not decomposed into
    # words -- an earlier version pooled individual words from every
    # title into one set, which let "ALIGNMENT TECHNIQUES" false-pass by
    # sharing just the single word "alignment" with the real "Alignment
    # Technician" title; whole-string substring matching doesn't have
    # that failure mode), and qualification names (some real entries,
    # like "Stand-up comedy", live only there, not in experience_
    # inventory) -- deliberately STRING containment only, no word-level
    # fuzzy matching.
    known_entry_names: set[str] = set(authoritative_titles.keys())
    known_entry_names.update(t.lower() for t in authoritative_titles.values())
    for item in profile.get("qualifications") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            known_entry_names.add(name.lower())

    if not preserved and not authoritative_titles and not known_entry_names:
        return {"errors": [], "warnings": []}

    for entry in tailored_data.get("experience") or []:
        header = (entry.get("header") or "").strip()
        split = _split_experience_header(header)
        if split is None:
            if header and known_entry_names:
                header_lower = header.lower()
                matches_known = any(name in header_lower or header_lower in name for name in known_entry_names)
                if not matches_known:
                    warnings.append(
                        f"Experience header could not be matched to any "
                        f"known employer, role, or qualification: '{header}' "
                        f"-- possible fabricated entry"
                    )
            continue
        title_part, company_part = split
        company_lower = company_part.lower()

        if preserved and company_lower and not any(co in company_lower for co in preserved):
            warnings.append(
                f"Experience header may reference unrecognized employer: "
                f"'{company_part}' "
                f"(known: {', '.join(c.title() for c in preserved[:3])})"
            )

        for known_name, authoritative_title in authoritative_titles.items():
            if known_name not in company_lower and company_lower not in known_name:
                continue
            auth_words = _title_words(authoritative_title)
            tailored_words = _title_words(title_part)
            added_elevated = {w for w in _ELEVATED_TITLE_WORDS_RE.findall(title_part.lower()) if w not in auth_words}
            if added_elevated and not tailored_words <= auth_words:
                errors.append(
                    f"Experience header for '{company_part}' uses title "
                    f"'{title_part}', which is not supported by the "
                    f"authoritative title '{authoritative_title}' -- "
                    f"invented/upgraded job title"
                )
            break  # matched a known employer; don't also check other entries

    return {"errors": errors, "warnings": warnings}


# 2026-09-01: keyword-injection fabrication check. The prompt bake-off
# found a systematic pattern (IterateCV especially) of grafting distinctive
# JOB-POSTING vocabulary onto a bullet whose employer/entry is otherwise
# correctly attributed -- e.g. "utilizing data to inform comedic timing and
# audience connection" appended to a Stand-up Comedian bullet. This is a
# genuinely harder case than the other checks: the injected WORD ("data")
# is often already legitimately in the candidate's OWN vocabulary (e.g.
# from a real data-analytics project) -- it's the CONTEXT that's fabricated,
# not the word itself. A whole-profile word-overlap check would miss this
# entirely, since "data" is grounded SOMEWHERE in the profile. The fix:
# ground each bullet against ONLY its OWN entry's evidence text (not the
# whole profile), so a word that's real for one entry but injected into an
# unrelated one is still caught. Deliberately WARNING-tier, not ERROR --
# natural-language vocabulary overlap is inherently fuzzier than the
# structural title/date/skill checks above, so this is advisory (matching
# check_claim_inflation/check_agency_inflation's precedent for coarser,
# natural-language-derived signals) rather than a hard reject.
_VOCAB_STOPWORDS: set[str] = {
    "the", "and", "with", "that", "this", "from", "have", "will", "your",
    "their", "about", "when", "where", "which", "while", "being", "been",
    "were", "into", "onto", "upon", "across", "during", "after", "before",
    "between", "through", "under", "over", "using", "used", "able", "also",
    "such", "some", "more", "most", "other", "each", "every", "both",
    "only", "just", "very", "much", "many", "less", "least", "same",
    "than", "then", "them", "they", "these", "those", "what", "who",
    "whom", "whose", "how", "why", "team", "role", "work", "years",
    "including", "strong", "ability", "experience", "skills",
    "knowledge", "environment", "opportunity", "candidate", "position",
    "company", "join", "looking", "required", "preferred", "responsible",
    "responsibilities", "requirements", "qualifications", "job",
    "description", "please", "apply", "help", "customer", "customers",
    "service", "services", "support", "provide", "provided", "ensure",
    "ensuring", "maintain", "maintained", "deliver", "delivered",
    "communication", "communicate", "communicated",
}


def _distinctive_words(text: str) -> set[str]:
    """Lowercased words of length >= 4 from ``text``, excluding generic
    English connectors/filler (_VOCAB_STOPWORDS) -- deliberately grammar-
    based, not domain/technology-specific, so it never needs updating for
    a new job posting's vocabulary the way a denylist would."""
    words = {w for w in re.findall(r"[a-z][a-z0-9+#.\-]{3,}", text.lower())}
    return words - _VOCAB_STOPWORDS


def _entry_evidence_text(item: dict) -> str:
    """Flatten one experience_inventory/historical_experience_inventory/
    qualifications item's own descriptive fields into one text blob --
    everything the candidate could truthfully draw language from for THIS
    specific entry (not the whole profile)."""
    parts = [str(item.get("description") or "")]
    for field in ("responsibilities", "evidence", "factual_concepts", "relevance_categories"):
        parts.extend(str(v) for v in item.get(field) or [])
    return " ".join(parts)


def check_bullet_vocabulary_injection(data: dict, profile: dict, job: dict | None = None) -> list[str]:
    """Flag resume bullets that use distinctive vocabulary lifted from the
    TARGET JOB POSTING but grounded NOWHERE in that SPECIFIC entry's own
    evidence -- the "keyword injection" fabrication pattern found in the
    prompt bake-off (2026-09-01). Deliberately scoped per-entry rather than
    whole-profile: a word can be genuinely grounded for one job (e.g. "data"
    for a real analytics project) and still be a fabricated graft onto an
    unrelated entry (e.g. "data" tied to stand-up comedy). Requires the job
    dict (title/full_description) -- returns [] when not provided, since
    there's nothing to compare against."""
    if not job or not (job.get("full_description") or "").strip():
        return []
    job_vocab = _distinctive_words((job.get("title") or "") + " " + (job.get("full_description") or ""))
    if not job_vocab:
        return []

    entry_evidence: dict[str, str] = {}
    for key in ("experience_inventory", "historical_experience_inventory", "qualifications", "project_inventory"):
        for item in profile.get(key) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name:
                entry_evidence[name.lower()] = _entry_evidence_text(item).lower()
    if not entry_evidence:
        return []

    def _match_evidence(header: str) -> str | None:
        split = _split_experience_header(header)
        needle = split[1].strip().lower() if split else header.strip().lower()
        if not needle:
            return None
        for name, ev_text in entry_evidence.items():
            if name in needle or needle in name:
                return ev_text
        return None

    warnings: list[str] = []
    for section in ("experience", "projects"):
        for entry in data.get(section) or []:
            if not isinstance(entry, dict):
                continue
            header = str(entry.get("header") or "").strip()
            if not header:
                continue
            matched_evidence = _match_evidence(header)
            if matched_evidence is None:
                continue
            evidence_words = _distinctive_words(matched_evidence)
            for bullet in entry.get("bullets") or []:
                bullet_words = _distinctive_words(str(bullet))
                injected = (bullet_words & job_vocab) - evidence_words
                if len(injected) >= 2:
                    warnings.append(
                        f"Bullet for '{header}' may graft job-posting "
                        f"language not grounded in this entry's own "
                        f"evidence ({', '.join(sorted(injected)[:5])}): "
                        f"'{str(bullet)[:120]}'"
                    )
    return warnings


# 2026-08-28: `subtitle` ("Tech | Dates" per experience/project entry) was
# never validated at all -- confirmed by reading validate_json_fields: it
# collects summary/bullets/header/title into checks but never touches
# subtitle. Live measurement against all 354 real generated resumes: the
# literal prompt-schema example "Tech | Dates" never leaks verbatim, but
# the LLM regularly invents its OWN placeholder text when it lacks real
# dates -- "Dates not specified" (1492 occurrences/196 files), "Dates N/A"
# (259/36), "Dates Not Specified" (146), "Dates Unspecified" (133), "Dates
# Not Provided" (18/7) -- affecting 271/354 (76%) of real output. The
# prompt now explicitly instructs omitting the date portion instead of
# inventing filler text (see tailor.py's HARD RULES); this is the
# deterministic backstop, same ERROR-tier/retry pattern as
# check_title_inflation below.
_DATE_PLACEHOLDER_RE = re.compile(r"dates?\s+(?:not\s+(?:specified|provided)|n/?a|unspecified)", re.IGNORECASE)


# 2026-09-01: check_date_placeholder_fabrication above only catches the
# ABSENCE of a real date (an invented "Dates not specified"-style
# placeholder) -- it has no way to catch a model that instead invents a
# PLAUSIBLE-LOOKING specific date range, which is more dangerous (more
# falsifiable, more likely to pass a casual read) and was found to slip
# straight past every existing check during the specialized-model prompt
# bake-off (2026-09-01). This compares any year(s) mentioned in an entry's
# subtitle against the profile's own authoritative start_date/end_date for
# that employer (experience_inventory / historical_experience_inventory)
# -- profile-driven, so it generalizes to any candidate's resume without a
# hardcoded date, and stays silent for any employer the profile has no
# dates on file for (never a false positive from missing data).
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _extract_year(date_str: str | None) -> int | None:
    """Pull a 4-digit year out of a profile.json date field via _YEAR_RE,
    rather than assuming the strict 'YYYY-MM' shape _authoritative_date_
    ranges' docstring hopes for. 2026-09-04: found via a real production
    crash -- historical_experience_inventory's "Lowe's Foods" entry has
    start_date='circa 2016'/end_date='circa 2017' (legitimate real data;
    someone genuinely doesn't remember the exact month), and the old code
    did int(start[:4]) unconditionally, raising ValueError on 'circ' and
    aborting the entire tailoring job. Returns None for anything with no
    recognizable year (never raises) -- check_date_fabrication already
    treats "no year on file" as "nothing to check," so this just extends
    that same silent-skip behavior to malformed/approximate date text."""
    if not date_str:
        return None
    match = _YEAR_RE.search(str(date_str))
    return int(match.group(1)) if match else None


def _authoritative_date_ranges(profile: dict) -> dict[str, tuple[str | None, str | None]]:
    """employer name (lowercased) -> (start_date, end_date) as 'YYYY-MM'
    strings (or None), from experience_inventory / historical_experience_
    inventory -- the same authoritative source validate_factual_anchors
    already uses for titles, extended to carry dates too."""
    ranges: dict[str, tuple[str | None, str | None]] = {}
    for key in ("experience_inventory", "historical_experience_inventory"):
        for item in profile.get(key) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name:
                ranges[name.lower()] = (item.get("start_date"), item.get("end_date"))
    return ranges


def check_date_fabrication(data: dict, profile: dict) -> list[str]:
    """Return one error per experience/project entry whose subtitle states
    a specific year that falls OUTSIDE the authoritative start/end date
    range on file for that employer -- catches an INVENTED plausible date,
    as opposed to check_date_placeholder_fabrication's invented-placeholder
    case. Silent whenever the matched employer has no dates on file (e.g.
    a test fixture, or a profile entry that never recorded them) -- there
    must be real ground truth to contradict before this flags anything."""
    errors: list[str] = []
    date_ranges = _authoritative_date_ranges(profile)
    if not date_ranges:
        return errors

    for section in ("experience", "projects"):
        entries = data.get(section)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            header = str(entry.get("header") or "")
            subtitle = str(entry.get("subtitle") or "")
            split = _split_experience_header(header)
            if split is None:
                continue
            _title_part, company_part = split
            company_lower = company_part.strip().lower()

            matched_range = None
            for known_name, rng in date_ranges.items():
                if known_name in company_lower or company_lower in known_name:
                    matched_range = rng
                    break
            if matched_range is None:
                continue

            start, end = matched_range
            start_year = _extract_year(start)
            end_year = _extract_year(end)
            allowed_years: set[int] = set()
            if start_year:
                allowed_years.add(start_year)
            if end_year:
                allowed_years.add(end_year)
            if start_year and end_year:
                allowed_years.update(range(start_year, end_year + 1))
            if not allowed_years:
                continue

            claimed_years = {int(y) for y in _YEAR_RE.findall(subtitle)}
            bad_years = claimed_years - allowed_years
            if bad_years:
                errors.append(
                    f"Fabricated date(s) in subtitle for '{header}': claims "
                    f"year(s) {sorted(bad_years)}, but the profile's "
                    f"authoritative record for this employer is "
                    f"{start or '?'} to {end or 'present'}"
                )
    return errors


def check_date_placeholder_fabrication(data: dict) -> list[str]:
    """Return one error string per experience/project entry whose
    `subtitle` contains invented placeholder date text (e.g. "Dates not
    specified") instead of either a real date or an omitted date portion.
    Empty list when nothing is flagged."""
    errors: list[str] = []
    for section in ("experience", "projects"):
        entries = data.get(section)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            subtitle = str(entry.get("subtitle") or "")
            if _DATE_PLACEHOLDER_RE.search(subtitle):
                header = str(entry.get("header") or "")[:80]
                errors.append(f"Fabricated date placeholder in subtitle for '{header}': '{subtitle}'")
    return errors


# 2026-08-25: companion to validate_factual_anchors' per-employer title
# check, for the one title field that isn't tied to any single employer --
# the top-level resume headline (data["title"]). This is exactly where the
# most severe known-bad regression case appeared ("Sr Architect - Emerging
# Technologies" for a candidate with no professional architecture
# experience): tailor.py's prompt used to instruct copying the JOB
# POSTING's title verbatim into this field with no grounding check at all.
def check_title_inflation(title: str, profile: dict) -> str:
    """Return an error string if `title` (the resume's top-level headline)
    uses an elevated role word (see _ELEVATED_TITLE_WORDS_RE) that no
    authoritative title anywhere in the candidate's profile supports --
    empty string when there's nothing to flag. Unlike the coarser claim/
    agency inflation warnings, this is deliberately an ERROR: an invented
    professional title/seniority claim in the resume's own headline is
    exactly the class of fabrication the profile-authority principle exists
    to prevent, not an advisory style note."""
    if not title:
        return ""
    auth_words: set[str] = set()
    for key in ("experience_inventory", "historical_experience_inventory"):
        for item in profile.get(key) or []:
            if not isinstance(item, dict):
                continue
            t = item.get("role_title") or item.get("role_type")
            if t:
                auth_words |= _title_words(str(t))

    if not auth_words:
        # No authoritative titles anywhere in the profile to check against
        # -- nothing to ground a judgment on, so this stays silent rather
        # than flagging every title that happens to contain a role word
        # (e.g. a bare "Engineer" title in a profile/test fixture with no
        # experience_inventory at all). Matches validate_factual_anchors'
        # same early-exit when there's no preserved-company data.
        return ""

    added_elevated = {w for w in _ELEVATED_TITLE_WORDS_RE.findall(title.lower()) if w not in auth_words}
    if not added_elevated:
        return ""
    return (
        f"Resume title '{title}' uses role word(s) "
        f"({', '.join(sorted(added_elevated))}) not supported by any of the "
        f"candidate's authoritative job titles -- invented/upgraded "
        f"professional title"
    )


# ── Full Resume Text Validation ───────────────────────────────────────────


def validate_tailored_resume(
    text: str,
    profile: dict,
    original_text: str = "",
    standup_decision: str | None = None,
) -> dict:
    """Programmatic validation of a tailored resume against the user's profile.

    Args:
        text: The tailored resume text to validate.
        profile: User profile dict from load_profile().
        original_text: The original base resume text (for fabrication comparison).

    Returns:
        {"passed": bool, "errors": list[str], "warnings": list[str]}
    """
    errors: list[str] = []
    warnings: list[str] = []
    text_lower = text.lower()

    personal = profile.get("personal", {})
    resume_facts = profile.get("resume_facts", {})
    allowed_skills = _build_skills_set(profile)

    # 1. Check required sections exist (flexible matching)
    section_variants: dict[str, list[str]] = {
        "SUMMARY": ["summary", "professional summary", "profile"],
        "TECHNICAL SKILLS": ["technical skills", "skills", "tech stack", "core skills", "technologies"],
        "EXPERIENCE": ["experience", "work experience", "professional experience"],
        "PROJECTS": ["projects", "personal projects", "key projects", "selected projects"],
        "EDUCATION": ["education", "academic background"],
    }
    for section, variants in section_variants.items():
        if not any(v in text_lower for v in variants):
            errors.append(f"Missing required section: {section} (or variant)")

    # 2. Check name preserved (warn, don't error -- we can inject it)
    full_name = personal.get("full_name", "")
    if full_name and full_name.lower() not in text_lower:
        warnings.append(f"Name '{full_name}' missing -- will be injected")

    # 3. Check companies preserved (warning, not error — 1-page resumes may drop early-career roles)
    for company in resume_facts.get("preserved_companies", []):
        if company.lower() not in text_lower:
            warnings.append(f"Company '{company}' not in resume (may be omitted for space)")

    # 4. Check projects preserved
    for project in resume_facts.get("preserved_projects", []):
        if project.lower() not in text_lower:
            warnings.append(f"Project '{project}' not found -- may have been renamed")

    # 5. Check school preserved (each school checked individually so structured
    # multi-school education sections validate)
    preserved_school = resume_facts.get("preserved_school", "")
    if preserved_school:
        missing = _missing_schools(preserved_school, text)
        if missing:
            errors.append(f"Education missing school(s): {', '.join(missing)}")

    # 6. Check contact info preserved (warn, don't error -- we can inject)
    email = personal.get("email", "")
    phone = personal.get("phone", "")
    if email and email.lower() not in text_lower:
        warnings.append("Email missing -- will be injected")
    if phone and phone not in text:
        warnings.append("Phone missing -- will be injected")

    # 7. Scan TECHNICAL SKILLS section for fabricated tools
    skills_start = text_lower.find("technical skills")
    skills_end = text_lower.find("experience", skills_start) if skills_start != -1 else -1
    if skills_start != -1 and skills_end != -1:
        skills_block = text_lower[skills_start:skills_end]
        for fake in FABRICATION_WATCHLIST:
            if len(fake) <= 2:
                continue
            if any(fake in skill for skill in allowed_skills):
                continue
            if fake in skills_block:
                errors.append(f"FABRICATED SKILL in Technical Skills: '{fake}'")

    # 8. Scan full document for fabrication watchlist items not in original
    if original_text:
        original_lower = original_text.lower()
        for fake in FABRICATION_WATCHLIST:
            if len(fake) <= 2:
                continue
            if fake in text_lower and fake not in original_lower:
                warnings.append(f"New tool/skill appeared: '{fake}' (not in original)")

    # 9. Em dashes (should be auto-fixed by sanitize_text, but safety net)
    if "\u2014" in text or "\u2013" in text:
        errors.append("Contains em dash or en dash.")

    # 10. Banned words (style warning, not hard error — judge layer evaluates tone)
    found_banned = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", text_lower)]
    if found_banned:
        warnings.append(f"Banned words (style): {', '.join(found_banned[:5])}")

    # 11. LLM self-talk leak detection
    found_leaks = [p for p in LLM_LEAK_PHRASES if p in text_lower]
    if found_leaks:
        errors.append(f"LLM self-talk: '{found_leaks[0]}'")

    _append_profile_integrity_errors(text, profile, errors)

    if standup_decision == "EXCLUDE" and re.search(r"\b(stand[- ]?up|comed(y|ian)|open mic|improv)\b", text_lower):
        errors.append("Stand-up content present while decision is EXCLUDE")

    # 12. Duplicate section detection
    for section_name in ["summary", "experience", "education", "projects"]:
        count = text_lower.count(f"\n{section_name}\n") + text_lower.count(f"\n{section_name} \n")
        if text_lower.startswith(f"{section_name}\n"):
            count += 1
        if count > 1:
            errors.append(f"Section '{section_name}' appears {count} times.")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ── Cover Letter Validation ──────────────────────────────────────────────


def validate_cover_letter(text: str, profile: dict | None = None) -> dict:
    """Programmatic validation of a cover letter.

    Args:
        text: The cover letter text to validate.
        profile: Optional user profile dict -- when given, also runs the
            same profile-integrity checks the resume validator applies
            (private/ApplyPilot project leakage, unfinished-project
            success claims, degree/field consistency). 2026-08-25:
            previously this function took no profile at all, so a cover
            letter had NO backstop against exactly the leaks the resume
            path already guarded against -- "ApplyPilot" (private,
            resume_allowed=False) could appear in a cover letter with
            nothing to catch it. Optional and defaulted to preserve every
            existing call site.

    Returns:
        {"passed": bool, "errors": list[str]}
    """
    errors: list[str] = []
    warnings: list[str] = []
    text_lower = text.lower()

    if profile:
        _append_profile_integrity_errors(text, profile, errors)

    # 1. Em dashes
    if "\u2014" in text or "\u2013" in text:
        errors.append("Contains em dash or en dash.")

    # 2. Banned words (style warning, not hard error — judge layer evaluates tone)
    found = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", text_lower)]
    if found:
        warnings.append(f"Banned words (style): {', '.join(found[:5])}")

    # 2b. Hard-reject phrase patterns (error tier — these force a retry).
    hits = [label for label, pat in CL_BANNED_PATTERNS if re.search(pat, text_lower)]
    if hits:
        errors.append(f"Banned phrase(s): {', '.join(hits)}")

    # 3. Word count policy: hard minimum + target window + hard maximum.
    words = len(text.split())
    if words > _CL_HARD_MAX_WORDS:
        errors.append(
            f"Too long ({words} words). "
            f"Target {_CL_TARGET_MIN_WORDS}-{_CL_TARGET_MAX_WORDS} words; "
            f"maximum {_CL_HARD_MAX_WORDS}."
        )
    if words < _CL_MIN_WORDS:
        errors.append(
            f"Too short ({words} words). "
            f"Minimum {_CL_MIN_WORDS} words; "
            f"target {_CL_TARGET_MIN_WORDS}-{_CL_TARGET_MAX_WORDS}."
        )
    elif words < _CL_TARGET_MIN_WORDS:
        warnings.append(f"Below target ({words} words). Target {_CL_TARGET_MIN_WORDS}-{_CL_TARGET_MAX_WORDS} words.")
    elif words > _CL_TARGET_MAX_WORDS:
        warnings.append(f"Above target ({words} words). Target {_CL_TARGET_MIN_WORDS}-{_CL_TARGET_MAX_WORDS} words.")

    # 4. LLM self-talk
    found_leaks = [p for p in LLM_LEAK_PHRASES if p in text_lower]
    if found_leaks:
        errors.append(f"LLM self-talk: '{found_leaks[0]}'")

    # 5. Must start with "Dear"
    stripped = text.strip()
    if not stripped.lower().startswith("dear"):
        errors.append("Must start with 'Dear Hiring Manager,'")

    # 6. Structure: requires substantial body paragraphs (hook, evidence,
    # company fit, close). "Substantial" = >= 15 words.
    body_paragraphs = [p for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 15]
    if len(body_paragraphs) < _CL_REQUIRED_PARAGRAPHS:
        errors.append(
            f"Only {len(body_paragraphs)} body paragraph(s); structure requires "
            f"{_CL_REQUIRED_PARAGRAPHS} (hook, evidence, company fit, close)."
        )
    elif len(body_paragraphs) > _CL_REQUIRED_PARAGRAPHS:
        warnings.append(f"{len(body_paragraphs)} body paragraphs; target structure is {_CL_REQUIRED_PARAGRAPHS}.")

    return {"passed": len(errors) == 0, "errors": errors, "warnings": warnings}
