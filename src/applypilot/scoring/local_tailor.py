"""Local LLM support: a cheap, structured resume/job matching PLANNER.

The local model's job is narrow and deliberate, and deliberately NOT
generative: given a numbered list of job requirements and a numbered list of
already-vetted candidate evidence (both built with zero LLM calls), it picks
which evidence numbers (if any) support each requirement number. It never
writes prose, never restates text, and structurally cannot fabricate a
skill/employer/tool -- its only possible outputs are references into two
lists that were assembled deterministically before it ever ran. Everything
that WAS previously asked of the model in free text (skills_to_emphasize,
keyword_targets, matching_projects/certifications, summary_focus) is now
derived in code from which evidence numbers got matched; anything that
still requires real prose generation (bullet rewrites, resume bullet
prioritization) is left to the cloud model, which has the full resume in
hand and is far more reliable at freeform generation than a CPU-only 1-2B
local model. See validate_local_plan() for where that derivation happens.

Public API
----------
  rank_profile_evidence(job, profile, top_n=6) -> list[dict]
      Deterministic (no LLM, no embeddings) retrieval: ranks experience_
      inventory/project_inventory/skills_inventory/certifications entries
      against the job description by term overlap with their existing
      relevance_categories/factual_concepts/name fields. Reduces the
      problem handed to the local model instead of a flat resume dump.
      Each result records WHY it matched (matched_terms) so the retrieval
      is inspectable, not a black box -- see the `applypilot
      debug-local-plan` command. List order is also each item's E-number
      in the local model's prompt (see format_evidence_for_prompt).

  format_evidence_for_prompt(ranked) -> str
      Renders ranked evidence (numbered E1, E2, ... plus each item's
      existing hand-written `constraints`) as compact text for the local
      model's prompt.

  get_local_tailoring_plan(resume_text, job, profile) -> dict | None
      Deterministically extracts requirement lines (see
      _extract_requirement_lines / _is_benefit_line -- employer-benefit
      lines like PTO/401(k)/tuition assistance/career-growth opportunities
      are filtered out here, before the model ever sees them, so they can
      never be "matched" to candidate evidence) and retrieves
      candidate evidence (rank_profile_evidence), asks the local model only
      to match requirement numbers to evidence numbers, then runs that
      through validate_local_plan() -- which does the actual field
      derivation, not fabrication-checking, since closed-set number
      references can't be ungrounded. Returns None if the local model is
      unreachable, the response is empty, or the output can't be parsed as
      JSON at all. Skips the LLM call entirely (returns an empty-but-valid
      plan) when there are no requirement lines or no evidence to match
      against -- there is nothing for the model to usefully decide.

  format_local_plan_for_cloud(plan) -> str
      Renders a validated plan as compact text for the cloud model's user
      message.  Returns "" for an empty/all-dropped plan so callers can use
      `if rendered:` to decide whether to include a "TAILORING GUIDANCE"
      block at all.

  validate_local_plan(raw_plan, requirement_lines, ranked_evidence) -> dict
      Deterministic, code-only (no LLM call). Reads the model's `matches`
      (requirement number -> evidence numbers), drops any reference outside
      the given lists' bounds, and builds the full structured plan dict
      from that: per-requirement supported/resume_evidence, plus
      skills_to_emphasize/matching_projects/matching_certifications/
      keyword_targets/summary_focus/unsupported_requirements, all derived
      from which evidence items got matched -- not asked of the model as
      free text. bullets_to_prioritize/bullets_to_deemphasize/safe_rewrites
      are always empty here (real generation tasks, left to the cloud
      model); the keys are kept for output-shape compatibility. Never
      raises; malformed matches are dropped and noted in "_warnings".

  debug_plan_for_job(job, profile) -> dict | None
      Backing function for the `applypilot debug-local-plan` CLI command:
      resolves the resume text the same way the real tailor pipeline does,
      then returns {"plan": ..., "evidence": ..., "requirement_lines": ...}
      -- the validated plan plus the deterministic retrieval trace -- WITHOUT
      generating or modifying any resume. None if local planning failed
      entirely (unreachable model, unparseable response).

  _build_compact_local_prompt(profile) -> str
      DEGRADED-MODE full-resume-generation prompt for small local models.
      Only used by tailor.py when every cloud model is exhausted -- an
      emergency fallback, not a substitute for normal cloud finalization.
      See LOCAL_FULL_GENERATION_IS_DEGRADED below.

Neither capability runs automatically:
  - Set APPLYPILOT_LOCAL_LLM_URL to enable the local model at all (both as a
    planner and as the last-resort fallback entry in the cloud chain).
  - Use ``applypilot run tailor --local-first`` (or APPLYPILOT_LOCAL_PLAN=1)
    to enable the planning first-pass before cloud tailoring.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

log = logging.getLogger(__name__)

_DEFAULT_LOCAL_URL = "http://localhost:11434"
_DEFAULT_LOCAL_MODEL = "llama3.2"

# The compact-prompt full-generation path (see _build_compact_local_prompt)
# is an emergency degraded mode: it's only reached when every cloud tailoring
# model is exhausted AND a local model is configured. It is NOT equivalent to
# normal cloud finalization -- callers should log a visible warning when they
# take this branch (see scoring/tailor.py's tailor_resume()).
LOCAL_FULL_GENERATION_IS_DEGRADED = True


# ---------------------------------------------------------------------------
# Connectivity probe
# ---------------------------------------------------------------------------

def is_local_available() -> bool:
    """Return True if the configured local LLM endpoint responds.

    Thin wrapper around applypilot.llm.local_available() so connectivity
    logic isn't duplicated across the two modules.
    """
    from applypilot.llm import local_available
    return local_available()


# ---------------------------------------------------------------------------
# Compact system prompt for DEGRADED-MODE full-resume generation
# ---------------------------------------------------------------------------

def _build_compact_local_prompt(profile: dict) -> str:
    """Return a compact (<600-token) system prompt suitable for small local models.

    DEGRADED MODE ONLY (see LOCAL_FULL_GENERATION_IS_DEGRADED). The full cloud
    prompt is 3 000+ tokens; local 7B models produce poor JSON output with
    prompts that long, so this version keeps only the structural contract and
    the factual constraints that prevent hallucination.
    """
    from applypilot.scoring.validator import _build_skills_set  # lazy to avoid circular import

    resume_facts = profile.get("resume_facts", {})
    companies = (
        ", ".join(resume_facts.get("preserved_companies", []))
        or "same companies as in the original resume"
    )
    school = resume_facts.get("preserved_school", "keep original exactly")
    metrics_list = resume_facts.get("real_metrics", [])
    metrics = (
        ", ".join(metrics_list[:6]) if metrics_list
        else "only numbers already in the resume; do not invent any"
    )
    allowed = sorted(_build_skills_set(profile))[:30]
    skills_str = ", ".join(allowed) if allowed else "only skills already in the resume"
    full_name = (profile.get("personal") or {}).get("full_name", "")

    return (
        "You are a resume tailoring assistant. Return ONLY valid JSON \u2014 no preamble, "
        "no markdown fences, no explanation.\n\n"
        "ABSOLUTE RULES \u2014 violation causes immediate rejection:\n"
        f"- NEVER invent employer names, dates, degrees, or certifications\n"
        f"- NEVER add technologies not in this list: {skills_str}\n"
        f"- NEVER change real numbers; real metrics are only: {metrics}\n"
        f"- Keep these companies verbatim in experience headers: {companies}\n"
        f"- Keep education verbatim: {school}\n"
        f"- The name at the top must be: {full_name}\n\n"
        "ALLOWED: reorder bullets, rewrite wording, change role title to match job, "
        "rewrite summary from scratch, drop low-relevance bullets.\n\n"
        "OUTPUT SCHEMA (return exactly this structure, no extra keys):\n"
        '{\n'
        '  "title": "<role title variant matching the job>",\n'
        '  "summary": "<2-3 sentences specific to this job>",\n'
        '  "skills": {"<category>": "<comma-separated items>"},\n'
        '  "experience": [{"header": "<Role | Company | Dates>", '
        '"subtitle": "<optional or empty string>", "bullets": ["<bullet>"]}],\n'
        '  "projects": [{"header": "<Project Name>", '
        '"subtitle": "<optional or empty string>", "bullets": ["<bullet>"]}],\n'
        '  "education": "<keep original text, or structured list>"\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Deterministic job-signal extraction and evidence retrieval
#
# No LLM call, no embedding model. profile.json's experience_inventory /
# project_inventory / skills_inventory are already hand-tagged with
# relevance_categories (and, for projects, factual_concepts) -- an
# already-vetted vocabulary that's a better retrieval substrate for this
# project than either (a) re-deriving keywords from free job-description
# text with NLP heuristics, or (b) adding a real embedding dependency. This
# reduces the problem the local model sees to a curated, small evidence set
# instead of a flat, truncated resume dump.
# ---------------------------------------------------------------------------

_REQUIREMENT_MARKER_RE = re.compile(r"^\s*(?:[-*•‣▪]|\d+[.)])\s+(.+)$", re.MULTILINE)
_PREFERRED_HINT_RE = re.compile(r"\b(preferred|nice[- ]to[- ]have|bonus|a\s+plus)\b", re.IGNORECASE)
_REQUIRED_HINT_RE = re.compile(r"\b(required|must\s+have|minimum\s+qualif|required\s+qualif)\b", re.IGNORECASE)
_TERM_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.#/_-]*")

# Employer-offered benefits/perks/compensation lines are not candidate
# qualifications and must never reach the local model as a "requirement" to
# reason about -- deterministic preprocessing so a CPU-only small model
# doesn't have to (re-)learn that PTO/401(k)/insurance aren't skills. It
# gets this wrong in exactly the way that matters: shown "Tuition
# Educational Assistance Programs" as a requirement, a small model happily
# marks it "supported" because the candidate once worked somewhere that
# offered tuition assistance -- turning an employer's offer into a claimed
# candidate qualification. That distinction is decided here, in code,
# before the model can blur it; it is not something to ask the model to
# reason better about.
#
# Matching is two-tiered rather than a flat phrase list, because employers
# phrase the same perk a dozen ways ("Tuition Reimbursement", "Tuition
# Educational Assistance Programs", "Educational Assistance"):
#
#   _BENEFIT_LINE_RE   -- phrases that are unambiguous on their own. No
#       employer asks a candidate to bring their own 401(k) or FSA.
#   _BENEFIT_TOPIC_RE + _BENEFIT_FRAME_RE -- a benefit SUBJECT ("health",
#       "career", "stock", "wages") appearing alongside benefit FRAMING
#       ("programs", "opportunities", "assistance", "employer
#       contributions"). Neither half drops a line by itself: "health",
#       "education" and "career" are ordinary words inside real
#       qualifications, and so are "plans" and "programs".
#   _CANDIDATE_SIGNAL_RE -- vetoes the two-part rule when the line is
#       phrased as something asked OF the candidate ("experience with
#       health insurance claims", "knowledge of retirement plan
#       administration"). Not applied to the unambiguous tier.
#
# This is a filter over an already-narrow set (bullet/numbered lines), so a
# false positive here just drops one line, not a whole section.
_BENEFIT_LINE_RE = re.compile(
    r"("
    r"\bpaid\s+(?:time\s+off|holidays?|sick\s+(?:leave|time)|parental\s+leave|"
    r"vacation|maternity\s+leave|paternity\s+leave)\b|"
    r"\bpto\b|\b401\s?\(?k\)?\b|\bhsa\b|\bfsa\b|\bespp\b|"
    r"\bflexible\s+spending\b|"
    r"\b(?:employee\s+)?stock\s+(?:purchase|option|award|grant)s?\b|"
    r"\bequity\s+(?:grants?|awards?|package)\b|"
    r"\b(?:medical|health|dental|vision)\s*[,/&]\s*(?:medical|health|dental|vision)\b|"
    r"\b(?:life|disability|medical|dental|vision)\s+insurance\b|"
    r"\b(?:tuition|education|educational)\s+(?:\w+\s+){0,2}?"
    r"(?:reimbursement|assistance|benefits?)\b|"
    r"\bemployer\s+(?:contribution|match)\w*\b|\bcompany\s+match\b|"
    r"\bcompetitive\s+(?:salary|salaries|pay|wages?|compensation|benefits?)\b|"
    r"\b(?:generous|comprehensive|robust|excellent)\s+(?:benefits?|pto|"
    r"paid\s+time\s+off|compensation)\b|"
    r"\b(?:benefits?|compensation|total\s+rewards?)\s+package\b|"
    r"\bemployee\s+discounts?\b|\bgym\s+membership\b|\bcommuter\s+benefits?\b|"
    r"\bprofessional\s+development\s+(?:budget|stipend|allowance|fund)\b"
    r")", re.IGNORECASE,
)

# Benefit SUBJECTS. Deliberately broad -- on its own this matches plenty of
# legitimate requirements, which is why it is never used alone. Some
# categories live here rather than in the unambiguous tier on purpose:
# "retirement plan" is a benefit on a perks list but a subject-matter
# requirement in "knowledge of retirement plan administration", and only
# this tier is subject to the _CANDIDATE_SIGNAL_RE veto.
_BENEFIT_TOPIC_RE = re.compile(
    r"\b("
    r"health|healthcare|health\s?care|wellness|well-?being|medical|dental|vision|"
    r"insurance|coverage|tuition|education(?:al)?|career|advancement|"
    r"professional\s+development|retirement|pension|401k?|stock|equity|"
    r"compensation|salary|salaries|wages?|pay|payroll|bonus(?:es)?|"
    r"time\s+off|holidays?|vacation|leave|discounts?|perks?|benefits?|rewards?"
    r")\b", re.IGNORECASE,
)

# Benefit FRAMING -- the words employers use when OFFERING something rather
# than asking for it.
_BENEFIT_FRAME_RE = re.compile(
    r"\b("
    r"programs?|plans?|packages?|benefits?|perks?|offerings?|opportunit\w+|"
    r"assistance|reimbursement|insurance|coverage|contributions?|match(?:es|ing)?|"
    r"discounts?|stipends?|allowances?|savings|purchase|bonus(?:es)?|eligib\w+|"
    r"enrollment|competitive|generous|comprehensive|paid|"
    r"employer[-\s]paid|company[-\s]paid"
    r")\b", re.IGNORECASE,
)

# Phrasing that marks a line as something asked OF the candidate. Only
# vetoes the two-part topic+frame rule, so genuine qualifications that
# happen to be ABOUT benefits ("experience administering 401(k) plans" is
# still dropped, "experience with health insurance claims" is not) stay in.
_CANDIDATE_SIGNAL_RE = re.compile(
    r"\b("
    r"years?\s+of\s+experience|experience\s+(?:with|in|using|administering|"
    r"supporting|managing)|abilit(?:y|ies)\s+to|able\s+to|proficien\w+|"
    r"knowledge\s+of|familiar(?:ity)?\s+with|degree\s+in|certif\w+\s+in|"
    r"background\s+in|understanding\s+of|skilled\s+in|expertise\s+in|"
    r"responsible\s+for|must\s+(?:have|be|possess)|demonstrated\s+\w+|"
    r"track\s+record\s+of"
    r")\b", re.IGNORECASE,
)


def _is_benefit_line(text: str) -> bool:
    """True if `text` reads as something the employer OFFERS (a benefit,
    perk, or compensation item) rather than something it wants FROM the
    candidate. See the comment block above for the two tiers."""
    if _BENEFIT_LINE_RE.search(text):
        return True
    if _CANDIDATE_SIGNAL_RE.search(text):
        return False
    return bool(_BENEFIT_TOPIC_RE.search(text) and _BENEFIT_FRAME_RE.search(text))


def _split_requirement_lines(
    description: str, max_lines: int = 8,
) -> tuple[list[dict], list[str]]:
    """Same extraction as _extract_requirement_lines, but also returns the
    benefit/perk lines that were dropped.

    Callers that report on their own behaviour (get_local_tailoring_plan,
    which notes WHY it skipped the model) need to distinguish "this posting
    had no bullet lines at all" from "every bullet line was an employer
    benefit" -- the second is the interesting one.
    """
    if not description:
        return [], []
    lines: list[dict] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for match in _REQUIREMENT_MARKER_RE.finditer(description):
        text = match.group(1).strip()
        if not text or len(text) < 8 or len(text) > 220:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        if _is_benefit_line(text):
            dropped.append(text)
            continue
        if _PREFERRED_HINT_RE.search(text):
            importance = "preferred"
        elif _REQUIRED_HINT_RE.search(text):
            importance = "required"
        else:
            importance = "unspecified"
        lines.append({"text": text, "importance": importance})
        if len(lines) >= max_lines:
            break
    return lines, dropped


def _extract_requirement_lines(description: str, max_lines: int = 8) -> list[dict]:
    """Pull bullet/numbered lines from the job description and tag each as
    required/preferred/unspecified via simple keyword sniffing.

    Pure text processing -- no LLM call. Gives the local model (and the
    debug command) a pre-highlighted starting point instead of re-deriving
    requirements from a wall of text every time; it's a hint the model can
    still disagree with, not a hard constraint. Lines that read as employer
    benefits/perks/compensation (see _is_benefit_line) are dropped here --
    they aren't candidate requirements at all, so there's no reason to spend
    local-model reasoning re-discovering that on every job (and every reason
    not to: a small model shown a benefit as a "requirement" tends to mark
    it supported by association with a past employer).
    """
    return _split_requirement_lines(description, max_lines=max_lines)[0]


def _format_requirement_lines(lines: list[dict]) -> str:
    if not lines:
        return ""
    return "\n".join(f"R{i} [{l['importance']}] {l['text']}" for i, l in enumerate(lines, start=1))


def _normalize_term(term: str) -> str:
    return term.strip().lower()


## Generic connector words pulled from multi-word item names (e.g. "National
## Tire and Battery") that would otherwise trivially match almost any job
## description via the word "and"/"the"/etc. Only filters the individual-
## word split of `name` -- relevance_categories/factual_concepts are
## already curated signal, not filtered.
_NAME_TOKEN_STOPWORDS = frozenset({
    "and", "the", "for", "with", "from", "into", "onto", "your", "you",
    "our", "their", "his", "her", "its", "this", "that", "these", "those",
    "are", "was", "were", "has", "have", "had", "not", "but", "can",
})


def _item_terms(item: dict) -> set[str]:
    """Collect the matchable normalized terms already curated on one
    experience_inventory / project_inventory / skills_inventory entry."""
    terms: set[str] = set()
    name = item.get("name")
    if isinstance(name, str) and name.strip():
        terms.add(_normalize_term(name))
        for w in _TERM_WORD_RE.findall(name):
            wl = _normalize_term(w)
            if len(wl) >= 3 and wl not in _NAME_TOKEN_STOPWORDS:
                terms.add(wl)
    for cat in item.get("relevance_categories") or []:
        if isinstance(cat, str) and cat.strip():
            terms.add(_normalize_term(cat))
    for concept in item.get("factual_concepts") or []:
        if isinstance(concept, str) and concept.strip():
            terms.add(_normalize_term(concept))
    return {t for t in terms if t}


def _term_in_text(term: str, haystack_lower: str) -> bool:
    """Word-boundary containment check -- cheap and deterministic, not fuzzy."""
    if not term or len(term) < 2:
        return False
    return re.search(rf"\b{re.escape(term)}\b", haystack_lower) is not None


def _job_text_lower(job: dict) -> str:
    return f"{job.get('title', '')}\n{job.get('full_description') or ''}".lower()


def rank_profile_evidence(job: dict, profile: dict, top_n: int = 6) -> list[dict]:
    """Rank experience_inventory/project_inventory/skills_inventory/
    certifications items against the job description via deterministic term
    overlap.

    Items marked resume_allowed=False (private/unfinished) are excluded --
    same rule validator.py already applies elsewhere for these inventories.
    Only items with at least one matched term are returned, sorted by score
    (matched-term count) descending, ties broken by inventory order.

    Returns entries shaped for both prompt-building and the debug CLI:
      {"type": "experience" | "project" | "skill" | "certification",
       "name": str, "score": int, "matched_terms": list[str], "item": dict}
    `matched_terms` is exactly what makes this inspectable -- the specific
    job-description terms/categories that caused the match, not a black box.
    List order also fixes each item's E-number in the local model's prompt
    (see format_evidence_for_prompt) -- the model only ever cites these
    numbers back, never free text, so this order IS the grounding contract.
    """
    haystack = _job_text_lower(job)
    sources = (
        ("experience", profile.get("experience_inventory") or []),
        ("project", profile.get("project_inventory") or []),
        ("skill", profile.get("skills_inventory") or []),
        ("certification", profile.get("certifications") or []),
    )
    ranked: list[dict] = []
    for kind, items in sources:
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or item.get("resume_allowed") is False:
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            matched = sorted(t for t in _item_terms(item) if _term_in_text(t, haystack))
            if not matched:
                continue
            ranked.append({
                "type": kind, "name": name, "score": len(matched),
                "matched_terms": matched, "item": item,
            })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]


def format_evidence_for_prompt(ranked: list[dict]) -> str:
    """Render ranked evidence as compact, numbered (E1, E2, ...) text for
    the local model's prompt. The number is the ONLY thing the model may
    cite back (see _PLAN_SYSTEM) -- free-text restatement is not part of
    its output anymore, so there's no need to give it prose to echo.

    Includes each item's existing per-item `constraints` (project_inventory
    already carries hand-written anti-fabrication notes, e.g. "do not claim
    production deployment without explicit evidence") so the local model
    sees the same factual guardrails a human reviewer already wrote.
    """
    if not ranked:
        return ""
    lines: list[str] = []
    for i, r in enumerate(ranked, start=1):
        item, kind, name = r["item"], r["type"], r["name"]
        idx = f"E{i}"
        if kind == "skill":
            level = item.get("evidence_level", "")
            lines.append(f"{idx} [skill] {name}" + (f" ({level})" if level else ""))
        elif kind == "project":
            concepts = ", ".join(c for c in (item.get("factual_concepts") or []) if isinstance(c, str))
            lines.append(f"{idx} [project] {name}" + (f": {concepts}" if concepts else ""))
            for c in item.get("constraints") or []:
                if isinstance(c, str) and c.strip():
                    lines.append(f"    constraint: {c.strip()}")
        elif kind == "certification":
            lines.append(f"{idx} [certification] {name}")
        else:  # experience
            desc = (item.get("description") or "")[:200]
            role_type = item.get("role_type", "")
            header = f"{idx} [experience] {name}" + (f" ({role_type})" if role_type else "")
            lines.append(header + (f": {desc}" if desc else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured tailoring PLAN (the normal, recommended local-model role)
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You match a job's REQUIREMENTS against a candidate's EVIDENCE. Both "
    "lists are pre-built and numbered; you only ever refer to them by "
    "number. You never write sentences, never restate their text, and "
    "never explain your answer -- you only output number references.\n\n"
    "REQUIREMENTS are numbered R1, R2, ... -- lines pulled from the job "
    "posting. Employer benefits/perks/pay are already excluded from this "
    "list; every requirement here is something the employer wants FROM the "
    "candidate.\n"
    "EVIDENCE is numbered E1, E2, ... -- things the candidate has actually "
    "done, holds, or knows. It was already verified; treat it as fact, not "
    "as something to second-guess.\n\n"
    "TASK: for each requirement number, decide which evidence numbers (zero "
    "or more) show the candidate satisfies THAT SPECIFIC requirement. Being "
    "topically related is not enough -- the evidence must actually cover "
    "the requirement's subject matter, not just share a job title or "
    "industry with it. If nothing in EVIDENCE supports a requirement, give "
    "it an empty list; do not guess or stretch a match.\n\n"
    "Output ONLY this JSON shape, nothing else -- no markdown fences, no "
    "prose before or after it:\n"
    '{"matches": [{"r": 1, "e": [2, 5]}, {"r": 2, "e": []}]}\n'
    "Include exactly one entry per requirement number you were shown, in "
    "order. Never cite an evidence number that wasn't listed."
)


def get_local_tailoring_plan(
    resume_text: str,
    job: dict,
    profile: dict,
) -> dict | None:
    """Ask the local model to match job requirements to candidate evidence,
    then deterministically build and return the full plan (see
    validate_local_plan). Returns None if the local model is unavailable,
    the response is empty, or it isn't parseable as JSON at all. A plan
    that IS parseable but matched nothing still comes back as a dict with
    empty list fields -- callers should treat that the same as "no useful
    guidance" via format_local_plan_for_cloud().

    `resume_text` is accepted for interface/logging parity with callers
    (tailor.py, debug_plan_for_job) but is not sent to the model or used
    for grounding: grounding now comes entirely from rank_profile_evidence's
    already-vetted profile data, referenced by number, which the model
    cannot fabricate around (see the module docstring).

    Deterministic preprocessing does the rest of the work up front:
    _extract_requirement_lines pulls bullet/numbered lines from the
    description and drops employer-benefit lines before the model ever
    sees them; rank_profile_evidence retrieves up to `top_n` (default 6,
    override via APPLYPILOT_LOCAL_EVIDENCE_TOPN) already-vetted profile
    items. The model's only job is picking which evidence numbers support
    each requirement number -- if either list is empty there is nothing to
    match, so the local model isn't called at all.
    """
    # Avoid circular import
    from applypilot.scoring.tailor import display_company

    top_n = int(os.environ.get("APPLYPILOT_LOCAL_EVIDENCE_TOPN", "6"))
    ranked_evidence = rank_profile_evidence(job, profile, top_n=top_n)
    requirement_lines, dropped_benefits = _split_requirement_lines(
        job.get("full_description") or ""
    )

    if not requirement_lines or not ranked_evidence:
        reason = (
            f"Skipped local LLM call: {len(requirement_lines)} requirement line(s), "
            f"{len(ranked_evidence)} evidence item(s) -- nothing to match."
        )
        if not requirement_lines and dropped_benefits:
            reason += (
                f" All {len(dropped_benefits)} bullet line(s) in this posting were "
                "employer benefits/perks, not candidate requirements: "
                + "; ".join(dropped_benefits[:6])
            )
        log.debug("Local tailoring plan for %s: %s",
                  (job.get("title") or "")[:40], reason)
        plan = validate_local_plan({"matches": []}, requirement_lines, ranked_evidence)
        plan["_warnings"].append(reason)
        return plan

    url = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", _DEFAULT_LOCAL_URL).rstrip("/")
    model = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", _DEFAULT_LOCAL_MODEL)
    timeout = float(os.environ.get("APPLYPILOT_LOCAL_LLM_TIMEOUT", "60"))

    company = display_company(job)
    job_text = (
        f"TITLE: {job.get('title', '')}\n"
        f"COMPANY: {company or 'unknown'}"
    )
    evidence_text = format_evidence_for_prompt(ranked_evidence)
    requirements_text = _format_requirement_lines(requirement_lines)

    user_msg = (
        f"JOB:\n{job_text}\n\n"
        f"REQUIREMENTS:\n{requirements_text}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "Return the JSON match list now:"
    )

    # Native Ollama API disables reasoning explicitly with think=false.
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "think": False,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": int(os.environ.get("APPLYPILOT_LOCAL_LLM_MAX_TOKENS", "300")),
        },
    }

    try:
        resp = httpx.post(f"{url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            log.warning("Local tailoring plan for %s: empty message content in response",
                        (job.get("title") or "")[:40])
            return None
        raw_plan = _parse_plan(text)
        if not isinstance(raw_plan, dict):
            log.warning("Local tailoring plan for %s: parsed JSON was not an object (got %s)",
                        (job.get("title") or "")[:40], type(raw_plan).__name__)
            return None
        sanitized = validate_local_plan(raw_plan, requirement_lines, ranked_evidence)
        log.debug(
            "Local tailoring plan for %s: %d/%d requirement(s) supported, %d warning(s)",
            (job.get("title") or "")[:40],
            sum(1 for r in sanitized["requirements"] if r["supported"]),
            len(sanitized["requirements"]), len(sanitized["_warnings"]),
        )
        return sanitized
    except httpx.TimeoutException as exc:
        log.warning(
            "Local tailoring plan for %s timed out after %.0fs (model=%s). "
            "Qwen-family models in particular can be slow; raise "
            "APPLYPILOT_LOCAL_LLM_TIMEOUT or use a smaller/faster model if "
            "this persists. (%s)",
            (job.get("title") or "")[:40], timeout, model, exc,
        )
        return None
    except Exception as exc:
        log.warning("Local tailoring plan failed for %s: %s: %s",
                    (job.get("title") or "")[:40], type(exc).__name__, exc)
        return None


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_plan(text: str) -> dict:
    """Parse plan JSON; raise ValueError if malformed.

    Defense-in-depth for Qwen3/other hybrid-reasoning models: even with
    "think": false requested (see get_local_tailoring_plan), some Ollama
    builds still emit an empty (or populated) <think>...</think> block
    before the real content. Stripped unconditionally before fence/brace
    extraction -- a cheap, pure-text operation, not dependent on which
    failure mode is actually occurring for a given model/build.
    """
    text = text.strip()
    text = _THINK_BLOCK_RE.sub("", text).strip()
    # Strip markdown fences if present
    if "```" in text:
        for part in text.split("```")[1::2]:
            part = part.lstrip("json").strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    return json.loads(text)


def _as_list(value) -> list:
    """Defensive coercion: local models occasionally return a string or dict
    where a list was expected. Treat anything non-list as empty rather than
    crashing or silently iterating characters of a string."""
    return value if isinstance(value, list) else []


# ---------------------------------------------------------------------------
# Grounding / fabrication-safety validation
# ---------------------------------------------------------------------------

def validate_local_plan(
    raw_plan: dict, requirement_lines: list[dict], ranked_evidence: list[dict],
) -> dict:
    """Deterministically build the full plan from the model's `matches`
    (requirement number -> evidence numbers).

    There is nothing to "ground" here in the old sense: every number the
    model can possibly cite resolves to a requirement_lines/ranked_evidence
    entry that was already deterministically extracted/retrieved before the
    model ran, so a valid reference is grounded by construction. The only
    work left is (a) dropping references outside the given lists' bounds
    (a malformed/hallucinated number, not a hallucinated fact) and (b)
    deriving every other plan field from which evidence numbers ended up
    matched to which requirements -- skills_to_emphasize, matching_projects,
    matching_certifications, and keyword_targets are exactly "which matched
    evidence items were skills/projects/certifications, and which job terms
    made them match" (rank_profile_evidence already recorded that), not
    something asked of the model as free text.

    bullets_to_prioritize/bullets_to_deemphasize/safe_rewrites are always
    returned empty -- those are real text-generation tasks (quoting/
    rewriting actual resume prose), which this planner deliberately leaves
    to the cloud model that has the full resume and is far more reliable at
    freeform generation. The keys are kept so format_local_plan_for_cloud
    and the debug CLI don't need to special-case an older/newer plan shape.

    Never raises. Returns a plan dict with the same shape as before, plus a
    "_warnings" key (list[str]) noting any out-of-range references that
    were dropped -- intended for the debug CLI command, not for the cloud
    prompt.
    """
    warnings: list[str] = []
    n_reqs = len(requirement_lines)
    n_evid = len(ranked_evidence)

    # requirement number -> evidence numbers cited for it (deduped)
    matched: dict[int, list[int]] = {}
    for entry in _as_list(raw_plan.get("matches")):
        if not isinstance(entry, dict):
            continue
        r = entry.get("r")
        if not isinstance(r, int) or isinstance(r, bool) or not (1 <= r <= n_reqs):
            warnings.append(f"Dropped match with out-of-range requirement id: {r!r}")
            continue
        evidence_ids: list[int] = []
        for e in _as_list(entry.get("e")):
            if isinstance(e, int) and not isinstance(e, bool) and 1 <= e <= n_evid:
                if e not in evidence_ids:
                    evidence_ids.append(e)
            else:
                warnings.append(f"Dropped invalid evidence id {e!r} for requirement R{r}")
        if evidence_ids:
            matched.setdefault(r, [])
            matched[r] = sorted(set(matched[r]) | set(evidence_ids))

    clean_requirements: list[dict] = []
    unsupported: list[str] = []
    skills_seen: dict[str, None] = {}
    projects_seen: dict[str, None] = {}
    certs_seen: dict[str, None] = {}
    keywords_seen: dict[str, None] = {}
    required_supported: list[str] = []
    any_supported: list[str] = []

    for i, line in enumerate(requirement_lines, start=1):
        evidence_ids = matched.get(i, [])
        evidence_names: list[str] = []
        for eid in evidence_ids:
            item = ranked_evidence[eid - 1]
            evidence_names.append(item["name"])
            if item["type"] == "skill":
                skills_seen.setdefault(item["name"], None)
            elif item["type"] == "project":
                projects_seen.setdefault(item["name"], None)
            elif item["type"] == "certification":
                certs_seen.setdefault(item["name"], None)
            for term in item.get("matched_terms") or []:
                keywords_seen.setdefault(term, None)
        supported = bool(evidence_names)
        importance = line["importance"] if line["importance"] in ("required", "preferred") else "preferred"
        clean_requirements.append({
            "requirement": line["text"],
            "importance": importance,
            "resume_evidence": evidence_names,
            "supported": supported,
        })
        if supported:
            any_supported.append(line["text"])
            if importance == "required":
                required_supported.append(line["text"])
        else:
            unsupported.append(line["text"])

    summary_focus = (required_supported or any_supported)[:3]

    return {
        "requirements": clean_requirements[:12],
        "skills_to_emphasize": list(skills_seen)[:10],
        "bullets_to_prioritize": [],
        "bullets_to_deemphasize": [],
        "matching_projects": list(projects_seen)[:5],
        "matching_certifications": list(certs_seen)[:5],
        "summary_focus": summary_focus,
        "keyword_targets": list(keywords_seen)[:10],
        "safe_rewrites": [],
        "unsupported_requirements": unsupported[:10],
        "_warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Rendering for the cloud prompt
# ---------------------------------------------------------------------------

def format_local_plan_for_cloud(plan: dict | None) -> str:
    """Render a validated plan as compact text for the cloud model's user message.

    Returns "" when the plan is empty/None (nothing survived validation) so
    callers can treat an empty string as "no plan" and skip the guidance
    block entirely rather than sending a useless empty section.
    """
    if not plan:
        return ""
    lines: list[str] = []

    reqs = plan.get("requirements") or []
    if reqs:
        lines.append("Job requirements (grounded against the resume):")
        for r in reqs:
            tag = "REQUIRED" if r.get("importance") == "required" else "preferred"
            status = "supported by resume" if r.get("supported") else "NOT supported by resume"
            lines.append(f"- [{tag}] {r.get('requirement', '')} -- {status}")

    if plan.get("skills_to_emphasize"):
        lines.append("Skills to emphasize: " + ", ".join(plan["skills_to_emphasize"]))
    if plan.get("keyword_targets"):
        lines.append("Keywords worth incorporating (already supported): " + ", ".join(plan["keyword_targets"]))
    if plan.get("bullets_to_prioritize"):
        lines.append("Bullets to prioritize/bring forward:")
        lines.extend(f"  - {b}" for b in plan["bullets_to_prioritize"])
    if plan.get("bullets_to_deemphasize"):
        lines.append("Bullets to de-emphasize or drop:")
        lines.extend(f"  - {b}" for b in plan["bullets_to_deemphasize"])
    if plan.get("matching_projects"):
        lines.append("Relevant projects: " + ", ".join(plan["matching_projects"]))
    if plan.get("matching_certifications"):
        lines.append("Relevant certifications: " + ", ".join(plan["matching_certifications"]))
    if plan.get("summary_focus"):
        lines.append("Summary should emphasize: " + "; ".join(plan["summary_focus"]))
    if plan.get("safe_rewrites"):
        lines.append("Suggested factual-preserving wording changes:")
        for rw in plan["safe_rewrites"]:
            lines.append(f'  - "{rw["original"]}" -> "{rw["suggested"]}"')
    if plan.get("unsupported_requirements"):
        lines.append(
            "Requirements NOT supported by the resume (do not fabricate evidence for these): "
            + "; ".join(plan["unsupported_requirements"])
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Debug / evaluation helper (used by `applypilot debug-local-plan`)
# ---------------------------------------------------------------------------

def debug_plan_for_job(job: dict, profile: dict) -> dict | None:
    """Resolve the resume for `job` the same way the real pipeline does, then
    fetch and validate a local tailoring plan -- WITHOUT generating or
    modifying any resume. Used by the `applypilot debug-local-plan` CLI
    command so a user can manually judge whether the local model (and the
    deterministic retrieval feeding it) is useful.

    Returns None if local planning failed entirely (unreachable model,
    unparseable response -- same condition get_local_tailoring_plan
    signals with None). Otherwise returns:
      {"plan": dict, "evidence": list[dict], "requirement_lines": list[dict]}
    `evidence`/`requirement_lines` are recomputed here (cheap, deterministic,
    no LLM call) purely for CLI display -- get_local_tailoring_plan already
    used the same functions internally to build the model's prompt.
    """
    from applypilot.scoring.resume_router import load_resume_text_for_job
    resume_text, _ = load_resume_text_for_job(job)
    plan = get_local_tailoring_plan(resume_text, job, profile)
    if plan is None:
        return None
    top_n = int(os.environ.get("APPLYPILOT_LOCAL_EVIDENCE_TOPN", "6"))
    return {
        "plan": plan,
        "evidence": rank_profile_evidence(job, profile, top_n=top_n),
        "requirement_lines": _extract_requirement_lines(job.get("full_description") or ""),
    }



