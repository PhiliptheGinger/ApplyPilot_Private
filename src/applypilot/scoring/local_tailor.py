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
      Compact full-resume-generation prompt for small local models. No
      longer tailor.py's default DEGRADED MODE path (see
      compose_degraded_resume_json below, 2026-08-23) -- kept for
      reference/manual use, not called by the normal pipeline.

  compose_degraded_resume_json(client, resume_text, job, profile, job_schema)
      -> (dict, dict)
      The current DEGRADED-MODE path: Python deterministically parses the
      ORIGINAL resume (scoring/pdf.py) and asks the local model for only a
      small, schema-bounded REALIZATION step (see _build_realization_prompt)
      -- short bullet sentences + a summary -- rather than an entire resume.
      Everything else (headers, dates, company names, skills, education)
      stays verbatim from the original. Still exactly one LLM call, just a
      much smaller one than the old full-generation approach. Used by
      tailor.py's tailor_resume() when every cloud model is exhausted.

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
import time

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
    companies = ", ".join(resume_facts.get("preserved_companies", [])) or "same companies as in the original resume"
    school = resume_facts.get("preserved_school", "keep original exactly")
    metrics_list = resume_facts.get("real_metrics", [])
    metrics = ", ".join(metrics_list[:6]) if metrics_list else "only numbers already in the resume; do not invent any"
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
        "{\n"
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
    r")",
    re.IGNORECASE,
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
    r")\b",
    re.IGNORECASE,
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
    r")\b",
    re.IGNORECASE,
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
    r")\b",
    re.IGNORECASE,
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


def _classify_candidate_lines(
    texts: list[str],
    max_lines: int,
) -> tuple[list[dict], list[str]]:
    """Shared tail for both extraction strategies below: drop employer-
    benefit lines, tag required/preferred/unspecified importance, cap at
    max_lines. `texts` must already be a deduplicated, order-preserved
    list of candidate strings -- this function makes no structural
    judgment about whether a candidate is well-formed, only whether it's
    a benefit line and how it should be tagged."""
    lines: list[dict] = []
    dropped: list[str] = []
    for text in texts:
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


def _extract_marker_lines(
    description: str,
    max_lines: int = 8,
) -> tuple[list[dict], list[str]]:
    """Primary extraction strategy: pull bullet/numbered lines via
    _REQUIREMENT_MARKER_RE. Unchanged behavior from before this function
    was split out -- see _split_requirement_lines for the fallback this
    feeds into."""
    if not description:
        return [], []
    texts: list[str] = []
    seen: set[str] = set()
    for match in _REQUIREMENT_MARKER_RE.finditer(description):
        text = match.group(1).strip()
        if not text or len(text) < 8 or len(text) > 220:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
    return _classify_candidate_lines(texts, max_lines)


# 2026-08-25 (Direction 1 of the extraction audit): some ATS sources flatten
# genuinely itemized requirements to plain, UNMARKED lines -- confirmed
# against real postings for three separate NO_SUPPORTED_EVIDENCE incidents
# (Axon, Twilio, Pinterest/Greenhouse, all single-newline-separated) and
# against a live Workday posting (blank-line-separated, i.e. two "\n" in a
# row -- an empty line between each item). Splitting on every physical line
# handles both shapes uniformly: a blank "paragraph break" line is just an
# empty string after stripping, which the per-line filter below already
# discards, so no separate blank-line-vs-single-newline branch is needed.
#
# A marker character is itself evidence of intentional itemization, so
# _extract_marker_lines doesn't need to judge WHAT a marked line says --
# only that it's not a benefit line. A markerless line has no such signal,
# so _looks_like_list_item substitutes a structural (not semantic) proxy
# for "this reads as one discrete item, not a paragraph of flowing prose or
# a section label": short, no trailing colon (colons mark section headers
# like "Qualifications:", never a requirement itself), and -- the load-
# bearing check -- no INTERNAL sentence boundary. A single Workday-style
# clause ("4+ years in deploying large-scale, complex applications...")
# ends in exactly one period, at the end, so it passes; a flattened "About
# us" paragraph ("...to craft personalized customer experiences. Our
# dedication to remote-first work... Your career at Twilio is in your
# hands.") contains multiple ". <Capital>" boundaries and is rejected.
#
# _PARAGRAPH_FALLBACK_MIN_ITEMS is the other half of the safety margin, and
# is enforced as a CONTIGUOUS-RUN requirement, not a whole-description
# count -- see _extract_paragraph_lines. Real postings were observed (via
# the three named incidents) to also contain a handful of short, structurally-
# qualifying header/label lines ("Who we are", "Responsibilities", "About
# the job") scattered through the surrounding prose. Checking a flat count
# across the whole description let this noise crowd out the real list
# entirely: on the Axon and Twilio postings, seven of the first eight
# qualifying lines in document order were section headers, and the actual
# requirements never got extracted at all under a flat-count design.
# Structurally, though, these two classes are NOT the same shape: a real
# requirements list renders as many qualifying lines in a row (one `<li>`/
# `<p>` after another, nothing else between them), while a section header
# is always isolated -- immediately followed by a rejected multi-sentence
# prose paragraph or another header, never by more qualifying lines. Only a
# RUN of _PARAGRAPH_FALLBACK_MIN_ITEMS consecutive qualifying lines (blank
# lines tolerated as gaps within a run, since that's exactly Workday's
# blank-line-per-item formatting; any REJECTED non-blank line breaks the
# run) is treated as evidence of a genuine list. A single qualifying line
# surrounded by rejected/prose lines is never, by itself, a list.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s+[A-Z]")
_PARAGRAPH_FALLBACK_MIN_ITEMS = 3
_PARAGRAPH_MIN_LEN = 2
_PARAGRAPH_MAX_LEN = 220


def _looks_like_list_item(text: str) -> bool:
    """Conservative structural proxy for "this markerless line is plausibly
    one discrete requirement, not flowing prose or a section label" -- see
    the extraction-fallback comment block above."""
    if not (_PARAGRAPH_MIN_LEN <= len(text) <= _PARAGRAPH_MAX_LEN):
        return False
    if text.endswith(":"):
        return False
    return not _SENTENCE_BOUNDARY_RE.search(text)


def _extract_paragraph_lines(
    description: str,
    max_lines: int = 8,
) -> tuple[list[dict], list[str]]:
    """Fallback extraction strategy -- only ever called by
    _split_requirement_lines when marker-based extraction found ZERO lines,
    never when it found any (see that function).

    Scans physical lines in order, tracking a running streak of
    consecutive qualifying (_looks_like_list_item) lines; a streak is only
    kept once it reaches _PARAGRAPH_FALLBACK_MIN_ITEMS, discarding shorter
    ones (isolated header/label lines) as they're found. Blank lines are
    skipped without breaking a streak (Workday's blank-line-per-item
    format); any REJECTED non-blank line ends the current streak. See the
    module comment above _SENTENCE_BOUNDARY_RE for the full rationale and
    the real postings that motivated this design.
    """
    if not description:
        return [], []
    texts: list[str] = []
    seen: set[str] = set()
    streak: list[str] = []

    def _flush_streak() -> None:
        if len(streak) >= _PARAGRAPH_FALLBACK_MIN_ITEMS:
            for text in streak:
                key = text.lower()
                if key not in seen:
                    seen.add(key)
                    texts.append(text)
        streak.clear()

    for raw_line in description.split("\n"):
        text = raw_line.strip()
        if not text:
            continue  # blank line: tolerated gap, does not break a streak
        if _looks_like_list_item(text):
            streak.append(text)
        else:
            _flush_streak()
    _flush_streak()

    if not texts:
        return [], []
    return _classify_candidate_lines(texts, max_lines)


def _split_requirement_lines(
    description: str,
    max_lines: int = 8,
) -> tuple[list[dict], list[str]]:
    """Same extraction as _extract_requirement_lines, but also returns the
    benefit/perk lines that were dropped.

    Callers that report on their own behaviour (get_local_tailoring_plan,
    which notes WHY it skipped the model) need to distinguish "this posting
    had no bullet lines at all" from "every bullet line was an employer
    benefit" -- the second is the interesting one.

    Marker-based extraction (_extract_marker_lines) is always tried first
    and, if it finds ANYTHING at all, its result is used as-is -- the
    paragraph fallback (_extract_paragraph_lines) never competes with or
    overrides it, only fills in for postings where it found nothing.
    """
    if not description:
        return [], []
    lines, dropped = _extract_marker_lines(description, max_lines=max_lines)
    if not lines:
        lines, dropped = _extract_paragraph_lines(description, max_lines=max_lines)
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


def _format_requirement_lines(
    lines: list[dict],
    candidates: dict[int, list[int]] | None = None,
    only_ids: set[int] | None = None,
) -> str:
    """Render requirement lines as "R1 [required] ..." text.

    `only_ids`, if given, renders just those requirement numbers (1-based,
    matching `lines`' position) -- get_local_tailoring_plan uses this to
    show the model only the requirements that are still genuinely
    ambiguous after deterministic pair-scoring (see
    _auto_resolve_requirements), not ones already resolved without it.

    `candidates`, if given, appends a "(candidates: E2, E5)" hint naming
    the deterministically top-scoring evidence numbers for that
    requirement -- steering the model away from evidence a strictly
    stronger candidate already outranks. This is only a hint: the actual
    enforcement happens in code afterward (get_local_tailoring_plan
    intersects the model's answer against this same candidate set), so an
    ignored hint can't let a low-relevance pick through.
    """
    if not lines:
        return ""
    rendered: list[str] = []
    for i, l in enumerate(lines, start=1):
        if only_ids is not None and i not in only_ids:
            continue
        line = f"R{i} [{l['importance']}] {l['text']}"
        if candidates and candidates.get(i):
            line += " (candidates: " + ", ".join(f"E{c}" for c in candidates[i]) + ")"
        rendered.append(line)
    return "\n".join(rendered)


def _normalize_term(term: str) -> str:
    return term.strip().lower()


## Generic connector words pulled from multi-word item names (e.g. "National
## Tire and Battery") that would otherwise trivially match almost any job
## description via the word "and"/"the"/etc. Only filters the individual-
## word split of `name` -- relevance_categories/factual_concepts were
## historically assumed to already be curated signal that needed no
## filtering. That assumption is what let "it"/"technical" through (see
## _GENERIC_EVIDENCE_TERMS below): profile.json data, not code, is where
## these two originate, but nothing downstream ever questioned them.
_NAME_TOKEN_STOPWORDS = frozenset(
    {
        "and",
        "the",
        "for",
        "with",
        "from",
        "into",
        "onto",
        "your",
        "you",
        "our",
        "their",
        "his",
        "her",
        "its",
        "this",
        "that",
        "these",
        "those",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "not",
        "but",
        "can",
    }
)

# A term this generic carries no discriminating signal on its own, no
# matter which source it came from (name, name-word-split,
# relevance_categories, or factual_concepts) -- it's excluded once, here,
# rather than re-litigated per source or (worse) papered over downstream
# in _pair_candidate_evidence with special-case exceptions.
#
# Two independent, both DATA-observed (not hypothetical) failure shapes:
#   - Too short to be anything but a function word. profile.json tags
#     CompTIA A+ with the category "IT" -- at 2 characters it's
#     indistinguishable from the pronoun "it" and matches almost any
#     sentence in English ("...as IT relates to the selection...").
#     Nothing upstream normalizes/expands "IT" to "information technology"
#     first; _normalize_term just lowercases it, so it's literally the
#     2-letter word "it" by the time it reaches _term_in_text.
#   - A broad descriptor of ANY domain's work, not a specific skill/
#     industry. profile.json tags the Python skill with the bare category
#     "technical" (alongside "automation"/"data"/"OCR", which ARE
#     specific). "Technical assistance" shows up in auto-parts retail,
#     nursing, and IT support alike -- tagging Python with the single word
#     "technical" makes it match every one of those, not just software
#     ones. It's a different failure mode than "it" (not short, not a
#     function word) but the same root cause: too generic to mean anything
#     standalone.
# Neither entry is dropped when it's part of a longer, more specific
# phrase -- "technical support", "customer-facing technical", and
# "desktop support" (also on CompTIA A+) are untouched; only the bare
# single-word term is excluded. Add entries here ONLY when a real generic
# false-positive is observed (as both of these were), not speculatively --
# this is a short, explainable exception list, not a general stopword
# dictionary standing in for real domain-specificity judgment.
_GENERIC_EVIDENCE_TERMS = frozenset({"it", "technical"})


def _is_generic_evidence_term(term: str) -> bool:
    """True if `term` (an already-normalized name/category/concept term)
    is too generic to serve as a standalone evidence-match signal -- see
    _GENERIC_EVIDENCE_TERMS. Multi-word terms are never caught by this:
    a category like "customer service" or "technical support" is always
    well over the length floor and never equals one of the bare single
    words in the exception set, so only a truly bare, truly generic term
    is excluded."""
    return len(term) < 3 or term in _GENERIC_EVIDENCE_TERMS


def _item_terms(item: dict) -> set[str]:
    """Collect the matchable normalized terms already curated on one
    experience_inventory / project_inventory / skills_inventory entry.

    Every term -- from the item's name, the individual words split from
    it, its relevance_categories, and its factual_concepts alike -- passes
    through the SAME final _is_generic_evidence_term filter, so a term too
    generic to be useful can't slip through just because it came from a
    source that historically wasn't filtered (see _NAME_TOKEN_STOPWORDS'
    comment above for how that happened before).
    """
    terms: set[str] = set()
    name = item.get("name")
    if isinstance(name, str) and name.strip():
        terms.add(_normalize_term(name))
        for w in _TERM_WORD_RE.findall(name):
            wl = _normalize_term(w)
            if wl not in _NAME_TOKEN_STOPWORDS:
                terms.add(wl)
    for cat in item.get("relevance_categories") or []:
        if isinstance(cat, str) and cat.strip():
            terms.add(_normalize_term(cat))
    for concept in item.get("factual_concepts") or []:
        if isinstance(concept, str) and concept.strip():
            terms.add(_normalize_term(concept))
    return {t for t in terms if t and not _is_generic_evidence_term(t)}


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
            ranked.append(
                {
                    "type": kind,
                    "name": name,
                    "score": len(matched),
                    "matched_terms": matched,
                    "item": item,
                }
            )
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
# Deterministic requirement <-> evidence PAIR scoring
#
# rank_profile_evidence already tells us WHY each evidence item matched the
# job as a whole (matched_terms). What it doesn't do is tell the model
# which of those terms are actually relevant to any ONE requirement line --
# so a CPU-only 1-2B model, faced with a flat E1..En list and no per-
# requirement signal, ends up re-deriving relevance from scratch per pair
# and gets it wrong exactly where it matters: picking Python for a
# "customer service" requirement because Python was topically present
# *somewhere* in the evidence, or rejecting a genuinely relevant employer
# because nothing in its own free-text reasoning connected the dots.
#
# This closes that gap without asking the model to reason better: for each
# requirement line, re-score each evidence item's ALREADY-COMPUTED
# matched_terms against just that line's text (not the whole job
# description). An evidence item whose matched_terms don't appear anywhere
# in a given requirement line has, by definition, no demonstrated overlap
# with THAT requirement -- it's excluded outright, not just deprioritized.
# Only the TOP-SCORING tier per requirement survives as a "candidate";
# anything scoring lower is dropped even if nonzero, because a strictly
# stronger deterministic match already exists for that requirement.
#
# What this buys the pipeline:
#   - 0 candidates  -> requirement is unsupported. No LLM call needed.
#   - 1 candidate   -> requirement is unambiguously supported by that one
#                      item. No LLM call needed.
#   - 2+ candidates -> genuinely ambiguous (deterministic scoring can't
#                      break the tie) -- and ONLY THEN is the local model
#                      asked, restricted to choosing among that tier.
# If every requirement resolves in the first two buckets, the local model
# is never called at all for this job.
# ---------------------------------------------------------------------------

# Controlled concept-synonym vocabulary.
#
# _pair_candidate_evidence (below) only recognizes a requirement/evidence
# relationship when the requirement line LITERALLY contains one of the
# evidence item's matched_terms (e.g. the exact phrase "customer service").
# That is precise but misses obvious paraphrases a human reads instantly:
# "Accept and respond to telephone inquiries from customers in a polite
# manner..." is self-evidently a customer-service requirement, yet never
# says the words "customer service" -- so it scored zero candidates and
# came back unsupported even though Mavis/Waffle House plainly cover it.
#
# This closes that gap without reopening free-text/fuzzy matching: a small,
# hand-curated table of alternate phrasings for a couple of the most common
# relevance_categories values. Two things keep it bounded rather than a
# general synonym-expansion engine:
#   - It only widens which REQUIREMENT LINES an evidence item's
#     already-established matched_term can apply to. It can never make an
#     evidence item relevant to a job at all if rank_profile_evidence didn't
#     already find literal category/name overlap with the job description
#     as a whole -- synonym expansion never invents a new matched_term, it
#     only rechecks existing ones against different phrasing.
#   - A synonym hit scores identically to a literal hit, so the
#     "top-scoring tier only" rule in _pair_candidate_evidence is untouched
#     -- this only changes WHICH items can reach that tier for a given
#     line, never how many survive once there.
#
# Deliberately small and specific: each entry is a handful of unambiguous
# rephrasings for ONE category, not a broad thesaurus. A category like
# "technical" is intentionally NOT here -- "technical assistance"/"product
# knowledge" show up in plenty of non-software contexts (e.g. hardware
# retail) and would reintroduce exactly the false-positive failure mode
# (Python matched to an unrelated "technical" requirement) this whole
# closed-set design exists to prevent. Add entries only when a real
# false-negative is observed, narrowly enough that it can't bleed into an
# unrelated domain.
_CONCEPT_SYNONYM_PATTERNS: dict[str, re.Pattern] = {
    term: re.compile(pattern, re.IGNORECASE)
    for term, pattern in {
        "customer service": (
            r"\btelephone\s+inquir\w*\b|"
            r"\bcustomer\s+(?:inquir\w*|feedback|needs?|complaints?|requests?|confidence)\b|"
            r"\brespond(?:ing|s)?\s+to\s+(?:customer|client)s?\b|"
            r"\bassist(?:ing|s)?\s+(?:customer|client|guest)s?\b|"
            r"\bserv(?:e|ing|es)\s+(?:customer|client|guest)s?\b|"
            r"\b(?:customer|client|guest)s?\s+(?:in\s+a\s+)?(?:polite|friendly|courteous)\s+manner\b"
        ),
        "sales": (
            r"\blead\s+generation\b|\bgenerate\s+leads?\b|\bclose\s+(?:a\s+)?sales?\b|"
            r"\bupsell(?:ing)?\b|\boutreach\b|\bmembership\s+growth\b|"
            r"\bnew\s+business\s+opportunit\w*\b|\bfollow-?ups?\b"
        ),
    }.items()
}


def _synonym_hit(term: str, text_lower: str) -> bool:
    """True if `text_lower` contains a curated alternate phrasing of the
    concept named by `term` (see _CONCEPT_SYNONYM_PATTERNS).

    `term` must exactly match one of the table's keys -- this never guesses
    at arbitrary terms, only the small curated set above. Most matched_terms
    (project names, one-off skills) simply aren't in the table and always
    return False here, falling back entirely to the literal _term_in_text
    check in _pair_candidate_evidence.
    """
    pattern = _CONCEPT_SYNONYM_PATTERNS.get(term)
    return bool(pattern and pattern.search(text_lower))


def _pair_candidate_evidence(requirement_text: str, ranked_evidence: list[dict]) -> list[int]:
    """Return the 1-based evidence indices that are the STRONGEST
    deterministic match for one requirement line, by re-checking each
    evidence item's job-level `matched_terms` against just this line's
    text -- either literally (_term_in_text) or via a curated paraphrase
    (_synonym_hit) -- see the module comments above.

    Returns only the top-scoring tier -- ties for first place, never a
    runner-up -- so a caller can treat this list as "the evidence this
    requirement could plausibly cite" rather than "everything topically
    nearby". Empty if no evidence item has any matched term (literal or
    synonym) in this line.
    """
    text_lower = requirement_text.lower()
    scored: list[tuple[int, int]] = []
    for idx, item in enumerate(ranked_evidence, start=1):
        score = sum(
            1 for t in (item.get("matched_terms") or []) if _term_in_text(t, text_lower) or _synonym_hit(t, text_lower)
        )
        if score > 0:
            scored.append((idx, score))
    if not scored:
        return []
    top = max(s for _, s in scored)
    return [idx for idx, s in scored if s == top]


def _auto_resolve_requirements(
    requirement_lines: list[dict],
    ranked_evidence: list[dict],
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Run _pair_candidate_evidence for every requirement and sort the
    results into what can be settled without the model vs. what's
    genuinely ambiguous.

    Returns (resolved, candidates):
      resolved   -- requirement number -> evidence ids, for every
                    requirement with 0 or 1 top-tier candidate (nothing
                    for the model to decide).
      candidates -- requirement number -> top-tier evidence ids, for
                    EVERY requirement (including resolved ones, where it's
                    just their 0/1-length entry). Used to (a) build the
                    "(candidates: ...)" hint shown to the model for the
                    still-ambiguous requirements and (b) hard-filter
                    whatever the model answers afterward, so it can never
                    select an evidence id outside this deterministic tier
                    even if it ignores the hint.
    """
    resolved: dict[int, list[int]] = {}
    candidates: dict[int, list[int]] = {}
    for i, line in enumerate(requirement_lines, start=1):
        cands = _pair_candidate_evidence(line["text"], ranked_evidence)
        candidates[i] = cands
        if len(cands) <= 1:
            resolved[i] = cands
    return resolved, candidates


def _merge_model_matches_with_resolved(
    raw_plan: dict,
    resolved: dict[int, list[int]],
    candidates: dict[int, list[int]],
    ambiguous_ids: set[int],
) -> tuple[dict, list[str]]:
    """Combine the deterministically-resolved requirements with the local
    model's answer for the (only) ambiguous ones, into the same
    {"matches": [...]} shape validate_local_plan already expects.

    The model was shown a hint naming each ambiguous requirement's
    deterministic candidate tier (see _format_requirement_lines), but that
    hint is advisory -- this is the actual enforcement: any evidence id the
    model cites for an ambiguous requirement that isn't in that
    requirement's candidate tier is dropped (not the whole match, just the
    offending id), and any requirement number the model answers that
    wasn't even shown to it (hallucinated, or one of the resolved ones) is
    ignored entirely -- the deterministic answer already stands for those.
    Returns (combined_plan_dict, extra_warnings).
    """
    warnings: list[str] = []
    combined: dict[int, list[int]] = dict(resolved)
    for entry in _as_list(raw_plan.get("matches")):
        if not isinstance(entry, dict):
            continue
        r = entry.get("r")
        if not isinstance(r, int) or isinstance(r, bool) or r not in ambiguous_ids:
            continue
        allowed = set(candidates.get(r, []))
        picked: list[int] = []
        for e in _as_list(entry.get("e")):
            if isinstance(e, int) and not isinstance(e, bool) and e in allowed:
                picked.append(e)
            else:
                warnings.append(
                    f"Dropped model-selected evidence {e!r} for R{r}: not in its deterministic candidate tier"
                )
        combined[r] = picked
    matches = [{"r": r, "e": ids} for r, ids in combined.items()]
    return {"matches": matches}, warnings


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
    "You are only ever shown requirements that were too close to call "
    "deterministically -- every requirement here already has a "
    "'(candidates: E2, E5)' hint listing the ONLY evidence numbers worth "
    "considering for it, pre-computed by scoring shared terms/categories "
    "between the requirement and each evidence item. You MUST choose only "
    "from a requirement's own candidates (or none) -- never cite an "
    "evidence number for a requirement that isn't in its candidate list, "
    "even if it appears elsewhere in EVIDENCE and seems related.\n\n"
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
    items. If either list is empty there is nothing to match, so the local
    model isn't called at all.

    Otherwise, _auto_resolve_requirements pair-scores every requirement
    against every evidence item's already-computed matched_terms (see the
    module comment above _pair_candidate_evidence) and settles anything
    with 0 or 1 top-tier candidate deterministically -- 0 means
    unsupported, 1 means unambiguously supported by that item, neither
    needs the model to weigh in. The local model is asked ONLY about
    requirements left with 2+ tied top-tier candidates (genuine
    ambiguity a lexical-overlap heuristic can't break), and even then its
    answer is hard-filtered back down to that requirement's own candidate
    set afterward -- so it can never select an evidence item a
    deterministically stronger candidate already outranks. If nothing is
    ambiguous, the model is never called for this job at all.
    """
    # Avoid circular import
    from applypilot.scoring.tailor import display_company

    top_n = int(os.environ.get("APPLYPILOT_LOCAL_EVIDENCE_TOPN", "6"))
    ranked_evidence = rank_profile_evidence(job, profile, top_n=top_n)
    requirement_lines, dropped_benefits = _split_requirement_lines(job.get("full_description") or "")

    if not requirement_lines or not ranked_evidence:
        reason = (
            f"Skipped local LLM call: {len(requirement_lines)} requirement line(s), "
            f"{len(ranked_evidence)} evidence item(s) -- nothing to match."
        )
        if not requirement_lines and dropped_benefits:
            reason += (
                f" All {len(dropped_benefits)} bullet line(s) in this posting were "
                "employer benefits/perks, not candidate requirements: " + "; ".join(dropped_benefits[:6])
            )
        log.debug("Local tailoring plan for %s: %s", (job.get("title") or "")[:40], reason)
        plan = validate_local_plan({"matches": []}, requirement_lines, ranked_evidence)
        plan["_warnings"].append(reason)
        return plan

    resolved, candidates = _auto_resolve_requirements(requirement_lines, ranked_evidence)
    ambiguous_ids = {i for i in range(1, len(requirement_lines) + 1) if i not in resolved}

    if not ambiguous_ids:
        log.debug(
            "Local tailoring plan for %s: skipped LLM call -- all %d requirement(s) "
            "resolved deterministically via requirement/evidence term overlap "
            "(no requirement had 2+ evidence items tied for top relevance)",
            (job.get("title") or "")[:40],
            len(requirement_lines),
        )
        combined = {"matches": [{"r": r, "e": ids} for r, ids in resolved.items()]}
        plan = validate_local_plan(combined, requirement_lines, ranked_evidence)
        plan["_warnings"].append(
            "Skipped local LLM call: every requirement resolved deterministically "
            "via requirement/evidence term overlap -- no requirement had 2+ "
            "evidence items tied for top relevance."
        )
        return plan

    url = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", _DEFAULT_LOCAL_URL).rstrip("/")
    model = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", _DEFAULT_LOCAL_MODEL)
    timeout = float(os.environ.get("APPLYPILOT_LOCAL_LLM_TIMEOUT", "60"))

    company = display_company(job)
    job_text = f"TITLE: {job.get('title', '')}\nCOMPANY: {company or 'unknown'}"
    evidence_text = format_evidence_for_prompt(ranked_evidence)
    requirements_text = _format_requirement_lines(
        requirement_lines,
        candidates=candidates,
        only_ids=ambiguous_ids,
    )

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
            log.warning("Local tailoring plan for %s: empty message content in response", (job.get("title") or "")[:40])
            return None
        raw_plan = _parse_plan(text)
        if not isinstance(raw_plan, dict):
            log.warning(
                "Local tailoring plan for %s: parsed JSON was not an object (got %s)",
                (job.get("title") or "")[:40],
                type(raw_plan).__name__,
            )
            return None
        combined, filter_warnings = _merge_model_matches_with_resolved(
            raw_plan,
            resolved,
            candidates,
            ambiguous_ids,
        )
        sanitized = validate_local_plan(combined, requirement_lines, ranked_evidence)
        sanitized["_warnings"].extend(filter_warnings)
        log.debug(
            "Local tailoring plan for %s: %d/%d requirement(s) supported, %d warning(s), "
            "%d/%d requirement(s) sent to the model (rest resolved deterministically)",
            (job.get("title") or "")[:40],
            sum(1 for r in sanitized["requirements"] if r["supported"]),
            len(sanitized["requirements"]),
            len(sanitized["_warnings"]),
            len(ambiguous_ids),
            len(requirement_lines),
        )
        return sanitized
    except httpx.TimeoutException as exc:
        log.warning(
            "Local tailoring plan for %s timed out after %.0fs (model=%s). "
            "Qwen-family models in particular can be slow; raise "
            "APPLYPILOT_LOCAL_LLM_TIMEOUT or use a smaller/faster model if "
            "this persists. (%s)",
            (job.get("title") or "")[:40],
            timeout,
            model,
            exc,
        )
        return None
    except httpx.ConnectError as exc:
        # Nothing listening at `url` -- the configured local model endpoint
        # (APPLYPILOT_LOCAL_LLM_URL) isn't running or isn't reachable.
        log.warning(
            "Local tailoring plan for %s: could not reach %s (%s). Falling back to cloud tailoring only.",
            (job.get("title") or "")[:40],
            url,
            exc,
        )
        return None
    except Exception as exc:  # noqa: BLE001 -- final fallback of a documented
        # "returns None if the local model is unreachable, the response is
        # empty, or the output can't be parsed" contract (see this
        # function's docstring); the two specific httpx handlers above
        # already cover the common cases, this catches anything else
        # (malformed client state, unexpected library exceptions) so a
        # local-model glitch degrades to "skip local planning", never
        # crashes the tailoring pipeline.
        log.warning(
            "Local tailoring plan failed for %s: %s: %s", (job.get("title") or "")[:40], type(exc).__name__, exc
        )
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
            except json.JSONDecodeError:
                continue
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
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
    raw_plan: dict,
    requirement_lines: list[dict],
    ranked_evidence: list[dict],
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
        clean_requirements.append(
            {
                "requirement": line["text"],
                "importance": importance,
                "resume_evidence": evidence_names,
                "supported": supported,
            }
        )
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


# ---------------------------------------------------------------------------
# DEGRADED-MODE structured realization (2026-08-23)
#
# Replaces the old approach of asking the local model for an entire tailored
# resume JSON (_build_compact_local_prompt, still defined above for
# reference but no longer the default DEGRADED MODE path). That approach
# asked a CPU-bound 1.7B model to reproduce structure it didn't need to
# touch at all -- section headers, company names, dates, full skills
# categorization, every bullet in every section -- which is both slow
# (large output budget) and risky (small models "helpfully" alter text
# they were told to copy verbatim).
#
# New shape: Python already knows, from the schema representation
# (scoring/schemas.py), exactly which requirements are grounded/supported
# and what rhetorical shape each one's bullet should take. The local model's
# ONLY job is a small, bounded REALIZATION step -- turn each schema+evidence
# pairing into one short sentence, plus one short summary -- never asked to
# invent structure, headers, or facts. Python then deterministically
# composes the full resume JSON by parsing the ORIGINAL resume text (via
# scoring/pdf.py's existing, already-trusted parser) and splicing in the
# realized bullets/summary, keeping everything else -- headers, dates,
# company names, skills, education -- verbatim. This is still exactly ONE
# LLM call (same as before), just a much smaller one.
# ---------------------------------------------------------------------------

# Cap on how many bullets the local model is asked to realize in one call --
# keeps the prompt/response small regardless of how many requirements a
# posting has. Overridable for testing/tuning; not expected to need raising
# in practice since format_schema_guidance already caps at a similar size.
_DEGRADED_MAX_REALIZED_BULLETS = 5

_REALIZATION_SYSTEM = (
    "You write SHORT resume content from pre-verified facts. You are given "
    "a list of bullet SLOTS to fill and a summary schema. For each slot you "
    "are given: which evidence it's based on, the rhetorical shape to "
    "follow (a sequence of slot names, e.g. action -> object -> outcome), "
    "the underlying fact, a CLAIM CEILING, an AGENCY CEILING, and -- when "
    "given -- exact terms to use verbatim or a note that this is "
    "transferable (not identical-domain) experience. Write exactly ONE "
    "sentence per bullet slot, realizing that shape. Never invent a fact, "
    "employer, tool, or number beyond what you're given -- never change or "
    "add a number not already in the fact text. When a slot is marked "
    "transferable, do not claim the source experience IS the target domain "
    "-- state the transferable capability. Use ACTIVE voice, candidate as "
    "subject ('Diagnosed X'), never passive ('X was diagnosed').\n\n"
    "CLAIM CEILING limits TECHNICAL DEPTH: participation < execution < "
    "implementation < design < authority (architected). AGENCY CEILING is "
    "a SEPARATE limit on PEOPLE authority: individual_contributor < owner "
    "< team_lead (led/managed) < director (directed/spearheaded). Never "
    "use a verb stronger than either given ceiling.\n\n"
    "Also write one short (2-3 sentence) professional summary following "
    "the given summary schema's shape, from the given viewpoint where it "
    "fits naturally -- the viewpoint changes emphasis, never facts.\n\n"
    "Output ONLY this JSON, nothing else -- no markdown fences, no prose:\n"
    '{"summary": "...", "bullets": [{"evidence": "<evidence name exactly '
    'as given>", "text": "<one sentence>"}]}\n'
    "Include exactly one bullet entry per slot you were given, using the "
    "evidence name verbatim so it can be matched back."
)


def _build_realization_prompt(
    job_schema: dict,
    max_items: int = _DEGRADED_MAX_REALIZED_BULLETS,
) -> tuple[str, str] | None:
    """Build a small system+user prompt asking the local model to realize
    short text for a bounded set of already schema-assigned requirements,
    plus one summary. Returns None when the schema representation has
    nothing supported to realize -- callers should skip the LLM call
    entirely in that case (nothing useful for the model to add) rather
    than send an empty request.

    Only ever draws on requirements the deterministic schema layer already
    marked `supported` (see schemas.build_job_schema_representation) --
    ambiguous/unsupported requirements never reach this prompt, so the
    model has no opportunity to turn an uncertain match into a confident
    sentence. The claim ceiling given here is advisory (a prompt
    instruction) -- request_local_realization enforces it deterministically
    afterward via schemas.check_claim_strength, so a model that ignores
    this instruction still can't produce a claim stronger than the
    evidence supports.
    """
    from applypilot.scoring.schemas import BULLET_SCHEMAS

    supported = [r for r in (job_schema.get("requirements") or []) if r.get("supported") and r.get("schema")][
        :max_items
    ]
    if not supported:
        return None

    lines = [
        f"VIEWPOINT: {job_schema.get('viewpoint', 'general')}",
        f"SUMMARY SCHEMA: {job_schema.get('summary_schema', '')}",
        "",
        "BULLET SLOTS TO FILL (one sentence each):",
    ]
    for r in supported:
        bullet = BULLET_SCHEMAS.get(r["schema"]["bullet_schema"], {})
        if r.get("exact_keywords"):
            anchor = f" | use these exact terms: {', '.join(r['exact_keywords'])}"
        elif r.get("synonym_concepts"):
            anchor = " | TRANSFERABLE experience -- do not claim identical domain"
        else:
            anchor = ""
        force_note = (
            " | this is a PREVENTION fact -- name the risk and the outcome it avoided, not just the action taken"
            if r.get("force_relation") == "prevention"
            else ""
        )
        lines.append(
            f"- evidence: {', '.join(r['resume_evidence'])} | "
            f"shape: {' -> '.join(bullet.get('slots', []))} | "
            f"claim ceiling: {r.get('claim_ceiling', 'participation')} | "
            f"agency ceiling: {r.get('agency_ceiling', 'individual_contributor')} | "
            f"fact: {r['requirement']}{anchor}{force_note}"
        )
    user = "\n".join(lines) + "\n\nReturn the JSON now:"
    return _REALIZATION_SYSTEM, user


def _fuzzy_evidence_match(header: str, evidence_name: str) -> bool:
    """Best-effort match between a parsed resume entry's header text and a
    schema representation's evidence name. Not required to be exact --
    profile.json's inventory `name` and the base resume's rendered header
    text aren't guaranteed to be byte-identical, so this checks substring
    containment either direction, case-insensitive. A missed match just
    means that entry's bullets stay verbatim (always safe), never a crash
    or a wrong splice."""
    h = (header or "").strip().lower()
    e = (evidence_name or "").strip().lower()
    if not h or not e:
        return False
    return e in h or h in e


def build_base_resume_model(resume_text: str, profile: dict) -> dict:
    """Deterministically parse the ORIGINAL resume into a COMPLETE resume
    model. No LLM call. Always has every key (title/summary/skills/
    experience/projects/education) with a usable value -- this is the
    fallback-of-last-resort shape, and it's also exactly what
    validate_json_fields/assemble_resume_text expect, so "do nothing at
    all" (no realization available) is always a structurally valid result,
    never a partial one.

    "experience"/"projects" entries here use scoring/pdf.py's parse_entries
    shape ({"title", "subtitle", "meta", "bullets"}) rather than the
    tailor.py JSON contract's {"header", "subtitle", "bullets"} -- see
    merge_realization, which does that translation at the one place it's
    needed.
    """
    from applypilot.scoring.pdf import parse_entries, parse_resume, parse_skills

    parsed = parse_resume(resume_text or "")
    sections = parsed.get("sections") or {}

    skills = {cat: val for cat, val in parse_skills(sections.get("TECHNICAL SKILLS", ""))}
    if not skills:
        # The resume's TECHNICAL SKILLS section didn't parse (unexpected
        # format) -- fall back to the profile's own skills_boundary rather
        # than a structurally-present-but-empty skills dict (validate_
        # json_fields treats an empty "skills" the same as a missing one).
        boundary = (profile or {}).get("skills_boundary") or {}
        for category, items in boundary.items():
            if isinstance(items, list) and items:
                skills[category.replace("_", " ").title()] = ", ".join(items)

    return {
        "title": parsed.get("title") or "",
        "summary": sections.get("SUMMARY", ""),
        "skills": skills,
        "experience": parse_entries(sections.get("EXPERIENCE", "")),
        "projects": parse_entries(sections.get("PROJECTS", "")),
        "education": sections.get("EDUCATION", ""),
    }


def request_local_realization(
    client,
    job: dict,
    job_schema: dict,
    profile: dict | None = None,
) -> tuple[dict | None, dict]:
    """The ONE bounded local-model call for DEGRADED MODE. Returns
    (realization, meta):
      realization -- {"summary": str | None, "bullets": {evidence_name: text}}
        or None if there was nothing to realize, or the call/parse failed.
        ALWAYS partial by contract -- callers must merge this onto a
        complete base resume model (see merge_realization), never treat it
        as a resume on its own.
      meta -- {"llm_called": bool, "realized_bullets": int,
        "prompt_chars": int, "max_tokens": int} -- prompt_chars/max_tokens
        are logged so a slow call is diagnosable from the run log without
        re-instrumenting by hand.

    Never raises: an exception from the call or an unparseable response
    both result in (None, meta) -- the caller falls back to the verbatim
    base model.
    """
    meta: dict = {
        "llm_called": False,
        "realized_bullets": 0,
        "prompt_chars": 0,
        "max_tokens": 0,
        "claim_strength_violations": 0,
        "passive_voice_warnings": 0,
    }

    prompt = _build_realization_prompt(job_schema)
    if prompt is None:
        log.info(
            "Degraded-mode realization skipped for %s: no schema-supported "
            "requirements to realize -- keeping the original resume verbatim.",
            (job.get("title") or "")[:40],
        )
        return None, meta

    system, user = prompt
    max_tokens = int(os.environ.get("APPLYPILOT_LOCAL_LLM_MAX_TOKENS", "600"))
    meta["llm_called"] = True
    meta["prompt_chars"] = len(system) + len(user)
    meta["max_tokens"] = max_tokens
    # Rough token estimate (chars/4, the usual ballpark for English text) --
    # not exact, but enough to sanity-check "this is small" from the log
    # without pulling in a real tokenizer.
    approx_tokens = meta["prompt_chars"] // 4
    log.info(
        "Degraded-mode realization prompt for %s: %d chars (~%d tokens), max_tokens=%d, %d requirement(s) to realize.",
        (job.get("title") or "")[:40],
        meta["prompt_chars"],
        approx_tokens,
        max_tokens,
        user.count("- evidence:"),
    )

    try:
        t0 = time.time()
        raw = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        elapsed = time.time() - t0
        log.info("Degraded-mode realization call for %s finished in %.1fs", (job.get("title") or "")[:40], elapsed)
        parsed_realization = _parse_plan(raw)
    except Exception as exc:  # noqa: BLE001 -- this function's docstring
        # guarantees "never raises" (an LLMClient.chat() failure and a
        # _parse_plan() JSON failure must both degrade to (None, meta), not
        # propagate) -- narrowing this would break that documented contract.
        log.warning(
            "Degraded-mode realization call failed for %s: %s: %s",
            (job.get("title") or "")[:40],
            type(exc).__name__,
            exc,
        )
        return None, meta

    if not isinstance(parsed_realization, dict):
        log.warning(
            "Degraded-mode realization for %s: response did not parse as a "
            "JSON object -- keeping the original resume verbatim.",
            (job.get("title") or "")[:40],
        )
        return None, meta

    # Deterministic claim-strength enforcement (schemas.py #22): the prompt
    # ASKS the model to respect each slot's claim ceiling, but a small
    # model can still ignore instructions -- this is the actual safety
    # net. Any bullet whose realized text uses a stronger verb than its
    # evidence supports is dropped outright (never downgraded/rewritten,
    # which risks producing an ungrammatical patch) -- merge_realization
    # then falls back to that entry's ORIGINAL bullets, which is always
    # safe. The advisory ceiling was already shown to the model per-slot
    # in the prompt; this check doesn't need that context again, only the
    # evidence-name -> ceiling mapping.
    from applypilot.scoring.schemas import (
        AGENCY_TIERS,
        CLAIM_TIERS,
        check_agency_strength,
        check_causal_claim,
        check_claim_strength,
        check_metric_fabrication,
        check_passive_voice,
    )

    known_metrics = ((profile or {}).get("resume_facts") or {}).get("real_metrics") or []

    ceiling_by_evidence: dict[str, str] = {}
    agency_ceiling_by_evidence: dict[str, str] = {}
    provenance_by_evidence: dict[str, str] = {}
    for r in job_schema.get("requirements") or []:
        if not r.get("supported"):
            continue
        for name in r.get("resume_evidence") or []:
            ceiling_by_evidence[name] = r.get("claim_ceiling") or "participation"
            agency_ceiling_by_evidence[name] = r.get("agency_ceiling") or "individual_contributor"
            provenance = r.get("provenance") or []
            if provenance:
                first = provenance[0]
                provenance_by_evidence[name] = first.get("text", "") if isinstance(first, dict) else str(first)

    bullets: dict[str, str] = {}
    violations = 0
    passive_warnings = 0
    for b in _as_list(parsed_realization.get("bullets")):
        if not (isinstance(b, dict) and b.get("evidence") and b.get("text")):
            continue
        evidence_name = str(b["evidence"]).strip()
        text = str(b["text"]).strip()
        evidence_text = provenance_by_evidence.get(evidence_name, "")
        ceiling = ceiling_by_evidence.get(evidence_name, "participation")
        strength_check = check_claim_strength(text, ceiling)
        if not strength_check["passed"]:
            violations += 1
            log.warning(
                "Degraded-mode realization dropped a bullet for %s (%s): %s",
                (job.get("title") or "")[:40],
                evidence_name,
                strength_check["violation"],
            )
            continue
        agency_ceiling = agency_ceiling_by_evidence.get(evidence_name, "individual_contributor")
        agency_check = check_agency_strength(text, agency_ceiling)
        if not agency_check["passed"]:
            violations += 1
            log.warning(
                "Degraded-mode realization dropped a bullet for %s (%s): %s",
                (job.get("title") or "")[:40],
                evidence_name,
                agency_check["violation"],
            )
            continue
        causal_check = check_causal_claim(text, evidence_text)
        if not causal_check["passed"]:
            violations += 1
            log.warning(
                "Degraded-mode realization dropped a bullet for %s (%s): %s",
                (job.get("title") or "")[:40],
                evidence_name,
                causal_check["violation"],
            )
            continue
        metric_check = check_metric_fabrication(text, evidence_text, known_metrics=known_metrics)
        if not metric_check["passed"]:
            violations += 1
            log.warning(
                "Degraded-mode realization dropped a bullet for %s (%s): %s",
                (job.get("title") or "")[:40],
                evidence_name,
                metric_check["violation"],
            )
            continue
        if not check_passive_voice(text)["passed"]:
            # Soft signal only -- logged, not dropped. See schemas.py's
            # check_passive_voice docstring for the false-positive rationale.
            passive_warnings += 1
            log.info(
                "Degraded-mode realization: possible passive voice for %s (%s): %r",
                (job.get("title") or "")[:40],
                evidence_name,
                text,
            )
        bullets[evidence_name] = text

    summary = str(parsed_realization.get("summary") or "").strip() or None
    if summary:
        # The summary aggregates multiple evidence items -- its ceiling is
        # the strongest any of THIS job's supported requirements' evidence
        # actually earns, not an unconstrained one.
        summary_ceiling = max(
            ceiling_by_evidence.values(),
            key=lambda t: CLAIM_TIERS.index(t) if t in CLAIM_TIERS else 0,
            default="participation",
        )
        summary_check = check_claim_strength(summary, summary_ceiling)
        if summary_check["passed"]:
            summary_agency_ceiling = max(
                agency_ceiling_by_evidence.values(),
                key=lambda t: AGENCY_TIERS.index(t) if t in AGENCY_TIERS else 0,
                default="individual_contributor",
            )
            summary_check = check_agency_strength(summary, summary_agency_ceiling)
        if not summary_check["passed"]:
            violations += 1
            log.warning(
                "Degraded-mode realization dropped the summary for %s: %s",
                (job.get("title") or "")[:40],
                summary_check["violation"],
            )
            summary = None

    meta["realized_bullets"] = len(bullets)
    meta["claim_strength_violations"] = violations
    meta["passive_voice_warnings"] = passive_warnings

    if not bullets and not summary:
        return None, meta
    return {"summary": summary, "bullets": bullets}, meta


def merge_realization(base_resume: dict, realization: dict | None, job: dict) -> dict:
    """Deterministically merge a PARTIAL realization onto the COMPLETE base
    resume model. Never drops a section the realization didn't touch --
    realization is None or missing a key means "keep the base model's
    value for that key", always.

    Returns a dict in tailor.py's JSON contract shape (title/summary/
    skills/experience/projects/education, with experience/projects entries
    as {"header", "subtitle", "bullets"}) -- ready for validate_json_fields/
    assemble_resume_text exactly as-is.
    """
    realization = realization or {}
    bullets_by_evidence: dict[str, str] = realization.get("bullets") or {}

    def _translate(entries: list[dict]) -> list[dict]:
        out: list[dict] = []
        for e in entries:
            header = (e.get("title") or "").strip()
            bullets = list(e.get("bullets") or [])
            realized = next(
                (text for name, text in bullets_by_evidence.items() if _fuzzy_evidence_match(header, name)),
                None,
            )
            if realized:
                bullets = [realized] + [b for b in bullets if b != realized][:2]
            subtitle = " | ".join(p for p in (e.get("subtitle", ""), e.get("meta", "")) if p)
            out.append({"header": header, "subtitle": subtitle, "bullets": bullets[:4]})
        return out

    return {
        "title": job.get("title") or base_resume.get("title") or "",
        "summary": realization.get("summary") or base_resume.get("summary") or "",
        "skills": base_resume.get("skills") or {},
        "experience": _translate(base_resume.get("experience") or []),
        "projects": _translate(base_resume.get("projects") or []),
        "education": base_resume.get("education") or "",
    }


def compose_degraded_resume_json(
    client,
    resume_text: str,
    job: dict,
    profile: dict,
    job_schema: dict,
) -> tuple[dict, dict]:
    """Orchestrates DEGRADED MODE: build_base_resume_model (always
    complete) -> request_local_realization (always partial, the ONE local
    call) -> merge_realization (deterministic, never drops an untouched
    section).

    Returns (data, meta):
      data -- matches tailor.py's normal JSON contract, so
        validate_json_fields, assemble_resume_text, and
        judge_tailored_resume all work completely unchanged on the result.
        ALWAYS structurally complete, even when realization is None.
      meta -- {"tier": "degraded_structured", "llm_called": bool,
        "realized_bullets": int, "prompt_chars": int, "max_tokens": int}.

    2026-08-23: tailor.py's tailor_resume() no longer calls this directly
    -- it calls build_base_resume_model/request_local_realization/
    merge_realization itself, because it needs to see the RAW realization
    result (None vs. populated) to distinguish "nothing was safely
    groundable" / "the local model failed" from "content was genuinely
    realized", which this function's always-merged return value collapses
    together. Kept as a standalone, independently useful/tested entry
    point (e.g. for introspection) -- not dead code, just not on the
    pipeline's critical path anymore.
    """
    base_resume = build_base_resume_model(resume_text, profile)
    realization, meta = request_local_realization(client, job, job_schema)
    meta["tier"] = "degraded_structured"
    data = merge_realization(base_resume, realization, job)
    return data, meta
