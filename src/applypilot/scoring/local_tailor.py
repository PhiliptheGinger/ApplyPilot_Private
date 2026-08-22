"""Local LLM support: a cheap, structured resume/job matching PLANNER.

The local model's job is narrow and deliberate: given a job description and
the candidate's existing resume, identify what ALREADY EXISTS that should be
emphasized, reordered, or lightly reworded -- never to invent new facts.  It
is a semantic preprocessor, not a resume writer.  The stronger cloud model
remains responsible for final prose quality and can reject a bad local
recommendation because it always receives the original resume and job too.

Public API
----------
  rank_profile_evidence(job, profile, top_n=6) -> list[dict]
      Deterministic (no LLM, no embeddings) retrieval: ranks experience_
      inventory/project_inventory/skills_inventory entries against the job
      description by term overlap with their existing relevance_categories/
      factual_concepts/name fields. Reduces the problem handed to the local
      model instead of a flat resume dump. Each result records WHY it
      matched (matched_terms) so the retrieval is inspectable, not a black
      box -- see the `applypilot debug-local-plan` command.

  format_evidence_for_prompt(ranked) -> str
      Renders ranked evidence (plus each item's existing hand-written
      `constraints`) as compact text for the local model's prompt.

  get_local_tailoring_plan(resume_text, job, profile) -> dict | None
      Calls the local model with the job, deterministically-extracted
      requirement lines, deterministically-retrieved candidate evidence,
      and a short resume excerpt; parses its structured JSON plan; runs it
      through validate_local_plan() (grounding + fabrication checks) before
      returning it.  Returns None if the local model is unreachable, the
      response is empty, or the output can't be parsed as JSON at all.

  format_local_plan_for_cloud(plan) -> str
      Renders a validated plan as compact text for the cloud model's user
      message.  Returns "" for an empty/all-dropped plan so callers can use
      `if rendered:` to decide whether to include a "TAILORING GUIDANCE"
      block at all.

  validate_local_plan(plan, profile, resume_text, extra_grounding_text="") -> dict
      Deterministic, code-only grounding checks (no LLM call).  Drops any
      requirement/skill/bullet/keyword/rewrite that cannot be traced back to
      text that already exists in the resume or extra_grounding_text (the
      retrieved evidence block).  Never raises; a completely ungrounded
      plan just comes back empty.

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

_DEFAULT_LOCAL_URL = "http://localhost:11434/v1"
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


def _extract_requirement_lines(description: str, max_lines: int = 10) -> list[dict]:
    """Pull bullet/numbered lines from the job description and tag each as
    required/preferred/unspecified via simple keyword sniffing.

    Pure text processing -- no LLM call. Gives the local model (and the
    debug command) a pre-highlighted starting point instead of re-deriving
    requirements from a wall of text every time; it's a hint the model can
    still disagree with, not a hard constraint.
    """
    if not description:
        return []
    lines: list[dict] = []
    seen: set[str] = set()
    for match in _REQUIREMENT_MARKER_RE.finditer(description):
        text = match.group(1).strip()
        if not text or len(text) < 8 or len(text) > 220:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        if _PREFERRED_HINT_RE.search(text):
            importance = "preferred"
        elif _REQUIRED_HINT_RE.search(text):
            importance = "required"
        else:
            importance = "unspecified"
        lines.append({"text": text, "importance": importance})
        if len(lines) >= max_lines:
            break
    return lines


def _format_requirement_lines(lines: list[dict]) -> str:
    if not lines:
        return ""
    return "\n".join(f"- [{l['importance']}] {l['text']}" for l in lines)


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
    """Rank experience_inventory/project_inventory/skills_inventory items
    against the job description via deterministic term overlap.

    Items marked resume_allowed=False (private/unfinished) are excluded --
    same rule validator.py already applies elsewhere for these inventories.
    Only items with at least one matched term are returned, sorted by score
    (matched-term count) descending, ties broken by inventory order.

    Returns entries shaped for both prompt-building and the debug CLI:
      {"type": "experience" | "project" | "skill", "name": str, "score": int,
       "matched_terms": list[str], "item": dict}
    `matched_terms` is exactly what makes this inspectable -- the specific
    job-description terms/categories that caused the match, not a black box.
    """
    haystack = _job_text_lower(job)
    sources = (
        ("experience", profile.get("experience_inventory") or []),
        ("project", profile.get("project_inventory") or []),
        ("skill", profile.get("skills_inventory") or []),
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
    """Render ranked evidence as compact text for the local model's prompt.

    Includes each item's existing per-item `constraints` (project_inventory
    already carries hand-written anti-fabrication notes, e.g. "do not claim
    production deployment without explicit evidence") so the local model
    sees the same factual guardrails a human reviewer already wrote.
    """
    if not ranked:
        return ""
    lines: list[str] = []
    for r in ranked:
        item, kind, name = r["item"], r["type"], r["name"]
        if kind == "skill":
            level = item.get("evidence_level", "")
            lines.append(f"- [skill] {name}" + (f" ({level})" if level else ""))
        elif kind == "project":
            concepts = ", ".join(c for c in (item.get("factual_concepts") or []) if isinstance(c, str))
            lines.append(f"- [project] {name}" + (f": {concepts}" if concepts else ""))
            for c in item.get("constraints") or []:
                if isinstance(c, str) and c.strip():
                    lines.append(f"    constraint: {c.strip()}")
        else:  # experience
            desc = (item.get("description") or "")[:200]
            role_type = item.get("role_type", "")
            header = f"- [experience] {name}" + (f" ({role_type})" if role_type else "")
            lines.append(header + (f": {desc}" if desc else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured tailoring PLAN (the normal, recommended local-model role)
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You are a resume-to-job matching analyst. Do NOT rewrite the resume. "
    "Compare the JOB to the RESUME (and CANDIDATE EVIDENCE, if provided) and output ONLY "
    "compact JSON \u2014 no prose, no fences, no explanation.\n\n"
    "RULES:\n"
    "- Only reference skills, tools, employers, projects, and facts that ALREADY appear in "
    "the RESUME or CANDIDATE EVIDENCE sections below.\n"
    "- If a requirement has no evidence in either section, list it in unsupported_requirements. "
    "Do NOT invent evidence.\n"
    "- bullets_to_prioritize and bullets_to_deemphasize must quote or closely paraphrase EXISTING resume bullets.\n"
    "- safe_rewrites must preserve every fact (same employer, same numbers, same tools) \u2014 wording only.\n\n"
    "Schema (all keys required, empty arrays if nothing applies):\n"
    "{\n"
    '  "requirements": [{"requirement": "...", "importance": "required or preferred", '
    '"resume_evidence": ["..."], "supported": true or false}],\n'
    '  "skills_to_emphasize": ["..."],\n'
    '  "bullets_to_prioritize": ["..."],\n'
    '  "bullets_to_deemphasize": ["..."],\n'
    '  "matching_projects": ["..."],\n'
    '  "matching_certifications": ["..."],\n'
    '  "summary_focus": ["..."],\n'
    '  "keyword_targets": ["..."],\n'
    '  "safe_rewrites": [{"original": "...", "suggested": "..."}],\n'
    '  "unsupported_requirements": ["..."]\n'
    "}"
)

# List-typed keys in the plan schema (used for defensive type-coercion when
# the local model's output has a malformed field, e.g. a string where a list
# was expected -- a common small-model mistake).
_PLAN_LIST_KEYS = (
    "requirements", "skills_to_emphasize", "bullets_to_prioritize",
    "bullets_to_deemphasize", "matching_projects", "matching_certifications",
    "summary_focus", "keyword_targets", "safe_rewrites", "unsupported_requirements",
)


def get_local_tailoring_plan(
    resume_text: str,
    job: dict,
    profile: dict,
) -> dict | None:
    """Ask the local model for a structured tailoring plan and validate it.

    Returns the sanitized plan dict (see validate_local_plan) or None if the
    local model is unavailable, the response is empty, or it isn't parseable
    as JSON at all. A plan that IS parseable but fails every grounding check
    still comes back as a dict with empty list fields -- callers should treat
    that the same as "no useful guidance" via format_local_plan_for_cloud().

    Rather than sending the whole (truncated) resume as the sole evidence
    source, this also deterministically ranks experience_inventory/
    project_inventory/skills_inventory items against the job description
    (see rank_profile_evidence) and includes a curated CANDIDATE EVIDENCE
    section built ONLY from that already-vetted profile data -- reducing
    the problem handed to the local model instead of relying on it to find
    relevance in a large, undifferentiated block of text. The resume
    excerpt is kept too, shortened, purely for narrative/style continuity.

    Inputs are trimmed so the whole prompt fits within a local model's context:
    resume excerpt to 800 chars, description to 2 000 chars, evidence to
    `top_n` (default 6, override via APPLYPILOT_LOCAL_EVIDENCE_TOPN) items.
    """
    url = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", _DEFAULT_LOCAL_URL).rstrip("/")
    model = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", _DEFAULT_LOCAL_MODEL)
    timeout = float(os.environ.get("APPLYPILOT_LOCAL_LLM_TIMEOUT", "60"))

    # Avoid circular import
    from applypilot.scoring.tailor import display_company

    company = display_company(job)
    job_text = (
        f"TITLE: {job.get('title', '')}\n"
        f"COMPANY: {company or 'unknown'}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:2000]}"
    )

    top_n = int(os.environ.get("APPLYPILOT_LOCAL_EVIDENCE_TOPN", "6"))
    ranked_evidence = rank_profile_evidence(job, profile, top_n=top_n)
    evidence_text = format_evidence_for_prompt(ranked_evidence)
    requirement_lines = _extract_requirement_lines(job.get("full_description") or "")
    requirements_text = _format_requirement_lines(requirement_lines)

    resume_excerpt = resume_text[:800]

    user_msg_parts = [f"JOB:\n{job_text}"]
    if requirements_text:
        user_msg_parts.append(
            f"LIKELY REQUIREMENTS (auto-extracted from the posting; verify against "
            f"the full JOB text above, this may be incomplete):\n{requirements_text}"
        )
    if evidence_text:
        user_msg_parts.append(
            f"CANDIDATE EVIDENCE (already-verified facts from the candidate's profile "
            f"-- treat as equally authoritative to the resume below):\n{evidence_text}"
        )
    user_msg_parts.append(f"RESUME (excerpt, for style/context):\n{resume_excerpt}")
    user_msg_parts.append("Return the JSON plan:")
    user_msg = "\n\n".join(user_msg_parts)

    # Qwen3 (and other hybrid-reasoning models served the same way) default
    # to an extended internal "thinking" pass before producing output --
    # confirmed live: an equivalent-shaped request against qwen3:8b via
    # Ollama exceeded a 120s timeout for this planning task, while a bare,
    # short single-user-message probe (no system prompt -- see cli.py's
    # test-local, and the "Qwen3 optimization" in llm.py's LLMClient.chat())
    # returned promptly. /no_think is the same lightweight control token
    # llm.py already relies on for Qwen models; applied here directly since
    # get_local_tailoring_plan talks to the local endpoint itself rather
    # than going through LLMClient.chat() (whose check only fires when the
    # FIRST message has role "user", which isn't the case for a system+user
    # prompt pair like this one).
    if "qwen" in model.lower() and not user_msg.startswith("/no_think"):
        user_msg = f"/no_think\n{user_msg}"

    payload: dict = {
	    "model": model,
	    "messages": [
		    {"role": "system", "content": _PLAN_SYSTEM},
	    	{"role": "user", "content": user_msg},
	    ],
	    "temperature": 0.0,
	    "max_tokens": int(os.environ.get("APPLYPILOT_LOCAL_LLM_MAX_TOKENS", "700")),
	    "response_format": {"type": "json_object"},
    }

    try:
        resp = httpx.post(f"{url}/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            log.warning("Local tailoring plan for %s: empty choices in response",
                        (job.get("title") or "")[:40])
            return None
        text = (choices[0].get("message") or {}).get("content", "").strip()
        if not text:
            log.warning("Local tailoring plan for %s: empty message content in response",
                        (job.get("title") or "")[:40])
            return None
        raw_plan = _parse_plan(text)
        if not isinstance(raw_plan, dict):
            log.warning("Local tailoring plan for %s: parsed JSON was not an object (got %s)",
                        (job.get("title") or "")[:40], type(raw_plan).__name__)
            return None
        sanitized = validate_local_plan(
            raw_plan, profile, resume_text, extra_grounding_text=evidence_text,
        )
        log.debug(
            "Local tailoring plan for %s: %d requirement(s), %d bullet(s) prioritized, "
            "%d dropped by validation",
            (job.get("title") or "")[:40], len(sanitized["requirements"]),
            len(sanitized["bullets_to_prioritize"]), len(sanitized["_warnings"]),
        )
        return sanitized
    except httpx.TimeoutException as exc:
        log.warning(
            "Local tailoring plan for %s timed out after %.0fs (model=%s). "
            "Qwen-family models in particular can be slow without /no_think; "
            "raise APPLYPILOT_LOCAL_LLM_TIMEOUT or use a smaller/faster model "
            "if this persists. (%s)",
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
    /no_think requested (see get_local_tailoring_plan), some Ollama builds
    still emit an empty (or populated) <think>...</think> block before the
    real content. Stripped unconditionally before fence/brace extraction --
    a cheap, pure-text operation, not dependent on which failure mode is
    actually occurring for a given model/build.
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
    plan: dict, profile: dict, resume_text: str, extra_grounding_text: str = "",
) -> dict:
    """Deterministically strip any plan entry that isn't grounded in the
    actual resume text (plus `extra_grounding_text`), or that reintroduces
    a fabricated skill/tool.

    `extra_grounding_text` extends the grounding corpus beyond the literal
    resume text -- used to include the deterministically-retrieved
    CANDIDATE EVIDENCE block (see rank_profile_evidence/format_evidence_
    for_prompt) that the local model was shown. Without this, evidence
    phrased in profile.json's experience_inventory/project_inventory
    (rather than verbatim in the resume file) would be wrongly treated as
    ungrounded, since it's real, already-vetted candidate data, not
    something the model invented.

    This is the factual-safety layer for the PLAN (separate from, and much
    narrower than, applypilot.scoring.validator's full resume validation --
    it only checks that the local model's suggestions reference things that
    already exist, never that a whole generated resume is well-formed).

    Never raises. Returns a plan dict with the same shape as the schema,
    plus a "_warnings" key (list[str]) describing what was dropped and why --
    intended for the debug CLI command, not for the cloud prompt.
    """
    from applypilot.scoring.validator import FABRICATION_WATCHLIST, _build_skills_set

    warnings: list[str] = []
    resume_lower = f"{resume_text}\n{extra_grounding_text}".lower()
    allowed_skills = _build_skills_set(profile)
    # Also allow resume-permitted skills_inventory entries -- the new
    # evidence-retrieval layer can surface a skill from skills_inventory
    # that isn't separately listed in skills_boundary; both are equally
    # profile-vetted, so treating only one as authoritative would reject
    # legitimate emphasis recommendations grounded in the new evidence.
    for _skill in profile.get("skills_inventory") or []:
        if (
            isinstance(_skill, dict)
            and _skill.get("resume_allowed") is not False
            and isinstance(_skill.get("name"), str)
            and _skill["name"].strip()
        ):
            allowed_skills.add(_skill["name"].strip().lower())

    def _grounded(text: str, min_word_len: int = 4) -> bool:
        """Loose check: at least one non-trivial word from `text` appears in
        the resume, or the whole (short) phrase appears verbatim."""
        if not text or not text.strip():
            return False
        if text.strip().lower() in resume_lower:
            return True
        words = [w.strip(".,;:()\"'").lower() for w in text.split()]
        significant = [w for w in words if len(w) >= min_word_len]
        if not significant:
            return False
        return any(w in resume_lower for w in significant)

    # requirements: keep, but downgrade "supported" if the claimed evidence
    # doesn't actually appear in the resume.
    clean_requirements: list[dict] = []
    for req in _as_list(plan.get("requirements")):
        if not isinstance(req, dict) or not req.get("requirement"):
            continue
        evidence = [e for e in _as_list(req.get("resume_evidence")) if isinstance(e, str) and e.strip()]
        grounded_evidence = [e for e in evidence if _grounded(e)]
        claimed_supported = bool(req.get("supported"))
        supported = claimed_supported and bool(grounded_evidence)
        if claimed_supported and not grounded_evidence:
            warnings.append(
                f"Downgraded unsupported claim: '{req['requirement']}' had no grounded evidence"
            )
        importance = req.get("importance") if req.get("importance") in ("required", "preferred") else "preferred"
        clean_requirements.append({
            "requirement": str(req["requirement"])[:200],
            "importance": importance,
            "resume_evidence": grounded_evidence[:3],
            "supported": supported,
        })

    # skills_to_emphasize: must be an allowed skill AND appear in the resume.
    clean_skills: list[str] = []
    for s in _as_list(plan.get("skills_to_emphasize")):
        if not isinstance(s, str) or not s.strip():
            continue
        if s.lower() in allowed_skills and _grounded(s):
            clean_skills.append(s)
        else:
            warnings.append(f"Dropped unsupported skill: '{s}'")

    # keyword_targets: domain terms are allowed even if not in skills_boundary,
    # but they must still be grounded in the actual resume text.
    clean_keywords: list[str] = []
    for k in _as_list(plan.get("keyword_targets")):
        if not isinstance(k, str) or not k.strip():
            continue
        if _grounded(k):
            clean_keywords.append(k)
        else:
            warnings.append(f"Dropped ungrounded keyword: '{k}'")

    def _clean_bullets(items: list, label: str) -> list[str]:
        out: list[str] = []
        for b in items:
            if not isinstance(b, str) or not b.strip():
                continue
            if _grounded(b, min_word_len=5):
                out.append(b)
            else:
                warnings.append(f"Dropped unverified {label}: '{b[:60]}'")
        return out

    clean_prioritize = _clean_bullets(_as_list(plan.get("bullets_to_prioritize")), "bullet (prioritize)")
    clean_deemphasize = _clean_bullets(_as_list(plan.get("bullets_to_deemphasize")), "bullet (de-emphasize)")

    clean_projects = [
        p for p in _as_list(plan.get("matching_projects"))
        if isinstance(p, str) and _grounded(p)
    ]
    clean_certs = [
        c for c in _as_list(plan.get("matching_certifications"))
        if isinstance(c, str) and _grounded(c)
    ]

    # safe_rewrites: the ORIGINAL text must be grounded (it's supposed to be
    # an existing bullet/phrase); the SUGGESTED text must not introduce a
    # fabrication-watchlist term unless that term is already an allowed skill.
    clean_rewrites: list[dict] = []
    for rw in _as_list(plan.get("safe_rewrites")):
        if not isinstance(rw, dict):
            continue
        original = str(rw.get("original") or "").strip()
        suggested = str(rw.get("suggested") or "").strip()
        if not original or not suggested:
            continue
        if not _grounded(original, min_word_len=5):
            warnings.append(f"Dropped rewrite with ungrounded original: '{original[:60]}'")
            continue
        suggested_lower = suggested.lower()
        fabricated = [
            f for f in FABRICATION_WATCHLIST
            if len(f) > 2 and f in suggested_lower and not any(f in s for s in allowed_skills)
        ]
        if fabricated:
            warnings.append(
                f"Dropped rewrite introducing fabricated term(s): {', '.join(fabricated)}"
            )
            continue
        clean_rewrites.append({"original": original[:300], "suggested": suggested[:300]})

    # summary_focus: accept either a list of short phrases (schema) or a
    # single string (backward-compat with the earlier one-sentence version).
    raw_summary = plan.get("summary_focus")
    if isinstance(raw_summary, str) and raw_summary.strip():
        summary_focus = [raw_summary.strip()]
    else:
        summary_focus = [s for s in _as_list(raw_summary) if isinstance(s, str) and s.strip()][:5]

    # unsupported_requirements: pass through untouched -- this list represents
    # an ABSENCE of evidence, which is exactly the signal we want preserved so
    # the cloud model knows not to fabricate for these.
    unsupported = [
        u for u in _as_list(plan.get("unsupported_requirements"))
        if isinstance(u, str) and u.strip()
    ][:10]

    return {
        "requirements": clean_requirements[:12],
        "skills_to_emphasize": clean_skills[:10],
        "bullets_to_prioritize": clean_prioritize[:8],
        "bullets_to_deemphasize": clean_deemphasize[:8],
        "matching_projects": clean_projects[:5],
        "matching_certifications": clean_certs[:5],
        "summary_focus": summary_focus,
        "keyword_targets": clean_keywords[:10],
        "safe_rewrites": clean_rewrites[:6],
        "unsupported_requirements": unsupported,
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

