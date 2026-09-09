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

Deliberately NOT wired into the automatic score_job() fallback chain, and
NOT run automatically as part of the normal pipeline. This is real,
imperfect data (81-87% gate agreement, not 100%) -- an honest
score_error/pending-retry is recoverable once quota resets; a bad
deterministic score silently reaching `tailored`/`ready_to_apply` is not
(see the real fabricated-identity-score incident, decision #75, for exactly
what that class of mistake costs). So this module is only ever invoked
explicitly (`applypilot revalidate-deterministic-fallback-scores` covers
the cleanup half; the scoring half is `run_deterministic_fallback_scoring`,
wired to a dedicated CLI flag) against jobs already stuck specifically on a
quota-cooldown score_error -- never against jobs that simply haven't been
scored yet for other reasons. Every fallback-scored row is tagged
`score_method = 'deterministic_fallback'` (see database.py's `_ALL_COLUMNS`)
so it stays visibly distinguishable from a real LLM score and is
revalidation-eligible once quota returns.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime

from applypilot.llm import LLMClient, ModelEntry, local_openai_base_url
from applypilot.scoring.compensation import classify_compensation, compensation_score_adjustment
from applypilot.scoring.scorer import _check_ineligible, _classify_ineligibility, _flush_score_batch

SCORE_METHOD = "deterministic_fallback"

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

_YEARS_MENTION_RE = re.compile(
    r"\b(\d{1,2})\+?\s*years?\b[^.\n]{0,50}\b(?:of\s+)?(?:professional\s+)?experience\b",
    re.IGNORECASE,
)
_REQUIRED_CONTEXT_RE = re.compile(r"\b(?:required|must have|minimum of)\b", re.IGNORECASE)

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
_REQUIRED_SECTION_HEADER_RE = re.compile(
    r"\b(?:minimum|required|basic)\s+qualifications\b",
    re.IGNORECASE,
)
_NEXT_SECTION_HEADER_RE = re.compile(
    r"\b(?:preferred|desired|nice.to.have|bonus)\s+qualifications\b|\bpreferred\s+skills\b|\bnice.to.haves?\b",
    re.IGNORECASE,
)
_CS_DEGREE_REQUIRED_RE = re.compile(
    r"\b(?:bachelor'?s?|b\.?s\.?)\s+degree\b[^.\n]{0,60}\b"
    r"(?:computer science|computer engineering|software engineering)\b[^.\n]{0,30}\brequired\b"
    r"|\brequired\b[^.\n]{0,30}\b(?:bachelor'?s?|b\.?s\.?)\s+degree\b[^.\n]{0,60}\b"
    r"(?:computer science|computer engineering|software engineering)\b",
    re.IGNORECASE,
)


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
    stop_m = _NEXT_SECTION_HEADER_RE.search(text, start)
    end = stop_m.start() if stop_m else min(len(text), start + 1500)
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
    requirement. Takes the smallest qualifying number in the first 3000
    chars."""
    text = (description or "")[:3000]
    qualifying: list[int] = []
    required_span = _required_section_span(text)
    for m in _YEARS_MENTION_RE.finditer(text):
        window = text[max(0, m.start() - 60) : m.end() + 60]
        in_required_section = required_span is not None and required_span[0] <= m.start() < required_span[1]
        if in_required_section or _REQUIRED_CONTEXT_RE.search(window):
            qualifying.append(int(m.group(1)))
    if not qualifying:
        return None
    return min(qualifying)


def extract_cs_degree_required(description: str) -> bool:
    return bool(_CS_DEGREE_REQUIRED_RE.search((description or "")[:3000]))


def local_only_client(model: str) -> LLMClient:
    client = LLMClient(LOCAL_URL, model, "", quality=True)
    client._fallback_chain = [ModelEntry(model, "local", local_openai_base_url(LOCAL_URL), "")]
    return client


def classify_family(client: LLMClient, job: dict) -> str | None:
    job_text = (
        f"TITLE: {job['title']}\nCOMPANY: {job['site']}\n\nDESCRIPTION:\n{(job.get('full_description') or '')[:3000]}"
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
        return 3  # years >= 1 required professional experience
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
) -> dict:
    """Score ONLY jobs currently stuck on a quota-cooldown score_error.
    Never touches a job that hasn't been scored for any other reason --
    this is explicitly a quota-outage rescue, not a general scoring path.

    ``limit`` defaults to DEFAULT_LIMIT (NOT unlimited -- see the 2026-09-07
    near-miss note above); pass ``limit=0`` explicitly to process every
    candidate.

    ``escalate_model``: opt-in title-keyword escalation (see
    score_job_deterministic / _AMBIGUOUS_TITLE_RE) -- off by default.

    Writes are flushed after EVERY job, not batched until the end -- a
    long run (qwen3:8b: ~76s/job) must be safely interruptible without
    losing already-scored progress, and should show real-time DB progress
    to anyone watching, not go silent until the whole run finishes.
    """
    from applypilot.config import load_profile
    from applypilot.database import get_connection

    if conn is None:
        conn = get_connection()
    profile = load_profile()

    rows = conn.execute(
        "SELECT * FROM jobs WHERE fit_score IS NULL AND score_error IS NOT NULL "
        "AND score_error LIKE '%quota cooldown%'"
    ).fetchall()
    jobs = [dict(r) for r in rows]
    if limit:
        jobs = jobs[:limit]

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
