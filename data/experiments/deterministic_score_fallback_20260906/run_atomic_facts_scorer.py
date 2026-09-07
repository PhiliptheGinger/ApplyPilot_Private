"""V2 of the decomposed scorer, per direct user correction: v1 (tier A/B/C/D)
still bundled multiple facts (occupation family + experience-gap + degree)
into ONE compound letter choice the model had to blend itself -- same shape
of task that failed before, just with fewer options. It showed a real
letter-position bias (defaulted to B regardless of content).

This version asks the model exactly ONE atomic, genuinely single-fact
question (occupation family), and gets every other fact (years required,
degree required) via deterministic regex extraction -- no LLM judgment
involved in those at all. A plain Python function then combines the
checkbox answers into a score. This mirrors _check_ineligible's own
"regex for what's mechanical, model only for what's genuinely semantic"
split, just applied one level deeper.
"""
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

os.environ["APPLYPILOT_LOCAL_OLLAMA_NATIVE"] = "1"

from applypilot import config  # noqa: E402

config.load_env()

from applypilot import database  # noqa: E402
from applypilot.config import load_profile  # noqa: E402
from applypilot.llm import LLMClient, ModelEntry, local_openai_base_url  # noqa: E402
from applypilot.scoring.scorer import _check_ineligible  # noqa: E402

LOCAL_URL = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_MODEL = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", "qwen3:1.7b")

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

# Generalized past tailor.py's _SENIOR_YEARS_RE (which only targets 8+/senior-tier) --
# this needs the full graded picture (0/1/2+) to feed the deterministic table below,
# not just a hard disqualifying threshold.
_YEARS_MENTION_RE = re.compile(
    r"\b(\d{1,2})\+?\s*years?\b[^.\n]{0,50}\b(?:of\s+)?(?:professional\s+)?experience\b",
    re.IGNORECASE,
)
# 2026-09-06 bug found via live testing: a years-mention alone doesn't
# distinguish "3+ years required" from "3+ years preferred" -- the rubric's
# real B-vs-C boundary. A real Desktop Support job (fit_score=8) mentioned
# "3 years" but scored a false low (5) because the old extractor treated any
# years-mention as a hard requirement. Now requires "required"/"must have"
# within the same nearby window, not just a bare number.
_REQUIRED_CONTEXT_RE = re.compile(r"\b(?:required|must have|minimum of)\b", re.IGNORECASE)
_CS_DEGREE_REQUIRED_RE = re.compile(
    r"\b(?:bachelor'?s?|b\.?s\.?)\s+degree\b[^.\n]{0,60}\b"
    r"(?:computer science|computer engineering|software engineering)\b[^.\n]{0,30}\brequired\b"
    r"|\brequired\b[^.\n]{0,30}\b(?:bachelor'?s?|b\.?s\.?)\s+degree\b[^.\n]{0,60}\b"
    r"(?:computer science|computer engineering|software engineering)\b",
    re.IGNORECASE,
)


def extract_years_required(description: str) -> int | None:
    """Minimum years-of-experience explicitly stated as REQUIRED (not just
    "preferred"), or None if no such hard requirement is found. Only counts
    a years-mention if "required"/"must have"/"minimum of" appears within
    a 100-char window around it -- a bare years-mention with no required-
    context is treated as not-a-hard-requirement (None), matching the
    rubric's real B-vs-C distinction between "preferred" and "required"
    stretch items. Takes the SMALLEST qualifying number in the first 3000
    chars (multiple mentions in a real posting are often the SAME
    requirement restated for different sub-skills; erring toward the
    lower bound is safer than over-rejecting a real match)."""
    text = (description or "")[:3000]
    qualifying: list[int] = []
    for m in _YEARS_MENTION_RE.finditer(text):
        window = text[max(0, m.start() - 60) : m.end() + 60]
        if _REQUIRED_CONTEXT_RE.search(window):
            qualifying.append(int(m.group(1)))
    if not qualifying:
        return None
    return min(qualifying)


def extract_cs_degree_required(description: str) -> bool:
    return bool(_CS_DEGREE_REQUIRED_RE.search((description or "")[:3000]))


def local_only_client() -> LLMClient:
    client = LLMClient(LOCAL_URL, LOCAL_MODEL, "", quality=True)
    client._fallback_chain = [ModelEntry(LOCAL_MODEL, "local", local_openai_base_url(LOCAL_URL), "")]
    return client


def classify_family(client, job) -> str | None:
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
    """Pure Python, zero LLM. Reconstructs the real SCORE_PROMPT_TEMPLATE's
    9-10/7-8/5-6/3-4 logic as an explicit table over the three checkbox
    facts, instead of asking the model to blend them itself.

    2026-09-06, second real finding at n=58: the original single broad
    "hands_on_support" bucket swept in genuinely different occupations
    (mechanical assembly, lab/QC work, formal structural/IC-design
    engineering, enterprise IT service delivery) that sound "hands-on" or
    "technical" by title but don't actually share transferable overlap with
    THIS candidate's specific demonstrated background (automotive repair,
    warehouse, installation, sales, IT support at the CompTIA A+ level) --
    12 false positives, all with this exact shape. Split into narrower,
    candidate-specific families; anything that doesn't clearly match one of
    them now defaults conservative (specialized_or_other), not optimistic."""
    if family is None:
        return 5  # couldn't classify -- neutral, not a guess in either direction
    if family == "specialized_or_other":
        # Deliberately conservative, not neutral: this bucket exists
        # specifically to catch occupations that sound plausible but don't
        # match the candidate's real background (manufacturing, lab/QC,
        # formal engineering, enterprise IT management, etc.) -- assuming
        # "unknown" here repeats the exact false-positive pattern just found.
        return 3
    if family in ("it_or_tech_support", "hands_on_repair_or_trade"):
        if cs_degree:
            return 5  # unusual for this family, but a hard-ish requirement caps it
        if years is None or years == 0:
            return 9
        if years == 1:
            return 7
        return 5  # years >= 2
    if family == "customer_facing_or_sales":
        # 2026-09-06: reverted a same-night overcorrection. A single miss
        # (Sales Development Rep, real=6, predicted=9) led to capping this
        # family's default at 7 -- but the very next larger run (n=58) showed
        # that cap suppressing MULTIPLE real 8-10 matches (Greeter/Counter
        # Desk Attendant, Universal Banker, Store Operations Administrator)
        # that should have scored high. One anecdote isn't a pattern; net
        # effect of the cap was more harm than good. Back to the same
        # optimistic default as the other direct-match families.
        if cs_degree:
            return 5
        if years is None or years == 0:
            return 9
        if years == 1:
            return 7
        return 5
    if family == "software_engineering":
        if cs_degree:
            return 3
        if years == 0:
            return 7  # explicitly states no experience required -- genuinely entry-level
        if years is None:
            # 2026-09-06 real finding: a software-engineering posting with NO
            # explicit years-required number is often still senior-scoped via
            # ownership/scope language regex can't see (e.g. a real "DevOps
            # Engineer... take ownership of cloud infrastructure... exceptional"
            # posting, real fit_score=1, no years regex match at all). Defaulting
            # to "entry-level" here was a false positive in the wrong direction --
            # a missed good match just stays unscored/retryable, a false "good
            # fit" risks auto-tailoring toward a bad match. Uncertain -> neutral,
            # not optimistic.
            return 5
        return 3  # years >= 1 required professional experience


def decomposed_score(client, profile, job) -> tuple[int, str]:
    reason = _check_ineligible(job, profile)
    if reason:
        return 2, f"gate:{reason[:40]}"
    family = classify_family(client, job)
    years = extract_years_required(job.get("full_description") or "")
    cs_degree = extract_cs_degree_required(job.get("full_description") or "")
    score = deterministic_combine(family, years, cs_degree)
    return score, f"family:{family} years:{years} cs_degree:{cs_degree}"


def main():
    conn = database.get_connection()
    profile = load_profile()

    rows = conn.execute(
        """
        SELECT url, title, site, location, fit_score, full_description FROM jobs
        WHERE fit_score IS NOT NULL
          AND (score_reasoning IS NULL OR score_reasoning NOT LIKE '%Ineligible:%')
          AND full_description IS NOT NULL
          AND scored_at >= '2026-08-28'
          AND full_description NOT LIKE '%Replace All Your Work Tools%'
          AND full_description NOT LIKE '%official careers website%'
        """
    ).fetchall()
    rows = [dict(r) for r in rows]
    positives = [r for r in rows if r["fit_score"] >= 8]
    negatives = [r for r in rows if r["fit_score"] < 8]
    import random

    random.seed(20260906)
    random.shuffle(positives)
    random.shuffle(negatives)
    N = int(os.environ.get("N_SAMPLE", "10"))
    sample = positives[: N // 2] + negatives[: N - N // 2]
    random.shuffle(sample)
    print(f"clean pool: {len(rows)} (pos={len(positives)}, neg={len(negatives)}); testing n={len(sample)}")

    client = local_only_client()
    results = []
    for i, job in enumerate(sample):
        t0 = time.time()
        score, detail = decomposed_score(client, profile, job)
        dt = time.time() - t0
        results.append({**job, "local_score": score, "detail": detail, "seconds": round(dt, 1)})
        print(f"[{i+1}/{len(sample)}] real={job['fit_score']} local={score} ({dt:.1f}s, {detail}) {job['title'][:40]}")

    have = [r for r in results if r["local_score"] is not None]
    n = len(have)
    print(f"\nn={len(results)}, parseable={n} ({100*n/len(results):.0f}%)")
    exact = sum(1 for r in have if r["local_score"] == r["fit_score"])
    within2 = sum(1 for r in have if abs(r["local_score"] - r["fit_score"]) <= 2)
    real_pos = [r for r in have if r["fit_score"] >= 8]
    tp = sum(1 for r in real_pos if r["local_score"] >= 8)
    fn = len(real_pos) - tp
    pred_pos = [r for r in have if r["local_score"] >= 8]
    fp = sum(1 for r in pred_pos if r["fit_score"] < 8)
    recall = tp / len(real_pos) if real_pos else float("nan")
    precision = tp / len(pred_pos) if pred_pos else float("nan")
    gate_agree = sum(1 for r in have if (r["local_score"] >= 8) == (r["fit_score"] >= 8)) / n
    avg_seconds = sum(r["seconds"] for r in have) / n
    print(f"exact match: {100*exact/n:.0f}%  within +-2: {100*within2/n:.0f}%  gate_agree: {100*gate_agree:.0f}%")
    print(f"real positives (>=8): n={len(real_pos)}  TP={tp} FN={fn}  recall={recall:.2f} precision={precision:.2f}")
    print(f"avg latency: {avg_seconds:.1f}s")

    import json

    with open(Path(__file__).parent / "atomic_facts_scorer_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
