"""Deterministic compensation classification and conservative base-pay
estimation for job postings.

2026-08-29 investigation found the existing commission-only gate
(``scorer._COMMISSION_ONLY_PATTERN`` / ``_check_ineligible``) correctly
rejects genuine commission-only postings and correctly allows base+
commission/base+bonus/ordinary salaried postings -- but a posting with NO
compensation information at all was invisible to scoring: neither rejected
nor flagged, simply scored as if pay had been confirmed. This module closes
that visibility gap without touching the commission-only gate at all.

Four statuses, never collapsed into each other:

- "stated"           -- an unambiguous dollar figure is present in the
                         posting (or the `salary` column). Never a claim
                         from ApplyPilot -- always the employer's own text.
- "explicitly_absent" -- the existing commission-only gate's pattern
                         matches (100% commission / commission-only /
                         straight commission / no base salary / etc.).
                         This module does not re-derive that judgment; it
                         imports and reuses the exact same regex so there
                         is only ever one source of truth for this case.
- "estimated"         -- no dollar figure in this posting, but enough
                         same-employer comparable postings with real
                         stated pay exist in the database to support a
                         conservative range. Always carries a confidence
                         tier and a basis string; never a point estimate.
- "unknown"           -- no dollar figure, and insufficient evidence to
                         estimate. This is the ONLY case where ApplyPilot
                         has genuinely no information to offer, and it is
                         also the expected outcome for MOST unstated-pay
                         postings today -- see the module-level note on the
                         employer-identity heuristic below for why.

Deliberately NOT built: no external salary API, no web scraping, no LLM
call, no general salary-prediction model, no nearest-neighbor engine, no
salary threshold. The estimator uses exactly one evidence path -- other
postings from what is judged to be the SAME employer, with an overlapping
significant title token, that themselves have stated compensation -- and
abstains (returns None / status "unknown") whenever that bar isn't met.
Per the investigation's own live measurement, only 6.6% of described jobs
have anything in the `salary` column, and `company` is essentially never
populated even for postings from single-employer ATS sources (Boeing,
Amazon, Anduril Industries, Databricks, ... all show 0 rows with `company`
set) -- for those single-employer ATS sources, `site` itself already IS
the employer name (the scraper wrote it that way), so `site` is the usable
employer identifier. For aggregator-sourced postings (LinkedIn, Indeed,
Dice, Glassdoor, ...) `site` is the AGGREGATOR, not the employer, and
`company` is also empty for those rows in practice -- there is no reliable
structured employer identifier for them at all today, so the estimator
must and does abstain for them. `_AGGREGATOR_SITES` below is the same
small, explicit, hardcoded denylist convention already used by
view.py's `colors` dict for exactly this "is this a platform, not an
employer" distinction -- reused/extended here, not invented fresh.
"""

from __future__ import annotations

import re

COMPENSATION_STATUSES = frozenset({"stated", "estimated", "unknown", "explicitly_absent"})

STATED_SUBTYPES = frozenset(
    {"annual", "hourly", "base_plus_commission", "base_plus_bonus", "other_stated"}
)

CONFIDENCE_LEVELS = frozenset({"medium", "high"})

# Same denylist convention as view.py's `colors` dict (aggregator display
# names that are NOT the employer). Extended with a few more aggregator/
# job-board names observed directly in the live DB during the 2026-08-29/30
# investigations (Talent.com, Startup.jobs, Nodesk). Lowercased for
# case-insensitive comparison. An unrecognized aggregator slipping through
# this list would be wrongly treated as an employer identity -- a known,
# accepted limitation of a deliberately small hardcoded list, documented
# rather than solved with a bigger classifier here.
_AGGREGATOR_SITES = frozenset(
    {
        "linkedin",
        "indeed",
        "dice",
        "glassdoor",
        "remoteok",
        "welcometothejungle",
        "hacker news jobs",
        "builtin remote",
        "talent.com",
        "startup.jobs",
        "nodesk",
        "ziprecruiter",
        "monster",
        "simplyhired",
    }
)

# ── Stated-compensation detection ───────────────────────────────────────
#
# Deliberately anchored on an actual number. Vague phrases like
# "competitive salary", "generous compensation", "attractive package",
# "market-rate pay", "uncapped earnings" contain no number to point at and
# must NOT be treated as stated -- there is nothing here to distinguish
# "$150K" from "$40K" if the posting itself never says either.
#
# Two shapes are recognized (found necessary by real-data validation
# against the live database, 2026-08-30):
# 1. "$"-prefixed, e.g. "$60,000-$75,000/year", "$20/hour", "$45K". The
#    second side of a range may omit its own "$" (e.g. Intel's
#    "$60,450.00-85,340.00 USD") -- a real, common shorthand; without this
#    the second number fails to match _MONEY at all and the range
#    collapses to a misleading single-value match on the first number only.
# 2. Postfix-currency-code ranges with NO "$" at all, e.g. Amazon's
#    federal-pay-transparency-mandated "60,700.00 - 106,300.00 USD
#    annually" -- deliberately narrow (requires BOTH a "-" range AND a
#    trailing "USD" AND a trailing unit word all together) rather than
#    "any bare number is money", which would false-positive constantly on
#    years/phone numbers/zip codes with no currency signal at all.
_MONEY = r"\$\s?\d[\d,]*(?:\.\d+)?\s?[kK]?"
_MONEY_BARE = r"\d[\d,]*(?:\.\d+)?\s?[kK]?"
_MONEY_RANGE_RE = re.compile(rf"{_MONEY}(?:\s*(?:-|–|—|to)\s*\$?\s?{_MONEY_BARE})?")
_USD_SUFFIX_RANGE_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:-|–|—|to)\s*\d[\d,]*(?:\.\d+)?\s*USD\s*"
    r"(?P<unit>annually|per\s+year|/\s?year|yearly|hourly|per\s+hour|/\s?hour)",
    re.IGNORECASE,
)
_ANNUAL_UNIT_RE = re.compile(r"/\s?(?:year|yr)\b|per\s+year\b|\bannually\b|\byearly\b", re.IGNORECASE)
_HOURLY_UNIT_RE = re.compile(r"/\s?(?:hour|hr)\b|per\s+hour\b|\bhourly\b", re.IGNORECASE)

# 2026-08-30 real-data fix: a live production row (Intel, "Module Equipment
# Technician - Metrology") reads "Annual Salary Range ... $60,450.00-
# 85,340.00 USD (Hourly Role)" -- Intel's own template labels the SAME
# figure both "Annual" (68 chars before the number, just outside
# _CONTEXT_WINDOW) and "(Hourly Role)" (a few chars after, inside the
# window) in one sentence. The bare "hourly" word there describes the
# position's employment classification, not a per-unit rate on this
# number. No real hourly wage is anywhere near $60,450 -- when the bare
# "hourly" match is this implausible, the plain magnitude heuristic below
# (lo >= 1000 -> annual) is trusted instead. Deliberately generous (no
# genuine hourly wage is remotely close to $1,000) so this never
# second-guesses an actual "$45/hour" or "$120/hr" mention.
_MAX_PLAUSIBLE_HOURLY_RATE = 1000
_COMMISSION_WORD_RE = re.compile(r"\bcommission\b", re.IGNORECASE)
_BONUS_WORD_RE = re.compile(r"\bbonus\b", re.IGNORECASE)

# 2026-08-29 real-data fix: a ONE-TIME sign-on/signing/referral/relocation/
# hiring bonus figure is not base pay at all -- it must never be read as
# "base_plus_bonus" annual compensation. Found via a live production run:
# Raytheon's "External candidates will receive a sign-on bonus of $3,000"
# was misread as "$3,000/year", producing a nonsensical $3,000-$90,571
# same-employer estimate for an unrelated Avionics Technician posting.
# Deliberately structural (money figure directly adjacent to the specific
# one-time-bonus phrase), NOT a wide context-window word search like the
# other checks in this module -- "Base salary $80,000 plus a $3,000
# sign-on bonus" must still read $80,000 as real base pay even though
# "sign-on bonus" appears later in the same sentence; only the $3,000
# figure structurally attached to the bonus phrase is excluded.
_ONE_TIME_BONUS_TYPES = r"sign[\s-]?on|signing|referral|relocation|hiring|starting"
_ONE_TIME_BONUS_RE = re.compile(
    rf"(?:{_ONE_TIME_BONUS_TYPES})\s+bonus\s+of\s+(?P<money1>{_MONEY})"
    rf"|(?P<money2>{_MONEY})\s+(?:{_ONE_TIME_BONUS_TYPES})\s+bonus",
    re.IGNORECASE,
)


def _one_time_bonus_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of dollar figures that are structurally a one-time
    bonus amount, not base pay -- these must be excluded from every money
    scan in this module, not just the plain-"bonus" subtype check."""
    spans = []
    for m in _ONE_TIME_BONUS_RE.finditer(text):
        group = "money1" if m.group("money1") else "money2"
        spans.append(m.span(group))
    return spans


def _overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < s_end and end > s_start for s_start, s_end in spans)

# How far around a matched dollar figure to look for a unit/structure word
# (e.g. "/year", "commission", "bonus"). Kept small and local -- this is
# about disambiguating ONE compensation mention, not scanning the whole
# posting for stray unrelated uses of "commission" (that job already
# belongs to _COMMISSION_ONLY_PATTERN, which scans the full description).
_CONTEXT_WINDOW = 60


def _parse_money(token: str) -> float | None:
    """Parse a single "$X,XXX" / "$XXK" / "X,XXX.XX" token to a plain number."""
    token = token.strip().lstrip("$").strip()
    is_k = token.lower().endswith("k")
    if is_k:
        token = token[:-1]
    token = token.replace(",", "")
    try:
        value = float(token)
    except ValueError:
        return None
    return value * 1000 if is_k else value


def _parse_money_range(span: str) -> tuple[float | None, float | None]:
    """Parse a matched "$X" or "$X-$Y" span into (min, max)."""
    parts = re.split(r"\s*(?:-|–|—|to)\s*", span)
    values = [_parse_money(p) for p in parts if p.strip()]
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return min(values), max(values)


def _classify_one_match(text: str, start: int, end: int, lo: float, hi: float | None) -> dict:
    """Build the {subtype, min, max, unit, is_range} candidate for a single
    matched money span, using nearby (+/- _CONTEXT_WINDOW chars) structure/
    unit words to disambiguate."""
    window = text[max(0, start - _CONTEXT_WINDOW) : end + _CONTEXT_WINDOW]
    is_range = hi is not None and hi != lo

    if _COMMISSION_WORD_RE.search(window):
        subtype = "base_plus_commission"
        unit = "annual" if lo >= 1000 else None
    elif _BONUS_WORD_RE.search(window):
        subtype = "base_plus_bonus"
        unit = "annual" if lo >= 1000 else None
    elif _ANNUAL_UNIT_RE.search(window):
        subtype, unit = "annual", "annual"
    elif _HOURLY_UNIT_RE.search(window) and lo <= _MAX_PLAUSIBLE_HOURLY_RATE:
        subtype, unit = "hourly", "hourly"
    elif lo >= 1000:
        # Bare "$45,000" / "$60K" with no unit word: hourly wages are never
        # quoted in this magnitude, so annual is the only sane reading.
        subtype, unit = "annual", "annual"
    else:
        subtype, unit = "other_stated", None

    has_context = unit is not None or subtype in ("base_plus_commission", "base_plus_bonus")
    return {
        "subtype": subtype,
        "min": lo,
        "max": hi if hi is not None else lo,
        "unit": unit,
        "is_range": is_range,
        "has_context": has_context,
        "start": start,
    }


def classify_stated_compensation(text: str) -> dict | None:
    """Return the single most informative unambiguous compensation mention
    in ``text``, or None if nothing unambiguous is found.

    A posting commonly mentions dollar figures more than once (e.g. an
    isolated "up to $100,000" teaser earlier in the copy, followed later by
    the real "$60,300 to $100,000" range) -- 2026-08-30 real-data
    validation found taking the textually-FIRST match picked the wrong one
    in exactly this shape. All matches are scanned; the one preferred is,
    in order: (1) an actual two-sided range over a single bare figure, (2)
    has explicit unit/commission/bonus context nearby over a bare number
    with none, (3) earliest position as a final tiebreaker.

    Returns {"subtype": ..., "min": float, "max": float, "unit": str|None}.
    ``unit`` is "annual"/"hourly" when an explicit unit word is nearby, else
    None (subtype "other_stated" in that case -- a real number exists, but
    ApplyPilot does not guess whether it is a yearly or hourly figure).
    """
    if not text:
        return None

    bonus_spans = _one_time_bonus_spans(text)

    candidates = []
    for m in _MONEY_RANGE_RE.finditer(text):
        if _overlaps_any(m.start(), m.end(), bonus_spans):
            continue
        lo, hi = _parse_money_range(m.group(0))
        if lo is None:
            continue
        candidates.append(_classify_one_match(text, m.start(), m.end(), lo, hi))

    for m in _USD_SUFFIX_RANGE_RE.finditer(text):
        if _overlaps_any(m.start(), m.end(), bonus_spans):
            continue
        lo, hi = _parse_money_range(m.group(0).rsplit("USD", 1)[0])
        if lo is None:
            continue
        unit_word = m.group("unit").lower()
        unit = "hourly" if ("hour" in unit_word or "hr" in unit_word) else "annual"
        candidates.append(
            {
                "subtype": unit,
                "min": lo,
                "max": hi if hi is not None else lo,
                "unit": unit,
                "is_range": hi is not None and hi != lo,
                "has_context": True,
                "start": m.start(),
            }
        )

    if not candidates:
        return None

    best = max(candidates, key=lambda c: (c["is_range"], c["has_context"], -c["start"]))
    return {"subtype": best["subtype"], "min": best["min"], "max": best["max"], "unit": best["unit"]}


# ── Estimation (same-employer comparable lookup) ────────────────────────

MIN_COMPARABLES_FOR_ESTIMATE = 2
MIN_COMPARABLES_FOR_HIGH_CONFIDENCE = 5
HIGH_CONFIDENCE_SPREAD_RATIO = 1.5

# 2026-08-29 real-data fix: a live production run produced a same-employer
# "estimate" of $3,000-$90,571/year (30x spread) for a Raytheon Avionics
# Technician posting -- min-of-mins/max-of-maxes across N comparables had
# no ceiling on how incoherent the resulting range could be, only a floor
# on how many comparables existed. A range this wide isn't a conservative
# estimate, it's noise wearing a confidence label. Above this ratio the
# estimator abstains entirely (status stays "unknown") regardless of N --
# deliberately looser than HIGH_CONFIDENCE_SPREAD_RATIO (1.5), which only
# decides high- vs medium-confidence for ranges that already passed this
# gate; a real, wide-but-genuine range (e.g. a company's Software Engineer
# postings legitimately spanning ~2x across very different roles) should
# still surface as a medium-confidence estimate, just never one this wide.
MAX_SPREAD_RATIO_FOR_ANY_ESTIMATE = 3.0

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "senior", "sr", "junior", "jr", "staff",
        "lead", "principal", "director", "manager", "chief", "head", "vp",
        "team", "member", "specialist", "associate", "assistant",
    }
)


def _title_tokens(title: str) -> set[str]:
    """Significant (length >= 4, non-stopword) lowercase word tokens."""
    tokens = re.findall(r"[a-z]+", (title or "").lower())
    return {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS}


def _is_usable_employer_identifier(site: str | None) -> bool:
    return bool(site) and site.strip().lower() not in _AGGREGATOR_SITES


def estimate_compensation(job: dict, conn) -> dict | None:
    """Conservative same-employer, overlapping-title comparable estimate.

    Requires the job's own `site` to look like a real employer (not a
    known aggregator platform), a non-empty title, and at least
    MIN_COMPARABLES_FOR_ESTIMATE OTHER postings from that same `site` that
    (a) share at least one significant title token and (b) themselves have
    unambiguous stated compensation in the SAME unit (annual or hourly --
    never mixed/converted, to avoid inventing a conversion the posting
    never made). Returns None (never guesses) if that bar isn't met --
    this is intentionally the common case given how little of the live
    database has any employer identity or stated pay at all.
    """
    site = job.get("site")
    title = job.get("title") or ""
    if not _is_usable_employer_identifier(site):
        return None
    my_tokens = _title_tokens(title)
    if not my_tokens:
        return None

    self_url = job.get("url") or ""
    rows = conn.execute(
        "SELECT title, salary, full_description FROM jobs "
        "WHERE site = ? AND url != ? AND full_description IS NOT NULL",
        (site, self_url),
    ).fetchall()

    by_unit: dict[str, list[dict]] = {"annual": [], "hourly": []}
    matched_tokens: set[str] = set()
    for row in rows:
        r_title = row["title"] or ""
        shared = _title_tokens(r_title) & my_tokens
        if not shared:
            continue
        text = f"{row['salary'] or ''}\n{row['full_description'] or ''}"
        stated = classify_stated_compensation(text)
        if stated and stated["unit"] in ("annual", "hourly"):
            by_unit[stated["unit"]].append(stated)
            matched_tokens |= shared

    unit = "annual" if len(by_unit["annual"]) >= len(by_unit["hourly"]) else "hourly"
    comparables = by_unit[unit]
    if len(comparables) < MIN_COMPARABLES_FOR_ESTIMATE:
        return None

    lo = min(c["min"] for c in comparables)
    hi = max(c["max"] for c in comparables)
    n = len(comparables)
    spread_ratio = (hi / lo) if lo else float("inf")

    if spread_ratio > MAX_SPREAD_RATIO_FOR_ANY_ESTIMATE:
        # The comparable set disagrees too much with itself to produce a
        # defensible range at any confidence level -- abstain rather than
        # publish noise with a confidence label attached.
        return None

    if n >= MIN_COMPARABLES_FOR_HIGH_CONFIDENCE and spread_ratio <= HIGH_CONFIDENCE_SPREAD_RATIO:
        confidence = "high"
    else:
        confidence = "medium"

    basis = (
        f"{n} comparable stated-pay posting(s) at '{site}' sharing title term(s) "
        f"{', '.join(sorted(matched_tokens))}"
    )

    return {
        "estimated_min": lo,
        "estimated_max": hi,
        "estimated_unit": unit,
        "estimated_confidence": confidence,
        "estimate_basis": basis,
    }


# ── Top-level classifier ─────────────────────────────────────────────────


def classify_compensation(job: dict, conn=None) -> dict:
    """Classify a job's compensation and, if unstated, attempt a
    conservative estimate.

    Imports scorer._COMMISSION_ONLY_PATTERN lazily (call time, not module
    load time) so this module never creates a top-level circular import
    with scoring.scorer -- scorer.py is the primary caller and already has
    that pattern defined in the same module by the time score_job() runs;
    view.py (dashboard) is a secondary caller with no other dependency on
    scoring.scorer at all. Always the SAME compiled pattern
    scorer._check_ineligible already uses for its own commission-only
    rejection, so "explicitly_absent" here is always in exact agreement
    with what the deterministic gate rejects -- one source of truth, never
    two regexes that could drift apart.

    ``conn``: optional DB connection for the same-employer comparable
    estimator. When None, estimation is skipped entirely (cheap callers
    like the dashboard, which classifies many rows per page render, pass
    conn=None deliberately to avoid a DB query per row) and unstated
    compensation simply classifies as "unknown".

    Returns a dict with at least {"status": ...}, plus:
    - status "stated": subtype, stated_min, stated_max, stated_unit
    - status "estimated": estimated_min/max/unit/confidence, estimate_basis
    - status "unknown" / "explicitly_absent": no extra keys
    """
    from applypilot.scoring.scorer import _COMMISSION_ONLY_PATTERN

    full_description = job.get("full_description") or ""
    salary_field = job.get("salary") or ""
    combined_text = f"{salary_field}\n{full_description}"

    stated = classify_stated_compensation(combined_text)
    if stated is not None:
        return {
            "status": "stated",
            "subtype": stated["subtype"],
            "stated_min": stated["min"],
            "stated_max": stated["max"],
            "stated_unit": stated["unit"],
        }

    if _COMMISSION_ONLY_PATTERN.search(full_description):
        return {"status": "explicitly_absent"}

    if conn is not None:
        estimate = estimate_compensation(job, conn)
        if estimate is not None:
            return {"status": "estimated", **estimate}

    return {"status": "unknown"}


# ── Scoring integration ──────────────────────────────────────────────────
#
# Deliberately small and flat -- three numeric outcomes, never invented as
# a bigger scale. "explicitly_absent" is never passed here: that status is
# already a hard pre-filter rejection in scorer._check_ineligible/
# score_job, which returns before any LLM score (and before this function)
# is ever produced.
#
# Ordering rationale (documented since the source brief's own suggested
# tiers -- "unknown -> modest penalty" vs "very low-confidence/no-estimate
# -> somewhat larger penalty" -- describe what, in this module's actual
# architecture, is the SAME state: the estimator either produces a
# confidence-tagged estimate or abstains to "unknown" outright; there is no
# third "tried and got a low-confidence result" state to distinguish from
# plain "unknown". Resolved by taking the "somewhat larger" magnitude for
# "unknown" specifically because it is a strictly higher-uncertainty state
# than a medium-confidence estimate backed by real same-employer data.
COMPENSATION_PENALTY_ESTIMATED_HIGH = 0
COMPENSATION_PENALTY_ESTIMATED_MEDIUM = -1
COMPENSATION_PENALTY_UNKNOWN = -2


def compensation_score_adjustment(comp: dict) -> tuple[int, str]:
    """Deterministic, small, explainable score delta for compensation
    uncertainty. Never a claim that the job pays poorly -- only that
    ApplyPilot could not confirm what it pays.
    """
    status = comp.get("status")

    if status == "stated":
        return 0, ""

    if status == "estimated":
        rng = f"${comp['estimated_min']:,.0f}–${comp['estimated_max']:,.0f}/{comp['estimated_unit']}"
        if comp.get("estimated_confidence") == "high":
            return COMPENSATION_PENALTY_ESTIMATED_HIGH, (
                f"Estimated base: {rng} (high confidence, based on {comp['estimate_basis']}); "
                "not employer-stated."
            )
        return COMPENSATION_PENALTY_ESTIMATED_MEDIUM, (
            f"Estimated base: {rng} (medium confidence, based on {comp['estimate_basis']}); "
            "not employer-stated, so a small uncertainty adjustment was applied."
        )

    if status == "unknown":
        return COMPENSATION_PENALTY_UNKNOWN, (
            "Compensation is not stated in this posting, so there is some uncertainty about "
            "job quality; a small adjustment was applied for that uncertainty, not a claim "
            "that the job pays poorly."
        )

    # "explicitly_absent" (or any future status) intentionally has no
    # adjustment here -- the deterministic gate already handles rejection
    # before this function can ever be reached for that case.
    return 0, ""
