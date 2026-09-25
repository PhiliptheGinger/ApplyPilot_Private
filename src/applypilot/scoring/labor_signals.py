"""Deterministic positive labor/ownership-structure signals for job postings
(CLAUDE.md decision #206 addendum, built 2026-09-25, per explicit user request).

Complements the existing `exclude_description_keywords` ethical-exclusion
list (searches.example.yaml): that mechanism only ever subtracts for
identified ill-will (defense/surveillance/policing exclusions). Nothing
previously added points for what a candidate might actually be looking
FOR. User's own framing, preserved: employee ownership and unionization
deserve credit even though "there's not a lot of ethical consumption
under our underregulated market economy" -- most jobs won't clear a
maximalist ethical bar, so this is a small, additive bonus for concrete,
verifiable positive structures, not a purity test.

Verified against the real 14,420-job corpus before shipping (same
discipline as every other pattern in this codebase): every candidate
pattern below was sample-checked against real matched snippets, confirming
genuine hits (real Employee Stock Ownership Plan benefit lines, real
Collective Bargaining Agreement clauses, real "public benefit corporation"
self-descriptions including a real Anthropic posting, a real "BCorp
Certified" line). One real false positive was found and fixed before
shipping: a naive "union members/employees/workforce" branch substring-
matched "union membership" inside ordinary EEO/non-discrimination
boilerplate ("political affiliation, union membership, or any other
characteristic protected by law") -- the exact same false-positive SHAPE
already documented for bare "military" in the ethical-exclusion list
(searches.example.yaml). Fixed by dropping that branch entirely and
relying only on unambiguous unionization phrasing (collective bargaining,
union shop, unionized, "represented by a union").
"""

import re

LABOR_SIGNAL_CATEGORIES = frozenset(
    {
        "employee_ownership",
        "worker_cooperative",
        "unionized",
        "b_corp_certified",
        "benefit_corporation",
        "profit_sharing",
    }
)

_EMPLOYEE_OWNED_RE = re.compile(r"employee[\s-]owned|employee\s+ownership|\bESOP\b", re.I)
_WORKER_COOPERATIVE_RE = re.compile(r"worker[\s-]owned\s+cooperative|worker\s+cooperative", re.I)
_UNIONIZED_RE = re.compile(
    r"collective\s+bargaining|union\s+shop|unioniz(?:ed|ation)|represented\s+by\s+(?:a\s+|the\s+)?union\b",
    re.I,
)
_B_CORP_RE = re.compile(r"certified\s+b\s*corp(?:oration)?|\bb\s*corp\s+certified\b", re.I)
_BENEFIT_CORP_RE = re.compile(r"public\s+benefit\s+corporation|\bbenefit\s+corporation\b", re.I)
_PROFIT_SHARING_RE = re.compile(r"profit[\s-]sharing", re.I)

_PATTERNS = {
    "employee_ownership": _EMPLOYEE_OWNED_RE,
    "worker_cooperative": _WORKER_COOPERATIVE_RE,
    "unionized": _UNIONIZED_RE,
    "b_corp_certified": _B_CORP_RE,
    "benefit_corporation": _BENEFIT_CORP_RE,
    "profit_sharing": _PROFIT_SHARING_RE,
}

# Small, symmetric with the compensation module's own penalty magnitude
# (COMPENSATION_PENALTY_UNKNOWN = -2) -- capped so several categories
# firing on one posting doesn't inflate the score more than a genuinely
# large compensation uncertainty penalty could move it down.
LABOR_SIGNAL_BONUS_PER_CATEGORY = 1
LABOR_SIGNAL_BONUS_CAP = 2


def detect_labor_signals(job: dict) -> list[str]:
    """Return the list of matched positive labor/ownership-signal category names,
    empty if none. Checked against title + full description, matching the
    ethical-exclusion keyword check's own scope.
    """
    text = f"{job.get('title', '')}\n{job.get('full_description') or ''}"
    return [name for name, pat in _PATTERNS.items() if pat.search(text)]


def labor_signal_score_adjustment(signals: list[str]) -> tuple[int, str]:
    """Deterministic, small, explainable score bonus for positive labor/ownership
    structures. Never a claim the job is ethically ideal overall -- only that a
    specific, verifiable positive structure is present in the posting's own text.
    """
    if not signals:
        return 0, ""
    delta = min(LABOR_SIGNAL_BONUS_CAP, len(signals) * LABOR_SIGNAL_BONUS_PER_CATEGORY)
    readable = ", ".join(s.replace("_", " ") for s in signals)
    return delta, f"Positive labor/ownership signal(s) found: {readable}."
