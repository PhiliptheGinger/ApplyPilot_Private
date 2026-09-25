"""Deterministic scam-likelihood signals for application-response emails
(CLAUDE.md Future Work item 34, built 2026-09-25).

Grounded in a real, already-caught incident (decision #174): a fake
"You're Invited to Interview with Forestar Group" email, sent from a
personal Gmail address impersonating a real company, bundling several
unrelated generic job titles with "strictly work-from-home and requires
training" framing.

Deliberately NEVER auto-rejects or hides anything -- this only ever
annotates a match with a warning (`detect_scam_signals` returns a list of
matched signal names, empty if none). A real, live check against this
pipeline's own JOB POSTING corpus (2026-09-25) found several intuitive
single-keyword scam phrases are dangerous false-positive traps on
legitimate postings (e.g. "bank details" hit 602 real, legitimate
financial-services job descriptions) -- there's no equivalent corpus of
real application-response emails available to calibrate against here, so
this module requires at least TWO independent signal categories to
co-occur before returning anything, rather than trusting any single
keyword alone.
"""

import re

# Common personal webmail providers -- a real company's OFFICIAL hiring
# communication essentially never comes from one of these (small
# legitimate startups using Gmail for business do exist, which is why
# this alone is never sufficient -- see _looks_like_corporate_claim below).
_PERSONAL_EMAIL_DOMAINS = frozenset(
    {
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "aol.com",
        "icloud.com",
        "protonmail.com",
        "mail.com",
    }
)

# Phrasing that implies the sender is speaking FOR a company/hiring
# process, not just a personal note -- the co-occurrence with a personal
# email domain is the actual signal, not either alone.
_CORPORATE_CLAIM_RE = re.compile(
    r"\b(?:invited?\s+to\s+interview|we(?:'re| are)\s+hiring|"
    r"on\s+behalf\s+of\s+[A-Z]|job\s+offer|congratulations.{0,20}(?:hired|selected)|"
    r"welcome\s+to\s+the\s+team)\b",
    re.I,
)

# Several distinct, unrelated generic job titles bundled in one message --
# the real Forestar Group case listed "Remote Data Entry Clerk, Customer
# Service Representative, and Administrative roles" together.
_GENERIC_BAIT_TITLES = [
    "data entry",
    "customer service representative",
    "administrative",
    "virtual assistant",
    "remote assistant",
    "mystery shopper",
    "package handler",
    "form filler",
]

_UPFRONT_PAYMENT_RE = re.compile(
    r"\b(?:purchase\s+(?:your\s+own\s+)?equipment|processing\s+fee|"
    r"registration\s+fee|starter\s+kit\s+fee|pay\s+for\s+(?:your\s+own\s+)?training|"
    r"wire\s+(?:us\s+|transfer\s+)?(?:a\s+)?(?:deposit|fee))\b",
    re.I,
)

_URGENCY_RE = re.compile(
    r"\b(?:respond\s+within\s+24\s+hours|act\s+immediately|limited\s+spots|"
    r"apply\s+within\s+24\s+hours|urgent(?:ly)?\s+(?:hiring|need)|"
    r"offer\s+expires\s+(?:today|soon))\b",
    re.I,
)


def _sender_domain(sender: str) -> str:
    match = re.search(r"@([\w.-]+)", sender or "")
    return match.group(1).lower() if match else ""


def detect_scam_signals(email: dict) -> list[str]:
    """Return the list of matched scam-signal category names, empty if none.

    Requires at least 2 independent categories to co-occur before
    returning anything -- see module docstring for why a single keyword
    alone isn't trusted here.
    """
    sender = email.get("sender") or ""
    body = f"{email.get('subject', '')}\n{email.get('body') or email.get('snippet') or ''}"

    signals: list[str] = []

    domain = _sender_domain(sender)
    if domain in _PERSONAL_EMAIL_DOMAINS and _CORPORATE_CLAIM_RE.search(body):
        signals.append("personal_email_claiming_corporate")

    bait_hits = sum(1 for title in _GENERIC_BAIT_TITLES if title in body.lower())
    if bait_hits >= 2:
        signals.append("bundled_generic_titles")

    if _UPFRONT_PAYMENT_RE.search(body):
        signals.append("upfront_payment_language")

    if _URGENCY_RE.search(body):
        signals.append("urgency_pressure")

    return signals if len(signals) >= 2 else []
