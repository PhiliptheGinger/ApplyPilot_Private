"""Requirement-framing classifier (FW28, decision #236).

``deterministic_fallback.extract_years_required`` decides whether a years
mention is a hard requirement from a short list of trigger words and known
section headers. Every new phrasing ("Seeking 10+ years", "What you'll
need", "commercial experience") has needed its own regex patch, and several
patches caused false positives (#87, #90, #92).

This module classifies the SENTENCE around a years mention from several weak
signals instead of one keyword:

- modal strength in the sentence: required / must / need / at least /
  seeking (hard) vs. preferred / nice to have / a plus / bonus (soft);
- the nearest section header above it (or an inline ``Label:`` prefix):
  requirements-like, preferred-like, company/benefits-like, or duties;
- who the sentence is about: the applicant ("you", "candidate", "someone")
  vs. the company ("we have", "our team", "founded");
- whether the line is a bullet.

A mention is ``hard_requirement`` only when at least two signals agree, one
of them a modal or a requirements header, and nothing points the other way.
Callers use it ADDITIVELY: it can fill in a requirement the existing
extractor missed, never remove or lower one the extractor found.

``implicit_seniority`` is the second half of FW28 (FW15 a/b): postings with
no stated years whose own wording asks for authority-level work ("set the
technical direction", "founding engineer", "architect our platform").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

HARD = "hard_requirement"
PREFERRED = "preferred"
DESCRIPTIVE = "descriptive"
UNRELATED = "unrelated"

# Same pattern as deterministic_fallback._YEARS_MENTION_RE (kept in step by
# test_requirement_framing): group 1 = range start, 2 = single number, 3 = word.
_YEARS_MENTION_RE = re.compile(
    r"\b(\d{1,2}(?:\.\d+)?)\s*[-–—]\s*\d{1,2}(?:\.\d+)?(?:\+|\s+or\s+more)?[\s-]*(?:years?|yrs?)\b"
    r"|\b(?:[a-z]+\s*\()?(\d{1,2}(?:\.\d+)?)\)?(?:\+|\s+or\s+more)?[\s-]*(?:years?|yrs?)\b"
    r"|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:years?|yrs?)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

# A years mention that is not about work experience at all.
_NOT_EXPERIENCE_RE = re.compile(
    r"\b(?:renew\w*|valid\s+for|expir\w*|warrant\w*|tenure|anniversary|\bpto\b|paid\s+time\s+off|"
    r"of\s+age|or\s+older|years?\s+old|full.?time\s+education|years?\s+ago|years?\s+later|"
    r"in\s+business|year.round|per\s+year|a\s+year\b|each\s+year|every\s+\w*\s*years?)\b",
    re.IGNORECASE,
)
# "4-year degree", "2-year contract": the number describes the thing after it.
_FOLLOWING_NOUN_RE = re.compile(
    r"^\W{0,3}(?:\(\w+\)\s*)?(?:degree|college|university|program|programme|contract|term|plan|"
    r"commitment|vesting|cliff|warranty|history)\b",
    re.IGNORECASE,
)

# "less than 2 years of experience" is a ceiling (an entry-level ask), not a minimum.
_UPPER_BOUND_RE = re.compile(
    r"\b(?:less\s+than|fewer\s+than|under|up\s+to|no\s+more\s+than|maximum\s+of|max\.?|at\s+most)\s*$",
    re.IGNORECASE,
)
_MODAL_HARD_RE = re.compile(
    r"\b(?:required|requires?|requirement|must|mandatory|essential|minimum|at\s+least|"
    r"need(?:s|ed)?|seeking|looking\s+for|demonstrated|proven)\b",
    re.IGNORECASE,
)
_MODAL_SOFT_RE = re.compile(
    r"\b(?:prefer(?:red|ably|able)?|nice.to.have|bonus|(?:is|are|a|big)\s+plus|ideal(?:ly)?|"
    r"desired|desirable|advantage(?:ous)?|helpful|optional|would\s+be\s+great|not\s+required|"
    r"good\s+to\s+have|beneficial)\b",
    re.IGNORECASE,
)
_APPLICANT_RE = re.compile(
    r"\b(?:you|your|you['’]ll|you['’]ve|you['’]re|candidates?|applicants?|someone|individual|"
    r"incumbent|the\s+successful|the\s+right\s+person|this\s+(?:role|position))\b",
    re.IGNORECASE,
)
_COMPANY_RE = re.compile(
    r"\b(?:we|we['’]ve|we['’]re|our|us|founded|established|since\s+\d{4}|company|legacy)\b",
    re.IGNORECASE,
)

_HEADER_SOFT_RE = re.compile(
    r"prefer|nice.to.have|bonus|\bplus\b|desired|desirable|optional|extra\s+credit|good\s+to\s+have|ideally",
    re.IGNORECASE,
)
_HEADER_COMPANY_RE = re.compile(
    r"^about\s+(?!you\b|the\s+role\b|this\s+role\b|the\s+job\b|the\s+position\b)|who\s+we\s+are|"
    r"\bour\s+(?:story|mission|company|culture|values|history)|benefits|perks|what\s+we\s+offer|"
    r"compensation|\bpay\b|salary|why\s+(?:join|work|aws|us)|life\s+at|equal\s+opportunity|\beeo\b|"
    r"diverse\s+experiences|inclusi|work.life|career\s+growth|mentorship",
    re.IGNORECASE,
)
_HEADER_HARD_RE = re.compile(
    r"requirement|required|qualification|\bmust\b|minimum|basic|what\s+you.?ll\s+need|what\s+you\s+need|"
    r"you\s+have|about\s+you|who\s+you\s+are|what\s+you.?ll\s+bring|what\s+you\s+bring|looking\s+for|"
    r"skills\s*(?:and|&)\s*experience|^(?:relevant\s+|required\s+|work\s+|professional\s+|the\s+)?experience\b(?!\s+with)|"
    r"education\s*(?:and|&|/)\s*(?:work\s+)?experience|experience\s+(?:needed|required|perspective)|you\s+should\s+have|ideal\s+candidate|need\s+to\s+have|"
    r"what\s+we\s+(?:look|require)|competenc",
    re.IGNORECASE,
)
_HEADER_DUTIES_RE = re.compile(
    r"responsibilit|what\s+you.?ll\s+do|the\s+role|duties|day.to.day|your\s+impact|job\s+description|"
    r"overview|summary|what\s+you\s+will\s+do",
    re.IGNORECASE,
)

_ITEM_START_RE = re.compile(
    r"(?:experience\s+(?!and\b|needed\b|required\b)\w+|must\s+(?:be|have)|bachelor|master|associate|"
    r"b\.?s\.?\b|b\.?a\.?\b|m\.?s\.?\b|degree|u\.?s\.?\s|ability|able\s+to|knowledge|proficien|"
    r"strong|excellent|familiar|understanding|working\s+knowledge|\d)",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•·▪◦‣–]|\d{1,2}[.)])\s+")
_INLINE_LABEL_RE = re.compile(r"^\s*\**([A-Za-z][^:\n]{1,40}?)\**\s*:\s+\S")
_SENTENCE_BREAK_RE = re.compile(r"[.!?;](?=\s|$)|\n")


@dataclass
class Mention:
    years: int
    frame: str
    signals: list[str] = field(default_factory=list)
    sentence: str = ""


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, end if end != -1 else len(text)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for m in _SENTENCE_BREAK_RE.finditer(text, 0, start):
        left = m.end()
    m = _SENTENCE_BREAK_RE.search(text, end)
    return left, m.start() if m else len(text)


def _sentence(text: str, start: int, end: int) -> str:
    left, right = _sentence_bounds(text, start, end)
    return text[left:right].strip()


def _soft_wins(sentence: str, mention_start: int, mention_end: int) -> bool:
    """True when the sentence's softening word governs the years mention.

    "3 years preferred" -> soft. "Minimum 7 years of design experience,
    preferably in consumer electronics" -> the hard word sits closer to the
    number, so the preference is about the domain, not the years.
    """
    soft = [(m.start(), m.end()) for m in _MODAL_SOFT_RE.finditer(sentence)]
    if not soft:
        return False
    hard = [
        (m.start(), m.end())
        for m in _MODAL_HARD_RE.finditer(sentence)
        if not any(s <= m.start() < e for s, e in soft)  # "not required" is soft
    ]
    if not hard:
        return True

    def dist(span: tuple[int, int]) -> int:
        return max(span[0] - mention_end, mention_start - span[1], 0)

    return min(map(dist, soft)) < min(map(dist, hard))


def _is_header_line(line: str) -> bool:
    s = line.strip().strip("*#").strip()
    if not s or len(s) > 60 or len(s.split()) > 8 or _BULLET_RE.match(line):
        return False
    if s.endswith(".") or _YEARS_MENTION_RE.search(s):
        return False
    if s.endswith(":") or line.strip().startswith(("#", "**")):
        return True
    # A bare short line counts only if it reads like a known kind of header,
    # not like a requirement item ("Experience with Altium", "Must be a US
    # Citizen", "Bachelor's degree or equivalent experience").
    return _header_category(s) != "other" and not _ITEM_START_RE.match(s)


def _header_category(header: str) -> str:
    h = header.strip().strip("*#:").strip()
    if _HEADER_SOFT_RE.search(h):
        return "soft"
    if _HEADER_COMPANY_RE.search(h):
        return "company"
    if _HEADER_HARD_RE.search(h):
        return "hard"
    if _HEADER_DUTIES_RE.search(h):
        return "duties"
    return "other"


def _nearest_header(text: str, line_start: int, max_lines: int = 40) -> str | None:
    """The closest header-looking line above ``line_start``."""
    lines = text[:line_start].split("\n")
    for line in reversed(lines[-max_lines:]):
        if _is_header_line(line):
            return line
    return None


def _mention_years(m: re.Match) -> int | None:
    numeric = m.group(1) or m.group(2)
    if numeric is not None:
        return math.ceil(float(numeric))
    return _NUMBER_WORDS.get((m.group(3) or "").lower())


def classify_mention(text: str, m: re.Match) -> Mention | None:
    """Classify one years mention. None if it isn't a usable number."""
    years = _mention_years(m)
    if years is None:
        return None
    ls, le = _line_bounds(text, m.start())
    line = text[ls:le]
    ss, se = _sentence_bounds(text, m.start(), m.end())
    sentence = text[ss:se]
    if _NOT_EXPERIENCE_RE.search(sentence) or _FOLLOWING_NOUN_RE.match(text[m.end() : m.end() + 40]):
        return Mention(years, UNRELATED, ["not_experience"], sentence)
    if _UPPER_BOUND_RE.search(text[max(0, m.start() - 20) : m.start()]):
        return Mention(years, UNRELATED, ["upper_bound"], sentence)

    label = _INLINE_LABEL_RE.match(line)
    if label and label.end(1) <= m.start() - ls:
        header_cat = _header_category(label.group(1))
    else:
        header = _nearest_header(text, ls)
        header_cat = _header_category(header) if header else None

    signals: list[str] = []
    if header_cat:
        signals.append(f"header:{header_cat}")
    if header_cat == "soft" or _soft_wins(sentence, m.start() - ss, m.end() - ss):
        return Mention(years, PREFERRED, signals + ["soft"], sentence.strip())
    sentence = sentence.strip()
    applicant = bool(_APPLICANT_RE.search(sentence))
    if header_cat == "company" or (_COMPANY_RE.search(sentence) and not applicant):
        return Mention(years, DESCRIPTIVE, signals + ["company"], sentence)

    hard: list[str] = []
    if _MODAL_HARD_RE.search(sentence):
        hard.append("modal")
    if header_cat == "hard":
        hard.append("header")
    if applicant:
        hard.append("applicant")
    # A list item: a bullet, or (since scraped text often drops the bullet
    # markers) a short line of its own that starts with the requirement.
    standalone_item = header_cat == "hard" and len(line.strip()) < 300 and m.start() - ls < 80
    if (_BULLET_RE.match(line) and header_cat in ("hard", None)) or standalone_item:
        hard.append("list_item")
    signals += hard
    if len(hard) >= 2 and ("modal" in hard or "header" in hard):
        return Mention(years, HARD, signals, sentence)
    return Mention(years, DESCRIPTIVE, signals, sentence)


def classify_years_mentions(description: str) -> list[Mention]:
    text = description or ""
    out = []
    for m in _YEARS_MENTION_RE.finditer(text):
        mention = classify_mention(text, m)
        if mention is not None:
            out.append(mention)
    return out


def framed_years_required(description: str) -> int | None:
    """Smallest years figure among mentions classified as hard requirements."""
    hard = [m.years for m in classify_years_mentions(description) if m.frame == HARD]
    return min(hard) if hard else None


# ---------------------------------------------------------------------------
# Implicit seniority (FW15 a/b)
# ---------------------------------------------------------------------------

# Posting-voice phrases that ask for authority-level work. Each pattern counts
# once per posting; two distinct ones are needed (one is too easy to hit in a
# mid-level posting's aspirational paragraph).
_SENIORITY_PATTERNS: dict[str, re.Pattern] = {
    "founding": re.compile(r"\bfounding\s+(?:engineer|developer|member|team)", re.IGNORECASE),
    "technical_direction": re.compile(
        r"\b(?:set|sets|setting|define|defines|defining|drive|drives|driving|own|owns|owning)\s+"
        r"(?:the\s+|our\s+)?(?:technical|engineering|architectural)\s+(?:direction|vision|strategy|roadmap)",
        re.IGNORECASE,
    ),
    "architect": re.compile(
        r"\b(?:architect|design\s+and\s+own|own\s+the\s+architecture|lead\s+the\s+(?:design|architecture))\b"
        r"(?!\s*(?:role|position|title))",
        re.IGNORECASE,
    ),
    "mentor_engineers": re.compile(
        r"\b(?:mentor|mentoring|coach|coaching)\s+(?:other\s+|junior\s+|more\s+junior\s+)?(?:engineers|developers)",
        re.IGNORECASE,
    ),
    "lead_team": re.compile(
        r"\b(?:lead|leading|manage|managing|build\s+and\s+lead|grow\s+and\s+lead)\s+(?:a|the|our)\s+"
        r"(?:\w+\s+)?(?:team|engineers|organization)",
        re.IGNORECASE,
    ),
    "deep_expertise": re.compile(
        r"\b(?:deep|extensive|expert.level|world.class)\s+(?:expertise|experience|knowledge)\b",
        re.IGNORECASE,
    ),
}


def implicit_seniority(description: str) -> list[str]:
    """Names of distinct authority-level asks found outside company/benefit
    sections and preferred-only sentences. Two or more means senior."""
    text = description or ""
    found: list[str] = []
    for name, pattern in _SENIORITY_PATTERNS.items():
        for m in pattern.finditer(text):
            ls, _le = _line_bounds(text, m.start())
            sentence = _sentence(text, m.start(), m.end())
            header = _nearest_header(text, ls)
            if header and _header_category(header) in ("company", "soft"):
                continue
            if _MODAL_SOFT_RE.search(sentence):
                continue
            found.append(name)
            break
    return found


def is_implicitly_senior(description: str) -> bool:
    return len(implicit_seniority(description)) >= 2
