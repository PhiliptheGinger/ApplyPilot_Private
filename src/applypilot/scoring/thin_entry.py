"""Follow-up questions for profile entries too thin to build a phrase bank (FW26).

2026-10-07. When `applypilot expand-bank` gets zero surviving phrasings for an
entry (decision #143's Freelance Photography case: one modest sentence plus an
anti-embellishment constraint), the fix is more real facts, never generated
ones. This asks the candidate a few factual questions, shows exactly what
would be added, and only writes to profile.json after they confirm.

Facts are appended to the same list the phrase bank reads from
(phrase_bank.source_facts): `responsibilities` when the entry has them,
otherwise `factual_concepts` for project entries.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

FOLLOWUP_QUESTIONS = (
    "Describe 2-3 specific tasks or responsibilities you handled.",
    "Which tools, software, or equipment did you use?",
    "Who was it for, and roughly how often or over what period?",
    "Any outcome you're comfortable stating as fact (numbers, frequency, duration)?",
)


def _as_fact(answer: str) -> str | None:
    text = " ".join((answer or "").split())
    if not text:
        return None
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def collect_followup_facts(item_name: str, ask: Callable[[str], str]) -> list[str]:
    """Ask each follow-up question; return the non-empty answers as fact sentences."""
    facts: list[str] = []
    for question in FOLLOWUP_QUESTIONS:
        fact = _as_fact(ask(f"[{item_name}] {question} (Enter to skip)"))
        if fact:
            facts.append(fact)
    return facts


def _fact_list_key(item: dict, section: str) -> str:
    if [r for r in (item.get("responsibilities") or []) if isinstance(r, str) and r.strip()]:
        return "responsibilities"
    return "factual_concepts" if section == "project_inventory" else "responsibilities"


def add_facts_to_profile(profile_path: Path, item_name: str, facts: list[str]) -> dict:
    """Append `facts` to the named entry in profile.json and return the updated entry.

    Raises KeyError if no experience/project entry has that name.
    """
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    for section in ("experience_inventory", "project_inventory"):
        for item in profile.get(section) or []:
            if isinstance(item, dict) and item.get("name") == item_name:
                key = _fact_list_key(item, section)
                existing = [f for f in (item.get(key) or []) if isinstance(f, str)]
                item[key] = existing + [f for f in facts if f not in existing]
                Path(profile_path).write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
                return item
    raise KeyError(item_name)
