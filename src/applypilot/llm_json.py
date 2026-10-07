"""Shared parser for JSON returned by an LLM (cloud or local).

2026-10-07 consolidation (pattern audit, docs/audit_2026-10.md). Five call
sites each parsed model output their own way, with different coverage:
tailor.extract_json handled fences and an outer {...}; local_tailor's
_parse_plan also stripped <think> blocks; smartextract handled prose
around JSON; hackernews handled fences only; and phrase-bank generation
used a bare json.loads, so any local-model reply wrapped in a fence,
prose, or a <think> block silently lost that whole generation round.

parse_llm_json is the superset. It only ever returns a value that parsed
as a complete JSON document starting at the first top-level { or [ it
finds -- it never digs into nested fragments, so a malformed outer object
still raises rather than returning some inner piece of it.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


def _strip_think(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    # An unterminated block followed by a closing tag only, e.g. "...</think>{...}"
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


def _decode_from_first_bracket(text: str) -> Any:
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    for start in sorted(starts):
        try:
            value, _end = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("no JSON object or array found")


def parse_llm_json(text: str) -> Any:
    """Parse the JSON payload of a model reply.

    Tries, in order: the whole reply; each fenced code block; the first
    complete JSON object/array in the reply (tolerates prose before and
    after). <think>...</think> reasoning blocks are removed first.

    Raises:
        ValueError: if no complete JSON document can be found.
    """
    if text is None:
        raise ValueError("No valid JSON found in LLM response: empty reply")
    cleaned = _strip_think(str(text))
    candidates = [cleaned]
    candidates.extend(block.strip() for block in _FENCE_RE.findall(cleaned))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        try:
            return _decode_from_first_bracket(candidate)
        except ValueError:
            continue
    raise ValueError("No valid JSON found in LLM response")
