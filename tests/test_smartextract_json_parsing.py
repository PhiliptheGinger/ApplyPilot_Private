"""Regression tests for CLAUDE.md Future Work item 63 (closed 2026-09-26,
per decision #203's finding): `discovery/smartextract.py::extract_json`
used to give up entirely once a response was neither a clean JSON blob nor
a code-fenced one -- but the local model, confirmed live during a full-day
cloud outage, sometimes wraps a real JSON object in conversational prose
("It seems like the list of job postings is truncated... {...}. Let me
know if you need more.") instead of returning ONLY the JSON. 16 real
PARSE_ERROR occurrences in one night, all this same shape.

Fix: a last-resort regex pull of the outermost brace/bracket-delimited
span anywhere in the text, tried only after every existing (clean-JSON,
code-fence, trailing-truncation) path has already failed.
"""

from __future__ import annotations

import json

import pytest

from applypilot.discovery.smartextract import extract_json


def test_clean_json_still_parses_directly_no_regression():
    raw = '{"card_selector": ".job-card", "title_selector": ".title"}'
    assert extract_json(raw) == {"card_selector": ".job-card", "title_selector": ".title"}


def test_code_fenced_json_still_parses_no_regression():
    raw = '```json\n{"card_selector": ".job-card"}\n```'
    assert extract_json(raw) == {"card_selector": ".job-card"}


def test_json_wrapped_in_leading_prose_is_rescued():
    """The exact real-incident shape: the model narrates instead of just
    returning JSON, but the real object is still present somewhere in the
    response."""
    raw = (
        "It seems like the list of job postings is truncated. Based on what "
        'I can see, here are the selectors: {"card_selector": ".posting", '
        '"title_selector": ".posting-title"}'
    )
    assert extract_json(raw) == {"card_selector": ".posting", "title_selector": ".posting-title"}


def test_json_wrapped_in_leading_and_trailing_prose_is_rescued():
    raw = (
        'The provided content appears to be a job board. {"card_selector": ".card"} '
        "Let me know if you need anything else!"
    )
    assert extract_json(raw) == {"card_selector": ".card"}


def test_pure_prose_with_no_json_at_all_still_raises_cleanly():
    """Not every rescue attempt succeeds -- a response with genuinely no
    JSON anywhere must still fail cleanly (matching decision #203's own
    "confirmed safe: 0 jobs, no crash" finding), not hang or return junk."""
    raw = "It seems like the list of job postings is truncated. I cannot generate selectors from this content."
    with pytest.raises(json.JSONDecodeError):
        extract_json(raw)


def test_malformed_bracket_content_still_raises_cleanly():
    """A stray, non-JSON bracket pair in real prose (e.g. a citation-style
    reference) must not be mistaken for real JSON and crash the parser --
    the regex match attempt itself must fail gracefully."""
    raw = "See the array [not, valid, json, here without quotes] for details."
    with pytest.raises(json.JSONDecodeError):
        extract_json(raw)
