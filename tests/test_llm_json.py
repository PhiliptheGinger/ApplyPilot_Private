import pytest

from applypilot.llm_json import parse_llm_json


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ("  [1, 2]  ", [1, 2]),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Here you go:\n```json\n{"a": 1}\n```\nLet me know!', {"a": 1}),
        ('<think>reasoning {not json}</think>\n{"a": 1}', {"a": 1}),
        ('<think></think>{"a": 1}', {"a": 1}),
        ('It seems the list is truncated. {"variants": ["x", "y"]} Hope that helps.', {"variants": ["x", "y"]}),
        ('Sure: {"a": {"b": [1, {"c": 2}]}} trailing } junk', {"a": {"b": [1, {"c": 2}]}}),
        ('prefix [{"x": 1}] suffix', [{"x": 1}]),
    ],
)
def test_parses_common_model_reply_shapes(reply, expected):
    assert parse_llm_json(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "",
        None,
        "I could not produce JSON for this.",
        '{"a": 1, "b": }',
        # A malformed outer object must not be "rescued" by returning a nested fragment.
        '{"experience": [{"header": "Tech"}, ], "title": }',
    ],
)
def test_raises_value_error_when_no_complete_document(reply):
    with pytest.raises(ValueError):
        parse_llm_json(reply)


def test_callers_share_the_parser():
    """Each former one-off parser now accepts the shapes the shared one does."""
    from applypilot.discovery import hackernews
    from applypilot.scoring import local_tailor, tailor

    wrapped = 'Here is the result:\n{"title": "x"}\nThanks'
    assert tailor.extract_json(wrapped) == {"title": "x"}
    assert local_tailor._parse_plan(wrapped) == {"title": "x"}
    assert hackernews._parse_extracted_job(wrapped) == {"title": "x"}
