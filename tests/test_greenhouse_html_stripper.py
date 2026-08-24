"""Regression coverage for discovery/greenhouse.py's HTML-to-text stripper.

Root cause traced in a live investigation: Greenhouse job postings commonly
use a genuine <ul><li>...</li></ul> requirements list (confirmed against the
real Greenhouse API for Axon's "What You Bring" section), but
_HTMLStripper.handle_starttag treated <li> identically to <p>/<div>/etc. --
just a bare newline, no bullet marker. local_tailor._REQUIREMENT_MARKER_RE
looks specifically for a leading "-"/"*"/"•"/"‣"/"▪" or "N." / "N)" marker to
recognize a requirement line, so every Greenhouse-sourced job's genuinely
itemized requirements silently extracted as zero requirement lines --
get_local_tailoring_plan then (correctly, given that input, but misleadingly
given the real posting) skipped the local model entirely, before its
grounding/pair-scoring logic ever ran.

The fix is narrow: <li> now gets its own branch that inserts "- " instead of
a bare "\n", matching the precedent already set in enrichment/detail.py's
clean_description(). These tests cover that fix, guard that everything else
(<p>, <div>, headings, <br>, script/style skipping) is unchanged, and bridge
all the way to local_tailor's own extractor to prove the two modules now
actually connect for this case.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_li_items_get_bullet_marker():
    from applypilot.discovery.greenhouse import _strip_html

    html = (
        "<ul>"
        "<li>Experience with Python</li>"
        "<li>Experience with Linux</li>"
        "<li>Ability to troubleshoot hardware</li>"
        "</ul>"
    )
    text = _strip_html(html)
    assert "- Experience with Python" in text
    assert "- Experience with Linux" in text
    assert "- Ability to troubleshoot hardware" in text


def test_li_items_appear_on_their_own_lines_in_order():
    from applypilot.discovery.greenhouse import _strip_html

    html = "<ul><li>First requirement</li><li>Second requirement</li></ul>"
    text = _strip_html(html)
    lines = [l for l in text.split("\n") if l.strip()]
    assert lines == ["- First requirement", "- Second requirement"]


def test_real_axon_style_requirements_list_extracts_correctly():
    """The exact shape confirmed against the live Greenhouse API for
    Axon's "What You Bring" section, condensed."""
    from applypilot.discovery.greenhouse import _strip_html

    html = (
        "<h4><strong>What You Bring</strong></h4>\n"
        "<ul>\n"
        "<li>Bachelor's Degree in Computer Science, Engineering, or related field</li>\n"
        "<li>8+ years of professional software development experience</li>\n"
        "<li>Experience working with SQL or NoSQL data stores</li>\n"
        "</ul>\n"
    )
    text = _strip_html(html)
    assert "- Bachelor's Degree in Computer Science, Engineering, or related field" in text
    assert "- 8+ years of professional software development experience" in text
    assert "- Experience working with SQL or NoSQL data stores" in text


def test_stripped_li_lines_are_recognized_by_requirement_marker_regex():
    """Bridges the two modules: the stripper's output must actually be
    something local_tailor._REQUIREMENT_MARKER_RE recognizes -- a bullet
    character alone proves nothing if the downstream regex still can't see
    it (e.g. wrong spacing)."""
    from applypilot.discovery.greenhouse import _strip_html
    from applypilot.scoring.local_tailor import _REQUIREMENT_MARKER_RE

    html = "<ul><li>Experience with Python</li><li>Experience with Linux</li></ul>"
    text = _strip_html(html)
    matches = _REQUIREMENT_MARKER_RE.findall(text)
    assert matches == ["Experience with Python", "Experience with Linux"]


def test_stripped_li_lines_extracted_as_requirement_lines_end_to_end():
    """Full path: HTML -> greenhouse stripper -> local_tailor's own
    requirement-line extractor, with no other change to local_tailor.py."""
    from applypilot.discovery.greenhouse import _strip_html
    from applypilot.scoring.local_tailor import _extract_requirement_lines

    html = (
        "<ul>"
        "<li>Experience with Python</li>"
        "<li>Experience with Linux</li>"
        "<li>Ability to troubleshoot hardware</li>"
        "</ul>"
    )
    text = _strip_html(html)
    lines = _extract_requirement_lines(text)
    texts = [l["text"] for l in lines]
    assert texts == [
        "Experience with Python",
        "Experience with Linux",
        "Ability to troubleshoot hardware",
    ]


def test_nested_inline_tags_inside_li_do_not_break_the_bullet():
    from applypilot.discovery.greenhouse import _strip_html

    html = "<ul><li><strong>Experience</strong> with <em>Python</em></li></ul>"
    text = _strip_html(html)
    assert "- Experience with Python" in text


def test_ordered_list_items_also_get_a_bullet_marker():
    """<ol><li> is treated the same as <ul><li> -- _REQUIREMENT_MARKER_RE
    accepts either a bullet char or a numeric marker, so a plain "- " is a
    valid (if not numerically faithful) requirement-line marker either way."""
    from applypilot.discovery.greenhouse import _strip_html
    from applypilot.scoring.local_tailor import _REQUIREMENT_MARKER_RE

    html = "<ol><li>Step one</li><li>Step two</li></ol>"
    text = _strip_html(html)
    assert _REQUIREMENT_MARKER_RE.findall(text) == ["Step one", "Step two"]


# ---------------------------------------------------------------------------
# Regression guards: everything else must stay exactly as it was.
# ---------------------------------------------------------------------------


def test_paragraph_and_div_still_produce_bare_newlines_not_bullets():
    from applypilot.discovery.greenhouse import _strip_html

    html = "<p>First paragraph.</p><div>Second block.</div>"
    text = _strip_html(html)
    assert text == "First paragraph.\nSecond block."
    assert "-" not in text


def test_headings_still_produce_bare_newlines_not_bullets():
    from applypilot.discovery.greenhouse import _strip_html

    html = "<h1>Title</h1><h2>Subtitle</h2><h3>Sub</h3><h4>Sub sub</h4>"
    text = _strip_html(html)
    assert text == "Title\nSubtitle\nSub\nSub sub"


def test_br_still_produces_a_bare_newline():
    from applypilot.discovery.greenhouse import _strip_html

    html = "Line one<br>Line two"
    text = _strip_html(html)
    assert text == "Line one\nLine two"


def test_table_row_still_produces_a_bare_newline():
    from applypilot.discovery.greenhouse import _strip_html

    html = "<table><tr>Row one</tr><tr>Row two</tr></table>"
    text = _strip_html(html)
    assert text == "Row one\nRow two"


def test_script_and_style_content_still_skipped():
    from applypilot.discovery.greenhouse import _strip_html

    html = "<p>Visible text</p><script>var x = 1;</script><style>.a{color:red}</style>"
    text = _strip_html(html)
    assert text == "Visible text"
    assert "var x" not in text
    assert "color:red" not in text


def test_mixed_list_and_prose_preserves_both_forms():
    """A realistic mixed posting: prose paragraphs stay bare, only the
    <li> items get bulleted -- the fix must not bleed into unrelated tags."""
    from applypilot.discovery.greenhouse import _strip_html

    html = (
        "<p>About the role: we build things.</p>"
        "<h4>Requirements</h4>"
        "<ul><li>Python</li><li>Linux</li></ul>"
        "<p>Benefits: great snacks.</p>"
    )
    text = _strip_html(html)
    lines = [l for l in text.split("\n") if l.strip()]
    assert lines == [
        "About the role: we build things.",
        "Requirements",
        "- Python",
        "- Linux",
        "Benefits: great snacks.",
    ]


def test_empty_html_returns_empty_string():
    from applypilot.discovery.greenhouse import _strip_html

    assert _strip_html("") == ""
    assert _strip_html(None) == ""
