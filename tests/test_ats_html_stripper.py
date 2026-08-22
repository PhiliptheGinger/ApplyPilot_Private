"""Tests for the ONE canonical HTML-to-text converter shared by every
direct-API ATS scraper: applypilot.discovery.ats_common.strip_html_to_text.

Background: greenhouse.py, amazon.py, builtin.py, costco.py, and workday.py
each used to carry an independent, copy-pasted HTMLParser subclass for
converting a job posting's raw HTML `content` field to plain text -- and
every one of them flattened `<li>` to a bare newline, indistinguishable
from a `<p>` or `<div>`. That's how the same bug (genuinely itemized
`<ul><li>` requirements silently losing their bullet marker, so
local_tailor._REQUIREMENT_MARKER_RE finds nothing to extract) got
introduced independently across five scrapers.

Since these scrapers all populate `full_description` directly at DISCOVERY
time (Greenhouse/Amazon/Costco/Workday's public APIs return full content in
one call; see insert_normalized_jobs), enrichment/detail.py's OWN correct
`<li>`-handling (clean_description) never gets a second chance to fix it --
these are the only converters these jobs' HTML ever passes through.

This file tests the canonical implementation directly and confirms every
scraper's thin wrapper delegates to it (not a separate copy). See
test_greenhouse_pipeline_requirement_extraction.py for the pipeline-boundary
tests that prove this actually reaches local_tailor for a freshly-scraped
job, not just a standalone parser fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Every scraper's public stripper function, parametrized so a fix (or a
# regression) is verified across all of them at once, not just Greenhouse.
# ---------------------------------------------------------------------------

def _all_scraper_strip_fns():
    from applypilot.discovery import amazon, builtin, costco, greenhouse, workday
    return [
        ("greenhouse", greenhouse._strip_html),
        ("lever", __import__("applypilot.discovery.lever", fromlist=["_strip_html"])._strip_html),
        ("amazon", amazon._strip_html),
        ("builtin", builtin._strip_html),
        ("costco", costco._strip_html),
        ("workday", workday.strip_html),
    ]


@pytest.mark.parametrize("name,fn", _all_scraper_strip_fns())
def test_every_scraper_wrapper_preserves_li_bullets(name, fn):
    """All six modules delegate to the same canonical implementation --
    this proves it, rather than trusting that each file's plumbing is wired
    correctly by inspection alone."""
    html = "<ul><li>Experience with Python</li><li>Experience with Linux</li></ul>"
    text = fn(html)
    assert "- Experience with Python" in text, f"{name} did not preserve the bullet"
    assert "- Experience with Linux" in text, f"{name} did not preserve the bullet"


@pytest.mark.parametrize("name,fn", _all_scraper_strip_fns())
def test_every_scraper_wrapper_handles_empty_input(name, fn):
    assert fn("") == ""
    assert fn(None) == ""


# ---------------------------------------------------------------------------
# The canonical implementation itself.
# ---------------------------------------------------------------------------

def test_li_gets_bullet_marker():
    from applypilot.discovery.ats_common import strip_html_to_text

    html = "<ul><li>Experience with Python</li><li>Experience with Linux</li></ul>"
    text = strip_html_to_text(html)
    lines = [l for l in text.split("\n") if l.strip()]
    assert lines == ["- Experience with Python", "- Experience with Linux"]


def test_ordered_list_items_also_get_bullet_marker():
    from applypilot.discovery.ats_common import strip_html_to_text

    text = strip_html_to_text("<ol><li>Step one</li><li>Step two</li></ol>")
    lines = [l for l in text.split("\n") if l.strip()]
    assert lines == ["- Step one", "- Step two"]


def test_nested_list_inside_a_list_item():
    """A <li> containing its own nested <ul><li> must not crash and must
    still bullet both levels -- flattening nested items to their own bullet
    lines (rather than trying to preserve indentation) is an acceptable,
    deliberate simplification: local_tailor only needs "this is a candidate
    requirement line", not a faithful outline structure."""
    from applypilot.discovery.ats_common import strip_html_to_text

    html = "<ul><li>Outer item<ul><li>Inner item</li></ul></li><li>Second outer item</li></ul>"
    text = strip_html_to_text(html)
    assert "- Outer item" in text
    assert "- Inner item" in text
    assert "- Second outer item" in text


def test_nested_inline_tags_inside_li_do_not_break_the_bullet():
    from applypilot.discovery.ats_common import strip_html_to_text

    html = "<ul><li><strong>Experience</strong> with <em>Python</em> and <a href='#'>Linux</a></li></ul>"
    text = strip_html_to_text(html)
    assert "- Experience with Python and Linux" in text


def test_malformed_unclosed_tags_do_not_raise():
    from applypilot.discovery.ats_common import strip_html_to_text

    html = "<ul><li>Unclosed item<li>Second item</ul>"
    text = strip_html_to_text(html)  # must not raise
    assert "- Unclosed item" in text
    assert "- Second item" in text


def test_html_entities_are_decoded():
    """HTMLParser's default convert_charrefs=True handles this already --
    guarded here because our downstream regex/keyword matching depends on
    real characters, not literal "&amp;"/"&#39;" sequences."""
    from applypilot.discovery.ats_common import strip_html_to_text

    html = "<ul><li>Experience with C&amp;C++ and R&amp;D</li><li>Bachelor&#39;s degree</li></ul>"
    text = strip_html_to_text(html)
    assert "- Experience with C&C++ and R&D" in text
    assert "- Bachelor's degree" in text
    assert "&amp;" not in text
    assert "&#39;" not in text


def test_empty_list_items_do_not_produce_spurious_lines():
    from applypilot.discovery.ats_common import strip_html_to_text

    html = "<ul><li></li><li>Real requirement</li><li>   </li></ul>"
    text = strip_html_to_text(html)
    lines = [l for l in text.split("\n") if l.strip()]
    assert lines == ["- Real requirement"]


def test_list_inside_a_table_cell_still_bullets():
    from applypilot.discovery.ats_common import strip_html_to_text

    html = "<table><tr><td><ul><li>Requirement in a cell</li></ul></td></tr></table>"
    text = strip_html_to_text(html)
    assert "- Requirement in a cell" in text


def test_whitespace_collapsed_around_bullets():
    from applypilot.discovery.ats_common import strip_html_to_text

    html = "<ul>\n  <li>  Padded   requirement  </li>\n</ul>"
    text = strip_html_to_text(html)
    assert "- Padded requirement" in text


def test_script_and_style_content_skipped():
    from applypilot.discovery.ats_common import strip_html_to_text

    html = (
        "<p>Visible text</p>"
        "<script>var x = 1; document.write('<li>fake</li>');</script>"
        "<style>.a{color:red}</style>"
    )
    text = strip_html_to_text(html)
    assert text == "Visible text"
    assert "var x" not in text
    assert "color:red" not in text
    assert "fake" not in text


def test_block_tags_unchanged_bare_newline_not_bulleted():
    from applypilot.discovery.ats_common import strip_html_to_text

    html = "<p>Para</p><div>Div</div><h1>H1</h1><h2>H2</h2>text<br>after br<tr>Row</tr>"
    text = strip_html_to_text(html)
    assert "-" not in text
    lines = [l for l in text.split("\n") if l.strip()]
    assert lines == ["Para", "Div", "H1", "H2text", "after br", "Row"]


def test_mixed_prose_and_list_only_list_gets_bulleted():
    from applypilot.discovery.ats_common import strip_html_to_text

    html = (
        "<p>About the role: we build things.</p>"
        "<h4>Requirements</h4>"
        "<ul><li>Python</li><li>Linux</li></ul>"
        "<p>Benefits: great snacks.</p>"
    )
    text = strip_html_to_text(html)
    lines = [l for l in text.split("\n") if l.strip()]
    assert lines == [
        "About the role: we build things.",
        "Requirements",
        "- Python",
        "- Linux",
        "Benefits: great snacks.",
    ]


def test_feed_exception_falls_back_to_raw_input(monkeypatch):
    """A converter that could raise on some pathological input would lose
    the whole description; strip_html_to_text degrades to the raw string
    instead."""
    from applypilot.discovery import ats_common

    class _ExplodingStripper(ats_common._StructureAwareHTMLStripper):
        def feed(self, data):
            raise ValueError("boom")

    monkeypatch.setattr(ats_common, "_StructureAwareHTMLStripper", _ExplodingStripper)
    assert ats_common.strip_html_to_text("<li>whatever</li>") == "<li>whatever</li>"
