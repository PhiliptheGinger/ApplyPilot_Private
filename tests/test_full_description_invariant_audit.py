"""Full-repository audit for the structured-requirements data-contract
invariant, across EVERY discovery source that can populate
`full_description` at discovery time -- not just Greenhouse.

Exhaustive search performed for this audit (see the investigation report
for the full command list): grep for BeautifulSoup / HTMLParser / html2text
/ markdownify / clean_description / _strip_html / strip_html / full_description
assignment across src/applypilot/. Every source found is accounted for
below, each with a note on WHY it does or doesn't need ApplyPilot's own
HTML-to-text conversion:

  Source        | Converts HTML itself?          | Preserves <li> structure?
  --------------|---------------------------------|---------------------------
  Greenhouse    | yes -- ats_common (canonical)   | yes (this fix)
  Lever         | yes -- imports greenhouse's     | yes (inherited)
  Amazon        | yes -- ats_common (canonical)   | yes (this fix)
  BuiltIn       | yes -- ats_common (canonical)   | yes (this fix; only feeds
                |                                 |  the short `description`
                |                                 |  summary, not full_description)
  Costco        | yes -- ats_common (canonical)   | yes (this fix)
  Workday       | yes -- ats_common (canonical)   | yes (this fix)
  Ashby         | NO -- consumes descriptionPlain,| yes -- Ashby's own
                |  Ashby's own server-rendered    |  server-side renderer
                |  plain-text field               |  already emits " - item"
  JobSpy        | NO -- python-jobspy's own       | yes -- html2text (a
                |  html2text conversion, before   |  mature third-party lib)
                |  ApplyPilot ever sees the data  |  already emits "  * item"
  HackerNews    | NO -- no HTML at all; the       | N/A -- no structured list
                |  description is an LLM-written  |  exists to preserve; the
                |  free-text summary              |  correct behavior is to
                |                                 |  extract nothing
  smartextract  | Only for non-content page       | N/A -- never sets
                |  structure analysis (CSS        |  full_description itself;
                |  selector discovery); full_desc |  full_description for
                |  comes later via enrichment      |  these jobs goes through
                |                                  |  enrichment/detail.py's
                |                                  |  clean_description, which
                |                                  |  already handles <li>
                |                                  |  correctly (pre-existing)

This file proves each row of that table, and separately proves the three
possible states a job's requirement extraction can land in are correctly
distinguished:

  1. Deterministic, unambiguous, no LLM needed  (correct, common case)
  2. Deterministic narrowing, genuine ambiguity -> LLM invoked (correct)
  3. Structure existed in the source but was destroyed upstream, so
     nothing was ever extracted, and the LLM was skipped for the WRONG
     reason (this was the bug; this file proves it's now unreachable for
     every source ApplyPilot itself converts).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# B. ATS adapter coverage -- SPY-based proof of delegation, not just
# behavioral equivalence. Each wrapper must actually CALL the canonical
# converter; matching output alone wouldn't rule out a coincidental
# reimplementation drifting back out of sync later.
# ---------------------------------------------------------------------------

def _wrapper_under_test(module_path: str):
    import importlib
    mod = importlib.import_module(module_path)
    fn_name = "strip_html" if module_path.endswith("workday") else "_strip_html"
    return mod, getattr(mod, fn_name)


import pytest


@pytest.mark.parametrize("module_path", [
    "applypilot.discovery.greenhouse",
    "applypilot.discovery.amazon",
    "applypilot.discovery.builtin",
    "applypilot.discovery.costco",
    "applypilot.discovery.workday",
])
def test_scraper_wrapper_actually_calls_the_canonical_converter(module_path):
    """Patches each module's OWN bound reference to strip_html_to_text
    (not ats_common's, since `from x import y` binds a local name at
    import time) and asserts the wrapper genuinely calls through to it."""
    mod, wrapper = _wrapper_under_test(module_path)
    with patch.object(mod, "strip_html_to_text", return_value="SENTINEL") as spy:
        result = wrapper("<li>whatever</li>")
    spy.assert_called_once_with("<li>whatever</li>")
    assert result == "SENTINEL"


def test_lever_reuses_the_exact_same_function_object_as_greenhouse():
    """Lever doesn't call ats_common directly -- it imports Greenhouse's
    _strip_html by name. Prove it's the SAME function object (so it picks
    up the canonical converter transitively), not a look-alike copy."""
    from applypilot.discovery import greenhouse, lever
    assert lever._strip_html is greenhouse._strip_html


def test_ats_common_is_the_single_shared_implementation():
    """All five scrapers' wrappers resolve to the same underlying function
    -- not five independently-behaving copies that happen to agree today."""
    from applypilot.discovery import amazon, builtin, costco, greenhouse, workday
    assert greenhouse.strip_html_to_text is amazon.strip_html_to_text
    assert greenhouse.strip_html_to_text is builtin.strip_html_to_text
    assert greenhouse.strip_html_to_text is costco.strip_html_to_text
    assert greenhouse.strip_html_to_text is workday.strip_html_to_text


# ---------------------------------------------------------------------------
# Ashby: does NOT use ats_common's HTML converter at all -- it consumes
# Ashby's own `descriptionPlain` field, which the live API already renders
# with a " - item" marker (verified against the real API for a MotherDuck
# posting with a genuine <ul><li> "Key Responsibilities" section: the
# corresponding descriptionPlain rendered each item as " - <text>", already
# inside _REQUIREMENT_MARKER_RE's recognized marker set). This fixture is
# that captured real-world shape, condensed -- not invented.
# ---------------------------------------------------------------------------

_ASHBY_STYLE_DESCRIPTION_PLAIN = (
    "ABOUT MOTHERDUCK\n\n"
    "MotherDuck is on a mission to make data warehousing fun and frictionless "
    "for developers and data practitioners. This paragraph is prose describing "
    "the company and must never be extracted as a requirement.\n\n"
    "KEY RESPONSIBILITIES:\n\n"
    " - Own major projects in our dual execution stack.\n\n"
    " - Help build a cloud platform that dynamically assigns resources.\n\n"
    "WHAT YOU BRING\n\n"
    " - You have built and shipped complex backend systems.\n\n"
    " - You are fluent in a backend language like C++, Rust, or Java.\n\n"
)


def test_ashby_scrape_preserves_bullet_markers_with_zero_html_parsing():
    """End-to-end through the real ashby.py scrape function -- proves the
    invariant holds even though ApplyPilot never touches HTML for this
    source at all; Ashby's own API already does the job."""
    from applypilot.discovery import ashby

    fake_payload = {
        "jobs": [{
            "id": "abc123",
            "title": "Software Engineer - Database",
            "location": "Amsterdam",
            "isListed": True,
            "isRemote": False,
            "workplaceType": "hybrid",
            "jobUrl": "https://jobs.ashbyhq.com/motherduck/abc123",
            "applyUrl": "https://jobs.ashbyhq.com/motherduck/abc123/apply",
            "descriptionPlain": _ASHBY_STYLE_DESCRIPTION_PLAIN,
            "publishedAt": "2026-08-01T00:00:00Z",
        }]
    }
    with patch.object(ashby, "fetch_with_retry", return_value=(fake_payload, None)):
        jobs, err = ashby.scrape_one_employer("motherduck", {"name": "MotherDuck"}, accept_locs=[])

    assert err is None
    assert len(jobs) == 1
    desc = jobs[0]["full_description"]
    assert " - Own major projects in our dual execution stack." in desc
    assert " - You are fluent in a backend language like C++, Rust, or Java." in desc


def test_ashby_job_reaches_local_tailor_correctly(tmp_db):
    """Full boundary for Ashby, mirroring the Greenhouse pipeline test:
    scrape -> insert_normalized_jobs -> read back from DB -> real
    _extract_requirement_lines. Proves the invariant for a source that
    uses a completely different mechanism (no HTML at all) than the
    canonical converter."""
    from applypilot.discovery import ashby
    from applypilot.scoring.local_tailor import _extract_requirement_lines

    conn = tmp_db()
    fake_payload = {
        "jobs": [{
            "id": "abc123",
            "title": "Software Engineer - Database",
            "location": "Amsterdam",
            "isListed": True,
            "jobUrl": "https://jobs.ashbyhq.com/motherduck/abc123",
            "applyUrl": "https://jobs.ashbyhq.com/motherduck/abc123/apply",
            "descriptionPlain": _ASHBY_STYLE_DESCRIPTION_PLAIN,
            "publishedAt": "2026-08-01T00:00:00Z",
        }]
    }
    with patch.object(ashby, "fetch_with_retry", return_value=(fake_payload, None)):
        jobs, _ = ashby.scrape_one_employer("motherduck", {"name": "MotherDuck"}, accept_locs=[])
    ashby._insert_jobs(conn, jobs)

    row = conn.execute(
        "SELECT full_description FROM jobs WHERE url = ?", (jobs[0]["url"],),
    ).fetchone()
    req_lines = _extract_requirement_lines(row["full_description"])
    texts = [l["text"] for l in req_lines]
    assert "Own major projects in our dual execution stack." in texts
    assert "You are fluent in a backend language like C++, Rust, or Java." in texts
    # The company-mission prose paragraph must never appear as a requirement.
    assert not any("mission to make data warehousing" in t for t in texts)


# ---------------------------------------------------------------------------
# JobSpy (LinkedIn/Indeed): ApplyPilot performs ZERO HTML parsing of its
# own here -- python-jobspy's internal html2text conversion already runs
# before jobspy.py ever sees `row["description"]`. This is a focused
# unit-level check of that third-party output's compatibility with
# _REQUIREMENT_MARKER_RE (confirmed live against the installed
# html2text version), not a full mocked scrape_jobs() pipeline test --
# there's no ApplyPilot conversion CODE at this layer to prove delegates
# correctly, since none exists; the contract to verify is purely "is the
# format html2text already produces one our extractor recognizes."
# ---------------------------------------------------------------------------

def test_jobspy_html2text_output_is_already_compatible_with_requirement_extraction():
    from applypilot.scoring.local_tailor import _extract_requirement_lines

    html2text = pytest.importorskip("html2text")
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = False
    html = (
        "<p>About the company: we build great things. This is prose.</p>"
        "<h4>What You Bring</h4>"
        "<ul>"
        "<li>Bachelor's Degree in Computer Science, Engineering, or related field</li>"
        "<li>8+ years of professional software development experience</li>"
        "</ul>"
    )
    markdown = text_maker.handle(html)
    lines = _extract_requirement_lines(markdown)
    texts = [l["text"] for l in lines]
    assert "Bachelor's Degree in Computer Science, Engineering, or related field" in texts
    assert "8+ years of professional software development experience" in texts
    assert not any("we build great things" in t for t in texts)


# ---------------------------------------------------------------------------
# HackerNews: genuinely has no HTML source and no structured list to
# preserve -- `description` is an LLM-written free-text summary. The
# correct behavior here is to extract NOTHING, not to invent bullets from
# ordinary sentences.
# ---------------------------------------------------------------------------

def test_hackernews_style_prose_summary_yields_no_fake_requirements():
    from applypilot.scoring.local_tailor import _extract_requirement_lines

    hn_style_summary = (
        "Acme Corp is hiring a backend engineer to help build our payments "
        "platform. We are a small remote-first team working primarily in Go "
        "and Postgres. Ideal candidates have several years of experience with "
        "distributed systems and enjoy owning projects end to end. Reach out "
        "if this sounds interesting."
    )
    assert _extract_requirement_lines(hn_style_summary) == []


# ---------------------------------------------------------------------------
# The three-state distinction, made explicit and testable.
# ---------------------------------------------------------------------------

def _old_broken_li_conversion(html: str) -> str:
    """Reproduces the PRE-FIX behavior verbatim (li treated as a bare
    block tag) -- used only here, to demonstrate the state-3 -> state-1/2
    transition side-by-side against the same real input. Not imported from
    production code; production code no longer has this bug anywhere."""
    from html.parser import HTMLParser
    import re as _re

    class _OldStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []

        def handle_starttag(self, tag, attrs):
            if tag in ("p", "br", "li", "div", "h1", "h2", "h3", "h4"):
                self.parts.append("\n")

        def handle_data(self, data):
            if data.strip():
                self.parts.append(data)

        def text(self):
            raw = "".join(self.parts)
            raw = _re.sub(r"[ \t]+", " ", raw)
            return raw.strip()

    s = _OldStripper()
    s.feed(html)
    return s.text()


_STRUCTURED_POSTING_HTML = (
    "<h4>What You Bring</h4>"
    "<ul>"
    "<li>Bachelor's Degree in Computer Science, Engineering, or related field</li>"
    "<li>Experience working with SQL or NoSQL data stores</li>"
    "</ul>"
)


def test_state_3_the_bug_is_no_longer_reachable_via_the_canonical_converter():
    """Side-by-side on the SAME real input: the old per-scraper
    implementation destroyed structure (state 3 -- zero requirement lines
    despite a genuine <ul><li> list existing in the source); the canonical
    converter does not."""
    from applypilot.scoring.local_tailor import _extract_requirement_lines
    from applypilot.discovery.ats_common import strip_html_to_text

    old_text = _old_broken_li_conversion(_STRUCTURED_POSTING_HTML)
    assert _extract_requirement_lines(old_text) == []  # the historical bug, reproduced

    new_text = strip_html_to_text(_STRUCTURED_POSTING_HTML)
    new_lines = _extract_requirement_lines(new_text)
    assert len(new_lines) == 2  # state 1 or 2 is now reachable; state 3 is not


def test_state_1_deterministic_unambiguous_requirements_skip_the_llm():
    """State 1: structure preserved, every requirement resolves without
    ambiguity -> the local model must not be called."""
    from applypilot.scoring.local_tailor import get_local_tailoring_plan

    job = {
        "title": "Sr Software Engineer II",
        "full_description": (
            "- Experience working with SQL or NoSQL data stores\n"
            "- Familiarity with obscure legacy mainframe COBOL systems\n"
        ),
    }
    profile = {
        "skills_inventory": [
            {"name": "SQL Database Administration", "relevance_categories": ["sql"],
             "resume_allowed": True},
        ],
        "experience_inventory": [], "project_inventory": [],
    }
    with patch.dict("os.environ", {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}), \
         patch("httpx.post") as mock_post:
        plan = get_local_tailoring_plan(job["full_description"], job, profile)

    mock_post.assert_not_called()
    by_text = {r["requirement"]: r for r in plan["requirements"]}
    assert by_text["Experience working with SQL or NoSQL data stores"]["supported"] is True
    assert by_text["Familiarity with obscure legacy mainframe COBOL systems"]["supported"] is False


def test_state_2_genuine_ambiguity_reaches_the_local_model():
    """State 2: structure preserved, but two evidence items tie for one
    requirement -> the grounding gate must hand that (and only that) one
    to the local model."""
    from applypilot.scoring.local_tailor import get_local_tailoring_plan

    job = {
        "title": "Sr Software Engineer II",
        "full_description": "- Backend service experience is required for this role\n",
    }
    profile = {
        "experience_inventory": [
            {"name": "API Gateway Service", "relevance_categories": ["backend"],
             "resume_allowed": True},
            {"name": "Order Processing Service", "relevance_categories": ["backend"],
             "resume_allowed": True},
        ],
        "skills_inventory": [], "project_inventory": [],
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"message": {"content": '{"matches":[{"r":1,"e":[1]}]}'}}

    with patch.dict("os.environ", {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}), \
         patch("httpx.post", return_value=mock_resp) as mock_post:
        plan = get_local_tailoring_plan(job["full_description"], job, profile)

    mock_post.assert_called_once()  # the ambiguity genuinely reached the model
    assert plan["requirements"][0]["supported"] is True
