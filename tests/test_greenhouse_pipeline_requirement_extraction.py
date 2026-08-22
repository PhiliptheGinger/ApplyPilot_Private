"""Pipeline-boundary tests: prove a FRESHLY SCRAPED Greenhouse job reaches
local_tailor's requirement extraction correctly -- not just that
_HTMLStripper/strip_html_to_text behaves correctly in isolation.

A prior fix attempt demonstrated exactly the gap this file closes: unit
tests of the HTML stripper passed, while `debug-local-plan` against an
EXISTING database record still reported zero requirement lines, because
that record was scraped before the fix and the database is never
automatically re-processed after a parser change (see the "stale rows"
tests at the bottom of this file, and the module docstring in
discovery/ats_common.py for the full incident).

This file drives the real functions at each hop of the actual data
contract, with only the network call mocked:

    Greenhouse API JSON (content=true)
        -> greenhouse.scrape_one_employer()      [real function]
        -> ats_common.insert_normalized_jobs()    [real function, tmp_db]
        -> read back from the database            [real schema]
        -> local_tailor.rank_profile_evidence() +
           local_tailor.get_local_tailoring_plan() [real functions]

covering both halves the task asked for:
    A. structural preservation (requirement lines actually extracted)
    B. end-to-end tailoring behavior (deterministic resolution AND
       genuine ambiguity both reach the correct destination, and
       unrelated prose never becomes a fake requirement)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# A realistic Greenhouse posting: prose (must NOT become requirements),
# a genuinely itemized requirements list with three deliberately different
# grounding outcomes baked in via the profile fixture below:
#   - "Bachelor's Degree..."        -> no matching evidence at all
#                                      (deterministic: unsupported, no LLM)
#   - "...SQL or NoSQL..."          -> exactly one matching evidence item
#                                      (deterministic: supported, no LLM)
#   - "Backend service experience..." -> two evidence items tied
#                                      (genuinely ambiguous -> LLM)
# ---------------------------------------------------------------------------
_AXON_STYLE_CONTENT_HTML = (
    "<p>At Axon, we&#39;re on a mission to Protect Life. This sentence is "
    "prose describing the company, not a requirement, and must never be "
    "extracted as one.</p>"
    "<h4><strong>What You Bring</strong></h4>"
    "<ul>"
    "<li>Bachelor's Degree in Computer Science, Engineering, or related field</li>"
    "<li>Experience working with SQL or NoSQL data stores</li>"
    "<li>Backend service experience is required for this role</li>"
    "</ul>"
)


def _fake_greenhouse_payload():
    return {
        "jobs": [
            {
                "id": 7577015003,
                "title": "Sr Software Engineer II",
                "location": {"name": "Seattle, WA"},
                "absolute_url": "https://job-boards.greenhouse.io/axon/jobs/7577015003",
                "content": _AXON_STYLE_CONTENT_HTML,
                "first_published": "2026-08-01T00:00:00Z",
                "updated_at": "2026-08-10T00:00:00Z",
            }
        ]
    }


def _profile_with_deliberate_grounding_outcomes():
    return {
        "skills_inventory": [
            {"name": "SQL Database Administration", "relevance_categories": ["sql"],
             "resume_allowed": True},
        ],
        "experience_inventory": [
            {"name": "API Gateway Service", "relevance_categories": ["backend"],
             "resume_allowed": True},
            {"name": "Order Processing Service", "relevance_categories": ["backend"],
             "resume_allowed": True},
        ],
        "project_inventory": [],
    }


# ---------------------------------------------------------------------------
# A. Structural preservation, through the REAL discovery function
# ---------------------------------------------------------------------------

def test_scrape_one_employer_preserves_bulleted_requirements():
    """discovery/greenhouse.py's real scrape function, given a mocked API
    response shaped exactly like Greenhouse's actual payload, must produce
    a full_description with bullet-marked requirement lines -- not the
    HTML-stripper unit test's synthetic fixture, the actual scraper path."""
    from applypilot.discovery import greenhouse

    with patch("applypilot.discovery.ats_common.fetch_with_retry",
               return_value=(_fake_greenhouse_payload(), None)):
        jobs, err = greenhouse.scrape_one_employer("axon", {"name": "Axon"}, accept_locs=[])

    assert err is None
    assert len(jobs) == 1
    desc = jobs[0]["full_description"]
    assert "- Bachelor's Degree in Computer Science, Engineering, or related field" in desc
    assert "- Experience working with SQL or NoSQL data stores" in desc
    assert "- Backend service experience is required for this role" in desc
    # The prose sentence must survive (it's real description content) but
    # without ever picking up a bullet marker of its own.
    assert "mission to Protect Life" in desc
    assert "- At Axon" not in desc


def test_bullets_survive_the_database_round_trip(tmp_db):
    """insert_normalized_jobs -> read back from a real (tmp) database --
    proves the bullet markers aren't an artifact of the in-memory dict that
    gets lost on the way into storage."""
    from applypilot.discovery import greenhouse
    from applypilot.discovery.ats_common import insert_normalized_jobs

    conn = tmp_db()
    with patch("applypilot.discovery.ats_common.fetch_with_retry",
               return_value=(_fake_greenhouse_payload(), None)):
        jobs, _ = greenhouse.scrape_one_employer("axon", {"name": "Axon"}, accept_locs=[])

    new, existing = insert_normalized_jobs(conn, jobs, "Greenhouse", "greenhouse_api")
    assert new == 1

    row = conn.execute(
        "SELECT full_description, detail_scraped_at, state FROM jobs WHERE url = ?",
        (jobs[0]["url"],),
    ).fetchone()
    assert "- Experience working with SQL or NoSQL data stores" in row["full_description"]
    # Confirms the "enrichment never touches this job" finding from the
    # investigation: full_description present at insert time means
    # detail_scraped_at is set immediately and the job starts life already
    # "enriched" -- there is no second pass where a different converter
    # could have fixed (or re-broken) this.
    assert row["detail_scraped_at"] is not None
    assert row["state"] == "enriched"


# ---------------------------------------------------------------------------
# B. End-to-end tailoring behavior, fed by the freshly-scraped-and-stored row
# ---------------------------------------------------------------------------

def test_freshly_scraped_job_extracts_and_grounds_correctly(tmp_db):
    """The full boundary in one test: scrape -> insert -> read back ->
    local_tailor. Proves all three deliberately-designed outcomes land
    where they should:
      - no evidence at all -> deterministically unsupported, no LLM
      - exactly one matching item -> deterministically supported, no LLM
      - two tied items -> genuine ambiguity -> the local model IS invoked
    and that the prose sentence never became a fourth fake requirement.
    """
    from applypilot.discovery import greenhouse
    from applypilot.discovery.ats_common import insert_normalized_jobs
    from applypilot.scoring.local_tailor import (
        _extract_requirement_lines, get_local_tailoring_plan, rank_profile_evidence,
    )

    conn = tmp_db()
    with patch("applypilot.discovery.ats_common.fetch_with_retry",
               return_value=(_fake_greenhouse_payload(), None)):
        jobs, _ = greenhouse.scrape_one_employer("axon", {"name": "Axon"}, accept_locs=[])
    insert_normalized_jobs(conn, jobs, "Greenhouse", "greenhouse_api")

    row = conn.execute(
        "SELECT title, full_description FROM jobs WHERE url = ?", (jobs[0]["url"],),
    ).fetchone()
    job = {"title": row["title"], "full_description": row["full_description"]}
    profile = _profile_with_deliberate_grounding_outcomes()

    # --- A: structural preservation, via the real extractor -----------
    req_lines = _extract_requirement_lines(job["full_description"])
    texts = [l["text"] for l in req_lines]
    assert "Bachelor's Degree in Computer Science, Engineering, or related field" in texts
    assert "Experience working with SQL or NoSQL data stores" in texts
    assert "Backend service experience is required for this role" in texts
    # The prose sentence must never appear as a requirement line.
    assert not any("mission to Protect Life" in t for t in texts)
    assert len(texts) == 3  # exactly the three <li> items, nothing invented

    evidence = rank_profile_evidence(job, profile, top_n=6)
    assert {e["name"] for e in evidence} == {
        "SQL Database Administration", "API Gateway Service", "Order Processing Service",
    }

    # --- B: deterministic resolution + genuine ambiguity ----------------
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    # The model is only ever shown the ambiguous requirement; whichever of
    # the two tied evidence numbers it names, both are legitimate.
    mock_resp.json.return_value = {"message": {"content": '{"matches":[{"r":3,"e":[1]}]}'}}

    with patch.dict("os.environ", {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}), \
         patch("httpx.post", return_value=mock_resp) as mock_post:
        plan = get_local_tailoring_plan(job["full_description"], job, profile)

    assert plan is not None
    mock_post.assert_called_once()  # the grounding gate was NOT bypassed
    by_text = {r["requirement"]: r for r in plan["requirements"]}

    bachelors = by_text["Bachelor's Degree in Computer Science, Engineering, or related field"]
    assert bachelors["supported"] is False
    assert bachelors["resume_evidence"] == []

    sql_req = by_text["Experience working with SQL or NoSQL data stores"]
    assert sql_req["supported"] is True
    assert sql_req["resume_evidence"] == ["SQL Database Administration"]

    backend_req = by_text["Backend service experience is required for this role"]
    assert backend_req["supported"] is True
    assert backend_req["resume_evidence"] == ["API Gateway Service"]

    # The prompt actually sent to the model must show ONLY the ambiguous
    # requirement -- the deterministically-resolved two must never be
    # re-litigated by the LLM.
    sent_payload = mock_post.call_args.kwargs["json"]
    user_msg = sent_payload["messages"][1]["content"]
    assert "Backend service experience is required for this role" in user_msg
    assert "Bachelor's Degree" not in user_msg
    assert "SQL or NoSQL" not in user_msg


def test_unambiguous_job_skips_the_llm_entirely(tmp_db):
    """A job where every requirement resolves deterministically (0 or 1
    candidate each) must never invoke the local model at all -- confirming
    the fix does not make the grounding gate more trigger-happy than it
    needs to be."""
    from applypilot.discovery import greenhouse
    from applypilot.discovery.ats_common import insert_normalized_jobs
    from applypilot.scoring.local_tailor import get_local_tailoring_plan

    conn = tmp_db()
    with patch("applypilot.discovery.ats_common.fetch_with_retry",
               return_value=(_fake_greenhouse_payload(), None)):
        jobs, _ = greenhouse.scrape_one_employer("axon", {"name": "Axon"}, accept_locs=[])
    insert_normalized_jobs(conn, jobs, "Greenhouse", "greenhouse_api")

    row = conn.execute(
        "SELECT title, full_description FROM jobs WHERE url = ?", (jobs[0]["url"],),
    ).fetchone()
    job = {"title": row["title"], "full_description": row["full_description"]}

    # Only the SQL item is present -- no second "backend" item to tie with,
    # so every requirement resolves to 0 or 1 candidates.
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
    assert plan is not None
    by_text = {r["requirement"]: r for r in plan["requirements"]}
    assert by_text["Experience working with SQL or NoSQL data stores"]["supported"] is True
    assert by_text["Backend service experience is required for this role"]["supported"] is False


# ---------------------------------------------------------------------------
# Stale database rows: existing records are NOT retroactively reprocessed.
# ---------------------------------------------------------------------------

def test_stale_row_from_before_the_fix_is_not_retroactively_repaired(tmp_db):
    """Documents, as a test (not just prose), that a row inserted with the
    OLD unmarked-flat text stays exactly as stored -- there is no
    background job that walks the database and re-derives
    full_description after a parser change. A fresh scrape (the tests
    above) is what's fixed; existing rows need an explicit re-scrape,
    covered in the report's "how to verify/re-enrich" section rather than
    an automatic mechanism."""
    from applypilot.discovery.ats_common import insert_normalized_jobs
    from applypilot.scoring.local_tailor import _extract_requirement_lines

    conn = tmp_db()
    pre_fix_style_description = (
        "What You Bring\n"
        "Bachelor's Degree in Computer Science, Engineering, or related field\n"
        "Experience working with SQL or NoSQL data stores\n"
    )
    jobs = [{
        "url": "https://job-boards.greenhouse.io/axon/jobs/stale-example",
        "title": "Sr Software Engineer II",
        "full_description": pre_fix_style_description,
        "application_url": "https://job-boards.greenhouse.io/axon/jobs/stale-example",
        "employer_name": "Axon",
    }]
    insert_normalized_jobs(conn, jobs, "Greenhouse", "greenhouse_api")

    row = conn.execute(
        "SELECT full_description FROM jobs WHERE url = ?", (jobs[0]["url"],),
    ).fetchone()
    # Still flat, still zero requirement lines -- the fix only changes what
    # NEW scrapes produce, not what's already stored.
    assert _extract_requirement_lines(row["full_description"]) == []
