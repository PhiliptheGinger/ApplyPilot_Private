"""Regression tests for the 2026-08-25 fix: malformed/non-conforming LLM
scoring responses must fail rather than silently becoming fit_score=0.

Two independent layers were involved, tested here end-to-end:
  1. scorer.py's _parse_score_response now returns score=None (not 0) when
     no "SCORE: <digits>" line can be found -- see
     test_scorer_prefilter.py's dedicated parser-level tests for that half.
  2. scorer.py's score_job() must attach an "error" key whenever the parsed
     score is None, since it returns that dict directly (not through the
     except-branch, which already added "error") -- without this,
     _flush_score_batch's failure branch (`r["error"]`) would KeyError on
     any malformed-but-non-exception response.
  3. _flush_score_batch itself must route a None-score result through the
     existing failure path (retry/backoff), never write a fake fit_score,
     and never clear score_error -- the downstream half of the guarantee.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _job(title="Backend Engineer"):
    return {
        "title": title,
        "site": "TestCo",
        "location": "Remote (US)",
        "full_description": "We need a backend engineer with Python experience.",
    }


# ---------------------------------------------------------------------------
# score_job(): malformed LLM output must come back with score=None + "error"
# ---------------------------------------------------------------------------


def test_score_job_empty_llm_response_returns_none_score_with_error():
    import applypilot.scoring.scorer as scorer_mod

    with patch.object(scorer_mod, "get_stage_client") as mock_get_client:
        mock_get_client.return_value.chat.return_value = ""
        result = scorer_mod.score_job("Resume text.", _job(), {"education": []})
    assert result["score"] is None
    assert "error" in result
    assert "parseable SCORE" in result["error"]


def test_score_job_refusal_response_returns_none_score_with_error():
    import applypilot.scoring.scorer as scorer_mod

    with patch.object(scorer_mod, "get_stage_client") as mock_get_client:
        mock_get_client.return_value.chat.return_value = "I'm not able to evaluate this job posting."
        result = scorer_mod.score_job("Resume text.", _job(), {"education": []})
    assert result["score"] is None
    assert "error" in result


def test_score_job_malformed_score_value_returns_none_score_with_error():
    import applypilot.scoring.scorer as scorer_mod

    with patch.object(scorer_mod, "get_stage_client") as mock_get_client:
        mock_get_client.return_value.chat.return_value = "SCORE: unclear\nREASONING: hmm"
        result = scorer_mod.score_job("Resume text.", _job(), {"education": []})
    assert result["score"] is None
    assert "error" in result


def test_score_job_legitimate_score_has_no_error_key():
    """Contrast case: a real, well-formed response must not gain a spurious
    "error" key -- the key's presence now means exactly "score is None".

    Job description states explicit compensation (2026-08-30 compensation
    layer): this test's own point is the "error" key contract, not
    compensation, so the fixture is given stated pay to keep the
    compensation adjustment a no-op and preserve the original exact-score
    assertion -- an unstated-compensation job would legitimately (and
    correctly, per the new feature) score below 8 here, which would be
    testing the wrong thing for this test's stated purpose."""
    import applypilot.scoring.scorer as scorer_mod

    job = _job()
    job["full_description"] += " Pay: $90,000-$110,000 annually."

    with patch.object(scorer_mod, "get_stage_client") as mock_get_client:
        mock_get_client.return_value.chat.return_value = (
            "ELIGIBILITY: eligible\nSCORE: 8\nKEYWORDS: python\nREASONING: solid match."
        )
        result = scorer_mod.score_job("Resume text.", job, {"education": []})
    assert result["score"] == 8
    assert "error" not in result


# ---------------------------------------------------------------------------
# _flush_score_batch(): a None-score result must take the failure path, not
# silently persist fit_score=0
# ---------------------------------------------------------------------------


def test_flush_score_batch_malformed_response_does_not_write_fake_zero_score(tmp_db, seed_job):
    from applypilot.scoring.scorer import _flush_score_batch

    conn = tmp_db()
    job = seed_job(conn, fit_score=None, full_description="x", state="enriched")

    batch = [
        {
            "url": job["url"],
            "score": None,
            "keywords": "",
            "reasoning": "",
            "error": "LLM error: response did not contain a parseable SCORE line",
        }
    ]
    _flush_score_batch(conn, batch, _now())

    row = conn.execute(
        "SELECT fit_score, score_error, state FROM jobs WHERE url = ?",
        (job["url"],),
    ).fetchone()
    assert row["fit_score"] is None
    assert row["score_error"] is not None
    assert row["state"] == "enriched"  # never transitioned to scored/low_score


def test_flush_score_batch_legitimate_zero_would_still_be_impossible_but_real_score_is_written(tmp_db, seed_job):
    """Contrast/regression case: a genuine successful score is still written
    normally -- the guard added for the None case must not affect the
    ordinary success path."""
    from applypilot.scoring.scorer import _flush_score_batch

    conn = tmp_db()
    job = seed_job(conn, fit_score=None, full_description="x", state="enriched")

    batch = [
        {
            "url": job["url"],
            "score": 8,
            "keywords": "python",
            "reasoning": "solid match",
            "eligibility": "eligible",
        }
    ]
    _flush_score_batch(conn, batch, _now())

    row = conn.execute(
        "SELECT fit_score, score_error, state FROM jobs WHERE url = ?",
        (job["url"],),
    ).fetchone()
    assert row["fit_score"] == 8
    assert row["score_error"] is None
    assert row["state"] == "scored"
