"""Regression tests for the 2026-08-27 P0 fix: scorer.run_scoring() used to
read ~/.applypilot/resume.txt (RESUME_PATH) directly, bypassing
resume_router -- the module every other stage (tailor, cover_letter) already
uses. That file had drifted into a fabricated placeholder ("Senior Software
Engineer", "Seattle, WA", unsupported backend/AI skills) that contradicted
the truthful profile-derived system prompt and contaminated every score's
reasoning (e.g. citing "the candidate is based in Seattle, WA" for a
Mebane, NC candidate).

See docs/audit-2026-08-27.md sections 1, 4.1, 18, 19 (fix #1) for the full
evidence trail.
"""

from __future__ import annotations

import applypilot.scoring.scorer as scorer_mod


def test_scorer_module_does_not_reference_resume_path():
    """The fabricated ~/.applypilot/resume.txt must not be reachable as an
    independent scoring source of truth at all -- not merely unused."""
    assert not hasattr(scorer_mod, "RESUME_PATH")


def test_run_scoring_resume_text_is_profile_derived_not_fabricated(tmp_db, seed_job, monkeypatch):
    """run_scoring's resume_text must come from the candidate's real
    profile, and must never contain the fabricated placeholder's content
    even if a stale/fabricated ~/.applypilot/resume.txt still exists on
    disk (this test doesn't even create one -- the point is the code path
    to read it no longer exists in run_scoring at all)."""
    conn = tmp_db()
    seed_job(conn, full_description="x", state="enriched", fit_score=None)

    captured: dict = {}

    def fake_score_job(resume_text, job, profile=None):
        captured["resume_text"] = resume_text
        captured["profile"] = profile
        return {"score": 5, "keywords": "", "reasoning": ""}

    fake_profile = {
        "education": [{"official_degree": "Bachelor of Arts in Media Studies"}],
        "experience_inventory": [{"name": "Warehouse Associate", "role_title": "Packaging Associate"}],
    }
    monkeypatch.setattr(scorer_mod, "load_profile", lambda: fake_profile)
    monkeypatch.setattr(scorer_mod, "score_job", fake_score_job)

    scorer_mod.run_scoring(limit=1)

    resume_text = captured["resume_text"]
    assert "Senior Software Engineer" not in resume_text
    assert "Seattle" not in resume_text
    assert "Packaging Associate" in resume_text
    # profile is threaded into score_job so it isn't re-read from disk per job.
    assert captured["profile"] is fake_profile


def test_run_scoring_uses_real_canonical_profile_truthfully(tmp_db, seed_job, monkeypatch):
    """End-to-end (still no LLM call) against the real, checked-in
    data/profile.json: the rendered resume reference must reflect the
    candidate's actual truthful facts (Mebane NC, Media Studies degree)
    and must not contain any trace of the old fabricated identity."""
    conn = tmp_db()
    seed_job(conn, full_description="x", state="enriched", fit_score=None)

    captured: dict = {}

    def fake_score_job(resume_text, job, profile=None):
        captured["resume_text"] = resume_text
        return {"score": 5, "keywords": "", "reasoning": ""}

    monkeypatch.setattr(scorer_mod, "score_job", fake_score_job)

    scorer_mod.run_scoring(limit=1)

    resume_text = captured["resume_text"]
    assert "Media Studies" in resume_text
    assert "Senior Software Engineer" not in resume_text
    assert "Seattle" not in resume_text


def test_score_job_prompt_carries_whatever_resume_text_it_is_given():
    """score_job must pass resume_text through to the LLM prompt verbatim
    (not re-derive it from RESUME_PATH internally) -- proves the fix lives
    at the call site (run_scoring), not just incidentally, and that a
    truthful resume_text actually reaches the model with no fabricated
    content re-injected along the way."""
    captured: dict = {}

    class _FakeClient:
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            return "ELIGIBILITY: eligible\nSCORE: 5\nKEYWORDS: none\nREASONING: ok"

    scorer_mod_local = scorer_mod
    orig_get_stage_client = scorer_mod_local.get_stage_client
    try:
        scorer_mod_local.get_stage_client = lambda *a, **k: _FakeClient()
        fake_profile = {
            "education": [],
            "experience_inventory": [],
            "application_profile": {"location": {"current_city": "Mebane", "current_state": "NC"}},
        }
        job = {
            "title": "IT Support Technician",
            "site": "Acme",
            "location": "Remote",
            "full_description": "Support role.",
        }
        resume_text = "CANONICAL PROFILE REFERENCE\neducation: Bachelor of Arts in Media Studies"

        scorer_mod_local.score_job(resume_text, job, profile=fake_profile)

        user_msg = next(m["content"] for m in captured["messages"] if m["role"] == "user")
        assert resume_text in user_msg
        assert "Senior Software Engineer" not in user_msg
        assert "Seattle" not in user_msg
    finally:
        scorer_mod_local.get_stage_client = orig_get_stage_client


def test_render_profile_reference_is_public_and_used_by_scorer():
    """resume_router.render_profile_reference must be importable (public,
    not the old module-private _render_profile_reference) since scorer.py
    now depends on it directly."""
    from applypilot.scoring.resume_router import render_profile_reference

    text = render_profile_reference({"education": [{"official_degree": "Bachelor of Arts"}]})
    assert "Bachelor of Arts" in text
