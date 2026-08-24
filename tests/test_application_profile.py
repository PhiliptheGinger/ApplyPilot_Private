import re

from applypilot import config
from applypilot.apply import prompt as apply_prompt
from applypilot.scoring.tailor import assemble_resume_text


def test_application_profile_loaded():
    profile = config.load_profile()
    assert "application_profile" in profile
    ap = profile["application_profile"]
    assert isinstance(ap.get("work_authorization"), dict)
    assert ap["work_authorization"]["authorized_to_work_us"] is True
    assert ap["work_authorization"]["requires_sponsorship"] is False


def test_application_profile_not_leaked_into_resume():
    # Minimal fake LLM output that assemble_resume_text expects
    data = {
        "title": "Test Engineer",
        "summary": "Test summary.",
        "skills": {"Languages": "Python"},
        "experience": [{"header": "Test Role at Company", "subtitle": "Tech | 2020 - 2021", "bullets": ["Did work"]}],
        "projects": [],
        "education": [],
    }
    profile = config.load_profile()
    txt = assemble_resume_text(data, profile)
    # Application-profile phrases must not appear verbatim in the resume
    forbidden = ["Work Auth", "Sponsorship Needed", "Willing to relocate", "preferred_commute_minutes"]
    for f in forbidden:
        assert f not in txt


def test_autofill_prompt_uses_application_profile():
    profile = config.load_profile()
    summary = apply_prompt._build_profile_summary(profile)
    # Ensure the autofill/profile summary includes the explicit work auth lines
    assert "Work Auth:" in summary
    assert "Sponsorship Needed:" in summary
    # Spanish CEFR should be present as B1-B2 in the summary
    assert re.search(r"Spanish.*B1-B2|B1-B2.*Spanish", summary, re.IGNORECASE)
