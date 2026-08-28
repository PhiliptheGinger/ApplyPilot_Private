"""Tests for the 2026-08-28 apply-prompt honesty fix in prompt.py.

Investigation found four independent, live-confirmed truthfulness defects
in the apply-agent prompt, all rendering on every real submitted
application:

1. `_build_screening_section` hardcoded "cannot relocate" unconditionally
   -- the real profile (data/profile.json) has willing_to_relocate=True.
2. `exp.get(key, default)` only falls back when a key is ABSENT, not when
   it's present-but-empty -- the real profile has both
   years_of_experience_total and target_role present as "", rendering the
   literal malformed sentence "This candidate is a  with  years
   experience."
3. Work-authorization was read via `legally_authorized_to_work`, a key
   that doesn't exist in the real application_profile.work_authorization
   schema (authorized_to_work_us / requires_sponsorship) -- always
   rendered the useless literal fallback "see profile".
4. A blanket "If it's in the same domain, answer YES" instruction told the
   agent to fabricate professional experience with any unlisted-but-
   adjacent tool -- the exact anti-fabrication violation the tailoring
   prompt already guards against, never closed at the apply stage.

Also removes a hardcoded Seattle/Bellevue/Kirkland/Redmond office-location
preference from build_prompt's WHICH OFFICE section that applied
regardless of the actual candidate's location.

These tests exercise the new helpers directly (fast, isolated) plus the
full rendered _build_screening_section output and a real build_prompt()
integration check for the office-location fix, following
test_prompt_doc_format.py's established fixture conventions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply.prompt import (
    _build_experience_line,
    _build_relocation_line,
    _build_screening_section,
    _build_skills_tools_rule,
    _build_work_auth_line,
)

PROJECT_PROFILE_PATH = Path(__file__).parent.parent / "data" / "profile.json"


def _profile(**overrides):
    base = {
        "personal": {"full_name": "Test Candidate", "city": "Testville"},
        "experience": {},
        "application_profile": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Relocation
# ---------------------------------------------------------------------------


def test_relocation_willing_true_with_preference():
    app_prof = {"location": {"willing_to_relocate": True, "relocation_preference": "North Carolina"}}
    line = _build_relocation_line(app_prof, "Mebane")
    assert "willing to relocate" in line
    assert "North Carolina" in line
    assert "cannot relocate" not in line


def test_relocation_willing_true_no_preference():
    app_prof = {"location": {"willing_to_relocate": True}}
    line = _build_relocation_line(app_prof, "Mebane")
    assert "willing to relocate" in line
    assert "cannot relocate" not in line


def test_relocation_willing_false():
    """Positive control: when the profile genuinely says unwilling, the
    line must say so -- this fix must not swing to always claiming
    willing either."""
    app_prof = {"location": {"willing_to_relocate": False}}
    line = _build_relocation_line(app_prof, "Mebane")
    assert "not willing to relocate" in line
    assert line.count("willing to relocate") == 1  # "not willing to relocate", not also a bare "willing to relocate"


def test_relocation_field_absent_uses_neutral_instruction_not_a_guess():
    """The core fix: when the field is genuinely unavailable, never invent
    a relocation constraint in either direction."""
    app_prof = {"location": {}}
    line = _build_relocation_line(app_prof, "Mebane")
    assert "cannot relocate" not in line
    assert "willing to relocate" not in line
    assert "answer truthfully from the profile" in line


# ---------------------------------------------------------------------------
# Experience / target role
# ---------------------------------------------------------------------------


def test_experience_line_empty_strings_do_not_render_malformed_text():
    """The exact live regression: both fields present as empty strings."""
    exp = {"years_of_experience_total": "", "target_role": ""}
    personal = {"current_job_title": ""}
    line = _build_experience_line(exp, personal)
    assert "with  years" not in line
    assert "is a  with" not in line
    assert "candidate is a with years experience" not in line.lower()
    assert "answer" in line.lower()  # falls back to a neutral instruction


def test_experience_line_both_present():
    exp = {"years_of_experience_total": "5", "target_role": "Support Specialist"}
    personal = {}
    line = _build_experience_line(exp, personal)
    assert "Support Specialist" in line
    assert "5" in line


def test_experience_line_only_years():
    exp = {"years_of_experience_total": "3", "target_role": ""}
    personal = {"current_job_title": ""}
    line = _build_experience_line(exp, personal)
    assert "3" in line
    assert "years" in line.lower()


def test_experience_line_only_target_role():
    exp = {"years_of_experience_total": "", "target_role": "IT Support Technician"}
    personal = {}
    line = _build_experience_line(exp, personal)
    assert "IT Support Technician" in line


def test_experience_line_falls_back_to_current_job_title_when_target_role_absent():
    exp = {"years_of_experience_total": "", "target_role": ""}
    personal = {"current_job_title": "Warehouse Associate"}
    line = _build_experience_line(exp, personal)
    assert "Warehouse Associate" in line


# ---------------------------------------------------------------------------
# Work authorization
# ---------------------------------------------------------------------------


def test_work_auth_uses_real_schema_keys():
    work_auth = {"authorized_to_work_us": True, "requires_sponsorship": False}
    line = _build_work_auth_line(work_auth)
    assert "True" in line
    assert "False" in line
    assert "see profile" not in line.lower()


def test_work_auth_legacy_keys_still_produce_a_truthful_line():
    """Legacy field names must still work, not just degrade to the
    neutral fallback -- this is a real answer, not an invented one."""
    work_auth = {"legally_authorized_to_work": "Yes", "require_sponsorship": "No"}
    line = _build_work_auth_line(work_auth)
    assert "Yes" in line
    assert "No" in line


def test_work_auth_real_keys_take_priority_over_legacy_when_both_present():
    work_auth = {
        "authorized_to_work_us": True,
        "requires_sponsorship": False,
        "legally_authorized_to_work": "STALE",
        "require_sponsorship": "STALE",
    }
    line = _build_work_auth_line(work_auth)
    assert "STALE" not in line


def test_work_auth_missing_fields_degrades_to_neutral_not_fabricated():
    """Neither current nor legacy keys present -- must not fail, must not
    invent an answer, must not render the old dead 'see profile' literal
    that gave the agent nothing to work with."""
    line = _build_work_auth_line({})
    assert line == "answer truthfully from the profile"


# ---------------------------------------------------------------------------
# Skills / tools -- no blanket fabrication instruction
# ---------------------------------------------------------------------------


def test_skills_rule_has_no_blanket_answer_yes_instruction():
    profile = _profile(skills_inventory=[{"name": "Python", "resume_allowed": True}])
    rule = _build_skills_tools_rule(profile)
    assert "answer YES" not in rule
    assert "Software engineers learn tools fast" not in rule


def test_skills_rule_lists_resume_allowed_skills_as_professional():
    profile = _profile(
        skills_inventory=[
            {"name": "Python", "resume_allowed": True, "proficiency": "learning_developing"},
            {"name": "APIs", "resume_allowed": False, "proficiency": "learning"},
        ]
    )
    rule = _build_skills_tools_rule(profile)
    assert "Python" in rule
    professional_section = rule.split("Familiarity")[0]
    assert "Python" in professional_section


def test_skills_rule_marks_resume_allowed_false_as_familiarity_only():
    profile = _profile(skills_inventory=[{"name": "Docker", "resume_allowed": False, "proficiency": "learning"}])
    rule = _build_skills_tools_rule(profile)
    assert "Docker" in rule
    assert "not professional experience" in rule.lower()
    # Docker must not appear in a way that claims it as safe professional experience.
    professional_section = rule.split("Familiarity")[0]
    assert "Docker" not in professional_section


def test_skills_rule_still_allows_claiming_legitimate_professional_skills():
    """Must not become so conservative that real, resume_allowed skills
    can no longer be confidently represented."""
    profile = _profile(skills_inventory=[{"name": "Python", "resume_allowed": True}])
    rule = _build_skills_tools_rule(profile)
    assert "Safe to claim professional" in rule
    assert "Python" in rule


def test_skills_rule_handles_empty_inventory_without_crashing():
    profile = _profile(skills_inventory=[])
    rule = _build_skills_tools_rule(profile)
    assert "answer YES" not in rule
    assert isinstance(rule, str) and rule


def test_skills_rule_missing_inventory_key_without_crashing():
    profile = _profile()  # no skills_inventory key at all
    rule = _build_skills_tools_rule(profile)
    assert "answer YES" not in rule


# ---------------------------------------------------------------------------
# Full _build_screening_section output
# ---------------------------------------------------------------------------


def test_full_screening_section_has_no_hardcoded_relocation_denial():
    profile = _profile(
        application_profile={"location": {"willing_to_relocate": True, "relocation_preference": "NC"}},
    )
    section = _build_screening_section(profile)
    assert "cannot relocate" not in section


def test_full_screening_section_has_no_dead_see_profile_fallback_when_data_exists():
    profile = _profile(
        application_profile={"work_authorization": {"authorized_to_work_us": True, "requires_sponsorship": False}},
    )
    section = _build_screening_section(profile)
    assert "Work authorization: see profile" not in section


def test_full_screening_section_has_no_blanket_tool_fabrication_instruction():
    profile = _profile()
    section = _build_screening_section(profile)
    assert "answer YES" not in section
    assert "Software engineers learn tools fast" not in section


# ---------------------------------------------------------------------------
# Regression against the actual live data/profile.json
# ---------------------------------------------------------------------------


def test_regression_against_real_profile_json():
    if not PROJECT_PROFILE_PATH.exists():
        import pytest

        pytest.skip("data/profile.json not present in this environment")

    profile = json.loads(PROJECT_PROFILE_PATH.read_text(encoding="utf-8"))
    section = _build_screening_section(profile)

    # The three concrete, verified-live defects must all be gone.
    assert "cannot relocate" not in section
    assert "Work authorization: see profile" not in section
    assert "candidate is a  with  years" not in section.lower()
    assert "answer YES" not in section

    # And the real profile's actual truthful values must be reflected.
    assert "willing to relocate" in section  # profile: willing_to_relocate=True
    assert "North Carolina" in section  # profile: relocation_preference


# ---------------------------------------------------------------------------
# build_prompt integration: office-location section
# ---------------------------------------------------------------------------

_MINIMAL_PROFILE = {
    "personal": {
        "full_name": "Test User",
        "preferred_name": "Test",
        "email": "test@example.com",
        "password": "hunter2",
        "phone": "555-000-0000",
        "address": "1 Main St",
        "city": "Mebane",
        "province_state": "NC",
        "country": "USA",
        "postal_code": "27302",
    },
    "application_profile": {
        "location": {"willing_to_relocate": True, "relocation_preference": "North Carolina"},
        "work_authorization": {"authorized_to_work_us": True, "requires_sponsorship": False},
    },
    "availability": {
        "earliest_start_date": "Immediately",
    },
    "compensation": {
        "salary_expectation": "50000",
        "salary_currency": "USD",
    },
    "experience": {
        "years_of_experience_total": "",
        "target_role": "",
    },
    "skills_inventory": [{"name": "Python", "resume_allowed": True}],
    "eeo_voluntary": {},
    "skills_boundary": {},
    "resume_facts": {},
    "site_credentials": {},
    "files": {},
}

_MINIMAL_SEARCH = {
    "location": {
        "primary": "Mebane",
        "accept_patterns": ["Mebane", "Remote"],
        "linkedin_type_chars": 3,
    },
    "queries": [{"query": "software engineer", "tier": 1}],
}


def _setup_paths(tmp_path, monkeypatch):
    from applypilot import config

    app_dir = tmp_path / "applypilot_home"
    app_dir.mkdir()
    apply_worker_dir = app_dir / "apply-workers"
    apply_worker_dir.mkdir()

    profile_path = app_dir / "profile.json"
    profile_path.write_text(json.dumps(_MINIMAL_PROFILE), encoding="utf-8")

    search_path = app_dir / "searches.yaml"
    import yaml

    search_path.write_text(yaml.safe_dump(_MINIMAL_SEARCH), encoding="utf-8")

    monkeypatch.setattr(config, "APP_DIR", app_dir)
    monkeypatch.setattr(config, "PROFILE_PATH", profile_path)
    monkeypatch.setattr(config, "SEARCH_CONFIG_PATH", search_path)
    monkeypatch.setattr(config, "APPLY_WORKER_DIR", apply_worker_dir)

    return app_dir


def _make_resume(tmp_path, ext: str) -> Path:
    resume_dir = tmp_path / "tailored"
    resume_dir.mkdir()
    txt = resume_dir / "acme_test_role_abc123.txt"
    txt.write_text("Test User\nTest Role\n", encoding="utf-8")
    doc = txt.with_suffix(f".{ext}")
    doc.write_bytes(b"fake-doc-content")
    return txt


def _build_job(resume_txt: Path) -> dict:
    return {
        "url": "https://example.com/job/1",
        "title": "Test Role",
        "site": "acme",
        "application_url": "https://boards.greenhouse.io/acme/jobs/1",
        "fit_score": 9,
        "tailored_resume_path": str(resume_txt),
        "cover_letter_path": None,
    }


def _mock_db_calls(monkeypatch):
    from applypilot import database
    from applypilot.apply import prompt as prompt_module

    monkeypatch.setattr(prompt_module, "get_all_qa", lambda **_kw: [])
    monkeypatch.setattr(database, "get_accounts_for_prompt", dict)


def test_build_prompt_has_no_hardcoded_seattle_office_preference(tmp_path, monkeypatch):
    _setup_paths(tmp_path, monkeypatch)
    resume_txt = _make_resume(tmp_path, "docx")
    _mock_db_calls(monkeypatch)

    from applypilot.apply.prompt import build_prompt

    job = _build_job(resume_txt)
    result = build_prompt(job, tailored_resume="Resume text", doc_format="docx")

    for hardcoded_city in ("Bellevue", "Kirkland", "Redmond", "Everett", "Bothell", "Renton", "Tacoma"):
        assert hardcoded_city not in result, f"Hardcoded office-location city resurfaced: {hardcoded_city}"


def test_build_prompt_office_location_still_uses_location_accept_priority(tmp_path, monkeypatch):
    """The removed hardcoded fallback must not be replaced with silence --
    the existing profile-driven priority list must still be the source of
    truth for the WHICH OFFICE section."""
    _setup_paths(tmp_path, monkeypatch)
    resume_txt = _make_resume(tmp_path, "docx")
    _mock_db_calls(monkeypatch)

    from applypilot.apply.prompt import build_prompt

    job = _build_job(resume_txt)
    result = build_prompt(job, tailored_resume="Resume text", doc_format="docx")

    assert "WHICH OFFICE" in result
    assert "Selection priority order: Mebane, Remote" in result
