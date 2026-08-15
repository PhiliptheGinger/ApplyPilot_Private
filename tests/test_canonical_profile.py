from applypilot.scoring import resume_router
from applypilot.scoring.tailor import _build_canonical_inventory_block, _build_tailor_prompt


def _profile():
    return {
        "education": [{
            "institution": "University of North Carolina at Greensboro",
            "official_degree": "Bachelor of Arts in Media Studies",
            "field_of_study": "Media Studies",
            "start_year": 2019,
            "end_year": 2022,
        }],
        "experience_inventory": [{"name": "AMP Smart", "responsibilities": ["In-person consultations"]}],
        "project_inventory": [
            {"name": "Standup-OCR", "status": "completed_or_active", "resume_allowed": True},
            {"name": "ApplyPilot", "private": True, "resume_allowed": False},
            {"name": "Sunburn", "status": "unfinished_experimental", "resume_allowed": True},
        ],
        "skills_inventory": [
            {"name": "Python", "evidence_level": "demonstrated_project_use", "proficiency": "learning_developing", "resume_allowed": True},
            {"name": "Docker", "evidence_level": "learning_or_exposure", "proficiency": "learning", "resume_allowed": False},
        ],
        "skills_boundary": {"demonstrated_projects": ["Python"], "learning_or_exposure_not_expertise": ["Docker"]},
    }


def test_canonical_inventory_excludes_private_project():
    block = _build_canonical_inventory_block(_profile())
    assert "Standup-OCR" in block
    assert "ApplyPilot" not in block
    assert "AMP Smart" in block


def test_tailor_prompt_preserves_official_degree_and_skill_evidence():
    prompt = _build_tailor_prompt(_profile(), standup_decision="EXCLUDE")
    assert "Bachelor of Arts in Media Studies" in prompt
    assert "University of North Carolina at Greensboro" in prompt
    assert "LEARNING/EXPOSURE, NOT EXPERTISE" in prompt
    assert "Never include ApplyPilot" in prompt


def test_load_resume_text_uses_canonical_profile_reference(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{}", encoding="utf-8")
    profile = _profile()
    monkeypatch.setattr(resume_router, "PROJECT_PROFILE_PATH", profile_path)
    monkeypatch.setattr(resume_router, "load_profile", lambda: profile)

    text, source = resume_router.load_resume_text_for_job({"title": "Software Engineer", "full_description": "backend"})

    assert source == profile_path
    assert "CANONICAL PROFILE REFERENCE" in text
    assert "ApplyPilot" not in text
    assert "Standup-OCR" in text
