import json

from applypilot.scoring.tailor import (
    STANDUP_EXCLUDE,
    STANDUP_INCLUDE,
    STANDUP_OPTIONAL,
    _build_tailor_prompt,
    classify_standup_relevance,
    tailor_resume,
)
from applypilot.scoring.validator import validate_json_fields, validate_tailored_resume


def _job(title: str, desc: str) -> dict:
    return {
        "url": "https://example.com/job/1",
        "title": title,
        "site": "example",
        "location": "Seattle, WA",
        "full_description": desc,
    }


def test_classifier_output_strings_are_exact():
    out = classify_standup_relevance(_job("Help Desk Technician", "Customer-facing support and troubleshooting"))
    assert out in {STANDUP_INCLUDE, STANDUP_OPTIONAL, STANDUP_EXCLUDE}


def test_software_engineer_exclude():
    j = _job("Software Engineer", "Design and implement backend services, microservices, CI/CD.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_backend_developer_exclude():
    j = _job("Backend Developer", "Build APIs, optimize database queries, maintain infrastructure.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_systems_administrator_exclude():
    j = _job("Systems Administrator", "Manage Linux servers, patching, backups, and IAM.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_network_engineer_exclude():
    j = _job("Network Engineer", "Design network architecture, routing, switching, and firewalls.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_cybersecurity_engineer_exclude():
    j = _job("Cybersecurity Engineer", "Threat detection, incident response, SIEM tuning.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_help_desk_exclude():
    """2026-08-25: explicit user correction overrides the prior design --
    help desk/IT-support-shaped titles are a HARD exclude regardless of how
    much customer-facing/communication language the description uses (this
    fixture has plenty and must still exclude). See
    classify_standup_relevance's docstring and _TECHNICAL_EXCLUDE_TITLE_RE."""
    j = _job(
        "Help Desk Technician",
        "End-user support, customer-facing troubleshooting, explain technical issues to non-technical users.",
    )
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_technical_support_engineer_exclude():
    """Same hard-exclude override as test_help_desk_exclude -- "technical
    support" is in the user's enumerated technical-role exclude list."""
    j = _job(
        "Technical Support Engineer",
        "Customer-facing troubleshooting, lead customer communications, explain complex technical concepts to users.",
    )
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_sales_engineer_include():
    j = _job(
        "Sales Engineer",
        "Lead product demonstrations and presentations, partner with account management and sales teams.",
    )
    assert classify_standup_relevance(j) == STANDUP_INCLUDE


def test_recruiter_include():
    j = _job("Recruiter", "Run interviews, candidate outreach, stakeholder communication.")
    assert classify_standup_relevance(j) == STANDUP_INCLUDE


def test_technical_trainer_include():
    j = _job("Technical Trainer", "Deliver training workshops and presentations for customer teams.")
    assert classify_standup_relevance(j) == STANDUP_INCLUDE


def test_desktop_engineer_generic_comm_optional_or_exclude():
    j = _job(
        "Desktop Engineer",
        "Maintain endpoint infrastructure, software packaging, and asset lifecycle. Excellent communication skills.",
    )
    out = classify_standup_relevance(j)
    assert out in {STANDUP_OPTIONAL, STANDUP_EXCLUDE}


def test_desktop_engineer_substantial_communication_still_excludes():
    """2026-08-25: "Desktop Engineer" is in the hard-exclude title list
    (desktop support/desktop engineer), so even substantial communication
    signals in the description no longer override it -- unlike the softer
    technical_ic_title penalty other titles get, this one is unconditional."""
    j = _job(
        "Desktop Engineer",
        "Primary point of contact for end-user escalations, stakeholder communication, and customer-facing support.",
    )
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_generic_communication_phrase_alone_not_include():
    j = _job(
        "Backend Developer",
        "Build distributed services and optimize infrastructure. Excellent communication skills required.",
    )
    out = classify_standup_relevance(j)
    assert out in {STANDUP_OPTIONAL, STANDUP_EXCLUDE}
    assert out != STANDUP_INCLUDE


def test_communications_media_marketing_include():
    j = _job(
        "Communications Manager",
        "Lead media outreach, audience-facing communication, content strategy, and presentations.",
    )
    assert classify_standup_relevance(j) == STANDUP_INCLUDE


def test_software_architect_hard_exclude():
    j = _job(
        "Software Architect",
        "Own the technical vision. Explain complex technical concepts to non-technical stakeholders. Public speaking at conferences.",
    )
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_it_support_specialist_hard_exclude():
    j = _job("IT Support Specialist", "Customer-facing troubleshooting and end-user support.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_data_engineer_hard_exclude():
    j = _job("Data Engineer", "Build data pipelines. Excellent communication skills required.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_ml_engineer_hard_exclude():
    j = _job("ML Engineer", "Train and deploy models. Present findings to stakeholders.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_infrastructure_engineer_hard_exclude():
    j = _job("Infrastructure Engineer", "Own cloud infrastructure. Customer-facing support required.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


# 2026-08-25 adversarial-review follow-up: these are the exact real job
# titles/descriptions (pulled from the live database) that produced actual
# historical bad output -- a resume with "Public Speaker & Performer at
# Stand-up Comedy" as a fabricated employer entry. The original hard-
# exclude list only matched "software architect"/"solutions architect",
# never bare "architect", and had no coverage for "full stack"/"member of
# staff"-shaped titles at all -- so with the REAL (non-empty, lengthy) job
# description, these fell through to the soft scoring system and still
# returned INCLUDE. Confirmed broken, then fixed; regression-tested here
# with the real description text, not an empty stand-in that would trivially
# EXCLUDE via the separate "no description" rule regardless of title
# coverage.
def test_bare_architect_title_hard_exclude():
    j = _job(
        "Sr Architect - Emerging Technologies (Fraud Detection and Governance)",
        "Own the architectural vision for fraud detection systems. Explain complex "
        "technical concepts to non-technical stakeholders. Excellent communication skills.",
    )
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_ai_architect_title_hard_exclude():
    j = _job(
        "Sr AI Architect - Conversational AI",
        "Lead the architectural vision. Explain complex technical concepts to "
        "stakeholders. Present findings to executives. Strong communication skills.",
    )
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_full_stack_member_of_technical_staff_hard_exclude():
    j = _job(
        "Sr. Full Stack Member of Technical Staff",
        "Collaborate cross-functionally with product and design. Explain technical "
        "tradeoffs to non-technical stakeholders. Strong written and verbal "
        "communication skills required. Present architecture decisions to the team.",
    )
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_naval_architect_is_a_minor_accepted_over_inclusion():
    """Documents a known, deliberately accepted tradeoff: bare "architect"
    also catches the rare non-software case (naval/landscape architect).
    Not a meaningful risk for this pipeline's actual job mix, and not
    harmful even if it occurred -- see _TECHNICAL_EXCLUDE_TITLE_RE's
    comment."""
    j = _job("Naval Architect", "Design ship hulls and structures.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_no_meaningful_communication_signals_exclude():
    j = _job("Platform Engineer", "Maintain Kubernetes clusters and automate deployments.")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_missing_description_exclude():
    j = _job("Anything", "")
    assert classify_standup_relevance(j) == STANDUP_EXCLUDE


def test_prompt_includes_explicit_include_instruction():
    p = _build_tailor_prompt({}, standup_decision=STANDUP_INCLUDE)
    assert "STAND-UP EXPERIENCE DECISION: INCLUDE" in p


def test_prompt_includes_explicit_optional_instruction():
    p = _build_tailor_prompt({}, standup_decision=STANDUP_OPTIONAL)
    assert "STAND-UP EXPERIENCE DECISION: OPTIONAL" in p


def test_prompt_includes_explicit_exclude_instruction():
    p = _build_tailor_prompt({}, standup_decision=STANDUP_EXCLUDE)
    assert "STAND-UP EXPERIENCE DECISION: EXCLUDE" in p


class _StubClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return json.dumps(self.payload)


def test_tailor_resume_passes_job_description_and_profile_source(monkeypatch):
    from applypilot.scoring import tailor as t

    profile = {
        "personal": {"full_name": "Jordan Lee", "email": "j@example.com"},
        "skills_boundary": {"languages": ["Python"]},
        "resume_facts": {
            "preserved_companies": ["Acme Corp"],
            "preserved_projects": ["Project X"],
            "preserved_school": "State University",
            "real_metrics": ["50% faster"],
        },
        "experience": {"education_level": "Bachelor's Degree"},
    }
    job = _job("Help Desk Technician", "Customer-facing support and explain technical issues to users.")

    payload = {
        "title": "Help Desk Technician",
        "summary": "Support specialist with user-facing troubleshooting experience.",
        "skills": {"Languages": "Python"},
        "experience": [
            {
                "header": "Support Engineer at Acme Corp",
                "subtitle": "Python | 2020-2024",
                "bullets": ["Resolved incidents"],
            }
        ],
        "projects": [{"header": "Project X", "subtitle": "Python | 2023", "bullets": ["Built tooling"]}],
        "education": [{"institution": "State University", "degree": "BS", "dates": "2016 - 2020"}],
    }
    stub = _StubClient(payload)
    monkeypatch.setattr(t, "get_stage_client", lambda stage, quality=False: stub)

    tailored, report = tailor_resume("ORIGINAL RESUME", job, profile, max_retries=0)

    # "Help Desk Technician" is a hard-exclude title (2026-08-25) -- this
    # test's actual purpose is verifying prompt/profile wiring, not standup
    # classification, so the assertion just tracks the correct
    # classification for this fixture rather than picking a different job.
    assert report["standup_decision"] == STANDUP_EXCLUDE
    system_prompt = stub.calls[0][0]["content"]
    user_prompt = stub.calls[0][1]["content"]

    assert "STAND-UP EXPERIENCE DECISION: EXCLUDE" in system_prompt
    assert "Preserved companies: Acme Corp" in system_prompt
    assert "TARGET JOB:" in user_prompt
    assert "DESCRIPTION:" in user_prompt
    assert "Customer-facing support" in user_prompt
    assert "ORIGINAL RESUME:" in user_prompt
    assert "SUMMARY" in tailored


def test_validator_blocks_standup_when_exclude():
    profile = {"skills_boundary": {"languages": ["Python"]}, "resume_facts": {}}
    data = {
        "title": "Software Engineer",
        "summary": "Backend engineer.",
        "skills": {"Languages": "Python"},
        "experience": [
            {
                "header": "Engineer at Acme",
                "subtitle": "Python | 2022-2024",
                "bullets": ["Performed stand-up comedy in local venues."],
            }
        ],
        "projects": [],
        "education": "State University",
    }
    out = validate_json_fields(data, profile, standup_decision=STANDUP_EXCLUDE)
    assert not out["passed"]
    assert any("Stand-up content present" in e for e in out["errors"])


def test_text_validator_blocks_standup_when_exclude():
    profile = {"skills_boundary": {"languages": ["Python"]}, "resume_facts": {}}
    text = """SUMMARY
Backend engineer.

EXPERIENCE
- Did stand-up comedy weekly.

TECHNICAL SKILLS
Python

PROJECTS
Project X

EDUCATION
State University
"""
    out = validate_tailored_resume(text, profile, standup_decision=STANDUP_EXCLUDE)
    assert not out["passed"]
    assert any("Stand-up content present" in e for e in out["errors"])
