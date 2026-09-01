"""Tests for the profile-authority hardening pass (2026-08-25): profile.json
is the single source of truth for job titles, seniority, and skills. Covers
the concrete known-bad regression cases named in the task:

- "Sr Architect - Emerging Technologies..." for a candidate with no
  professional architecture experience (check_title_inflation).
- "High-Volume Operations Specialist" invented for UPS instead of the
  authoritative "Packaging Associate / Warehouse Associate"
  (validate_factual_anchors' per-employer title check).
- A technical resume including stand-up comedy merely because it exists in
  the profile (classify_standup_relevance's hard-exclude title list,
  covered in test_standup_relevance.py -- not duplicated here).
"""

import json

import pytest

from applypilot.scoring.tailor import (
    _build_canonical_inventory_block,
    classify_seniority_mismatch,
    tailor_resume,
)
from applypilot.scoring.validator import (
    check_bullet_vocabulary_injection,
    check_date_fabrication,
    check_date_placeholder_fabrication,
    check_title_inflation,
    check_unsupported_skill_claims,
    check_unsupported_technical_skills,
    validate_cover_letter,
    validate_factual_anchors,
    validate_json_fields,
)


def _profile():
    """A profile shaped like the real data/profile.json's relevant parts --
    enough authoritative title/skill data for every check in this file to
    have real ground truth to compare against."""
    return {
        "resume_facts": {
            "preserved_companies": ["UPS", "National Tire and Battery / Mavis", "AMP Smart"],
            "preserved_school": "University of North Carolina at Greensboro",
            "real_metrics": [],
        },
        "experience_inventory": [
            {
                "name": "AMP Smart",
                "role_title": "Sales Representative / Sales Tech",
                "resume_allowed": True,
                "relevance_categories": ["sales"],
            },
            {
                "name": "UPS",
                "role_title": "Packaging Associate / Warehouse Associate",
                "resume_allowed": True,
                "relevance_categories": ["operations"],
            },
            {
                "name": "National Tire and Battery / Mavis",
                "role_title": "Alignment Technician",
                "resume_allowed": True,
                "relevance_categories": ["technical support"],
            },
        ],
        "historical_experience_inventory": [],
        "skills_inventory": [
            {
                "name": "Python",
                "evidence_level": "demonstrated_project_use",
                "proficiency": "learning_developing",
                "resume_allowed": True,
            },
            {
                "name": "APIs",
                "evidence_level": "learning_or_exposure",
                "proficiency": "learning",
                "resume_allowed": False,
            },
            {
                "name": "Docker",
                "evidence_level": "learning_or_exposure",
                "proficiency": "learning",
                "resume_allowed": False,
            },
        ],
        "skills_boundary": {
            "demonstrated_projects": ["Python"],
            "learning_or_exposure_not_expertise": ["APIs", "Docker"],
        },
        "project_inventory": [
            {
                "name": "You Power You",
                "resume_allowed": True,
                "relevance_categories": ["web"],
                "factual_concepts": ["Website development", "HTML/CSS/JavaScript"],
            },
        ],
        "qualifications": [
            {
                "name": "Stand-up comedy",
                "resume_allowed": True,
                "relevance_categories": ["communications", "public speaking"],
                "evidence": ["Public speaking", "Verbal communication", "Audience engagement"],
            },
        ],
    }


# ---------------------------------------------------------------------------
# check_title_inflation -- the top-level resume headline check
# ---------------------------------------------------------------------------


class TestTitleInflation:
    def test_known_bad_sr_architect_is_rejected(self):
        """The exact known-bad regression example from the task."""
        err = check_title_inflation("Sr Architect - Emerging Technologies", _profile())
        assert err
        assert "architect" in err.lower()

    def test_bare_authoritative_title_passes(self):
        assert check_title_inflation("Packaging Associate / Warehouse Associate", _profile()) == ""

    def test_modest_descriptive_title_passes(self):
        assert check_title_inflation("Technical Support & Customer Service", _profile()) == ""

    def test_engineer_title_rejected_when_unsupported(self):
        err = check_title_inflation("Software Engineer", _profile())
        assert err
        assert "engineer" in err.lower()

    def test_no_authoritative_data_never_flags(self):
        """No experience_inventory anywhere -- nothing to ground a
        judgment on, so this stays silent (matches validate_factual_
        anchors' identical early-exit)."""
        assert check_title_inflation("Senior Engineer", {"experience_inventory": []}) == ""

    def test_candidates_own_history_supporting_the_word_is_not_flagged(self):
        profile = _profile()
        profile["experience_inventory"].append(
            {"name": "Acme", "role_title": "Senior Engineer", "resume_allowed": True}
        )
        assert check_title_inflation("Senior Engineer", profile) == ""


# ---------------------------------------------------------------------------
# validate_factual_anchors -- per-employer title + header-format handling
# ---------------------------------------------------------------------------


class TestFactualAnchorsTitleCheck:
    def test_known_bad_ups_title_upgrade_is_rejected(self):
        """The exact known-bad regression example: UPS becoming
        "High-Volume Operations Specialist" instead of its authoritative
        title. Uses the real "Title at Company" header shape (no pipe) --
        the actual JSON contract tailor.py's prompt asks the LLM for."""
        data = {"experience": [{"header": "High-Volume Operations Specialist at United Parcel Service (UPS)"}]}
        result = validate_factual_anchors(data, _profile())
        assert result["errors"]
        assert "ups" in result["errors"][0].lower() or "united parcel" in result["errors"][0].lower()

    def test_authoritative_ups_title_passes(self):
        data = {"experience": [{"header": "Packaging Associate / Warehouse Associate at United Parcel Service (UPS)"}]}
        result = validate_factual_anchors(data, _profile())
        assert result["errors"] == []

    def test_subset_of_authoritative_title_passes(self):
        """A truthful SUBSET of the authoritative title ("Warehouse
        Associate" out of "Packaging Associate / Warehouse Associate")
        must not be flagged -- only an ADDED elevated word is."""
        data = {"experience": [{"header": "Warehouse Associate at United Parcel Service (UPS)"}]}
        result = validate_factual_anchors(data, _profile())
        assert result["errors"] == []

    def test_mavis_technical_support_specialist_is_rejected(self):
        """Second named regression example: Mavis (Alignment Technician)
        must not become a generic "Technical Support Specialist"."""
        data = {"experience": [{"header": "Technical Support Specialist at National Tire and Battery / Mavis"}]}
        result = validate_factual_anchors(data, _profile())
        assert result["errors"]

    def test_amp_smart_consultant_title_is_rejected(self):
        """Third named regression example: AMP Smart must not become an
        engineering/consulting position."""
        data = {"experience": [{"header": "Solutions Consultant at AMP Smart"}]}
        result = validate_factual_anchors(data, _profile())
        assert result["errors"]

    def test_pipe_separated_header_format_still_works(self):
        """Backward compatibility with the older pipe-separated shape
        (still what local_tailor.py's degraded-mode base-resume parser
        produces)."""
        data = {"experience": [{"header": "High-Volume Operations Specialist | UPS | 2024-2025"}]}
        result = validate_factual_anchors(data, _profile())
        assert result["errors"]

    def test_unrecognized_employer_still_warns_not_errors(self):
        data = {"experience": [{"header": "Warehouse Associate at Totally Fake Inc"}]}
        result = validate_factual_anchors(data, _profile())
        assert result["errors"] == []
        assert any("unrecognized employer" in w for w in result["warnings"])

    def test_no_profile_data_never_flags(self):
        data = {"experience": [{"header": "Anything at Anywhere"}]}
        result = validate_factual_anchors(data, {"resume_facts": {}})
        assert result == {"errors": [], "warnings": []}


class TestUnparseableHeaderFabrication:
    """2026-09-01: the prompt bake-off's worst single fabrication
    (IterateCV inventing a standalone "ALIGNMENT TECHNIQUES" experience
    entry with no employer name at all) evaded every check that only fires
    once a header splits into (title, company) -- it never reached that
    branch. Any unparseable header is now checked against every known
    entry name/authoritative title in the profile (whole-STRING
    containment, not word-level overlap -- an earlier version pooled
    individual words from every title into one set, which let "ALIGNMENT
    TECHNIQUES" false-pass by coincidentally sharing the single word
    "alignment" with the real "Alignment Technician" title; caught by this
    exact test failing before the fix), including qualifications (some
    real entries, like "Stand-up comedy", live only there)."""

    def test_fabricated_standalone_entry_is_flagged(self):
        data = {"experience": [{"header": "ALIGNMENT TECHNIQUES", "bullets": ["Built RESTful services."]}]}
        result = validate_factual_anchors(data, _profile())
        assert result["errors"] == []
        assert any("could not be matched" in w for w in result["warnings"])
        assert any("ALIGNMENT TECHNIQUES" in w for w in result["warnings"])

    def test_legitimate_bare_qualification_header_is_not_flagged(self):
        """A bare header using the qualification's OWN name text is
        recognized -- must not be false-flagged just because it doesn't
        split into (title, company)."""
        data = {"experience": [{"header": "Stand-up comedy", "bullets": ["Delivered live performances."]}]}
        result = validate_factual_anchors(data, _profile())
        assert result["warnings"] == []

    def test_derived_agent_noun_form_is_a_known_limitation(self):
        """KNOWN LIMITATION, documented rather than silently accepted: a
        stylistic rendering of a qualification as a person/role noun
        ("Stand-up Comedian", from "Stand-up comedy") does not match via
        string containment, and gets the same (low-cost, warning-tier,
        non-blocking) treatment as a genuine fabrication would. Confirmed
        during development that no simple string-similarity heuristic can
        reliably tell this case apart from the real "ALIGNMENT TECHNIQUES"
        fabrication above -- by character overlap, "Alignment Technician"
        vs "ALIGNMENT TECHNIQUES" is MORE similar than "Stand-up comedy"
        vs "Stand-up Comedian", so any threshold that accepts the second
        would also accept the first. A warning here is an acceptable,
        low-cost false positive (a human reviewer dismisses it on sight),
        traded off against reliably catching the real fabrication."""
        data = {"experience": [{"header": "Stand-up Comedian", "bullets": ["Delivered live performances."]}]}
        result = validate_factual_anchors(data, _profile())
        assert any("could not be matched" in w for w in result["warnings"])

    def test_legitimate_bare_role_title_is_not_flagged(self):
        """A bare authoritative role title with no company suffix, used
        verbatim, must not be flagged."""
        data = {"experience": [{"header": "Alignment Technician", "bullets": ["Vehicle alignment."]}]}
        result = validate_factual_anchors(data, _profile())
        assert result["warnings"] == []

    def test_no_profile_data_never_flags_unparseable_header(self):
        data = {"experience": [{"header": "Something Made Up"}]}
        result = validate_factual_anchors(data, {"resume_facts": {}})
        assert result == {"errors": [], "warnings": []}


class TestBulletVocabularyInjection:
    """2026-09-01: the prompt bake-off found a systematic pattern
    (IterateCV especially) of grafting distinctive JOB-POSTING vocabulary
    onto a bullet whose entry is otherwise correctly attributed -- e.g.
    "utilizing data to inform comedic timing" on a Stand-up Comedian
    bullet. The injected word ("data") is often already legitimate
    SOMEWHERE in the candidate's profile (a real analytics project), so
    this must be checked per-entry, not whole-profile. Uses a "Title at
    Company" header (not the bare "Stand-up Comedian" form) so the test
    exercises this mechanism directly, without also depending on the
    separate comedian/comedy string-matching limitation documented in
    TestUnparseableHeaderFabrication."""

    def _job(self):
        return {
            "title": "Data Engineer",
            "full_description": (
                "We need a data engineer to build analytics pipelines and "
                "dashboards, working with data at scale."
            ),
        }

    def test_injected_job_vocabulary_on_unrelated_entry_is_flagged(self):
        data = {
            "experience": [
                {
                    "header": "Alignment Technician at National Tire and Battery / Mavis",
                    "bullets": [
                        "Performed vehicle alignment work, then helped build analytics "
                        "pipelines and dashboards as part of a data engineer role."
                    ],
                }
            ]
        }
        warnings = check_bullet_vocabulary_injection(data, _profile(), job=self._job())
        assert warnings
        assert "Alignment Technician" in warnings[0]

    def test_word_grounded_in_its_own_entry_is_not_flagged(self):
        """The SAME job-vocabulary word ("analytics"/"pipelines") applied
        to an entry whose OWN evidence actually supports it must not be
        flagged -- this is legitimate, truthful tailoring, not injection."""
        profile = _profile()
        profile["project_inventory"].append(
            {
                "name": "CAP Predictor",
                "resume_allowed": True,
                "factual_concepts": ["data analytics", "pipelines", "feature engineering"],
            }
        )
        data = {
            "projects": [
                {
                    "header": "CAP Predictor",
                    "bullets": ["Built data analytics pipelines for feature engineering."],
                }
            ]
        }
        warnings = check_bullet_vocabulary_injection(data, profile, job=self._job())
        assert warnings == []

    def test_no_job_provided_returns_empty(self):
        data = {"experience": [{"header": "Stand-up Comedian", "bullets": ["Utilizing data to inform comedic timing."]}]}
        assert check_bullet_vocabulary_injection(data, _profile(), job=None) == []

    def test_validate_json_fields_surfaces_injection_as_warning_only(self):
        """Deliberately WARNING-tier, not ERROR -- natural-language
        vocabulary overlap is fuzzier than the structural checks."""
        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Languages": "Python"},
            "experience": [
                {
                    "header": "Alignment Technician at National Tire and Battery / Mavis",
                    "subtitle": "Technical Support",
                    "bullets": [
                        "Performed vehicle alignment work, then helped build analytics "
                        "pipelines and dashboards as part of a data engineer role."
                    ],
                }
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, _profile(), job=self._job())
        assert any("graft job-posting language" in w for w in result["warnings"])
        assert not any("graft job-posting language" in e for e in result["errors"])


class TestSkillsCategoryJoinBoundary:
    """2026-09-01: found by the large-sample statistical fabrication-
    detection test, not by any hand-picked example. validate_json_fields
    used to join the SKILLS dict's category values with a bare space
    (" ".join(...)) before checking them -- when neither adjacent value
    ended in a delimiter, two DIFFERENT categories could merge into one
    claim (e.g. {"Other": "Outreach", "Fabricated": "Salesforce"} ->
    "outreach salesforce"), which then falsely substring-matched the real
    "outreach" skill and silently exempted the fabricated content riding
    along with it. Every hand-picked test case up to this point happened
    to keep multi-item claims within a single, comma-separated category
    value, so this never surfaced until a genuinely large, varied sample
    was run. Fixed by joining with ", " instead of " "."""

    def test_fabrication_in_a_separate_category_is_not_hidden_by_an_adjacent_real_skill(self):
        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Real": "Outreach", "Fabricated": "Salesforce"},
            "experience": [
                {"header": "Packaging Associate / Warehouse Associate at UPS", "bullets": ["Package handling."]}
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, _profile())
        assert not result["passed"]
        assert any("salesforce" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------------
# check_unsupported_technical_skills -- skills-section inflation
# ---------------------------------------------------------------------------


class TestUnsupportedTechnicalSkills:
    def test_disallowed_skill_inventory_item_is_flagged(self):
        violations = check_unsupported_technical_skills("Python, APIs, Docker", _profile())
        assert "APIs" in violations
        assert "Docker" in violations

    def test_allowed_skill_is_not_flagged(self):
        violations = check_unsupported_technical_skills("Python", _profile())
        assert violations == []

    def test_project_only_technology_promoted_to_skills_is_flagged(self):
        """The exact known-bad regression: "Python, JavaScript, HTML, CSS"
        under Languages -- JS/HTML/CSS only ever appear as You Power You's
        project-scoped factual_concepts, never as a vetted skill."""
        violations = check_unsupported_technical_skills("Python, JavaScript, HTML, CSS", _profile())
        assert "javascript" in violations
        assert "html" in violations
        assert "css" in violations
        assert "python" not in violations

    def test_validate_json_fields_rejects_unsupported_skills(self):
        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Languages": "Python, JavaScript, HTML, CSS"},
            "experience": [
                {"header": "Packaging Associate / Warehouse Associate at UPS", "bullets": ["Package handling."]}
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, _profile())
        assert not result["passed"]
        assert any("javascript" in e.lower() for e in result["errors"])


class TestFabricationWatchlistUngroundedTechnologies:
    """2026-08-25 adversarial-review follow-up: check_unsupported_technical_
    skills only catches two SPECIFIC known patterns (explicitly-disallowed
    skills_inventory items, and project-only factual_concepts wrongly
    promoted) -- it does NOT catch a technology with ZERO grounding
    anywhere in the profile at all. Confirmed as a REAL historical
    fabrication via an actual pre-fix generated resume: "Kubernetes
    (Exposure)" appeared on a real tailored resume, traceable to the OLD
    tailoring prompt's own "you MAY add 2-3 closely related tools
    (Kubernetes if Docker...)" instruction (since removed) and the OLD
    FABRICATION_WATCHLIST's explicit "reasonable stretches... are ALLOWED"
    design (since reversed). This is now caught via FABRICATION_WATCHLIST,
    which was expanded with common cloud/infra/framework terms verified
    absent from the real profile.json."""

    def test_kubernetes_with_zero_grounding_is_rejected(self):
        from applypilot.scoring.validator import FABRICATION_WATCHLIST

        assert "kubernetes" in FABRICATION_WATCHLIST

        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Tools & Infrastructure": "Docker (Exposure), Kubernetes (Exposure)"},
            "experience": [
                {"header": "Packaging Associate / Warehouse Associate at UPS", "bullets": ["Package handling."]}
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, _profile())
        assert not result["passed"]
        assert any("kubernetes" in e.lower() for e in result["errors"])

    def test_terraform_with_zero_grounding_is_rejected(self):
        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Tools & Infrastructure": "Terraform"},
            "experience": [
                {"header": "Packaging Associate / Warehouse Associate at UPS", "bullets": ["Package handling."]}
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, _profile())
        assert not result["passed"]
        assert any("terraform" in e.lower() for e in result["errors"])

    def test_watchlist_still_yields_to_a_genuinely_supported_skill(self):
        """The watchlist's existing cross-reference-against-profile skip
        logic (validate_json_fields: "skip if this 'fabrication' is
        actually a real skill in the profile") must still work after the
        expansion -- a watchlisted term the profile actually grounds is
        never flagged as a fabricated skill."""
        profile = _profile()
        profile["skills_boundary"]["cloud"] = ["Kubernetes"]
        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Tools & Infrastructure": "Kubernetes"},
            "experience": [
                {"header": "Packaging Associate / Warehouse Associate at UPS", "bullets": ["Package handling."]}
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, profile)
        assert not any("fabricated skill: 'kubernetes'" in e.lower() for e in result["errors"])


class TestUnsupportedSkillClaimsGeneralized:
    """2026-09-01: FABRICATION_WATCHLIST and check_unsupported_technical_
    skills both only catch fabrications someone already observed and
    hand-enumerated -- confirmed as a real, live gap by the specialized-
    model prompt bake-off, which found models inventing skills (e.g.
    "Databricks") not on the watchlist at all. check_unsupported_skill_
    claims is a POSITIVE check (must trace to the profile) instead of a
    denylist, so it generalizes to any not-yet-seen fabrication."""

    def test_novel_fabrication_not_on_watchlist_is_caught(self):
        """"Databricks" is deliberately NOT in FABRICATION_WATCHLIST --
        this is exactly the class of gap the watchlist-only approach
        can't close."""
        from applypilot.scoring.validator import FABRICATION_WATCHLIST

        assert "databricks" not in FABRICATION_WATCHLIST
        violations = check_unsupported_skill_claims("Python, Databricks", _profile())
        assert any("databricks" in v.lower() for v in violations)
        assert not any("python" in v.lower() for v in violations)

    def test_validate_json_fields_rejects_novel_fabrication(self):
        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Tools": "Python, Databricks"},
            "experience": [
                {"header": "Packaging Associate / Warehouse Associate at UPS", "bullets": ["Package handling."]}
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, _profile())
        assert not result["passed"]
        assert any("databricks" in e.lower() for e in result["errors"])

    def test_proficiency_qualifiers_are_not_flagged(self):
        """"(Exposure)"-style qualifiers must not be treated as skill
        claims in their own right -- this is generic English grammar, not
        a technology, and must never need a per-candidate update."""
        violations = check_unsupported_skill_claims("Python (Exposure), Docker (Basic Familiarity)", _profile())
        assert violations == []

    def test_category_labels_are_not_flagged(self):
        violations = check_unsupported_skill_claims("Languages: Python; Tools: Docker", _profile())
        assert violations == []

    def test_no_profile_skill_data_stays_silent(self):
        """Matches check_title_inflation/validate_factual_anchors' same
        early-exit -- nothing to ground a judgment on, so this doesn't
        flag everything."""
        assert check_unsupported_skill_claims("Python, Databricks", {}) == []

    def test_empty_skills_text_returns_empty(self):
        assert check_unsupported_skill_claims("", _profile()) == []


# ---------------------------------------------------------------------------
# check_date_placeholder_fabrication -- subtitle date-placeholder fabrication
# ---------------------------------------------------------------------------


class TestDatePlaceholderFabrication:
    """2026-08-28: `subtitle` ("Tech | Dates" per entry) was never validated
    at all. Live measurement against all 354 real generated resumes found
    the exact prompt-schema example ("Tech | Dates") never leaks verbatim,
    but the LLM regularly invents its OWN placeholder text when it lacks
    real dates -- affecting 271/354 (76%) of real output. These are the 5
    real observed variants plus a negative control."""

    @pytest.mark.parametrize(
        "subtitle",
        [
            "Python, SQL | Dates not specified",
            "Python, SQL | Dates N/A",
            "Python, SQL | Dates Not Specified",
            "Python, SQL | Dates Unspecified",
            "Python, SQL | Dates Not Provided",
            "Python, SQL | dates not provided",  # case-insensitive
        ],
    )
    def test_known_bad_variants_are_rejected(self, subtitle):
        data = {"experience": [{"header": "Role at Company", "subtitle": subtitle}]}
        errors = check_date_placeholder_fabrication(data)
        assert errors
        assert "Role at Company" in errors[0]

    def test_real_dates_pass(self):
        data = {"experience": [{"header": "Role at Company", "subtitle": "Python, SQL | 2021 - 2023"}]}
        assert check_date_placeholder_fabrication(data) == []

    def test_omitted_dates_pass(self):
        """The prompt now instructs omitting the date portion entirely
        when genuinely unknown -- this must never be flagged, only
        INVENTED filler text should be."""
        data = {"experience": [{"header": "Role at Company", "subtitle": "Python, SQL"}]}
        assert check_date_placeholder_fabrication(data) == []

    def test_projects_section_also_checked(self):
        data = {"projects": [{"header": "Side Project", "subtitle": "React | Dates not specified"}]}
        errors = check_date_placeholder_fabrication(data)
        assert errors
        assert "Side Project" in errors[0]

    def test_missing_subtitle_key_does_not_crash(self):
        data = {"experience": [{"header": "Role at Company"}]}
        assert check_date_placeholder_fabrication(data) == []

    def test_non_list_sections_do_not_crash(self):
        assert check_date_placeholder_fabrication({}) == []
        assert check_date_placeholder_fabrication({"experience": None, "projects": "not a list"}) == []


class TestValidateJsonFieldsRejectsDatePlaceholder:
    def test_validate_json_fields_rejects_fabricated_date_placeholder(self):
        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Languages": "Python"},
            "experience": [
                {
                    "header": "Packaging Associate / Warehouse Associate at UPS",
                    "subtitle": "Logistics | Dates not specified",
                    "bullets": ["Package handling."],
                }
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, _profile())
        assert not result["passed"]
        assert any("fabricated date placeholder" in e.lower() for e in result["errors"])

    def test_validate_json_fields_accepts_real_dates(self):
        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Languages": "Python"},
            "experience": [
                {
                    "header": "Packaging Associate / Warehouse Associate at UPS",
                    "subtitle": "Logistics | 2019 - 2021",
                    "bullets": ["Package handling."],
                }
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, _profile())
        assert not any("fabricated date placeholder" in e.lower() for e in result["errors"])


def _profile_with_dates():
    """_profile() extended with real start_date/end_date on the UPS entry
    -- _profile() itself deliberately has none (so the existing placeholder
    tests above can't accidentally trip the new date-fabrication check),
    matching the shape data/profile.json actually has."""
    profile = _profile()
    profile["experience_inventory"][1]["start_date"] = "2024-05"
    profile["experience_inventory"][1]["end_date"] = "2025-04"
    return profile


class TestDateFabricationAgainstAuthoritativeRange:
    """2026-09-01: check_date_placeholder_fabrication only catches an
    invented PLACEHOLDER ("Dates not specified") -- it has no way to catch
    a model that instead invents a plausible-looking specific date range,
    which is more dangerous (more falsifiable) and was confirmed to slip
    past every existing check during the specialized-model prompt bake-off
    (qwen3 invented "2019-2021" for a job the profile only ever shows as
    "Dates not specified" -- nothing flagged it). check_date_fabrication
    compares against the profile's own recorded start_date/end_date
    instead of a string pattern."""

    def test_out_of_range_year_is_rejected(self):
        data = {
            "experience": [
                {"header": "Packaging Associate / Warehouse Associate at UPS", "subtitle": "Logistics | 2019 - 2021"}
            ]
        }
        errors = check_date_fabrication(data, _profile_with_dates())
        assert errors
        assert "UPS" in errors[0] or "Packaging" in errors[0]

    def test_correct_year_passes(self):
        data = {
            "experience": [
                {"header": "Packaging Associate / Warehouse Associate at UPS", "subtitle": "Logistics | 2024 - 2025"}
            ]
        }
        assert check_date_fabrication(data, _profile_with_dates()) == []

    def test_employer_with_no_dates_on_file_stays_silent(self):
        """_profile() (no dates recorded anywhere) must never flag --
        there's no ground truth to contradict."""
        data = {
            "experience": [
                {"header": "Packaging Associate / Warehouse Associate at UPS", "subtitle": "Logistics | 2019 - 2021"}
            ]
        }
        assert check_date_fabrication(data, _profile()) == []

    def test_no_year_mentioned_passes(self):
        data = {"experience": [{"header": "Packaging Associate / Warehouse Associate at UPS", "subtitle": "Logistics"}]}
        assert check_date_fabrication(data, _profile_with_dates()) == []

    def test_validate_json_fields_rejects_invented_date_range(self):
        data = {
            "title": "Technical Support & Customer Service",
            "summary": "Support and troubleshooting background.",
            "skills": {"Languages": "Python"},
            "experience": [
                {
                    "header": "Packaging Associate / Warehouse Associate at UPS",
                    "subtitle": "Logistics | 2019 - 2021",
                    "bullets": ["Package handling."],
                }
            ],
            "education": "University of North Carolina at Greensboro",
        }
        result = validate_json_fields(data, _profile_with_dates())
        assert not result["passed"]
        assert any("fabricated date(s)" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------------
# classify_seniority_mismatch -- pre-tailor eligibility gate
# ---------------------------------------------------------------------------


class TestSeniorityMismatch:
    def _job(self, title, desc="Some job description."):
        return {"title": title, "full_description": desc}

    def test_sr_architect_title_is_mismatched(self):
        assert classify_seniority_mismatch(self._job("Sr Architect - Emerging Technologies"), _profile()) is True

    def test_staff_software_engineer_is_mismatched(self):
        assert classify_seniority_mismatch(self._job("Staff Software Engineer, Observability"), _profile()) is True

    def test_principal_engineer_is_mismatched(self):
        assert classify_seniority_mismatch(self._job("Principal Engineer"), _profile()) is True

    def test_engineering_manager_is_mismatched(self):
        assert classify_seniority_mismatch(self._job("Engineering Manager"), _profile()) is True

    def test_member_of_technical_staff_is_mismatched(self):
        """The exact real Axon regression title. Now caught via the
        canonical applypilot.eligibility.seniority_disqualifier predicate's
        bare "staff" match (see the consolidation note below) -- no
        special-casing needed for "Member of Staff" specifically."""
        assert classify_seniority_mismatch(self._job("Sr. Full Stack Member of Technical Staff"), _profile()) is True

    def test_ordinary_entry_level_title_not_mismatched(self):
        assert classify_seniority_mismatch(self._job("IT Support Technician"), _profile()) is False

    def test_director_of_engineering_is_mismatched(self):
        assert classify_seniority_mismatch(self._job("Director of Engineering"), _profile()) is True

    def test_cto_is_mismatched(self):
        assert classify_seniority_mismatch(self._job("CTO"), _profile()) is True

    # 2026-08-25 adversarial-review follow-up (round 1): this function used
    # to have its own separately-maintained title regex, which originally
    # matched bare "director"/"chief"/"vp" anywhere in the title -- wrongly
    # flagging "Funeral Director"/"Camp Director"/"Program Director"/"Chief
    # of Staff"/"Store Director" as seniority mismatches -- and was narrowed
    # to require an engineering/technical qualifier.
    #
    # 2026-08-25 follow-up (round 2, consolidation): that separate regex was
    # then found to be exactly the "independently drifted copy" anti-pattern
    # applypilot.eligibility.py was built to prevent (a 2026-08-21 incident
    # already documents THREE such drifted copies; this was becoming a
    # fourth). Consolidated to delegate to the canonical
    # eligibility.seniority_disqualifier predicate -- the SAME one scorer.py
    # (pre-LLM scoring) and apply/launcher.py (apply-time re-check) already
    # use. That canonical predicate is DELIBERATELY broad by explicit,
    # documented design (scorer.py: "a Senior/Staff/.../Director/Manager/
    # Head/VP/Chief/... title is always disqualifying regardless of keyword
    # overlap elsewhere") -- so these bare director/chief titles are
    # correctly "mismatched" again here too, for consistency with the rest
    # of the pipeline. This is a deliberate architectural tradeoff (one
    # canonical, consistent predicate over a locally "smarter" but
    # independently-drifting one), not a regression -- the same bare-word
    # breadth already applies at scoring time today, unrelated to this
    # function's own history.
    def test_funeral_director_is_mismatched_via_canonical_predicate(self):
        assert classify_seniority_mismatch(self._job("Funeral Director"), _profile()) is True

    def test_chief_of_staff_is_mismatched_via_canonical_predicate(self):
        assert classify_seniority_mismatch(self._job("Chief of Staff"), _profile()) is True

    # 2026-08-25 adversarial-review follow-up: the first version of
    # _SENIOR_YEARS_RE's trailing-context alternation included the bare word
    # "technical", which matched ordinary phrasing like "8+ years of
    # technical support experience" -- wrongly flagging exactly the kind of
    # IT-support role this candidate's own profile evidence (Mavis, CompTIA
    # A+) actually supports. Confirmed as a real false positive, removed.
    def test_years_of_technical_support_not_mismatched(self):
        j = self._job(
            "IT Support Specialist",
            "8+ years of technical support experience required.",
        )
        assert classify_seniority_mismatch(j, _profile()) is False

    def test_years_of_customer_facing_technical_not_mismatched(self):
        j = self._job(
            "Support Engineer",
            "10+ years of experience in customer-facing technical roles.",
        )
        assert classify_seniority_mismatch(j, _profile()) is False

    def test_sales_engineer_not_mismatched_by_bare_engineer_word(self):
        """ "Engineer" alone (not senior/staff/principal/director/architect/
        lead+engineer) isn't a seniority-tier signal on its own -- avoids
        over-triggering on every entry-level "Engineer I" posting."""
        assert classify_seniority_mismatch(self._job("Support Engineer"), _profile()) is False

    def test_years_of_experience_signal_without_senior_title(self):
        j = self._job(
            "Software Developer",
            "15+ years of experience in software engineering required.",
        )
        assert classify_seniority_mismatch(j, _profile()) is True

    def test_candidates_own_history_prevents_false_positive(self):
        """Profile-driven, not hardcoded: a candidate whose OWN profile
        contains a matching senior title is never flagged."""
        profile = _profile()
        profile["experience_inventory"].append(
            {"name": "Acme", "role_title": "Senior Software Architect", "resume_allowed": True}
        )
        assert classify_seniority_mismatch(self._job("Senior Software Architect"), profile) is False

    def test_no_seniority_signal_in_job_never_flags(self):
        assert classify_seniority_mismatch(self._job("Cashier"), _profile()) is False


class _StubClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return json.dumps(self.payload)


class TestSeniorityMismatchIntegration:
    def test_tailor_resume_skips_llm_call_on_mismatch(self, monkeypatch):
        from applypilot.scoring import tailor as t

        stub = _StubClient({})
        monkeypatch.setattr(t, "get_stage_client", lambda stage, quality=False: stub)
        job = {
            "title": "Sr Architect - Emerging Technologies",
            "full_description": "Own the technical architecture vision.",
            "site": "example",
            "location": "Remote",
        }
        tailored, report = tailor_resume("ORIGINAL RESUME", job, _profile(), max_retries=2)

        assert report["status"] == "seniority_mismatch"
        assert tailored == ""
        assert stub.calls == []  # no LLM call was made at all


class TestEndToEndRejectsExactKnownBadContent:
    """Full tailor_resume() integration -- not the isolated validator
    functions -- with a mocked LLM that persistently returns EXACTLY the
    combined known-bad content from the regression report on every retry
    attempt: an inflated top-level title, unsupported JS/HTML/CSS skills, an
    invented UPS title, and stand-up comedy on a technical-role resume. The
    job itself is deliberately NOT seniority-mismatched (a legitimate,
    eligible job) so the seniority gate doesn't short-circuit before the LLM
    call -- this specifically exercises the validate_json_fields path
    inside the real retry loop, proving the fix works end-to-end and not
    merely in each validator function's own isolated unit tests."""

    def _bad_payload(self):
        return {
            "title": "Sr Architect - Emerging Technologies (Fraud Detection and Governance)",
            "summary": "Experienced architect with a passion for stand-up comedy and audience engagement.",
            "skills": {"Languages": "Python, JavaScript, HTML, CSS"},
            "experience": [
                {
                    "header": "High-Volume Operations Specialist at United Parcel Service (UPS)",
                    "bullets": ["Performed stand-up comedy routines to entertain coworkers during breaks."],
                }
            ],
            "projects": [],
            "education": "University of North Carolina at Greensboro",
        }

    def test_persistently_bad_llm_output_never_reaches_approved(self, monkeypatch):
        from applypilot.scoring import tailor as t

        stub = _StubClient(self._bad_payload())
        monkeypatch.setattr(t, "get_stage_client", lambda stage, quality=False: stub)
        monkeypatch.setattr(t, "is_local_configured", lambda: False)
        job = {
            "title": "IT Support Technician",  # eligible job -- not seniority-mismatched
            "full_description": "Provide desktop and end-user technical support.",
            "site": "example",
            "location": "Remote",
        }
        _tailored, report = tailor_resume("ORIGINAL RESUME", job, _profile(), max_retries=1)

        # The LLM WAS called (this job is eligible) -- proves the gate
        # didn't just get lucky by short-circuiting.
        assert len(stub.calls) >= 1
        # But it must never be approved, no matter how many retries are
        # exhausted, since every attempt returns the same bad content.
        assert report["status"] != "approved"
        assert report["status"] == "failed_validation"

        validator = report["validator"]
        assert not validator["passed"]
        all_errors = " ".join(validator["errors"]).lower()
        assert "javascript" in all_errors
        assert "html" in all_errors
        assert "css" in all_errors
        assert "invented/upgraded job title" in all_errors  # UPS header
        assert "stand-up" in all_errors.lower() or "stand up" in all_errors.lower()

    def test_the_avoid_notes_fed_back_name_the_actual_violations(self, monkeypatch):
        """The retry loop feeds validation errors back to the LLM as
        'avoid this' notes -- confirms the SPECIFIC violations (not a vague
        generic message) are what gets surfaced, so a real LLM would have
        an actual chance to self-correct on the next attempt."""
        from applypilot.scoring import tailor as t

        stub = _StubClient(self._bad_payload())
        monkeypatch.setattr(t, "get_stage_client", lambda stage, quality=False: stub)
        monkeypatch.setattr(t, "is_local_configured", lambda: False)
        job = {
            "title": "IT Support Technician",
            "full_description": "Provide desktop and end-user technical support.",
            "site": "example",
            "location": "Remote",
        }
        _tailored, _report = tailor_resume("ORIGINAL RESUME", job, _profile(), max_retries=1)
        assert len(stub.calls) == 2  # both attempts made (max_retries=1 -> 2 total)
        second_attempt_system_prompt = stub.calls[1][0]["content"]
        assert "AVOID THESE ISSUES" in second_attempt_system_prompt
        assert "javascript" in second_attempt_system_prompt.lower()


# ---------------------------------------------------------------------------
# _build_canonical_inventory_block -- role_title/role_type now rendered
# ---------------------------------------------------------------------------

_SENT = (
    "At the company I built a tool used by fifty engineering teams, "
    "moving billions of operations across regions every single day with measured uptime. "
)


def _para(n: int) -> str:
    return (_SENT * n).strip()


def _good_letter(body: str = "") -> str:
    return (
        "Dear Hiring Manager,\n\n"
        + _para(4)
        + "\n\n"
        + _para(6)
        + "\n\n"
        + _para(4)
        + (f" {body}" if body else "")
        + "\n\n"
        + _para(2)
        + "\n\n"
        + "Philip"
    )


class TestCoverLetterProfileIntegrity:
    def test_applypilot_leak_is_rejected_when_profile_given(self):
        profile = {"resume_facts": {"private_projects": ["ApplyPilot"]}}
        letter = _good_letter("Built with ApplyPilot, an internal automation tool.")
        result = validate_cover_letter(letter, profile)
        assert not result["passed"]
        assert any("ApplyPilot" in e for e in result["errors"])

    def test_no_profile_arg_still_works_unchanged(self):
        """Backward compatibility -- existing call sites that don't pass a
        profile at all must keep working exactly as before."""
        letter = _good_letter("Built with ApplyPilot, an internal automation tool.")
        result = validate_cover_letter(letter)
        assert "ApplyPilot" not in " ".join(result["errors"])  # no profile, no check

    def test_clean_letter_passes_with_profile(self):
        profile = {"resume_facts": {"private_projects": ["ApplyPilot"]}}
        letter = _good_letter()
        result = validate_cover_letter(letter, profile)
        assert result["passed"]


class TestCanonicalInventoryAuthoritativeTitle:
    def test_role_title_is_rendered(self):
        block = _build_canonical_inventory_block(_profile())
        assert "AUTHORITATIVE TITLE: Packaging Associate / Warehouse Associate" in block

    def test_disallowed_skills_get_explicit_do_not_list_line(self):
        block = _build_canonical_inventory_block(_profile())
        assert "DO NOT LIST AS SKILLS" in block
        assert "APIs" in block
        assert "Docker" in block
