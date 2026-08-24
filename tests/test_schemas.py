"""Tests for the cognitive-linguistic schema layer (scoring/schemas.py).

2026-08-23: introduced to reduce unconstrained LLM generation in resume
tailoring and cover-letter writing by computing one deterministic, LLM-free
structured representation per job (requirements -> ranked evidence ->
assigned cognitive/bullet schema -> exact-keyword anchors) and sharing it
between tailor.py and cover_letter.py.

Covers: cognitive schema selection, resume schema selection, schema
composition, job-keyword preservation, evidence-to-schema mapping,
unsupported-claim prevention, resume/cover-letter realization (guidance
reaches the actual outgoing LLM prompt), caching/reuse, and local-vs-cloud
routing (this whole module must never call any LLM client).
"""

import unittest
from unittest.mock import MagicMock, patch

from applypilot.scoring import schemas


def _job(description: str, url: str = "https://example.com/job1") -> dict:
    return {"url": url, "title": "Support Technician", "full_description": description}


TROUBLESHOOTING_PROFILE = {
    "experience_inventory": [
        {
            "name": "Auto Shop Diagnostic Tech",
            "relevance_categories": ["troubleshooting", "root cause", "diagnostics"],
            "resume_allowed": True,
            "description": "Diagnosed and repaired vehicle electrical faults.",
        },
    ],
    "project_inventory": [],
    "skills_inventory": [
        {
            "name": "Python",
            "relevance_categories": ["automation", "technical"],
            "resume_allowed": True,
            "evidence_level": "working",
        },
        {
            "name": "Customer Service",
            "relevance_categories": ["customer service"],
            "resume_allowed": True,
            "evidence_level": "expert",
        },
    ],
    "certifications": [],
}

JOB_DESCRIPTION = (
    "- Troubleshoot hardware and software issues and identify root cause\n"
    "- Provide excellent customer service to clients\n"
    "- Experience with Python automation is a plus\n"
    "- Competitive salary and health benefits\n"
    "- Familiarity with quantum cryptography\n"  # deliberately unsupported
)


def setUp_clear_cache():
    schemas.clear_schema_cache()


# ---------------------------------------------------------------------------
# 1. Cognitive schema selection
# ---------------------------------------------------------------------------


class TestCognitiveSchemaSelection(unittest.TestCase):
    def test_troubleshooting_requirement_gets_problem_diagnosis_schema(self):
        result = schemas.select_schema_for_requirement(
            "Troubleshoot hardware and software issues and identify root cause",
            {"type": "experience", "name": "Auto Shop Diagnostic Tech"},
            match_kind="literal",
        )
        self.assertEqual(result["cognitive_schema"], "problem_diagnosis_repair_verification")

    def test_generic_experience_requirement_gets_action_object_outcome(self):
        result = schemas.select_schema_for_requirement(
            "Build and maintain internal tooling",
            {"type": "experience", "name": "Some Job"},
            match_kind="literal",
        )
        self.assertEqual(result["cognitive_schema"], "action_object_outcome")

    def test_skill_requirement_gets_evidence_claim(self):
        result = schemas.select_schema_for_requirement(
            "Experience with Python automation is a plus",
            {"type": "skill", "name": "Python"},
            match_kind="literal",
        )
        self.assertEqual(result["cognitive_schema"], "evidence_claim")

    def test_non_literal_match_always_gets_domain_transfer_regardless_of_content(self):
        """A synonym/paraphrase match must never get a direct-match schema,
        even when the requirement text itself contains troubleshooting
        language -- the MATCH is transferable, not the requirement's topic."""
        result = schemas.select_schema_for_requirement(
            "Troubleshoot hardware and software issues",
            {"type": "experience", "name": "Auto Shop Diagnostic Tech"},
            match_kind="synonym",
        )
        self.assertEqual(result["cognitive_schema"], "domain_transfer")
        self.assertEqual(result["bullet_schema"], "domain_transfer_bullet")


# ---------------------------------------------------------------------------
# 2. Resume schema selection (bullet + summary)
# ---------------------------------------------------------------------------


class TestResumeSchemaSelection(unittest.TestCase):
    def test_summary_schema_prefers_identity_strength_when_two_plus_literal_matches(self):
        requirements = [
            {"supported": True, "schema": {"cognitive_schema": "action_object_outcome"}},
            {"supported": True, "schema": {"cognitive_schema": "evidence_claim"}},
        ]
        self.assertEqual(
            schemas.select_summary_schema(requirements),
            "identity_domain_strength_evidence_value",
        )

    def test_summary_schema_falls_back_to_transfer_when_support_is_thin(self):
        requirements = [
            {"supported": True, "schema": {"cognitive_schema": "domain_transfer"}},
            {"supported": False, "schema": None},
        ]
        self.assertEqual(
            schemas.select_summary_schema(requirements),
            "capability_background_transfer_value",
        )

    def test_all_bullet_and_summary_schemas_have_named_slots(self):
        """A schema with no slots is useless as a rhetorical scaffold --
        catches an accidental empty entry in the library."""
        for name, schema in {**schemas.BULLET_SCHEMAS, **schemas.SUMMARY_SCHEMAS}.items():
            self.assertTrue(schema.get("slots"), f"{name} has no slots")


# ---------------------------------------------------------------------------
# 3. Schema composition (build_job_schema_representation)
# ---------------------------------------------------------------------------


class TestSchemaComposition(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_composition_produces_one_entry_per_requirement_line(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        # 4 candidate requirement lines survive the benefits filter (the
        # "Competitive salary and health benefits" line is dropped entirely
        # by local_tailor.py's benefit-line detection, reused here).
        self.assertEqual(len(rep["requirements"]), 4)
        self.assertTrue(all("benefit" not in r["requirement"].lower() for r in rep["requirements"]))

    def test_supported_requirement_gets_resume_evidence_and_schema(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        troubleshooting = next(r for r in rep["requirements"] if "troubleshoot" in r["requirement"].lower())
        self.assertTrue(troubleshooting["supported"])
        self.assertIn("Auto Shop Diagnostic Tech", troubleshooting["resume_evidence"])
        self.assertIsNotNone(troubleshooting["schema"])

    def test_representation_has_a_summary_schema(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        self.assertIn(rep["summary_schema"], schemas.SUMMARY_SCHEMAS)


# ---------------------------------------------------------------------------
# 4. Job-keyword preservation (exact vs synonym/conceptual)
# ---------------------------------------------------------------------------


class TestKeywordPreservation(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_literal_match_records_exact_keywords_not_synonym_concepts(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        python_req = next(r for r in rep["requirements"] if "python" in r["requirement"].lower())
        self.assertIn("python", python_req["exact_keywords"])
        self.assertEqual(python_req["synonym_concepts"], [])

    def test_synonym_match_records_synonym_concepts_not_exact_keywords(self):
        """'Accept and respond to telephone inquiries...' matches the
        curated 'customer service' synonym pattern in local_tailor.py, not a
        literal 'customer service' phrase -- must be flagged as transferable,
        not as an exact-keyword hit.

        A second bullet containing "customer service" literally is required
        so rank_profile_evidence's job-level literal scan includes the
        "Customer Service" skill in ranked_evidence at all -- synonym
        matching only re-scores evidence ALREADY found relevant to the job
        as a whole against one specific requirement line; it never invents
        job-level relevance on its own (see local_tailor.py's module
        comment above _CONCEPT_SYNONYM_PATTERNS)."""
        job = _job(
            "- Accept and respond to telephone inquiries from customers in a polite manner\n"
            "- Prior customer service experience needed\n"
        )
        profile = {
            "experience_inventory": [],
            "project_inventory": [],
            "skills_inventory": [
                {
                    "name": "Customer Service",
                    "relevance_categories": ["customer service"],
                    "resume_allowed": True,
                    "evidence_level": "expert",
                },
            ],
            "certifications": [],
        }
        rep = schemas.build_job_schema_representation(job, profile)
        self.assertEqual(len(rep["requirements"]), 2)
        req = rep["requirements"][0]  # the telephone-inquiries (synonym-only) line
        self.assertIn("telephone", req["requirement"].lower())
        self.assertTrue(req["supported"])
        self.assertEqual(req["exact_keywords"], [])
        self.assertIn("Customer Service", req["synonym_concepts"])
        self.assertEqual(req["schema"]["cognitive_schema"], "domain_transfer")

    def test_format_schema_guidance_surfaces_exact_keywords_verbatim(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        guidance = schemas.format_schema_guidance(rep)
        self.assertIn("root cause", guidance)
        self.assertIn("use these exact terms", guidance)


# ---------------------------------------------------------------------------
# 5. Evidence-to-schema mapping
# ---------------------------------------------------------------------------


class TestEvidenceToSchemaMapping(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_experience_evidence_maps_to_experience_appropriate_schema(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        troubleshooting = next(r for r in rep["requirements"] if "troubleshoot" in r["requirement"].lower())
        self.assertEqual(troubleshooting["schema"]["bullet_schema"], "problem_action_result")

    def test_skill_evidence_maps_to_evidence_claim(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        python_req = next(r for r in rep["requirements"] if "python" in r["requirement"].lower())
        self.assertEqual(python_req["schema"]["cognitive_schema"], "evidence_claim")


# ---------------------------------------------------------------------------
# 6. Unsupported-claim prevention
# ---------------------------------------------------------------------------


class TestUnsupportedClaimPrevention(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_unsupported_requirement_gets_no_schema(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        quantum_req = next(r for r in rep["requirements"] if "quantum" in r["requirement"].lower())
        self.assertFalse(quantum_req["supported"])
        self.assertIsNone(quantum_req["schema"])
        self.assertEqual(quantum_req["resume_evidence"], [])

    def test_unsupported_requirement_omitted_from_rendered_guidance(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        guidance = schemas.format_schema_guidance(rep)
        self.assertNotIn("quantum", guidance.lower())

    def test_ambiguous_requirement_gets_no_schema_either(self):
        """Two evidence items tied for top score on the same requirement --
        genuinely ambiguous, resolved neither way by this deterministic
        layer (that's local_tailor.py's LLM-assisted job) -- must not be
        assigned a schema just to have something to show."""
        job = _job("- Python experience required\n")
        profile = {
            "experience_inventory": [],
            "project_inventory": [],
            "skills_inventory": [
                {"name": "Python", "relevance_categories": ["python"], "resume_allowed": True},
                {"name": "Python Fundamentals", "relevance_categories": ["python"], "resume_allowed": True},
            ],
            "certifications": [],
        }
        rep = schemas.build_job_schema_representation(job, profile)
        self.assertEqual(len(rep["requirements"]), 1)
        req = rep["requirements"][0]
        self.assertFalse(req["supported"])
        self.assertTrue(req["ambiguous"])
        self.assertIsNone(req["schema"])

    def test_empty_guidance_when_nothing_is_supported(self):
        job = _job("- Familiarity with quantum cryptography\n")
        profile = {"experience_inventory": [], "project_inventory": [], "skills_inventory": [], "certifications": []}
        rep = schemas.build_job_schema_representation(job, profile)
        self.assertEqual(schemas.format_schema_guidance(rep), "")


# ---------------------------------------------------------------------------
# 7 & 8. Resume / cover-letter realization -- the guidance actually reaches
#         the outgoing LLM prompt for both generators.
# ---------------------------------------------------------------------------


class TestResumeRealization(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_tailor_resume_includes_schema_guidance_in_the_user_message(self):
        from applypilot.scoring import tailor

        job = _job(JOB_DESCRIPTION)
        mock_client = MagicMock()
        mock_client.chat.return_value = (
            '{"title":"Tech","summary":"S","skills":{"Languages":"Python"},'
            '"experience":[],"projects":[],"education":[]}'
        )
        mock_client.has_cloud_available = lambda: True

        with (
            patch.object(tailor, "get_stage_client", return_value=mock_client),
            patch.object(tailor, "is_local_configured", return_value=False),
            patch.object(tailor, "extract_facts_from_resume_json", return_value=set()),
            patch.object(tailor, "is_auto_approvable", return_value=True),
        ):
            tailor.tailor_resume("Original resume.", job, TROUBLESHOOTING_PROFILE, max_retries=0)

        self.assertTrue(mock_client.chat.called)
        sent_messages = mock_client.chat.call_args[0][0]
        user_content = next(m["content"] for m in sent_messages if m["role"] == "user")
        self.assertIn("JOB SCHEMA GUIDANCE", user_content)
        self.assertIn("root cause", user_content)


class TestCoverLetterRealization(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_generate_cover_letter_includes_schema_guidance_in_the_user_message(self):
        from applypilot.scoring import cover_letter

        job = _job(JOB_DESCRIPTION)
        job["title"] = "Support Technician"
        mock_client = MagicMock()
        # A long-enough plausible letter so validation doesn't force retries
        # that would complicate call-arg inspection.
        mock_client.chat.return_value = "Dear Hiring Manager,\n\n" + ("Solid engineering work. " * 40) + "\n\nName"

        with (
            patch.object(cover_letter, "get_client", return_value=mock_client),
            patch.object(
                cover_letter, "validate_cover_letter", return_value={"passed": True, "errors": [], "warnings": []}
            ),
        ):
            cover_letter.generate_cover_letter("Resume text.", job, TROUBLESHOOTING_PROFILE, max_retries=0)

        self.assertTrue(mock_client.chat.called)
        sent_messages = mock_client.chat.call_args[0][0]
        user_content = next(m["content"] for m in sent_messages if m["role"] == "user")
        self.assertIn("JOB SCHEMA GUIDANCE", user_content)
        self.assertIn("root cause", user_content)


# ---------------------------------------------------------------------------
# 9. Caching / reuse across tailor and cover-letter generation
# ---------------------------------------------------------------------------


class TestCaching(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_second_call_for_same_job_returns_cached_object(self):
        job = _job(JOB_DESCRIPTION)
        first = schemas.get_or_build_job_schema(job, TROUBLESHOOTING_PROFILE)
        second = schemas.get_or_build_job_schema(job, TROUBLESHOOTING_PROFILE)
        self.assertIs(first, second)

    def test_build_is_only_invoked_once_for_two_callers_of_the_same_job(self):
        """Simulates tailor.py then cover_letter.py hitting the same job in
        one pipeline run -- the expensive-ish build must run once."""
        job = _job(JOB_DESCRIPTION)
        with patch.object(
            schemas,
            "build_job_schema_representation",
            wraps=schemas.build_job_schema_representation,
        ) as wrapped:
            schemas.get_or_build_job_schema(job, TROUBLESHOOTING_PROFILE)  # tailor.py's call
            schemas.get_or_build_job_schema(job, TROUBLESHOOTING_PROFILE)  # cover_letter.py's call
        self.assertEqual(wrapped.call_count, 1)

    def test_different_description_for_same_url_is_not_served_stale(self):
        """Guards against a re-enriched job (same URL, updated description)
        silently reusing an old representation."""
        job_v1 = _job("- Python experience required\n")
        job_v2 = _job("- Java experience required\n")  # same URL, new description
        rep1 = schemas.get_or_build_job_schema(job_v1, TROUBLESHOOTING_PROFILE)
        rep2 = schemas.get_or_build_job_schema(job_v2, TROUBLESHOOTING_PROFILE)
        self.assertNotEqual(
            [r["requirement"] for r in rep1["requirements"]],
            [r["requirement"] for r in rep2["requirements"]],
        )


# ---------------------------------------------------------------------------
# 10. Local-vs-cloud routing: this module must never call any LLM client
# ---------------------------------------------------------------------------


class TestNoLLMCallsInSchemaComputation(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_build_job_schema_representation_never_touches_the_llm_module(self):
        job = _job(JOB_DESCRIPTION)
        with (
            patch("applypilot.llm.get_client") as mock_get_client,
            patch("applypilot.llm.get_stage_client") as mock_get_stage_client,
        ):
            schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        mock_get_client.assert_not_called()
        mock_get_stage_client.assert_not_called()

    def test_get_or_build_job_schema_never_touches_the_llm_module(self):
        job = _job(JOB_DESCRIPTION)
        with (
            patch("applypilot.llm.get_client") as mock_get_client,
            patch("applypilot.llm.get_stage_client") as mock_get_stage_client,
        ):
            schemas.get_or_build_job_schema(job, TROUBLESHOOTING_PROFILE)
        mock_get_client.assert_not_called()
        mock_get_stage_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
