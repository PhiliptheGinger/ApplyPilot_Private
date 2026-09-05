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
        comment above _CONCEPT_SYNONYM_PATTERNS).

        Uses "clients" rather than "customers" in the first line deliberately
        (2026-09-03): _term_in_text is now inflection-tolerant, so "customer"
        (one of the "Customer Service" skill's own matched_terms) would
        otherwise literally match "customers" via plain pluralization,
        turning this into a literal match and defeating the point of this
        test. "clients" has zero word-level relationship to "customer" and
        still trips the same curated pattern via "telephone inquiries" /
        "clients in a polite manner", so the synonym-only path stays clean."""
        job = _job(
            "- Accept and respond to telephone inquiries from clients in a polite manner\n"
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

    def test_inflected_match_still_recorded_as_exact_keyword(self):
        """2026-09-03: an inflected (plural/singular) literal match must be
        classified as "literal" / land in exact_keywords, not fall through
        to synonym_concepts -- _term_in_text's inflection tolerance is a
        same-word variant, not a different word, so it should behave exactly
        like any other literal hit from _match_kind's perspective.

        2026-09-04: swapped the fixture word from "customer"/"customers" to
        "database"/"databases" -- "customer" (idf=1.80) is now excluded by
        the separate IDF-based generic-term check added the same day (see
        test_local_llm.py::TestGenericEvidenceTermFiltering), which would
        make this item unsupported for an unrelated reason and defeat the
        actual thing this test checks. "database" (idf=3.50) isn't generic,
        so it isolates the inflection-tolerance behavior cleanly."""
        job = _job("- Prior experience working with databases is required\n")
        profile = {
            "experience_inventory": [],
            "project_inventory": [],
            "skills_inventory": [
                {
                    "name": "Database",
                    "relevance_categories": ["database administration"],
                    "resume_allowed": True,
                    "evidence_level": "expert",
                },
            ],
            "certifications": [],
        }
        rep = schemas.build_job_schema_representation(job, profile)
        req = rep["requirements"][0]
        self.assertTrue(req["supported"])
        self.assertIn("database", req["exact_keywords"])
        self.assertEqual(req["synonym_concepts"], [])

    def test_format_schema_guidance_surfaces_exact_keywords_verbatim(self):
        job = _job(JOB_DESCRIPTION)
        rep = schemas.build_job_schema_representation(job, TROUBLESHOOTING_PROFILE)
        guidance = schemas.format_schema_guidance(rep)
        self.assertIn("root cause", guidance)
        self.assertIn("use these exact terms", guidance)

    def test_cross_domain_ambiguous_term_dropped_from_exact_keywords(self):
        """2026-09-04 regression: 'alignment' is globally common vocabulary
        in two unrelated domains (corporate 'strategic alignment' vs.
        automotive 'wheel alignment') -- a real case found via the v6 IDF-
        gate experiment (v6_idf_gate_validation_report.md), where an auto-
        repair sentence got cited as evidence for a Product Manager
        requirement purely because both texts happen to contain the word
        'alignment'. The two usages share no other local context, so
        _context_senses_agree must reject it -- 'alignment' should not
        appear in exact_keywords even though it's a literal string match."""
        job = _job(
            "- Lead strategic engagements with stakeholders to drive organizational "
            "alignment on business priorities\n"
        )
        profile = {
            "experience_inventory": [
                {
                    "name": "Auto Shop",
                    "relevance_categories": ["automotive"],
                    "resume_allowed": True,
                    "responsibilities": [
                        "Diagnosed and corrected vehicle alignment issues using specialized "
                        "equipment and established troubleshooting procedures."
                    ],
                },
            ],
            "project_inventory": [],
            "skills_inventory": [],
            "certifications": [],
        }
        rep = schemas.build_job_schema_representation(job, profile)
        req = rep["requirements"][0]
        self.assertTrue(req["supported"])
        self.assertNotIn("alignment", req["exact_keywords"])
        # "alignment" was the ONLY literal term this evidence item shared
        # with the requirement -- with it correctly dropped, exact_keywords
        # is empty and the tier must demote all the way to "unsupported"
        # (a zero-keyword "literal" match is no match at all, not a weaker
        # one -- see classify_category_tier's 2026-09-04 handling of
        # exact_keyword_count == 0).
        self.assertEqual(req["exact_keywords"], [])
        self.assertEqual(req["category_tier"], "unsupported")

    def test_same_domain_ambiguous_term_still_counted(self):
        """Mirror case: when both texts genuinely use 'alignment' in the
        SAME (automotive) sense, the context check must not over-block it
        -- this is the false-negative side of the same mechanism."""
        job = _job("- Perform wheel alignment and diagnose vehicle alignment issues\n")
        profile = {
            "experience_inventory": [
                {
                    "name": "Auto Shop",
                    "relevance_categories": ["automotive"],
                    "resume_allowed": True,
                    "responsibilities": [
                        "Diagnosed and corrected vehicle alignment issues using specialized "
                        "equipment and established troubleshooting procedures."
                    ],
                },
            ],
            "project_inventory": [],
            "skills_inventory": [],
            "certifications": [],
        }
        rep = schemas.build_job_schema_representation(job, profile)
        req = rep["requirements"][0]
        self.assertIn("alignment", req["exact_keywords"])

    def test_installation_cross_domain_ambiguous_term_dropped(self):
        """2026-09-04 regression: found via a real batch scan of 1477 DB
        jobs during the deterministic-selector work -- "install"/
        "installation" is the same shape of collision as "alignment":
        Alex Prosperity Group's evidence is genuinely about PHYSICAL
        appliance installation, but the literal word also appears in
        totally unrelated SOFTWARE installation contexts (this fixture
        mirrors a real hit: "RedHat Linux operating systems" installation)."""
        job = _job(
            "- Experience in the installation, configuration, and maintenance of "
            "RedHat Linux operating systems\n"
        )
        profile = {
            "experience_inventory": [
                {
                    "name": "Alex Prosperity Group / UST Logistics",
                    "resume_allowed": True,
                    "responsibilities": [
                        "Installed home appliances contracted through Lowe's",
                        "Performed hands-on installation work while following customer "
                        "requirements and established procedures.",
                        "Communicated clearly with customers throughout installations.",
                    ],
                },
            ],
            "project_inventory": [],
            "skills_inventory": [],
            "certifications": [],
        }
        rep = schemas.build_job_schema_representation(job, profile)
        req = rep["requirements"][0]
        self.assertNotIn("installation", req["exact_keywords"])
        self.assertEqual(req["exact_keywords"], [])
        self.assertEqual(req["category_tier"], "unsupported")
        # NOTE: req["supported"] stays True here -- same as the existing
        # "alignment" test above. The ambiguous-term check demotes
        # category_tier (the confidence label) but does not flip the
        # top-level `supported` flag set earlier by local_tailor.py's
        # _auto_resolve_requirements, which has no knowledge of
        # _AMBIGUOUS_TERMS. This is a real, pre-existing gap (not
        # introduced by this test) -- consumers that gate on `supported`
        # (e.g. local_tailor.build_pool_realization) are NOT actually
        # blocked by this safety net, only schemas.py's own reported
        # confidence fields are. Documented, not fixed, here.

    def test_installation_same_domain_still_counted(self):
        """Mirror case: a job genuinely about physical/appliance
        installation must still match -- the context check shouldn't
        over-block legitimate same-domain hits."""
        job = _job("- Install home appliances per manufacturer specifications and customer requirements\n")
        profile = {
            "experience_inventory": [
                {
                    "name": "Alex Prosperity Group / UST Logistics",
                    "resume_allowed": True,
                    "responsibilities": [
                        "Installed home appliances contracted through Lowe's",
                        "Performed hands-on installation work while following customer "
                        "requirements and established procedures.",
                    ],
                },
            ],
            "project_inventory": [],
            "skills_inventory": [],
            "certifications": [],
        }
        rep = schemas.build_job_schema_representation(job, profile)
        req = rep["requirements"][0]
        self.assertTrue(req["supported"])

    def test_context_senses_agree_direct(self):
        """Direct unit coverage of the helper, independent of the full
        schema-build pipeline."""
        # same sense (both automotive) -> agree
        self.assertTrue(
            schemas._context_senses_agree(
                "Perform wheel alignment and brake inspections on vehicles",
                "Diagnosed vehicle alignment issues using specialized equipment",
                "alignment",
            )
        )
        # different senses (corporate vs. automotive) -> disagree
        self.assertFalse(
            schemas._context_senses_agree(
                "Drive organizational alignment on strategic business priorities",
                "Diagnosed vehicle alignment issues using specialized equipment",
                "alignment",
            )
        )
        # no context to compare on one side -> conservative False, not True
        self.assertFalse(schemas._context_senses_agree("alignment.", "Diagnosed vehicle alignment issues.", "alignment"))


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
