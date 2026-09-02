"""Tests for the 2026-08-24 cognitive-linguistic architecture expansion.

Adds, on top of the existing evidence-grounding + rhetorical-schema layer
(scoring/schemas.py, introduced 2026-08-23): frame semantics, event
structure, a graded category tier, a claim-strength lattice (the central
safety layer -- ceiling derived from the evidence's OWN description text,
enforced deterministically post-generation), viewpoint, salience ordering,
construction patterns, discourse-repetition flagging, and explicit
provenance. See scoring/schemas.py's module docstring for the full
architecture assessment (which systems were implemented vs. deliberately
deferred as redundant with what's implemented, and why).

Covers every category from the review request: evidence safety (literal /
synonym / peripheral / unsupported / ambiguous / illegitimate transfer),
construal (viewpoint / salience / event-type-driven profiling), event
structure (accomplishment vs. activity), claim strength ("used" cannot
become "architected"; "participated in" cannot become "led"), causality
(unsupported outcome claims rejected), domain transfer (structural
similarity never becomes literal domain experience), provenance (every
realized claim traceable to evidence text), and full-suite regression.
"""

import unittest
from unittest.mock import MagicMock

from applypilot.scoring import local_tailor, schemas, validator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _job(description: str, title: str = "Support Technician", url: str = "https://example.com/clj1") -> dict:
    return {"title": title, "url": url, "full_description": description}


PROFILE = {
    "experience_inventory": [
        {
            "name": "Auto Shop Diagnostic Tech",
            "relevance_categories": ["troubleshooting", "root cause"],
            "resume_allowed": True,
            "description": "Diagnosed and repaired vehicle electrical faults, restoring reliable operation.",
        },
        {
            "name": "Ops Assistant",
            "relevance_categories": ["automation"],
            "resume_allowed": True,
            "description": "Used scripts to automate repetitive reporting tasks.",
        },
        {
            "name": "Warehouse Lead",
            "relevance_categories": ["process improvement"],
            "resume_allowed": True,
            "description": "Led a small team; reduced order-picking time by streamlining the workflow.",
        },
    ],
    "project_inventory": [],
    "skills_inventory": [
        {
            "name": "Customer Service",
            "relevance_categories": ["customer service"],
            "resume_allowed": True,
            "evidence_level": "expert",
        },
        {
            "name": "Python",
            "relevance_categories": ["automation", "technical"],
            "resume_allowed": True,
            "evidence_level": "working",
        },
    ],
    "certifications": [],
    "skills_boundary": {},
}


# ---------------------------------------------------------------------------
# 1. Frame semantics
# ---------------------------------------------------------------------------


class TestFrameSelection(unittest.TestCase):
    def test_troubleshooting_requirement_selects_diagnostic_support(self):
        self.assertEqual(
            schemas.select_frame("Troubleshoot hardware and software issues and identify root cause"),
            "diagnostic_support",
        )

    def test_automation_requirement_selects_automation_frame(self):
        self.assertEqual(
            schemas.select_frame("Automate repetitive manual reporting workflows"),
            "automation",
        )

    def test_deployment_requirement_selects_deployment_frame(self):
        self.assertEqual(
            schemas.select_frame("Manage CI/CD pipelines and deployment rollouts"),
            "deployment",
        )

    def test_migration_requirement_selects_migration_frame(self):
        self.assertEqual(
            schemas.select_frame("Migrate legacy systems to a modernized platform"),
            "migration",
        )

    def test_customer_requirement_selects_customer_support_frame(self):
        self.assertEqual(
            schemas.select_frame("Respond to customer inquiries and support requests"),
            "customer_support",
        )

    def test_leadership_requirement_selects_leadership_project_frame(self):
        self.assertEqual(
            schemas.select_frame("Lead and mentor a team of engineers"),
            "leadership_project",
        )

    def test_generic_requirement_falls_back_to_general_capability(self):
        self.assertEqual(
            schemas.select_frame("Some entirely generic requirement with no frame hints"),
            "general_capability",
        )

    def test_automation_takes_priority_over_deployment_when_both_hint(self):
        """'Automate the deployment pipeline' is fundamentally an automation
        requirement, not a deployment one -- priority order matters."""
        self.assertEqual(
            schemas.select_frame("Automate the deployment pipeline to reduce manual releases"),
            "automation",
        )

    def test_every_frame_has_a_description_and_slots(self):
        for name, frame in schemas.FRAMES.items():
            self.assertTrue(frame.get("description"), f"{name} has no description")
            self.assertTrue(frame.get("slots"), f"{name} has no slots")


# ---------------------------------------------------------------------------
# 2. Event structure
# ---------------------------------------------------------------------------


class TestEventStructure(unittest.TestCase):
    def test_evidence_with_outcome_verb_is_accomplishment(self):
        self.assertEqual(
            schemas.classify_event_type("Automated reporting, reducing manual processing time."),
            "accomplishment",
        )

    def test_evidence_with_no_outcome_verb_is_activity(self):
        self.assertEqual(
            schemas.classify_event_type("Responsible for daily monitoring of the ticket queue."),
            "activity",
        )

    def test_empty_evidence_text_is_activity(self):
        self.assertEqual(schemas.classify_event_type(""), "activity")

    def test_gerund_form_of_outcome_verb_is_still_detected(self):
        """'reducing' (gerund) must be recognized, not just 'reduced' (past
        tense) -- LLM-generated and human-written text uses both forms."""
        self.assertEqual(
            schemas.classify_event_type("Currently reducing backlog size through triage."),
            "accomplishment",
        )


# ---------------------------------------------------------------------------
# 3. Category tier (graded prototype scale)
# ---------------------------------------------------------------------------


class TestCategoryTier(unittest.TestCase):
    def test_literal_match_with_multiple_keywords_is_prototype(self):
        self.assertEqual(schemas.classify_category_tier("literal", 2), "prototype")
        self.assertEqual(schemas.classify_category_tier("literal", 3), "prototype")

    def test_literal_match_with_one_keyword_is_near_prototype(self):
        self.assertEqual(schemas.classify_category_tier("literal", 1), "near_prototype")

    def test_synonym_match_is_peripheral(self):
        self.assertEqual(schemas.classify_category_tier("synonym", 0), "peripheral")

    def test_unknown_match_is_unsupported(self):
        self.assertEqual(schemas.classify_category_tier("unknown", 0), "unsupported")

    def test_semantic_match_is_unsupported_never_prototype_or_peripheral(self):
        """2026-08-31: embedding-similarity provenance must never receive
        the same category tier as a literal or curated-synonym match --
        real-data validation found raw cosine similarity unreliable at
        distinguishing genuine matches from false positives."""
        self.assertEqual(schemas.classify_category_tier("semantic", 0), "unsupported")
        self.assertEqual(schemas.classify_category_tier("semantic", 5), "unsupported")  # keyword count irrelevant


class TestMatchKindSemanticProvenance(unittest.TestCase):
    def test_semantic_provenance_with_no_term_overlap_classified_semantic(self):
        item = {"matched_terms": [], "provenance": "semantic"}
        self.assertEqual(schemas._match_kind("Install PC hardware.", item), "semantic")

    def test_no_provenance_marker_falls_back_to_unknown(self):
        item = {"matched_terms": []}
        self.assertEqual(schemas._match_kind("Install PC hardware.", item), "unknown")

    def test_literal_overlap_wins_over_semantic_provenance(self):
        """An item _auto_resolve_requirements added via semantic recall
        that ALSO happens to share a literal term with this specific
        requirement line is classified by the stronger signal, not
        weakened to "semantic"."""
        item = {"matched_terms": ["python"], "provenance": "semantic"}
        self.assertEqual(schemas._match_kind("Use python for automation.", item), "literal")


# ---------------------------------------------------------------------------
# 4. Claim-strength lattice: detection, ceiling derivation, and the
#    deterministic post-generation check (the safety-critical layer).
# ---------------------------------------------------------------------------


class TestClaimTierDetection(unittest.TestCase):
    def test_tier_ordering_is_correct(self):
        self.assertEqual(
            list(schemas.CLAIM_TIERS),
            ["participation", "execution", "implementation", "design", "authority"],
        )

    def test_used_is_participation_tier(self):
        self.assertEqual(schemas.detect_claim_tier("Used Python for scripting."), "participation")

    def test_configured_is_execution_tier(self):
        self.assertEqual(schemas.detect_claim_tier("Configured the monitoring dashboards."), "execution")

    def test_implemented_is_implementation_tier(self):
        self.assertEqual(schemas.detect_claim_tier("Implemented a new deployment pipeline."), "implementation")

    def test_designed_is_design_tier(self):
        self.assertEqual(schemas.detect_claim_tier("Designed the system architecture."), "design")

    def test_architected_is_authority_tier(self):
        self.assertEqual(schemas.detect_claim_tier("Architected the infrastructure."), "authority")

    def test_engineers_as_a_job_title_noun_is_not_design_tier(self):
        """'engineers' is far more often the plural-noun job title ('a
        team of engineers') than the rare 3rd-person verb -- a false
        positive here would wrongly inflate a requirement's ceiling."""
        self.assertIsNone(schemas.detect_claim_tier("Led a team of on-call engineers."))

    def test_architects_as_a_job_title_noun_is_not_authority_tier(self):
        """Same noun/verb collision as 'engineers', but on the HIGHEST
        tier -- the most dangerous case to get wrong."""
        self.assertIsNone(schemas.detect_claim_tier("Collaborated with senior solutions architects on the design."))

    def test_engineered_verb_form_still_detected(self):
        self.assertEqual(schemas.detect_claim_tier("Engineered a resilient failover system."), "design")

    def test_feature_engineering_as_a_technique_noun_is_not_design_tier(self):
        """2026-09-02: found via the deterministic slot-filler prototype
        scanning real profile.json factual_concepts -- 'Feature Engineering'
        (a data-science technique name) matched the bare 'engineering'
        gerund and wrongly elevated that evidence's claim ceiling to
        'design'. Same noun/verb-collision class as the engineers/
        architects bare-plural exclusions above: 'X engineering'
        (feature/data/software/prompt engineering) is overwhelmingly a
        field name in resume/job text, not a first-person action verb."""
        self.assertIsNone(schemas.detect_claim_tier("Feature engineering and statistical modeling experimentation."))

    def test_led_is_not_a_technical_depth_verb(self):
        """2026-08-24: 'led' moved OUT of the technical-depth lattice into
        the separate agency axis (see TestAgencyAxis) -- evidence
        supporting 'architected X' (technical authority) must not silently
        license 'led the team that built X' (people authority), which is
        a genuinely different claim. detect_claim_tier no longer
        recognizes people-management verbs at all."""
        self.assertIsNone(schemas.detect_claim_tier("Led the migration effort."))

    def test_gerund_forms_are_detected(self):
        """'implementing' etc. must resolve the same as its base form --
        LLM realizations commonly use -ing constructions."""
        self.assertEqual(schemas.detect_claim_tier("Implementing new pipelines."), "implementation")

    def test_highest_tier_wins_when_multiple_verbs_present(self):
        self.assertEqual(
            schemas.detect_claim_tier("Used Python and later architected the full platform."),
            "authority",
        )

    def test_no_recognized_verb_returns_none(self):
        self.assertIsNone(schemas.detect_claim_tier("A noun phrase fragment with no verb"))

    def test_empty_text_returns_none(self):
        self.assertIsNone(schemas.detect_claim_tier(""))

    def test_user_does_not_false_positive_as_used(self):
        """A loose 'us\\w*' stem would wrongly match 'user'/'usable' -- the
        explicit verb-form lists must not do this."""
        self.assertIsNone(schemas.detect_claim_tier("Trained new users on the ticketing system."))


class TestClaimCeilingDerivation(unittest.TestCase):
    def test_ceiling_derived_from_evidence_own_verb(self):
        item = {"description": "Designed and implemented the reporting pipeline."}
        self.assertEqual(schemas.claim_ceiling_for_evidence(item), "design")

    def test_ceiling_defaults_to_participation_when_no_verb_found(self):
        """Silence in the evidence is never license to claim MORE -- the
        default is the most conservative tier, not a middle one."""
        item = {"description": "Backend systems and reporting infrastructure."}
        self.assertEqual(schemas.claim_ceiling_for_evidence(item), "participation")

    def test_ceiling_considers_factual_concepts_too(self):
        item = {"description": "", "factual_concepts": ["Architected the ingestion pipeline"]}
        self.assertEqual(schemas.claim_ceiling_for_evidence(item), "authority")


class TestClaimStrengthCheck(unittest.TestCase):
    """The exact example from the review request: 'used' cannot become
    'architected' (technical-depth axis). 'participated in' cannot become
    'led' is now TestAgencyAxis's job -- 'led' is a people-authority claim,
    not a technical-depth one (see the 2026-08-24 axis split)."""

    def test_used_cannot_become_architected(self):
        result = schemas.check_claim_strength("Architected the core platform.", ceiling="participation")
        self.assertFalse(result["passed"])
        self.assertEqual(result["detected_tier"], "authority")

    def test_claim_at_or_below_ceiling_passes(self):
        self.assertTrue(schemas.check_claim_strength("Used Python for automation.", ceiling="participation")["passed"])
        self.assertTrue(schemas.check_claim_strength("Configured the pipeline.", ceiling="execution")["passed"])
        self.assertTrue(schemas.check_claim_strength("Architected the platform.", ceiling="authority")["passed"])

    def test_claim_with_no_recognized_verb_always_passes(self):
        """This lattice only rejects OVERCLAIMING -- it never requires a
        claim verb to be present at all."""
        result = schemas.check_claim_strength("Backend systems and reporting infrastructure.", ceiling="participation")
        self.assertTrue(result["passed"])
        self.assertIsNone(result["detected_tier"])

    def test_exact_boundary_tier_passes(self):
        """A claim at EXACTLY the ceiling tier (not below) must pass."""
        result = schemas.check_claim_strength("Designed the workflow.", ceiling="design")
        self.assertTrue(result["passed"])


class TestAgencyAxis(unittest.TestCase):
    """2026-08-24: the agency axis, orthogonal to technical depth. The
    review request's exact example: 'participated in' cannot become
    'led'. Gap this closes: evidence supporting 'architected X' (solo
    technical authority) does not, by itself, license 'led the team that
    architected X' (people authority) -- those are different claims that
    the OLD single merged lattice could not tell apart (both used to rank
    as the single 'authority' tier)."""

    def test_tier_ordering_is_correct(self):
        self.assertEqual(
            list(schemas.AGENCY_TIERS),
            ["individual_contributor", "owner", "team_lead", "director"],
        )

    def test_led_and_managed_are_team_lead_tier(self):
        self.assertEqual(schemas.detect_agency_tier("Led the migration effort."), "team_lead")
        self.assertEqual(schemas.detect_agency_tier("Managed the on-call rotation."), "team_lead")

    def test_directed_and_spearheaded_are_director_tier(self):
        self.assertEqual(schemas.detect_agency_tier("Directed the initiative."), "director")
        self.assertEqual(schemas.detect_agency_tier("Spearheaded the effort."), "director")

    def test_owned_is_owner_tier(self):
        self.assertEqual(schemas.detect_agency_tier("Owned the deployment process."), "owner")

    def test_no_agency_vocabulary_returns_none(self):
        """individual_contributor has no dedicated vocabulary -- it's the
        default absence of a stronger agency signal."""
        self.assertIsNone(schemas.detect_agency_tier("Used Python for scripting."))

    def test_agency_ceiling_defaults_to_individual_contributor(self):
        item = {"description": "Implemented the reporting pipeline."}
        self.assertEqual(schemas.agency_ceiling_for_evidence(item), "individual_contributor")

    def test_participated_in_cannot_become_led(self):
        """The exact review-request example, now on its correct axis: an
        individual-contributor-ceiling evidence item cannot license a
        realization claiming 'led'."""
        result = schemas.check_agency_strength("Led the migration effort.", ceiling="individual_contributor")
        self.assertFalse(result["passed"])
        self.assertEqual(result["detected_tier"], "team_lead")

    def test_architected_alone_does_not_license_led(self):
        """The core gap this axis closes: technical-depth evidence
        ('architected X') must NOT license a people-authority claim
        ('led the team') just because both used to rank as 'authority' on
        the old single merged lattice. Checked on the agency axis
        specifically -- the evidence never mentions a team."""
        evidence_item = {"description": "Architected the payments platform."}
        agency_ceiling = schemas.agency_ceiling_for_evidence(evidence_item)
        self.assertEqual(agency_ceiling, "individual_contributor")
        result = schemas.check_agency_strength("Led the team that architected the platform.", ceiling=agency_ceiling)
        self.assertFalse(result["passed"])

    def test_led_evidence_licenses_led_claim(self):
        """A genuinely team-lead-ceiling evidence item DOES license a
        'led' realization -- this axis rejects overclaiming, not all
        leadership language."""
        result = schemas.check_agency_strength("Led the migration effort.", ceiling="team_lead")
        self.assertTrue(result["passed"])

    def test_no_agency_claim_in_realized_text_always_passes(self):
        result = schemas.check_agency_strength("Used Python for scripting.", ceiling="individual_contributor")
        self.assertTrue(result["passed"])
        self.assertIsNone(result["detected_tier"])


class TestCausalClaimCheck(unittest.TestCase):
    def test_unsupported_causal_claim_is_rejected(self):
        result = schemas.check_causal_claim(
            "Used scripts to automate reporting, reducing manual work by 80%.",
            evidence_text="Used scripts to automate repetitive reporting tasks.",
        )
        self.assertFalse(result["passed"])

    def test_causal_claim_grounded_in_evidence_passes(self):
        result = schemas.check_causal_claim(
            "Automated reporting, reducing manual processing time.",
            evidence_text="Automated reporting tasks, reducing manual processing time significantly.",
        )
        self.assertTrue(result["passed"])

    def test_weak_construction_never_triggers_the_check(self):
        """'used'/'supported'/'helped' etc. never claim a causal outcome,
        so they're exempt from this check regardless of evidence."""
        result = schemas.check_causal_claim("Used Python for scripting.", evidence_text="")
        self.assertTrue(result["passed"])


class TestMetricFabricationCheck(unittest.TestCase):
    """A verb can be legitimately grounded while the NUMBER riding along
    with it is fabricated -- check_causal_claim alone doesn't catch this."""

    def test_ungrounded_number_is_rejected(self):
        result = schemas.check_metric_fabrication(
            "Reduced deployment time by 30%.",
            evidence_text="Reduced deployment time through automation.",
        )
        self.assertFalse(result["passed"])

    def test_number_present_in_evidence_text_passes(self):
        result = schemas.check_metric_fabrication(
            "Reduced deployment time by 30%.",
            evidence_text="Reduced deployment time by 30% through automation.",
        )
        self.assertTrue(result["passed"])

    def test_number_present_in_known_metrics_passes(self):
        result = schemas.check_metric_fabrication(
            "Reduced deployment time by 30%.",
            evidence_text="Reduced deployment time through automation.",
            known_metrics=["30% faster deployments"],
        )
        self.assertTrue(result["passed"])

    def test_text_with_no_numbers_always_passes(self):
        result = schemas.check_metric_fabrication(
            "Reduced deployment time through automation.",
            evidence_text="",
        )
        self.assertTrue(result["passed"])


class TestPassiveVoiceCheck(unittest.TestCase):
    """Soft/logged, not a hard reject -- see schemas.check_passive_voice's
    docstring for the false-positive rationale."""

    def test_passive_construction_is_detected(self):
        result = schemas.check_passive_voice("The network failures were diagnosed.")
        self.assertFalse(result["passed"])

    def test_active_construction_passes(self):
        result = schemas.check_passive_voice("Diagnosed the network failures.")
        self.assertTrue(result["passed"])


# ---------------------------------------------------------------------------
# 5. Viewpoint (construal: perspective)
# ---------------------------------------------------------------------------


class TestViewpointSelection(unittest.TestCase):
    def test_engineering_title_selects_engineering_viewpoint(self):
        self.assertEqual(schemas.select_viewpoint({"title": "Software Engineer"}), "engineering")

    def test_devops_title_selects_operations_viewpoint(self):
        self.assertEqual(schemas.select_viewpoint({"title": "DevOps Engineer"}), "operations")

    def test_data_title_selects_data_viewpoint(self):
        self.assertEqual(schemas.select_viewpoint({"title": "Data Engineer"}), "data")

    def test_support_title_selects_support_viewpoint(self):
        self.assertEqual(schemas.select_viewpoint({"title": "Technical Support Engineer"}), "support")

    def test_manager_title_selects_leadership_viewpoint(self):
        self.assertEqual(schemas.select_viewpoint({"title": "Engineering Manager"}), "leadership")

    def test_unrecognized_title_falls_back_to_general(self):
        self.assertEqual(schemas.select_viewpoint({"title": "Marine Biologist"}), "general")

    def test_same_facts_different_viewpoint_for_different_jobs(self):
        """The core construal claim: the SAME evidence can be legitimately
        presented from different angles depending on the target job -- no
        fact changes, only which viewpoint tag is attached."""
        engineering_job = _job("- Build backend services\n", title="Software Engineer")
        operations_job = _job("- Build backend services\n", title="Site Reliability Engineer")
        rep_eng = schemas.build_job_schema_representation(engineering_job, PROFILE)
        rep_ops = schemas.build_job_schema_representation(operations_job, PROFILE)
        self.assertNotEqual(rep_eng["viewpoint"], rep_ops["viewpoint"])


# ---------------------------------------------------------------------------
# 6. Salience ordering (figure-ground / profiling / information structure)
# ---------------------------------------------------------------------------


class TestSalienceRanking(unittest.TestCase):
    def test_accomplishment_profiles_outcome_first(self):
        self.assertEqual(schemas.rank_salience("accomplishment")[0], "outcome")

    def test_activity_profiles_action_first(self):
        self.assertEqual(schemas.rank_salience("activity")[0], "action")

    def test_salience_order_is_a_complete_four_element_ranking(self):
        """Accomplishment and activity profile different element sets
        (mechanism vs. action) by design -- both must still be complete,
        duplicate-free 4-element orderings."""
        for event_type in ("accomplishment", "activity"):
            order = schemas.rank_salience(event_type)
            self.assertEqual(len(order), 4)
            self.assertEqual(len(set(order)), 4)  # no duplicates
            self.assertIn("outcome", order)
            self.assertIn("technology", order)
            self.assertIn("context", order)


# ---------------------------------------------------------------------------
# 7. Discourse coherence (repetition flagging)
# ---------------------------------------------------------------------------


class TestDiscourseRepetitionFlagging(unittest.TestCase):
    def _requirement(self, frame, cog_schema):
        return {
            "supported": True,
            "frame": frame,
            "schema": {"cognitive_schema": cog_schema, "bullet_schema": "action_object_context_outcome"},
            "vary_phrasing": False,
        }

    def test_three_or_more_repeats_get_flagged(self):
        reqs = [self._requirement("automation", "action_object_outcome") for _ in range(3)]
        schemas.flag_repetition(reqs)
        self.assertTrue(all(r["vary_phrasing"] for r in reqs))

    def test_two_repeats_are_not_flagged(self):
        reqs = [self._requirement("automation", "action_object_outcome") for _ in range(2)]
        schemas.flag_repetition(reqs)
        self.assertTrue(all(not r["vary_phrasing"] for r in reqs))

    def test_diverse_frames_are_never_flagged(self):
        reqs = [
            self._requirement("automation", "action_object_outcome"),
            self._requirement("diagnostic_support", "problem_diagnosis_repair_verification"),
            self._requirement("customer_support", "evidence_claim"),
        ]
        schemas.flag_repetition(reqs)
        self.assertTrue(all(not r["vary_phrasing"] for r in reqs))

    def test_flagging_never_changes_which_schema_was_assigned(self):
        """Diversity is a phrasing nudge, never a reinterpretation of
        evidence -- the schema assignment itself must be untouched."""
        reqs = [self._requirement("automation", "action_object_outcome") for _ in range(3)]
        original_schemas = [r["schema"]["cognitive_schema"] for r in reqs]
        schemas.flag_repetition(reqs)
        self.assertEqual([r["schema"]["cognitive_schema"] for r in reqs], original_schemas)


# ---------------------------------------------------------------------------
# 8. Provenance
# ---------------------------------------------------------------------------


class TestProvenance(unittest.TestCase):
    """2026-08-24 (#T): provenance is now structured
    {"source_type": ..., "text": ...} rather than a bare string -- the LLM
    (and any human auditor) should be able to see WHERE a claim came from
    (experience/project/skill/certification), not just its text, so a
    derived inference can never be silently confused with direct source
    evidence."""

    def test_supported_requirement_carries_evidence_source_text_and_type(self):
        job = _job("- Troubleshoot hardware and software issues and identify root cause\n")
        rep = schemas.build_job_schema_representation(job, PROFILE)
        req = rep["requirements"][0]
        self.assertTrue(req["provenance"])
        self.assertIn("Diagnosed and repaired", req["provenance"][0]["text"])
        self.assertEqual(req["provenance"][0]["source_type"], "experience")

    def test_unsupported_requirement_has_no_provenance(self):
        job = _job("- Conduct deep-sea coral reef surveys\n")
        rep = schemas.build_job_schema_representation(job, PROFILE)
        req = rep["requirements"][0]
        self.assertFalse(req["supported"])
        self.assertEqual(req["provenance"], [])


# ---------------------------------------------------------------------------
# 9. Evidence safety matrix (unsupported / ambiguous / literal / synonym /
#    peripheral / illegitimate transfer)
# ---------------------------------------------------------------------------


class TestEvidenceSafetyMatrix(unittest.TestCase):
    def test_unsupported_requirement_gets_no_schema_frame_or_ceiling(self):
        job = _job("- Conduct deep-sea coral reef surveys using submersible equipment\n")
        rep = schemas.build_job_schema_representation(job, PROFILE)
        req = rep["requirements"][0]
        self.assertFalse(req["supported"])
        self.assertIsNone(req["schema"])
        self.assertIsNone(req["frame"])
        self.assertIsNone(req["claim_ceiling"])
        self.assertIsNone(req["event_type"])

    def test_ambiguous_requirement_gets_no_schema_frame_or_ceiling(self):
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
        req = rep["requirements"][0]
        self.assertTrue(req["ambiguous"])
        self.assertFalse(req["supported"])
        self.assertIsNone(req["schema"])

    def test_literal_match_yields_prototype_or_near_prototype_tier(self):
        job = _job("- Troubleshoot hardware and software issues and identify root cause\n")
        rep = schemas.build_job_schema_representation(job, PROFILE)
        req = rep["requirements"][0]
        self.assertIn(req["category_tier"], ("prototype", "near_prototype"))
        self.assertTrue(req["exact_keywords"])
        self.assertFalse(req["synonym_concepts"])

    def test_synonym_match_yields_peripheral_tier_never_prototype(self):
        job = _job(
            "- Accept and respond to telephone inquiries from customers in a polite manner\n"
            "- Prior customer service experience needed\n"  # anchors the job-level literal match
        )
        rep = schemas.build_job_schema_representation(job, PROFILE)
        req = rep["requirements"][0]
        self.assertIn("telephone", req["requirement"].lower())
        self.assertEqual(req["category_tier"], "peripheral")
        self.assertFalse(req["exact_keywords"])
        self.assertTrue(req["synonym_concepts"])

    def test_illegitimate_transfer_never_gets_a_direct_match_schema(self):
        """A synonym/peripheral match must NEVER be assigned a direct-match
        cognitive schema (action_object_outcome/evidence_claim/problem_
        diagnosis...) regardless of how the requirement text reads -- only
        domain_transfer, which explicitly instructs against claiming
        identical-domain experience."""
        job = _job(
            "- Accept and respond to telephone inquiries from customers in a polite manner\n"
            "- Prior customer service experience needed\n"
        )
        rep = schemas.build_job_schema_representation(job, PROFILE)
        req = rep["requirements"][0]
        self.assertEqual(req["schema"]["cognitive_schema"], "domain_transfer")
        self.assertEqual(req["schema"]["bullet_schema"], "domain_transfer_bullet")

    def test_legitimate_transfer_is_grounded_not_fabricated(self):
        """A transferable/peripheral match still traces to REAL evidence
        (resume_evidence is non-empty) -- it's grounded transfer, not an
        invented capability."""
        job = _job(
            "- Accept and respond to telephone inquiries from customers in a polite manner\n"
            "- Prior customer service experience needed\n"
        )
        rep = schemas.build_job_schema_representation(job, PROFILE)
        req = rep["requirements"][0]
        self.assertTrue(req["supported"])
        self.assertTrue(req["resume_evidence"])


# ---------------------------------------------------------------------------
# 10. Construction patterns (BULLET_SCHEMAS)
# ---------------------------------------------------------------------------


class TestConstructionPatterns(unittest.TestCase):
    def test_every_bullet_schema_has_a_concrete_pattern(self):
        for name, schema in schemas.BULLET_SCHEMAS.items():
            self.assertTrue(schema.get("pattern"), f"{name} has no construction pattern")

    def test_pattern_references_the_schema_slots(self):
        for schema in schemas.BULLET_SCHEMAS.values():
            pattern_lower = schema["pattern"].lower()
            # At least the first slot should be recognizable in the pattern text.
            self.assertIn(schema["slots"][0].split("_")[0], pattern_lower.replace("/", " "))


# ---------------------------------------------------------------------------
# 11. Claim-strength enforcement integration (degraded-mode realization)
# ---------------------------------------------------------------------------


class TestClaimStrengthEnforcementIntegration(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_overclaiming_bullet_is_dropped_not_shipped(self):
        job = _job("- Troubleshoot hardware and software issues and identify root cause\n")
        job_schema = schemas.build_job_schema_representation(job, PROFILE)
        client = MagicMock()
        client.chat.return_value = (
            '{"summary": "Architected enterprise diagnostic infrastructure.", '
            '"bullets": [{"evidence": "Auto Shop Diagnostic Tech", '
            '"text": "Architected the full vehicle diagnostic system."}]}'
        )
        realization, meta = local_tailor.request_local_realization(client, job, job_schema)
        self.assertIsNone(realization)  # both bullet and summary rejected
        self.assertEqual(meta["claim_strength_violations"], 2)

    def test_legitimate_claim_within_ceiling_is_kept(self):
        job = _job("- Troubleshoot hardware and software issues and identify root cause\n")
        job_schema = schemas.build_job_schema_representation(job, PROFILE)
        client = MagicMock()
        client.chat.return_value = (
            '{"summary": "Support technician with hands-on diagnostic experience.", '
            '"bullets": [{"evidence": "Auto Shop Diagnostic Tech", '
            '"text": "Diagnosed and repaired electrical faults, restoring reliable operation."}]}'
        )
        realization, meta = local_tailor.request_local_realization(client, job, job_schema)
        self.assertIsNotNone(realization)
        self.assertEqual(meta["claim_strength_violations"], 0)
        self.assertIn("Auto Shop Diagnostic Tech", realization["bullets"])

    def test_unsupported_causal_claim_is_dropped(self):
        job = _job("- Experience with automation is required\n")
        profile = {
            "experience_inventory": [
                {
                    "name": "Ops Assistant",
                    "relevance_categories": ["automation"],
                    "resume_allowed": True,
                    "description": "Used scripts to automate repetitive reporting tasks.",
                }
            ],
            "project_inventory": [],
            "skills_inventory": [],
            "certifications": [],
        }
        job_schema = schemas.build_job_schema_representation(job, profile)
        client = MagicMock()
        client.chat.return_value = (
            '{"summary": "Automation specialist.", '
            '"bullets": [{"evidence": "Ops Assistant", '
            '"text": "Used scripts to automate reporting, reducing manual work by 80 percent."}]}'
        )
        _realization, meta = local_tailor.request_local_realization(client, job, job_schema)
        self.assertEqual(meta["realized_bullets"], 0)
        self.assertEqual(meta["claim_strength_violations"], 1)


# ---------------------------------------------------------------------------
# 12. Cloud-path claim-inflation warning (validator.py)
# ---------------------------------------------------------------------------


class TestCloudPathClaimInflationWarning(unittest.TestCase):
    def test_authority_claim_with_no_supporting_evidence_warns(self):
        warning = validator.check_claim_inflation(
            "architected the full platform from scratch",
            profile={"experience_inventory": [{"description": "Used Python for scripting."}]},
        )
        self.assertTrue(warning)
        self.assertIn("claim-strength", warning.lower())

    def test_authority_claim_with_supporting_evidence_does_not_warn(self):
        warning = validator.check_claim_inflation(
            "architected the full platform from scratch",
            profile={"experience_inventory": [{"description": "Architected the payments platform."}]},
        )
        self.assertEqual(warning, "")

    def test_no_authority_claim_never_warns(self):
        warning = validator.check_claim_inflation(
            "used python and configured the pipeline",
            profile={"experience_inventory": []},
        )
        self.assertEqual(warning, "")

    def test_validate_json_fields_surfaces_it_as_a_warning_not_an_error(self):
        """Matches decision #19 (banned words = warnings, not errors) --
        this is advisory since the cloud path has no per-bullet provenance
        to pinpoint the exact offending claim."""
        data = {
            "title": "Eng",
            "summary": "Architected the entire platform.",
            "skills": {"Languages": "Python"},
            "experience": [{"header": "Eng | X", "bullets": ["Architected core systems."]}],
            "projects": [],
            "education": "Some School",
        }
        profile = {"resume_facts": {}, "experience_inventory": [{"description": "Used Python for scripting."}]}
        result = validator.validate_json_fields(data, profile)
        self.assertTrue(result["passed"])  # warning tier, not a hard failure
        self.assertTrue(any("claim-strength" in w.lower() for w in result["warnings"]))


class TestCloudPathAgencyInflationWarning(unittest.TestCase):
    """Companion to TestCloudPathClaimInflationWarning for the separate
    agency axis -- 'led the team' is no longer covered by claim_ceiling's
    authority tier (only 'architected' is), so it needs its own check."""

    def test_team_lead_claim_with_no_supporting_evidence_warns(self):
        warning = validator.check_agency_inflation(
            "led the team that migrated the platform",
            profile={"experience_inventory": [{"description": "Migrated the platform."}]},
        )
        self.assertTrue(warning)
        self.assertIn("agency", warning.lower())

    def test_architected_alone_does_not_license_led_claim(self):
        """The exact gap this axis exists to close: evidence that only
        earns technical-depth authority (architected) must not silence an
        agency-tier (led) claim it doesn't itself support."""
        warning = validator.check_agency_inflation(
            "led the team that architected the platform",
            profile={"experience_inventory": [{"description": "Architected the payments platform."}]},
        )
        self.assertTrue(warning)

    def test_team_lead_claim_with_supporting_evidence_does_not_warn(self):
        warning = validator.check_agency_inflation(
            "led the team that migrated the platform",
            profile={"experience_inventory": [{"description": "Led the platform migration team."}]},
        )
        self.assertEqual(warning, "")

    def test_no_agency_claim_never_warns(self):
        warning = validator.check_agency_inflation(
            "used python and configured the pipeline",
            profile={"experience_inventory": []},
        )
        self.assertEqual(warning, "")

    def test_individual_contributor_language_never_warns(self):
        warning = validator.check_agency_inflation(
            "architected the payments platform",
            profile={"experience_inventory": []},
        )
        self.assertEqual(warning, "")


if __name__ == "__main__":
    unittest.main()
