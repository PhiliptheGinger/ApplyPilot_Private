"""Tests for the 2026-08-23 DEGRADED MODE redesign.

Previously, when every cloud tailoring model was exhausted, tailor_resume()
asked the local (Qwen) model to generate an ENTIRE resume JSON via
local_tailor._build_compact_local_prompt -- slow (large output budget) and
unreliable (a small model reproducing headers/dates/skills it didn't need
to touch). Real-world testing showed ~0.1 jobs/min and 10-15+ minute
individual jobs, vs. 1-2 jobs/min on the normal cloud path.

The new design: Python deterministically parses the ORIGINAL resume (via
the existing, already-trusted scoring/pdf.py parser) and asks the local
model for only a small, schema-bounded REALIZATION step -- short bullet
sentences plus a summary for requirements the deterministic schema layer
(scoring/schemas.py) already marked supported. Everything else (headers,
dates, company names, skills, education) stays verbatim. Still exactly one
LLM call, just a much smaller one.

Covers: prompt construction (bounded, skips the call when nothing is
supported), fuzzy evidence-to-header matching, full composition (happy
path, LLM failure, malformed LLM output, verbatim fallback), and the
tailor_resume() integration (degraded path uses the new composer, not the
old full-generation prompt; exactly one LLM call; validation/judge/
auto-approval still run).

2026-08-23 follow-up fix: a real pipeline run showed the FIRST version of
this redesign still made THREE local calls for a job and produced
assemble_resume_text warnings about a missing "summary"/"skills" key. Root
cause: DEGRADED MODE's gate (`is_local_configured() and not
client_has_cloud_available()`) was checked ONCE, before the retry loop --
using exhaustion state that's necessarily stale at that point (a model is
only marked exhausted as a side effect of actually being tried, inside
client.chat()). The result: attempt 1 saw "cloud available" (nothing tried
yet THIS call), took the normal HEAVY cloud-quality prompt path, which
internally cascaded past newly-exhausted cloud entries all the way to
local anyway -- and since the resulting heavy-prompt response failed
validation, the loop retried the SAME heavy prompt against local AGAIN
(and again), instead of switching to the bounded composer. Fixed by
checking fresh on every loop iteration and having the degraded branch
always return immediately (never loop). Also split the composer into an
explicit three-part shape per the architecture review: build_base_
resume_model (always complete, no LLM), request_local_realization (always
partial, the one LLM call), merge_realization (deterministic, never drops
an untouched section) -- see the new classes below.

2026-08-23 third follow-up (issues A & B): a further real run showed (A)
the FIRST job to hit exhaustion in a process still spent ~5m38s in one
heavy cloud-quality-prompt call that cascaded through to local before
degraded mode kicked in, and (B) when the deterministic schema layer found
ZERO supported requirements, degraded mode kept the resume verbatim but
still ran it through validate_json_fields/assemble_resume_text, which
could fail for reasons unrelated to tailoring (the base resume's own
validity, not something tailoring touched) and reported a plain, confusing
status="failed_validation" with no way to tell "nothing was safely
groundable" apart from "the LLM produced a bad result" or "local
realization failed outright". Fixed: (A) LLMClient.chat() gained
exclude_providers -- tailor_resume()'s heavy prompt now passes
exclude_providers={"local"} and, on RuntimeError (all cloud exhausted),
redirects immediately into degraded mode instead of ever letting the heavy
prompt reach local. (B) tailor_resume()'s _run_degraded_mode now calls
build_base_resume_model/request_local_realization/merge_realization
directly (not the compose_degraded_resume_json orchestrator) so it can see
the raw realization result and branch: realization is None + llm_called
False -> status="no_supported_evidence" (grounding found nothing to
realize, local never called); realization is None + llm_called True ->
status="local_realization_failed" (the local call/response failed);
either way resume_text is returned LITERALLY unchanged and validation/
judge are skipped entirely -- there is nothing new to check. See
TestChatExcludeProviders in test_llm_cascade.py for (A)'s LLMClient-level
tests, and TestNoSupportedEvidenceAndRealizationFailureStatuses below for
(B).
"""

import unittest
from unittest.mock import MagicMock, patch

from applypilot.scoring import local_tailor, schemas

RESUME_TEXT = """Jordan Lee
Software Engineer
jordan@example.com | 555-0100

SUMMARY
An engineer who builds things.

TECHNICAL SKILLS
Languages: Python, Go

EXPERIENCE
Auto Shop Diagnostic Tech | Some Garage
2018 - 2020
- Diagnosed vehicle electrical issues
- Repaired components
- Maintained shop inventory

PROJECTS

EDUCATION
Some University
"""

PROFILE = {
    "skills_boundary": {"languages": ["Python"]},
    "experience_inventory": [
        {
            "name": "Auto Shop Diagnostic Tech",
            "relevance_categories": ["troubleshooting", "root cause"],
            "resume_allowed": True,
            "description": "Diagnosed and repaired vehicle electrical faults.",
        },
    ],
    "project_inventory": [],
    "skills_inventory": [],
    "certifications": [],
}

JOB = {
    "title": "Support Technician",
    "url": "https://example.com/j1",
    "full_description": ("- Troubleshoot hardware and software issues and identify root cause\n"),
}

REALIZATION_RESPONSE = (
    '{"summary": "Support tech with hands-on troubleshooting experience.", '
    '"bullets": [{"evidence": "Auto Shop Diagnostic Tech", '
    '"text": "Diagnosed intermittent electrical faults through systematic '
    "testing, identified root cause, completed repair, and verified reliable "
    'operation."}]}'
)


def _supported_job_schema():
    return schemas.build_job_schema_representation(JOB, PROFILE)


# A job whose requirements share zero literal/synonym overlap with PROFILE's
# evidence -- build_job_schema_representation legitimately finds nothing
# supported for it, exactly the "grounding found no safe evidence" case.
JOB_NO_MATCH = {
    "title": "Marine Biologist",
    "url": "https://example.com/j-no-match",
    "full_description": (
        "- Conduct deep-sea coral reef surveys using submersible equipment\n"
        "- Publish peer-reviewed research on cephalopod behavior\n"
    ),
}


# ---------------------------------------------------------------------------
# _build_realization_prompt
# ---------------------------------------------------------------------------


class TestBuildRealizationPrompt(unittest.TestCase):
    def test_returns_none_when_nothing_supported(self):
        empty_schema = {"requirements": [], "summary_schema": "x"}
        self.assertIsNone(local_tailor._build_realization_prompt(empty_schema))

    def test_returns_none_when_all_requirements_unsupported_or_ambiguous(self):
        job_schema = {
            "requirements": [
                {"supported": False, "schema": None, "ambiguous": True},
                {"supported": False, "schema": None, "ambiguous": False},
            ],
            "summary_schema": "x",
        }
        self.assertIsNone(local_tailor._build_realization_prompt(job_schema))

    def test_builds_a_small_bounded_prompt_for_supported_requirements(self):
        job_schema = _supported_job_schema()
        result = local_tailor._build_realization_prompt(job_schema)
        self.assertIsNotNone(result)
        system, user = result
        self.assertIn("ONE sentence per bullet slot", system)
        self.assertIn("Auto Shop Diagnostic Tech", user)
        self.assertIn("root cause", user)  # exact keyword anchor preserved

    def test_prompt_respects_max_items_cap(self):
        requirements = [
            {
                "requirement": f"Requirement {i}",
                "supported": True,
                "resume_evidence": [f"Evidence {i}"],
                "exact_keywords": [],
                "synonym_concepts": [],
                "schema": {"cognitive_schema": "evidence_claim", "bullet_schema": "action_object_context_outcome"},
                "category_tier": "near_prototype",
            }
            for i in range(10)
        ]
        job_schema = {"requirements": requirements, "summary_schema": "x"}
        _, user = local_tailor._build_realization_prompt(job_schema, max_items=3)
        self.assertEqual(user.count("- evidence:"), 3)


# ---------------------------------------------------------------------------
# _fuzzy_evidence_match
# ---------------------------------------------------------------------------


class TestFuzzyEvidenceMatch(unittest.TestCase):
    def test_matches_when_evidence_name_is_a_substring_of_header(self):
        self.assertTrue(
            local_tailor._fuzzy_evidence_match("Auto Shop Diagnostic Tech | Some Garage", "Auto Shop Diagnostic Tech")
        )

    def test_matches_when_header_is_a_substring_of_evidence_name(self):
        self.assertTrue(local_tailor._fuzzy_evidence_match("Diagnostic Tech", "Auto Shop Diagnostic Tech"))

    def test_does_not_match_unrelated_strings(self):
        self.assertFalse(local_tailor._fuzzy_evidence_match("Warehouse Associate", "Auto Shop Diagnostic Tech"))

    def test_empty_inputs_never_match(self):
        self.assertFalse(local_tailor._fuzzy_evidence_match("", "Auto Shop Diagnostic Tech"))
        self.assertFalse(local_tailor._fuzzy_evidence_match("Auto Shop Diagnostic Tech", ""))


# ---------------------------------------------------------------------------
# compose_degraded_resume_json
# ---------------------------------------------------------------------------


class TestComposeDegradedResumeJson(unittest.TestCase):
    def test_happy_path_splices_realized_bullet_and_keeps_rest_verbatim(self):
        client = MagicMock()
        client.chat.return_value = REALIZATION_RESPONSE
        job_schema = _supported_job_schema()

        data, meta = local_tailor.compose_degraded_resume_json(
            client,
            RESUME_TEXT,
            JOB,
            PROFILE,
            job_schema,
        )

        self.assertEqual(meta["tier"], "degraded_structured")
        self.assertTrue(meta["llm_called"])
        self.assertEqual(meta["realized_bullets"], 1)

        self.assertEqual(data["title"], "Support Technician")
        self.assertIn("hands-on troubleshooting", data["summary"])
        self.assertEqual(data["skills"], {"Languages": "Python, Go"})  # verbatim from resume

        entry = data["experience"][0]
        self.assertEqual(entry["header"], "Auto Shop Diagnostic Tech | Some Garage")  # verbatim
        self.assertEqual(entry["subtitle"], "2018 - 2020")  # verbatim
        self.assertIn("Diagnosed intermittent electrical faults", entry["bullets"][0])
        # Original bullets still present after the realized one, capped at 4 total.
        self.assertIn("Diagnosed vehicle electrical issues", entry["bullets"])
        self.assertLessEqual(len(entry["bullets"]), 4)

        self.assertEqual(data["education"], "Some University")  # verbatim

    def test_exactly_one_llm_call(self):
        client = MagicMock()
        client.chat.return_value = REALIZATION_RESPONSE
        job_schema = _supported_job_schema()
        local_tailor.compose_degraded_resume_json(client, RESUME_TEXT, JOB, PROFILE, job_schema)
        self.assertEqual(client.chat.call_count, 1)

    def test_no_supported_requirements_skips_the_llm_call_entirely(self):
        client = MagicMock()
        empty_schema = {"requirements": [], "summary_schema": "x"}
        data, meta = local_tailor.compose_degraded_resume_json(
            client,
            RESUME_TEXT,
            JOB,
            PROFILE,
            empty_schema,
        )
        client.chat.assert_not_called()
        self.assertFalse(meta["llm_called"])
        self.assertEqual(meta["realized_bullets"], 0)
        # Verbatim original content -- degraded gracefully, not empty/broken.
        self.assertEqual(data["summary"], "An engineer who builds things.")
        self.assertEqual(
            data["experience"][0]["bullets"],
            ["Diagnosed vehicle electrical issues", "Repaired components", "Maintained shop inventory"],
        )

    def test_llm_call_failure_degrades_to_verbatim_original_not_a_crash(self):
        client = MagicMock()
        client.chat.side_effect = TimeoutError("local model timed out")
        job_schema = _supported_job_schema()

        data, meta = local_tailor.compose_degraded_resume_json(
            client,
            RESUME_TEXT,
            JOB,
            PROFILE,
            job_schema,
        )  # must not raise

        self.assertTrue(meta["llm_called"])
        self.assertEqual(meta["realized_bullets"], 0)
        self.assertEqual(data["summary"], "An engineer who builds things.")  # verbatim fallback
        self.assertEqual(
            data["experience"][0]["bullets"],
            ["Diagnosed vehicle electrical issues", "Repaired components", "Maintained shop inventory"],
        )

    def test_malformed_llm_json_degrades_to_verbatim_not_a_crash(self):
        client = MagicMock()
        client.chat.return_value = "not json at all, sorry"
        job_schema = _supported_job_schema()

        data, meta = local_tailor.compose_degraded_resume_json(
            client,
            RESUME_TEXT,
            JOB,
            PROFILE,
            job_schema,
        )  # must not raise

        self.assertEqual(meta["realized_bullets"], 0)
        self.assertEqual(data["summary"], "An engineer who builds things.")

    def test_falls_back_to_profile_skills_boundary_when_skills_section_unparseable(self):
        client = MagicMock()
        client.chat.return_value = REALIZATION_RESPONSE
        resume_no_skills = RESUME_TEXT.replace(
            "TECHNICAL SKILLS\nLanguages: Python, Go\n\n",
            "TECHNICAL SKILLS\n\n",
        )
        job_schema = _supported_job_schema()

        data, _meta = local_tailor.compose_degraded_resume_json(
            client,
            resume_no_skills,
            JOB,
            PROFILE,
            job_schema,
        )
        self.assertIn("Languages", data["skills"])
        self.assertIn("Python", data["skills"]["Languages"])

    def test_never_invents_bullets_beyond_what_the_model_returned(self):
        """A schema with 2 supported requirements but a model response that
        only realizes 1 of them must not fabricate the second."""
        client = MagicMock()
        client.chat.return_value = REALIZATION_RESPONSE  # only realizes 1 evidence
        job_schema = _supported_job_schema()
        job_schema["requirements"].append(
            {
                "requirement": "Provide customer service",
                "supported": True,
                "resume_evidence": ["Customer Service Skill"],
                "exact_keywords": ["customer"],
                "synonym_concepts": [],
                "schema": {"cognitive_schema": "evidence_claim", "bullet_schema": "action_object_context_outcome"},
            }
        )
        data, meta = local_tailor.compose_degraded_resume_json(
            client,
            RESUME_TEXT,
            JOB,
            PROFILE,
            job_schema,
        )
        self.assertEqual(meta["realized_bullets"], 1)  # model only returned 1
        # No entry in the resume references "Customer Service Skill" content
        # that was never actually realized.
        all_bullets = [b for e in data["experience"] for b in e["bullets"]]
        self.assertFalse(any("customer" in b.lower() for b in all_bullets))


# ---------------------------------------------------------------------------
# tailor_resume() integration
# ---------------------------------------------------------------------------


class TestTailorResumeDegradedModeIntegration(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def _degraded_client(self, chat_return=REALIZATION_RESPONSE):
        client = MagicMock()
        client.chat.return_value = chat_return
        client.has_cloud_available = lambda: False  # triggers DEGRADED MODE
        return client

    def test_degraded_mode_uses_the_new_composer_not_full_generation(self):
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client()
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            tailored, report = tailor_mod.tailor_resume(RESUME_TEXT, JOB, PROFILE, max_retries=2)

        self.assertIn("degraded_mode", report)
        self.assertEqual(report["degraded_mode"]["tier"], "degraded_structured")
        self.assertIn("Diagnosed intermittent electrical faults", tailored)
        self.assertIn(report["status"], ("approved", "failed_validation"))

    def test_degraded_mode_makes_exactly_one_local_realization_call_not_a_retry_loop(self):
        """The old full-generation DEGRADED MODE could burn max_retries+1
        LLM calls against the local model; the new composer makes exactly
        one realization call to it, regardless of max_retries. (The
        separate judge stage -- a different client/model in production --
        is exercised independently below and isn't what this test guards.)"""
        from applypilot.scoring import tailor as tailor_mod

        tailor_client = self._degraded_client()
        judge_client = MagicMock()
        judge_client.chat.return_value = "VERDICT: PASS\nISSUES: none"

        def _fake_get_stage_client(stage, *, quality):
            return judge_client if stage == "judge" else tailor_client

        with (
            patch.object(tailor_mod, "get_stage_client", side_effect=_fake_get_stage_client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            tailor_mod.tailor_resume(RESUME_TEXT, JOB, PROFILE, max_retries=3)

        self.assertEqual(tailor_client.chat.call_count, 1)

    def test_degraded_mode_result_still_goes_through_validation(self):
        """A response missing required structure must still be caught by
        the existing validator, not silently shipped."""
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client()
        resume_without_education = RESUME_TEXT.split("EDUCATION")[0]  # no EDUCATION section at all
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            _tailored, report = tailor_mod.tailor_resume(
                resume_without_education,
                JOB,
                PROFILE,
                max_retries=2,
            )

        self.assertEqual(report["status"], "failed_validation")
        self.assertIn("education", " ".join(report["validator"]["errors"]).lower())

    def test_normal_cloud_path_unaffected_when_cloud_available(self):
        """Sanity check: when cloud IS available, tailor_resume must not
        touch the degraded-mode composer at all."""
        from applypilot.scoring import tailor as tailor_mod

        client = MagicMock()
        client.chat.return_value = (
            '{"title":"Tech","summary":"S","skills":{"Languages":"Python"},'
            '"experience":[],"projects":[],"education":[]}'
        )
        client.has_cloud_available = lambda: True

        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
            patch.object(tailor_mod, "extract_facts_from_resume_json", return_value=set()),
            patch.object(tailor_mod, "is_auto_approvable", return_value=True),
        ):
            _tailored, report = tailor_mod.tailor_resume(RESUME_TEXT, JOB, PROFILE, max_retries=0)

        self.assertNotIn("degraded_mode", report)


# ---------------------------------------------------------------------------
# 2026-08-23 follow-up: exact bug reproduction + the A/B/C architecture split
# ---------------------------------------------------------------------------


class TestStaleExhaustionCheckFix(unittest.TestCase):
    """Reproduces the exact reported bug and proves the fix."""

    def setUp(self):
        schemas.clear_schema_cache()

    def test_cloud_exhaustion_discovered_mid_job_never_reaches_local_with_the_heavy_prompt(self):
        """has_cloud_available() looks True at the very start of
        tailor_resume (nothing tried yet in THIS call) -- attempt 1 takes
        the heavy cloud-quality prompt. 2026-08-23 second fix: that attempt
        now passes exclude_providers={"local"}, so client.chat() itself can
        never let the heavy prompt fall through to local -- it fails FAST
        with a RuntimeError once every cloud provider is exhausted, which
        tailor_resume catches and redirects straight into the bounded
        degraded composer within the SAME attempt. Expected: the heavy call
        happens (and is given exclude_providers=local), it raises, and
        exactly ONE small bounded realization call follows -- the heavy
        prompt is NEVER sent with a chance of reaching local at all."""
        from applypilot.scoring import tailor as tailor_mod

        # First check (before attempt 1's heavy call) says "available" --
        # nothing tried yet. Second check (inside the except-RuntimeError
        # handler, after the cloud-only attempt just exhausted everything)
        # correctly reflects the now-real exhausted state.
        availability = iter([True, False, False, False])
        calls: list[dict] = []

        def fake_chat(messages, max_tokens=None, temperature=None, exclude_providers=None):
            calls.append({"max_tokens": max_tokens, "exclude_providers": exclude_providers})
            if exclude_providers and "local" in exclude_providers:
                # Cloud-only attempt: every cloud provider is exhausted --
                # must fail fast, never silently fall through to local.
                raise RuntimeError("All LLM providers are on quota cooldown (min wait: 1.0h).")
            return REALIZATION_RESPONSE

        tailor_client = MagicMock()
        tailor_client.has_cloud_available.side_effect = lambda: next(availability, False)
        tailor_client.chat.side_effect = fake_chat
        # A separate judge client (matches production: get_stage_client("judge",
        # quality=False) is a different fast-tier client than "tailor") so its
        # calls don't get miscounted as extra tailor/local calls.
        judge_client = MagicMock()
        judge_client.chat.return_value = "VERDICT: PASS\nISSUES: none"

        def _fake_get_stage_client(stage, *, quality):
            return judge_client if stage == "judge" else tailor_client

        with (
            patch.object(tailor_mod, "get_stage_client", side_effect=_fake_get_stage_client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            tailor_mod.tailor_resume(RESUME_TEXT, JOB, PROFILE, max_retries=3)

        cloud_only_calls = [c for c in calls if c["exclude_providers"] and "local" in c["exclude_providers"]]
        bounded_calls = [c for c in calls if not c["exclude_providers"]]
        self.assertEqual(
            len(cloud_only_calls),
            1,
            "expected exactly 1 cloud-only discovery attempt (fails fast, never reaches local)",
        )
        self.assertEqual(
            len(bounded_calls), 1, f"expected exactly 1 bounded realization call, got {len(bounded_calls)}"
        )
        self.assertEqual(tailor_client.chat.call_count, 2)

    def test_already_known_exhaustion_skips_the_heavy_call_entirely(self):
        """When exhaustion is ALREADY known at the start (the normal
        steady-state case for the 2nd+ job in a run after the 1st job
        discovered it), degraded mode fires on attempt 1 with ZERO heavy
        calls -- matches the observed correct behavior for the 3rd job in
        the real run that triggered this investigation."""
        from applypilot.scoring import tailor as tailor_mod

        tailor_client = MagicMock()
        tailor_client.has_cloud_available.return_value = False
        tailor_client.chat.return_value = REALIZATION_RESPONSE
        judge_client = MagicMock()
        judge_client.chat.return_value = "VERDICT: PASS\nISSUES: none"

        def _fake_get_stage_client(stage, *, quality):
            return judge_client if stage == "judge" else tailor_client

        with (
            patch.object(tailor_mod, "get_stage_client", side_effect=_fake_get_stage_client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            tailor_mod.tailor_resume(RESUME_TEXT, JOB, PROFILE, max_retries=3)

        self.assertEqual(tailor_client.chat.call_count, 1)
        called_max_tokens = tailor_client.chat.call_args.kwargs.get("max_tokens")
        self.assertLessEqual(called_max_tokens, 1000)


class TestBuildBaseResumeModel(unittest.TestCase):
    """build_base_resume_model: always complete, no LLM call."""

    def test_always_has_every_required_key(self):
        model = local_tailor.build_base_resume_model(RESUME_TEXT, PROFILE)
        for key in ("title", "summary", "skills", "experience", "projects", "education"):
            self.assertIn(key, model)

    def test_summary_and_skills_come_from_the_original_resume(self):
        model = local_tailor.build_base_resume_model(RESUME_TEXT, PROFILE)
        self.assertEqual(model["summary"], "An engineer who builds things.")
        self.assertEqual(model["skills"], {"Languages": "Python, Go"})

    def test_falls_back_to_profile_skills_boundary_when_section_missing(self):
        resume_no_skills = RESUME_TEXT.replace(
            "TECHNICAL SKILLS\nLanguages: Python, Go\n\n",
            "TECHNICAL SKILLS\n\n",
        )
        model = local_tailor.build_base_resume_model(resume_no_skills, PROFILE)
        self.assertIn("Languages", model["skills"])

    def test_empty_resume_text_still_returns_a_complete_dict(self):
        """Never raises, never returns a partial shape, even for degenerate input."""
        model = local_tailor.build_base_resume_model("", PROFILE)
        for key in ("title", "summary", "skills", "experience", "projects", "education"):
            self.assertIn(key, model)


class TestBuildBaseResumeModelFallsBackToProfileStructure(unittest.TestCase):
    """2026-09-04 regression: real production degraded-mode calls pass
    resume_router.load_resume_text_for_job's "CANONICAL PROFILE REFERENCE"
    rendering as resume_text -- a flat, profile-derived text with no
    EXPERIENCE/EDUCATION/SUMMARY section headers at all. parse_resume/
    parse_entries found nothing to parse and silently returned EMPTY
    experience/education every time, confirmed against a real production
    call (0 experience entries, empty education string) -- meaning
    degraded mode has been merging realized bullets onto a base resume
    with no work history at all, independent of whether the local model's
    realization step itself worked. Fixed by falling back to constructing
    entries directly from the STRUCTURED profile (already a function
    parameter) when text-parsing yields nothing -- same pattern already
    established for `skills` just above."""

    FLAT_PROFILE_TEXT = (
        "CANONICAL PROFILE REFERENCE\n"
        "education: Some University\n"
        "- Bachelor of Arts in Media Studies\n"
        "experience_inventory: Auto Shop Diagnostic Tech\n"
        "- Diagnosed and repaired vehicle electrical faults.\n"
    )

    def test_experience_falls_back_to_profile_when_no_experience_section_parses(self):
        model = local_tailor.build_base_resume_model(self.FLAT_PROFILE_TEXT, PROFILE)
        self.assertEqual(len(model["experience"]), 1)
        entry = model["experience"][0]
        self.assertEqual(entry["title"], "Auto Shop Diagnostic Tech")
        self.assertTrue(entry["bullets"])

    def test_education_falls_back_to_profile_when_no_education_section_parses(self):
        profile_with_education = {
            **PROFILE,
            "education": [
                {"official_degree": "Bachelor of Arts in Media Studies", "institution": "Some University", "start_year": 2019, "end_year": 2022}
            ],
        }
        model = local_tailor.build_base_resume_model(self.FLAT_PROFILE_TEXT, profile_with_education)
        self.assertIn("Media Studies", model["education"])
        self.assertIn("Some University", model["education"])

    def test_fallback_experience_entry_still_matches_realization_by_fuzzy_name(self):
        """The whole point: a realized bullet keyed by evidence name must
        still land on the right entry after the fallback construction, not
        just produce non-empty entries that realization can't attach to."""
        model = local_tailor.build_base_resume_model(self.FLAT_PROFILE_TEXT, PROFILE)
        realization = {"summary": None, "bullets": {"Auto Shop Diagnostic Tech": "Realized bullet text."}}
        merged = local_tailor.merge_realization(model, realization, JOB)
        entry = next(e for e in merged["experience"] if e["header"] == "Auto Shop Diagnostic Tech")
        self.assertIn("Realized bullet text.", entry["bullets"])

    def test_project_entries_fall_back_to_profile_factual_concepts(self):
        profile_with_project = {
            **PROFILE,
            "project_inventory": [
                {"name": "Standup-OCR", "resume_allowed": True, "factual_concepts": ["Python", "OCR tooling"]}
            ],
        }
        model = local_tailor.build_base_resume_model(self.FLAT_PROFILE_TEXT, profile_with_project)
        self.assertEqual(len(model["projects"]), 1)
        self.assertEqual(model["projects"][0]["title"], "Standup-OCR")
        self.assertIn("Python", model["projects"][0]["bullets"])

    def test_real_resume_document_still_parses_normally_unaffected_by_the_fallback(self):
        """The fallback must never fire when the text genuinely IS a
        parseable resume document -- RESUME_TEXT's real EXPERIENCE section
        must still win over the profile fallback."""
        model = local_tailor.build_base_resume_model(RESUME_TEXT, PROFILE)
        self.assertEqual(len(model["experience"]), 1)
        self.assertIn("Some Garage", model["experience"][0]["title"])
        self.assertEqual(model["experience"][0]["subtitle"], "2018 - 2020")


class TestMergeRealization(unittest.TestCase):
    """merge_realization: deterministic, never drops an untouched section."""

    def setUp(self):
        self.base = local_tailor.build_base_resume_model(RESUME_TEXT, PROFILE)

    def test_none_realization_returns_the_complete_base_resume_unchanged(self):
        data = local_tailor.merge_realization(self.base, None, JOB)
        for key in ("title", "summary", "skills", "experience", "projects", "education"):
            self.assertIn(key, data)
        self.assertEqual(data["summary"], "An engineer who builds things.")
        self.assertEqual(data["skills"], {"Languages": "Python, Go"})
        self.assertEqual(
            data["experience"][0]["bullets"],
            ["Diagnosed vehicle electrical issues", "Repaired components", "Maintained shop inventory"],
        )

    def test_partial_realization_with_only_summary_leaves_bullets_skills_education_intact(self):
        realization = {"summary": "A tailored summary.", "bullets": {}}
        data = local_tailor.merge_realization(self.base, realization, JOB)
        self.assertEqual(data["summary"], "A tailored summary.")
        self.assertEqual(data["skills"], {"Languages": "Python, Go"})  # untouched
        self.assertEqual(
            data["experience"][0]["bullets"],
            ["Diagnosed vehicle electrical issues", "Repaired components", "Maintained shop inventory"],
        )  # untouched -- no bullet realization was given
        self.assertEqual(data["education"], "Some University")  # untouched

    def test_partial_realization_with_only_bullets_leaves_summary_intact(self):
        realization = {"summary": None, "bullets": {"Auto Shop Diagnostic Tech": "A new bullet."}}
        data = local_tailor.merge_realization(self.base, realization, JOB)
        self.assertEqual(data["summary"], "An engineer who builds things.")  # untouched
        self.assertEqual(data["experience"][0]["bullets"][0], "A new bullet.")

    def test_result_always_has_every_key_regardless_of_realization_shape(self):
        for realization in (None, {}, {"summary": "X"}, {"bullets": {"A": "B"}}):
            data = local_tailor.merge_realization(self.base, realization, JOB)
            for key in ("title", "summary", "skills", "experience", "projects", "education"):
                self.assertIn(key, data, f"missing {key!r} for realization={realization!r}")

    def test_job_title_overrides_base_title(self):
        data = local_tailor.merge_realization(self.base, None, JOB)
        self.assertEqual(data["title"], JOB["title"])


class TestRequestLocalRealization(unittest.TestCase):
    """request_local_realization: always partial, the one LLM call, with
    diagnosable size/timing metadata."""

    def setUp(self):
        schemas.clear_schema_cache()

    def test_returns_none_and_meta_when_nothing_supported(self):
        client = MagicMock()
        realization, meta = local_tailor.request_local_realization(client, JOB, {"requirements": []})
        self.assertIsNone(realization)
        self.assertFalse(meta["llm_called"])
        client.chat.assert_not_called()

    def test_meta_reports_prompt_size_and_max_tokens(self):
        client = MagicMock()
        client.chat.return_value = REALIZATION_RESPONSE
        job_schema = _supported_job_schema()
        _realization, meta = local_tailor.request_local_realization(client, JOB, job_schema)
        self.assertTrue(meta["llm_called"])
        self.assertGreater(meta["prompt_chars"], 0)
        self.assertEqual(meta["max_tokens"], 600)

    def test_realization_is_partial_shape_not_a_full_resume(self):
        client = MagicMock()
        client.chat.return_value = REALIZATION_RESPONSE
        job_schema = _supported_job_schema()
        realization, _meta = local_tailor.request_local_realization(client, JOB, job_schema)
        self.assertEqual(set(realization.keys()), {"summary", "bullets"})
        self.assertNotIn("experience", realization)
        self.assertNotIn("skills", realization)
        self.assertNotIn("education", realization)

    def test_bullet_with_ungrounded_agency_claim_is_dropped(self):
        """PROFILE's only evidence for this slot ('Diagnosed and repaired
        vehicle electrical faults') earns no agency tier above
        individual_contributor -- a realized bullet claiming 'led the team'
        must be dropped, mirroring the existing claim-strength/causal
        drop behavior."""
        client = MagicMock()
        client.chat.return_value = (
            '{"summary": null, "bullets": [{"evidence": "Auto Shop Diagnostic Tech", '
            '"text": "Led the team that diagnosed vehicle electrical faults."}]}'
        )
        job_schema = _supported_job_schema()
        realization, meta = local_tailor.request_local_realization(client, JOB, job_schema)
        self.assertIsNone(realization)
        self.assertGreaterEqual(meta["claim_strength_violations"], 1)

    def test_bullet_with_fabricated_metric_is_dropped(self):
        """The evidence text has no number in it -- a realized bullet
        introducing one ('by 40%') must be dropped even though the verb
        itself is grounded."""
        client = MagicMock()
        client.chat.return_value = (
            '{"summary": null, "bullets": [{"evidence": "Auto Shop Diagnostic Tech", '
            '"text": "Diagnosed vehicle electrical faults, reducing repair time by 40%."}]}'
        )
        job_schema = _supported_job_schema()
        realization, meta = local_tailor.request_local_realization(client, JOB, job_schema)
        self.assertIsNone(realization)
        self.assertGreaterEqual(meta["claim_strength_violations"], 1)

    def test_grounded_bullet_with_no_agency_or_metric_issue_survives(self):
        client = MagicMock()
        client.chat.return_value = REALIZATION_RESPONSE
        job_schema = _supported_job_schema()
        realization, meta = local_tailor.request_local_realization(client, JOB, job_schema)
        self.assertIsNotNone(realization)
        self.assertEqual(meta["claim_strength_violations"], 0)

    def test_profile_real_metrics_are_accepted_as_known_numbers(self):
        client = MagicMock()
        client.chat.return_value = (
            '{"summary": null, "bullets": [{"evidence": "Auto Shop Diagnostic Tech", '
            '"text": "Diagnosed vehicle electrical faults, reducing repair time by 40%."}]}'
        )
        job_schema = _supported_job_schema()
        profile_with_metric = dict(PROFILE, resume_facts={"real_metrics": ["40%"]})
        realization, meta = local_tailor.request_local_realization(
            client,
            JOB,
            job_schema,
            profile_with_metric,
        )
        self.assertIsNotNone(realization)
        self.assertEqual(meta["claim_strength_violations"], 0)


class TestPromptBoundedness(unittest.TestCase):
    """Guards against silently reintroducing the full resume/job posting
    into the realization prompt in a future change."""

    def test_prompt_stays_small_even_with_many_supported_requirements(self):
        requirements = [
            {
                "requirement": f"Requirement number {i} about something specific",
                "supported": True,
                "resume_evidence": [f"Evidence {i}"],
                "exact_keywords": ["term"],
                "synonym_concepts": [],
                "schema": {"cognitive_schema": "evidence_claim", "bullet_schema": "action_object_context_outcome"},
                "category_tier": "prototype",
            }
            for i in range(20)  # far more than the cap
        ]
        job_schema = {"requirements": requirements, "summary_schema": "identity_domain_strength_evidence_value"}
        system, user = local_tailor._build_realization_prompt(job_schema)
        # Bounded to _DEGRADED_MAX_REALIZED_BULLETS regardless of how many
        # supported requirements exist.
        self.assertEqual(user.count("- evidence:"), local_tailor._DEGRADED_MAX_REALIZED_BULLETS)
        # A generous but real ceiling -- catches a regression that starts
        # dumping the full resume/job description into this prompt again.
        self.assertLess(len(system) + len(user), 3000)

    def test_prompt_never_contains_a_resume_or_job_description_dump(self):
        """The realization prompt must reference only requirement text/
        evidence names/schema names -- never the raw resume or job
        description text, which is exactly what the old full-generation
        prompt sent and what made it slow."""
        job_schema = _supported_job_schema()
        _system, user = local_tailor._build_realization_prompt(job_schema)
        self.assertNotIn("Auto Shop Diagnostic Tech | Some Garage", user)  # a resume header line
        self.assertNotIn(RESUME_TEXT.strip(), user)


class TestMalformedRealizationFallsBackToCompleteBase(unittest.TestCase):
    def setUp(self):
        schemas.clear_schema_cache()

    def test_malformed_json_falls_back_to_complete_base_not_a_partial_object(self):
        client = MagicMock()
        client.chat.return_value = "this is not json"
        job_schema = _supported_job_schema()
        data, _meta = local_tailor.compose_degraded_resume_json(client, RESUME_TEXT, JOB, PROFILE, job_schema)
        for key in ("title", "summary", "skills", "experience", "projects", "education"):
            self.assertIn(key, data)
            self.assertTrue(data[key] or key == "projects")  # projects legitimately empty here
        self.assertEqual(data["summary"], "An engineer who builds things.")

    def test_response_with_no_recognizable_fields_falls_back_to_complete_base(self):
        client = MagicMock()
        client.chat.return_value = '{"unrelated_field": "nothing useful here"}'
        job_schema = _supported_job_schema()
        data, meta = local_tailor.compose_degraded_resume_json(client, RESUME_TEXT, JOB, PROFILE, job_schema)
        self.assertEqual(meta["realized_bullets"], 0)
        self.assertEqual(data["summary"], "An engineer who builds things.")
        self.assertEqual(
            data["experience"][0]["bullets"],
            ["Diagnosed vehicle electrical issues", "Repaired components", "Maintained shop inventory"],
        )


class TestNoSupportedEvidenceAndRealizationFailureStatuses(unittest.TestCase):
    """2026-08-23 (Issue B): distinguishes three degraded-mode outcomes that
    used to collapse into the same status="failed_validation":
      1. no_supported_evidence -- the deterministic schema layer found
         zero supported requirements; local Qwen is never called.
      2. local_realization_failed -- Qwen WAS called (supported
         requirements existed) but its call/response failed or was empty.
      3. failed_validation -- Qwen returned something usable, but the
         MERGED result failed validate_json_fields (a genuine problem with
         the tailoring attempt, distinct from cases 1/2 where nothing was
         even attempted/usable).
    In cases 1 and 2, resume_text is returned LITERALLY unchanged and
    validation/judge are skipped -- there is nothing new to check."""

    def setUp(self):
        schemas.clear_schema_cache()

    def _degraded_client(self, chat_side_effect=None, chat_return=None):
        client = MagicMock()
        client.has_cloud_available = lambda: False  # already-known-exhausted (steady state)
        if chat_side_effect is not None:
            client.chat.side_effect = chat_side_effect
        else:
            client.chat.return_value = chat_return
        return client

    # --- 1. no_supported_evidence -------------------------------------

    def test_no_supported_requirements_yields_no_supported_evidence_status(self):
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client(chat_return=REALIZATION_RESPONSE)
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            _tailored, report = tailor_mod.tailor_resume(RESUME_TEXT, JOB_NO_MATCH, PROFILE, max_retries=2)

        self.assertEqual(report["status"], "no_supported_evidence")
        self.assertNotEqual(report["status"], "failed_validation")

    def test_no_supported_requirements_makes_no_local_realization_call(self):
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client(chat_return=REALIZATION_RESPONSE)
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            tailor_mod.tailor_resume(RESUME_TEXT, JOB_NO_MATCH, PROFILE, max_retries=2)

        client.chat.assert_not_called()

    def test_no_supported_requirements_preserves_the_resume_literally_unchanged(self):
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client(chat_return=REALIZATION_RESPONSE)
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            tailored, _report = tailor_mod.tailor_resume(RESUME_TEXT, JOB_NO_MATCH, PROFILE, max_retries=2)

        self.assertEqual(tailored, RESUME_TEXT)  # byte-for-byte, not re-parsed/reassembled

    def test_no_supported_requirements_skips_validation_entirely(self):
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client(chat_return=REALIZATION_RESPONSE)
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            _tailored, report = tailor_mod.tailor_resume(RESUME_TEXT, JOB_NO_MATCH, PROFILE, max_retries=2)

        self.assertIsNone(report["validator"])
        self.assertIsNone(report["judge"])

    def test_no_supported_requirements_does_not_loop(self):
        """No retry cycle -- degraded mode returns on the first attempt
        that reaches it, regardless of max_retries."""
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client(chat_return=REALIZATION_RESPONSE)
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            _tailored, report = tailor_mod.tailor_resume(RESUME_TEXT, JOB_NO_MATCH, PROFILE, max_retries=5)

        self.assertEqual(report["attempts"], 1)
        client.chat.assert_not_called()

    # --- 2. local_realization_failed ------------------------------------

    def test_realization_exception_yields_local_realization_failed_status(self):
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client(chat_side_effect=TimeoutError("local model timed out"))
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            tailored, report = tailor_mod.tailor_resume(RESUME_TEXT, JOB, PROFILE, max_retries=2)

        self.assertEqual(report["status"], "local_realization_failed")
        self.assertNotEqual(report["status"], "no_supported_evidence")
        self.assertNotEqual(report["status"], "failed_validation")
        self.assertEqual(tailored, RESUME_TEXT)

    def test_malformed_realization_yields_local_realization_failed_status(self):
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client(chat_return="not json at all")
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            tailored, report = tailor_mod.tailor_resume(RESUME_TEXT, JOB, PROFILE, max_retries=2)

        self.assertEqual(report["status"], "local_realization_failed")
        self.assertEqual(tailored, RESUME_TEXT)

    def test_realization_failure_makes_exactly_one_local_call_and_does_not_loop(self):
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client(chat_side_effect=TimeoutError("timed out"))
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            _tailored, report = tailor_mod.tailor_resume(RESUME_TEXT, JOB, PROFILE, max_retries=5)

        self.assertEqual(client.chat.call_count, 1)
        self.assertEqual(report["attempts"], 1)

    def test_realization_failure_skips_validation_entirely(self):
        from applypilot.scoring import tailor as tailor_mod

        client = self._degraded_client(chat_side_effect=TimeoutError("timed out"))
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            _tailored, report = tailor_mod.tailor_resume(RESUME_TEXT, JOB, PROFILE, max_retries=2)

        self.assertIsNone(report["validator"])

    # --- 3. distinct from a genuine failed_validation --------------------

    def test_successful_realization_still_goes_through_validation_and_can_fail_it(self):
        """Contrast case: when realization DOES produce usable content but
        the merged result is genuinely invalid, status must still be
        failed_validation -- proves cases 1/2 aren't just masking real
        validation failures."""
        from applypilot.scoring import tailor as tailor_mod

        resume_without_education = RESUME_TEXT.split("EDUCATION")[0]
        client = self._degraded_client(chat_return=REALIZATION_RESPONSE)
        with (
            patch.object(tailor_mod, "get_stage_client", return_value=client),
            patch.object(tailor_mod, "is_local_configured", return_value=True),
        ):
            _tailored, report = tailor_mod.tailor_resume(
                resume_without_education,
                JOB,
                PROFILE,
                max_retries=2,
            )

        self.assertEqual(report["status"], "failed_validation")
        self.assertIsNotNone(report["validator"])
        client.chat.assert_called_once()


class TestBuildPoolRealization(unittest.TestCase):
    """2026-09-04: deterministic counterpart to request_local_realization
    -- selects from an already-generated, already-diversity-filtered
    sentence pool instead of calling an LLM. Same output contract as
    request_local_realization ({"bullets": {...}}), so it plugs into the
    existing, unchanged merge_realization. Embeddings are mocked
    throughout -- no real Ollama call required, matching the convention
    in test_local_llm.py's semantic_match-adjacent tests."""

    def _patched_embed(self, dim=64):
        """Fixed-size hashed bag-of-words fake embedding. build_pool_
        realization calls embed_texts separately for the pool and for each
        requirement (different calls), so a fake embedding with a
        dynamically-growing per-call vocabulary produces DIFFERENT vector
        lengths across calls -- cosine_similarity's own contract returns
        0.0 for any length mismatch, silently making every comparison tie
        at 0.0 (an earlier version of this test had exactly that bug: it
        always "selected" the first pool sentence regardless of wording,
        because a stable sort over all-zero scores just preserves input
        order). Fixed dimensionality (hash each word into one of `dim`
        buckets by character-ordinal sum, not a growing index) guarantees
        every call produces same-length vectors regardless of call order,
        so real word overlap between a requirement and a pool sentence
        actually shows up as nonzero cosine similarity."""

        def _bucket(word: str) -> int:
            return sum(ord(c) for c in word) % dim

        def _embed(texts):
            vectors = []
            for t in texts:
                vec = [0.0] * dim
                for w in t.lower().split():
                    vec[_bucket(w)] = 1.0
                vectors.append(vec)
            return vectors

        return _embed

    def test_no_pools_returns_none(self):
        schema = {"requirements": [{"supported": True, "requirement": "x", "resume_evidence": ["Acme"]}]}
        self.assertIsNone(local_tailor.build_pool_realization(schema, None))
        self.assertIsNone(local_tailor.build_pool_realization(schema, {}))

    def test_unsupported_requirement_skipped(self):
        schema = {"requirements": [{"supported": False, "requirement": "handle customer calls", "resume_evidence": ["Acme"]}]}
        pools = {"Acme": ["Handled customer calls daily."]}
        with patch("applypilot.scoring.semantic_match.embed_texts", side_effect=self._patched_embed()):
            self.assertIsNone(local_tailor.build_pool_realization(schema, pools))

    def test_evidence_with_no_pool_is_skipped(self):
        schema = {"requirements": [{"supported": True, "requirement": "handle customer calls", "resume_evidence": ["OtherCo"]}]}
        pools = {"Acme": ["Handled customer calls daily."]}
        with patch("applypilot.scoring.semantic_match.embed_texts", side_effect=self._patched_embed()):
            self.assertIsNone(local_tailor.build_pool_realization(schema, pools))

    def test_selects_best_matching_pool_sentence(self):
        schema = {
            "requirements": [
                {"supported": True, "requirement": "communicate clearly with customers", "resume_evidence": ["Acme"]},
            ]
        }
        pools = {
            "Acme": [
                "Followed established company protocols during installation work.",
                "Maintained clear communication with customers throughout service visits.",
            ]
        }
        with patch("applypilot.scoring.semantic_match.embed_texts", side_effect=self._patched_embed()):
            result = local_tailor.build_pool_realization(schema, pools)
        self.assertIsNotNone(result)
        self.assertEqual(result["bullets"]["Acme"], "Maintained clear communication with customers throughout service visits.")

    def test_different_requirement_wording_selects_different_sentence(self):
        """The actual point of this function: two differently-worded
        requirements against the SAME pool pick DIFFERENT sentences."""
        pools = {
            "Acme": [
                "Followed established company protocols during installation work.",
                "Maintained clear communication with customers throughout service visits.",
            ]
        }
        schema_communication = {
            "requirements": [{"supported": True, "requirement": "communicate clearly with customers", "resume_evidence": ["Acme"]}]
        }
        schema_protocols = {
            "requirements": [{"supported": True, "requirement": "follow established company protocols", "resume_evidence": ["Acme"]}]
        }
        with patch("applypilot.scoring.semantic_match.embed_texts", side_effect=self._patched_embed()):
            result_a = local_tailor.build_pool_realization(schema_communication, pools)
            result_b = local_tailor.build_pool_realization(schema_protocols, pools)
        self.assertNotEqual(result_a["bullets"]["Acme"], result_b["bullets"]["Acme"])

    def test_first_matched_requirement_wins_for_an_entry_not_overwritten_by_a_later_one(self):
        schema = {
            "requirements": [
                {"supported": True, "requirement": "communicate clearly with customers", "resume_evidence": ["Acme"]},
                {"supported": True, "requirement": "follow established company protocols", "resume_evidence": ["Acme"]},
            ]
        }
        pools = {
            "Acme": [
                "Followed established company protocols during installation work.",
                "Maintained clear communication with customers throughout service visits.",
            ]
        }
        with patch("applypilot.scoring.semantic_match.embed_texts", side_effect=self._patched_embed()):
            result = local_tailor.build_pool_realization(schema, pools)
        self.assertEqual(result["bullets"]["Acme"], "Maintained clear communication with customers throughout service visits.")

    def test_embedding_failure_degrades_to_none_not_a_crash(self):
        schema = {"requirements": [{"supported": True, "requirement": "x", "resume_evidence": ["Acme"]}]}
        pools = {"Acme": ["some sentence"]}
        with patch("applypilot.scoring.semantic_match.embed_texts", return_value=None):
            self.assertIsNone(local_tailor.build_pool_realization(schema, pools))

    def test_plugs_into_existing_merge_realization_unchanged(self):
        """Integration: build_pool_realization's output flows through the
        SAME merge_realization every other realization source already
        uses -- no new merge path was written for this."""
        base_resume = {
            "title": "Old Title",
            "summary": "Old summary.",
            "skills": {"Languages": "Python"},
            "experience": [
                {"title": "Acme", "subtitle": "Installer | 2020-2021", "meta": "", "bullets": ["Original bullet one.", "Original bullet two."]}
            ],
            "projects": [],
            "education": "Some University",
        }
        schema = {
            "requirements": [{"supported": True, "requirement": "communicate clearly with customers", "resume_evidence": ["Acme"]}]
        }
        pools = {
            "Acme": [
                "Followed established company protocols during installation work.",
                "Maintained clear communication with customers throughout service visits.",
            ]
        }
        with patch("applypilot.scoring.semantic_match.embed_texts", side_effect=self._patched_embed()):
            realization = local_tailor.build_pool_realization(schema, pools)
        merged = local_tailor.merge_realization(base_resume, realization, {"title": "New Job Title"})
        self.assertEqual(merged["experience"][0]["bullets"][0],
                          "Maintained clear communication with customers throughout service visits.")
        # summary/education untouched -- build_pool_realization doesn't set "summary",
        # so merge_realization's existing base-resume fallback applies unchanged.
        self.assertEqual(merged["summary"], "Old summary.")
        self.assertEqual(merged["education"], "Some University")


class TestEditSentenceForRequirement(unittest.TestCase):
    """2026-09-04: editor mode -- reword ONE already-true sentence instead
    of writing a new one from evidence. Direct response to a real near-miss
    where the writer prompt (request_local_realization) invented "Built and
    implemented the tracking system... managing lanes, shoes, and other
    equipment" for Mavis -- caught by the agency check, but a dropped
    bullet costs the same as no bullet. Smaller task, smaller surface."""

    def test_returns_stripped_text_on_success(self):
        client = MagicMock()
        client.chat.return_value = '"Diagnosed vehicle alignment issues using specialized equipment."'
        result = local_tailor.edit_sentence_for_requirement(client, "orig", "req")
        self.assertEqual(result, "Diagnosed vehicle alignment issues using specialized equipment.")

    def test_returns_none_on_call_failure(self):
        client = MagicMock()
        client.chat.side_effect = RuntimeError("boom")
        self.assertIsNone(local_tailor.edit_sentence_for_requirement(client, "orig", "req"))

    def test_returns_none_on_empty_response(self):
        client = MagicMock()
        client.chat.return_value = "   "
        self.assertIsNone(local_tailor.edit_sentence_for_requirement(client, "orig", "req"))


class TestEditSentenceWithRetry(unittest.TestCase):
    _MAVIS = {
        "name": "National Tire and Battery / Mavis",
        "responsibilities": [
            "Diagnosed and corrected vehicle alignment issues using specialized equipment and "
            "established troubleshooting procedures.",
        ],
    }

    def test_first_attempt_passing_returns_immediately(self):
        client = MagicMock()
        client.chat.return_value = "Diagnosed vehicle alignment issues using specialized diagnostic equipment."
        sentence, was_edited, attempts = local_tailor.edit_sentence_with_retry(
            client, self._MAVIS["responsibilities"][0], "vehicle diagnostics", self._MAVIS, max_attempts=3
        )
        self.assertTrue(was_edited)
        self.assertEqual(attempts, 1)
        self.assertEqual(client.chat.call_count, 1)

    def test_retries_until_a_passing_edit_then_stops(self):
        client = MagicMock()
        client.chat.side_effect = [
            "Managed and led the vehicle alignment program.",  # attempt 1: agency violation
            "Diagnosed vehicle alignment issues using specialized diagnostic tools.",  # attempt 2: passes
        ]
        sentence, was_edited, attempts = local_tailor.edit_sentence_with_retry(
            client, self._MAVIS["responsibilities"][0], "vehicle diagnostics", self._MAVIS, max_attempts=3
        )
        self.assertTrue(was_edited)
        self.assertEqual(attempts, 2)
        self.assertEqual(client.chat.call_count, 2)
        self.assertIn("diagnostic tools", sentence)

    def test_exhausting_all_attempts_falls_back_to_original_unchanged(self):
        """The actual fix for the real near-miss: never return nothing --
        guaranteed non-empty, guaranteed true (it's the untouched original)."""
        client = MagicMock()
        client.chat.return_value = "Built and implemented the tracking system, managing the whole team."
        original = self._MAVIS["responsibilities"][0]
        sentence, was_edited, attempts = local_tailor.edit_sentence_with_retry(
            client, original, "vehicle diagnostics", self._MAVIS, max_attempts=3
        )
        self.assertFalse(was_edited)
        self.assertEqual(attempts, 3)
        self.assertEqual(sentence, original)
        self.assertEqual(client.chat.call_count, 3)

    def test_call_failures_count_as_attempts_and_still_fall_back_safely(self):
        client = MagicMock()
        client.chat.side_effect = RuntimeError("local model unreachable")
        original = self._MAVIS["responsibilities"][0]
        sentence, was_edited, attempts = local_tailor.edit_sentence_with_retry(
            client, original, "vehicle diagnostics", self._MAVIS, max_attempts=2
        )
        self.assertFalse(was_edited)
        self.assertEqual(sentence, original)
        self.assertEqual(client.chat.call_count, 2)

    def test_real_near_miss_sentence_is_correctly_rejected(self):
        """Direct regression pin for the actual sentence produced live
        tonight -- confirms edit_sentence_with_retry's checks reject it
        exactly like request_local_realization's did."""
        client = MagicMock()
        client.chat.return_value = (
            "Built and implemented the tracking system for National Tire and Battery / Mavis, "
            "improving efficiency by managing lanes, shoes, and other equipment."
        )
        sentence, was_edited, _ = local_tailor.edit_sentence_with_retry(
            client, self._MAVIS["responsibilities"][0], "resource tracking", self._MAVIS, max_attempts=1
        )
        self.assertFalse(was_edited)
        self.assertEqual(sentence, self._MAVIS["responsibilities"][0])


if __name__ == "__main__":
    unittest.main()
