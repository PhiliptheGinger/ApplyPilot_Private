"""Tests for local LLM support: local provider config, generation, fallback
integration, and factual-anchor validation on locally-drafted content.

Covers the required scenarios:
1. Local provider configuration (env vars)
2. Successful local generation
3. Local provider unavailable
4. Cloud models all exhausted but local model available
5. Daily-exhausted cloud models being skipped (not re-tried)
6. Local draft preserving factual resume information
7. Malformed local-model output
8. Validation failures
9. Existing cloud-only behavior remaining functional

No real Ollama/local model is required -- httpx calls are mocked throughout.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _matchable_job_and_profile():
    """A minimal (job, profile) pair that deterministically produces a
    genuinely AMBIGUOUS requirement/evidence pairing -- two evidence items
    tied for the top deterministic term-overlap score against the one
    requirement line (see _pair_candidate_evidence) -- so
    get_local_tailoring_plan actually calls the local model instead of
    resolving everything itself. Used by tests that exercise the HTTP call
    itself (errors, malformed output, payload shape)."""
    job = {"title": "Engineer", "full_description": "- Python experience required\n"}
    profile = {
        "resume_facts": {},
        "skills_inventory": [
            {"name": "Python", "resume_allowed": True, "relevance_categories": ["python"]},
            {"name": "Python Fundamentals", "resume_allowed": True, "relevance_categories": ["python"]},
        ],
    }
    return job, profile


# ---------------------------------------------------------------------------
# 1. Local provider configuration
# ---------------------------------------------------------------------------

class TestLocalProviderConfig(unittest.TestCase):
    def test_is_local_configured_false_by_default(self):
        from applypilot import llm as llm_mod
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("APPLYPILOT_LOCAL_LLM_URL", None)
            self.assertFalse(llm_mod.is_local_configured())

    def test_is_local_configured_true_when_url_set(self):
        from applypilot import llm as llm_mod
        with patch.dict("os.environ", {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}):
            self.assertTrue(llm_mod.is_local_configured())

    def test_fallback_chain_appends_local_entry_when_configured(self):
        from applypilot.llm import _build_fallback_chain
        env = {
            k: v for k, v in __import__("os").environ.items()
            if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                         "DEEPSEEK_API_KEY", "LLM_URL",
                         "APPLYPILOT_LOCAL_LLM_URL", "APPLYPILOT_LOCAL_LLM_MODEL")
        }
        env["GEMINI_API_KEY"] = "fake-gemini-key"
        env["APPLYPILOT_LOCAL_LLM_URL"] = "http://localhost:11434/v1"
        env["APPLYPILOT_LOCAL_LLM_MODEL"] = "llama3.2"
        with patch.dict("os.environ", env, clear=True), \
             patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-3.6-flash", quality=False)
        local_entries = [e for e in chain if e.provider == "local"]
        self.assertEqual(len(local_entries), 1)
        self.assertEqual(local_entries[0].name, "llama3.2")
        self.assertEqual(local_entries[0].base_url, "http://localhost:11434/v1")
        # Local must come after all cloud entries (last resort)
        self.assertIs(chain[-1], local_entries[0])

    def test_fallback_chain_no_local_entry_when_not_configured(self):
        from applypilot.llm import _build_fallback_chain
        env = {
            k: v for k, v in __import__("os").environ.items()
            if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                         "DEEPSEEK_API_KEY", "LLM_URL", "APPLYPILOT_LOCAL_LLM_URL")
        }
        env["GEMINI_API_KEY"] = "fake-gemini-key"
        with patch.dict("os.environ", env, clear=True), \
             patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-3.6-flash", quality=False)
        self.assertFalse(any(e.provider == "local" for e in chain))

    def test_local_only_chain_when_no_cloud_keys(self):
        """Local can be the ONLY provider (no cloud keys at all) -- doesn't raise."""
        from applypilot.llm import _build_fallback_chain
        env = {
            k: v for k, v in __import__("os").environ.items()
            if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                         "DEEPSEEK_API_KEY", "LLM_URL", "APPLYPILOT_LOCAL_LLM_URL")
        }
        env["APPLYPILOT_LOCAL_LLM_URL"] = "http://localhost:11434/v1"
        with patch.dict("os.environ", env, clear=True), \
             patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("llama3.2", quality=False)
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].provider, "local")


# ---------------------------------------------------------------------------
# 2. Successful local generation
# ---------------------------------------------------------------------------

class TestLocalGenerationSuccess(unittest.TestCase):
    def _client_with_local_only(self):
        from applypilot.llm import LLMClient, ModelEntry
        fake_chain = [ModelEntry("llama3.2", "local", "http://localhost:11434/v1", "")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="http://localhost:11434/v1", model="llama3.2", api_key="")
        return client

    def test_chat_returns_local_response(self):
        client = self._client_with_local_only()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"status":"ok"}'}}]
        }
        mock_resp.raise_for_status.return_value = None
        with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
            result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, '{"status":"ok"}')
        # No Authorization header when api_key is empty (local doesn't require auth)
        _, kwargs = mock_post.call_args
        self.assertNotIn("Authorization", kwargs["headers"])

    def test_get_local_tailoring_plan_success(self):
        """Matches the actual local_tailor.py schema: the model returns ONLY
        requirement-number -> evidence-number matches (see _PLAN_SYSTEM);
        every other plan field (skills_to_emphasize, matching_projects,
        keyword_targets, summary_focus) is deterministically derived by
        validate_local_plan() from which evidence numbers got matched --
        not asked of the model as free text. This exercises that whole
        derivation, not just JSON parsing."""
        from applypilot.scoring import local_tailor

        resume_text = "Built things using Python, SQL, and ETL pipelines."
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        # Evidence order (see rank_profile_evidence: experience, project,
        # skill, certification, each stable-sorted by score desc): the ETL
        # Pipeline project matches 3 terms (score 3) and comes first as E1;
        # Python and SQL each match 1 term (score 1) and follow as E2/E3.
        mock_resp.json.return_value = {
            "message": {"content": (
                '{"matches":[{"r":1,"e":[2]},{"r":2,"e":[3]},{"r":3,"e":[1]}]}'
            )}
        }
        job = {"title": "Data Engineer", "full_description": (
            "- Python experience required\n"
            "- SQL experience required\n"
            "- ETL pipeline experience required\n"
        )}
        profile = {
            "resume_facts": {},
            "skills_inventory": [
                {"name": "Python", "resume_allowed": True, "relevance_categories": ["python"]},
                {"name": "SQL", "resume_allowed": True, "relevance_categories": ["sql"]},
            ],
            "project_inventory": [
                {"name": "ETL Pipeline", "resume_allowed": True,
                 "relevance_categories": ["etl"], "factual_concepts": ["ETL"]},
            ],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", return_value=mock_resp):
            plan = local_tailor.get_local_tailoring_plan(resume_text, job, profile)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan["requirements"]), 3)
        self.assertTrue(all(r["supported"] for r in plan["requirements"]))
        self.assertIn("Python", plan["skills_to_emphasize"])
        self.assertIn("SQL", plan["skills_to_emphasize"])
        self.assertIn("ETL Pipeline", plan["matching_projects"])
        self.assertIn("etl", plan["keyword_targets"])
        self.assertEqual(len(plan["summary_focus"]), 3)
        # The model never generates prose -- these stay empty by design.
        self.assertEqual(plan["bullets_to_prioritize"], [])
        self.assertEqual(plan["safe_rewrites"], [])


# ---------------------------------------------------------------------------
# 3. Local provider unavailable
# ---------------------------------------------------------------------------

class TestLocalProviderUnavailable(unittest.TestCase):
    def test_local_available_false_when_not_configured(self):
        from applypilot import llm as llm_mod
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(llm_mod.local_available())

    def test_local_available_false_on_connection_error(self):
        from applypilot import llm as llm_mod
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.get", side_effect=ConnectionError("refused")):
            self.assertFalse(llm_mod.local_available())

    def test_get_local_tailoring_plan_returns_none_on_connection_error(self):
        from applypilot.scoring import local_tailor
        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", side_effect=ConnectionError("refused")):
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        self.assertIsNone(plan)

    def test_tailor_gracefully_skips_plan_when_local_unavailable(self):
        """_tailor_one_job must not raise when local plan generation fails --
        it should proceed with tailor_resume(local_plan=None)."""
        from applypilot.scoring import tailor as tailor_mod

        job = {"url": "https://example.com/j1", "title": "Engineer", "site": "test",
               "full_description": "desc", "fit_score": 9}
        profile = {"personal": {"full_name": "Jane Doe"}, "resume_facts": {}}

        with patch.dict("os.environ", {"APPLYPILOT_LOCAL_PLAN": "1",
                                        "APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}), \
             patch("applypilot.scoring.local_tailor.get_local_tailoring_plan",
                   side_effect=Exception("boom")), \
             patch.object(tailor_mod, "tailor_resume",
                          return_value=("RESUME TEXT", {"status": "approved", "attempts": 1,
                                                         "approved_facts": []})) as mock_tailor, \
             patch.object(tailor_mod, "TAILORED_DIR", Path("/tmp/applypilot-test-tailor")):
            tailor_mod.TAILORED_DIR.mkdir(parents=True, exist_ok=True)
            result = tailor_mod._tailor_one_job(job, "Original resume text.", profile)

        self.assertEqual(result["status"], "approved")
        # local_plan kwarg was passed as None since generation failed
        _, kwargs = mock_tailor.call_args
        self.assertIsNone(kwargs.get("local_plan"))


# ---------------------------------------------------------------------------
# 4. Cloud models all exhausted but local model available
# ---------------------------------------------------------------------------

class TestCloudExhaustedLocalAvailable(unittest.TestCase):
    def _client(self):
        from applypilot.llm import LLMClient, ModelEntry
        fake_chain = [
            ModelEntry("cloud-a", "openai_compat", "https://fake.api/v1", "fake-key"),
            ModelEntry("cloud-b", "openai_compat", "https://fake.api/v1", "fake-key"),
            ModelEntry("llama3.2", "local", "http://localhost:11434/v1", ""),
        ]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="https://fake.api/v1", model="cloud-a", api_key="fake-key")
        return client

    def test_falls_through_to_local_when_cloud_exhausted(self):
        client = self._client()
        # Mark both cloud entries as daily-exhausted (24h)
        client._exhausted["cloud-a"] = time.time() + 86400
        client._exhausted["cloud-b"] = time.time() + 86400

        local_resp = MagicMock()
        local_resp.status_code = 200
        local_resp.raise_for_status.return_value = None
        local_resp.json.return_value = {"choices": [{"message": {"content": "local reply"}}]}

        with patch.object(client._client, "post", return_value=local_resp) as mock_post:
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "local reply")
        # Only the local endpoint should have been hit -- cloud entries skipped
        self.assertEqual(mock_post.call_count, 1)
        call_url = mock_post.call_args[0][0]
        self.assertIn("localhost:11434", call_url)

    def test_has_cloud_available_false_when_only_local_left(self):
        client = self._client()
        client._exhausted["cloud-a"] = time.time() + 86400
        client._exhausted["cloud-b"] = time.time() + 86400
        self.assertFalse(client.has_cloud_available())
        # has_available_model still True (local counts)
        self.assertTrue(client.has_available_model())

    def test_has_cloud_available_true_when_cloud_not_exhausted(self):
        client = self._client()
        self.assertTrue(client.has_cloud_available())


# ---------------------------------------------------------------------------
# 5. Daily-exhausted cloud models being skipped (not re-tried)
# ---------------------------------------------------------------------------

class TestDailyExhaustionNotReset(unittest.TestCase):
    def _client(self, n=2):
        from applypilot.llm import LLMClient, ModelEntry
        fake_chain = [
            ModelEntry(f"cloud-{i}", "openai_compat", "https://fake.api/v1", "fake-key")
            for i in range(n)
        ]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="https://fake.api/v1", model="cloud-0", api_key="fake-key")
        return client

    def test_all_models_daily_exhausted_raises_without_retrying(self):
        """Regression test for the bug: previously self._exhausted.clear() reset
        24h daily-quota marks, causing repeated hammering. Now it must raise
        instead of clearing and retrying exhausted entries."""
        client = self._client(2)
        now = time.time()
        client._exhausted["cloud-0"] = now + 86400  # daily exhausted
        client._exhausted["cloud-1"] = now + 86400  # daily exhausted

        with patch.object(client._client, "post") as mock_post:
            with self.assertRaises(RuntimeError):
                client.chat([{"role": "user", "content": "hi"}])
        # No HTTP call should have been attempted -- both entries are still
        # marked exhausted and there's no local fallback in this chain.
        mock_post.assert_not_called()

    def test_short_cooldown_entries_do_reset_when_all_exhausted(self):
        """Short-term (<=5min) rate-limit cooldowns ARE cleared when every
        entry is temporarily exhausted -- only long daily blocks persist."""
        client = self._client(2)
        now = time.time()
        client._exhausted["cloud-0"] = now + 30   # short transient cooldown
        client._exhausted["cloud-1"] = now + 60   # short transient cooldown

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        with patch.object(client._client, "post", return_value=resp):
            result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")

    def test_429_quota_marks_24h_not_cleared_by_subsequent_call(self):
        client = self._client(2)
        quota_resp = MagicMock()
        quota_resp.status_code = 429
        quota_resp.text = "Resource has been exhausted (e.g. check quota)."

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status.return_value = None
        ok_resp.json.return_value = {"choices": [{"message": {"content": "fallback ok"}}]}

        with patch.object(client._client, "post", side_effect=[quota_resp, ok_resp]):
            result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "fallback ok")

        now = time.time()
        self.assertIn("cloud-0", client._exhausted)
        # Marked ~24h out, not a short cooldown
        self.assertGreater(client._exhausted["cloud-0"] - now, 3600)

        # Second call: cloud-0 must be skipped (not retried) since it's still
        # within its 24h window, and cloud-1 (not exhausted) serves it directly.
        with patch.object(client._client, "post", return_value=ok_resp) as mock_post2:
            client.chat([{"role": "user", "content": "hi again"}])
        self.assertEqual(mock_post2.call_count, 1)


# ---------------------------------------------------------------------------
# 6. Local draft preserving factual resume information
# ---------------------------------------------------------------------------

class TestFactualAnchorValidation(unittest.TestCase):
    def _profile(self):
        return {
            "personal": {"full_name": "Jane Doe"},
            "resume_facts": {
                "preserved_companies": ["Acme Corp", "Initech"],
                "preserved_school": "State University",
                "real_metrics": ["40% reduction in latency"],
            },
            "skills_boundary": {"languages": ["Python", "SQL"]},
        }

    def test_validate_factual_anchors_passes_known_employer(self):
        from applypilot.scoring.validator import validate_factual_anchors
        data = {"experience": [{"header": "Software Engineer | Acme Corp | 2020-2023"}]}
        result = validate_factual_anchors(data, self._profile())
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["warnings"], [])

    def test_validate_factual_anchors_warns_on_unknown_employer(self):
        from applypilot.scoring.validator import validate_factual_anchors
        data = {"experience": [{"header": "Software Engineer | Totally Fake Inc | 2020-2023"}]}
        result = validate_factual_anchors(data, self._profile())
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("Fake Inc", result["warnings"][0])

    def test_validate_factual_anchors_noop_when_no_preserved_companies(self):
        from applypilot.scoring.validator import validate_factual_anchors
        data = {"experience": [{"header": "Software Engineer | Anything | 2020-2023"}]}
        profile = {"resume_facts": {}}
        result = validate_factual_anchors(data, profile)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["warnings"], [])

    def test_validate_json_fields_surfaces_anchor_warning(self):
        """validate_json_fields wires in validate_factual_anchors as warnings."""
        from applypilot.scoring.validator import validate_json_fields
        data = {
            "title": "Engineer",
            "summary": "Experienced engineer.",
            "skills": {"languages": "Python, SQL"},
            "experience": [{"header": "Software Engineer | Made Up LLC | 2020-2023", "bullets": ["Did work."]}],
            "education": "State University",
        }
        result = validate_json_fields(data, self._profile())
        self.assertTrue(any("unrecognized employer" in w for w in result["warnings"]))
        # A single unknown-employer mention is a warning, not a hard failure
        self.assertTrue(result["passed"])


# ---------------------------------------------------------------------------
# 7. Malformed local-model output
# ---------------------------------------------------------------------------

class TestMalformedLocalOutput(unittest.TestCase):
    def test_get_local_tailoring_plan_returns_none_on_non_json(self):
        from applypilot.scoring import local_tailor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "message": {"content": "Sure! Here's your plan: it looks great."}
        }
        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", return_value=mock_resp):
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        self.assertIsNone(plan)

    def test_get_local_tailoring_plan_returns_none_on_empty_message(self):
        from applypilot.scoring import local_tailor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {}}
        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", return_value=mock_resp):
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        self.assertIsNone(plan)

    def test_tailor_resume_handles_malformed_json_and_retries(self):
        """When the (local-first-pass-informed) cloud model returns invalid
        JSON, tailor_resume's existing retry loop still recovers on attempt 2."""
        from applypilot.scoring import tailor as tailor_mod

        profile = {
            "personal": {"full_name": "Jane Doe", "email": "jane@example.com"},
            "resume_facts": {"preserved_companies": [], "preserved_school": ""},
            "skills_boundary": {"languages": ["Python"]},
        }
        job = {"title": "Engineer", "site": "test", "full_description": "desc", "location": "Remote"}

        good_json = (
            '{"title":"Engineer","summary":"Solid engineer.",'
            '"skills":{"languages":"Python"},'
            '"experience":[{"header":"Engineer | Acme | 2020-2023","bullets":["Built things."]}],'
            '"projects":[],"education":"State University"}'
        )

        mock_client = MagicMock()
        mock_client.chat.side_effect = ["not json at all", good_json]

        with patch("applypilot.scoring.tailor.get_stage_client", return_value=mock_client), \
             patch("applypilot.scoring.tailor.is_local_configured", return_value=False), \
             patch("applypilot.scoring.tailor.judge_tailored_resume",
                   return_value={"passed": True, "verdict": "SKIP", "issues": "none", "raw": "n/a"}):
            tailored, report = tailor_mod.tailor_resume("Original resume.", job, profile, max_retries=2)

        self.assertEqual(report["status"], "approved")
        self.assertIn("Engineer", tailored)

    def test_parse_plan_strips_think_block_before_json(self):
        """Defense-in-depth for Qwen3/hybrid-reasoning models: a <think>...
        </think> block (populated or empty) preceding the real JSON must not
        break parsing."""
        from applypilot.scoring.local_tailor import _parse_plan
        text = (
            "<think>\nLet me consider the {curly brace} in this reasoning "
            "text first.\n</think>\n"
            '{"requirements": [], "skills_to_emphasize": ["Python"]}'
        )
        result = _parse_plan(text)
        self.assertEqual(result["skills_to_emphasize"], ["Python"])

    def test_parse_plan_strips_empty_think_block(self):
        from applypilot.scoring.local_tailor import _parse_plan
        text = '<think>\n\n</think>\n{"requirements": []}'
        result = _parse_plan(text)
        self.assertEqual(result, {"requirements": []})

    def test_parse_plan_still_works_with_no_think_block(self):
        """Regression guard: the new stripping step must not affect plain
        JSON responses (the common case with /no_think honored)."""
        from applypilot.scoring.local_tailor import _parse_plan
        text = '{"requirements": [], "skills_to_emphasize": ["SQL"]}'
        result = _parse_plan(text)
        self.assertEqual(result["skills_to_emphasize"], ["SQL"])

    def test_get_local_tailoring_plan_handles_think_wrapped_response_end_to_end(self):
        from applypilot.scoring import local_tailor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "message": {"content": (
                '<think>\nThe candidate has Python experience.\n</think>\n'
                '{"matches":[{"r":1,"e":[1]}]}'
            )}
        }
        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", return_value=mock_resp):
            plan = local_tailor.get_local_tailoring_plan("Uses Python daily.", job, profile)
        self.assertIsNotNone(plan)
        self.assertIn("Python", plan["skills_to_emphasize"])


# ---------------------------------------------------------------------------
# 8. Validation failures
# ---------------------------------------------------------------------------

class TestValidationFailures(unittest.TestCase):
    def test_validate_json_fields_fails_on_missing_required_field(self):
        from applypilot.scoring.validator import validate_json_fields
        data = {"title": "Engineer", "summary": "", "skills": {}, "experience": [], "education": ""}
        result = validate_json_fields(data, {"resume_facts": {}})
        self.assertFalse(result["passed"])
        self.assertTrue(any("Missing required field" in e for e in result["errors"]))

    def test_validate_json_fields_fails_on_fabricated_skill(self):
        from applypilot.scoring.validator import validate_json_fields
        data = {
            "title": "Engineer",
            "summary": "Experienced engineer.",
            "skills": {"languages": "Python, Rust"},
            "experience": [{"header": "Engineer | Acme | 2020", "bullets": ["Did work."]}],
            "education": "State University",
        }
        profile = {
            "resume_facts": {"preserved_companies": [], "preserved_school": ""},
            "skills_boundary": {"languages": ["Python"]},
        }
        result = validate_json_fields(data, profile)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Fabricated skill" in e for e in result["errors"]))

    def test_tailor_resume_returns_failed_validation_after_exhausting_retries(self):
        from applypilot.scoring import tailor as tailor_mod

        profile = {
            "personal": {"full_name": "Jane Doe"},
            "resume_facts": {"preserved_companies": [], "preserved_school": ""},
            "skills_boundary": {"languages": ["Python"]},
        }
        job = {"title": "Engineer", "site": "test", "full_description": "desc", "location": "Remote"}

        bad_json = (
            '{"title":"Engineer","summary":"Summary.",'
            '"skills":{"languages":"Rust"},'
            '"experience":[{"header":"Engineer | Acme | 2020","bullets":["Did work."]}],'
            '"projects":[],"education":""}'
        )
        mock_client = MagicMock()
        mock_client.chat.return_value = bad_json

        with patch("applypilot.scoring.tailor.get_stage_client", return_value=mock_client), \
             patch("applypilot.scoring.tailor.is_local_configured", return_value=False):
            tailored, report = tailor_mod.tailor_resume("Original resume.", job, profile, max_retries=1)

        self.assertEqual(report["status"], "failed_validation")


# ---------------------------------------------------------------------------
# 9. Existing cloud-only behavior remains functional
# ---------------------------------------------------------------------------

class TestExistingCloudBehaviorUnaffected(unittest.TestCase):
    def test_gemini_only_chain_unaffected_by_local_support(self):
        from applypilot.llm import _build_fallback_chain
        env = {
            k: v for k, v in __import__("os").environ.items()
            if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                         "DEEPSEEK_API_KEY", "LLM_URL", "APPLYPILOT_LOCAL_LLM_URL")
        }
        env["GEMINI_API_KEY"] = "fake-key"
        with patch.dict("os.environ", env, clear=True), \
             patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-3.6-flash", quality=False)
        self.assertTrue(all(e.provider == "gemini" for e in chain))
        self.assertFalse(any(e.provider == "local" for e in chain))

    def test_cloud_chat_success_unaffected(self):
        from applypilot.llm import LLMClient, ModelEntry
        fake_chain = [ModelEntry("cloud-a", "openai_compat", "https://fake.api/v1", "fake-key")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="https://fake.api/v1", model="cloud-a", api_key="fake-key")

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "cloud reply"}}]}
        with patch.object(client._client, "post", return_value=resp) as mock_post:
            result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "cloud reply")
        # Authorization header present for cloud (non-empty api_key)
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer fake-key")

    def test_tailor_resume_without_local_plan_unaffected(self):
        """Calling tailor_resume with no local_plan kwarg (existing call sites)
        still works exactly as before."""
        from applypilot.scoring import tailor as tailor_mod

        profile = {
            "personal": {"full_name": "Jane Doe"},
            "resume_facts": {"preserved_companies": [], "preserved_school": ""},
            "skills_boundary": {"languages": ["Python"]},
        }
        job = {"title": "Engineer", "site": "test", "full_description": "desc", "location": "Remote"}
        good_json = (
            '{"title":"Engineer","summary":"Solid engineer.",'
            '"skills":{"languages":"Python"},'
            '"experience":[{"header":"Engineer | Acme | 2020-2023","bullets":["Built things."]}],'
            '"projects":[],"education":"State University"}'
        )
        mock_client = MagicMock()
        mock_client.chat.return_value = good_json

        with patch("applypilot.scoring.tailor.get_stage_client", return_value=mock_client), \
             patch("applypilot.scoring.tailor.is_local_configured", return_value=False):
            tailored, report = tailor_mod.tailor_resume("Original resume.", job, profile, max_retries=1)

        self.assertEqual(report["status"], "approved")


# ---------------------------------------------------------------------------
# 10. Deterministic requirement-line extraction (no LLM, no embeddings)
# ---------------------------------------------------------------------------

class TestDeterministicRequirementExtraction(unittest.TestCase):
    def test_extracts_bulleted_lines(self):
        from applypilot.scoring.local_tailor import _extract_requirement_lines
        desc = (
            "About the role:\n"
            "- Experience with Python required\n"
            "- SQL is a plus\n"
            "- Familiarity with Docker preferred\n"
        )
        lines = _extract_requirement_lines(desc)
        texts = [l["text"] for l in lines]
        self.assertIn("Experience with Python required", texts)
        self.assertIn("SQL is a plus", texts)
        self.assertIn("Familiarity with Docker preferred", texts)

    def test_tags_required_vs_preferred(self):
        from applypilot.scoring.local_tailor import _extract_requirement_lines
        desc = (
            "- Python experience is required\n"
            "- Docker knowledge preferred\n"
            "- Owns a pet dinosaur\n"
        )
        lines = {l["text"]: l["importance"] for l in _extract_requirement_lines(desc)}
        self.assertEqual(lines["Python experience is required"], "required")
        self.assertEqual(lines["Docker knowledge preferred"], "preferred")
        self.assertEqual(lines["Owns a pet dinosaur"], "unspecified")

    def test_numbered_lines_also_extracted(self):
        from applypilot.scoring.local_tailor import _extract_requirement_lines
        desc = "1. Must have 3+ years of experience\n2. Bachelor's degree preferred\n"
        texts = [l["text"] for l in _extract_requirement_lines(desc)]
        self.assertIn("Must have 3+ years of experience", texts)
        self.assertIn("Bachelor's degree preferred", texts)

    def test_empty_description_returns_empty_list(self):
        from applypilot.scoring.local_tailor import _extract_requirement_lines
        self.assertEqual(_extract_requirement_lines(""), [])
        self.assertEqual(_extract_requirement_lines(None), [])

    def test_deduplicates_and_caps_line_count(self):
        from applypilot.scoring.local_tailor import _extract_requirement_lines
        desc = "\n".join(f"- Requirement number {i} needs real content here" for i in range(20))
        desc += "\n- Requirement number 0 needs real content here"  # exact duplicate
        lines = _extract_requirement_lines(desc, max_lines=10)
        self.assertLessEqual(len(lines), 10)

    def test_prose_paragraph_with_no_bullets_yields_nothing(self):
        from applypilot.scoring.local_tailor import _extract_requirement_lines
        desc = "This is a great job at a great company doing great things for great people."
        self.assertEqual(_extract_requirement_lines(desc), [])

    def test_employer_benefit_lines_are_filtered_out(self):
        """Benefits/perks/compensation lines are not candidate requirements
        -- deterministic preprocessing drops them before the local model
        ever sees them, instead of relying on the model to recognize this
        (which it can get wrong, and which costs it reasoning time even
        when it gets it right)."""
        from applypilot.scoring.local_tailor import _extract_requirement_lines
        desc = (
            "- 3+ years of Python experience required\n"
            "- Generous PTO and paid holidays\n"
            "- 401(k) with company match\n"
            "- Comprehensive health, dental, and vision insurance\n"
            "- Tuition reimbursement program\n"
            "- Career growth opportunities\n"
            "- Bachelor's degree preferred\n"
        )
        texts = [l["text"] for l in _extract_requirement_lines(desc)]
        self.assertIn("3+ years of Python experience required", texts)
        self.assertIn("Bachelor's degree preferred", texts)
        self.assertFalse(any("PTO" in t for t in texts))
        self.assertFalse(any("401(k)" in t for t in texts))
        self.assertFalse(any("insurance" in t.lower() for t in texts))
        self.assertFalse(any("Tuition" in t for t in texts))
        self.assertFalse(any("Career growth" in t for t in texts))

    def test_real_world_total_compensation_bullets_are_all_filtered(self):
        """The exact 'Total Compensation Package' block from a real posting
        (O'Reilly Auto Parts, job 20172). The last three previously survived
        the filter and the local model then marked two of them 'supported'
        because a past employer plausibly offered them -- an employer's
        offer laundered into a candidate qualification. Every one of these
        must be dropped before the model ever sees it."""
        from applypilot.scoring.local_tailor import _extract_requirement_lines
        desc = (
            "  * Competitive Wages & Paid Time Off \n"
            "  * Stock Purchase Plan & 401k with Employer Contributions Starting Day One \n"
            "  * Medical, Dental, & Vision Insurance with Optional Flexible Spending Account (FSA) \n"
            "  * Team Member Health/Wellbeing Programs \n"
            "  * Tuition Educational Assistance Programs \n"
            "  * Opportunities for Career Growth \n"
        )
        self.assertEqual(_extract_requirement_lines(desc), [])

    def test_benefit_phrasing_variants_are_recognized(self):
        """Phrase/category matching, not one-off strings: each of these is a
        differently-worded instance of a category the filter must cover."""
        from applypilot.scoring.local_tailor import _is_benefit_line
        for line in [
            "Team Member Health/Wellbeing Programs",
            "Employee Wellness Programs",
            "Team wellness program and on-site fitness classes",
            "Tuition Educational Assistance Programs",
            "Tuition reimbursement after one year",
            "Educational Assistance available day one",
            "Opportunities for Career Growth",
            "Career growth opportunities",
            "Advancement opportunities within the organization",
            "Generous PTO and paid holidays",
            "Paid Time Off and paid parental leave",
            "Competitive Wages & Paid Time Off",
            "Competitive compensation and annual bonus",
            "Competitive pay based on market data",
            "401k with Employer Contributions Starting Day One",
            "401(k) with company match",
            "Retirement savings plan",
            "Stock Purchase Plan & employee stock options",
            "Employee stock purchase plan",
            "Medical, Dental, & Vision Insurance",
            "Comprehensive health, dental, and vision insurance",
            "Vision insurance and life insurance",
            "Optional Flexible Spending Account (FSA)",
            "FSA and HSA options",
        ]:
            with self.subTest(line=line):
                self.assertTrue(_is_benefit_line(line), f"not filtered: {line!r}")

    def test_real_qualifications_are_not_filtered_as_benefits(self):
        """The mirror-image failure: 'health', 'education' and 'career' are
        ordinary words inside genuine requirements, so their presence alone
        must never drop a line."""
        from applypilot.scoring.local_tailor import _is_benefit_line
        for line in [
            "Bachelor's degree in Health Sciences required",
            "5+ years of experience in health care operations",
            "Experience with health insurance claims processing systems",
            "Knowledge of retirement plan administration",
            "Ability to educate customers on product options",
            "Familiarity with career services software preferred",
            "Must have a background in education technology",
            "Responsible for payroll processing and reconciliation",
            "Demonstrated track record of career-long technical leadership",
            "3+ years of Python experience required",
            "Ability to quickly match alphanumeric sequences",
            "Ability to provide outstanding, friendly and professional customer service",
            "Familiar with automotive parts, cataloging, and automotive sales or service",
            "ASE certification",
            "Fluency in multiple languages (Spanish is highly desired)",
            "Bachelor's degree preferred",
        ]:
            with self.subTest(line=line):
                self.assertFalse(_is_benefit_line(line), f"wrongly filtered: {line!r}")

    def test_split_reports_dropped_benefit_lines(self):
        from applypilot.scoring.local_tailor import _split_requirement_lines
        desc = (
            "- 3+ years of Python experience required\n"
            "- Opportunities for Career Growth\n"
            "- Tuition Educational Assistance Programs\n"
        )
        lines, dropped = _split_requirement_lines(desc)
        self.assertEqual([l["text"] for l in lines],
                         ["3+ years of Python experience required"])
        self.assertEqual(dropped,
                         ["Opportunities for Career Growth",
                          "Tuition Educational Assistance Programs"])


# ---------------------------------------------------------------------------
# 11. Deterministic evidence retrieval/ranking (no LLM, no embeddings)
# ---------------------------------------------------------------------------

def _sample_profile():
    return {
        "experience_inventory": [
            {
                "name": "Freelance Photography",
                "role_type": "freelance",
                "relevance_categories": ["media", "content", "creative"],
                "resume_allowed": True,
                "description": "DSLR photography and videography.",
            },
            {
                "name": "Private Consulting Gig",
                "relevance_categories": ["consulting"],
                "resume_allowed": False,  # must never be surfaced
                "description": "Confidential engagement.",
            },
        ],
        "project_inventory": [
            {
                "name": "Standup-OCR",
                "relevance_categories": ["python", "ocr", "automation"],
                "factual_concepts": ["Python", "OCR", "Image processing"],
                "resume_allowed": True,
                "constraints": ["Do not claim production deployment."],
            },
            {
                "name": "Unrelated Woodworking Project",
                "relevance_categories": ["woodworking"],
                "resume_allowed": True,
            },
        ],
        "skills_inventory": [
            {"name": "Python", "evidence_level": "demonstrated", "resume_allowed": True,
             "relevance_categories": ["python", "automation"]},
            {"name": "Photography", "evidence_level": "demonstrated", "resume_allowed": True,
             "relevance_categories": ["media"]},
        ],
    }


class TestEvidenceRetrieval(unittest.TestCase):
    def _job(self, description, title="Python Automation Engineer"):
        return {"title": title, "full_description": description}

    def test_ranks_matching_items_above_unrelated(self):
        from applypilot.scoring.local_tailor import rank_profile_evidence
        job = self._job("Looking for Python automation experience with OCR tooling.")
        ranked = rank_profile_evidence(job, _sample_profile(), top_n=10)
        names = [r["name"] for r in ranked]
        self.assertIn("Standup-OCR", names)
        self.assertIn("Python", names)
        self.assertNotIn("Unrelated Woodworking Project", names)
        # Best match (most overlapping terms) should be first.
        self.assertEqual(names[0], "Standup-OCR")

    def test_excludes_resume_allowed_false_items(self):
        from applypilot.scoring.local_tailor import rank_profile_evidence
        job = self._job("Looking for a consulting background.")
        ranked = rank_profile_evidence(job, _sample_profile(), top_n=10)
        names = [r["name"] for r in ranked]
        self.assertNotIn("Private Consulting Gig", names)

    def test_matched_terms_are_inspectable(self):
        """The retrieval must expose WHY an item matched, not just that it did."""
        from applypilot.scoring.local_tailor import rank_profile_evidence
        job = self._job("We need strong Python automation skills.")
        ranked = rank_profile_evidence(job, _sample_profile(), top_n=10)
        python_skill = next(r for r in ranked if r["name"] == "Python" and r["type"] == "skill")
        self.assertIn("python", python_skill["matched_terms"])
        self.assertIn("automation", python_skill["matched_terms"])

    def test_reports_source_inventory_type(self):
        from applypilot.scoring.local_tailor import rank_profile_evidence
        job = self._job("Python automation and OCR and media and photography work.")
        ranked = rank_profile_evidence(job, _sample_profile(), top_n=10)
        types = {r["name"]: r["type"] for r in ranked}
        self.assertEqual(types["Standup-OCR"], "project")
        self.assertEqual(types["Python"], "skill")
        self.assertEqual(types["Freelance Photography"], "experience")

    def test_no_match_returns_empty(self):
        from applypilot.scoring.local_tailor import rank_profile_evidence
        job = self._job(
            "Seeking an experienced neurosurgeon for our hospital.",
            title="Neurosurgeon",
        )
        ranked = rank_profile_evidence(job, _sample_profile(), top_n=10)
        self.assertEqual(ranked, [])

    def test_top_n_is_respected(self):
        from applypilot.scoring.local_tailor import rank_profile_evidence
        job = self._job("Python automation OCR media photography content creative.")
        ranked = rank_profile_evidence(job, _sample_profile(), top_n=2)
        self.assertLessEqual(len(ranked), 2)

    def test_common_connector_word_in_item_name_does_not_cause_false_match(self):
        """Regression: a multi-word item name like 'X and Y' must not match
        via the word 'and', which appears in virtually every job posting."""
        from applypilot.scoring.local_tailor import rank_profile_evidence
        profile = {
            "experience_inventory": [{
                "name": "National Tire and Battery",
                "relevance_categories": ["retail"],
                "resume_allowed": True,
            }],
            "project_inventory": [], "skills_inventory": [],
        }
        job = self._job("We need someone who works well with a team and communicates clearly.")
        ranked = rank_profile_evidence(job, profile, top_n=10)
        self.assertEqual(ranked, [])

    def test_format_evidence_for_prompt_includes_constraints(self):
        from applypilot.scoring.local_tailor import format_evidence_for_prompt, rank_profile_evidence
        job = self._job("Python automation OCR experience needed.")
        ranked = rank_profile_evidence(job, _sample_profile(), top_n=10)
        rendered = format_evidence_for_prompt(ranked)
        self.assertIn("Standup-OCR", rendered)
        self.assertIn("Do not claim production deployment.", rendered)

    def test_format_evidence_for_prompt_empty_list_returns_empty_string(self):
        from applypilot.scoring.local_tailor import format_evidence_for_prompt
        self.assertEqual(format_evidence_for_prompt([]), "")

    def test_certifications_are_included_as_evidence(self):
        """certifications is a fourth deterministic evidence source (same
        shape as the other inventories) -- so matching_certifications can be
        derived from real matches instead of asked of the model as free
        text, which is exactly where a small model is most likely to
        hallucinate a certification name."""
        from applypilot.scoring.local_tailor import rank_profile_evidence
        profile = _sample_profile()
        profile["certifications"] = [{
            "name": "CompTIA A+", "resume_allowed": True,
            "relevance_categories": ["IT", "help desk"],
        }]
        job = self._job("Looking for CompTIA A+ certified help desk support.")
        ranked = rank_profile_evidence(job, profile, top_n=10)
        cert = next(r for r in ranked if r["name"] == "CompTIA A+")
        self.assertEqual(cert["type"], "certification")

    def test_malformed_inventory_entries_do_not_crash(self):
        """Non-dict entries / missing fields must be skipped, not raise."""
        from applypilot.scoring.local_tailor import rank_profile_evidence
        profile = {
            "experience_inventory": ["not a dict", None, {"no_name_field": True}],
            "project_inventory": [{"name": 123}],  # name not a string
            "skills_inventory": None,  # not even a list
        }
        job = self._job("Python automation.")
        ranked = rank_profile_evidence(job, profile, top_n=10)  # must not raise
        self.assertEqual(ranked, [])


# ---------------------------------------------------------------------------
# 12. Cloud/local plan generation now uses retrieved evidence as grounding
# ---------------------------------------------------------------------------

class TestPlanUsesEvidenceGrounding(unittest.TestCase):
    def _job(self):
        return {"title": "Python Automation Engineer",
                "full_description": "- Python automation and OCR experience required\n"}

    def test_prompt_includes_evidence_section_when_matched(self):
        """get_local_tailoring_plan must actually send the retrieved
        evidence to the local model when a requirement is genuinely
        ambiguous between multiple deterministically-tied candidates (see
        _pair_candidate_evidence) -- not just compute it and discard it."""
        from applypilot.scoring import local_tailor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": '{"matches":[]}'}}
        job = self._job()
        # Two projects with identical category/concept tags, so both tie
        # for the top pair-score against the single requirement line --
        # a genuine ambiguity that must reach the model.
        profile = {
            "project_inventory": [
                {"name": "Standup-OCR", "relevance_categories": ["python", "ocr", "automation"],
                 "factual_concepts": ["Python", "OCR"], "resume_allowed": True},
                {"name": "Doc-OCR Pipeline", "relevance_categories": ["python", "ocr", "automation"],
                 "factual_concepts": ["Python", "OCR"], "resume_allowed": True},
            ],
            "skills_inventory": [], "experience_inventory": [],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", return_value=mock_resp) as mock_post:
            local_tailor.get_local_tailoring_plan("Resume text.", job, profile)
        sent_payload = mock_post.call_args.kwargs["json"]
        user_content = sent_payload["messages"][1]["content"]
        self.assertIn("EVIDENCE", user_content)
        self.assertIn("Standup-OCR", user_content)
        self.assertIn("Doc-OCR Pipeline", user_content)
        self.assertIn("candidates:", user_content)

    def test_plan_grounded_in_evidence_not_literal_resume_survives_validation(self):
        """A plan entry grounded in the retrieved EVIDENCE block
        (profile-derived, e.g. a project matched by number) but never
        mentioned in the raw resume_text string must still come back as
        supported -- the local model no longer sees or references
        resume_text at all, so grounding comes entirely from the closed
        evidence list it was shown (see validate_local_plan)."""
        from applypilot.scoring import local_tailor

        resume_text = "A short resume that never mentions OCR by name."
        job = self._job()
        profile = _sample_profile()
        ranked = local_tailor.rank_profile_evidence(job, profile, top_n=6)
        ocr_project_id = next(
            i for i, r in enumerate(ranked, start=1) if r["name"] == "Standup-OCR"
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "message": {"content": f'{{"matches":[{{"r":1,"e":[{ocr_project_id}]}}]}}'}
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", return_value=mock_resp):
            plan = local_tailor.get_local_tailoring_plan(resume_text, job, profile)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan["requirements"]), 1)
        self.assertTrue(plan["requirements"][0]["supported"])
        self.assertIn("ocr", plan["keyword_targets"])
        self.assertIn("Standup-OCR", plan["matching_projects"])


# ---------------------------------------------------------------------------
# 12c. Deterministic requirement<->evidence PAIR scoring narrows candidates
#
# Regression coverage for a real live-test failure (jobs 20171/20170): with
# a flat E1..En evidence list and no per-requirement signal, Qwen3:1.7b
# matched a "Practice excellent customer service" requirement to the
# Python skill (topically present *somewhere* in the evidence, but with no
# actual bearing on customer service) while missing genuinely relevant
# customer-service/sales evidence. _pair_candidate_evidence closes that gap
# deterministically -- these tests exercise it directly and end-to-end.
# ---------------------------------------------------------------------------

def _job_domain_mix_profile():
    """Evidence spanning three unrelated domains (customer service,
    sales, technical) so a requirement about one domain has a
    deterministically obvious in-domain candidate and several
    deterministically obvious NON-candidates -- exactly the shape of the
    live 20171/20170 failures (Mavis/Waffle House misread as Python)."""
    return {
        "experience_inventory": [
            {"name": "National Tire and Battery / Mavis",
             "relevance_categories": ["automotive", "customer service"], "resume_allowed": True},
            {"name": "AMP Smart", "relevance_categories": ["sales"], "resume_allowed": True},
            {"name": "Waffle House", "relevance_categories": ["customer service"], "resume_allowed": True},
        ],
        "skills_inventory": [
            {"name": "Python", "relevance_categories": ["technical"], "resume_allowed": True},
            {"name": "CompTIA A+", "relevance_categories": ["information technology"], "resume_allowed": True},
        ],
        "project_inventory": [],
    }


def _job_domain_mix_job():
    # Every category term below must appear somewhere in the description
    # so rank_profile_evidence retrieves all five items -- the prose line
    # (not a bullet) supplies terms for items whose specific requirement
    # bullet won't mention them, mirroring how a real posting's "About us"
    # boilerplate primes retrieval without being a requirement itself.
    return {
        "title": "Hardware Sales Associate",
        "full_description": (
            "We value sales, technical, information technology, and "
            "automotive skills, plus excellent customer service.\n"
        ),
    }


class TestPairScoringNarrowsCandidates(unittest.TestCase):
    def _ranked(self):
        from applypilot.scoring.local_tailor import rank_profile_evidence
        return rank_profile_evidence(_job_domain_mix_job(), _job_domain_mix_profile(), top_n=10)

    def test_customer_service_requirement_excludes_python(self):
        from applypilot.scoring.local_tailor import _pair_candidate_evidence
        ranked = self._ranked()
        names = [r["name"] for r in ranked]
        cands = _pair_candidate_evidence(
            "Practice excellent customer service at all times", ranked,
        )
        cand_names = {names[i - 1] for i in cands}
        self.assertTrue(cand_names & {"National Tire and Battery / Mavis", "Waffle House"})
        self.assertNotIn("Python", cand_names)
        self.assertNotIn("CompTIA A+", cand_names)

    def test_sales_outreach_requirement_favors_amp_smart(self):
        from applypilot.scoring.local_tailor import _pair_candidate_evidence
        ranked = self._ranked()
        names = [r["name"] for r in ranked]
        cands = _pair_candidate_evidence(
            "Generate leads and close sales with new customers", ranked,
        )
        cand_names = {names[i - 1] for i in cands}
        self.assertIn("AMP Smart", cand_names)
        self.assertNotIn("Python", cand_names)
        self.assertNotIn("National Tire and Battery / Mavis", cand_names)

    def test_technical_requirement_favors_python_and_a_plus(self):
        from applypilot.scoring.local_tailor import _pair_candidate_evidence
        ranked = self._ranked()
        names = [r["name"] for r in ranked]
        cands = _pair_candidate_evidence(
            "Strong technical and information technology troubleshooting skills", ranked,
        )
        cand_names = {names[i - 1] for i in cands}
        self.assertTrue(cand_names & {"Python", "CompTIA A+"})
        self.assertNotIn("Waffle House", cand_names)
        self.assertNotIn("AMP Smart", cand_names)

    def test_irrelevant_evidence_excluded_from_candidates(self):
        from applypilot.scoring.local_tailor import _pair_candidate_evidence
        ranked = self._ranked()
        cands = _pair_candidate_evidence("Lift up to 50 pounds of merchandise", ranked)
        self.assertEqual(cands, [])

    def test_end_to_end_model_cannot_select_python_for_customer_service_requirement(self):
        """Even if the local model ignores its candidate hint and cites
        Python for a customer-service requirement, get_local_tailoring_plan
        must strip that pick -- enforcement happens in code (see
        _merge_model_matches_with_resolved), not by trusting the prompt."""
        from applypilot.scoring import local_tailor

        job = {
            "title": "Hardware Sales Associate",
            "full_description": (
                "- Practice excellent customer service at all times\n"
                "\n"
                "We value sales, technical, information technology, and "
                "automotive skills.\n"
            ),
        }
        profile = _job_domain_mix_profile()
        ranked = local_tailor.rank_profile_evidence(job, profile, top_n=10)
        names = [r["name"] for r in ranked]
        waffle_id = names.index("Waffle House") + 1
        python_id = names.index("Python") + 1

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "message": {"content": (
                f'{{"matches":[{{"r":1,"e":[{waffle_id},{python_id}]}}]}}'
            )}
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", return_value=mock_resp):
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)

        self.assertIsNotNone(plan)
        self.assertEqual(plan["requirements"][0]["resume_evidence"], ["Waffle House"])
        self.assertNotIn("Python", plan["skills_to_emphasize"])
        joined = " ".join(plan["_warnings"])
        self.assertIn("not in its deterministic candidate tier", joined)


# ---------------------------------------------------------------------------
# 12b. Reasoning disabled via the native `think` param.
#
# Root cause of a real reported timeout: get_local_tailoring_plan() talks to
# the local endpoint directly (not via LLMClient.chat()), so it never
# benefited from the existing "Qwen3 optimization" in llm.py. It now calls
# Ollama's native /api/chat endpoint with "think": false, which -- unlike
# the older /no_think prompt-prefix convention it replaced -- is honored
# uniformly by Ollama regardless of which model is loaded, so there's no
# longer a per-model branch to test here.
# ---------------------------------------------------------------------------

class TestSkipsLlmCallWhenNothingToMatch(unittest.TestCase):
    """Deterministic short-circuit: if there's no requirement line or no
    retrieved evidence, there is nothing for the local model to usefully
    decide, so it's never called -- pure latency win, and still returns a
    valid (empty) plan rather than None."""

    def test_no_requirement_lines_skips_llm_call(self):
        from applypilot.scoring import local_tailor
        job = {"title": "Engineer", "full_description": "A great team doing great work."}
        profile = {
            "resume_facts": {},
            "skills_inventory": [{"name": "Python", "resume_allowed": True,
                                   "relevance_categories": ["python"]}],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post") as mock_post:
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        mock_post.assert_not_called()
        self.assertIsNotNone(plan)
        self.assertEqual(plan["requirements"], [])

    def test_no_matching_evidence_skips_llm_call(self):
        from applypilot.scoring import local_tailor
        job = {"title": "Chef", "full_description": "- Culinary arts degree required\n"}
        profile = {
            "resume_facts": {},
            "skills_inventory": [{"name": "Python", "resume_allowed": True,
                                   "relevance_categories": ["python"]}],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post") as mock_post:
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        mock_post.assert_not_called()
        self.assertIsNotNone(plan)
        self.assertEqual(plan["unsupported_requirements"],
                          ["Culinary arts degree required"])

    def test_benefits_only_posting_skips_llm_call(self):
        """A posting whose only bullet lines are employer benefits has
        nothing for the model to match, so it must not be called at all --
        and the plan must say so, since 'no requirements' and 'all
        requirements were perks' look identical in the debug output
        otherwise."""
        from applypilot.scoring import local_tailor
        job = {
            "title": "Parts Specialist",
            "full_description": (
                "Total Compensation Package\n"
                "  * Competitive Wages & Paid Time Off \n"
                "  * Stock Purchase Plan & 401k with Employer Contributions Starting Day One \n"
                "  * Medical, Dental, & Vision Insurance with Optional Flexible Spending Account (FSA) \n"
                "  * Team Member Health/Wellbeing Programs \n"
                "  * Tuition Educational Assistance Programs \n"
                "  * Opportunities for Career Growth \n"
            ),
        }
        profile = {
            "resume_facts": {},
            "skills_inventory": [
                {"name": "Customer Service", "resume_allowed": True,
                 "relevance_categories": ["customer", "service", "retail"]},
            ],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post") as mock_post:
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)

        mock_post.assert_not_called()
        self.assertIsNotNone(plan)
        self.assertEqual(plan["requirements"], [])
        self.assertEqual(plan["unsupported_requirements"], [])
        # No fabricated candidate evidence anywhere in the plan.
        self.assertEqual(plan["skills_to_emphasize"], [])
        self.assertEqual(plan["matching_projects"], [])
        self.assertEqual(plan["summary_focus"], [])
        joined = " ".join(plan["_warnings"])
        self.assertIn("Skipped local LLM call", joined)
        self.assertIn("employer benefits/perks", joined)
        self.assertIn("Tuition Educational Assistance Programs", joined)


class TestThinkDisabled(unittest.TestCase):
    def test_think_false_sent_regardless_of_model_name(self):
        from applypilot.scoring import local_tailor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": '{"matches":[]}'}}
        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1",
               "APPLYPILOT_LOCAL_LLM_MODEL": "qwen3:8b"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", return_value=mock_resp) as mock_post:
            local_tailor.get_local_tailoring_plan("resume", job, profile)
        sent = mock_post.call_args.kwargs["json"]
        self.assertIs(sent["think"], False)
        self.assertEqual(sent["model"], "qwen3:8b")

    def test_timeout_is_logged_with_specific_cause(self):
        """A timeout must produce a specific, visible (WARNING-level)
        explanation rather than being silently swallowed at debug level --
        'a local planning failure should remain a local planning failure
        and explain why'."""
        from applypilot.scoring import local_tailor
        import httpx as httpx_mod

        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1",
               "APPLYPILOT_LOCAL_LLM_MODEL": "qwen3:8b"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", side_effect=httpx_mod.ReadTimeout("timed out")), \
             patch.object(local_tailor.log, "warning") as mock_warn:
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        self.assertIsNone(plan)
        mock_warn.assert_called_once()
        logged_msg = mock_warn.call_args[0][0]
        self.assertIn("timed out", logged_msg.lower())


# ---------------------------------------------------------------------------
# 13. debug_plan_for_job: combined plan + evidence + requirement lines
# ---------------------------------------------------------------------------

class TestDebugPlanForJob(unittest.TestCase):
    def test_returns_plan_evidence_and_requirement_lines(self):
        from applypilot.scoring import local_tailor

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": '{"matches":[]}'}}
        # "automation" only appears via the title, not this specific
        # requirement line, so Standup-OCR (categories python/ocr/
        # automation) and the Python skill (categories python/automation)
        # tie for the top pair-score against it -- genuinely ambiguous,
        # so the HTTP call actually happens.
        job = {"url": "https://example.com/j1", "title": "Python Automation Engineer",
               "full_description": "- Python experience required\n"}
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", return_value=mock_resp), \
             patch("applypilot.scoring.resume_router.load_resume_text_for_job",
                   return_value=("Resume text.", None)):
            result = local_tailor.debug_plan_for_job(job, _sample_profile())
        self.assertIsNotNone(result)
        self.assertIn("plan", result)
        self.assertIn("evidence", result)
        self.assertIn("requirement_lines", result)
        self.assertTrue(any(e["name"] == "Standup-OCR" for e in result["evidence"]))
        self.assertTrue(any("required" in l["importance"] for l in result["requirement_lines"]))

    def test_returns_none_when_local_planning_fails(self):
        from applypilot.scoring import local_tailor

        # See the sibling test above for why this description (rather than
        # the full "Python automation and OCR..." line) is needed to force
        # a genuinely ambiguous pair-score and therefore an actual HTTP call.
        job = {"url": "https://example.com/j2", "title": "Python Automation Engineer",
               "full_description": "- Python experience required\n"}
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), \
             patch("httpx.post", side_effect=ConnectionError("refused")), \
             patch("applypilot.scoring.resume_router.load_resume_text_for_job",
                   return_value=("Resume text.", None)):
            result = local_tailor.debug_plan_for_job(job, _sample_profile())
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 14. Fallback-chain provider separation
#
# 2026-08-22 regression: with GEMINI_API_KEY set, constructing an LLMClient
# whose primary model is the configured local model name (e.g. cli.py's
# `test-local` command building LLMClient(local_url, "qwen3:8b", ...))
# caused _build_fallback_chain to treat "qwen3:8b" as an unrecognized
# (therefore assumed-new) GEMINI model and add it there too -- producing a
# bogus "qwen3:8b (gemini)" entry that 404s against the real Gemini API.
# ---------------------------------------------------------------------------

class TestFallbackChainProviderSeparation(unittest.TestCase):
    def _env(self, **overrides):
        env = {
            k: v for k, v in __import__("os").environ.items()
            if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                         "DEEPSEEK_API_KEY", "LLM_URL", "LLM_MODEL",
                         "APPLYPILOT_LOCAL_LLM_URL", "APPLYPILOT_LOCAL_LLM_MODEL")
        }
        env.update(overrides)
        return env

    def test_local_model_name_never_added_as_gemini_entry(self):
        """The exact reported bug: APPLYPILOT_LOCAL_LLM_MODEL=qwen3:8b with
        GEMINI_API_KEY also set must not produce a 'qwen3:8b' entry under
        the gemini provider, and gemini/qwen3:8b must never be a valid
        (name, provider) pair anywhere in the chain."""
        from applypilot.llm import _build_fallback_chain
        env = self._env(
            GEMINI_API_KEY="fake-gemini-key",
            APPLYPILOT_LOCAL_LLM_URL="http://localhost:11434/v1",
            APPLYPILOT_LOCAL_LLM_MODEL="qwen3:8b",
        )
        with patch.dict("os.environ", env, clear=True), \
             patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("qwen3:8b", quality=False)
        self.assertFalse(
            any(e.name == "qwen3:8b" and e.provider == "gemini" for e in chain)
        )

    def test_local_entry_still_present_as_final_fallback(self):
        """The local model must still appear -- exactly once, under the
        correct provider, and last in the chain (final fallback)."""
        from applypilot.llm import _build_fallback_chain
        env = self._env(
            GEMINI_API_KEY="fake-gemini-key",
            APPLYPILOT_LOCAL_LLM_URL="http://localhost:11434/v1",
            APPLYPILOT_LOCAL_LLM_MODEL="qwen3:8b",
        )
        with patch.dict("os.environ", env, clear=True), \
             patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("qwen3:8b", quality=False)
        local_entries = [e for e in chain if e.provider == "local"]
        self.assertEqual(len(local_entries), 1)
        self.assertEqual(local_entries[0].name, "qwen3:8b")
        self.assertIs(chain[-1], local_entries[0])
        # The regular gemini fallback chain must still be present and intact
        # ("existing cloud-provider configuration continues to work unchanged").
        gemini_names = [e.name for e in chain if e.provider == "gemini"]
        self.assertEqual(gemini_names, ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"])

    def test_gemini_only_chain_unaffected_when_local_not_configured(self):
        from applypilot.llm import _build_fallback_chain
        env = self._env(GEMINI_API_KEY="fake-gemini-key")
        with patch.dict("os.environ", env, clear=True), \
             patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-3.6-flash", quality=False)
        self.assertTrue(all(e.provider == "gemini" for e in chain))
        self.assertEqual(chain[0].name, "gemini-3.6-flash")

    def test_genuinely_unlisted_gemini_model_still_added_as_gemini(self):
        """A real (not-yet-hardcoded) Gemini model name that is NOT the
        configured local model must still be treated as a Gemini primary --
        this pre-existing behavior must survive the fix."""
        from applypilot.llm import _build_fallback_chain
        env = self._env(GEMINI_API_KEY="fake-gemini-key")
        with patch.dict("os.environ", env, clear=True), \
             patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-4.0-custom-preview", quality=False)
        self.assertEqual(chain[0].name, "gemini-4.0-custom-preview")
        self.assertEqual(chain[0].provider, "gemini")

    def test_test_local_cmd_probe_only_hits_local_provider(self):
        """cli.py's `test-local` command must isolate its chat probe to the
        local endpoint -- it must never silently succeed via a configured
        cloud fallback (which would mask a broken local setup as 'working')."""
        import applypilot.cli as cli_mod

        env = {"GEMINI_API_KEY": "fake-gemini-key",
               "APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1",
               "APPLYPILOT_LOCAL_LLM_MODEL": "qwen3:8b"}
        with patch.dict("os.environ", env, clear=False), \
             patch("applypilot.llm.local_available", return_value=True), \
             patch("applypilot.llm.LLMClient._try_openai_compat",
                   return_value='{"status":"ok"}') as mock_try:
            cli_mod.test_local_cmd()
        self.assertTrue(mock_try.called)
        called_entry = mock_try.call_args[0][0]
        self.assertEqual(called_entry.provider, "local")
        self.assertEqual(called_entry.name, "qwen3:8b")


# ---------------------------------------------------------------------------
# 15. debug-local-plan CLI: --job-id / --url selection and validation
#
# pytest-style (not unittest.TestCase) so the tmp_db/seed_job fixtures from
# conftest.py are usable directly. debug_local_plan_cmd is called as a plain
# function (not through typer's CLI dispatch) -- its typer.Option(...)
# defaults only apply when invoked via the CLI, so calling it directly with
# explicit url=/job_id= kwargs exercises the exact same command body.
# ---------------------------------------------------------------------------

def _empty_debug_result():
    return {
        "plan": {
            "requirements": [], "skills_to_emphasize": [], "bullets_to_prioritize": [],
            "bullets_to_deemphasize": [], "matching_projects": [], "matching_certifications": [],
            "summary_focus": [], "keyword_targets": [], "safe_rewrites": [],
            "unsupported_requirements": [], "_warnings": [],
        },
        "evidence": [], "requirement_lines": [],
    }


def test_debug_local_plan_both_url_and_job_id_is_a_clean_error():
    import applypilot.cli as cli_mod
    with pytest.raises(cli_mod.typer.Exit) as excinfo:
        cli_mod.debug_local_plan_cmd(url="https://example.com/j1", job_id=5)
    assert excinfo.value.exit_code == 1


def test_debug_local_plan_neither_url_nor_job_id_is_a_clean_error():
    import applypilot.cli as cli_mod
    with pytest.raises(cli_mod.typer.Exit) as excinfo:
        cli_mod.debug_local_plan_cmd(url=None, job_id=None)
    assert excinfo.value.exit_code == 1


def test_debug_local_plan_job_id_resolves_correct_job(tmp_db, seed_job, monkeypatch):
    import applypilot.cli as cli_mod

    conn = tmp_db()
    seed_job(conn, url_suffix="job-a", title="Job A")
    row_b = seed_job(conn, url_suffix="job-b", title="Job B")
    rowid_b = conn.execute("SELECT rowid FROM jobs WHERE url = ?", (row_b["url"],)).fetchone()[0]

    monkeypatch.setenv("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434/v1")
    captured: dict = {}
    monkeypatch.setattr("applypilot.config.load_profile", lambda: {"resume_facts": {}})
    monkeypatch.setattr(
        "applypilot.scoring.local_tailor.debug_plan_for_job",
        lambda job, profile: (captured.__setitem__("job", job), _empty_debug_result())[1],
    )

    cli_mod.debug_local_plan_cmd(url=None, job_id=rowid_b)

    assert captured["job"]["title"] == "Job B"


def test_debug_local_plan_url_still_works(tmp_db, seed_job, monkeypatch):
    """--url continues to work exactly as before (partial-match lookup)."""
    import applypilot.cli as cli_mod

    conn = tmp_db()
    row = seed_job(conn, url_suffix="job-c", title="Job C")

    monkeypatch.setenv("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434/v1")
    captured: dict = {}
    monkeypatch.setattr("applypilot.config.load_profile", lambda: {"resume_facts": {}})
    monkeypatch.setattr(
        "applypilot.scoring.local_tailor.debug_plan_for_job",
        lambda job, profile: (captured.__setitem__("job", job), _empty_debug_result())[1],
    )

    cli_mod.debug_local_plan_cmd(url=row["url"], job_id=None)

    assert captured["job"]["title"] == "Job C"


def test_debug_local_plan_unknown_job_id_is_a_clean_error(tmp_db, seed_job, monkeypatch):
    import applypilot.cli as cli_mod

    tmp_db()  # empty DB
    monkeypatch.setenv("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434/v1")
    with pytest.raises(cli_mod.typer.Exit) as excinfo:
        cli_mod.debug_local_plan_cmd(url=None, job_id=999999)
    assert excinfo.value.exit_code == 1


if __name__ == "__main__":
    unittest.main()
