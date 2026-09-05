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
        """include_local=True is explicit here: this test is about the
        append-local mechanism itself (used by the quality tier -- tailor/
        cover -- as a last-resort fallback), not the fast/quality routing
        split. See TestLocalExcludedFromFastTier for that."""
        from applypilot.llm import _build_fallback_chain

        env = {
            k: v
            for k, v in __import__("os").environ.items()
            if k
            not in (
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "DEEPSEEK_API_KEY",
                "LLM_URL",
                "APPLYPILOT_LOCAL_LLM_URL",
                "APPLYPILOT_LOCAL_LLM_MODEL",
            )
        }
        env["GEMINI_API_KEY"] = "fake-gemini-key"
        env["APPLYPILOT_LOCAL_LLM_URL"] = "http://localhost:11434/v1"
        env["APPLYPILOT_LOCAL_LLM_MODEL"] = "llama3.2"
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-3.6-flash", quality=False, include_local=True)
        local_entries = [e for e in chain if e.provider == "local"]
        self.assertEqual(len(local_entries), 1)
        self.assertEqual(local_entries[0].name, "llama3.2")
        self.assertEqual(local_entries[0].base_url, "http://localhost:11434/v1")
        # Local must come after all cloud entries (last resort)
        self.assertIs(chain[-1], local_entries[0])

    def test_fallback_chain_no_local_entry_when_not_configured(self):
        from applypilot.llm import _build_fallback_chain

        env = {
            k: v
            for k, v in __import__("os").environ.items()
            if k
            not in (
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "DEEPSEEK_API_KEY",
                "LLM_URL",
                "APPLYPILOT_LOCAL_LLM_URL",
            )
        }
        env["GEMINI_API_KEY"] = "fake-gemini-key"
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-3.6-flash", quality=False)
        self.assertFalse(any(e.provider == "local" for e in chain))

    def test_local_only_chain_when_no_cloud_keys(self):
        """Local can be the ONLY provider (no cloud keys at all) -- doesn't
        raise, when the caller explicitly opts into local (include_local=True;
        the fast tier's implicit default is now False -- see
        TestLocalExcludedFromFastTier)."""
        from applypilot.llm import _build_fallback_chain

        env = {
            k: v
            for k, v in __import__("os").environ.items()
            if k
            not in (
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "DEEPSEEK_API_KEY",
                "LLM_URL",
                "APPLYPILOT_LOCAL_LLM_URL",
            )
        }
        env["APPLYPILOT_LOCAL_LLM_URL"] = "http://localhost:11434/v1"
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("llama3.2", quality=False, include_local=True)
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
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"status":"ok"}'}}]}
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
            "message": {"content": ('{"matches":[{"r":1,"e":[2]},{"r":2,"e":[3]},{"r":3,"e":[1]}]}')}
        }
        job = {
            "title": "Data Engineer",
            "full_description": (
                "- Python experience required\n- SQL experience required\n- ETL pipeline experience required\n"
            ),
        }
        profile = {
            "resume_facts": {},
            "skills_inventory": [
                {"name": "Python", "resume_allowed": True, "relevance_categories": ["python"]},
                {"name": "SQL", "resume_allowed": True, "relevance_categories": ["sql"]},
            ],
            "project_inventory": [
                {
                    "name": "ETL Pipeline",
                    "resume_allowed": True,
                    "relevance_categories": ["etl"],
                    "factual_concepts": ["ETL"],
                },
            ],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp):
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
# 2b. Ollama native URL normalization (2026-08-31 fix)
#
# Root cause reproduced live: APPLYPILOT_LOCAL_LLM_URL is documented for,
# and llm.py's local_openai_base_url() always normalizes TO, the OpenAI-
# compatible /v1 convention -- but get_local_tailoring_plan() posts
# straight to Ollama's native /api/chat, which lives at the bare server
# root. A URL configured the documented way (WITH /v1) previously built
# {url}/v1/api/chat here, a route Ollama doesn't serve -- 404, confirmed
# against a real Ollama instance running `debug-local-plan`. None of the
# pre-existing tests in this file caught it: they all mock httpx.post at
# the function level and only inspect the mocked return value, never the
# URL the call was actually made with. These tests assert on the URL
# specifically to close that gap.
# ---------------------------------------------------------------------------


class TestOllamaNativeBaseUrlNormalization(unittest.TestCase):
    def test_strips_trailing_v1(self):
        from applypilot.scoring.local_tailor import _ollama_native_base_url

        self.assertEqual(
            _ollama_native_base_url("http://localhost:11434/v1"),
            "http://localhost:11434",
        )

    def test_strips_trailing_v1_with_trailing_slash(self):
        from applypilot.scoring.local_tailor import _ollama_native_base_url

        self.assertEqual(
            _ollama_native_base_url("http://localhost:11434/v1/"),
            "http://localhost:11434",
        )

    def test_bare_root_unaffected(self):
        """The already-correct, default-matching form must be untouched."""
        from applypilot.scoring.local_tailor import _ollama_native_base_url

        self.assertEqual(
            _ollama_native_base_url("http://localhost:11434"),
            "http://localhost:11434",
        )

    def test_bare_root_with_trailing_slash(self):
        from applypilot.scoring.local_tailor import _ollama_native_base_url

        self.assertEqual(
            _ollama_native_base_url("http://localhost:11434/"),
            "http://localhost:11434",
        )

    def test_does_not_strip_v1_that_is_not_a_trailing_segment(self):
        """A host/path that merely CONTAINS "v1" elsewhere (not as the
        trailing path segment) must not be mangled."""
        from applypilot.scoring.local_tailor import _ollama_native_base_url

        self.assertEqual(
            _ollama_native_base_url("http://v1.example.com:11434"),
            "http://v1.example.com:11434",
        )

    def test_get_local_tailoring_plan_posts_to_bare_root_when_v1_configured(self):
        """End-to-end regression for the exact reported/reproduced bug:
        with APPLYPILOT_LOCAL_LLM_URL including /v1 (the documented,
        test-local-matching convention), the actual outgoing POST must
        hit the bare-root /api/chat, not /v1/api/chat."""
        from applypilot.scoring import local_tailor

        job, profile = _matchable_job_and_profile()
        resume_text = "resume text"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": '{"matches":[]}'}}

        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp) as mock_post:
            local_tailor.get_local_tailoring_plan(resume_text, job, profile)

        self.assertTrue(mock_post.called)
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "http://localhost:11434/api/chat")
        self.assertNotEqual(called_url, "http://localhost:11434/v1/api/chat")

    def test_get_local_tailoring_plan_posts_to_bare_root_when_bare_root_configured(self):
        """The already-working configuration (no /v1) must keep working
        identically -- regression guard for the non-broken case."""
        from applypilot.scoring import local_tailor

        job, profile = _matchable_job_and_profile()
        resume_text = "resume text"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": '{"matches":[]}'}}

        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp) as mock_post:
            local_tailor.get_local_tailoring_plan(resume_text, job, profile)

        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "http://localhost:11434/api/chat")


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
        with patch.dict("os.environ", env, clear=False), patch("httpx.get", side_effect=ConnectionError("refused")):
            self.assertFalse(llm_mod.local_available())

    def test_get_local_tailoring_plan_returns_none_on_connection_error(self):
        from applypilot.scoring import local_tailor

        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", side_effect=ConnectionError("refused")):
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        self.assertIsNone(plan)

    def test_tailor_gracefully_skips_plan_when_local_unavailable(self):
        """_tailor_one_job must not raise when local plan generation fails --
        it should proceed with tailor_resume(local_plan=None)."""
        from applypilot.scoring import tailor as tailor_mod

        job = {
            "url": "https://example.com/j1",
            "title": "Engineer",
            "site": "test",
            "full_description": "desc",
            "fit_score": 9,
        }
        profile = {"personal": {"full_name": "Jane Doe"}, "resume_facts": {}}

        with (
            patch.dict(
                "os.environ", {"APPLYPILOT_LOCAL_PLAN": "1", "APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
            ),
            patch("applypilot.scoring.local_tailor.get_local_tailoring_plan", side_effect=Exception("boom")),
            patch.object(
                tailor_mod,
                "tailor_resume",
                return_value=("RESUME TEXT", {"status": "approved", "attempts": 1, "approved_facts": []}),
            ) as mock_tailor,
            patch.object(tailor_mod, "TAILORED_DIR", Path("/tmp/applypilot-test-tailor")),
        ):
            tailor_mod.TAILORED_DIR.mkdir(parents=True, exist_ok=True)
            result = tailor_mod._tailor_one_job(job, "Original resume text.", profile)

        self.assertEqual(result["status"], "approved")
        # local_plan kwarg was passed as None since generation failed
        _, kwargs = mock_tailor.call_args
        self.assertIsNone(kwargs.get("local_plan"))


class TestLocalFirstObservability(unittest.TestCase):
    """INFO-level observability for --local-first.

    Regression coverage for a real integration gap: every log statement
    inside get_local_tailoring_plan (and _tailor_one_job's own except
    block) was at DEBUG, while cli.py's root logger defaults to INFO --
    so a --local-first run produced zero lines in the log file grep'able
    by "local|Ollama|planner|plan" REGARDLESS of whether the planner ran
    successfully, found nothing useful, or failed outright. These tests
    exercise the four outcomes _tailor_one_job must now report at INFO
    (WARNING for the exception case), while confirming the local-first
    failure still never blocks cloud tailoring from proceeding.
    """

    def _job_and_profile(self):
        job = {
            "url": "https://example.com/j1",
            "title": "Help Desk Technician",
            "site": "test",
            "full_description": "desc",
            "fit_score": 9,
        }
        profile = {"personal": {"full_name": "Jane Doe"}, "resume_facts": {}}
        return job, profile

    def _patched_tailor_resume(self, tailor_mod):
        return patch.object(
            tailor_mod,
            "tailor_resume",
            return_value=("RESUME TEXT", {"status": "approved", "attempts": 1, "approved_facts": []}),
        )

    def test_logs_info_when_plan_accepted(self):
        from applypilot.scoring import tailor as tailor_mod

        job, profile = self._job_and_profile()
        plan = {
            "requirements": [{"requirement": "Python", "supported": True}, {"requirement": "SQL", "supported": False}]
        }

        with (
            patch.dict("os.environ", {"APPLYPILOT_LOCAL_PLAN": "1"}),
            patch("applypilot.scoring.local_tailor.get_local_tailoring_plan", return_value=plan),
            patch(
                "applypilot.scoring.local_tailor.format_local_plan_for_cloud",
                return_value="Job requirements:\n- Python -- supported",
            ),
            self._patched_tailor_resume(tailor_mod) as mock_tailor,
            patch.object(tailor_mod, "TAILORED_DIR", Path("/tmp/applypilot-test-tailor")),
        ):
            tailor_mod.TAILORED_DIR.mkdir(parents=True, exist_ok=True)
            with self.assertLogs("applypilot.scoring.tailor", level="INFO") as cm:
                tailor_mod._tailor_one_job(job, "Original resume text.", profile)

        joined = " ".join(cm.output)
        self.assertIn("plan accepted", joined)
        self.assertIn("1/2 requirement", joined)
        # The accepted plan text must actually reach tailor_resume.
        _, kwargs = mock_tailor.call_args
        self.assertIn("supported", kwargs.get("local_plan") or "")

    def test_logs_info_when_no_usable_plan(self):
        """A plan the model DID produce but that has nothing to say (e.g.
        every requirement unsupported, or a benefits-only posting) must be
        distinguished from an outright planner failure."""
        from applypilot.scoring import tailor as tailor_mod

        job, profile = self._job_and_profile()
        plan = {"requirements": []}

        with (
            patch.dict("os.environ", {"APPLYPILOT_LOCAL_PLAN": "1"}),
            patch("applypilot.scoring.local_tailor.get_local_tailoring_plan", return_value=plan),
            patch("applypilot.scoring.local_tailor.format_local_plan_for_cloud", return_value=""),
            self._patched_tailor_resume(tailor_mod) as mock_tailor,
            patch.object(tailor_mod, "TAILORED_DIR", Path("/tmp/applypilot-test-tailor")),
        ):
            tailor_mod.TAILORED_DIR.mkdir(parents=True, exist_ok=True)
            with self.assertLogs("applypilot.scoring.tailor", level="INFO") as cm:
                tailor_mod._tailor_one_job(job, "Original resume text.", profile)

        joined = " ".join(cm.output)
        self.assertIn("no usable plan", joined)
        _, kwargs = mock_tailor.call_args
        self.assertIsNone(kwargs.get("local_plan"))

    def test_logs_info_when_planner_returns_none(self):
        """get_local_tailoring_plan returning None (unreachable/timed out/
        unparseable -- it already logs the specific cause at WARNING
        internally) must still produce a visible, at-a-glance INFO line
        confirming local-first ran and got nothing, distinct from the
        'model answered but had nothing useful' case above."""
        from applypilot.scoring import tailor as tailor_mod

        job, profile = self._job_and_profile()

        with (
            patch.dict("os.environ", {"APPLYPILOT_LOCAL_PLAN": "1"}),
            patch("applypilot.scoring.local_tailor.get_local_tailoring_plan", return_value=None),
            self._patched_tailor_resume(tailor_mod) as mock_tailor,
            patch.object(tailor_mod, "TAILORED_DIR", Path("/tmp/applypilot-test-tailor")),
        ):
            tailor_mod.TAILORED_DIR.mkdir(parents=True, exist_ok=True)
            with self.assertLogs("applypilot.scoring.tailor", level="INFO") as cm:
                tailor_mod._tailor_one_job(job, "Original resume text.", profile)

        joined = " ".join(cm.output)
        self.assertIn("planner unavailable", joined)
        _, kwargs = mock_tailor.call_args
        self.assertIsNone(kwargs.get("local_plan"))

    def test_logs_warning_on_planner_exception_and_still_tailors(self):
        """A raised exception must produce a concise WARNING (type + message,
        not a full traceback at that level) and must NOT prevent cloud
        tailoring from proceeding."""
        from applypilot.scoring import tailor as tailor_mod

        job, profile = self._job_and_profile()

        with (
            patch.dict("os.environ", {"APPLYPILOT_LOCAL_PLAN": "1"}),
            patch("applypilot.scoring.local_tailor.get_local_tailoring_plan", side_effect=ValueError("boom")),
            self._patched_tailor_resume(tailor_mod) as mock_tailor,
            patch.object(tailor_mod, "TAILORED_DIR", Path("/tmp/applypilot-test-tailor")),
        ):
            tailor_mod.TAILORED_DIR.mkdir(parents=True, exist_ok=True)
            with self.assertLogs("applypilot.scoring.tailor", level="WARNING") as cm:
                result = tailor_mod._tailor_one_job(job, "Original resume text.", profile)

        self.assertEqual(result["status"], "approved")
        joined = " ".join(cm.output)
        self.assertIn("ValueError", joined)
        self.assertIn("boom", joined)
        _, kwargs = mock_tailor.call_args
        self.assertIsNone(kwargs.get("local_plan"))
        mock_tailor.assert_called_once()

    def test_no_local_first_logging_when_disabled(self):
        """With APPLYPILOT_LOCAL_PLAN unset, none of the local-first log
        lines should appear at all -- it must stay fully opt-in."""
        from applypilot.scoring import tailor as tailor_mod

        job, profile = self._job_and_profile()

        env = {k: v for k, v in __import__("os").environ.items() if k != "APPLYPILOT_LOCAL_PLAN"}
        with (
            patch.dict("os.environ", env, clear=True),
            self._patched_tailor_resume(tailor_mod),
            patch.object(tailor_mod, "TAILORED_DIR", Path("/tmp/applypilot-test-tailor")),
        ):
            tailor_mod.TAILORED_DIR.mkdir(parents=True, exist_ok=True)
            with self.assertLogs("applypilot.scoring.tailor", level="INFO") as cm:
                # at least one INFO line must exist elsewhere in
                # _tailor_one_job for assertLogs not to raise; if this
                # becomes a problem, swap for a plain caplog-free check.
                tailor_mod.log.info("marker so assertLogs has something to capture")
                tailor_mod._tailor_one_job(job, "Original resume text.", profile)

        self.assertFalse(any("local-first" in line for line in cm.output))


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

        fake_chain = [ModelEntry(f"cloud-{i}", "openai_compat", "https://fake.api/v1", "fake-key") for i in range(n)]
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

        with patch.object(client._client, "post") as mock_post, self.assertRaises(RuntimeError):
            client.chat([{"role": "user", "content": "hi"}])
        # No HTTP call should have been attempted -- both entries are still
        # marked exhausted and there's no local fallback in this chain.
        mock_post.assert_not_called()

    def test_short_cooldown_entries_do_reset_when_all_exhausted(self):
        """Short-term (<=5min) rate-limit cooldowns ARE cleared when every
        entry is temporarily exhausted -- only long daily blocks persist."""
        client = self._client(2)
        now = time.time()
        client._exhausted["cloud-0"] = now + 30  # short transient cooldown
        client._exhausted["cloud-1"] = now + 60  # short transient cooldown

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
        mock_resp.json.return_value = {"message": {"content": "Sure! Here's your plan: it looks great."}}
        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp):
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
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp):
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

        with (
            patch("applypilot.scoring.tailor.get_stage_client", return_value=mock_client),
            patch("applypilot.scoring.tailor.is_local_configured", return_value=False),
            patch(
                "applypilot.scoring.tailor.judge_tailored_resume",
                return_value={"passed": True, "verdict": "SKIP", "issues": "none", "raw": "n/a"},
            ),
        ):
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
            "message": {
                "content": ('<think>\nThe candidate has Python experience.\n</think>\n{"matches":[{"r":1,"e":[1]}]}')
            }
        }
        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp):
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

        with (
            patch("applypilot.scoring.tailor.get_stage_client", return_value=mock_client),
            patch("applypilot.scoring.tailor.is_local_configured", return_value=False),
        ):
            _tailored, report = tailor_mod.tailor_resume("Original resume.", job, profile, max_retries=1)

        self.assertEqual(report["status"], "failed_validation")


# ---------------------------------------------------------------------------
# 9. Existing cloud-only behavior remains functional
# ---------------------------------------------------------------------------


class TestExistingCloudBehaviorUnaffected(unittest.TestCase):
    def test_gemini_only_chain_unaffected_by_local_support(self):
        from applypilot.llm import _build_fallback_chain

        env = {
            k: v
            for k, v in __import__("os").environ.items()
            if k
            not in (
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "DEEPSEEK_API_KEY",
                "LLM_URL",
                "APPLYPILOT_LOCAL_LLM_URL",
            )
        }
        env["GEMINI_API_KEY"] = "fake-key"
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
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

        with (
            patch("applypilot.scoring.tailor.get_stage_client", return_value=mock_client),
            patch("applypilot.scoring.tailor.is_local_configured", return_value=False),
        ):
            _tailored, report = tailor_mod.tailor_resume("Original resume.", job, profile, max_retries=1)

        self.assertEqual(report["status"], "approved")


# ---------------------------------------------------------------------------
# 10. Deterministic requirement-line extraction (no LLM, no embeddings)
# ---------------------------------------------------------------------------


class TestDeterministicRequirementExtraction(unittest.TestCase):
    def test_extracts_bulleted_lines(self):
        from applypilot.scoring.local_tailor import _extract_requirement_lines

        desc = (
            "About the role:\n- Experience with Python required\n- SQL is a plus\n- Familiarity with Docker preferred\n"
        )
        lines = _extract_requirement_lines(desc)
        texts = [l["text"] for l in lines]
        self.assertIn("Experience with Python required", texts)
        self.assertIn("SQL is a plus", texts)
        self.assertIn("Familiarity with Docker preferred", texts)

    def test_tags_required_vs_preferred(self):
        from applypilot.scoring.local_tailor import _extract_requirement_lines

        desc = "- Python experience is required\n- Docker knowledge preferred\n- Owns a pet dinosaur\n"
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
        self.assertEqual([l["text"] for l in lines], ["3+ years of Python experience required"])
        self.assertEqual(dropped, ["Opportunities for Career Growth", "Tuition Educational Assistance Programs"])


# ---------------------------------------------------------------------------
# Markerless paragraph-fallback extraction (2026-08-25, Direction 1 of the
# extraction audit)
#
# Root cause (measured against the live database, see the audit report):
# three real NO_SUPPORTED_EVIDENCE incidents (Axon, Twilio, Pinterest) and
# ~1,478 active Workday-sourced rows all extracted ZERO requirement lines
# via _REQUIREMENT_MARKER_RE, because their itemized requirements were
# flattened by HTML-to-text conversion to plain, unmarked lines -- not
# because evidence matching rejected the candidate's qualifications. This
# fallback recovers that lost requirement TEXT only; it does not change
# what counts as evidence for a requirement (_pair_candidate_evidence,
# _auto_resolve_requirements, claim/agency/causal/metric safety checks are
# all untouched).
# ---------------------------------------------------------------------------


class TestParagraphFallbackExtraction(unittest.TestCase):
    def test_a_bare_terms_separated_by_blank_lines_are_extracted(self):
        """Test A: markerless, blank-line-separated bare skill terms."""
        from applypilot.scoring.local_tailor import _extract_requirement_lines

        desc = "Python\n\nDistributed systems\n\nKubernetes\n\nObservability\n"
        lines = _extract_requirement_lines(desc)
        self.assertEqual(len(lines), 4)
        texts = {l["text"] for l in lines}
        self.assertEqual(texts, {"Python", "Distributed systems", "Kubernetes", "Observability"})

    def test_b_fallback_never_fires_when_marker_extraction_found_anything(self):
        """Test B: the single most important test in this suite -- marker
        extraction winning must mean the fallback contributes NOTHING, even
        when markerless-looking lines sit right next to the marked one."""
        from applypilot.scoring.local_tailor import _extract_requirement_lines

        desc = "- Experience with Python programming\n\nDistributed systems\n\nKubernetes\n\nSomething else entirely\n"
        lines = _extract_requirement_lines(desc)
        texts = [l["text"] for l in lines]
        self.assertEqual(texts, ["Experience with Python programming"])
        self.assertNotIn("Distributed systems", texts)
        self.assertNotIn("Kubernetes", texts)

    def test_c_benefit_paragraphs_remain_excluded(self):
        """Test C: reuses _is_benefit_line, not a second classifier."""
        from applypilot.scoring.local_tailor import _split_requirement_lines

        desc = (
            "Competitive salary\n\nMedical, dental, and vision insurance\n\nGenerous paid time off\n\n401(k) matching\n"
        )
        lines, dropped = _split_requirement_lines(desc)
        self.assertEqual(lines, [])
        self.assertEqual(len(dropped), 4)

    def test_d_multi_sentence_prose_paragraphs_are_rejected(self):
        """Test D: ordinary multi-sentence employer prose, even several
        paragraphs of it (enough to clear a naive count-only gate), must
        never become requirement candidates."""
        from applypilot.scoring.local_tailor import _extract_requirement_lines

        desc = (
            "We are a rapidly growing company committed to innovation. "
            "Our team loves solving hard problems together.\n\n"
            "Our employees receive great support and mentorship. "
            "Everyone is encouraged to grow their career here.\n\n"
            "We believe in doing right by our customers. "
            "That philosophy guides everything we build.\n\n"
            "Diversity and inclusion are core to how we operate. "
            "We want everyone to feel like they belong.\n"
        )
        self.assertEqual(_extract_requirement_lines(desc), [])

    def test_e_existing_marker_based_fixtures_are_unchanged(self):
        """Test E: regression -- the refactor into _extract_marker_lines/
        _extract_paragraph_lines must not change marker-path behavior."""
        from applypilot.scoring.local_tailor import _extract_requirement_lines

        desc = (
            "About the role:\n- Experience with Python required\n- SQL is a plus\n- Familiarity with Docker preferred\n"
        )
        lines = _extract_requirement_lines(desc)
        texts = [l["text"] for l in lines]
        self.assertIn("Experience with Python required", texts)
        self.assertIn("SQL is a plus", texts)
        self.assertIn("Familiarity with Docker preferred", texts)

    def test_f_real_workday_fixture_recovers_requirements(self):
        """Test F: a real Workday posting (captured 2026-08-25), paragraph-
        per-requirement with blank-line separation and no marker
        characters at all -- the single largest affected source (~1,478
        active rows) identified by the audit."""
        from applypilot.scoring.local_tailor import _extract_marker_lines, _split_requirement_lines

        desc = (
            "Your work days are brighter here.\n\n"
            "We're obsessed with making hard work pay off, for our people, "
            "our customers, and the world around us.\n\n"
            "Basic Qualifications:\n\n"
            "4+ years in deploying large-scale, complex applications, with "
            "3+ years in cloud-based deployments on AWS or GCP\n\n"
            "2+ years experience with automation tools and languages, "
            "including Ansible, Python, Go, Chef, and Puppet\n\n"
            "2+ years experience with networks, servers, storage, and "
            "operating systems in cloud environments\n\n"
            "Hands-on experience with infrastructure automation tools like "
            "Chef, Puppet, and Ansible\n"
        )
        marker_lines, _ = _extract_marker_lines(desc)
        self.assertEqual(marker_lines, [])

        lines, _dropped = _split_requirement_lines(desc)
        self.assertGreater(len(lines), 0)
        texts = " ".join(l["text"] for l in lines)
        self.assertIn("Ansible", texts)
        self.assertIn("AWS", texts)

    def test_g_extraction_does_not_weaken_the_evidence_boundary(self):
        """Test G: a markerless requirement is extracted, but with no
        matching profile evidence it must remain unsupported -- proves the
        fallback only recovers TEXT, never invents a match."""
        from applypilot.scoring.local_tailor import (
            _auto_resolve_requirements,
            _split_requirement_lines,
            rank_profile_evidence,
        )

        desc = (
            "Experience with Kubernetes orchestration\n\n"
            "Experience with distributed tracing systems\n\n"
            "Experience with service mesh architectures\n"
        )
        lines, _dropped = _split_requirement_lines(desc)
        self.assertEqual(len(lines), 3)

        profile_no_k8s_evidence = {
            "experience_inventory": [
                {
                    "name": "Retail Associate",
                    "resume_allowed": True,
                    "relevance_categories": ["customer service"],
                    "description": "Assisted customers on the sales floor.",
                },
            ],
            "project_inventory": [],
            "skills_inventory": [],
            "certifications": [],
        }
        ranked = rank_profile_evidence({"title": "SRE", "full_description": desc}, profile_no_k8s_evidence, top_n=6)
        self.assertEqual(ranked, [])  # nothing in this profile matches at all
        resolved, candidates = _auto_resolve_requirements(lines, ranked)
        for i in range(1, len(lines) + 1):
            self.assertEqual(resolved.get(i, []), [])
            self.assertEqual(candidates.get(i, []), [])

    def test_h_extraction_lets_legitimate_evidence_resolve(self):
        """Test H: the mirror case -- when the profile DOES contain
        matching evidence, the existing (unmodified) matcher must still be
        able to resolve it once the text is recovered."""
        from applypilot.scoring.local_tailor import (
            _auto_resolve_requirements,
            _split_requirement_lines,
            rank_profile_evidence,
        )

        desc = (
            "Experience with Python automation\n\n"
            "Experience with distributed tracing systems\n\n"
            "Experience with service mesh architectures\n"
        )
        lines, _dropped = _split_requirement_lines(desc)
        self.assertEqual(len(lines), 3)

        profile_with_python = {
            "experience_inventory": [
                {
                    "name": "Automation Engineer",
                    "resume_allowed": True,
                    "relevance_categories": ["automation", "python"],
                    "description": "Built Python automation scripts.",
                },
            ],
            "project_inventory": [],
            "skills_inventory": [],
            "certifications": [],
        }
        job = {"title": "Platform Engineer", "full_description": desc}
        ranked = rank_profile_evidence(job, profile_with_python, top_n=6)
        self.assertTrue(any(r["name"] == "Automation Engineer" for r in ranked))
        resolved, _candidates = _auto_resolve_requirements(lines, ranked)
        python_req_idx = next(i for i, l in enumerate(lines, start=1) if "Python" in l["text"])
        self.assertTrue(resolved.get(python_req_idx))


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
            {
                "name": "Python",
                "evidence_level": "demonstrated",
                "resume_allowed": True,
                "relevance_categories": ["python", "automation"],
            },
            {
                "name": "Photography",
                "evidence_level": "demonstrated",
                "resume_allowed": True,
                "relevance_categories": ["media"],
            },
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
            "experience_inventory": [
                {
                    "name": "National Tire and Battery",
                    "relevance_categories": ["retail"],
                    "resume_allowed": True,
                }
            ],
            "project_inventory": [],
            "skills_inventory": [],
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

    def test_format_evidence_for_prompt_includes_experience_constraints(self):
        """2026-09-03 regression: constraints were only ever rendered for
        project_inventory entries -- an experience_inventory item's
        constraints (e.g. "keep claims modest") silently never reached the
        local model's prompt at all."""
        from applypilot.scoring.local_tailor import format_evidence_for_prompt, rank_profile_evidence

        profile = {
            "experience_inventory": [
                {
                    "name": "Freelance Photography",
                    "description": "Competent use of DSLR equipment for photography.",
                    "constraints": ["Keep claims modest; do not describe extensive professional experience."],
                }
            ],
            "project_inventory": [],
            "skills_inventory": [],
        }
        job = self._job("Photography and videography experience with DSLR equipment.")
        ranked = rank_profile_evidence(job, profile, top_n=10)
        rendered = format_evidence_for_prompt(ranked)
        self.assertIn("Freelance Photography", rendered)
        self.assertIn("Keep claims modest; do not describe extensive professional experience.", rendered)

    def test_certifications_are_included_as_evidence(self):
        """certifications is a fourth deterministic evidence source (same
        shape as the other inventories) -- so matching_certifications can be
        derived from real matches instead of asked of the model as free
        text, which is exactly where a small model is most likely to
        hallucinate a certification name."""
        from applypilot.scoring.local_tailor import rank_profile_evidence

        profile = _sample_profile()
        profile["certifications"] = [
            {
                "name": "CompTIA A+",
                "resume_allowed": True,
                "relevance_categories": ["IT", "help desk"],
            }
        ]
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
# 11b. Generic single-word categories excluded at the retrieval source
#
# Regression coverage for a precision bug traced to profile.json itself,
# not to any matching heuristic: CompTIA A+ is tagged with the bare
# relevance_category "IT" (2 characters -- literally the pronoun "it" once
# normalized) and Python with the bare category "technical" (broad enough
# to describe almost any domain's work, not just software). Both slipped
# through because relevance_categories/factual_concepts were the one term
# source _item_terms never filtered -- name-derived words already had a
# length/stopword filter (_NAME_TOKEN_STOPWORDS), categories didn't. See
# _GENERIC_EVIDENCE_TERMS / _is_generic_evidence_term, which now filters
# every term source uniformly, at the single place they're all collected
# -- not as a special case bolted onto _pair_candidate_evidence.
# ---------------------------------------------------------------------------


class TestGenericEvidenceTermFiltering(unittest.TestCase):
    def setUp(self):
        # _idf_weights_cache/_idf_weights_load_failed are module-level
        # globals (same class of leak as the LLM-exhaustion-state file
        # found and fixed earlier this session) -- reset before AND after
        # every test so one test's simulated load failure/success can't
        # leak into another.
        from applypilot.scoring import local_tailor

        local_tailor._idf_weights_cache = None
        local_tailor._idf_weights_load_failed = False

    def tearDown(self):
        from applypilot.scoring import local_tailor

        local_tailor._idf_weights_cache = None
        local_tailor._idf_weights_load_failed = False

    def test_bare_it_category_is_not_a_usable_term(self):
        from applypilot.scoring.local_tailor import _item_terms

        item = {"name": "CompTIA A+", "relevance_categories": ["IT", "help desk"]}
        terms = _item_terms(item)
        self.assertNotIn("it", terms)
        self.assertIn("help desk", terms)

    def test_bare_technical_category_is_not_a_usable_term(self):
        from applypilot.scoring.local_tailor import _item_terms

        item = {"name": "Python Tooling", "relevance_categories": ["technical", "automation", "data", "OCR"]}
        terms = _item_terms(item)
        self.assertNotIn("technical", terms)
        self.assertIn("automation", terms)
        self.assertIn("ocr", terms)
        # 2026-09-04: "data" (idf=1.40, real weight from idf_weights.json)
        # is now ALSO excluded by the new IDF-based generic-term check --
        # it appears in a huge fraction of job postings regardless of
        # domain, same failure shape as "technical", just not previously
        # discovered by a real bug. This is the new mechanism correctly
        # doing its job, not a regression -- "Python Tooling" is still
        # matchable via "automation"/"ocr", which are genuinely specific.
        self.assertNotIn("data", terms)

    def test_bare_ups_employer_name_is_not_a_usable_term(self):
        """2026-09-04, found via the rarity-weighting bake-off: the UPS
        experience_inventory entry's whole-name term "ups" (len==3, so it
        survived the length floor) word-boundary-matched ANY word ending in
        "-ups" -- "follow-ups", "backups", "startups" -- because a hyphen
        is a word boundary for \\b. UPS-the-employer must stay matchable
        via its other, more specific terms. (_term_in_text itself is a
        generic, term-agnostic word-boundary matcher and correctly still
        matches the literal string "ups" against "follow-ups" if ever
        asked to -- the fix belongs at term-CANDIDACY time, in _item_terms/
        _is_generic_evidence_term, which is what this test exercises.)"""
        from applypilot.scoring.local_tailor import _item_terms

        item = {"name": "UPS", "responsibilities": ["Handled package delivery, logistics, and warehouse operations daily."]}
        terms = _item_terms(item)
        self.assertNotIn("ups", terms)
        self.assertIn("package", terms)
        self.assertIn("warehouse", terms)
        # 2026-09-04: "operations" (idf=1.69) and "delivery" (idf=1.99) are
        # now ALSO excluded by the IDF-based generic-term check -- both
        # common across a huge fraction of postings regardless of domain.
        # UPS stays matchable via "package"/"warehouse"/"logistics", which
        # are genuinely more specific (idf 2.26/4.65/4.27 respectively).
        self.assertNotIn("operations", terms)

    def test_common_preposition_in_responsibilities_text_is_not_a_usable_term(self):
        """2026-09-04, found via a real production run: 388 jobs (out of
        1477 scanned) matched Alex Prosperity Group / UST Logistics
        evidence, including "Pay Increases throughout the first year of
        employment and annual merit increases" -- a benefits blurb from an
        unrelated job, not a skill requirement. Root cause: the entry's
        responsibilities text ("...Communicated clearly with customers
        throughout installations.") contributed the bare word "throughout"
        as a matchable term, and "throughout" is a common preposition with
        zero domain-discriminating signal -- it just happened to also
        appear in the unrelated job's benefits section."""
        from applypilot.scoring.local_tailor import _item_terms

        item = {
            "name": "Alex Prosperity Group / UST Logistics",
            "responsibilities": [
                "Communicated clearly with customers throughout installations.",
            ],
        }
        terms = _item_terms(item)
        self.assertNotIn("throughout", terms)
        self.assertIn("communicated", terms)
        # "customers" (idf=1.90) is excluded too, by the separate IDF-based
        # check added right after this fix -- see
        # test_idf_weights_load_real_file_and_flag_known_generic_word.
        # "communicated" remains as this item's matchable term here.

    def test_benefits_blurb_no_longer_matches_installation_evidence(self):
        """Integration-level reproduction of the exact real false positive
        via rank_profile_evidence, not just the unit-level _item_terms
        check above."""
        from applypilot.scoring.local_tailor import rank_profile_evidence

        job = {
            "title": "QC Lab Technician - 1st Shift",
            "full_description": (
                "Pay Increases throughout the first year of employment and annual merit increases.\n"
            ),
        }
        profile = {
            "experience_inventory": [
                {
                    "name": "Alex Prosperity Group / UST Logistics",
                    "resume_allowed": True,
                    "responsibilities": [
                        "Installed home appliances contracted through Lowe's",
                        "Performed hands-on installation work while following customer requirements "
                        "and established procedures.",
                        "Communicated clearly with customers throughout installations.",
                    ],
                },
            ],
            "project_inventory": [],
            "skills_inventory": [],
        }
        ranked = rank_profile_evidence(job, profile, top_n=10)
        self.assertEqual(ranked, [])

    def test_idf_weights_load_real_file_and_flag_known_generic_word(self):
        """Uses the REAL idf_weights.json shipped in src/applypilot/config/
        -- not mocked -- since the whole point is confirming the actual
        packaged artifact behaves as calibrated against real data."""
        from applypilot.scoring.local_tailor import _is_generic_evidence_term

        self.assertTrue(_is_generic_evidence_term("customer"))  # idf=1.80
        self.assertTrue(_is_generic_evidence_term("using"))  # idf=1.81

    def test_idf_weights_preserve_specific_single_word_terms(self):
        from applypilot.scoring.local_tailor import _is_generic_evidence_term

        self.assertFalse(_is_generic_evidence_term("troubleshooting"))  # idf=2.48
        self.assertFalse(_is_generic_evidence_term("python"))  # idf=2.68
        self.assertFalse(_is_generic_evidence_term("alignment"))  # idf=3.19

    def test_idf_cannot_see_multiword_terms_unaffected(self):
        from applypilot.scoring.local_tailor import _is_generic_evidence_term

        self.assertFalse(_is_generic_evidence_term("customer service"))

    def test_real_bug_customer_alone_no_longer_matches_unrelated_job(self):
        """Integration-level reproduction of the exact real false positive:
        a job requirement about AI tools, matched to Alex Prosperity Group
        installation evidence purely via the shared word "customer"."""
        from applypilot.scoring.local_tailor import rank_profile_evidence

        job = {
            "title": "Customer Experience Associate",
            "full_description": (
                "You actively use AI tools — Claude, Notion AI, research and workflow tools "
                "— and can demonstrate how you've applied them to get real work done.\n"
            ),
        }
        profile = {
            "experience_inventory": [
                {
                    "name": "Alex Prosperity Group / UST Logistics",
                    "resume_allowed": True,
                    "responsibilities": [
                        "Installed home appliances contracted through Lowe's",
                        "Performed hands-on installation work while following customer requirements "
                        "and established procedures.",
                        "Communicated clearly with customers throughout installations.",
                    ],
                },
            ],
            "project_inventory": [],
            "skills_inventory": [],
        }
        ranked = rank_profile_evidence(job, profile, top_n=10)
        self.assertEqual(ranked, [])

    def test_business_entity_suffix_in_employer_name_is_not_a_usable_term(self):
        """2026-09-04, found via a real production false positive: the
        employer's own name "Alex Prosperity Group / UST Logistics"
        contributed the bare word "group" as a matchable term (idf=2.90,
        ABOVE the IDF-generic threshold, so the IDF check doesn't catch
        this -- it's a business-entity-name suffix, not a generic English
        word), which then matched an unrelated "Senior Manager, Machine
        Learning Ops Engineering" job purely via its own use of "group"
        in "a high-performing MLOps engineering group"."""
        from applypilot.scoring.local_tailor import _item_terms, rank_profile_evidence

        item = {"name": "Alex Prosperity Group / UST Logistics", "responsibilities": ["Installed home appliances."]}
        terms = _item_terms(item)
        self.assertNotIn("group", terms)
        self.assertIn("installed", terms)
        self.assertIn("home", terms)

        job = {
            "title": "Senior Manager, Machine Learning Ops Engineering",
            "full_description": (
                "Lead and grow a high-performing MLOps engineering group tasked with managing "
                "end-to-end data pipelines.\n"
            ),
        }
        profile = {
            "experience_inventory": [
                {
                    "name": "Alex Prosperity Group / UST Logistics",
                    "resume_allowed": True,
                    "responsibilities": ["Installed home appliances."],
                },
            ],
            "project_inventory": [],
            "skills_inventory": [],
        }
        self.assertEqual(rank_profile_evidence(job, profile, top_n=10), [])

    def test_idf_lookup_failure_degrades_to_curated_list_only(self):
        """If idf_weights.json is missing/corrupt, _is_generic_evidence_term
        must still work using ONLY _GENERIC_EVIDENCE_TERMS -- never crash,
        never silently treat everything as generic."""
        from applypilot.scoring import local_tailor

        with patch.object(local_tailor, "_idf_weights", return_value={}):
            self.assertTrue(local_tailor._is_generic_evidence_term("it"))  # still curated-excluded
            self.assertFalse(local_tailor._is_generic_evidence_term("customer"))  # no IDF data -> not excluded
            self.assertFalse(local_tailor._is_generic_evidence_term("python"))

    def test_multiword_technical_phrases_are_unaffected(self):
        """Only the bare single-word term is excluded -- a category that
        merely CONTAINS "technical" as part of a longer, specific phrase
        is untouched."""
        from applypilot.scoring.local_tailor import _item_terms

        item = {
            "name": "Field Support Tech",
            "relevance_categories": [
                "technical support",
                "customer-facing technical",
                "desktop support",
            ],
        }
        terms = _item_terms(item)
        self.assertIn("technical support", terms)
        self.assertIn("customer-facing technical", terms)
        self.assertIn("desktop support", terms)

    def test_job_20171_technical_assistance_line_no_longer_retrieves_python(self):
        """The exact reported failure: a job whose only mention of
        'technical' is generic prose ('technical assistance... as it
        relates to the selection, use, and installation of products') must
        not retrieve a Python skill tagged only with the bare category
        "technical" -- there is no OTHER literal overlap (no mention of
        "python" anywhere), so with "technical" excluded there is nothing
        left for it to match on."""
        from applypilot.scoring.local_tailor import rank_profile_evidence

        job = {
            "title": "Hardware Sales Associate",
            "full_description": (
                "- Build customer confidence by supplying product knowledge "
                "and technical assistance as it relates to the selection, "
                "use, and installation of products sold by CNRG.\n"
            ),
        }
        profile = {
            "skills_inventory": [
                {"name": "Python", "relevance_categories": ["technical"], "resume_allowed": True},
            ],
            "experience_inventory": [],
            "project_inventory": [],
        }
        ranked = rank_profile_evidence(job, profile, top_n=10)
        self.assertEqual(ranked, [])

    def test_standalone_it_pronoun_does_not_create_a_false_match(self):
        """CompTIA A+'s "IT" category must not match ordinary prose that
        happens to use the pronoun "it" -- the exact live-test failure
        ("...as it relates to the selection...")."""
        from applypilot.scoring.local_tailor import rank_profile_evidence

        job = {
            "title": "Retail Associate",
            "full_description": ("Build customer confidence as it relates to the selection of products.\n"),
        }
        profile = {
            "certifications": [
                {"name": "CompTIA A+", "relevance_categories": ["IT"], "resume_allowed": True},
            ],
            "experience_inventory": [],
            "project_inventory": [],
            "skills_inventory": [],
        }
        ranked = rank_profile_evidence(job, profile, top_n=10)
        self.assertEqual(ranked, [])

    def test_specific_technical_requirement_still_matches_python(self):
        """The fix must not blind the system to a GENUINELY technical/
        software requirement -- Python is also tagged with the specific
        category "python", so a real programming requirement still
        matches it. Only the bare generic word is removed, not the item."""
        from applypilot.scoring.local_tailor import _pair_candidate_evidence, rank_profile_evidence

        job = {
            "title": "Software Engineer",
            "full_description": "- 3+ years of Python programming experience required\n",
        }
        profile = {
            "skills_inventory": [
                {
                    "name": "Python",
                    "relevance_categories": ["technical", "python", "automation"],
                    "resume_allowed": True,
                },
            ],
            "experience_inventory": [],
            "project_inventory": [],
        }
        ranked = rank_profile_evidence(job, profile, top_n=10)
        names = [r["name"] for r in ranked]
        self.assertIn("Python", names)
        cands = _pair_candidate_evidence(
            "3+ years of Python programming experience required",
            ranked,
        )
        self.assertEqual({names[i - 1] for i in cands}, {"Python"})

    def test_existing_specific_categories_still_match(self):
        """Existing legitimate categories -- automotive, sales, customer
        service -- are multi-word or specific enough single words and must
        be completely unaffected by the generic-term filter."""
        from applypilot.scoring.local_tailor import rank_profile_evidence

        job = {
            "title": "Hardware Sales Associate",
            "full_description": "We value sales, customer service, and automotive skills.\n",
        }
        profile = {
            "experience_inventory": [
                {
                    "name": "National Tire and Battery / Mavis",
                    "relevance_categories": ["automotive", "customer service"],
                    "resume_allowed": True,
                },
                {"name": "AMP Smart", "relevance_categories": ["sales"], "resume_allowed": True},
            ],
            "skills_inventory": [],
            "project_inventory": [],
        }
        ranked = rank_profile_evidence(job, profile, top_n=10)
        names = {r["name"] for r in ranked}
        self.assertEqual(names, {"National Tire and Battery / Mavis", "AMP Smart"})


# ---------------------------------------------------------------------------
# 12. Cloud/local plan generation now uses retrieved evidence as grounding
# ---------------------------------------------------------------------------


class TestPlanUsesEvidenceGrounding(unittest.TestCase):
    def _job(self):
        return {
            "title": "Python Automation Engineer",
            "full_description": "- Python automation and OCR experience required\n",
        }

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
                {
                    "name": "Standup-OCR",
                    "relevance_categories": ["python", "ocr", "automation"],
                    "factual_concepts": ["Python", "OCR"],
                    "resume_allowed": True,
                },
                {
                    "name": "Doc-OCR Pipeline",
                    "relevance_categories": ["python", "ocr", "automation"],
                    "factual_concepts": ["Python", "OCR"],
                    "resume_allowed": True,
                },
            ],
            "skills_inventory": [],
            "experience_inventory": [],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp) as mock_post:
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
        ocr_project_id = next(i for i, r in enumerate(ranked, start=1) if r["name"] == "Standup-OCR")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"message": {"content": f'{{"matches":[{{"r":1,"e":[{ocr_project_id}]}}]}}'}}
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp):
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
            {
                "name": "National Tire and Battery / Mavis",
                "relevance_categories": ["automotive", "customer service"],
                "resume_allowed": True,
            },
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
            "Practice excellent customer service at all times",
            ranked,
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
            "Generate leads and close sales with new customers",
            ranked,
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
            "Strong technical and information technology troubleshooting skills",
            ranked,
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

    def test_end_to_end_model_cannot_select_comptia_for_customer_service_requirement(self):
        """Even if the local model ignores its candidate hint and cites
        CompTIA A+ for a customer-service requirement, get_local_tailoring_plan
        must strip that pick -- enforcement happens in code (see
        _merge_model_matches_with_resolved), not by trusting the prompt.

        Uses CompTIA A+ (matched via the specific multi-word category
        "information technology") rather than Python here: Python's only
        category in this fixture is the bare word "technical", which
        _is_generic_evidence_term now excludes outright (see
        _GENERIC_EVIDENCE_TERMS) -- Python is no longer even retrieved for
        this job, so it can't demonstrate the out-of-candidate-tier
        enforcement this test is about."""
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
        comptia_id = names.index("CompTIA A+") + 1

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "message": {"content": (f'{{"matches":[{{"r":1,"e":[{waffle_id},{comptia_id}]}}]}}')}
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp):
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)

        self.assertIsNotNone(plan)
        self.assertEqual(plan["requirements"][0]["resume_evidence"], ["Waffle House"])
        self.assertNotIn("CompTIA A+", plan["matching_certifications"])
        joined = " ".join(plan["_warnings"])
        self.assertIn("not in its deterministic candidate tier", joined)


# ---------------------------------------------------------------------------
# 12a2. Controlled concept-synonym vocabulary closes obvious paraphrase gaps
#
# Follow-up to the pair-scoring pass above: _pair_candidate_evidence only
# recognized a LITERAL shared term, so a genuinely obvious paraphrase like
# "Accept and respond to telephone inquiries from customers..." (job
# 20171's real false negative -- it never says "customer service") scored
# zero candidates and came back unsupported even though a human reads it
# instantly. _CONCEPT_SYNONYM_PATTERNS/_synonym_hit close that gap with a
# small curated table, not general fuzzy matching -- these tests exercise
# both the recall win and the guard against it becoming a new source of
# false positives.
# ---------------------------------------------------------------------------


class TestConceptSynonymRecall(unittest.TestCase):
    def _ranked(self):
        from applypilot.scoring.local_tailor import rank_profile_evidence

        return rank_profile_evidence(_job_domain_mix_job(), _job_domain_mix_profile(), top_n=10)

    def test_telephone_inquiries_paraphrase_matches_customer_service_evidence(self):
        """The exact live-test false negative from job 20171."""
        from applypilot.scoring.local_tailor import _pair_candidate_evidence

        ranked = self._ranked()
        names = [r["name"] for r in ranked]
        cands = _pair_candidate_evidence(
            "Accept and respond to telephone inquiries from customers in a "
            "polite manner that provides the information sought by the customer.",
            ranked,
        )
        cand_names = {names[i - 1] for i in cands}
        self.assertTrue(cand_names & {"National Tire and Battery / Mavis", "Waffle House"})
        self.assertNotIn("Python", cand_names)
        self.assertNotIn("AMP Smart", cand_names)

    def test_customer_feedback_paraphrase_matches_customer_service_evidence(self):
        """Another real line from job 20171: 'based on customer feedback'
        never says 'customer service' either."""
        from applypilot.scoring.local_tailor import _pair_candidate_evidence

        ranked = self._ranked()
        names = [r["name"] for r in ranked]
        cands = _pair_candidate_evidence(
            "Make new product recommendations to store management based on customer feedback.",
            ranked,
        )
        cand_names = {names[i - 1] for i in cands}
        self.assertTrue(cand_names & {"National Tire and Battery / Mavis", "Waffle House"})
        self.assertNotIn("Python", cand_names)

    def test_lead_generation_paraphrase_matches_sales_evidence_without_literal_sales_word(self):
        from applypilot.scoring.local_tailor import _pair_candidate_evidence

        ranked = self._ranked()
        names = [r["name"] for r in ranked]
        cands = _pair_candidate_evidence(
            "Drive membership growth through outreach and lead generation activities.",
            ranked,
        )
        cand_names = {names[i - 1] for i in cands}
        self.assertIn("AMP Smart", cand_names)
        self.assertNotIn("Python", cand_names)
        self.assertNotIn("National Tire and Battery / Mavis", cand_names)

    def test_synonym_hit_requires_exact_table_key(self):
        """_synonym_hit must never guess at a term outside the curated
        table -- otherwise this stops being a bounded vocabulary and
        becomes general fuzzy matching again."""
        from applypilot.scoring.local_tailor import _synonym_hit

        self.assertFalse(_synonym_hit("technical", "telephone inquiries from customers"))
        self.assertFalse(_synonym_hit("some random project name", "customer feedback"))

    def test_customer_service_synonyms_do_not_fire_on_generic_technical_language(self):
        """Guard against scope creep: the curated customer-service
        phrasings must not match generic 'technical assistance'/'product
        knowledge' language that has no customer-facing framing at all --
        that would reopen the exact false-positive failure mode (an
        unrelated category matched via an over-broad pattern) this
        closed-set design exists to prevent. This is also why 'technical'
        has no synonym entry at all -- see the comment above
        _CONCEPT_SYNONYM_PATTERNS."""
        from applypilot.scoring.local_tailor import _synonym_hit

        text = ("provide technical assistance and product knowledge for installation of parts.").lower()
        self.assertFalse(_synonym_hit("customer service", text))
        self.assertFalse(_synonym_hit("sales", text))

    def test_customer_confidence_phrasing_correctly_favors_customer_service(self):
        """The mirror case: a line that DOES explicitly frame the work as
        building customer confidence (even alongside product/technical
        language) is genuinely a customer-service signal, not a false
        positive -- 'customer confidence' is squarely about the customer
        relationship, unlike bare 'technical'/'product' language."""
        from applypilot.scoring.local_tailor import _synonym_hit

        text = (
            "build customer confidence by supplying product knowledge and "
            "technical assistance related to installation of products."
        ).lower()
        self.assertTrue(_synonym_hit("customer service", text))

    def test_end_to_end_paraphrase_resolves_deterministically_without_llm(self):
        """A single-candidate synonym match must auto-resolve exactly like
        a literal one -- no LLM call needed."""
        from applypilot.scoring import local_tailor

        job = {
            "title": "Hardware Sales Associate",
            "full_description": (
                "We provide excellent customer service to every guest.\n"
                "\n"
                "- Accept and respond to telephone inquiries from customers "
                "in a polite manner.\n"
            ),
        }
        profile = {
            "experience_inventory": [
                {"name": "Waffle House", "relevance_categories": ["customer service"], "resume_allowed": True},
            ],
            "skills_inventory": [],
            "project_inventory": [],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post") as mock_post:
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)

        mock_post.assert_not_called()
        self.assertIsNotNone(plan)
        self.assertEqual(plan["requirements"][0]["resume_evidence"], ["Waffle House"])
        self.assertTrue(plan["requirements"][0]["supported"])


class TestInflectionTolerantMatching(unittest.TestCase):
    """2026-09-03: _term_in_text now tolerates ordinary plural/singular
    variants and a small curated root-family table, found via real
    near-miss mining across a 100-job sample (mean 1.5 exact_keywords
    available per requirement -- most misses were the SAME word in a
    different inflection, e.g. "customer"/"customers", not a genuinely
    different synonym). No text is ever rewritten by this mechanism, only
    whether a keyword counts as already matched -- zero fabrication risk."""

    def test_plural_keyword_matches_singular_evidence_term(self):
        from applypilot.scoring.local_tailor import _term_in_text

        self.assertTrue(_term_in_text("customer", "we work with many customers daily"))
        self.assertTrue(_term_in_text("customers", "handled one customer at a time"))

    def test_root_family_variant_matches(self):
        from applypilot.scoring.local_tailor import _term_in_text

        self.assertTrue(_term_in_text("operations", "worked in a high-volume operational environment"))
        self.assertTrue(_term_in_text("operational", "5+ years of operations experience"))

    def test_term_in_text_still_requires_word_boundary(self):
        """Regression guard: pluralization tolerance must not become a
        substring/stem match -- "cat" must never match inside "category"."""
        from applypilot.scoring.local_tailor import _term_in_text

        self.assertFalse(_term_in_text("cat", "this role requires categorizing tickets"))

    def test_inflection_tolerance_does_not_invent_unrelated_matches(self):
        """A term with no plural/root-family rule at all still correctly
        returns False against unrelated text -- proving this stays a
        bounded variant set, not a general stemmer."""
        from applypilot.scoring.local_tailor import _term_in_text

        self.assertFalse(_term_in_text("goose", "we saw several geese at the park"))
        self.assertFalse(_term_in_text("python", "worked extensively with java"))


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
            "skills_inventory": [{"name": "Python", "resume_allowed": True, "relevance_categories": ["python"]}],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post") as mock_post:
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        mock_post.assert_not_called()
        self.assertIsNotNone(plan)
        self.assertEqual(plan["requirements"], [])

    def test_no_matching_evidence_skips_llm_call(self):
        from applypilot.scoring import local_tailor

        job = {"title": "Chef", "full_description": "- Culinary arts degree required\n"}
        profile = {
            "resume_facts": {},
            "skills_inventory": [{"name": "Python", "resume_allowed": True, "relevance_categories": ["python"]}],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post") as mock_post:
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        mock_post.assert_not_called()
        self.assertIsNotNone(plan)
        self.assertEqual(plan["unsupported_requirements"], ["Culinary arts degree required"])

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
                {
                    "name": "Customer Service",
                    "resume_allowed": True,
                    "relevance_categories": ["customer", "service", "retail"],
                },
            ],
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post") as mock_post:
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
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1", "APPLYPILOT_LOCAL_LLM_MODEL": "qwen3:8b"}
        with patch.dict("os.environ", env, clear=False), patch("httpx.post", return_value=mock_resp) as mock_post:
            local_tailor.get_local_tailoring_plan("resume", job, profile)
        sent = mock_post.call_args.kwargs["json"]
        self.assertIs(sent["think"], False)
        self.assertEqual(sent["model"], "qwen3:8b")

    def test_timeout_is_logged_with_specific_cause(self):
        """A timeout must produce a specific, visible (WARNING-level)
        explanation rather than being silently swallowed at debug level --
        'a local planning failure should remain a local planning failure
        and explain why'."""
        import httpx as httpx_mod

        from applypilot.scoring import local_tailor

        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1", "APPLYPILOT_LOCAL_LLM_MODEL": "qwen3:8b"}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("httpx.post", side_effect=httpx_mod.ReadTimeout("timed out")),
            patch.object(local_tailor.log, "warning") as mock_warn,
        ):
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)
        self.assertIsNone(plan)
        mock_warn.assert_called_once()
        logged_msg = mock_warn.call_args[0][0]
        self.assertIn("timed out", logged_msg.lower())

    def test_connect_error_is_logged_and_falls_back_gracefully(self):
        """A connection failure to the configured local endpoint must be
        logged with the unreachable URL, and (per the existing 'local
        planner failure must not block cloud tailoring' rule) the call must
        still return None cleanly rather than raise."""
        import httpx as httpx_mod

        from applypilot.scoring import local_tailor

        job, profile = _matchable_job_and_profile()
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://127.0.0.1:11434", "APPLYPILOT_LOCAL_LLM_MODEL": "qwen3:32b"}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("httpx.post", side_effect=httpx_mod.ConnectError("Connection refused")),
            patch.object(local_tailor.log, "warning") as mock_warn,
        ):
            plan = local_tailor.get_local_tailoring_plan("resume", job, profile)

        self.assertIsNone(plan)
        mock_warn.assert_called_once()
        logged_msg = mock_warn.call_args[0][0] % mock_warn.call_args[0][1:]
        self.assertIn("could not reach", logged_msg.lower())
        self.assertIn("127.0.0.1:11434", logged_msg)


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
        job = {
            "url": "https://example.com/j1",
            "title": "Python Automation Engineer",
            "full_description": "- Python experience required\n",
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("httpx.post", return_value=mock_resp),
            patch("applypilot.scoring.resume_router.load_resume_text_for_job", return_value=("Resume text.", None)),
        ):
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
        job = {
            "url": "https://example.com/j2",
            "title": "Python Automation Engineer",
            "full_description": "- Python experience required\n",
        }
        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("httpx.post", side_effect=ConnectionError("refused")),
            patch("applypilot.scoring.resume_router.load_resume_text_for_job", return_value=("Resume text.", None)),
        ):
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
            k: v
            for k, v in __import__("os").environ.items()
            if k
            not in (
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "DEEPSEEK_API_KEY",
                "LLM_URL",
                "LLM_MODEL",
                "APPLYPILOT_LOCAL_LLM_URL",
                "APPLYPILOT_LOCAL_LLM_MODEL",
            )
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
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("qwen3:8b", quality=False)
        self.assertFalse(any(e.name == "qwen3:8b" and e.provider == "gemini" for e in chain))

    def test_local_entry_still_present_as_final_fallback(self):
        """The local model must still appear -- exactly once, under the
        correct provider, and last in the chain (final fallback) -- when
        the caller opts in via include_local=True (the quality tier's
        default). The fast/scoring tier's implicit default is now False;
        see TestLocalExcludedFromFastTier."""
        from applypilot.llm import _build_fallback_chain

        env = self._env(
            GEMINI_API_KEY="fake-gemini-key",
            APPLYPILOT_LOCAL_LLM_URL="http://localhost:11434/v1",
            APPLYPILOT_LOCAL_LLM_MODEL="qwen3:8b",
        )
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("qwen3:8b", quality=False, include_local=True)
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
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-3.6-flash", quality=False)
        self.assertTrue(all(e.provider == "gemini" for e in chain))
        self.assertEqual(chain[0].name, "gemini-3.6-flash")

    def test_genuinely_unlisted_gemini_model_still_added_as_gemini(self):
        """A real (not-yet-hardcoded) Gemini model name that is NOT the
        configured local model must still be treated as a Gemini primary --
        this pre-existing behavior must survive the fix."""
        from applypilot.llm import _build_fallback_chain

        env = self._env(GEMINI_API_KEY="fake-gemini-key")
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-4.0-custom-preview", quality=False)
        self.assertEqual(chain[0].name, "gemini-4.0-custom-preview")
        self.assertEqual(chain[0].provider, "gemini")

    def test_test_local_cmd_probe_only_hits_local_provider(self):
        """cli.py's `test-local` command must isolate its chat probe to the
        local endpoint -- it must never silently succeed via a configured
        cloud fallback (which would mask a broken local setup as 'working')."""
        import applypilot.cli as cli_mod

        env = {
            "GEMINI_API_KEY": "fake-gemini-key",
            "APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1",
            "APPLYPILOT_LOCAL_LLM_MODEL": "qwen3:8b",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("applypilot.llm.local_available", return_value=True),
            patch("applypilot.llm.LLMClient._try_openai_compat", return_value='{"status":"ok"}') as mock_try,
        ):
            cli_mod.test_local_cmd()
        self.assertTrue(mock_try.called)
        called_entry = mock_try.call_args[0][0]
        self.assertEqual(called_entry.provider, "local")
        self.assertEqual(called_entry.name, "qwen3:8b")


# ---------------------------------------------------------------------------
# 14b. test-local probe vs. Qwen3's mandatory reasoning trace (2026-08-31)
#
# Root cause, reproduced live against a real Ollama 0.33.2 instance running
# qwen3:1.7b: `applypilot test-local` reported "Null/empty content" even
# though the exact same model answered a direct HTTP request fine. Ollama's
# OpenAI-compatible /v1/chat/completions response for qwen3 models ALWAYS
# carries a separate "reasoning" field ahead of "content" -- confirmed
# identical (empty content, finish_reason="length") with and without
# chat()'s existing "/no_think" prompt prefix, and confirmed that neither a
# "think": false nor "reasoning_effort" field in the OpenAI-compatible
# request body has any effect on this Ollama version's /v1 route (both
# silently ignored). The command's old max_tokens=30 gave the reasoning
# trace nowhere to go before hitting the limit. This is NOT a request-
# construction or response-parsing bug -- _try_openai_compat's extraction
# (data["choices"][0]["message"]["content"]) is correct and must keep
# treating a genuinely empty result as a failure. The fix only raises the
# probe's token budget (30 -> 300, confirmed live to be enough headroom).
# ---------------------------------------------------------------------------


class TestLocalProbeReasoningTokenBudget(unittest.TestCase):
    def _real_ollama_response(self, content: str, reasoning: str, finish_reason: str, completion_tokens: int):
        """Builds a response shaped exactly like the real, live Ollama
        0.33.2 payload captured for qwen3:1.7b via /v1/chat/completions --
        both a top-level "reasoning" field AND the standard "content"
        field are present on every response, success or not."""
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "model": "qwen3:1.7b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "reasoning": reasoning},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"completion_tokens": completion_tokens},
        }
        return resp

    def test_reasoning_only_response_at_small_budget_still_raises(self):
        """Locks down that an empty `content` alongside a populated
        `reasoning` field (the real shape observed at max_tokens=30) is
        NOT treated as success merely because a "reasoning" field is
        present -- the fix must not weaken this validation."""
        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", "")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient("http://localhost:11434/v1", "qwen3:1.7b", "", quality=False)
        client._fallback_chain = fake_chain

        resp = self._real_ollama_response(
            content="",
            reasoning='Okay, the user wants me to reply with exactly {"status":"ok"}.\n\nFirst, I need to',
            finish_reason="length",
            completion_tokens=30,
        )
        with patch.object(client._client, "post", return_value=resp):
            with self.assertRaises(RuntimeError) as ctx:
                client.chat([{"role": "user", "content": 'Reply with exactly: {"status":"ok"}'}], max_tokens=30)
        self.assertIn("Null/empty content", str(ctx.exception))

    def test_real_content_response_at_current_budget_succeeds(self):
        """The real success shape captured live at max_tokens=300: the
        reasoning trace still precedes the answer, but this time finishes
        (finish_reason="stop") with room left for real content."""
        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", "")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient("http://localhost:11434/v1", "qwen3:1.7b", "", quality=False)
        client._fallback_chain = fake_chain

        resp = self._real_ollama_response(
            content='{"status":"ok"}',
            reasoning="Okay, the user wants me to reply with exactly ... Let me confirm that's all.",
            finish_reason="stop",
            completion_tokens=135,
        )
        with patch.object(client._client, "post", return_value=resp):
            result = client.chat([{"role": "user", "content": 'Reply with exactly: {"status":"ok"}'}], max_tokens=300)
        self.assertEqual(result, '{"status":"ok"}')

    def test_test_local_cmd_succeeds_end_to_end_with_realistic_reasoning_response(self):
        """End-to-end through the actual CLI command (not a mocked
        _try_openai_compat): a fake HTTP responder mimics the real Ollama
        behavior -- reasoning consumes the whole budget below ~135
        completion tokens, succeeds above it -- so this only passes if
        test_local_cmd is actually requesting a large-enough max_tokens."""
        import applypilot.cli as cli_mod

        def fake_post(url, json=None, headers=None, **kwargs):  # noqa: A002 - matches httpx.Client.post's own param name
            requested = json["max_tokens"]
            reasoning = 'Okay, the user wants me to reply with exactly {"status":"ok"}. Let me confirm.'
            if requested < 135:
                return self._real_ollama_response("", reasoning, "length", requested)
            return self._real_ollama_response('{"status":"ok"}', reasoning, "stop", 135)

        env = {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1", "APPLYPILOT_LOCAL_LLM_MODEL": "qwen3:1.7b"}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("applypilot.llm.local_available", return_value=True),
            patch("httpx.Client.post", side_effect=fake_post),
        ):
            # Must not raise/exit -- a typer.Exit would fail this test.
            cli_mod.test_local_cmd()


# ---------------------------------------------------------------------------
# 14c. Semantic candidate-recall expansion (2026-08-31)
#
# Real-data validation (5 real jobs / 40 requirements, then
# arbitration_test.py) found: raw cosine similarity does not reliably
# separate genuine evidence from false positives, and neither does the
# existing Qwen3 arbitration prompt. So the one invariant these tests
# exist to lock down is architectural, not score-based: a semantic
# candidate may WIDEN a requirement's candidate pool, but must NEVER be
# able to auto-resolve a requirement the way a single literal candidate
# can.
# ---------------------------------------------------------------------------


class TestEvidenceSemanticText(unittest.TestCase):
    def test_experience_uses_role_title_and_role_type_when_description_empty(self):
        from applypilot.scoring.local_tailor import _evidence_semantic_text

        item = {
            "name": "Alex Prosperity Group / UST Logistics",
            "role_title": "Moving Specialist / Installation Tech",
            "role_type": "installation",
            "description": None,
            "relevance_categories": ["customer-facing", "hands-on work", "installation"],
        }
        text = _evidence_semantic_text("experience", item)
        self.assertIn("Moving Specialist / Installation Tech", text)
        self.assertIn("installation", text)
        self.assertIn("Alex Prosperity Group / UST Logistics", text)

    def test_experience_includes_description_when_present(self):
        from applypilot.scoring.local_tailor import _evidence_semantic_text

        item = {"name": "X", "description": "Real description text.", "relevance_categories": []}
        text = _evidence_semantic_text("experience", item)
        self.assertIn("Real description text.", text)

    def test_project_uses_factual_concepts(self):
        from applypilot.scoring.local_tailor import _evidence_semantic_text

        item = {"name": "Standup-OCR", "factual_concepts": ["Python", "OCR"], "relevance_categories": ["automation"]}
        text = _evidence_semantic_text("project", item)
        self.assertIn("Python", text)
        self.assertIn("OCR", text)

    def test_certification_uses_official_description(self):
        from applypilot.scoring.local_tailor import _evidence_semantic_text

        item = {
            "name": "CompTIA A+",
            "official_credential_description": "CompTIA A+ Core 1 and Core 2 Certification Exams",
            "relevance_categories": ["technical support", "IT"],
        }
        text = _evidence_semantic_text("certification", item)
        self.assertIn("CompTIA A+ Core 1 and Core 2 Certification Exams", text)

    def test_empty_item_returns_empty_string_not_raise(self):
        from applypilot.scoring.local_tailor import _evidence_semantic_text

        self.assertEqual(_evidence_semantic_text("experience", {}), "")


class TestBuildFullEvidenceCorpus(unittest.TestCase):
    def test_resume_allowed_false_excluded(self):
        from applypilot.scoring.local_tailor import _build_full_evidence_corpus

        corpus = _build_full_evidence_corpus(_sample_profile())
        names = [c["name"] for c in corpus]
        self.assertNotIn("Private Consulting Gig", names)

    def test_includes_items_literal_matching_would_never_score(self):
        """The whole point: no literal-overlap filter here -- everything
        resume_allowed=True is included regardless of any job description."""
        from applypilot.scoring.local_tailor import _build_full_evidence_corpus

        corpus = _build_full_evidence_corpus(_sample_profile())
        names = [c["name"] for c in corpus]
        self.assertIn("Unrelated Woodworking Project", names)  # matches no job at all, still in the full corpus

    def test_each_entry_has_nonempty_text(self):
        from applypilot.scoring.local_tailor import _build_full_evidence_corpus

        corpus = _build_full_evidence_corpus(_sample_profile())
        self.assertTrue(corpus)
        for c in corpus:
            self.assertTrue(c["text"].strip())


class TestSemanticExpandEvidence(unittest.TestCase):
    """_semantic_expand_evidence: the safe-degradation + candidate-expansion
    wrapper called from get_local_tailoring_plan."""

    def test_disabled_returns_unchanged(self):
        from applypilot.scoring.local_tailor import _semantic_expand_evidence

        lines = [{"text": "Install PC hardware.", "importance": "unspecified"}]
        ranked = []
        with patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH": "0"}):
            extended, semantic_candidates = _semantic_expand_evidence(lines, ranked, _sample_profile())
        self.assertIs(extended, ranked)
        self.assertEqual(semantic_candidates, {})

    def test_ollama_failure_degrades_to_unchanged_not_raise(self):
        from applypilot.scoring.local_tailor import _semantic_expand_evidence

        lines = [{"text": "Install PC hardware.", "importance": "unspecified"}]
        ranked = []
        with (
            patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH": "1"}),
            patch("httpx.post", side_effect=ConnectionError("refused")),
        ):
            extended, semantic_candidates = _semantic_expand_evidence(lines, ranked, _sample_profile())
        self.assertEqual(extended, ranked)
        self.assertEqual(semantic_candidates, {})

    def test_evidence_corpus_embedded_once_not_per_requirement(self):
        """Embeddings must be reused across all requirement lines in one
        call -- exactly two embed calls total (corpus once, requirements
        batched once), regardless of how many requirement lines exist."""
        from applypilot.scoring.local_tailor import _semantic_expand_evidence

        lines = [
            {"text": "Install PC hardware.", "importance": "unspecified"},
            {"text": "Troubleshoot connectivity issues.", "importance": "unspecified"},
            {"text": "Document incidents.", "importance": "unspecified"},
        ]
        profile = _sample_profile()

        def fake_embed(texts, timeout=30.0):
            return [[1.0, 0.0] for _ in texts]

        with (
            patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH": "1"}),
            patch("applypilot.scoring.semantic_match.embed_texts", side_effect=fake_embed) as mock_embed,
        ):
            _semantic_expand_evidence(lines, [], profile)

        self.assertEqual(mock_embed.call_count, 2)  # once for corpus, once for all 3 requirements batched

    def test_new_items_appended_with_semantic_provenance(self):
        from applypilot.scoring.local_tailor import _semantic_expand_evidence

        lines = [{"text": "Install PC hardware.", "importance": "unspecified"}]
        profile = _sample_profile()

        def fake_embed(texts, timeout=30.0):
            return [[1.0, 0.0] for _ in texts]

        with (
            patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH": "1"}),
            patch("applypilot.scoring.semantic_match.embed_texts", side_effect=fake_embed),
        ):
            extended, semantic_candidates = _semantic_expand_evidence(lines, [], profile)

        self.assertGreater(len(extended), 0)
        new_items = [e for e in extended if e.get("provenance") == "semantic"]
        self.assertTrue(new_items)
        for e in new_items:
            self.assertEqual(e["matched_terms"], [])
        self.assertIn(1, semantic_candidates)

    def test_items_already_in_literal_ranked_evidence_not_duplicated(self):
        from applypilot.scoring.local_tailor import _semantic_expand_evidence

        lines = [{"text": "Automate things with Python.", "importance": "unspecified"}]
        profile = _sample_profile()
        already_literal = [{"type": "skill", "name": "Python", "score": 1, "matched_terms": ["python"], "item": {}}]

        def fake_embed(texts, timeout=30.0):
            return [[1.0, 0.0] for _ in texts]

        with (
            patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH": "1"}),
            patch("applypilot.scoring.semantic_match.embed_texts", side_effect=fake_embed),
        ):
            extended, _ = _semantic_expand_evidence(lines, already_literal, profile)

        names = [(e["type"], e["name"]) for e in extended]
        self.assertEqual(names.count(("skill", "Python")), 1)  # not duplicated


class TestAutoResolveRequirementsSemanticProvenance(unittest.TestCase):
    """The central safety invariant: a semantic-only candidate must never
    auto-resolve a requirement, even a single unambiguous one."""

    def _ranked(self):
        return [
            {"type": "certification", "name": "CompTIA A+", "score": 0, "matched_terms": [], "item": {}},
            {"type": "project", "name": "You Power You", "score": 0, "matched_terms": [], "item": {}},
        ]

    def test_literal_only_behavior_unchanged_when_semantic_param_omitted(self):
        from applypilot.scoring.local_tailor import _auto_resolve_requirements

        lines = [{"text": "Nothing matches this at all.", "importance": "unspecified"}]
        resolved, candidates = _auto_resolve_requirements(lines, self._ranked())
        self.assertEqual(resolved, {1: []})
        self.assertEqual(candidates, {1: []})

    def test_semantic_only_candidate_does_not_auto_resolve(self):
        """Literal finds nothing; semantic proposes evidence id 1. Must
        NOT be auto-resolved -- must remain in the ambiguous/arbitration
        path, even though there's exactly one semantic candidate."""
        from applypilot.scoring.local_tailor import _auto_resolve_requirements

        lines = [{"text": "Nothing literal matches this at all.", "importance": "unspecified"}]
        resolved, candidates = _auto_resolve_requirements(lines, self._ranked(), {1: [1]})
        self.assertNotIn(1, resolved)
        self.assertEqual(candidates[1], [1])

    def test_single_semantic_candidate_still_not_auto_resolved(self):
        """Explicit single-candidate case named in the requirements: do
        not infer support from semantic candidate count alone."""
        from applypilot.scoring.local_tailor import _auto_resolve_requirements

        lines = [{"text": "Some unrelated requirement text.", "importance": "unspecified"}]
        resolved, candidates = _auto_resolve_requirements(lines, self._ranked(), {1: [2]})
        self.assertNotIn(1, resolved)
        self.assertEqual(candidates[1], [2])

    def test_existing_literal_resolution_untouched_by_semantic_addition(self):
        """Literal already resolved this requirement (1 candidate) -- a
        semantic addition for the SAME requirement must not perturb that
        resolution."""
        from applypilot.scoring.local_tailor import _auto_resolve_requirements

        ranked = [
            {"type": "skill", "name": "Python", "score": 1, "matched_terms": ["python"], "item": {}},
            {"type": "project", "name": "You Power You", "score": 0, "matched_terms": [], "item": {}},
        ]
        lines = [{"text": "Use python for automation.", "importance": "unspecified"}]
        resolved, candidates = _auto_resolve_requirements(lines, ranked, {1: [2]})
        self.assertEqual(resolved[1], [1])  # untouched -- literal's own answer, unchanged
        self.assertEqual(candidates[1], [1, 2])  # semantic addition still visible for transparency, just unused

    def test_semantic_extra_already_in_literal_set_not_duplicated(self):
        from applypilot.scoring.local_tailor import _auto_resolve_requirements

        ranked = [
            {"type": "skill", "name": "Python", "score": 1, "matched_terms": ["python"], "item": {}},
            {"type": "skill", "name": "SQL", "score": 1, "matched_terms": ["sql"], "item": {}},
        ]
        lines = [{"text": "Use python and sql.", "importance": "unspecified"}]
        # Both are already literal candidates (a genuine tie) -- semantic
        # "rediscovering" one of them must not duplicate it in candidates.
        resolved, candidates = _auto_resolve_requirements(lines, ranked, {1: [1, 2]})
        self.assertEqual(candidates[1], [1, 2])
        self.assertNotIn(1, resolved)  # still a genuine literal tie -- unchanged, ambiguous


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
            "requirements": [],
            "skills_to_emphasize": [],
            "bullets_to_prioritize": [],
            "bullets_to_deemphasize": [],
            "matching_projects": [],
            "matching_certifications": [],
            "summary_focus": [],
            "keyword_targets": [],
            "safe_rewrites": [],
            "unsupported_requirements": [],
            "_warnings": [],
        },
        "evidence": [],
        "requirement_lines": [],
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
