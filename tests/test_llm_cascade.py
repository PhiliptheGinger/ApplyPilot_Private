"""Tests for the LLM client cascade and exhaustion state machine.

Covers:
- Exhausted models are skipped within the cooldown window
- Exhaustion clears after the cooldown period
- All models exhausted: client clears state and retries
- Failed _try_entry marks model exhausted and falls through to next
- _build_fallback_chain returns quality vs fast model sets correctly
- _build_fallback_chain raises RuntimeError when no keys are configured
"""

import socket
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _local_ollama_reachable() -> bool:
    """Cheap TCP probe -- used only to skip the live Ollama integration
    test when nothing is listening (e.g. CI), not to gate any real logic."""
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.5):
            return True
    except OSError:
        return False


def _make_client(n_models: int = 2):
    """Instantiate an LLMClient with fake models injected directly into the chain."""
    from applypilot.llm import LLMClient, ModelEntry

    # Patch _build_fallback_chain to return fake models so no real API keys needed
    fake_chain = [
        ModelEntry(f"fake-model-{i}", "openai_compat", "https://fake.api/v1", "fake-key") for i in range(n_models)
    ]
    with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
        client = LLMClient(
            base_url="https://fake.api/v1",
            model="fake-model-0",
            api_key="fake-key",
        )
    return client


class TestModelExhaustion(unittest.TestCase):
    """Exhausted models are skipped; exhaustion clears after cooldown."""

    def test_exhausted_model_skipped(self):
        """A model marked exhausted is not tried again within the cooldown window."""
        client = _make_client(2)
        first_name = client._fallback_chain[0].name
        second_name = client._fallback_chain[1].name

        client._exhausted[first_name] = time.time()

        tried = []

        def fake_try(entry, messages, temperature, max_tokens, is_last, frequency_penalty=None, presence_penalty=None):
            tried.append(entry.name)
            return "ok"

        with patch.object(client, "_try_entry", side_effect=fake_try):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertNotIn(first_name, tried, "Exhausted model should not have been tried")
        self.assertIn(second_name, tried, "Non-exhausted model should have been tried")

    def test_exhausted_model_retried_after_cooldown(self):
        """A model marked exhausted 6+ minutes ago is eligible again (cooldown = 5 min)."""
        client = _make_client(2)
        first_name = client._fallback_chain[0].name

        # Mark exhausted 6 minutes ago
        client._exhausted[first_name] = time.time() - 361

        tried = []

        def fake_try(entry, messages, temperature, max_tokens, is_last, frequency_penalty=None, presence_penalty=None):
            tried.append(entry.name)
            return "ok"

        with patch.object(client, "_try_entry", side_effect=fake_try):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertIn(first_name, tried, "Model should be retried after cooldown expires")

    def test_exhaustion_clears_when_all_exhausted(self):
        """When every model is exhausted, the client clears state and retries all."""
        client = _make_client(2)
        now = time.time()
        for m in client._fallback_chain:
            client._exhausted[m.name] = now

        call_count = {"n": 0}

        def fake_try(entry, messages, temperature, max_tokens, is_last, frequency_penalty=None, presence_penalty=None):
            call_count["n"] += 1
            return "ok"

        with patch.object(client, "_try_entry", side_effect=fake_try):
            result = client.chat([{"role": "user", "content": "hello"}])

        self.assertGreater(call_count["n"], 0, "At least one model should have been tried")
        self.assertEqual(result, "ok")

    def test_failed_try_falls_through_to_next(self):
        """Returning None from _try_entry causes the cascade to try the next model."""
        client = _make_client(2)
        first_name = client._fallback_chain[0].name
        second_name = client._fallback_chain[1].name
        call_order = []

        def fake_try(entry, messages, temperature, max_tokens, is_last, frequency_penalty=None, presence_penalty=None):
            call_order.append(entry.name)
            if entry.name == first_name:
                return None  # simulate failure
            return "success from fallback"

        with patch.object(client, "_try_entry", side_effect=fake_try):
            result = client.chat([{"role": "user", "content": "test"}])

        self.assertIn(first_name, call_order)
        self.assertIn(second_name, call_order)
        self.assertEqual(result, "success from fallback")


class TestConnectionErrorFallsThroughChain(unittest.TestCase):
    """2026-08-23: a configured local entry (APPLYPILOT_LOCAL_LLM_URL) can
    genuinely have nothing listening -- httpx.ConnectError was previously
    uncaught in _try_openai_compat, so it aborted client.chat() entirely
    instead of falling through to the next provider, even when a perfectly
    good cloud fallback was next in the chain. These exercise the real
    _try_openai_compat retry/fallthrough logic (not a mocked _try_entry),
    with time.sleep patched out so the retry backoff doesn't slow the test
    down."""

    def _client_with_local_first(self):
        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [
            ModelEntry("qwen3:32b", "local", "http://127.0.0.1:11434/v1", ""),
            ModelEntry("cloud-fallback", "openai_compat", "https://fake.api/v1", "fake-key"),
        ]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="http://127.0.0.1:11434/v1", model="qwen3:32b", api_key="")
        return client

    def _success_resp(self, text="cloud reply"):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": text}}]}
        return resp

    def test_connect_error_falls_through_to_next_provider(self):
        import httpx

        client = self._client_with_local_first()

        def fake_post(url, **kwargs):
            if "127.0.0.1:11434" in url:
                raise httpx.ConnectError("Connection refused")
            return self._success_resp()

        with patch.object(client._client, "post", side_effect=fake_post), patch("applypilot.llm.time.sleep"):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "cloud reply")

    def test_connect_error_on_the_only_entry_raises_not_swallowed(self):
        """A ConnectError on the LAST (or only) entry must still surface --
        mirrors the existing httpx.TimeoutException behavior on is_last."""
        import httpx

        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [ModelEntry("qwen3:32b", "local", "http://127.0.0.1:11434/v1", "")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="http://127.0.0.1:11434/v1", model="qwen3:32b", api_key="")

        with (
            patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")),
            patch("applypilot.llm.time.sleep"),
            self.assertRaises(httpx.ConnectError),
        ):
            client.chat([{"role": "user", "content": "hi"}])

    def test_connect_error_log_message_hints_that_local_endpoint_is_down(self):
        import httpx

        client = self._client_with_local_first()

        def fake_post(url, **kwargs):
            if "127.0.0.1:11434" in url:
                raise httpx.ConnectError("Connection refused")
            return self._success_resp()

        with (
            patch.object(client._client, "post", side_effect=fake_post),
            patch("applypilot.llm.time.sleep"),
            patch("applypilot.llm.log") as mock_log,
        ):
            client.chat([{"role": "user", "content": "hi"}])

        warning_messages = [str(call) for call in mock_log.warning.call_args_list]
        self.assertTrue(
            any("local model endpoint running" in m.lower() for m in warning_messages),
            f"expected a local-endpoint-down hint in a warning log; got: {warning_messages}",
        )


class TestEmptyOrWhitespaceContentRejected(unittest.TestCase):
    """2026-08-25 fix: a real local/qwen3:1.7b call was observed returning
    HTTP 200 with content == "" (see ~/.applypilot/logs/local_qwen_output.log,
    a "0 chars" successful-looking response entry). _try_openai_compat only
    checked `text is None`, so an empty (or whitespace-only) string was
    treated as a genuine successful response and returned all the way up to
    score_job(), where _parse_score_response("") produced a fake fit_score=0
    "success" instead of a retryable failure. Mirrors
    TestConnectionErrorFallsThroughChain's exact HTTP-mocking approach
    (exercises the real _try_openai_compat retry/fallthrough logic, not a
    mocked _try_entry)."""

    def _client_with_local_first(self):
        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [
            ModelEntry("qwen3:1.7b", "local", "http://127.0.0.1:11434/v1", ""),
            ModelEntry("cloud-fallback", "openai_compat", "https://fake.api/v1", "fake-key"),
        ]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="http://127.0.0.1:11434/v1", model="qwen3:1.7b", api_key="")
        return client

    def _resp(self, content):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": content, "refusal": None}}]}
        return resp

    def test_empty_content_falls_through_to_next_provider(self):
        client = self._client_with_local_first()

        def fake_post(url, **kwargs):
            if "127.0.0.1:11434" in url:
                return self._resp("")
            return self._resp("cloud reply")

        with patch.object(client._client, "post", side_effect=fake_post), patch("applypilot.llm.time.sleep"):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "cloud reply")

    def test_whitespace_only_content_falls_through_to_next_provider(self):
        client = self._client_with_local_first()

        def fake_post(url, **kwargs):
            if "127.0.0.1:11434" in url:
                return self._resp("   \n\t  ")
            return self._resp("cloud reply")

        with patch.object(client._client, "post", side_effect=fake_post), patch("applypilot.llm.time.sleep"):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "cloud reply")

    def test_empty_content_on_the_only_entry_raises_not_returned_as_success(self):
        """A blank response on the LAST (or only) entry must still surface
        as a failure -- mirrors the existing null-content/timeout is_last
        behavior. The critical assertion: chat() must NOT return "" as if
        it were a successful result."""
        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [ModelEntry("qwen3:1.7b", "local", "http://127.0.0.1:11434/v1", "")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="http://127.0.0.1:11434/v1", model="qwen3:1.7b", api_key="")

        with (
            patch.object(client._client, "post", return_value=self._resp("")),
            patch("applypilot.llm.time.sleep"),
            self.assertRaises(RuntimeError),
        ):
            client.chat([{"role": "user", "content": "hi"}])

    def test_nonempty_content_is_still_returned_normally(self):
        """Regression sanity: a real, non-blank response is unaffected."""
        client = self._client_with_local_first()

        with (
            patch.object(client._client, "post", return_value=self._resp("a real answer")),
            patch("applypilot.llm.time.sleep"),
        ):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "a real answer")


class TestLocalOpenAIBaseURLNormalization(unittest.TestCase):
    """2026-08-22 incident: a real Ollama endpoint configured via
    APPLYPILOT_LOCAL_LLM_URL=http://127.0.0.1:11434 (no /v1) responded 200
    to a direct httpx POST at /api/chat, but `applypilot test-local`
    reported "model not found (404)" for the exact same model.

    Root cause: APPLYPILOT_LOCAL_LLM_URL is consumed by two independent
    code paths that want different URL shapes --
    local_tailor.get_local_tailoring_plan() posts straight to Ollama's
    NATIVE {url}/api/chat (needs the bare root), while this module's
    LLMClient is OpenAI-compatible-only: _try_openai_compat always posts to
    {entry.base_url}/chat/completions, exactly like it does for Gemini/
    OpenAI whose base_urls already include their versioned API root.
    Ollama only serves an OpenAI-compatible surface under /v1, so a bare
    root resolves to POST http://host:port/chat/completions -- a route
    Ollama doesn't register at all, hence the (misleadingly-labeled)
    "model not found (404)".

    local_openai_base_url() normalizes for the OpenAI-compatible side only
    (appending /v1 if not already present); local_tailor.py's own
    /api/chat construction is untouched and still wants the bare root.
    """

    def test_local_openai_base_url_appends_v1_when_missing(self):
        from applypilot.llm import local_openai_base_url

        self.assertEqual(
            local_openai_base_url("http://127.0.0.1:11434"),
            "http://127.0.0.1:11434/v1",
        )

    def test_local_openai_base_url_strips_trailing_slash_first(self):
        from applypilot.llm import local_openai_base_url

        self.assertEqual(
            local_openai_base_url("http://127.0.0.1:11434/"),
            "http://127.0.0.1:11434/v1",
        )

    def test_local_openai_base_url_idempotent_when_already_present(self):
        """A URL already following the documented /v1 convention (as
        every pre-existing test/example in this codebase uses) must be
        left completely unchanged -- not doubled to .../v1/v1."""
        from applypilot.llm import local_openai_base_url

        self.assertEqual(
            local_openai_base_url("http://localhost:11434/v1"),
            "http://localhost:11434/v1",
        )
        self.assertEqual(
            local_openai_base_url("http://localhost:11434/v1/"),
            "http://localhost:11434/v1",
        )

    def test_fallback_chain_local_entry_gets_normalized_base_url(self):
        """The exact regression: APPLYPILOT_LOCAL_LLM_URL configured as the
        bare Ollama root must still produce a local ModelEntry whose
        base_url is usable by _try_openai_compat's {base_url}/chat/completions
        construction.

        include_local=True is passed explicitly: this test is about URL
        normalization, not about the fast/quality routing split (see
        TestLocalExcludedFromFastTier) -- quality=False chains no longer
        include local by default (2026-08-23), so without the explicit
        override there'd be no local entry at all to normalize."""
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
        env["APPLYPILOT_LOCAL_LLM_URL"] = "http://127.0.0.1:11434"  # bare root, no /v1
        env["APPLYPILOT_LOCAL_LLM_MODEL"] = "qwen3:1.7b"
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("qwen3:1.7b", quality=False, include_local=True)

        local_entries = [e for e in chain if e.provider == "local"]
        self.assertEqual(len(local_entries), 1)
        self.assertEqual(local_entries[0].base_url, "http://127.0.0.1:11434/v1")

    def test_chat_posts_to_v1_chat_completions_not_bare_chat_completions(self):
        """End-to-end at the request-construction level: with a bare-root
        local URL, the actual outgoing POST must hit /v1/chat/completions
        -- the route Ollama actually serves -- not bare /chat/completions,
        which 404s (reproduced live against a real Ollama instance)."""
        from applypilot.llm import LLMClient, ModelEntry, local_openai_base_url

        raw_url = "http://127.0.0.1:11434"
        fake_chain = [ModelEntry("qwen3:1.7b", "local", local_openai_base_url(raw_url), "")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url=raw_url, model="qwen3:1.7b", api_key="")

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "OK"}}]}

        with patch.object(client._client, "post", return_value=resp) as mock_post:
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "OK")
        called_url = mock_post.call_args[0][0]
        self.assertEqual(called_url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertNotEqual(called_url, "http://127.0.0.1:11434/chat/completions")


@pytest.mark.skipif(
    not _local_ollama_reachable(),
    reason="No local Ollama instance reachable at 127.0.0.1:11434",
)
class TestRealOllamaIntegration(unittest.TestCase):
    """Live regression test against a real, running Ollama instance with
    qwen3:1.7b -- reproduces the exact reported scenario end-to-end rather
    than mocking httpx. Skipped automatically when no local Ollama is
    reachable (e.g. in CI)."""

    def test_test_local_style_chat_succeeds_against_real_ollama(self):
        from applypilot.llm import LLMClient, ModelEntry, local_openai_base_url

        raw_url = "http://127.0.0.1:11434"  # bare root, exactly as reported
        model = "qwen3:1.7b"
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
        env["APPLYPILOT_LOCAL_LLM_URL"] = raw_url
        env["APPLYPILOT_LOCAL_LLM_MODEL"] = model
        pinned_chain = [ModelEntry(model, "local", local_openai_base_url(raw_url), "")]
        with (
            patch.dict("os.environ", env, clear=True),
            patch("applypilot.llm._build_fallback_chain", return_value=pinned_chain),
        ):
            # 2026-08-23: quality=False (fast tier) no longer includes local
            # by default -- _build_fallback_chain is patched here purely so
            # construction doesn't raise on "no provider configured" with no
            # cloud keys in this env; the chain is pinned to local-only
            # either way, mirroring cli.py's test-local command.
            client = LLMClient(raw_url, model, "", quality=False)
            client._fallback_chain = pinned_chain

            # max_tokens generously high: qwen3's thinking tokens (visible
            # via Ollama's OpenAI-compat "reasoning" field, separate from
            # "content") consume budget before any content is emitted, and
            # this module's "/no_think" prompt-prefix convention (chat()'s
            # "Qwen3 optimization") does NOT actually suppress that on this
            # endpoint -- confirmed live: with max_tokens=200 the reasoning
            # alone exhausts the budget (finish_reason="length", content
            # ""), deterministically at temperature=0. That's a separate,
            # pre-existing behavior (local_tailor.py's own Ollama-native
            # call sidesteps it with the real "think": false API param
            # instead of a prompt prefix) -- not the 404/routing bug this
            # test targets, so it's worked around here with headroom
            # rather than fixed.
            reply = client.chat(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=1000,
            )
        self.assertIn("OK", reply)


class TestBuildFallbackChain(unittest.TestCase):
    """_build_fallback_chain returns different model sets for quality vs fast."""

    def _build(self, quality: bool) -> list[str]:
        with patch.dict(
            "os.environ",
            {
                "GEMINI_API_KEY": "fake-gemini",
                "OPENAI_API_KEY": "fake-openai",
            },
        ):
            from applypilot.llm import _build_fallback_chain

            primary = "gemini-2.5-pro" if quality else "gemini-2.5-flash"
            return [m.name for m in _build_fallback_chain(primary, quality=quality)]

    def test_fast_chain_includes_flash_or_mini(self):
        names = self._build(quality=False)
        self.assertTrue(
            any("flash" in n or "nano" in n or "mini" in n for n in names),
            f"Fast chain should include a flash/nano/mini model. Got: {names}",
        )

    def test_quality_chain_includes_pro_or_sonnet(self):
        names = self._build(quality=True)
        self.assertTrue(
            any("pro" in n or "sonnet" in n for n in names),
            f"Quality chain should include a pro/sonnet model. Got: {names}",
        )

    def test_fast_and_quality_chains_differ(self):
        fast = self._build(quality=False)
        quality = self._build(quality=True)
        self.assertNotEqual(fast, quality, "Fast and quality chains should differ")

    def test_chain_nonempty_with_keys(self):
        names = self._build(quality=False)
        self.assertGreater(len(names), 0, "Should have at least one model when keys are set")

    def test_raises_without_api_keys(self):
        """RuntimeError is raised when no API keys or Claude CLI are available."""
        env_without_keys = {
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
        with patch.dict("os.environ", env_without_keys, clear=True):
            import applypilot.llm as llm_mod

            with (
                patch.object(llm_mod, "_find_claude_cli", return_value=None),
                self.assertRaises(RuntimeError),
            ):
                llm_mod._build_fallback_chain("gemini-2.5-flash", quality=False)


class TestLocalExcludedFromScoringButAvailableForExplicitLocalTasks(unittest.TestCase):
    """2026-08-23: local (Qwen) must never be part of the scoring/fast-tier
    fallback chain -- real-job testing found qwen3:1.7b's scoring unreliable
    (missed obvious SWE experience, scored known-9s as 7). The quality tier
    (tailor/cover) keeps local as its existing, deliberate last-resort
    fallback (has_cloud_available()/DEGRADED MODE in tailor.py) -- that is
    unchanged. Only the fast tier (get_stage_client("score", quality=False))
    stopped including local.

    These exercise the REAL _build_fallback_chain / LLMClient.chat() /
    _try_openai_compat routing end to end -- only httpx.Client.post is
    mocked (and, where noted, only the diagnostic-log call site itself) --
    not the routing decision under test.
    """

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
                "APPLYPILOT_LOCAL_LLM_URL",
                "APPLYPILOT_LOCAL_LLM_MODEL",
            )
        }
        env.update(overrides)
        return env

    # 1. Normal scoring does not include a local model in its fallback chain.
    def test_1_score_tier_chain_construction_excludes_local(self):
        from applypilot.llm import _build_fallback_chain

        env = self._env(
            GEMINI_API_KEY="fake-gemini-key",
            APPLYPILOT_LOCAL_LLM_URL="http://localhost:11434/v1",
            APPLYPILOT_LOCAL_LLM_MODEL="qwen3:1.7b",
        )
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            # Exactly what get_stage_client("score", quality=False) builds.
            chain = _build_fallback_chain("gemini-3.6-flash", quality=False)
        self.assertFalse(any(e.provider == "local" for e in chain))
        self.assertTrue(any(e.provider == "gemini" for e in chain))

    def test_1b_get_stage_client_score_end_to_end_excludes_local(self):
        """Same proof through the real get_stage_client("score", ...)
        singleton path scorer.py actually calls -- not just
        _build_fallback_chain in isolation."""
        import applypilot.llm as llm_mod

        env = self._env(
            GEMINI_API_KEY="fake-gemini-key",
            APPLYPILOT_LOCAL_LLM_URL="http://localhost:11434/v1",
            APPLYPILOT_LOCAL_LLM_MODEL="qwen3:1.7b",
        )
        saved = (llm_mod._instance, llm_mod._quality_instance, dict(llm_mod._stage_instances))
        llm_mod._instance = None
        llm_mod._quality_instance = None
        llm_mod._stage_instances = {}
        try:
            with (
                patch.dict("os.environ", env, clear=True),
                patch.object(llm_mod, "_find_claude_cli", return_value=None),
            ):
                client = llm_mod.get_stage_client("score", quality=False)
                providers = [e.provider for e in client._fallback_chain]
        finally:
            llm_mod._instance, llm_mod._quality_instance = saved[0], saved[1]
            llm_mod._stage_instances = saved[2]
        self.assertNotIn("local", providers)

    def test_1c_quality_tier_still_includes_local_unchanged(self):
        """The quality tier (tailor/cover) must be unaffected -- local
        remains the existing last-resort fallback there."""
        from applypilot.llm import _build_fallback_chain

        env = self._env(
            GEMINI_API_KEY="fake-gemini-key",
            APPLYPILOT_LOCAL_LLM_URL="http://localhost:11434/v1",
            APPLYPILOT_LOCAL_LLM_MODEL="qwen3:1.7b",
        )
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("gemini-3.1-pro-preview", quality=True)
        self.assertTrue(any(e.provider == "local" for e in chain))
        self.assertEqual(chain[-1].provider, "local")

    # 2. A cloud scoring failure falls through to another cloud provider,
    #    never to local.
    def test_2_cloud_failure_falls_through_to_next_cloud_not_local(self):
        from applypilot.llm import LLMClient

        env = self._env(
            GEMINI_API_KEY="fake-gemini-key",
            OPENAI_API_KEY="fake-openai-key",
            APPLYPILOT_LOCAL_LLM_URL="http://localhost:11434/v1",
            APPLYPILOT_LOCAL_LLM_MODEL="qwen3:1.7b",
        )
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            client = LLMClient(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                model="gemini-3.6-flash",
                api_key="fake-gemini-key",
                quality=False,
            )
        self.assertFalse(any(e.provider == "local" for e in client._fallback_chain))
        self.assertGreaterEqual(len(client._fallback_chain), 2)
        first_model = client._fallback_chain[0].name

        called_urls = []

        def fake_post(url, **kwargs):
            called_urls.append(url)
            if kwargs["json"]["model"] == first_model:
                resp = MagicMock()
                resp.status_code = 429
                resp.text = "rate_limit exceeded, please retry"
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"choices": [{"message": {"content": "second cloud reply"}}]}
            return resp

        with patch.object(client._client, "post", side_effect=fake_post):
            result = client.chat([{"role": "user", "content": "score this job"}])

        self.assertEqual(result, "second cloud reply")
        self.assertTrue(all("localhost" not in u and "127.0.0.1" not in u for u in called_urls))

    # 3. An explicitly local task uses Qwen.
    def test_3_explicit_local_task_reaches_qwen(self):
        """include_local=True is how an explicitly-local call site opts in
        regardless of quality tier -- mirrors what a genuine local-only
        client construction looks like."""
        from applypilot.llm import LLMClient, _build_fallback_chain

        env = self._env(
            APPLYPILOT_LOCAL_LLM_URL="http://localhost:11434/v1",
            APPLYPILOT_LOCAL_LLM_MODEL="qwen3:1.7b",
        )
        with patch.dict("os.environ", env, clear=True), patch("applypilot.llm._find_claude_cli", return_value=None):
            chain = _build_fallback_chain("qwen3:1.7b", quality=False, include_local=True)

        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].provider, "local")

        # Reuse the chain just proven above to build the client, rather than
        # re-deriving it inside LLMClient.__init__ under a different ambient
        # env -- the real _build_fallback_chain(include_local=True) call
        # above is the thing actually under test here.
        with patch("applypilot.llm._build_fallback_chain", return_value=chain):
            client = LLMClient(base_url="http://localhost:11434/v1", model="qwen3:1.7b", api_key="", quality=False)

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "qwen reply"}}]}
        with patch.object(client._client, "post", return_value=resp) as mock_post:
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "qwen reply")
        self.assertIn("localhost:11434", mock_post.call_args[0][0])

    # 4. An explicitly local task does NOT fall through to a cloud provider
    #    when Qwen fails.
    def test_4_explicit_local_task_does_not_fall_through_to_cloud_on_failure(self):
        import httpx as httpx_mod

        from applypilot.llm import LLMClient, ModelEntry

        # Pinned to local-only, exactly like cli.py's `test-local` command
        # does -- this IS the "explicitly local operation" boundary in this
        # codebase: the chain structurally contains no cloud entry at all.
        fake_chain = [ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", "")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="http://localhost:11434/v1", model="qwen3:1.7b", api_key="", quality=False)

        with (
            patch.object(client._client, "post", side_effect=httpx_mod.ConnectError("refused")),
            patch("applypilot.llm.time.sleep"),
            self.assertRaises(httpx_mod.ConnectError),
        ):
            client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual([e.provider for e in client._fallback_chain], ["local"])

    # 5 & 6. /no_think keyed on the actually-attempted model (not self.model);
    #        the caller's original messages list is never mutated.
    def test_5_and_6_no_think_keyed_on_attempted_model_and_messages_not_mutated(self):
        import copy

        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [
            ModelEntry("gemini-3.6-flash", "gemini", "https://fake.gemini/v1", "fake-key"),
            ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", ""),
        ]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(
                base_url="https://fake.gemini/v1", model="gemini-3.6-flash", api_key="fake-key", quality=False
            )

        original_messages = [{"role": "user", "content": "score this job"}]
        original_snapshot = copy.deepcopy(original_messages)
        calls = []

        def fake_post(url, **kwargs):
            calls.append(kwargs["json"])
            if kwargs["json"]["model"] == "gemini-3.6-flash":
                resp = MagicMock()
                resp.status_code = 429
                resp.text = "rate_limit exceeded, please retry"
                return resp
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"choices": [{"message": {"content": "qwen reply"}}]}
            return resp

        with patch.object(client._client, "post", side_effect=fake_post):
            result = client.chat(original_messages)

        self.assertEqual(result, "qwen reply")
        self.assertEqual(len(calls), 2)
        gemini_payload, qwen_payload = calls
        self.assertFalse(gemini_payload["messages"][0]["content"].startswith("/no_think"))
        self.assertTrue(qwen_payload["messages"][0]["content"].startswith("/no_think"))

        # requirement 6: the caller's original list/dicts are untouched --
        # not mutated in place, and no /no_think leaked into it even though
        # a later fallback attempt needed the prefix.
        self.assertEqual(original_messages, original_snapshot)

    def test_no_think_applied_when_system_message_precedes_user_message(self):
        """2026-09-04 regression, found via a real degraded-mode realization
        pilot run (data/experiments/deterministic_slotfiller_20260902/
        proxy_label_pilot.py): the /no_think injection only ever checked
        messages[0]'s role, so any system+user prompt (exactly the shape
        local_tailor.request_local_realization sends -- [{"role":"system"},
        {"role":"user"}]) had its SYSTEM message at index 0, and the
        'first.get("role") == "user"' check was always False -- /no_think
        silently never applied at all for that call shape. Confirmed live:
        15/15 real degraded-mode realization pilot calls failed regardless
        of evidence strength, 8 with exactly 'Null/empty content' (qwen3
        spending its whole max_tokens budget on unsuppressed <think>
        reasoning); a direct A/B (system-first vs. user-first, same
        trivial prompt) showed 39s vs. 17s, consistent with thinking left
        enabled. Fix: search for the first user-role message anywhere in
        the list, not just index 0."""
        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", "")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="http://localhost:11434/v1", model="qwen3:1.7b", api_key="", quality=True)

        messages = [
            {"role": "system", "content": "You are a resume assistant."},
            {"role": "user", "content": "Realize this bullet."},
        ]
        calls = []

        def fake_post(url, **kwargs):
            calls.append(kwargs["json"])
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
            return resp

        with patch.object(client._client, "post", side_effect=fake_post):
            client.chat(messages)

        self.assertEqual(len(calls), 1)
        sent = calls[0]["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertFalse(sent[0]["content"].startswith("/no_think"))
        self.assertEqual(sent[1]["role"], "user")
        self.assertTrue(sent[1]["content"].startswith("/no_think"))

    # 7. The local diagnostic log still receives successful local responses
    #    (and only local responses -- never cloud ones).
    def test_7_successful_local_response_reaches_the_diagnostic_logger(self):
        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", "")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(base_url="http://localhost:11434/v1", model="qwen3:1.7b", api_key="", quality=False)

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "qwen diagnostic reply"}}]}

        with (
            patch.object(client._client, "post", return_value=resp),
            patch("applypilot.llm._log_local_qwen_response") as mock_diag,
        ):
            result = client.chat([{"role": "user", "content": "diagnostic probe"}])

        self.assertEqual(result, "qwen diagnostic reply")
        mock_diag.assert_called_once()
        called_entry, _called_messages, called_text = mock_diag.call_args[0]
        self.assertEqual(called_entry.provider, "local")
        self.assertEqual(called_text, "qwen diagnostic reply")

    def test_7b_cloud_response_does_not_reach_the_local_diagnostic_logger(self):
        from applypilot.llm import LLMClient, ModelEntry

        fake_chain = [ModelEntry("gemini-3.6-flash", "gemini", "https://fake.gemini/v1", "fake-key")]
        with patch("applypilot.llm._build_fallback_chain", return_value=fake_chain):
            client = LLMClient(
                base_url="https://fake.gemini/v1", model="gemini-3.6-flash", api_key="fake-key", quality=False
            )

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "cloud reply"}}]}

        with (
            patch.object(client._client, "post", return_value=resp),
            patch("applypilot.llm._log_local_qwen_response") as mock_diag,
        ):
            client.chat([{"role": "user", "content": "hi"}])

        mock_diag.assert_not_called()


class TestChatExcludeProviders(unittest.TestCase):
    """2026-08-23: chat(exclude_providers=...) lets a caller attempt only a
    subset of the fallback chain for ONE call, without mutating
    self._fallback_chain (thread-unsafe on a shared singleton client) and
    without adding any new LLM/network call. Added so tailor.py's heavy
    cloud-quality prompt can discover "all cloud exhausted" via a fast,
    cloud-only attempt (each rejection is a ~1-2s 429, not a multi-minute
    local generation) instead of silently falling through to local with a
    prompt/max_tokens combination that was never meant for it."""

    def _client(self, chain):
        from applypilot.llm import LLMClient

        with patch("applypilot.llm._build_fallback_chain", return_value=chain):
            return LLMClient(base_url="https://fake.cloud/v1", model="cloud-a", api_key="fake-key")

    def test_excluded_provider_is_never_attempted(self):
        from applypilot.llm import ModelEntry

        chain = [
            ModelEntry("cloud-a", "gemini", "https://fake.gemini/v1", "fake-key"),
            ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", ""),
        ]
        client = self._client(chain)
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "cloud reply"}}]}

        with patch.object(client._client, "post", return_value=resp) as mock_post:
            result = client.chat([{"role": "user", "content": "hi"}], exclude_providers=frozenset({"local"}))

        self.assertEqual(result, "cloud reply")
        self.assertEqual(mock_post.call_count, 1)  # only the gemini entry was ever tried

    def test_all_non_excluded_entries_exhausted_raises_without_trying_excluded_one(self):
        """The realistic "cloud fully exhausted" case: every non-local
        entry is on cooldown. Must raise (fast, no local attempt) rather
        than silently falling through to the excluded local entry."""
        import time as time_mod

        from applypilot.llm import ModelEntry

        chain = [
            ModelEntry("cloud-a", "gemini", "https://fake.gemini/v1", "fake-key"),
            ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", ""),
        ]
        client = self._client(chain)
        client._exhausted["cloud-a"] = time_mod.time() + 86400  # 24h daily-quota exhaustion

        with patch.object(client._client, "post") as mock_post, self.assertRaises(RuntimeError):
            client.chat([{"role": "user", "content": "hi"}], exclude_providers=frozenset({"local"}))

        mock_post.assert_not_called()  # never even tried the excluded local entry

    def test_excluding_a_provider_does_not_mutate_the_fallback_chain(self):
        """Must be safe for concurrent callers sharing the same client
        (e.g. --workers > 1): exclude_providers is a per-call filter, not a
        mutation of shared state."""
        from applypilot.llm import ModelEntry

        chain = [
            ModelEntry("cloud-a", "gemini", "https://fake.gemini/v1", "fake-key"),
            ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", ""),
        ]
        client = self._client(chain)
        original_chain = list(client._fallback_chain)
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "cloud reply"}}]}

        with patch.object(client._client, "post", return_value=resp):
            client.chat([{"role": "user", "content": "hi"}], exclude_providers=frozenset({"local"}))

        self.assertEqual(client._fallback_chain, original_chain)

    def test_no_exclude_providers_behaves_exactly_as_before(self):
        """Backward compatibility: omitting exclude_providers (the default,
        every existing call site except tailor.py's heavy-prompt call) must
        not change behavior at all."""
        from applypilot.llm import ModelEntry

        chain = [ModelEntry("qwen3:1.7b", "local", "http://localhost:11434/v1", "")]
        client = self._client(chain)
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "local reply"}}]}

        with patch.object(client._client, "post", return_value=resp):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "local reply")


class TestClaudeCliFailureClassification(unittest.TestCase):
    """_try_claude_cli's failure classification.

    2026-08-19 incident: a Max-plan 5-hour usage window at 100% utilization
    made claude_cli/sonnet fail 50/50 tailor calls with exit 1 and
    completely empty stdout/stderr. The text-substring rate-limit check
    ("usage limit"/"rate limit"/"overloaded") can never match empty text,
    so every call fell through to a hard RuntimeError instead of backing
    off like a real rate-limit message does. These tests cover the
    empty-output heuristic added for that case, and confirm it doesn't
    swallow genuine errors that do carry real diagnostic text.
    """

    def _entry(self):
        from applypilot.llm import ModelEntry

        return ModelEntry("sonnet", "claude_cli", "/fake/path/to/claude", "")

    def _messages(self):
        return [{"role": "user", "content": "hello"}]

    def _fake_proc(self, returncode, stdout, stderr):
        from types import SimpleNamespace

        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def test_empty_output_nonzero_exit_marks_exhausted_not_raised(self):
        """Exit 1, empty stdout/stderr -- must back off gracefully, not
        raise RuntimeError, even when this is the last chain entry."""
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run", return_value=self._fake_proc(1, "", "")):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(result)
        self.assertIn(entry.name, client._exhausted)

    def test_text_matched_rate_limit_marks_exhausted(self):
        """Existing substring-match path still works unchanged."""
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run", return_value=self._fake_proc(1, "", "Error: usage limit reached")):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(result)
        self.assertIn(entry.name, client._exhausted)

    def test_genuine_error_with_text_still_raises_when_last(self):
        """A real error message (non-empty stderr, no limit keywords) must
        still surface as RuntimeError -- the empty-output heuristic only
        applies when there's truly nothing to diagnose from."""
        client = _make_client(1)
        entry = self._entry()
        with patch(
            "applypilot.llm.subprocess.run", return_value=self._fake_proc(1, "", "invalid --system-prompt argument")
        ), self.assertRaises(RuntimeError):
            client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertNotIn(entry.name, client._exhausted)

    def test_genuine_error_with_text_returns_none_when_not_last(self):
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run", return_value=self._fake_proc(1, "", "some other error")):
            result = client._try_claude_cli(entry, self._messages(), is_last=False)
        self.assertIsNone(result)

    def test_success_returns_stripped_stdout(self):
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run", return_value=self._fake_proc(0, "CLAUDE_TEST\n", "")):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertEqual(result, "CLAUDE_TEST")

    def test_session_limit_text_marks_exhausted(self):
        """2026-08-20 incident: the real message is 'session limit', not
        'usage limit' -- the original three keywords didn't match it."""
        client = _make_client(1)
        entry = self._entry()
        with patch(
            "applypilot.llm.subprocess.run",
            return_value=self._fake_proc(1, "You've hit your session limit · resets 4pm (America/New_York)", ""),
        ):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(result)
        self.assertIn(entry.name, client._exhausted)

    def test_stdout_only_diagnostic_is_captured_not_dropped(self):
        """A failure whose only diagnostic text is on stdout (empty stderr)
        must still surface that text -- the original code only ever read
        proc.stderr, so this exact case silently produced an empty error."""
        client = _make_client(1)
        entry = self._entry()
        with patch(
            "applypilot.llm.subprocess.run", return_value=self._fake_proc(1, "Error: invalid model configuration", "")
        ), self.assertRaises(RuntimeError) as ctx:
            client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIn("invalid model configuration", str(ctx.exception))

    def test_lock_short_circuits_after_concurrent_exhaustion(self):
        """Simulates the race this incident's cascade came from: another
        thread marks the entry exhausted between this thread deciding to
        try claude_cli and actually acquiring the lock. The re-check inside
        the lock must skip the subprocess call entirely rather than spawning
        another doomed ~3-30s claude invocation."""
        client = _make_client(1)
        entry = self._entry()
        client._exhausted[entry.name] = time.time() + 1800  # pre-exhausted
        with patch("applypilot.llm.subprocess.run") as mock_run:
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(result)
        mock_run.assert_not_called()


class TestClaudeCliExhaustionReasonAndCapacityProbe(unittest.TestCase):
    """The scheduler.py-facing additions: claude_cli_exhaustion_reason()
    (distinguish quota-text-match vs empty-output-heuristic exhaustion
    without parsing logs) and has_available_model() (cheap read-only
    capacity probe). Both are purely additive -- _try_claude_cli's
    str | None return contract and existing callers are unchanged.
    """

    def _entry(self):
        from applypilot.llm import ModelEntry

        return ModelEntry("sonnet", "claude_cli", "/fake/path/to/claude", "")

    def _messages(self):
        return [{"role": "user", "content": "hello"}]

    def _fake_proc(self, returncode, stdout, stderr):
        from types import SimpleNamespace

        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def test_reason_is_quota_text_match(self):
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run", return_value=self._fake_proc(1, "", "Error: usage limit reached")):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(result)
        self.assertEqual(client.claude_cli_exhaustion_reason(entry.name), "quota_text_match")

    def test_reason_is_empty_output_heuristic(self):
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run", return_value=self._fake_proc(1, "", "")):
            client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertEqual(client.claude_cli_exhaustion_reason(entry.name), "empty_output_heuristic")

    def test_reason_is_none_when_not_exhausted(self):
        client = _make_client(1)
        entry = self._entry()
        self.assertIsNone(client.claude_cli_exhaustion_reason(entry.name))
        with patch("applypilot.llm.subprocess.run", return_value=self._fake_proc(0, "ok", "")):
            client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(client.claude_cli_exhaustion_reason(entry.name))

    def test_reason_cleared_once_exhaustion_window_expires(self):
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run", return_value=self._fake_proc(1, "", "")):
            client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNotNone(client.claude_cli_exhaustion_reason(entry.name))
        client._exhausted[entry.name] = time.time() - 1  # force-expire
        self.assertIsNone(client.claude_cli_exhaustion_reason(entry.name))

    def test_has_available_model_true_when_nothing_exhausted(self):
        client = _make_client(3)
        self.assertTrue(client.has_available_model())

    def test_has_available_model_false_when_everything_exhausted(self):
        client = _make_client(2)
        for entry in client._fallback_chain:
            client._exhausted[entry.name] = time.time() + 1800
        self.assertFalse(client.has_available_model())

    def test_has_available_model_true_when_some_entries_free(self):
        client = _make_client(3)
        client._exhausted[client._fallback_chain[0].name] = time.time() + 1800
        self.assertTrue(client.has_available_model())

    def test_has_available_model_makes_no_calls(self):
        """Pure read of in-memory state -- must never invoke a network or
        subprocess call just to answer the capacity question."""
        client = _make_client(3)
        with patch("applypilot.llm.subprocess.run") as mock_run, patch("applypilot.llm.httpx.Client.post") as mock_post:
            client.has_available_model()
        mock_run.assert_not_called()
        mock_post.assert_not_called()


class TestClaudeCliReserveForApply(unittest.TestCase):
    """APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY gates whether claude_cli is ever
    added to a fallback chain built by this module.

    Nothing in llm.py serves the auto-apply stage -- apply/launcher.py talks
    to the `claude` CLI directly and never imports this module -- so every
    caller of _build_fallback_chain (score/tailor/cover/judge/discover/
    enrich/tracking) is, by definition, a "normal" stage. A single global
    gate is therefore sufficient to express "reserve Claude for auto-apply"
    without needing per-stage plumbing.
    """

    def _chain(self, reserve_value: str | None, quality: bool = False):
        import applypilot.llm as llm_mod

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
                "APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY",
                "APPLYPILOT_LOCAL_LLM_URL",
            )
        }
        env["GEMINI_API_KEY"] = "fake-gemini-key"  # keep the chain non-empty
        if reserve_value is not None:
            env["APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY"] = reserve_value
        with (
            patch.dict("os.environ", env, clear=True),
            patch.object(llm_mod, "_find_claude_cli", return_value="/fake/path/to/claude"),
        ):
            return llm_mod._build_fallback_chain("gemini-3.6-flash", quality=quality)

    def test_claude_cli_excluded_by_default(self):
        providers = [e.provider for e in self._chain(reserve_value=None)]
        self.assertNotIn("claude_cli", providers)

    def test_claude_cli_excluded_when_explicitly_true(self):
        providers = [e.provider for e in self._chain(reserve_value="true")]
        self.assertNotIn("claude_cli", providers)

    def test_claude_cli_included_when_reserve_disabled(self):
        providers = [e.provider for e in self._chain(reserve_value="false")]
        self.assertIn("claude_cli", providers)

    def test_unreserved_claude_cli_still_quality_aware(self):
        """Turning the reserve off shouldn't change the existing
        sonnet-for-quality / haiku-for-fast model selection."""
        fast = next(e for e in self._chain("false", quality=False) if e.provider == "claude_cli")
        quality = next(e for e in self._chain("false", quality=True) if e.provider == "claude_cli")
        self.assertEqual(fast.name, "haiku")
        self.assertEqual(quality.name, "sonnet")

    def test_reserved_and_no_other_provider_raises_with_explanatory_note(self):
        """When claude_cli is the only thing installed and it's reserved,
        the resulting RuntimeError should say why rather than just looking
        like no provider was ever configured."""
        import applypilot.llm as llm_mod

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
                "APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY",
                "APPLYPILOT_LOCAL_LLM_URL",
            )
        }
        with (
            patch.dict("os.environ", env, clear=True),
            patch.object(llm_mod, "_find_claude_cli", return_value="/fake/path/to/claude"),
            self.assertRaises(RuntimeError) as ctx,
        ):
            llm_mod._build_fallback_chain("gemini-3.6-flash", quality=False)
        self.assertIn("reserved for auto-apply", str(ctx.exception))


class TestFullCascadeIntegration(unittest.TestCase):
    """End-to-end cascade tests using real provider names and the real
    _build_fallback_chain, exercising the actual quota-detection logic in
    _try_openai_compat/_try_claude_cli rather than a mocked _try_entry.

    2026-08-21 incident: a cover run exhausted all 4 configured API models
    (gemini-3.6-flash, gemini-3.5-flash, gpt-4.1-mini, gpt-4.1-nano) and the
    RuntimeError's model list didn't include Claude. Root cause found by
    inspecting the current repo (not assumed from memory): this is the
    intended behavior of APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY defaulting to
    true (added in adf6deb), which excludes claude_cli from every chain
    built by this module so score/tailor/cover/judge never compete with
    auto-apply for the shared Max-plan session window. These tests exercise
    that exact scenario end-to-end, both with and without the reserve.
    """

    def _fake_response(self, status_code, text="", json_data=None):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        if json_data is not None:
            resp.json.return_value = json_data
        resp.raise_for_status = lambda: None
        return resp

    def _success_json(self, text="ok"):
        return {"choices": [{"message": {"content": text}}]}

    def _fake_proc(self, returncode=0, stdout="", stderr=""):
        from types import SimpleNamespace

        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def _build_real_client(self, env, quality=False, primary=None, claude_cli_path=None):
        """Construct a real LLMClient (real fallback chain, real provider
        names) against a controlled environment. HTTP/subprocess calls are
        mocked by the individual tests, not here."""
        import applypilot.llm as llm_mod

        base_env = {
            k: v
            for k, v in __import__("os").environ.items()
            if k
            not in (
                "GEMINI_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "DEEPSEEK_API_KEY",
                "LLM_URL",
                "APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY",
                "APPLYPILOT_LOCAL_LLM_URL",
            )
        }
        base_env.update(env)
        primary = primary or ("gemini-3.1-pro-preview" if quality else "gemini-3.6-flash")
        with (
            patch.dict("os.environ", base_env, clear=True),
            patch.object(llm_mod, "_find_claude_cli", return_value=claude_cli_path),
        ):
            return llm_mod.LLMClient(
                "https://fake",
                primary,
                base_env.get("GEMINI_API_KEY", ""),
                quality=quality,
            )

    def test_gemini_quota_exhaustion_falls_through_to_openai(self):
        """Gemini exhaustion causes fallback to OpenAI."""
        client = self._build_real_client({"GEMINI_API_KEY": "fake-gemini", "OPENAI_API_KEY": "fake-openai"})
        gemini_names = {e.name for e in client._fallback_chain if e.provider == "gemini"}
        self.assertTrue(gemini_names, "test needs at least one gemini entry in the chain")

        def fake_post(url, json=None, headers=None):
            if json["model"] in gemini_names:
                return self._fake_response(429, text='{"error": "Quota exceeded"}')
            return self._fake_response(200, json_data=self._success_json("openai response"))

        with patch.object(client._client, "post", side_effect=fake_post):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "openai response")
        for name in gemini_names:
            self.assertIn(name, client._exhausted)

    def test_openai_quota_exhaustion_falls_through_within_openai(self):
        """OpenAI exhaustion causes fallback (here: to the next OpenAI
        model, since only OPENAI_API_KEY is configured)."""
        client = self._build_real_client({"OPENAI_API_KEY": "fake-openai"})
        openai_entries = [e for e in client._fallback_chain if e.provider == "openai"]
        self.assertGreaterEqual(len(openai_entries), 2, "need 2+ openai models to test fallback")
        first, second = openai_entries[0], openai_entries[1]

        def fake_post(url, json=None, headers=None):
            if json["model"] == first.name:
                return self._fake_response(429, text='{"error": "quota exceeded"}')
            return self._fake_response(200, json_data=self._success_json("second openai model"))

        with patch.object(client._client, "post", side_effect=fake_post):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "second openai model")
        self.assertIn(first.name, client._exhausted)
        self.assertNotIn(second.name, client._exhausted)

    def test_openai_insufficient_quota_marks_exhausted_much_longer_than_rate_limit(self):
        """2026-09-01 fix: OpenAI's actual wording for a $0 account balance
        is "insufficient_quota" -- a billing failure, not a rate limit --
        but it contains the substring "quota" and used to fall into the
        SAME 24h-cooldown branch as a genuine daily quota reset. Confirmed
        live against a real zero-balance account. A billing failure must
        get a much longer cooldown (it will not self-resolve tomorrow) and
        must still fall through to the next model in the same call."""
        client = self._build_real_client({"OPENAI_API_KEY": "fake-openai"})
        openai_entries = [e for e in client._fallback_chain if e.provider == "openai"]
        self.assertGreaterEqual(len(openai_entries), 2, "need 2+ openai models to test fallback")
        first, second = openai_entries[0], openai_entries[1]

        def fake_post(url, json=None, headers=None):
            if json["model"] == first.name:
                return self._fake_response(
                    429,
                    text=(
                        '{"error": {"message": "You have no credits remaining.", '
                        '"type": "insufficient_quota", "code": "credit_balance_exhausted"}}'
                    ),
                )
            return self._fake_response(200, json_data=self._success_json("second openai model"))

        with patch.object(client._client, "post", side_effect=fake_post):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "second openai model")
        self.assertIn(first.name, client._exhausted)
        # Real quota (test above) marks ~24h out; billing failure must be
        # marked meaningfully longer -- checked as "> 7 days from now" so
        # this doesn't hardcode the exact 30-day constant and break on a
        # future tuning change, while still failing if it regresses to 24h.
        self.assertGreater(client._exhausted[first.name], time.time() + 7 * 86400)

    def test_claude_cli_reached_when_apis_exhausted_and_unreserved(self):
        """Claude CLI is reached when earlier providers are exhausted --
        with the reserve explicitly turned off."""
        client = self._build_real_client(
            {
                "GEMINI_API_KEY": "fake-gemini",
                "OPENAI_API_KEY": "fake-openai",
                "APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY": "false",
            },
            claude_cli_path="/fake/claude",
        )
        self.assertTrue(
            any(e.provider == "claude_cli" for e in client._fallback_chain),
            "claude_cli should be in the chain when unreserved",
        )

        def fake_post(url, json=None, headers=None):
            return self._fake_response(429, text='{"error": "quota exceeded"}')

        with (
            patch.object(client._client, "post", side_effect=fake_post),
            patch("applypilot.llm.subprocess.run", return_value=self._fake_proc(0, "claude response", "")) as mock_run,
        ):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "claude response")
        mock_run.assert_called_once()

    def test_claude_cli_not_reached_when_reserved_matches_incident(self):
        """The exact 2026-08-21 incident: Claude installed, all 4 API
        models exhausted, reserve at its default (true) -- claude_cli must
        never be touched, and the RuntimeError's model list must match
        what the user actually saw (no sonnet/haiku in it)."""
        client = self._build_real_client(
            {"GEMINI_API_KEY": "fake-gemini", "OPENAI_API_KEY": "fake-openai"},
            claude_cli_path="/fake/claude",  # installed, but reserved by default
        )
        self.assertNotIn("claude_cli", [e.provider for e in client._fallback_chain])

        def fake_post(url, json=None, headers=None):
            return self._fake_response(429, text='{"error": "quota exceeded"}')

        with (
            patch.object(client._client, "post", side_effect=fake_post),
            patch("applypilot.llm.subprocess.run") as mock_run,
            self.assertRaises(RuntimeError) as ctx,
        ):
            client.chat([{"role": "user", "content": "hi"}])

        mock_run.assert_not_called()
        message = str(ctx.exception)
        self.assertIn("All models exhausted", message)
        self.assertNotIn("sonnet", message)
        self.assertNotIn("haiku", message)

    def test_unrelated_provider_not_marked_exhausted_on_sibling_failure(self):
        """A non-quota failure on one entry doesn't mark a sibling entry
        (or itself, for a per-request error) exhausted."""
        client = self._build_real_client({"GEMINI_API_KEY": "fake-gemini", "OPENAI_API_KEY": "fake-openai"})
        gemini_first = next(e for e in client._fallback_chain if e.provider == "gemini")

        def fake_post(url, json=None, headers=None):
            if json["model"] == gemini_first.name:
                # Content-safety-style 400 -- explicitly NOT a quota signal.
                return self._fake_response(400, text='{"error": "content filtered"}')
            return self._fake_response(200, json_data=self._success_json("fallback ok"))

        with patch.object(client._client, "post", side_effect=fake_post):
            client.chat([{"role": "user", "content": "hi"}])

        self.assertNotIn(gemini_first.name, client._exhausted)

    def test_all_exhausted_raises_runtime_error_with_model_list(self):
        """All-model exhaustion still raises the expected RuntimeError,
        naming every model that was attempted."""
        client = self._build_real_client({"GEMINI_API_KEY": "fake-gemini", "OPENAI_API_KEY": "fake-openai"})
        chain_names = [e.name for e in client._fallback_chain]

        def fake_post(url, json=None, headers=None):
            return self._fake_response(429, text='{"error": "quota exceeded"}')

        with (
            patch.object(client._client, "post", side_effect=fake_post),
            self.assertRaises(RuntimeError) as ctx,
        ):
            client.chat([{"role": "user", "content": "hi"}])

        message = str(ctx.exception)
        self.assertIn("All models exhausted", message)
        for name in chain_names:
            self.assertIn(name, message)


class TestExhaustionPersistence(unittest.TestCase):
    """Exhaustion state survives a process restart (2026-09-02) -- a fresh
    LLMClient should start already knowing what a previous instance
    learned, instead of re-discovering it via a real call every restart.
    Relies on conftest.py's suite-wide _isolate_llm_exhaustion_state
    autouse fixture (applies to unittest.TestCase methods too, and to
    every test file, not just this one -- see conftest.py for the
    2026-09-04 leak this was promoted to fix) for a fresh, isolated state
    path per test -- no manual setUp/tearDown needed."""

    def _state_path(self):
        import applypilot.llm as llm_mod

        return llm_mod._EXHAUSTION_STATE_PATH

    def test_mark_exhausted_persists_and_a_new_client_loads_it(self):
        client = _make_client(2)
        first_name = client._fallback_chain[0].name
        until = time.time() + 3600

        client._mark_exhausted(first_name, until)
        self.assertTrue(self._state_path().exists(), "marking exhausted should write the state file")

        second_client = _make_client(2)
        self.assertIn(first_name, second_client._exhausted)
        self.assertAlmostEqual(second_client._exhausted[first_name], until, delta=1)

    def test_expired_entries_are_dropped_on_load(self):
        import json

        state_path = self._state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"stale-model": time.time() - 100, "live-model": time.time() + 100}))

        client = _make_client(2)
        self.assertNotIn("stale-model", client._exhausted, "already-expired entries must not be loaded")
        self.assertIn("live-model", client._exhausted)

    def test_missing_state_file_is_not_an_error(self):
        self.assertFalse(self._state_path().exists())
        client = _make_client(2)  # must not raise
        self.assertEqual(client._exhausted, {})

    def test_corrupt_state_file_falls_back_to_empty(self):
        state_path = self._state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("not valid json{{{")

        client = _make_client(2)  # must not raise
        self.assertEqual(client._exhausted, {})

    def test_persisting_one_provider_does_not_clobber_another_instances_entry(self):
        """Fast-tier and quality-tier clients are separate LLMClient
        instances sharing the same on-disk file -- marking one provider
        exhausted from one instance must not erase an entry a different
        instance already wrote (a blind whole-file overwrite would)."""
        fast_client = _make_client(2)
        quality_client = _make_client(2)
        fast_name = fast_client._fallback_chain[0].name
        quality_name = quality_client._fallback_chain[1].name

        fast_client._mark_exhausted(fast_name, time.time() + 3600)
        quality_client._mark_exhausted(quality_name, time.time() + 3600)

        third_client = _make_client(2)
        self.assertIn(fast_name, third_client._exhausted)
        self.assertIn(quality_name, third_client._exhausted)

    def test_write_failure_does_not_raise(self):
        """A persist failure (disk full, permissions) must never break a
        live chat() call -- only the real exhaustion tracking matters."""
        client = _make_client(2)
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            client._mark_exhausted(client._fallback_chain[0].name, time.time())  # must not raise
        self.assertIn(client._fallback_chain[0].name, client._exhausted, "in-memory state still updates")


class TestPersistent503FallsThroughToNextProvider(unittest.TestCase):
    """2026-09-04 production crash: a real `applypilot run tailor` run hit
    gemini-3.5-flash returning 503 three times in a row (exhausting
    _MAX_RETRIES) while it was NOT the last entry in the fallback chain --
    gpt-4.1-mini/gpt-4.1-nano/qwen3:1.7b were still available -- and the
    whole job crashed with an uncaught httpx.HTTPStatusError instead of
    falling through. Root cause: the 503 branch in _try_openai_compat only
    ever handled "retry while attempts remain," then fell straight into an
    unconditional resp.raise_for_status() once retries were exhausted,
    regardless of is_last -- unlike the 429 branch a few lines above it,
    which explicitly has a third `elif not is_last: return None` case.
    Fixed by giving 503 the same three-way structure as 429."""

    def _persistent_503_response(self):
        resp = MagicMock()
        resp.status_code = 503
        resp.text = "Service Unavailable"

        def _raise():
            import httpx as httpx_mod

            raise httpx_mod.HTTPStatusError("503", request=MagicMock(), response=resp)

        resp.raise_for_status.side_effect = _raise
        return resp

    def _ok_response(self, content="ok"):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        return resp

    def test_persistent_503_on_non_last_entry_falls_through(self):
        client = _make_client(2)
        first_name = client._fallback_chain[0].name

        def fake_post(url, **kwargs):
            if kwargs["json"]["model"] == first_name:
                return self._persistent_503_response()
            return self._ok_response("second provider answered")

        with patch.object(client._client, "post", side_effect=fake_post), patch("time.sleep"):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "second provider answered")

    def test_persistent_503_on_last_entry_raises(self):
        """When there's truly nowhere else to fall through to, the 503
        must still surface as an error -- this fix only adds the missing
        middle case, it doesn't remove the legitimate last-resort raise."""
        client = _make_client(1)
        with patch.object(client._client, "post", return_value=self._persistent_503_response()), patch("time.sleep"):
            with self.assertRaises(Exception):
                client.chat([{"role": "user", "content": "hi"}])

    def test_503_that_resolves_within_retries_still_succeeds(self):
        """Regression guard for the retry-then-succeed path this fix sits
        next to -- must not have been broken by restructuring the branch."""
        client = _make_client(1)
        call_count = {"n": 0}

        def fake_post(url, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                return self._persistent_503_response()
            return self._ok_response("recovered")

        with patch.object(client._client, "post", side_effect=fake_post), patch("time.sleep"):
            result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result, "recovered")
        self.assertEqual(call_count["n"], 2)


class TestFrequencyPresencePenaltyPassthrough(unittest.TestCase):
    """2026-09-04, added for the sentence-diversity bake-off: chat() can
    forward frequency_penalty/presence_penalty to an OpenAI-compatible
    endpoint. Purely additive -- omitted by default, must not change
    payloads for any caller that doesn't pass them."""

    def _ok_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        return resp

    def test_omitted_by_default_not_sent(self):
        client = _make_client(1)
        with patch.object(client._client, "post", return_value=self._ok_response()) as mock_post:
            client.chat([{"role": "user", "content": "hi"}])
        payload = mock_post.call_args.kwargs["json"]
        self.assertNotIn("frequency_penalty", payload)
        self.assertNotIn("presence_penalty", payload)

    def test_forwarded_when_provided(self):
        client = _make_client(1)
        with patch.object(client._client, "post", return_value=self._ok_response()) as mock_post:
            client.chat(
                [{"role": "user", "content": "hi"}],
                frequency_penalty=0.6,
                presence_penalty=0.4,
            )
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["frequency_penalty"], 0.6)
        self.assertEqual(payload["presence_penalty"], 0.4)

    def test_zero_is_forwarded_not_treated_as_omitted(self):
        """0.0 is a meaningful, explicit value (the OpenAI-compat default),
        distinct from None (field omitted entirely) -- must not be dropped
        by an `if frequency_penalty:` falsy check."""
        client = _make_client(1)
        with patch.object(client._client, "post", return_value=self._ok_response()) as mock_post:
            client.chat([{"role": "user", "content": "hi"}], frequency_penalty=0.0)
        payload = mock_post.call_args.kwargs["json"]
        self.assertIn("frequency_penalty", payload)
        self.assertEqual(payload["frequency_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
