"""Tests for the LLM client cascade and exhaustion state machine.

Covers:
- Exhausted models are skipped within the cooldown window
- Exhaustion clears after the cooldown period
- All models exhausted: client clears state and retries
- Failed _try_entry marks model exhausted and falls through to next
- _build_fallback_chain returns quality vs fast model sets correctly
- _build_fallback_chain raises RuntimeError when no keys are configured
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _make_client(n_models: int = 2):
    """Instantiate an LLMClient with fake models injected directly into the chain."""
    from applypilot.llm import LLMClient, ModelEntry

    # Patch _build_fallback_chain to return fake models so no real API keys needed
    fake_chain = [
        ModelEntry(f"fake-model-{i}", "openai_compat", "https://fake.api/v1", "fake-key")
        for i in range(n_models)
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

        def fake_try(entry, messages, temperature, max_tokens, is_last):
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

        def fake_try(entry, messages, temperature, max_tokens, is_last):
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

        def fake_try(entry, messages, temperature, max_tokens, is_last):
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

        def fake_try(entry, messages, temperature, max_tokens, is_last):
            call_order.append(entry.name)
            if entry.name == first_name:
                return None  # simulate failure
            return "success from fallback"

        with patch.object(client, "_try_entry", side_effect=fake_try):
            result = client.chat([{"role": "user", "content": "test"}])

        self.assertIn(first_name, call_order)
        self.assertIn(second_name, call_order)
        self.assertEqual(result, "success from fallback")


class TestBuildFallbackChain(unittest.TestCase):
    """_build_fallback_chain returns different model sets for quality vs fast."""

    def _build(self, quality: bool) -> list[str]:
        with patch.dict("os.environ", {
            "GEMINI_API_KEY": "fake-gemini",
            "OPENAI_API_KEY": "fake-openai",
        }):
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
            k: v for k, v in __import__("os").environ.items()
            if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY",
                         "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "LLM_URL")
        }
        with patch.dict("os.environ", env_without_keys, clear=True):
            import applypilot.llm as llm_mod
            with patch.object(llm_mod, "_find_claude_cli", return_value=None):
                with self.assertRaises(RuntimeError):
                    llm_mod._build_fallback_chain("gemini-2.5-flash", quality=False)


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
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(1, "", "")):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(result)
        self.assertIn(entry.name, client._exhausted)

    def test_text_matched_rate_limit_marks_exhausted(self):
        """Existing substring-match path still works unchanged."""
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(1, "", "Error: usage limit reached")):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(result)
        self.assertIn(entry.name, client._exhausted)

    def test_genuine_error_with_text_still_raises_when_last(self):
        """A real error message (non-empty stderr, no limit keywords) must
        still surface as RuntimeError -- the empty-output heuristic only
        applies when there's truly nothing to diagnose from."""
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(1, "", "invalid --system-prompt argument")):
            with self.assertRaises(RuntimeError):
                client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertNotIn(entry.name, client._exhausted)

    def test_genuine_error_with_text_returns_none_when_not_last(self):
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(1, "", "some other error")):
            result = client._try_claude_cli(entry, self._messages(), is_last=False)
        self.assertIsNone(result)

    def test_success_returns_stripped_stdout(self):
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(0, "CLAUDE_TEST\n", "")):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertEqual(result, "CLAUDE_TEST")

    def test_session_limit_text_marks_exhausted(self):
        """2026-08-20 incident: the real message is 'session limit', not
        'usage limit' -- the original three keywords didn't match it."""
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(
                        1, "You've hit your session limit · resets 4pm (America/New_York)", "")):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(result)
        self.assertIn(entry.name, client._exhausted)

    def test_stdout_only_diagnostic_is_captured_not_dropped(self):
        """A failure whose only diagnostic text is on stdout (empty stderr)
        must still surface that text -- the original code only ever read
        proc.stderr, so this exact case silently produced an empty error."""
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(1, "Error: invalid model configuration", "")):
            with self.assertRaises(RuntimeError) as ctx:
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
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(1, "", "Error: usage limit reached")):
            result = client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(result)
        self.assertEqual(client.claude_cli_exhaustion_reason(entry.name), "quota_text_match")

    def test_reason_is_empty_output_heuristic(self):
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(1, "", "")):
            client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertEqual(client.claude_cli_exhaustion_reason(entry.name), "empty_output_heuristic")

    def test_reason_is_none_when_not_exhausted(self):
        client = _make_client(1)
        entry = self._entry()
        self.assertIsNone(client.claude_cli_exhaustion_reason(entry.name))
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(0, "ok", "")):
            client._try_claude_cli(entry, self._messages(), is_last=True)
        self.assertIsNone(client.claude_cli_exhaustion_reason(entry.name))

    def test_reason_cleared_once_exhaustion_window_expires(self):
        client = _make_client(1)
        entry = self._entry()
        with patch("applypilot.llm.subprocess.run",
                    return_value=self._fake_proc(1, "", "")):
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
        with patch("applypilot.llm.subprocess.run") as mock_run, \
             patch("applypilot.llm.httpx.Client.post") as mock_post:
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
            k: v for k, v in __import__("os").environ.items()
            if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                         "DEEPSEEK_API_KEY", "LLM_URL", "APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY")
        }
        env["GEMINI_API_KEY"] = "fake-gemini-key"  # keep the chain non-empty
        if reserve_value is not None:
            env["APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY"] = reserve_value
        with patch.dict("os.environ", env, clear=True):
            with patch.object(llm_mod, "_find_claude_cli", return_value="/fake/path/to/claude"):
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
            k: v for k, v in __import__("os").environ.items()
            if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                         "DEEPSEEK_API_KEY", "LLM_URL", "APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY")
        }
        with patch.dict("os.environ", env, clear=True):
            with patch.object(llm_mod, "_find_claude_cli", return_value="/fake/path/to/claude"):
                with self.assertRaises(RuntimeError) as ctx:
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
            k: v for k, v in __import__("os").environ.items()
            if k not in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                         "DEEPSEEK_API_KEY", "LLM_URL", "APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY")
        }
        base_env.update(env)
        primary = primary or ("gemini-3.1-pro-preview" if quality else "gemini-3.6-flash")
        with patch.dict("os.environ", base_env, clear=True):
            with patch.object(llm_mod, "_find_claude_cli", return_value=claude_cli_path):
                return llm_mod.LLMClient(
                    "https://fake", primary, base_env.get("GEMINI_API_KEY", ""), quality=quality,
                )

    def test_gemini_quota_exhaustion_falls_through_to_openai(self):
        """Gemini exhaustion causes fallback to OpenAI."""
        client = self._build_real_client(
            {"GEMINI_API_KEY": "fake-gemini", "OPENAI_API_KEY": "fake-openai"})
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

    def test_claude_cli_reached_when_apis_exhausted_and_unreserved(self):
        """Claude CLI is reached when earlier providers are exhausted --
        with the reserve explicitly turned off."""
        client = self._build_real_client(
            {"GEMINI_API_KEY": "fake-gemini", "OPENAI_API_KEY": "fake-openai",
             "APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY": "false"},
            claude_cli_path="/fake/claude",
        )
        self.assertTrue(
            any(e.provider == "claude_cli" for e in client._fallback_chain),
            "claude_cli should be in the chain when unreserved",
        )

        def fake_post(url, json=None, headers=None):
            return self._fake_response(429, text='{"error": "quota exceeded"}')

        with patch.object(client._client, "post", side_effect=fake_post):
            with patch("applypilot.llm.subprocess.run",
                        return_value=self._fake_proc(0, "claude response", "")) as mock_run:
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

        with patch.object(client._client, "post", side_effect=fake_post):
            with patch("applypilot.llm.subprocess.run") as mock_run:
                with self.assertRaises(RuntimeError) as ctx:
                    client.chat([{"role": "user", "content": "hi"}])

        mock_run.assert_not_called()
        message = str(ctx.exception)
        self.assertIn("All models exhausted", message)
        self.assertNotIn("sonnet", message)
        self.assertNotIn("haiku", message)

    def test_unrelated_provider_not_marked_exhausted_on_sibling_failure(self):
        """A non-quota failure on one entry doesn't mark a sibling entry
        (or itself, for a per-request error) exhausted."""
        client = self._build_real_client(
            {"GEMINI_API_KEY": "fake-gemini", "OPENAI_API_KEY": "fake-openai"})
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
        client = self._build_real_client(
            {"GEMINI_API_KEY": "fake-gemini", "OPENAI_API_KEY": "fake-openai"})
        chain_names = [e.name for e in client._fallback_chain]

        def fake_post(url, json=None, headers=None):
            return self._fake_response(429, text='{"error": "quota exceeded"}')

        with patch.object(client._client, "post", side_effect=fake_post):
            with self.assertRaises(RuntimeError) as ctx:
                client.chat([{"role": "user", "content": "hi"}])

        message = str(ctx.exception)
        self.assertIn("All models exhausted", message)
        for name in chain_names:
            self.assertIn(name, message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
