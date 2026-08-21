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


if __name__ == "__main__":
    unittest.main(verbosity=2)
