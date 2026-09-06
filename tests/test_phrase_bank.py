"""Tests for scoring/phrase_bank.py (persistence) and
local_tailor.build_phrase_bank (generation) -- "Stage 0" of tailoring,
see local_tailor.py's phrase-bank-builder module comment.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.scoring import local_tailor, phrase_bank  # noqa: E402


class TestSlugifyAndContentHash(unittest.TestCase):
    def test_slugify_produces_a_safe_filename_stem(self):
        self.assertEqual(phrase_bank.slugify("National Tire and Battery / Mavis"), "national_tire_and_battery_mavis")

    def test_slugify_never_empty(self):
        self.assertEqual(phrase_bank.slugify("!!!"), "entry")

    def test_content_hash_stable_for_same_facts(self):
        item = {"responsibilities": ["Diagnosed vehicle alignment issues.", "Operated diagnostic equipment."]}
        self.assertEqual(phrase_bank.content_hash(item), phrase_bank.content_hash(dict(item)))

    def test_content_hash_changes_when_facts_change(self):
        item_a = {"responsibilities": ["Diagnosed vehicle alignment issues."]}
        item_b = {"responsibilities": ["Diagnosed vehicle alignment issues.", "Operated diagnostic equipment."]}
        self.assertNotEqual(phrase_bank.content_hash(item_a), phrase_bank.content_hash(item_b))

    def test_content_hash_unaffected_by_unrelated_fields(self):
        item_a = {"name": "Mavis", "responsibilities": ["Diagnosed vehicle alignment issues."]}
        item_b = {"name": "Different Name", "responsibilities": ["Diagnosed vehicle alignment issues."]}
        self.assertEqual(phrase_bank.content_hash(item_a), phrase_bank.content_hash(item_b))

    def test_source_facts_falls_back_to_factual_concepts_when_no_responsibilities(self):
        item = {"factual_concepts": ["Python", "OCR"]}
        self.assertEqual(phrase_bank.source_facts(item), ["Python", "OCR"])

    def test_source_facts_prefers_responsibilities(self):
        item = {"responsibilities": ["Did the real thing."], "factual_concepts": ["ignored"]}
        self.assertEqual(phrase_bank.source_facts(item), ["Did the real thing."])


class TestSaveAndLoadBank(unittest.TestCase):
    def setUp(self):
        self._tmpdir_patch = patch.object(phrase_bank, "PHRASE_BANK_DIR", Path(self._make_tmp_dir()))
        self._tmpdir_patch.start()
        self.addCleanup(self._tmpdir_patch.stop)

    def _make_tmp_dir(self):
        import tempfile

        d = tempfile.mkdtemp()
        self._tmp_dirs = getattr(self, "_tmp_dirs", [])
        self._tmp_dirs.append(d)
        return d

    def test_save_then_load_roundtrips(self):
        bank = {"Diagnosed vehicle alignment issues.": ["Identified and resolved alignment problems."]}
        path = phrase_bank.save_bank("Mavis", bank, "hash123")
        self.assertIsNotNone(path)
        loaded = phrase_bank.load_bank("Mavis", "hash123")
        self.assertEqual(loaded, bank)

    def test_load_returns_none_when_hash_mismatches(self):
        bank = {"Diagnosed vehicle alignment issues.": ["Identified and resolved alignment problems."]}
        phrase_bank.save_bank("Mavis", bank, "hash123")
        self.assertIsNone(phrase_bank.load_bank("Mavis", "a-different-hash"))

    def test_load_returns_none_for_missing_entry(self):
        self.assertIsNone(phrase_bank.load_bank("Never Saved", "whatever"))

    def test_save_skips_empty_bank(self):
        self.assertIsNone(phrase_bank.save_bank("Mavis", {}, "hash123"))

    def test_load_returns_none_for_malformed_json(self):
        phrase_bank.PHRASE_BANK_DIR.mkdir(parents=True, exist_ok=True)
        (phrase_bank.PHRASE_BANK_DIR / f"{phrase_bank.slugify('Mavis')}.json").write_text("not json", encoding="utf-8")
        self.assertIsNone(phrase_bank.load_bank("Mavis", "hash123"))


class TestFlattenForSelector(unittest.TestCase):
    def test_flattens_all_variants_across_bullets(self):
        bank = {"bullet a": ["a1", "a2"], "bullet b": ["b1"]}
        self.assertEqual(sorted(phrase_bank.flatten_for_selector(bank)), ["a1", "a2", "b1"])

    def test_none_bank_flattens_to_empty_list(self):
        self.assertEqual(phrase_bank.flatten_for_selector(None), [])


ITEM = {
    "name": "National Tire and Battery / Mavis",
    "role_title": "Service Technician",
    "responsibilities": [
        "Diagnosed and corrected vehicle alignment issues using specialized equipment.",
        "Operated point-of-sale systems to process customer transactions.",
    ],
}


def _mock_client(responses):
    """responses: list of raw chat() return values, consumed in order,
    repeating the last one if more calls happen than responses provided."""
    client = MagicMock()
    calls = {"i": 0}

    def _chat(*args, **kwargs):
        i = min(calls["i"], len(responses) - 1)
        calls["i"] += 1
        return responses[i]

    client.chat.side_effect = _chat
    return client


class TestBuildPhraseBank(unittest.TestCase):
    def test_no_source_facts_makes_zero_llm_calls(self):
        client = _mock_client([])
        result = local_tailor.build_phrase_bank({"name": "Empty Entry"}, client)
        self.assertEqual(result, {})
        client.chat.assert_not_called()

    def test_happy_path_produces_bank_keyed_by_original_bullet(self):
        response = json.dumps(
            {
                "variants": [
                    "Identified and resolved vehicle alignment problems using specialized equipment.",
                    "Used specialized equipment to diagnose and fix alignment faults on vehicles.",
                ]
            }
        )
        client = _mock_client([response])
        with patch("applypilot.scoring.semantic_match.embed_texts", return_value=[[1.0, 0.0], [0.0, 1.0]]):
            with patch("applypilot.scoring.semantic_match.select_diverse_indices", return_value=[0, 1]):
                result = local_tailor.build_phrase_bank(ITEM, client, target_per_bullet=2, max_rounds=1)
        self.assertIn("Diagnosed and corrected vehicle alignment issues using specialized equipment.", result)
        self.assertEqual(len(result["Diagnosed and corrected vehicle alignment issues using specialized equipment."]), 2)

    def test_fabrication_violation_is_dropped_not_shipped(self):
        """A variant that overclaims (e.g. invents team-lead authority the
        evidence doesn't support) must never survive into the bank, even
        if the model returns it."""
        response = json.dumps(
            {"variants": ["Led a team that architected a new vehicle diagnostic platform from the ground up."]}
        )
        client = _mock_client([response])
        with patch("applypilot.scoring.semantic_match.embed_texts", return_value=[[1.0]]):
            with patch("applypilot.scoring.semantic_match.select_diverse_indices", return_value=[0]):
                result = local_tailor.build_phrase_bank(ITEM, client, target_per_bullet=2, max_rounds=1)
        self.assertNotIn("Diagnosed and corrected vehicle alignment issues using specialized equipment.", result)

    def test_llm_call_failure_degrades_to_no_bank_for_that_bullet_not_a_crash(self):
        client = MagicMock()
        client.chat.side_effect = TimeoutError("local model timed out")
        result = local_tailor.build_phrase_bank(ITEM, client, target_per_bullet=2, max_rounds=1)
        self.assertEqual(result, {})

    def test_malformed_json_response_degrades_to_no_bank_for_that_bullet(self):
        client = _mock_client(["not json at all"])
        result = local_tailor.build_phrase_bank(ITEM, client, target_per_bullet=2, max_rounds=1)
        self.assertEqual(result, {})

    def test_zero_survivors_for_a_bullet_means_absent_not_a_placeholder(self):
        """Every candidate fails fabrication checks -- that bullet should
        simply not appear in the result, never a fabricated placeholder
        entry."""
        response = json.dumps({"variants": ["Architected and led the enterprise-wide vehicle diagnostics rollout."]})
        client = _mock_client([response])
        result = local_tailor.build_phrase_bank(ITEM, client, target_per_bullet=2, max_rounds=1)
        for bullet in phrase_bank.source_facts(ITEM):
            self.assertNotIn(bullet, result)


if __name__ == "__main__":
    unittest.main()
