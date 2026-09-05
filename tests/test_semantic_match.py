"""Tests for applypilot.scoring.semantic_match -- the embedding-based
candidate-recall layer (2026-08-31).

No real Ollama call is required for these -- httpx is mocked throughout,
matching the existing convention in test_local_llm.py. See
TestRealOllamaIntegration at the bottom for the one opt-in, real-Ollama
test (skipped automatically when nothing is listening on :11434).
"""

from __future__ import annotations

import socket
import unittest
from unittest.mock import MagicMock, patch

import pytest

from applypilot.scoring import semantic_match


def _local_ollama_reachable() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.5):
            return True
    except OSError:
        return False


class TestConfig(unittest.TestCase):
    def test_enabled_by_default(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("APPLYPILOT_SEMANTIC_MATCH", None)
            self.assertTrue(semantic_match.is_semantic_match_enabled())

    def test_disabled_via_env_var(self):
        with patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH": "0"}):
            self.assertFalse(semantic_match.is_semantic_match_enabled())
        with patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH": "false"}):
            self.assertFalse(semantic_match.is_semantic_match_enabled())

    def test_default_threshold_is_point_three(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("APPLYPILOT_SEMANTIC_MATCH_THRESHOLD", None)
            self.assertEqual(semantic_match.semantic_match_threshold(), 0.30)

    def test_threshold_configurable_via_env_var(self):
        with patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH_THRESHOLD": "0.45"}):
            self.assertEqual(semantic_match.semantic_match_threshold(), 0.45)

    def test_malformed_threshold_env_var_falls_back_to_default(self):
        with patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH_THRESHOLD": "not-a-number"}):
            self.assertEqual(semantic_match.semantic_match_threshold(), 0.30)

    def test_ollama_url_strips_trailing_v1(self):
        with patch.dict("os.environ", {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434/v1"}):
            self.assertEqual(semantic_match._ollama_url(), "http://localhost:11434")

    def test_ollama_url_bare_root_unaffected(self):
        with patch.dict("os.environ", {"APPLYPILOT_LOCAL_LLM_URL": "http://localhost:11434"}):
            self.assertEqual(semantic_match._ollama_url(), "http://localhost:11434")

    def test_ollama_url_defaults_when_unset(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("APPLYPILOT_LOCAL_LLM_URL", None)
            self.assertEqual(semantic_match._ollama_url(), "http://localhost:11434")

    def test_embed_model_default_is_all_minilm(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("APPLYPILOT_SEMANTIC_MODEL", None)
            self.assertEqual(semantic_match._embed_model(), "all-minilm")


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(semantic_match.cosine_similarity(v, v), 1.0, places=6)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(semantic_match.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_empty_vectors_score_zero_not_raise(self):
        self.assertEqual(semantic_match.cosine_similarity([], []), 0.0)
        self.assertEqual(semantic_match.cosine_similarity([1.0], []), 0.0)

    def test_mismatched_length_scores_zero_not_raise(self):
        self.assertEqual(semantic_match.cosine_similarity([1.0, 2.0], [1.0]), 0.0)

    def test_zero_vector_scores_zero_not_divide_by_zero(self):
        self.assertEqual(semantic_match.cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)


class TestRankSemanticCandidates(unittest.TestCase):
    """Candidates ranked by cosine similarity, admission-threshold filtered."""

    def test_ranked_by_score_descending(self):
        req = [1.0, 0.0]
        corpus = [[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]]  # scores: 0.0, 1.0, ~0.707
        ranked = semantic_match.rank_semantic_candidates(req, corpus, threshold=0.0)
        indices = [i for i, _ in ranked]
        self.assertEqual(indices, [1, 2, 0])

    def test_below_threshold_excluded(self):
        req = [1.0, 0.0]
        corpus = [[0.0, 1.0], [1.0, 0.0]]  # scores: 0.0, 1.0
        ranked = semantic_match.rank_semantic_candidates(req, corpus, threshold=0.5)
        self.assertEqual([i for i, _ in ranked], [1])

    def test_empty_corpus_returns_empty(self):
        self.assertEqual(semantic_match.rank_semantic_candidates([1.0, 0.0], [], threshold=0.0), [])

    def test_default_threshold_used_when_not_given(self):
        with patch.dict("os.environ", {"APPLYPILOT_SEMANTIC_MATCH_THRESHOLD": "0.9"}):
            ranked = semantic_match.rank_semantic_candidates([1.0, 0.0], [[0.7, 0.7]])
            self.assertEqual(ranked, [])  # ~0.707 < 0.9


class TestEmbedTextsDegradesSafely(unittest.TestCase):
    """Ollama failures must never raise -- always None, so the caller falls
    back to literal-only behavior."""

    def test_empty_input_returns_empty_list_not_none(self):
        self.assertEqual(semantic_match.embed_texts([]), [])

    def test_connection_error_returns_none(self):
        with patch("httpx.post", side_effect=ConnectionError("refused")):
            self.assertIsNone(semantic_match.embed_texts(["hello"]))

    def test_timeout_returns_none(self):
        import httpx

        with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
            self.assertIsNone(semantic_match.embed_texts(["hello"]))

    def test_malformed_response_missing_embeddings_key_returns_none(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"model": "all-minilm"}  # no "embeddings"
        with patch("httpx.post", return_value=resp):
            self.assertIsNone(semantic_match.embed_texts(["hello"]))

    def test_mismatched_embedding_count_returns_none(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"embeddings": [[0.1, 0.2]]}  # 1 embedding for 2 inputs
        with patch("httpx.post", return_value=resp):
            self.assertIsNone(semantic_match.embed_texts(["hello", "world"]))

    def test_http_error_status_returns_none(self):
        import httpx

        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())
        with patch("httpx.post", return_value=resp):
            self.assertIsNone(semantic_match.embed_texts(["hello"]))

    def test_successful_response_returns_embeddings(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        with patch("httpx.post", return_value=resp):
            result = semantic_match.embed_texts(["a", "b"])
        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])


class TestCenterEmbeddings(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(semantic_match.center_embeddings([]), [])

    def test_mean_of_centered_output_is_zero(self):
        embeddings = [[1.0, 4.0], [3.0, 6.0], [5.0, 8.0]]
        centered = semantic_match.center_embeddings(embeddings)
        for d in range(2):
            self.assertAlmostEqual(sum(e[d] for e in centered) / len(centered), 0.0, places=9)

    def test_single_embedding_centers_to_zero_vector(self):
        centered = semantic_match.center_embeddings([[1.0, 2.0, 3.0]])
        self.assertEqual(centered, [[0.0, 0.0, 0.0]])

    def test_can_widen_similarity_gap_between_shared_direction_and_real_difference(self):
        """Three vectors sharing a big common component (simulating shared
        domain vocabulary) plus a small distinguishing offset -- centering
        should reduce their inflated raw cosine similarity."""
        shared = [10.0, 10.0, 10.0]
        a = [shared[0] + 1.0, shared[1], shared[2]]
        b = [shared[0], shared[1] + 1.0, shared[2]]
        raw_sim = semantic_match.cosine_similarity(a, b)
        centered = semantic_match.center_embeddings([a, b])
        centered_sim = semantic_match.cosine_similarity(centered[0], centered[1])
        self.assertLess(centered_sim, raw_sim)


class TestDiversityThreshold(unittest.TestCase):
    def test_default_is_point_nine(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("APPLYPILOT_DIVERSITY_THRESHOLD", None)
            self.assertEqual(semantic_match.diversity_threshold(), 0.90)

    def test_configurable_via_env_var(self):
        with patch.dict("os.environ", {"APPLYPILOT_DIVERSITY_THRESHOLD": "0.8"}):
            self.assertEqual(semantic_match.diversity_threshold(), 0.8)

    def test_malformed_env_var_falls_back_to_default(self):
        with patch.dict("os.environ", {"APPLYPILOT_DIVERSITY_THRESHOLD": "nope"}):
            self.assertEqual(semantic_match.diversity_threshold(), 0.90)


class TestSelectDiverseIndices(unittest.TestCase):
    """Greedy near-duplicate rejection over precomputed embeddings."""

    def test_first_item_always_kept(self):
        embeddings = [[1.0, 0.0]]
        self.assertEqual(semantic_match.select_diverse_indices(embeddings, threshold=0.9), [0])

    def test_empty_input_returns_empty(self):
        self.assertEqual(semantic_match.select_diverse_indices([], threshold=0.9), [])

    def test_near_duplicate_rejected(self):
        embeddings = [[1.0, 0.0], [0.99, 0.01]]  # ~cosine 0.9999
        self.assertEqual(semantic_match.select_diverse_indices(embeddings, threshold=0.9), [0])

    def test_distinct_vectors_both_kept(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0]]  # orthogonal, cosine 0.0
        self.assertEqual(semantic_match.select_diverse_indices(embeddings, threshold=0.9), [0, 1])

    def test_kept_must_be_dissimilar_to_all_prior_kept_not_just_immediate_predecessor(self):
        # b is far from a, c is close to a but far from b -- c must still
        # be rejected because it collides with a, even though it's the
        # second comparison, not the first.
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        c = [0.95, 0.0, 0.05]
        kept = semantic_match.select_diverse_indices([a, b, c], threshold=0.9)
        self.assertEqual(kept, [0, 1])

    def test_input_order_determines_which_of_a_colliding_pair_survives(self):
        near_dupes = [[1.0, 0.0], [0.99, 0.01]]
        self.assertEqual(semantic_match.select_diverse_indices(near_dupes, threshold=0.9), [0])
        self.assertEqual(semantic_match.select_diverse_indices(list(reversed(near_dupes)), threshold=0.9), [0])

    def test_default_threshold_used_when_not_given(self):
        with patch.dict("os.environ", {"APPLYPILOT_DIVERSITY_THRESHOLD": "0.5"}):
            # cosine([1,1],[1,0.9]) ~= 0.995 -- collides at threshold 0.5
            kept = semantic_match.select_diverse_indices([[1.0, 1.0], [1.0, 0.9]])
        self.assertEqual(kept, [0])

    def test_real_pilot_data_reproduces_documented_threshold_finding(self):
        """Regression pin for the exact empirical numbers cited in
        diversity_threshold()'s docstring, using real all-minilm
        embeddings would require a live Ollama call -- this instead pins
        the documented cosine scores directly (0.977/0.930 confirmed
        duplicates cleared 0.90; 0.851/0.840 confirmed template-duplicates
        did NOT), so a future change to the default can't silently
        invalidate the documented justification without a test noticing."""
        threshold = semantic_match.diversity_threshold()
        self.assertLess(0.851, threshold)
        self.assertLess(0.840, threshold)
        self.assertGreaterEqual(0.930, threshold)
        self.assertGreaterEqual(0.977, threshold)


class TestCrossEncoderTier(unittest.TestCase):
    """2026-09-04 bake-off winner -- optional, gated behind the
    `cross_encoder` extra (sentence-transformers not required for these
    tests; the loader is exercised via sys.modules mocking / direct
    patching of _get_cross_encoder, never a real import).

    Global singleton state (_cross_encoder_instance/_cross_encoder_load_
    failed) is reset before and after every test here so one test's
    simulated load failure/success can't leak into another -- same class
    of bug as the LLM-exhaustion-state file leak found and fixed earlier
    this session, guarded against explicitly this time."""

    def setUp(self):
        semantic_match._cross_encoder_instance = None
        semantic_match._cross_encoder_load_failed = False

    def tearDown(self):
        semantic_match._cross_encoder_instance = None
        semantic_match._cross_encoder_load_failed = False

    def test_get_cross_encoder_returns_none_when_import_fails(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            result = semantic_match._get_cross_encoder()
        self.assertIsNone(result)
        self.assertTrue(semantic_match._cross_encoder_load_failed)

    def test_get_cross_encoder_caches_failure_does_not_retry(self):
        semantic_match._cross_encoder_load_failed = True
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            result = semantic_match._get_cross_encoder()
        self.assertIsNone(result)
        mock_module.CrossEncoder.assert_not_called()

    def test_get_cross_encoder_caches_success_loads_once(self):
        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_module.CrossEncoder = MagicMock(return_value=mock_model)
        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            first = semantic_match._get_cross_encoder()
            second = semantic_match._get_cross_encoder()
        self.assertIs(first, mock_model)
        self.assertIs(second, mock_model)
        mock_module.CrossEncoder.assert_called_once()

    def test_cross_encoder_score_returns_none_when_unavailable(self):
        with patch.object(semantic_match, "_get_cross_encoder", return_value=None):
            self.assertIsNone(semantic_match.cross_encoder_score("a", "b"))

    def test_cross_encoder_score_returns_float_from_model(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.83]
        with patch.object(semantic_match, "_get_cross_encoder", return_value=mock_model):
            score = semantic_match.cross_encoder_score("a", "b")
        self.assertEqual(score, 0.83)
        mock_model.predict.assert_called_once_with([("a", "b")])

    def test_cross_encoder_score_inference_failure_returns_none(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("boom")
        with patch.object(semantic_match, "_get_cross_encoder", return_value=mock_model):
            self.assertIsNone(semantic_match.cross_encoder_score("a", "b"))


class TestIsDuplicatePair(unittest.TestCase):
    """Vectors below use a=[1.0, 0.0], b=[0.75, 0.661437...] -- since a is
    already unit-length and sin^2+cos^2=1, cosine_similarity(a, b) works
    out to exactly 0.75, cleanly inside the (0.60, 0.90) escalation band
    without relying on float-comparison luck."""

    _A = [1.0, 0.0]
    _B_IN_BAND = [0.75, 0.6614378277661477]

    def test_above_cosine_threshold_is_duplicate_without_cross_encoder_call(self):
        with patch.object(semantic_match, "cross_encoder_score") as mock_ce:
            result = semantic_match.is_duplicate_pair("a", "b", [1.0, 0.0], [0.99, 0.01], cosine_threshold=0.9)
        self.assertTrue(result)
        mock_ce.assert_not_called()

    def test_well_below_escalation_band_is_not_duplicate_without_cross_encoder_call(self):
        with patch.object(semantic_match, "cross_encoder_score") as mock_ce:
            result = semantic_match.is_duplicate_pair("a", "b", [1.0, 0.0], [0.0, 1.0], cosine_threshold=0.9)
        self.assertFalse(result)
        mock_ce.assert_not_called()

    def test_escalation_band_pair_is_confirmed_in_band(self):
        cos = semantic_match.cosine_similarity(self._A, self._B_IN_BAND)
        self.assertAlmostEqual(cos, 0.75, places=6)

    def test_escalation_band_confirmed_duplicate_by_cross_encoder(self):
        with patch.object(semantic_match, "cross_encoder_score", return_value=0.9) as mock_ce:
            result = semantic_match.is_duplicate_pair(
                "text a", "text b", self._A, self._B_IN_BAND, cosine_threshold=0.9
            )
        self.assertTrue(result)
        mock_ce.assert_called_once_with("text a", "text b")

    def test_escalation_band_denied_by_cross_encoder(self):
        with patch.object(semantic_match, "cross_encoder_score", return_value=0.2):
            result = semantic_match.is_duplicate_pair(
                "text a", "text b", self._A, self._B_IN_BAND, cosine_threshold=0.9
            )
        self.assertFalse(result)

    def test_escalation_band_falls_back_to_not_duplicate_when_cross_encoder_unavailable(self):
        with patch.object(semantic_match, "cross_encoder_score", return_value=None):
            result = semantic_match.is_duplicate_pair(
                "text a", "text b", self._A, self._B_IN_BAND, cosine_threshold=0.9
            )
        self.assertFalse(result)


class TestSelectDiverseIndicesVerified(unittest.TestCase):
    def test_identical_to_plain_version_when_cross_encoder_unavailable(self):
        texts = ["a", "b", "c"]
        embeddings = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
        with patch.object(semantic_match, "cross_encoder_score", return_value=None):
            verified = semantic_match.select_diverse_indices_verified(texts, embeddings, cosine_threshold=0.9)
        plain = semantic_match.select_diverse_indices(embeddings, threshold=0.9)
        self.assertEqual(verified, plain)

    def test_escalation_catches_a_pair_plain_cosine_would_keep(self):
        a = [1.0, 0.0]
        b = [0.75, 0.6614378277661477]  # cosine(a, b) == 0.75, inside the escalation band
        texts = ["sentence one", "sentence two"]
        embeddings = [a, b]

        plain = semantic_match.select_diverse_indices(embeddings, threshold=0.9)
        self.assertEqual(plain, [0, 1], "plain cosine at 0.9 keeps both -- the gap this tier fixes")

        with patch.object(semantic_match, "cross_encoder_score", return_value=0.9):
            verified = semantic_match.select_diverse_indices_verified(texts, embeddings, cosine_threshold=0.9)
        self.assertEqual(verified, [0], "cross-encoder escalation correctly rejects the duplicate")


@pytest.mark.skipif(not _local_ollama_reachable(), reason="No local Ollama instance reachable at 127.0.0.1:11434")
class TestRealOllamaIntegration(unittest.TestCase):
    """Live regression test against a real, running Ollama instance with
    all-minilm. Skipped automatically when no local Ollama is reachable
    (e.g. in CI) -- not required for ordinary pytest execution."""

    def test_embed_and_similarity_against_real_ollama(self):
        embeddings = semantic_match.embed_texts(["desktop hardware support", "website development project"])
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(embeddings), 2)
        self.assertGreater(len(embeddings[0]), 0)
        score = semantic_match.cosine_similarity(embeddings[0], embeddings[1])
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
