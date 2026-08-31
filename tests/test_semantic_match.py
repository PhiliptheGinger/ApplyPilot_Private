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
