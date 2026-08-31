"""Semantic (embedding-based) candidate-recall layer for local tailoring
evidence matching.

Talks directly to Ollama's native embeddings endpoint (POST /api/embed)
with the already-installed `all-minilm` model. No LLMClient coupling, no
new Python dependency (no torch/sentence-transformers/scikit-learn) -- pure
Python cosine similarity, matching the same native-Ollama-transport
pattern local_tailor.py already uses for get_local_tailoring_plan()'s
/api/chat calls.

CRITICAL SAFETY CONTRACT: this module only ever RANKS candidates by
similarity. It never decides that evidence supports a requirement -- see
local_tailor.py's _auto_resolve_requirements, the only place a semantic
candidate can be added to a requirement's `candidates` pool, which
explicitly never lets a semantic-only candidate enter `resolved`.

Real-data validation (2026-08-31 semantic-retrieval experiment, 5 real
jobs / 40 requirement lines, then arbitration_test.py) found: (a) raw
cosine similarity does NOT reliably separate genuine matches from false
positives -- true-positive and false-positive score ranges overlap
substantially (genuine hits ~0.32+, obvious noise ~0.13-0.22, but some
false positives also reached 0.30-0.42); (b) the existing Qwen3 arbitration
prompt is NOT a reliable relevance classifier either -- qwen3:1.7b accepted
a known false positive alongside the correct answer, and qwen3:8b accepted
ONLY the false positive and rejected the correct answer. Neither similarity
score nor LLM arbitration can be trusted as a support decision on their
own. This module's admission threshold therefore exists to bound candidate
VOLUME only, never to establish correctness.
"""

from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "all-minilm"
_DEFAULT_THRESHOLD = 0.30


def is_semantic_match_enabled() -> bool:
    """Opt-out switch. Semantic retrieval only ever runs inside
    get_local_tailoring_plan(), which is itself already gated behind
    APPLYPILOT_LOCAL_PLAN/--local-first -- this is an independent switch
    to disable just the semantic layer (e.g. if all-minilm isn't pulled)
    without disabling local planning entirely. Default on."""
    return os.environ.get("APPLYPILOT_SEMANTIC_MATCH", "1").strip().lower() not in ("0", "false", "no")


def semantic_match_threshold() -> float:
    """Candidate-ADMISSION threshold -- NOT a support threshold. See the
    module docstring: this value exists purely to bound how many marginal
    candidates reach the (already-imperfect) arbitration step, not to
    prove relevance. Configurable via APPLYPILOT_SEMANTIC_MATCH_THRESHOLD
    for easy adjustment without a new configuration system."""
    try:
        return float(os.environ.get("APPLYPILOT_SEMANTIC_MATCH_THRESHOLD", str(_DEFAULT_THRESHOLD)))
    except ValueError:
        return _DEFAULT_THRESHOLD


def _ollama_url() -> str:
    """Reuses APPLYPILOT_LOCAL_LLM_URL (same Ollama server the generative
    local model already talks to) rather than a new env var. Strips a
    trailing /v1 the OpenAI-compatible sibling path requires but this
    native endpoint doesn't -- identical precedent to
    local_tailor._ollama_native_base_url."""
    raw = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "").strip()
    if not raw:
        return _DEFAULT_OLLAMA_URL
    url = raw.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def _embed_model() -> str:
    return os.environ.get("APPLYPILOT_SEMANTIC_MODEL", _DEFAULT_MODEL)


def embed_texts(texts: list[str], timeout: float = 30.0) -> list[list[float]] | None:
    """Embed a batch of texts via Ollama's native /api/embed.

    Returns None (never raises) on ANY failure -- connection refused,
    timeout, malformed response -- so a semantic-retrieval failure always
    degrades to "no additional candidates this run", never breaks the
    caller's existing literal-only behavior.
    """
    if not texts:
        return []
    try:
        resp = httpx.post(
            f"{_ollama_url()}/api/embed",
            json={"model": _embed_model(), "input": texts},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            log.warning(
                "semantic_match: malformed /api/embed response (expected %d embeddings, got %r)",
                len(texts),
                type(embeddings).__name__,
            )
            return None
        return embeddings
    except Exception as exc:  # noqa: BLE001 -- documented "never raises" contract; any embedding failure must degrade to "no semantic candidates", not break local tailoring
        log.warning(
            "semantic_match: embedding call failed (%s: %s) -- semantic retrieval skipped this run",
            type(exc).__name__,
            exc,
        )
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity -- no numpy dependency. Corpus sizes
    here are tiny (a few dozen evidence items / requirement lines), so
    performance is a non-issue."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_semantic_candidates(
    requirement_embedding: list[float],
    corpus_embeddings: list[list[float]],
    threshold: float | None = None,
) -> list[tuple[int, float]]:
    """Rank corpus items (0-based index into corpus_embeddings) by cosine
    similarity to one requirement embedding.

    Returns only items at or above `threshold` (default:
    semantic_match_threshold()), sorted by score descending. Pure ranking
    -- see module docstring for why the caller must never treat this as a
    support decision, only as an admission-only candidate list.
    """
    if threshold is None:
        threshold = semantic_match_threshold()
    scored = [(i, cosine_similarity(requirement_embedding, vec)) for i, vec in enumerate(corpus_embeddings)]
    scored = [(i, s) for i, s in scored if s >= threshold]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
