"""Semantic (embedding-based) candidate-recall layer for local tailoring
evidence matching.

Talks directly to Ollama's native embeddings endpoint (POST /api/embed)
with the already-installed `all-minilm` model. No LLMClient coupling, and
no REQUIRED new Python dependency -- pure Python cosine similarity,
matching the same native-Ollama-transport pattern local_tailor.py already
uses for get_local_tailoring_plan()'s /api/chat calls.

2026-09-04: an OPTIONAL cross-encoder tier (cross_encoder_score /
resolve_pair below) was added on top of this, gated behind the
`cross_encoder` extra (`pip install -e ".[cross_encoder]"`, pulls in
sentence-transformers + torch, ~865MB on disk). It degrades to None
whenever that extra isn't installed, via the exact same try/except-and-
return-None discipline embed_texts already uses for a down Ollama --
nothing in this module requires it.

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

Revisited 2026-09-03 after a resume-tailoring RAG paper claimed ~7.8
fit-score-point improvement from a vector-DB "career vault" retrieval
approach. Declined to pursue further: the finding above is a SIGNAL-QUALITY
ceiling (overlapping true/false-positive score ranges; LLM arbitration
giving opposite wrong answers depending on model size), not a scale or
compute limitation -- a vector database, a bigger embedding model, or more
storage would not address a precision problem already measured on tiny
(dozens-of-strings) candidate sets. This module already is the lightest
practical version of "semantic retrieval" (local CPU embeddings, no vector
index, nothing persisted); the gap is in what similarity means here, not in
infrastructure.
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


def center_embeddings(embeddings: list[list[float]]) -> list[list[float]]:
    """Subtract the batch mean vector from every embedding -- a cheap,
    corpus-free partial mitigation for embedding anisotropy (2026-09-04
    bake-off, data/experiments/deterministic_slotfiller_20260902/): all-
    minilm sentence embeddings (like most transformer-based embeddings)
    don't spread evenly through the vector space -- they collapse toward a
    narrow cone, which inflates cosine similarity between EVERY pair
    (confirmed empirically: 36 real pairwise scores from one bake-off run
    all landed in a 0.744-0.889 band with no gap, including pairs a human
    reader judged as genuinely different sentences). Removing the shared
    mean direction (the dominant, most anisotropic component) before
    computing cosine similarity is the cheap version of published de-
    anisotropy techniques (e.g. BERT-flow / whitening-BERT), which also
    remove leading PCA components using a large reference corpus -- this
    function deliberately does NOT do that; it only centers within
    whatever batch of embeddings is passed in, so it needs no reference
    corpus to build or keep in sync, at the cost of being a weaker
    correction than full whitening. cosine_similarity re-normalizes
    internally, so the returned vectors don't need to be unit-length.

    NOT applied by any caller by default -- this is a bake-off candidate,
    not yet adopted into select_diverse_indices/diversity_threshold, which
    still operate on raw embeddings from embed_texts. See the bake-off
    results for whether it's actually worth wiring in."""
    if not embeddings:
        return []
    dim = len(embeddings[0])
    mean = [sum(e[d] for e in embeddings) / len(embeddings) for d in range(dim)]
    return [[e[d] - mean[d] for d in range(dim)] for e in embeddings]


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


_DEFAULT_DIVERSITY_THRESHOLD = 0.90


def diversity_threshold() -> float:
    """Near-duplicate REJECTION threshold for select_diverse_indices --
    deliberately a different, higher number than semantic_match_threshold()
    (an admission floor for a different task: is this candidate relevant
    at all, vs. is this candidate too similar to one we already kept).
    Configurable via APPLYPILOT_DIVERSITY_THRESHOLD.

    0.90 default is empirically grounded, not guessed: measured against a
    real local-model (qwen3:1.7b) sentence-pool run (2026-09-03/04,
    inventory_expansion_pilot.py / data/experiments/
    deterministic_slotfiller_20260902/inventory_expansion_pilot_results.json)
    where 12 fabrication-check survivors for one experience_inventory entry
    were, on inspection, largely the same handful of sentences with words
    swapped for near-synonyms ("customer specifications and company
    protocols" / "customer requirements and company procedures" / ...).
    Pairwise cosine similarity on all-minilm embeddings of that real data:
    genuinely near-verbatim restatements scored 0.93-0.98; a clearly
    distinct sentence (naming a specific retailer, "Handled the
    installation of home appliances sourced from Lowe's.") scored as low
    as 0.34-0.46 against the others. 0.90 sits just below the confirmed-
    duplicate cluster without touching that low end.

    IMPORTANT DOCUMENTED LIMITATION, same discipline as this module's
    other findings: this threshold does NOT reliably catch every
    near-duplicate. The same real dataset also contained sentences built
    from an identical template with synonym-substituted slots (e.g.
    "Carried out hands-on installation tasks in accordance with customer
    specifications and company protocols." vs. "Adhered to customer
    specifications and company protocols during installation tasks.")
    that a human reads as the same sentence restated, yet scored only
    ~0.82-0.85 cosine -- below this threshold, because mean-pooled
    sentence embeddings blur which words play which grammatical role (the
    exact "look at the constituencies of the phrases" concern raised
    2026-09-03 night). Word/bigram-overlap Jaccard was tested as a second
    candidate signal on the same data and performed WORSE, not better --
    synonym substitution defeats surface lexical overlap even more than it
    dilutes an embedding average. No cheap deterministic fix for that
    class was found; it is a real, open gap, mitigated (not solved) by
    feeding already-accepted sentences back into regeneration prompts as
    an explicit "vary structure, not just wording" instruction, not by
    this threshold alone.
    """
    try:
        return float(os.environ.get("APPLYPILOT_DIVERSITY_THRESHOLD", str(_DEFAULT_DIVERSITY_THRESHOLD)))
    except ValueError:
        return _DEFAULT_DIVERSITY_THRESHOLD


def select_diverse_indices(
    embeddings: list[list[float]],
    threshold: float | None = None,
) -> list[int]:
    """Greedily select a diverse subset of `embeddings`, in input order.

    Keeps embeddings[0] unconditionally, then keeps each subsequent index
    only if its cosine similarity to EVERY already-kept embedding is below
    `threshold` (default: diversity_threshold()). Deterministic, order-
    dependent by design -- callers that want a specific sentence favored
    when it collides with another should put it first (e.g. keep prior-run
    accepted sentences ahead of a new generation batch).

    Pure function over precomputed embeddings (mirrors rank_semantic_
    candidates' separation from embed_texts) so it's trivially unit-
    testable without a real Ollama call and reusable regardless of which
    provider or model produced the original candidate text -- this
    function has no opinion on that, by design, since the near-duplicate
    problem this addresses was observed on local-model output but per
    2026-09-04 direction should guard cloud-model output identically for
    consistency, even though cloud hasn't (yet) been observed to need it.
    """
    if threshold is None:
        threshold = diversity_threshold()
    kept: list[int] = []
    for i, vec in enumerate(embeddings):
        if all(cosine_similarity(vec, embeddings[k]) < threshold for k in kept):
            kept.append(i)
    return kept


# ---------------------------------------------------------------------------
# Optional cross-encoder tier (2026-09-04 bake-off, data/experiments/
# deterministic_slotfiller_20260902/bakeoff_diversity_techniques_20260904.py)
# ---------------------------------------------------------------------------
#
# The bake-off scored 4 techniques against a 12-pair hand-labeled ground-
# truth set (same-claim / different-claim, drawn from real generated
# sentences + real profile.json text): raw cosine at the shipped 0.90
# threshold reached only 0.75 accuracy / 0.40 recall (it misses real
# duplicates sitting in a 0.82-0.89 "disputed zone" where paraphrased
# sentences share a template but swap synonyms); per-batch mean-centering
# (center_embeddings above) matched raw cosine's own best-fit threshold but
# didn't beat it; local-LLM-as-judge reached 0.917 accuracy but cost ~18s
# per pair with no fixed cost to amortize. cross-encoder/quora-distilroberta-
# base (trained on duplicate-question detection -- closest off-the-shelf
# analog to "same claim, different words") reached 1.00 accuracy at its OWN
# untuned natural threshold (0.5), with a wide, clean separation margin
# (SAME pairs scored 0.82-0.98, DIFFERENT pairs scored 0.0-0.42) -- not a
# threshold fitted to this small sample, unlike cosine's 0.84.
#
# Cost profile is why this is a two-tier ESCALATION, not a replacement for
# cosine: cross-encoder needs one forward pass per PAIR (no cheap batch
# embed-once-compare-many-times trick), plus a one-time model load that
# measured at 67s on this project's HDD-backed dev machine (torch + a
# ~318MB model reading from a spinning disk, worse than an SSD would do) --
# expensive to pay for every pair when cosine already resolves most of them
# for free. _CROSS_ENCODER_ESCALATION_BAND restricts cross-encoder calls to
# exactly the score range the bake-off showed cosine can't be trusted in;
# outside that band, cosine's own verdict stands unchanged from before this
# tier existed.
_CROSS_ENCODER_MODEL_NAME = "cross-encoder/quora-distilroberta-base"
_CROSS_ENCODER_DUPLICATE_THRESHOLD = 0.5
_CROSS_ENCODER_ESCALATION_BAND = (0.60, 0.90)

_cross_encoder_instance = None
_cross_encoder_load_failed = False


def _get_cross_encoder():
    """Lazy singleton -- loaded at most once per process, then reused for
    every subsequent call. Deliberately module-level (not per-call) so the
    ~865MB-on-disk / ~67s-on-a-slow-HDD load cost is paid once per long-
    running pipeline process (e.g. one `applypilot run tailor` invocation
    handling many jobs), not once per job or once per pair -- see this
    section's module comment for why that cost matters here specifically.
    Returns None permanently after the first failed attempt (e.g. the
    `cross_encoder` extra isn't installed) rather than retrying on every
    call, matching embed_texts' best-effort/never-raises contract."""
    global _cross_encoder_instance, _cross_encoder_load_failed
    if _cross_encoder_load_failed:
        return None
    if _cross_encoder_instance is None:
        try:
            from sentence_transformers import CrossEncoder

            _cross_encoder_instance = CrossEncoder(_CROSS_ENCODER_MODEL_NAME)
        except Exception as exc:  # noqa: BLE001 -- optional dependency; any failure must degrade to "unavailable," never break a caller that doesn't have the extra installed
            log.warning(
                "semantic_match: cross-encoder unavailable (%s: %s) -- "
                "install the 'cross_encoder' extra to enable it. Falling back "
                "to cosine-only behavior.",
                type(exc).__name__,
                exc,
            )
            _cross_encoder_load_failed = True
            return None
    return _cross_encoder_instance


def cross_encoder_score(text_a: str, text_b: str) -> float | None:
    """Score how likely `text_a`/`text_b` restate the same underlying claim,
    via the cross-encoder (0.0-1.0, model's own calibrated scale -- 0.5 is
    its natural decision boundary, not a value tuned to any specific
    dataset). Returns None if the model isn't available (extra not
    installed) or inference fails -- never raises, mirrors embed_texts."""
    model = _get_cross_encoder()
    if model is None:
        return None
    try:
        return float(model.predict([(text_a, text_b)])[0])
    except Exception as exc:  # noqa: BLE001 -- must degrade to "unavailable," not break the caller
        log.warning("semantic_match: cross-encoder inference failed (%s: %s)", type(exc).__name__, exc)
        return None


def is_duplicate_pair(
    text_a: str,
    text_b: str,
    embedding_a: list[float],
    embedding_b: list[float],
    cosine_threshold: float | None = None,
) -> bool:
    """Two-tier duplicate check: cheap cosine first; only escalates to the
    (expensive, optional) cross-encoder when cosine's score falls inside
    _CROSS_ENCODER_ESCALATION_BAND, the exact range the 2026-09-04 bake-off
    found cosine alone gets wrong. Outside that band, or whenever the
    cross-encoder is unavailable, behavior is IDENTICAL to plain cosine-
    threshold checking -- this function is a strict enhancement, never a
    behavior change, when the optional tier can't run."""
    if cosine_threshold is None:
        cosine_threshold = diversity_threshold()
    cos = cosine_similarity(embedding_a, embedding_b)
    if cos >= cosine_threshold:
        return True
    lo, hi = _CROSS_ENCODER_ESCALATION_BAND
    if lo <= cos < hi:
        score = cross_encoder_score(text_a, text_b)
        if score is not None:
            return score >= _CROSS_ENCODER_DUPLICATE_THRESHOLD
    return False


def select_diverse_indices_verified(
    texts: list[str],
    embeddings: list[list[float]],
    cosine_threshold: float | None = None,
) -> list[int]:
    """Same greedy, order-dependent selection as select_diverse_indices,
    but using is_duplicate_pair's two-tier check instead of a raw cosine
    threshold -- catches the disputed-zone duplicates plain cosine misses,
    at the cost of a cross-encoder call for each pair that lands in the
    escalation band. Falls back to being behaviorally identical to
    select_diverse_indices whenever the cross-encoder extra isn't
    installed (is_duplicate_pair's own fallback)."""
    kept: list[int] = []
    for i in range(len(texts)):
        if all(
            not is_duplicate_pair(texts[i], texts[k], embeddings[i], embeddings[k], cosine_threshold)
            for k in kept
        ):
            kept.append(i)
    return kept
