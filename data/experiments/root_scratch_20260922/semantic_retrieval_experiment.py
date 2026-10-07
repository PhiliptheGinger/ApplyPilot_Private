"""THROWAWAY experiment script -- semantic retrieval vs. literal retrieval.

Not part of the production pipeline. Does not modify any production code,
pyproject.toml, tests, or the database (opened read-only). Safe to delete
after review.

Reuses the real, unmodified deterministic retrieval functions from
applypilot.scoring.local_tailor (_split_requirement_lines,
rank_profile_evidence, _pair_candidate_evidence) so the "literal baseline"
shown here is byte-for-byte what debug-local-plan already produces.

Adds an experimental semantic-similarity ranking (Ollama's native
/api/embed with the all-minilm model -- no Qwen3, no LLMClient, no
sentence-transformers/Torch, no generative call of any kind) purely as an
observational signal. Never turns a similarity score into a
supported/unsupported verdict. No threshold is applied to the raw cosine
scores -- all are printed as-is for manual review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import numpy as np

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot.scoring.local_tailor import (  # noqa: E402
    _pair_candidate_evidence,
    _split_requirement_lines,
    rank_profile_evidence,
)

DB_PATH = Path.home() / ".applypilot" / "applypilot.db"
PROFILE_PATH = REPO_ROOT / "data" / "profile.json"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "all-minilm"

JOB_IDS = [19447, 18638, 23006, 20187, 18637]


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts via Ollama's native /api/embed endpoint."""
    resp = httpx.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["embeddings"], dtype=np.float32)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def _join(*parts) -> str:
    out: list[str] = []
    for p in parts:
        if isinstance(p, str) and p.strip():
            out.append(p.strip())
        elif isinstance(p, list):
            out.extend(x.strip() for x in p if isinstance(x, str) and x.strip())
    return " ".join(out)


def build_evidence_corpus(profile: dict) -> list[dict]:
    """Full 23-item evidence corpus (experience/project/skill/certification),
    respecting resume_allowed=False exclusion -- same rule rank_profile_evidence
    applies, but WITHOUT its literal-term-overlap filter, so items literal
    matching never even considers still get a chance to be found semantically."""
    corpus: list[dict] = []

    for item in profile.get("experience_inventory") or []:
        if not isinstance(item, dict) or item.get("resume_allowed") is False:
            continue
        text = _join(
            item.get("name"),
            item.get("role_title"),
            item.get("role_type"),
            item.get("description"),
            item.get("relevance_categories"),
        )
        corpus.append({"type": "experience", "name": item.get("name"), "text": text, "item": item})

    for item in profile.get("project_inventory") or []:
        if not isinstance(item, dict) or item.get("resume_allowed") is False:
            continue
        text = _join(
            item.get("name"),
            item.get("description"),
            item.get("factual_concepts"),
            item.get("relevance_categories"),
        )
        corpus.append({"type": "project", "name": item.get("name"), "text": text, "item": item})

    for item in profile.get("skills_inventory") or []:
        if not isinstance(item, dict) or item.get("resume_allowed") is False:
            continue
        text = _join(
            item.get("name"),
            item.get("evidence_level"),
            item.get("proficiency"),
            item.get("relevance_categories"),
        )
        corpus.append({"type": "skill", "name": item.get("name"), "text": text, "item": item})

    for item in profile.get("certifications") or []:
        if not isinstance(item, dict) or item.get("resume_allowed") is False:
            continue
        text = _join(
            item.get("name"),
            item.get("official_credential_description"),
            item.get("relevance_categories"),
        )
        corpus.append({"type": "certification", "name": item.get("name"), "text": text, "item": item})

    return corpus


def literal_names_for(pair_indices: list[int], ranked_evidence: list[dict]) -> set[str]:
    return {f"{ranked_evidence[i - 1]['type']}:{ranked_evidence[i - 1]['name']}" for i in pair_indices}


def literal_desc_for(pair_indices: list[int], ranked_evidence: list[dict]) -> str:
    if not pair_indices:
        return "(none -- unsupported by literal matching)"
    names = [f"{ranked_evidence[i - 1]['type']}:{ranked_evidence[i - 1]['name']}" for i in pair_indices]
    return ", ".join(names)


def evidence_snippet(item: dict, kind: str) -> str:
    """Enough of the item's real text to manually judge the match."""
    if kind == "experience":
        return (item.get("description") or "")[:220]
    if kind == "project":
        concepts = ", ".join(c for c in (item.get("factual_concepts") or []) if isinstance(c, str))
        return concepts[:220]
    if kind == "skill":
        cats = ", ".join(c for c in (item.get("relevance_categories") or []) if isinstance(c, str))
        return f"{item.get('evidence_level', '')} / {item.get('proficiency', '')} / {cats}"[:220]
    if kind == "certification":
        return (item.get("official_credential_description") or "")[:220]
    return ""


def main() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    import sqlite3

    db_uri = DB_PATH.as_uri() + "?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row

    corpus = build_evidence_corpus(profile)
    print(f"Evidence corpus items embedded: {len(corpus)}")
    corpus_embeddings = embed([c["text"] for c in corpus])
    print(f"Evidence embedding matrix shape: {corpus_embeddings.shape}")
    print()

    total_req_lines_embedded = 0
    per_job_summary = []

    for job_id in JOB_IDS:
        row = conn.execute("SELECT rowid, title, full_description FROM jobs WHERE rowid = ?", (job_id,)).fetchone()
        if row is None:
            print(f"!! Job {job_id} not found in DB -- skipping")
            continue

        job = {"title": row["title"], "full_description": row["full_description"]}
        print("=" * 100)
        print(f"JOB {job_id}: {row['title']}")
        print("=" * 100)

        requirement_lines, dropped_benefits = _split_requirement_lines(job.get("full_description") or "")
        ranked_evidence = rank_profile_evidence(job, profile)

        print(f"Requirement lines extracted: {len(requirement_lines)}  (dropped benefit lines: {len(dropped_benefits)})")
        print(f"Literal whole-job ranked evidence (top {len(ranked_evidence)}): "
              + ", ".join(f"{e['type']}:{e['name']}(score={e['score']})" for e in ranked_evidence))
        print()

        if not requirement_lines:
            per_job_summary.append(
                {"job_id": job_id, "title": row["title"], "n_requirements": 0, "n_semantic_only": 0, "strongest": None}
            )
            continue

        req_texts = [r["text"] for r in requirement_lines]
        req_embeddings = embed(req_texts)
        total_req_lines_embedded += len(req_texts)
        sims = cosine_sim_matrix(req_embeddings, corpus_embeddings)

        n_semantic_only = 0
        n_printed = 0
        strongest_semantic_only = None  # (score, corpus_entry, requirement_text)

        for i, r in enumerate(requirement_lines):
            req_text = r["text"]
            literal_indices = _pair_candidate_evidence(req_text, ranked_evidence)
            literal_set = literal_names_for(literal_indices, ranked_evidence)

            row_sims = sims[i]
            top3_idx = np.argsort(-row_sims)[:3]
            top3 = [(corpus[j], float(row_sims[j])) for j in top3_idx]

            top_semantic_name = f"{top3[0][0]['type']}:{top3[0][0]['name']}"
            materially_differs = (not literal_set) or (top_semantic_name not in literal_set)

            if not literal_set and top3:
                n_semantic_only += 1
                best = top3[0]
                if strongest_semantic_only is None or best[1] > strongest_semantic_only[0]:
                    strongest_semantic_only = (best[1], best[0], req_text)

            if not materially_differs:
                continue

            n_printed += 1
            print(f"--- Requirement [{r['importance']}]: {req_text}")
            print(f"    Literal evidence: {literal_desc_for(literal_indices, ranked_evidence)}")
            print("    Top 3 semantic candidates:")
            for entry, score in top3:
                snippet = evidence_snippet(entry["item"], entry["type"])
                print(f"      [{score:.4f}] {entry['type']}:{entry['name']}  -- {snippet}")
            print()

        if n_printed == 0:
            print("(No requirement lines where semantic ranking materially differed from literal.)")
            print()

        per_job_summary.append(
            {
                "job_id": job_id,
                "title": row["title"],
                "n_requirements": len(requirement_lines),
                "n_semantic_only": n_semantic_only,
                "strongest": strongest_semantic_only,
            }
        )

    conn.close()

    print("=" * 100)
    print("SUMMARY (raw counts only -- no supported/unsupported judgment)")
    print("=" * 100)
    for s in per_job_summary:
        print(f"\nJob {s['job_id']} -- {s['title']}")
        print(f"  Requirements examined: {s['n_requirements']}")
        print(f"  Requirements with literal=NONE where semantic surfaced >=1 candidate: {s['n_semantic_only']}")
        if s["strongest"]:
            score, entry, req_text = s["strongest"]
            print(f"  Strongest semantic-only candidate: [{score:.4f}] {entry['type']}:{entry['name']}"
                  f" for requirement: \"{req_text[:100]}\"")
        else:
            print("  Strongest semantic-only candidate: (none)")

    print(f"\nTotal requirement lines embedded across all jobs: {total_req_lines_embedded}")
    print(f"Total evidence items embedded: {len(corpus)}")


if __name__ == "__main__":
    main()
