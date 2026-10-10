"""Per-job document storage + cross-site exact-duplicate dedup (FW50).

Real-world motivation: applications on different ATSes/employers sometimes
attach the SAME generic document (e.g. a state-mandated hiring-law notice)
verbatim. Saving N identical copies wastes disk and makes a job's own
folder noisy with boilerplate unrelated to that specific application.

v1 scope, per the explicit 2026-10-09 design call ("exact byte hash only"):
dedup is SHA-256 of the raw file bytes, nothing fuzzier. This has zero
false-positive risk (two files with the same hash ARE the same bytes) but
will NOT catch a document re-rendered slightly differently per employer
(different header/footer, different generation timestamp baked into the
PDF) -- that's a safe miss (an extra stored copy), not a safe-to-ignore
risk, consistent with this project's standing preference for erring
toward not losing/misclassifying real content over squeezing out every
possible dedup.

Storage layout:
    ~/.applypilot/documents/canonical/{sha256}_{filename}   -- one real copy
    ~/.applypilot/documents/by_job/{job_hash}/{filename}.json  -- a small
        JSON pointer per job, not a copy and not a symlink (Windows
        symlinks need admin privileges by default; a pointer file works
        everywhere and is just as readable).

NOT built in this pass (left as a real, separate follow-up): the actual
download TRIGGER during a live Claude Code apply run -- i.e. teaching the
apply agent's own prompt (prompt.py) to recognize and save ancillary
documents it encounters via its Playwright MCP tools. That needs its own
look at what playwright-mcp's tool surface actually supports for
downloads before touching prompt.py, a heavily live-tuned file. This
module is the deterministic storage/dedup half of FW50 -- callable today
via store_document(), ready for whatever triggers the actual save.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _documents_dir() -> Path:
    from applypilot import config

    return Path(config.APP_DIR) / "documents"


def _canonical_dir() -> Path:
    return _documents_dir() / "canonical"


def _job_hash(job_url: str) -> str:
    return hashlib.sha256(job_url.encode()).hexdigest()[:12]


def _job_dir(job_url: str) -> Path:
    return _documents_dir() / "by_job" / _job_hash(job_url)


def store_document(job_url: str, filename: str, content: bytes) -> dict:
    """Store `content` (encountered during `job_url`'s application) under
    `filename`, deduping by exact SHA-256 against every document already
    stored for ANY job. Returns:
        {canonical_path, pointer_path, sha256, is_new_document}
    Never raises on a storage-layer failure that isn't actionable by the
    caller (disk full, permissions) -- those surface as a real exception,
    since silently losing a document is worse than a crashed caller here.
    """
    digest = hashlib.sha256(content).hexdigest()
    canonical_dir = _canonical_dir()
    canonical_dir.mkdir(parents=True, exist_ok=True)

    # The canonical filename is keyed ONLY on the hash (plus extension, for
    # tools/humans that care about file type) -- not the original filename.
    # The whole point is catching the same notice re-named differently per
    # employer (the exact scenario FW50 was raised for); including the
    # original name here would make that case never dedup at all (caught
    # live by this module's own test suite before shipping).
    ext = "".join(c for c in Path(filename).suffix if c.isalnum() or c == ".")[:10]
    canonical_path = canonical_dir / f"{digest}{ext}"
    is_new_document = not canonical_path.exists()

    # Sanitize filename for the per-job pointer's own name on disk.
    safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename) or "document"
    if is_new_document:
        canonical_path.write_bytes(content)
        logger.info("Stored new canonical document: %s (%d bytes)", canonical_path.name, len(content))
    else:
        logger.info("Exact duplicate of an existing document (sha256=%s); reusing canonical copy", digest[:12])

    job_dir = _job_dir(job_url)
    job_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = job_dir / f"{safe_filename}.json"
    pointer_path.write_text(
        json.dumps(
            {
                "canonical_path": str(canonical_path),
                "sha256": digest,
                "original_filename": filename,
                "stored_at": datetime.now(timezone.utc).isoformat(),
                "is_new_document": is_new_document,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "canonical_path": str(canonical_path),
        "pointer_path": str(pointer_path),
        "sha256": digest,
        "is_new_document": is_new_document,
    }


def list_job_documents(job_url: str) -> list[dict]:
    """All documents stored for `job_url`, newest first. Each entry is the
    pointer JSON plus its own `pointer_path`."""
    job_dir = _job_dir(job_url)
    if not job_dir.is_dir():
        return []
    entries = []
    for pointer_path in job_dir.glob("*.json"):
        try:
            data = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        data["pointer_path"] = str(pointer_path)
        entries.append(data)
    entries.sort(key=lambda e: e.get("stored_at", ""), reverse=True)
    return entries
