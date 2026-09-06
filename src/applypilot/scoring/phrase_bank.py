"""Persisted, per-entry bank of alternate TRUE phrasings for resume bullets.

This is "Stage 0" of tailoring: today, every job-specific tailoring run
(cloud or degraded/local) reads `profile.json`'s raw `responsibilities`/
`factual_concepts` text completely fresh, every single time. This module
is the missing piece that runs ONCE per resume/profile instead -- for each
original bullet on an entry, generate several alternate phrasings of that
SAME fact (never a new fact), keep only the ones that survive the existing
claim/agency/causal/metric fabrication checks and a diversity filter, and
persist the survivors so every later job-specific run can just SELECT from
this bank (local_tailor.build_pool_realization) instead of generating text
from scratch on every job.

Generation itself lives in local_tailor.build_phrase_bank (reuses the
proven technique from data/experiments/deterministic_slotfiller_20260902/
inventory_expansion_pilot_v2_diversity.py, generalized to run per-bullet,
for any entry, not one hardcoded name). This module only handles
persistence -- same shape as apply/successful_paths.py's save_path/
load_path (one JSON file per entry under ~/.applypilot/), except staleness
here is detected via a CONTENT HASH of the entry's own source facts
(responsibilities/factual_concepts) rather than age: an entry whose facts
haven't changed should never need regenerating, no matter how old the
bank file is, but an edited resume must not silently keep serving
phrasings of facts that no longer match.

Storage: ~/.applypilot/phrase_bank/{slug}.json, one file per entry::

    {
      "entry_name": "National Tire and Battery / Mavis",
      "content_hash": "3f9a...",
      "generated_at": "2026-09-05T...",
      "bank": {
        "Diagnosed and corrected vehicle alignment issues...": [
          "Identified and resolved wheel alignment problems...",
          "Troubleshot and corrected vehicle alignment faults..."
        ],
        ...
      }
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from applypilot import config

logger = logging.getLogger(__name__)

PHRASE_BANK_DIR = config.APP_DIR / "phrase_bank"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(entry_name: str) -> str:
    """entry name -> safe filename stem. Not required to be reversible --
    only used as a cache key, the real name lives inside the file."""
    return _SLUG_RE.sub("_", entry_name.strip().lower()).strip("_") or "entry"


def source_facts(item: dict) -> list[str]:
    """The facts a phrase bank is generated FROM for one profile item --
    responsibilities for experience_inventory entries, factual_concepts
    for project_inventory entries (mirrors local_tailor._evidence_own_text
    /build_phrase_bank's own source selection, kept here too since the
    content hash must be computed from the SAME facts that gate staleness,
    independent of which caller is asking)."""
    resp = [r for r in (item.get("responsibilities") or []) if isinstance(r, str) and r.strip()]
    if resp:
        return resp
    return [c for c in (item.get("factual_concepts") or []) if isinstance(c, str) and c.strip()]


def content_hash(item: dict) -> str:
    """Hash of the item's own source facts -- changes if and only if the
    facts a bank would be generated from change. Order-sensitive (a
    reordered responsibilities list is a real edit, not a no-op) but not
    sensitive to unrelated profile.json fields (name, dates, etc.) -- the
    bank stays valid across an edit to something the bank never read."""
    facts = source_facts(item)
    return hashlib.sha256("\n".join(facts).encode("utf-8")).hexdigest()


def save_bank(entry_name: str, bank: dict[str, list[str]], item_content_hash: str) -> Path | None:
    """Persist a generated bank for one entry. Overwrites any existing
    file for the same entry (latest generation wins -- unlike successful_
    paths.py's keep-fastest rule, there's no meaningful "better" bank to
    protect here, just a current one)."""
    if not entry_name or not bank:
        return None
    try:
        PHRASE_BANK_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PHRASE_BANK_DIR / f"{slugify(entry_name)}.json"
        payload = {
            "entry_name": entry_name,
            "content_hash": item_content_hash,
            "generated_at": datetime.now(UTC).isoformat(),
            "bank": bank,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Saved phrase bank for %r (%d original bullets, %d total variants)",
            entry_name,
            len(bank),
            sum(len(v) for v in bank.values()),
        )
        return out_path
    except Exception:
        logger.debug("Could not save phrase bank for %r", entry_name, exc_info=True)
        return None


def load_bank(entry_name: str, current_content_hash: str) -> dict[str, list[str]] | None:
    """Load the per-bullet bank for one entry, keyed
    {original_bullet: [variant, ...]} -- or None if no bank exists, it
    failed to parse, or its content_hash doesn't match
    current_content_hash (the entry's facts changed since the bank was
    generated -- never silently serve stale phrasing)."""
    if not entry_name:
        return None
    path = PHRASE_BANK_DIR / f"{slugify(entry_name)}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Could not load phrase bank for %r", entry_name, exc_info=True)
        return None
    if payload.get("content_hash") != current_content_hash:
        logger.info(
            "Phrase bank for %r is stale (source facts changed since generation) -- ignoring.", entry_name
        )
        return None
    bank = payload.get("bank")
    return bank if isinstance(bank, dict) else None


def flatten_for_selector(bank: dict[str, list[str]] | None) -> list[str]:
    """local_tailor.build_pool_realization's `sentence_pools` wants a flat
    list of candidate sentences per evidence item -- it ranks by relevance
    to a job requirement's text, not by which original bullet a variant
    came from, so per-bullet grouping is provenance for a human reviewer,
    not something the selector itself needs preserved."""
    if not bank:
        return []
    flat: list[str] = []
    for variants in bank.values():
        flat.extend(variants)
    return flat
