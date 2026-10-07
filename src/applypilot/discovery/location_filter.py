"""Shared discovery-time location filter.

2026-10-07 consolidation (pattern audit, docs/audit_2026-10.md): four
scrapers each carried their own `_location_ok`, and they had drifted:

- jobspy/workday/smartextract read top-level `location_accept` /
  `location_reject_non_remote` keys, which neither `searches.example.yaml`
  nor `applypilot init` ever writes. With those keys absent, every
  non-remote job that had a location string was dropped.
- greenhouse (also used by lever and ashby) read `location.accept_patterns`,
  the documented key, but never applied any reject list, and kept
  everything when the accept list was empty.

This module is now the one implementation. It reads both config shapes
(documented keys first, legacy keys merged in) so existing searches.yaml
files keep working, and treats an empty accept list as "no location
preference" everywhere.

Accept and reject patterns both match whole words (2026-10-07). Accept used
to be a plain substring test, so "NC" accepted "San Francisco" and "US"
accepted "Austin, TX".
"""

from __future__ import annotations

import re
from collections.abc import Iterable

REMOTE_MARKERS = ("remote", "anywhere", "work from home", "wfh", "distributed")


def _merged(*lists: Iterable[str] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for items in lists:
        for item in items or []:
            if not isinstance(item, str):
                continue
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(item.strip())
    return out


def load_location_filter(search_cfg: dict | None = None) -> tuple[list[str], list[str]]:
    """Return (accept, reject) patterns from searches.yaml.

    Reads the documented `location.accept_patterns` / `location.reject_patterns`
    and merges in the legacy top-level `location_accept` /
    `location_reject_non_remote` keys.
    """
    if search_cfg is None:
        from applypilot import config

        search_cfg = config.load_search_config()
    loc = search_cfg.get("location") or {}
    if not isinstance(loc, dict):
        loc = {}
    accept = _merged(loc.get("accept_patterns"), search_cfg.get("location_accept"))
    reject = _merged(loc.get("reject_patterns"), search_cfg.get("location_reject_non_remote"))
    return accept, reject


def pattern_in_location(pattern: str, location_lower: str) -> bool:
    """Whole-word, case-insensitive match of one pattern inside a location.

    Uses alphanumeric lookarounds rather than \\b so patterns that end in
    punctuation ("Washington, D.C.") still match.
    """
    p = (pattern or "").strip().lower()
    if not p:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", location_lower) is not None


def location_ok(location: str | None, accept: Iterable[str] = (), reject: Iterable[str] = ()) -> bool:
    """Whether a posting's location passes the candidate's filter.

    - Unknown/blank location: keep (the scorer decides later).
    - Remote-style location: keep.
    - Matches a reject pattern (whole word, case-insensitive): drop. Whole
      words matter -- a substring check let "India" reject
      "Indianapolis, IN" and "NC" reject "France".
    - No accept patterns configured: keep.
    - Otherwise keep only if an accept pattern appears as a whole word.
    """
    if not location:
        return True
    loc = location.lower()
    if any(marker in loc for marker in REMOTE_MARKERS):
        return True
    if any(pattern_in_location(p, loc) for p in reject):
        return False
    accept = [a for a in accept if a and a.strip()]
    if not accept:
        return True
    return any(pattern_in_location(a, loc) for a in accept)
