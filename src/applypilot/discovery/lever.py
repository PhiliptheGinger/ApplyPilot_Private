"""Lever ATS direct API scraper.

Scrapes Lever-powered career sites (Highspot, Outreach, Rover, Plaid,
…) via the public postings endpoint. Zero LLM, zero browser — pure
HTTP.

Postings API: https://api.lever.co/v0/postings/{slug}?mode=json

Company slugs configured in ``config/lever_employers.yaml``. Same
shape and life-cycle as ``discovery/greenhouse.py``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import yaml

from applypilot import config
from applypilot.config import CONFIG_DIR
from applypilot.database import commit_with_retry, get_connection, init_db
from applypilot.discovery.greenhouse import _location_ok, _load_location_filter, _strip_html

log = logging.getLogger(__name__)


LEVER_API = "https://api.lever.co/v0/postings/{slug}?mode=json"
_HEADERS = {
    "User-Agent": "ApplyPilot/1.0 (job-discovery)",
    "Accept": "application/json",
}


# ── Employer registry ─────────────────────────────────────────────────

def load_employers() -> dict:
    """Load Lever employer registry from config/lever_employers.yaml."""
    path = CONFIG_DIR / "lever_employers.yaml"
    if not path.exists():
        log.warning("lever_employers.yaml not found at %s", path)
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("employers", {})


# ── HTTP fetch ────────────────────────────────────────────────────────

def _fetch_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Per-employer scrape ───────────────────────────────────────────────

def _location_string(posting: dict) -> str:
    """Lever scatters location across categories.location + workplaceType.

    Returns a human-readable composite, with workplaceType prepended for
    `remote` (so the location filter's remote-allowlist can fire).
    """
    cats = posting.get("categories") or {}
    parts: list[str] = []
    wt = (posting.get("workplaceType") or "").lower()
    if wt and wt != "on-site":
        parts.append(wt)
    loc = cats.get("location")
    if loc:
        parts.append(str(loc))
    # Lever sometimes lists multiple allLocations
    extra = cats.get("allLocations") or []
    if isinstance(extra, list):
        for e in extra:
            if e and e not in parts:
                parts.append(str(e))
    return ", ".join(parts)


def _description_text(posting: dict) -> str:
    """Compose a plain-text description from Lever's HTML and lists fields."""
    parts: list[str] = []
    desc_html = posting.get("description") or ""
    if desc_html:
        parts.append(_strip_html(desc_html))
    # Lever's `lists` is an array of {text, content} sections (e.g. "What
    # you'll do", "Requirements").
    for section in posting.get("lists") or []:
        title = section.get("text") or ""
        body = section.get("content") or ""
        if not body:
            continue
        if title:
            parts.append(f"\n{title}")
        parts.append(_strip_html(body))
    additional = posting.get("additional") or ""
    if additional:
        parts.append(_strip_html(additional))
    return "\n\n".join(p for p in parts if p)


def scrape_one_employer(
    slug: str,
    emp: dict,
    accept_locs: list[str],
    max_retries: int = 2,
) -> tuple[list[dict], str | None]:
    """Fetch all postings for one Lever board.

    Returns (jobs, error). ``jobs`` is a list of normalized dicts
    matching the shape used by ``_insert_jobs``.
    """
    url = LEVER_API.format(slug=slug)
    last_err: str | None = None

    postings: list[dict] = []
    for attempt in range(max_retries):
        try:
            data = _fetch_json(url)
            postings = data if isinstance(data, list) else []
            break
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code} {e.reason}"
            if e.code == 404:
                return [], last_err
            time.sleep(2 + attempt * 3)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 + attempt * 3)
    else:
        return [], last_err or "unknown error"

    name = emp.get("name", slug)
    out: list[dict] = []
    for posting in postings:
        location = _location_string(posting)
        if not _location_ok(location, accept_locs):
            continue
        hosted_url = posting.get("hostedUrl")
        apply_url = posting.get("applyUrl") or hosted_url
        if not hosted_url:
            continue

        description = _description_text(posting)
        # Lever timestamps are ms-since-epoch; convert to ISO if present.
        created = posting.get("createdAt")
        posted_at: str | None = None
        if isinstance(created, (int, float)) and created > 0:
            posted_at = datetime.fromtimestamp(
                created / 1000, tz=timezone.utc
            ).isoformat()

        out.append({
            "url": hosted_url,
            "title": posting.get("text") or "",
            "location": location or None,
            "description": (description[:500] if description else None),
            "full_description": description if len(description) > 200 else None,
            "application_url": apply_url,
            "employer_name": name,
            "employer_slug": slug,
            "posted_at": posted_at,
        })
    return out, None


# ── DB insert ─────────────────────────────────────────────────────────

def _insert_jobs(conn: sqlite3.Connection, jobs: list[dict]) -> tuple[int, int]:
    """Insert jobs. Returns (new, existing). Mirrors greenhouse._insert_jobs."""
    new = 0
    existing = 0
    now = datetime.now(timezone.utc).isoformat()

    for job in jobs:
        url = job.get("url")
        if not url:
            continue
        full_description = job.get("full_description")
        detail_scraped_at = now if full_description else None
        site = job.get("employer_name", "Lever")
        strategy = "lever_api"
        initial_state = "enriched" if full_description else "discovered"
        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, "
                "discovered_at, posted_at, full_description, application_url, "
                "detail_scraped_at, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (url, job.get("title"), None, job.get("description"),
                 job.get("location"), site, strategy, now,
                 job.get("posted_at"), full_description,
                 job.get("application_url"), detail_scraped_at, initial_state),
            )
            conn.execute(
                "INSERT INTO job_state_transitions "
                "(job_url, from_state, to_state, at, reason, metadata) "
                "VALUES (?, NULL, ?, ?, ?, ?)",
                (url, initial_state, now, f"discovered via {strategy}", None),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    commit_with_retry(conn)
    return new, existing


# ── Public entry point ────────────────────────────────────────────────

def run_lever_discovery(employers: dict | None = None, workers: int = 1) -> dict:
    """Discover jobs from Lever-powered career sites.

    Args:
        employers: Override the employer registry (for tests).
        workers: Currently unused — fetches are sequential.

    Returns:
        {'found', 'new', 'existing', 'employers', 'errors'}
    """
    if employers is None:
        employers = load_employers()

    if not employers:
        log.warning("No Lever employers configured. Create config/lever_employers.yaml.")
        return {"found": 0, "new": 0, "existing": 0, "employers": 0, "errors": []}

    accept_locs = _load_location_filter()
    conn = get_connection()
    init_db()

    grand_new = 0
    grand_existing = 0
    grand_found = 0
    errors: list[str] = []

    log.info("Lever crawl: %d employers", len(employers))
    for slug, emp in employers.items():
        name = emp.get("name", slug)
        try:
            jobs, err = scrape_one_employer(slug, emp, accept_locs)
            if err:
                log.warning("  [%s] %s", slug, err)
                errors.append(f"{slug}: {err}")
                continue
            new, existing = _insert_jobs(conn, jobs)
            grand_new += new
            grand_existing += existing
            grand_found += len(jobs)
            log.info("  [%s] %s: %d found (%d new, %d existing)",
                     slug, name, len(jobs), new, existing)
        except Exception as e:
            log.exception("Lever scrape failed for %s: %s", slug, e)
            errors.append(f"{slug}: {e}")

    log.info("Lever crawl done: %d found (%d new, %d existing) across %d employers",
             grand_found, grand_new, grand_existing, len(employers))

    return {
        "found": grand_found,
        "new": grand_new,
        "existing": grand_existing,
        "employers": len(employers),
        "errors": errors,
    }
