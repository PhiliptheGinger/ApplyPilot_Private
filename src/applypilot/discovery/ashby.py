"""Ashby ATS direct-API scraper.

Scrapes Ashby-powered career sites (MotherDuck, Statsig, Deepgram,
PostHog, Vercel, OpenAI, Linear, Notion, Ramp, …) via the public
posting API. Zero LLM, zero browser — pure HTTP.

Posting API: https://api.ashbyhq.com/posting-api/job-board/{slug}

Company slugs configured in ``config/ashby_employers.yaml``. Same shape
and life-cycle as ``discovery/greenhouse.py`` and
``discovery/lever.py`` — the three are now structurally identical and
P3.10 will DRY them up.
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
from applypilot.discovery.greenhouse import _location_ok, _load_location_filter

log = logging.getLogger(__name__)


ASHBY_API = "https://api.ashbyhq.com/posting-api/job-board/{slug}"
_HEADERS = {
    "User-Agent": "ApplyPilot/1.0 (job-discovery)",
    "Accept": "application/json",
}


# ── Employer registry ─────────────────────────────────────────────────

def load_employers() -> dict:
    """Load Ashby employer registry from config/ashby_employers.yaml."""
    path = CONFIG_DIR / "ashby_employers.yaml"
    if not path.exists():
        log.warning("ashby_employers.yaml not found at %s", path)
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
    """Compose a human-readable location from Ashby's scattered fields.

    Ashby returns ``location`` (primary), ``secondaryLocations`` (extras),
    ``isRemote`` (bool), and ``workplaceType`` (Hybrid/Remote/On-site).
    The location filter wants a single string with "remote" present when
    applicable so its remote-allowlist fires.
    """
    parts: list[str] = []
    wt = (posting.get("workplaceType") or "").lower()
    if wt == "remote" or posting.get("isRemote"):
        parts.append("remote")
    elif wt and wt != "on-site":
        parts.append(wt)
    primary = posting.get("location")
    if primary:
        parts.append(str(primary))
    for extra in posting.get("secondaryLocations") or []:
        if isinstance(extra, dict):
            extra = extra.get("location") or ""
        if extra and str(extra) not in parts:
            parts.append(str(extra))
    return ", ".join(parts)


def scrape_one_employer(
    slug: str,
    emp: dict,
    accept_locs: list[str],
    max_retries: int = 2,
) -> tuple[list[dict], str | None]:
    """Fetch all postings for one Ashby board.

    Returns (jobs, error). ``jobs`` is a list of normalized dicts
    matching the shape used by ``_insert_jobs``.
    """
    url = ASHBY_API.format(slug=slug)
    last_err: str | None = None

    payload: dict = {}
    for attempt in range(max_retries):
        try:
            payload = _fetch_json(url)
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

    postings = payload.get("jobs") or []
    name = emp.get("name", slug)
    out: list[dict] = []

    for posting in postings:
        # Skip unlisted jobs (drafts, internal, etc.).
        if posting.get("isListed") is False:
            continue
        location = _location_string(posting)
        if not _location_ok(location, accept_locs):
            continue
        job_url = posting.get("jobUrl")
        if not job_url:
            continue
        apply_url = posting.get("applyUrl") or job_url
        description = posting.get("descriptionPlain") or ""

        out.append({
            "url": job_url,
            "title": posting.get("title") or "",
            "location": location or None,
            "description": (description[:500] if description else None),
            "full_description": description if len(description) > 200 else None,
            "application_url": apply_url,
            "employer_name": name,
            "employer_slug": slug,
            "posted_at": posting.get("publishedAt") or None,
        })
    return out, None


# ── DB insert ─────────────────────────────────────────────────────────

def _insert_jobs(conn: sqlite3.Connection, jobs: list[dict]) -> tuple[int, int]:
    """Insert jobs. Returns (new, existing). Mirrors lever._insert_jobs."""
    new = 0
    existing = 0
    now = datetime.now(timezone.utc).isoformat()

    for job in jobs:
        url = job.get("url")
        if not url:
            continue
        full_description = job.get("full_description")
        detail_scraped_at = now if full_description else None
        site = job.get("employer_name", "Ashby")
        strategy = "ashby_api"
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

def run_ashby_discovery(employers: dict | None = None, workers: int = 1) -> dict:
    """Discover jobs from Ashby-powered career sites.

    Args:
        employers: Override the employer registry (for tests).
        workers: Currently unused — fetches are sequential.

    Returns:
        {'found', 'new', 'existing', 'employers', 'errors'}
    """
    if employers is None:
        employers = load_employers()

    if not employers:
        log.warning("No Ashby employers configured. Create config/ashby_employers.yaml.")
        return {"found": 0, "new": 0, "existing": 0, "employers": 0, "errors": []}

    accept_locs = _load_location_filter()
    conn = get_connection()
    init_db()

    grand_new = 0
    grand_existing = 0
    grand_found = 0
    errors: list[str] = []

    log.info("Ashby crawl: %d employers", len(employers))
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
            log.exception("Ashby scrape failed for %s: %s", slug, e)
            errors.append(f"{slug}: {e}")

    log.info("Ashby crawl done: %d found (%d new, %d existing) across %d employers",
             grand_found, grand_new, grand_existing, len(employers))

    return {
        "found": grand_found,
        "new": grand_new,
        "existing": grand_existing,
        "employers": len(employers),
        "errors": errors,
    }
