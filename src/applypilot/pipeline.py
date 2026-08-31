"""ApplyPilot Pipeline Orchestrator.

Runs pipeline stages in sequence or concurrently (streaming mode).

Usage (via CLI):
    applypilot run                        # all stages, sequential
    applypilot run --stream               # all stages, concurrent
    applypilot run discover enrich        # specific stages
    applypilot run score tailor cover     # LLM-only stages
    applypilot run --dry-run              # preview without executing
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from datetime import UTC, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from applypilot.config import APP_DIR, LOG_DIR, ensure_dirs, load_env
from applypilot.database import get_connection, get_stats, init_db

log = logging.getLogger(__name__)
console = Console()

_RUN_STATE_DIR = APP_DIR / "run-state"
_STREAM_STOP_FILE = _RUN_STATE_DIR / "stop-stream"
_STREAM_PID_FILE = _RUN_STATE_DIR / "stream.pid"


def _setup_file_logging(stages: list[str]) -> logging.FileHandler | None:
    """Add a FileHandler to the root logger for this pipeline run.

    Creates a log file named like: 2026-02-20_01-37-00_discover.log
    Returns the handler so it can be removed after the run.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    stage_tag = "+".join(stages) if len(stages) <= 4 else f"{stages[0]}+{len(stages) - 1}more"
    log_path = LOG_DIR / f"{ts}_{stage_tag}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    log.info("Log file: %s", log_path)
    return handler


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_ORDER = ("discover", "enrich", "score", "tailor", "cover", "pdf")

# Stages whose worker function actually accepts/consumes min_score
# (_run_tailor, _run_cover -- both filter pending_tailor/pending_cover on
# `fit_score >= ?`). _run_score/_run_enrich/discover/pdf have no min_score
# parameter at all. Used only to decide whether the run banner's "Min
# score" line is relevant to the requested stages -- see run_pipeline.
_MIN_SCORE_STAGES = frozenset({"tailor", "cover"})

STAGE_META: dict[str, dict] = {
    "discover": {"desc": "Job discovery (JobSpy + Workday + smart extract + HN)"},
    "enrich": {"desc": "Detail enrichment (full descriptions + apply URLs)"},
    "score": {"desc": "LLM scoring (fit 1-10)"},
    "tailor": {"desc": "Resume tailoring (LLM + validation)"},
    "cover": {"desc": "Cover letter generation"},
    "pdf": {"desc": "PDF conversion (tailored resumes + cover letters)"},
}

# Upstream dependency: a stage only finishes when its upstream is done AND
# it has no remaining pending work.
_UPSTREAM: dict[str, str | None] = {
    "discover": None,
    "enrich": "discover",
    "score": "enrich",
    "tailor": "score",
    "cover": "tailor",
    "pdf": "cover",
}


# ---------------------------------------------------------------------------
# Discovery sources
# ---------------------------------------------------------------------------

# Canonical name → description. Order determines default execution order.
DISCOVERY_SOURCES: dict[str, str] = {
    "jobspy": "JobSpy aggregator (LinkedIn, Indeed, ZipRecruiter)",
    "linkedin": "LinkedIn only (via JobSpy)",
    "indeed": "Indeed only (via JobSpy)",
    "workday": "Workday corporate career sites",
    "greenhouse": "Greenhouse ATS public board API",
    "lever": "Lever ATS public postings API",
    "ashby": "Ashby ATS public posting API",
    "amazon": "amazon.jobs public search API",
    "costco": "careers.costco.com public API (Issaquah HQ)",
    "builtin": "builtin.com (config-driven cities + categories)",
    "smartextract": "Smart extract (AI-powered scraping, incl. Dice via sites.yaml)",
    "hackernews": "Hacker News 'Who is Hiring?' thread",
}

# Alias → canonical name for CLI convenience
_SOURCE_ALIASES: dict[str, str] = {
    "hn": "hackernews",
    "smart": "smartextract",
    "dice": "smartextract",  # dice is scraped via smartextract + sites.yaml
}

# Sources that are jobspy with a specific site filter
_JOBSPY_SITE_SOURCES: dict[str, list[str]] = {
    "linkedin": ["linkedin"],
    "indeed": ["indeed"],
}


def resolve_source_names(names: list[str]) -> list[str]:
    """Resolve source aliases and validate names. Returns canonical names."""
    resolved = []
    for name in names:
        canonical = _SOURCE_ALIASES.get(name, name)
        if canonical not in DISCOVERY_SOURCES:
            valid = sorted(set(list(DISCOVERY_SOURCES.keys()) + list(_SOURCE_ALIASES.keys())))
            raise ValueError(f"Unknown discovery source: '{name}'. Valid sources: {', '.join(valid)}")
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


# ---------------------------------------------------------------------------
# Individual stage runners
# ---------------------------------------------------------------------------


def _run_discover(workers: int = 1, sources: list[str] | None = None) -> dict:
    """Stage: Job discovery — JobSpy, Workday, smart-extract, and HN scrapers.

    Args:
        workers: Thread count for sources that support parallelism.
        sources: Canonical source names to run, or None for all.
    """
    run_all = sources is None
    active = list(DISCOVERY_SOURCES.keys()) if run_all else sources
    stats: dict = {s: None for s in active}

    if "jobspy" in active:
        console.print("  [cyan]JobSpy full crawl...[/cyan]")
        try:
            from applypilot.discovery.jobspy import run_discovery

            run_discovery()
            stats["jobspy"] = "ok"
        except Exception as e:
            log.exception("JobSpy crawl failed")
            console.print(f"  [red]JobSpy error:[/red] {e}")
            stats["jobspy"] = f"error: {e}"

    # Site-specific JobSpy sources (dice, linkedin, indeed)
    for source_name, sites in _JOBSPY_SITE_SOURCES.items():
        if source_name in active:
            console.print(f"  [cyan]JobSpy ({source_name})...[/cyan]")
            try:
                from applypilot.discovery.jobspy import run_discovery

                run_discovery(sites_override=sites)
                stats[source_name] = "ok"
            except Exception as e:
                log.exception("JobSpy (%s) crawl failed", source_name)
                console.print(f"  [red]JobSpy ({source_name}) error:[/red] {e}")
                stats[source_name] = f"error: {e}"

    if "workday" in active:
        console.print("  [cyan]Workday corporate scraper...[/cyan]")
        try:
            from applypilot.discovery.workday import run_workday_discovery

            run_workday_discovery(workers=workers)
            stats["workday"] = "ok"
        except Exception as e:
            log.exception("Workday scraper failed")
            console.print(f"  [red]Workday error:[/red] {e}")
            stats["workday"] = f"error: {e}"

    if "greenhouse" in active:
        console.print("  [cyan]Greenhouse ATS scraper...[/cyan]")
        try:
            from applypilot.discovery.greenhouse import run_greenhouse_discovery

            run_greenhouse_discovery(workers=workers)
            stats["greenhouse"] = "ok"
        except Exception as e:
            log.exception("Greenhouse scraper failed")
            console.print(f"  [red]Greenhouse error:[/red] {e}")
            stats["greenhouse"] = f"error: {e}"

    if "lever" in active:
        console.print("  [cyan]Lever ATS scraper...[/cyan]")
        try:
            from applypilot.discovery.lever import run_lever_discovery

            run_lever_discovery(workers=workers)
            stats["lever"] = "ok"
        except Exception as e:
            log.exception("Lever scraper failed")
            console.print(f"  [red]Lever error:[/red] {e}")
            stats["lever"] = f"error: {e}"

    if "ashby" in active:
        console.print("  [cyan]Ashby ATS scraper...[/cyan]")
        try:
            from applypilot.discovery.ashby import run_ashby_discovery

            run_ashby_discovery(workers=workers)
            stats["ashby"] = "ok"
        except Exception as e:
            log.exception("Ashby scraper failed")
            console.print(f"  [red]Ashby error:[/red] {e}")
            stats["ashby"] = f"error: {e}"

    if "amazon" in active:
        console.print("  [cyan]Amazon.jobs scraper...[/cyan]")
        try:
            from applypilot.discovery.amazon import run_amazon_discovery

            run_amazon_discovery(workers=workers)
            stats["amazon"] = "ok"
        except Exception as e:
            log.exception("Amazon scraper failed")
            console.print(f"  [red]Amazon error:[/red] {e}")
            stats["amazon"] = f"error: {e}"

    if "costco" in active:
        console.print("  [cyan]Costco careers scraper...[/cyan]")
        try:
            from applypilot.discovery.costco import run_costco_discovery

            run_costco_discovery(workers=workers)
            stats["costco"] = "ok"
        except Exception as e:
            log.exception("Costco scraper failed")
            console.print(f"  [red]Costco error:[/red] {e}")
            stats["costco"] = f"error: {e}"

    if "builtin" in active:
        console.print("  [cyan]BuiltIn (builtin.com) scraper...[/cyan]")
        try:
            from applypilot.discovery.builtin import run_builtin_discovery

            run_builtin_discovery(workers=workers)
            stats["builtin"] = "ok"
        except Exception as e:
            log.exception("BuiltIn scraper failed")
            console.print(f"  [red]BuiltIn error:[/red] {e}")
            stats["builtin"] = f"error: {e}"

    if "smartextract" in active:
        console.print("  [cyan]Smart extract (AI-powered scraping)...[/cyan]")
        try:
            from applypilot.discovery.smartextract import run_smart_extract

            run_smart_extract(workers=workers)
            stats["smartextract"] = "ok"
        except Exception as e:
            log.exception("Smart extract failed")
            console.print(f"  [red]Smart extract error:[/red] {e}")
            stats["smartextract"] = f"error: {e}"

    if "hackernews" in active:
        console.print("  [cyan]Hacker News 'Who is Hiring?' thread...[/cyan]")
        try:
            from applypilot.discovery.hackernews import run_hn_discovery

            hn_result = run_hn_discovery()
            new = hn_result.get("new", 0)
            console.print(f"  [dim]HN: {new} new jobs from '{hn_result.get('thread_title', '?')}'[/dim]")
            stats["hackernews"] = "ok"
        except Exception as e:
            log.exception("HN discovery failed")
            console.print(f"  [red]HN error:[/red] {e}")
            stats["hackernews"] = f"error: {e}"

    return stats


def _run_enrich(workers: int = 1) -> dict:
    """Stage: Detail enrichment — scrape full descriptions and apply URLs."""
    try:
        from applypilot.enrichment.detail import run_enrichment

        # 2026-08-30 fix: used to discard run_enrichment()'s real stats dict
        # (its own docstring: "processed, ok, partial, error, tiers") and
        # return a placeholder {"status": "ok"} instead. _run_stage_streaming's
        # progress-detection sums specific keys out of whatever this returns
        # to decide whether to back off 10s between polls -- the placeholder
        # has none of those keys, so every successful enrich pass in
        # streaming mode looked like zero progress and triggered a needless
        # 10s backoff regardless of how many jobs were actually enriched.
        stats = run_enrichment(workers=workers)
        stats.setdefault("status", "ok")
        return stats
    except Exception as e:
        log.exception("Enrichment failed")
        return {"status": f"error: {e}"}


def _run_score(workers: int = 1, max_age_days: int | None = None, limit: int = 0) -> dict:
    """Stage: LLM scoring — assign fit scores 1-10."""
    from applypilot.config import DEFAULTS

    if max_age_days is None:
        max_age_days = DEFAULTS["max_job_age_days"]
    try:
        from applypilot.scoring.scorer import run_scoring

        result = run_scoring(workers=workers, max_age_days=max_age_days, limit=limit)
        # job_ids carries this run's exact batch identity forward so a
        # following `tailor` stage in the same sequential run can restrict
        # itself to these jobs instead of independently re-querying (see
        # _run_sequential and run_tailoring's job_ids parameter).
        #
        # 2026-08-30 fix: this used to return ONLY {"status": "ok",
        # "job_ids": [...]}, discarding run_scoring()'s real "scored"/
        # "errors" counts entirely -- confirmed live: a real streaming run
        # logged "Stage 'score' pass N had pending=X but progress=0;
        # backing off 10s" on EVERY pass, including ones that had just
        # successfully scored a job, because _stage_progress (see
        # _PROGRESS_KEYS above) never got to see the real numbers -- this
        # wrapper threw them away first. Forwarding them is what actually
        # fixes the false backoff; the _PROGRESS_KEYS list alone was
        # necessary but not sufficient. Same bug, same fix, as
        # _run_enrich/_run_tailor/_run_cover below.
        return {
            "status": "ok",
            "job_ids": result.get("job_urls") or [],
            "scored": result.get("scored", 0),
            "errors": result.get("errors", 0),
        }
    except Exception as e:
        log.exception("Scoring failed")
        return {"status": f"error: {e}"}


def _run_tailor(
    min_score: int | None = None,
    max_age_days: int | None = None,
    limit: int = 20,
    workers: int = 1,
    doc_format: str = "docx",
    job_ids: list[str] | None = None,
) -> dict:
    """Stage: Resume tailoring — generate tailored resumes for high-fit jobs.

    job_ids: an upstream score stage's exact batch (see _run_score/
        _run_sequential), or None to independently select up to `limit`
        eligible jobs (standalone `applypilot run tailor` behavior).
    """
    from applypilot.config import DEFAULTS

    if min_score is None:
        min_score = DEFAULTS["min_score"]
    if max_age_days is None:
        max_age_days = DEFAULTS["max_job_age_days"]
    try:
        from applypilot.scoring.tailor import run_tailoring

        result = run_tailoring(
            min_score=min_score,
            max_age_days=max_age_days,
            limit=limit,
            workers=workers,
            doc_format=doc_format,
            job_ids=job_ids,
        )
        # job_ids carries this run's exact eligible-after-cap batch forward
        # so a following `cover` stage in the same sequential run can
        # restrict itself to these jobs instead of independently
        # re-querying (see _run_sequential and run_cover_letters's job_ids
        # parameter). Mirrors _run_score's identical pattern.
        #
        # 2026-08-30 fix: see _run_score's identical fix above -- this used
        # to discard run_tailoring()'s real "approved"/"failed"/"errors"
        # counts, causing the same false progress=0 streaming backoff for
        # the tailor stage.
        return {
            "status": "ok",
            "job_ids": result.get("job_urls") or [],
            "approved": result.get("approved", 0),
            "failed": result.get("failed", 0),
            "errors": result.get("errors", 0),
        }
    except Exception as e:
        log.exception("Tailoring failed")
        return {"status": f"error: {e}"}


def _run_cover(
    min_score: int | None = None,
    max_age_days: int | None = None,
    limit: int = 20,
    workers: int = 1,
    doc_format: str = "docx",
    job_ids: list[str] | None = None,
) -> dict:
    """Stage: Cover letter generation.

    job_ids: an upstream tailor stage's exact batch (see _run_tailor/
        _run_sequential), or None to independently select up to `limit`
        eligible jobs (standalone `applypilot run cover` behavior).
    """
    from applypilot.config import DEFAULTS

    if min_score is None:
        min_score = DEFAULTS["min_score"]
    if max_age_days is None:
        max_age_days = DEFAULTS["max_job_age_days"]
    try:
        from applypilot.scoring.cover_letter import run_cover_letters

        result = run_cover_letters(
            min_score=min_score,
            max_age_days=max_age_days,
            limit=limit,
            workers=workers,
            doc_format=doc_format,
            job_ids=job_ids,
        )
        # 2026-08-30 fix: this used to discard run_cover_letters()'s return
        # value entirely (didn't even assign it to a variable), causing the
        # same false progress=0 streaming backoff for the cover stage --
        # see _run_score's identical fix above.
        return {
            "status": "ok",
            "generated": result.get("generated", 0),
            "rejected": result.get("rejected", 0),
            "errors": result.get("errors", 0),
        }
    except Exception as e:
        log.exception("Cover letter generation failed")
        return {"status": f"error: {e}"}


def _run_pdf(doc_format: str = "docx") -> dict:
    """Stage: Document conversion — convert tailored resumes and cover letters to PDF/DOCX."""
    try:
        from applypilot.scoring.pdf import batch_convert

        # 2026-08-30 fix: same bug class as _run_enrich/_run_score/_run_tailor/
        # _run_cover above -- this discarded batch_convert()'s real int count
        # of files converted and returned a bare {"status": "ok"} placeholder,
        # which _stage_progress (see _PROGRESS_KEYS) always reads as zero
        # progress. A "pdf" stage in a --stream run (e.g. `run all --stream`)
        # would log a false "progress=0; backing off 10s" after every
        # successful pass, no matter how many files it actually converted.
        processed = batch_convert(doc_format=doc_format)
        return {"status": "ok", "processed": processed}
    except Exception as e:
        log.exception("Document conversion failed")
        return {"status": f"error: {e}"}


# Map stage names to their runner functions
_STAGE_RUNNERS: dict[str, callable] = {
    "discover": _run_discover,
    "enrich": _run_enrich,
    "score": _run_score,
    "tailor": _run_tailor,
    "cover": _run_cover,
    "pdf": _run_pdf,
}


# ---------------------------------------------------------------------------
# Stage resolution
# ---------------------------------------------------------------------------


def _resolve_stages(stage_names: list[str]) -> list[str]:
    """Resolve 'all' and validate/order stage names."""
    if "all" in stage_names:
        return list(STAGE_ORDER)

    resolved = []
    for name in stage_names:
        if name not in STAGE_META:
            console.print(f"[red]Unknown stage:[/red] '{name}'. Available: {', '.join(STAGE_ORDER)}, all")
            raise SystemExit(1)
        if name not in resolved:
            resolved.append(name)

    # Maintain canonical order
    return [s for s in STAGE_ORDER if s in resolved]


# ---------------------------------------------------------------------------
# Streaming pipeline helpers
# ---------------------------------------------------------------------------


class _StageTracker:
    """Thread-safe tracker for which stages have finished producing work."""

    def __init__(self):
        self._events: dict[str, threading.Event] = {stage: threading.Event() for stage in STAGE_ORDER}
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()

    def mark_done(self, stage: str, result: dict | None = None) -> None:
        with self._lock:
            self._results[stage] = result or {"status": "ok"}
        self._events[stage].set()

    def is_done(self, stage: str) -> bool:
        return self._events[stage].is_set()

    def wait(self, stage: str, timeout: float | None = None) -> bool:
        return self._events[stage].wait(timeout=timeout)

    def get_results(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._results)


# 2026-08-28 fix: "enrich"/"score"/"tailor"/"cover" used to have their own
# independently-maintained SQL here (a second, drifted-from-database.py
# definition of "pending" -- see git history for the old `_PENDING_SQL`
# entries). They now delegate to `database.count_jobs_by_stage`, which
# shares the exact same WHERE clause `get_jobs_by_stage` uses for real
# selection, via `_CANONICAL_PENDING_STAGE` below.
#
# "pdf" is the deliberate exception: its selection criterion
# (`tailored_resume_path LIKE '%.txt'`) isn't a state-machine concept and
# has no equivalent `get_jobs_by_stage` stage to share -- inventing one
# would be scope creep with no demonstrated divergence to fix. It stays on
# its own simple, standalone query below.
_PENDING_SQL: dict[str, str] = {
    "pdf": "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND tailored_resume_path LIKE '%.txt'",
}

# Maps a pipeline stage name to its canonical `get_jobs_by_stage`/
# `count_jobs_by_stage` stage name, for stages whose "pending" definition
# is shared with the real selector rather than kept in `_PENDING_SQL` above.
_CANONICAL_PENDING_STAGE: dict[str, str] = {
    "enrich": "pending_detail",
    "score": "pending_score",
    "tailor": "pending_tailor",
    "cover": "pending_cover",
}

# How long to sleep between polling loops in streaming mode (seconds)
_STREAM_POLL_INTERVAL = 10


def _ensure_run_state_dir() -> None:
    _RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)


def _read_stream_pid() -> int | None:
    try:
        if not _STREAM_PID_FILE.exists():
            return None
        raw = _STREAM_PID_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return int(raw)
    except Exception:  # noqa: BLE001 - a missing/corrupt PID file degrades to "no known stream process", not a crash
        return None


def get_stream_pid() -> int | None:
    """Return the recorded stream-run PID, if present."""
    return _read_stream_pid()


def _write_stream_pid(pid: int) -> None:
    _ensure_run_state_dir()
    _STREAM_PID_FILE.write_text(str(pid), encoding="utf-8")


def _clear_stream_pid() -> None:
    try:
        _STREAM_PID_FILE.unlink(missing_ok=True)
    except Exception:
        log.debug("Failed to clear stream pid file", exc_info=True)


def request_stream_stop() -> bool:
    """Signal a running stream pipeline to stop on its next poll cycle."""
    _ensure_run_state_dir()
    _STREAM_STOP_FILE.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
    return True


def clear_stream_stop_request() -> None:
    """Remove stale stop requests so new stream runs do not auto-exit."""
    try:
        _STREAM_STOP_FILE.unlink(missing_ok=True)
    except Exception:
        log.debug("Failed to clear stream stop request", exc_info=True)


def request_stream_interrupt_signal() -> bool:
    """Try to send an interrupt signal to the active stream process, if known."""
    pid = _read_stream_pid()
    if not pid:
        return False

    try:
        if os.name == "nt":
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        else:
            os.kill(pid, signal.SIGINT)
        return True
    except Exception:
        log.debug("Failed to interrupt stream process pid=%s", pid, exc_info=True)
        return False


def _stop_requested(stop_event: threading.Event) -> bool:
    return stop_event.is_set() or _STREAM_STOP_FILE.exists()


def _wait_or_stop(stop_event: threading.Event, timeout: float) -> bool:
    """Wait up to timeout seconds, but wake early if an external stop is requested."""
    end = time.time() + max(0.0, timeout)
    while time.time() < end:
        if _stop_requested(stop_event):
            return True
        slice_s = min(0.5, max(0.0, end - time.time()))
        if stop_event.wait(timeout=slice_s):
            return True
    return _stop_requested(stop_event)


def _count_pending(stage: str, min_score: int | None = None, max_age_days: int | None = None) -> int:
    """Count pending work items for a stage, honoring min_score and max_age_days.

    "enrich"/"score"/"tailor"/"cover" delegate to `database.count_jobs_by_stage`
    (see `_CANONICAL_PENDING_STAGE`) so this can never again silently diverge
    from what the real stage runners (via `get_jobs_by_stage`) can select --
    e.g. a permanently `archived` job used to keep counting as pending forever
    here even though it could never be re-selected, which could stall
    `--stream` mode's zero-pending termination check indefinitely. "pdf" has
    no canonical equivalent and stays on the standalone `_PENDING_SQL` query.
    """
    from applypilot.config import DEFAULTS
    from applypilot.database import count_jobs_by_stage

    if min_score is None:
        min_score = DEFAULTS["min_score"]
    if max_age_days is None:
        max_age_days = DEFAULTS["max_job_age_days"]

    canonical_stage = _CANONICAL_PENDING_STAGE.get(stage)
    if canonical_stage is not None:
        return count_jobs_by_stage(get_connection(), canonical_stage, min_score=min_score, max_age_days=max_age_days)

    sql = _PENDING_SQL.get(stage)
    if sql is None:
        return 0

    params: list = []
    if max_age_days and max_age_days > 0:
        sql += " AND discovered_at > datetime('now', ?)"
        params.append(f"-{max_age_days} days")

    conn = get_connection()
    return conn.execute(sql, params).fetchone()[0]


# Keys summed to decide whether a streaming pass made real progress (vs. a
# cap-blocked/all-filtered no-op pass that should back off before retrying).
#
# 2026-08-30 fix: this list must cover every real runner's own success-count
# key(s), not just whichever ones happened to be added first. Confirmed by
# reading each runner's actual return dict: run_scoring returns "scored"
# (was missing -- every successful score pass in streaming mode logged
# "progress=0" and took a needless 10s backoff regardless of how many jobs
# were actually scored); run_cover_letters returns "generated"/"rejected"
# (also missing -- same bug for cover); run_tailoring's "approved"/"failed"
# were already covered. "rejected" counts as progress the same way "failed"
# already does for tailor -- a rejected-by-validation cover letter still
# consumed a real LLM call and produced a real terminal state
# (cover_failed), it did not "do nothing".
_PROGRESS_KEYS = (
    "approved",
    "failed",
    "errors",
    "processed",
    "ok",
    "partial",
    "error",
    "new",
    "scored",
    "generated",
    "rejected",
)


def _stage_progress(result: dict) -> int:
    """Sum of every recognized progress-count key present in a stage
    runner's result dict. 0 means the pass produced no real work (distinct
    from an error, which is checked separately by the caller)."""
    return sum(int(result.get(k, 0) or 0) for k in _PROGRESS_KEYS)


def _run_stage_streaming(
    stage: str,
    tracker: _StageTracker,
    stop_event: threading.Event,
    min_score: int | None = None,
    max_age_days: int | None = None,
    limit: int = 20,
    workers: int = 1,
    sources: list[str] | None = None,
    doc_format: str = "docx",
) -> None:
    """Run a single stage in streaming mode: loop until upstream done + no work.

    For discover: runs once, then marks done.
    For all others: polls DB for pending work, runs the batch processor,
    and repeats until upstream is done and no pending work remains.
    """
    from applypilot.config import DEFAULTS

    if min_score is None:
        min_score = DEFAULTS["min_score"]
    if max_age_days is None:
        max_age_days = DEFAULTS["max_job_age_days"]
    runner = _STAGE_RUNNERS[stage]
    kwargs: dict = {}
    if stage in ("tailor", "cover", "pdf"):
        kwargs["doc_format"] = doc_format
    if stage in ("score", "tailor", "cover"):
        kwargs["limit"] = limit
    if stage in ("tailor", "cover"):
        kwargs["min_score"] = min_score
    if stage in ("discover", "enrich", "score", "tailor", "cover"):
        kwargs["workers"] = workers
    if stage == "discover" and sources is not None:
        kwargs["sources"] = sources
    if stage in ("score", "tailor", "cover"):
        kwargs["max_age_days"] = max_age_days

    upstream = _UPSTREAM[stage]

    if stage == "discover":
        # Discover runs once (its sub-scrapers already do their full crawl)
        try:
            result = runner(**kwargs)
            tracker.mark_done(stage, result)
        except Exception as e:
            log.exception("Stage '%s' crashed", stage)
            tracker.mark_done(stage, {"status": f"error: {e}"})
        return

    # For downstream stages: loop until upstream done + no pending work
    passes = 0
    while not _stop_requested(stop_event):
        # Wait for upstream to start producing work (first pass only)
        if passes == 0 and upstream and not tracker.is_done(upstream):
            # Wait a bit for upstream to produce some work before first run
            _wait_or_stop(stop_event, _STREAM_POLL_INTERVAL)

        if _stop_requested(stop_event):
            break

        pending = _count_pending(stage, min_score, max_age_days)

        if pending > 0:
            try:
                result = runner(**kwargs)
                passes += 1
                # If runner returned an error status, back off before retry
                if isinstance(result, dict) and result.get("status", "").startswith("error"):
                    log.warning(
                        "Stage '%s' pass %d returned error, backing off %ds", stage, passes, _STREAM_POLL_INTERVAL
                    )
                    if _wait_or_stop(stop_event, _STREAM_POLL_INTERVAL):
                        break
                # Pending count looks like work but the runner did nothing
                # (cap-blocked, all candidates filtered, etc.). Back off so
                # we don't hot-loop logging the same "no jobs" message every
                # millisecond — let upstream produce real new work first.
                elif isinstance(result, dict):
                    progress = _stage_progress(result)
                    if progress == 0:
                        log.info(
                            "Stage '%s' pass %d had pending=%d but progress=0; backing off %ds",
                            stage,
                            passes,
                            pending,
                            _STREAM_POLL_INTERVAL,
                        )
                        if _wait_or_stop(stop_event, _STREAM_POLL_INTERVAL):
                            break
            except Exception:
                log.exception("Stage '%s' error (pass %d)", stage, passes)
                passes += 1
                if _wait_or_stop(stop_event, _STREAM_POLL_INTERVAL):
                    break
        else:
            # No work right now
            upstream_done = upstream is None or tracker.is_done(upstream)
            if upstream_done:
                # No work and upstream is done — this stage is finished
                break
            # Upstream still running, wait and retry
            if passes > 0 and passes % 6 == 0:
                log.info(
                    "Stage '%s' idle poll: no pending work yet, waiting for upstream '%s'",
                    stage,
                    upstream,
                )
            if _wait_or_stop(stop_event, _STREAM_POLL_INTERVAL):
                break  # Stop requested

    tracker.mark_done(stage, {"status": "ok", "passes": passes})


# ---------------------------------------------------------------------------
# Pipeline orchestrators
# ---------------------------------------------------------------------------


def _run_sequential(
    ordered: list[str],
    min_score: int,
    max_age_days: int | None = None,
    limit: int = 20,
    workers: int = 1,
    sources: list[str] | None = None,
    doc_format: str = "docx",
) -> dict:
    """Execute stages one at a time (original behavior)."""
    results: list[dict] = []
    errors: dict[str, str] = {}
    pipeline_start = time.time()
    # Batch identity carried from `score` to an immediately-following `tailor`
    # stage, and separately from `tailor` to an immediately-following `cover`
    # stage, in THIS SAME run -- see run_tailoring's / run_cover_letters's
    # job_ids parameters. Each stays None until its upstream stage actually
    # completes here, so a standalone `tailor`-only or `cover`-only run (no
    # `score`/`tailor` stage in `ordered`) is completely unaffected and keeps
    # its original independent-selection behavior. This is exactly the bug a
    # real 3-job run surfaced: cover had no batch of its own and silently
    # picked up whatever else was independently pending_cover-eligible.
    scored_job_ids: list[str] | None = None
    tailored_job_ids: list[str] | None = None

    for name in ordered:
        meta = STAGE_META[name]
        console.print(f"\n{'=' * 70}")
        console.print(f"  [bold]STAGE: {name}[/bold] — {meta['desc']}")
        console.print(f"  Started: {datetime.now(UTC).strftime('%H:%M:%S')}")
        console.print(f"{'=' * 70}")

        t0 = time.time()
        runner = _STAGE_RUNNERS[name]

        try:
            kwargs: dict = {}
            if name in ("tailor", "cover", "pdf"):
                kwargs["doc_format"] = doc_format
            if name in ("score", "tailor", "cover"):
                kwargs["limit"] = limit
            if name in ("tailor", "cover"):
                kwargs["min_score"] = min_score
            if name in ("discover", "enrich", "score", "tailor", "cover"):
                kwargs["workers"] = workers
            if name == "discover" and sources is not None:
                kwargs["sources"] = sources
            if name in ("score", "tailor", "cover"):
                kwargs["max_age_days"] = max_age_days
            if name == "tailor" and scored_job_ids is not None:
                kwargs["job_ids"] = scored_job_ids
            if name == "cover" and tailored_job_ids is not None:
                kwargs["job_ids"] = tailored_job_ids
            result = runner(**kwargs)
            elapsed = time.time() - t0

            if name == "score" and isinstance(result, dict) and "job_ids" in result:
                scored_job_ids = result["job_ids"]
            if name == "tailor" and isinstance(result, dict) and "job_ids" in result:
                tailored_job_ids = result["job_ids"]

            status = "ok"
            if isinstance(result, dict):
                status = result.get("status", "ok")
                if name == "discover":
                    sub_errors = [
                        f"{k}: {v}" for k, v in result.items() if isinstance(v, str) and v.startswith("error")
                    ]
                    if sub_errors:
                        status = "partial"

        except Exception as e:
            elapsed = time.time() - t0
            status = f"error: {e}"
            log.exception("Stage '%s' crashed", name)
            console.print(f"\n  [red]STAGE FAILED:[/red] {e}")

        results.append({"stage": name, "status": status, "elapsed": elapsed})
        if status not in ("ok", "partial"):
            errors[name] = status

        console.print(f"\n  Stage '{name}' completed in {elapsed:.1f}s — {status}")

    total_elapsed = time.time() - pipeline_start
    return {"stages": results, "errors": errors, "elapsed": total_elapsed}


def _run_streaming(
    ordered: list[str],
    min_score: int,
    max_age_days: int | None = None,
    limit: int = 20,
    workers: int = 1,
    sources: list[str] | None = None,
    doc_format: str = "docx",
) -> dict:
    """Execute stages concurrently with DB as conveyor belt."""
    tracker = _StageTracker()
    stop_event = threading.Event()
    pipeline_start = time.time()
    _write_stream_pid(os.getpid())

    console.print("\n  [bold cyan]STREAMING MODE[/bold cyan] — stages run concurrently")
    console.print(f"  Poll interval: {_STREAM_POLL_INTERVAL}s\n")
    console.print(f"  Stop flag:     {_STREAM_STOP_FILE}")
    console.print("  Stop command:  applypilot stop --stream\n")

    # Mark stages NOT in `ordered` as done so downstream doesn't wait for them
    for stage in STAGE_ORDER:
        if stage not in ordered:
            tracker.mark_done(stage, {"status": "skipped"})

    # Launch each stage in its own thread
    threads: dict[str, threading.Thread] = {}
    start_times: dict[str, float] = {}

    for name in ordered:
        start_times[name] = time.time()
        t = threading.Thread(
            target=_run_stage_streaming,
            args=(name, tracker, stop_event, min_score, max_age_days, limit, workers, sources, doc_format),
            name=f"stage-{name}",
            daemon=True,
        )
        threads[name] = t
        t.start()
        console.print(f"  [dim]Started thread:[/dim] {name}")

    # Wait for all threads to finish
    completed: set[str] = set()
    try:
        while len(completed) < len(ordered):
            if _stop_requested(stop_event):
                stop_event.set()
            for name in ordered:
                if name in completed:
                    continue
                t = threads[name]
                t.join(timeout=0.5)
                if not t.is_alive():
                    completed.add(name)
                    elapsed = time.time() - start_times[name]
                    console.print(f"  [green]Completed:[/green] {name} ({elapsed:.1f}s)")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — stopping stages...[/yellow]")
        stop_event.set()
    finally:
        stop_event.set()
        for t in threads.values():
            t.join(timeout=10)
        _clear_stream_pid()

    total_elapsed = time.time() - pipeline_start

    # Build results from tracker
    all_results = tracker.get_results()
    results: list[dict] = []
    errors: dict[str, str] = {}

    for name in ordered:
        r = all_results.get(name, {"status": "unknown"})
        elapsed = time.time() - start_times.get(name, pipeline_start)
        status = r.get("status", "ok")

        results.append({"stage": name, "status": status, "elapsed": elapsed})
        if status not in ("ok", "partial", "skipped"):
            errors[name] = status

    return {"stages": results, "errors": errors, "elapsed": total_elapsed}


def run_pipeline(
    stages: list[str] | None = None,
    min_score: int | None = None,
    max_age_days: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    stream: bool = False,
    workers: int = 1,
    sources: list[str] | None = None,
    doc_format: str = "docx",
) -> dict:
    """Run pipeline stages.

    Defaults for min_score and max_age_days read from config.DEFAULTS when None.

    Args:
        stages: List of stage names, or None / ["all"] for full pipeline.
        min_score: Minimum fit score for tailor/cover stages.
        max_age_days: Only process jobs discovered within this many days.
        limit: Max jobs per batch for score/tailor/cover stages. Default: 20.
        dry_run: If True, preview stages without executing.
        stream: If True, run stages concurrently (streaming mode).
        workers: Number of parallel threads for discovery/enrichment stages.
        sources: Discovery source names to run, or None for all.

    Returns:
        Dict with keys: stages (list of result dicts), errors (dict), elapsed (float).
    """
    from applypilot.config import DEFAULTS

    if min_score is None:
        min_score = DEFAULTS["min_score"]
    if max_age_days is None:
        max_age_days = DEFAULTS["max_job_age_days"]

    # Bootstrap
    load_env()
    ensure_dirs()
    init_db()

    # Resolve stages
    if stages is None:
        stages = ["all"]
    ordered = _resolve_stages(stages)
    effective_limit = limit if limit is not None else 20

    # Banner
    mode = "streaming" if stream else "sequential"
    console.print()
    console.print(
        Panel.fit(
            f"[bold]ApplyPilot Pipeline[/bold] ({mode})",
            border_style="blue",
        )
    )
    console.print(f"  Max age:   {max_age_days}d")
    console.print(f"  Limit:     {effective_limit} jobs/batch")
    console.print(f"  Workers:   {workers}")
    console.print(f"  Stages:    {' -> '.join(ordered)}")
    # 2026-08-30 fix: min_score is only ever consumed by _run_tailor/
    # _run_cover (pending_tailor/pending_cover's `fit_score >= ?` filter) --
    # _run_score has no min_score parameter at all and never applies one.
    # The banner used to print "Min score: N" unconditionally, which reads
    # as if it governs whatever stages were requested even for a
    # score-only run where it's a complete no-op. Only show it when a
    # stage that actually consumes it is part of this run.
    if _MIN_SCORE_STAGES & set(ordered):
        console.print(f"  Min score: {min_score}")
    if sources:
        console.print(f"  Sources:   {', '.join(sources)}")
    pre_stats = get_stats()
    console.print(f"  DB:        {pre_stats['total']} jobs, {pre_stats['pending_detail']} pending enrichment")

    if dry_run:
        console.print(f"\n  [yellow]DRY RUN[/yellow] — would execute ({mode}):")
        for name in ordered:
            meta = STAGE_META[name]
            console.print(f"    {name:<12s}  {meta['desc']}")
        console.print("\n  No changes made.")
        return {"stages": [], "errors": {}, "elapsed": 0.0}

    # Set up per-run file logging
    file_handler = _setup_file_logging(ordered)

    # Execute
    try:
        if stream:
            clear_stream_stop_request()
            result = _run_streaming(
                ordered,
                min_score,
                max_age_days=max_age_days,
                limit=effective_limit,
                workers=workers,
                sources=sources,
                doc_format=doc_format,
            )
        else:
            result = _run_sequential(
                ordered,
                min_score,
                max_age_days=max_age_days,
                limit=effective_limit,
                workers=workers,
                sources=sources,
                doc_format=doc_format,
            )
    finally:
        # Always remove file handler, even on crash
        if file_handler:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
        if stream:
            _clear_stream_pid()

    # Summary table
    console.print(f"\n{'=' * 70}")
    summary = Table(title="Pipeline Summary", show_header=True, header_style="bold")
    summary.add_column("Stage", style="bold")
    summary.add_column("Status")
    summary.add_column("Time", justify="right")

    for r in result["stages"]:
        elapsed_str = f"{r['elapsed']:.1f}s"
        status_display = r["status"][:30]
        if r["status"] == "ok":
            style = "green"
        elif r["status"] in ("partial", "skipped"):
            style = "yellow"
        else:
            style = "red"
        summary.add_row(r["stage"], f"[{style}]{status_display}[/{style}]", elapsed_str)

    summary.add_row("", "", "")
    summary.add_row("[bold]Total[/bold]", "", f"[bold]{result['elapsed']:.1f}s[/bold]")
    console.print(summary)

    # Final DB stats
    final = get_stats()
    console.print("\n  [bold]DB Final State:[/bold]")
    console.print(f"    Total jobs:     {final['total']}")
    console.print(f"    With desc:      {final['with_description']}")
    console.print(f"    Scored:         {final['scored']}")
    console.print(f"    Tailored:       {final['tailored']}")
    console.print(f"    Cover letters:  {final['with_cover_letter']}")
    console.print(f"    Ready to apply: {final['ready_to_apply']}")
    console.print(f"    Applied:        {final['applied']}")
    console.print(f"{'=' * 70}")

    if file_handler:
        console.print(f"  [dim]Log saved: {file_handler.baseFilename}[/dim]")
    console.print()

    return result
