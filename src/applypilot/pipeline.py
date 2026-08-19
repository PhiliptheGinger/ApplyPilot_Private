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
from datetime import datetime, timezone

UTC = timezone.utc

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
    stage_tag = "+".join(stages) if len(stages) <= 4 else f"{stages[0]}+{len(stages)-1}more"
    log_path = LOG_DIR / f"{ts}_{stage_tag}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(handler)
    log.info("Log file: %s", log_path)
    return handler


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_ORDER = ("discover", "enrich", "score", "tailor", "cover", "pdf")

STAGE_META: dict[str, dict] = {
    "discover": {"desc": "Job discovery (JobSpy + Workday + smart extract + HN)"},
    "enrich":   {"desc": "Detail enrichment (full descriptions + apply URLs)"},
    "score":    {"desc": "LLM scoring (fit 1-10)"},
    "tailor":   {"desc": "Resume tailoring (LLM + validation)"},
    "cover":    {"desc": "Cover letter generation"},
    "pdf":      {"desc": "PDF conversion (tailored resumes + cover letters)"},
}

# Upstream dependency: a stage only finishes when its upstream is done AND
# it has no remaining pending work.
_UPSTREAM: dict[str, str | None] = {
    "discover": None,
    "enrich":   "discover",
    "score":    "enrich",
    "tailor":   "score",
    "cover":    "tailor",
    "pdf":      "cover",
}


# ---------------------------------------------------------------------------
# Discovery sources
# ---------------------------------------------------------------------------

# Canonical name → description. Order determines default execution order.
DISCOVERY_SOURCES: dict[str, str] = {
    "jobspy":       "JobSpy aggregator (LinkedIn, Indeed, ZipRecruiter)",
    "linkedin":     "LinkedIn only (via JobSpy)",
    "indeed":       "Indeed only (via JobSpy)",
    "workday":      "Workday corporate career sites",
    "greenhouse":   "Greenhouse ATS public board API",
    "lever":        "Lever ATS public postings API",
    "ashby":        "Ashby ATS public posting API",
    "amazon":       "amazon.jobs public search API",
    "costco":       "careers.costco.com public API (Issaquah HQ)",
    "builtin":      "builtin.com (config-driven cities + categories)",
    "smartextract": "Smart extract (AI-powered scraping, incl. Dice via sites.yaml)",
    "hackernews":   "Hacker News 'Who is Hiring?' thread",
}

# Alias → canonical name for CLI convenience
_SOURCE_ALIASES: dict[str, str] = {
    "hn":    "hackernews",
    "smart": "smartextract",
    "dice":  "smartextract",  # dice is scraped via smartextract + sites.yaml
}

# Sources that are jobspy with a specific site filter
_JOBSPY_SITE_SOURCES: dict[str, list[str]] = {
    "linkedin": ["linkedin"],
    "indeed":   ["indeed"],
}


def resolve_source_names(names: list[str]) -> list[str]:
    """Resolve source aliases and validate names. Returns canonical names."""
    resolved = []
    for name in names:
        canonical = _SOURCE_ALIASES.get(name, name)
        if canonical not in DISCOVERY_SOURCES:
            valid = sorted(set(list(DISCOVERY_SOURCES.keys()) + list(_SOURCE_ALIASES.keys())))
            raise ValueError(
                f"Unknown discovery source: '{name}'. "
                f"Valid sources: {', '.join(valid)}"
            )
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
        run_enrichment(workers=workers)
        return {"status": "ok"}
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
        run_scoring(workers=workers, max_age_days=max_age_days, limit=limit)
        return {"status": "ok"}
    except Exception as e:
        log.exception("Scoring failed")
        return {"status": f"error: {e}"}


def _run_tailor(min_score: int | None = None, max_age_days: int | None = None,
                limit: int = 20, workers: int = 1, doc_format: str = "docx") -> dict:
    """Stage: Resume tailoring — generate tailored resumes for high-fit jobs."""
    from applypilot.config import DEFAULTS
    if min_score is None:
        min_score = DEFAULTS["min_score"]
    if max_age_days is None:
        max_age_days = DEFAULTS["max_job_age_days"]
    try:
        from applypilot.scoring.tailor import run_tailoring
        run_tailoring(min_score=min_score, max_age_days=max_age_days,
                      limit=limit, workers=workers, doc_format=doc_format)
        return {"status": "ok"}
    except Exception as e:
        log.exception("Tailoring failed")
        return {"status": f"error: {e}"}


def _run_cover(min_score: int | None = None, max_age_days: int | None = None,
               limit: int = 20, workers: int = 1, doc_format: str = "docx") -> dict:
    """Stage: Cover letter generation."""
    from applypilot.config import DEFAULTS
    if min_score is None:
        min_score = DEFAULTS["min_score"]
    if max_age_days is None:
        max_age_days = DEFAULTS["max_job_age_days"]
    try:
        from applypilot.scoring.cover_letter import run_cover_letters
        run_cover_letters(min_score=min_score, max_age_days=max_age_days,
                          limit=limit, workers=workers, doc_format=doc_format)
        return {"status": "ok"}
    except Exception as e:
        log.exception("Cover letter generation failed")
        return {"status": f"error: {e}"}


def _run_pdf(doc_format: str = "docx") -> dict:
    """Stage: Document conversion — convert tailored resumes and cover letters to PDF/DOCX."""
    try:
        from applypilot.scoring.pdf import batch_convert
        batch_convert(doc_format=doc_format)
        return {"status": "ok"}
    except Exception as e:
        log.exception("Document conversion failed")
        return {"status": f"error: {e}"}


# Map stage names to their runner functions
_STAGE_RUNNERS: dict[str, callable] = {
    "discover": _run_discover,
    "enrich":   _run_enrich,
    "score":    _run_score,
    "tailor":   _run_tailor,
    "cover":    _run_cover,
    "pdf":      _run_pdf,
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
            console.print(
                f"[red]Unknown stage:[/red] '{name}'. "
                f"Available: {', '.join(STAGE_ORDER)}, all"
            )
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
        self._events: dict[str, threading.Event] = {
            stage: threading.Event() for stage in STAGE_ORDER
        }
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


# SQL to count pending work for each stage.
# The `?` params are: (min_score, age_cutoff_iso_offset) when both present;
# scroll through _count_pending to see the binding order.
_PENDING_SQL: dict[str, str] = {
    "enrich": (
        "SELECT COUNT(*) FROM jobs "
        "WHERE detail_scraped_at IS NULL"
    ),
    "score":  (
        "SELECT COUNT(*) FROM jobs "
        "WHERE full_description IS NOT NULL AND fit_score IS NULL"
    ),
    "tailor": (
        "SELECT COUNT(*) FROM jobs WHERE fit_score >= ? "
        "AND full_description IS NOT NULL "
        "AND tailored_resume_path IS NULL "
        "AND COALESCE(tailor_attempts, 0) < 5"
    ),
    "cover": (
        "SELECT COUNT(*) FROM jobs WHERE fit_score >= ? "
        "AND tailored_resume_path IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        "AND COALESCE(cover_attempts, 0) < 5"
    ),
    "pdf": (
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL "
        "AND tailored_resume_path LIKE '%.txt'"
    ),
}

# Stages whose SQL takes a ? for min_score.
_PENDING_SQL_TAKES_MIN_SCORE = {"tailor", "cover"}

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
    except Exception:
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


def _count_pending(stage: str, min_score: int | None = None,
                   max_age_days: int | None = None) -> int:
    """Count pending work items for a stage, honoring min_score and max_age_days."""
    from applypilot.config import DEFAULTS

    if min_score is None:
        min_score = DEFAULTS["min_score"]
    if max_age_days is None:
        max_age_days = DEFAULTS["max_job_age_days"]

    sql = _PENDING_SQL.get(stage)
    if sql is None:
        return 0

    params: list = []
    if stage in _PENDING_SQL_TAKES_MIN_SCORE:
        params.append(min_score)

    if max_age_days and max_age_days > 0:
        sql += " AND discovered_at > datetime('now', ?)"
        params.append(f"-{max_age_days} days")

    conn = get_connection()
    return conn.execute(sql, params).fetchone()[0]


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
                    log.warning("Stage '%s' pass %d returned error, backing off %ds",
                                stage, passes, _STREAM_POLL_INTERVAL)
                    if _wait_or_stop(stop_event, _STREAM_POLL_INTERVAL):
                        break
                # Pending count looks like work but the runner did nothing
                # (cap-blocked, all candidates filtered, etc.). Back off so
                # we don't hot-loop logging the same "no jobs" message every
                # millisecond — let upstream produce real new work first.
                elif isinstance(result, dict):
                    progress = sum(int(result.get(k, 0) or 0) for k in
                                   ("approved", "failed", "errors", "processed",
                                    "ok", "partial", "error", "new"))
                    if progress == 0:
                        log.info(
                            "Stage '%s' pass %d had pending=%d but progress=0; backing off %ds",
                            stage, passes, pending, _STREAM_POLL_INTERVAL,
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
                    stage, upstream,
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
            if name in ("tailor", "cover"):
                kwargs["min_score"] = min_score
                kwargs["limit"] = limit
            if name in ("discover", "enrich", "score", "tailor", "cover"):
                kwargs["workers"] = workers
            if name == "discover" and sources is not None:
                kwargs["sources"] = sources
            if name in ("score", "tailor", "cover"):
                kwargs["max_age_days"] = max_age_days
            result = runner(**kwargs)
            elapsed = time.time() - t0

            status = "ok"
            if isinstance(result, dict):
                status = result.get("status", "ok")
                if name == "discover":
                    sub_errors = [
                        f"{k}: {v}" for k, v in result.items()
                        if isinstance(v, str) and v.startswith("error")
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


def _run_streaming(ordered: list[str], min_score: int,
                   max_age_days: int | None = None,
                   limit: int = 20, workers: int = 1,
                   sources: list[str] | None = None, doc_format: str = "docx") -> dict:
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
    console.print(Panel.fit(
        f"[bold]ApplyPilot Pipeline[/bold] ({mode})",
        border_style="blue",
    ))
    console.print(f"  Min score: {min_score}")
    console.print(f"  Max age:   {max_age_days}d")
    console.print(f"  Limit:     {effective_limit} jobs/batch")
    console.print(f"  Workers:   {workers}")
    console.print(f"  Stages:    {' -> '.join(ordered)}")
    if sources:
        console.print(f"  Sources:   {', '.join(sources)}")

    # Pre-run stats
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
            result = _run_streaming(ordered, min_score,
                                    max_age_days=max_age_days,
                                    limit=effective_limit, workers=workers,
                                    sources=sources, doc_format=doc_format)
        else:
            result = _run_sequential(ordered, min_score,
                                     max_age_days=max_age_days,
                                     limit=effective_limit, workers=workers,
                                     sources=sources, doc_format=doc_format)
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
