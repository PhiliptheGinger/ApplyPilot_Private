"""Continuous ApplyPilot scheduler (``applypilot run-continuous``).

Thin coordinating layer -- does NOT reimplement discover/enrich/score/
tailor/cover/apply. Every planning cycle re-queries the database and calls
back into the existing stage machinery (``pipeline.run_pipeline``) and the
existing apply worker loop (``apply.orchestrator.worker_loop``), so a newly
discovered high-scoring candidate enters the very next batch naturally --
the scheduler never holds its own stale candidate list, and candidate
ordering is untouched (``acquire_job``/``get_jobs_by_stage`` decide that,
same as every other caller).

Two pure, independently-tested pieces this module composes:
  * ``applypilot.claude_status`` -- Claude availability state machine.
  * ``plan_upstream_work`` (below) -- how much tailor/cover work is worth
    doing this cycle, given the ready-queue target, remaining time (if
    known), and recent measured throughput.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from applypilot import config
from applypilot.claude_status import (
    AUTH_FAILURE,
    AVAILABLE,
    EXHAUSTED_KNOWN_RESET,
    EXHAUSTED_UNKNOWN_RESET,
    TRANSIENT_ERROR,
    ClaudeAvailability,
    check_claude_availability,
    gate,
)
from applypilot.pipeline import _RUN_STATE_DIR

log = logging.getLogger(__name__)

_SCHEDULER_PID_FILE = _RUN_STATE_DIR / "scheduler.pid"

# Conservative fallback rates (jobs/minute) used only when fewer than
# MIN_OBSERVATIONS_FOR_MEASURED_RATE real transitions exist in the lookback
# window -- e.g. a fresh install with no history yet. These are deliberately
# low, documented placeholders, NOT measurements.
CONSERVATIVE_FALLBACK_RATES: dict[str, float] = {
    "tailor": 0.2,
    "cover": 0.2,
}
MIN_OBSERVATIONS_FOR_MEASURED_RATE = 3

# Unknown-reset probe backoff ladder (seconds): 5m -> 10m -> 20m -> 30m cap.
_UNKNOWN_RESET_BACKOFF_STEPS: tuple[int, ...] = (300, 600, 1200, 1800)
_TRANSIENT_ERROR_BACKOFF_SECONDS = 30


class SchedulerAlreadyRunning(RuntimeError):
    """Raised when a second `run-continuous` instance tries to start while
    another one (with a live PID) already holds the lock."""


# ---------------------------------------------------------------------------
# Durable run-level log (~/.applypilot/logs/continuous_YYYY-MM-DD_HH-MM-SS.log)
#
# Purely additive: attaches one extra FileHandler to a small, fixed set of
# already-existing loggers for the duration of one `run-continuous`
# invocation. Console output (root logger's own handler, set up by cli.py)
# and every existing log file -- pipeline.py's own per-stage-run log,
# apply's per-worker worker-N.log -- are completely untouched; this just
# also captures the scheduler's own decisions (and whatever the apply/
# claude_status loggers already emit) into one consolidated, durable place
# so a run can be reconstructed after the terminal is gone.
# ---------------------------------------------------------------------------

_CONTINUOUS_LOG_LOGGER_NAMES: tuple[str, ...] = (
    "applypilot.scheduler",
    "applypilot.claude_status",
    "applypilot.apply.orchestrator",
    "applypilot.apply.launcher",
    "applypilot.apply.hitl",
)


def _setup_continuous_file_logging() -> logging.FileHandler:
    """Create and attach the per-run continuous scheduler log file.

    Called exactly once per `run_continuous` invocation (after the
    single-instance lock is acquired), never per-cycle -- so repeated
    cycles within one run never attach duplicate handlers.

    Also ensures each instrumented logger's own level admits INFO records
    while attached (cli.py's logging.basicConfig(level=INFO) already makes
    this a no-op in normal `applypilot run-continuous` usage, since these
    loggers then inherit INFO from the root logger -- but a logger's own
    level otherwise defaults to NOTSET, which resolves to the stdlib
    default of WARNING for anything invoked without cli.py's setup, e.g.
    tests, silently dropping every decision record before it ever reaches
    the handler). Each logger's prior level is restored on teardown.
    """
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = config.LOG_DIR / f"continuous_{ts}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    prev_levels: dict[str, int] = {}
    for name in _CONTINUOUS_LOG_LOGGER_NAMES:
        logger_ = logging.getLogger(name)
        prev_levels[name] = logger_.level
        logger_.setLevel(logging.INFO)
        logger_.addHandler(handler)
    handler._applypilot_prev_levels = prev_levels  # read back on teardown

    log.info("Continuous scheduler run log: %s", log_path)
    return handler


def _teardown_continuous_file_logging(handler: logging.FileHandler) -> None:
    prev_levels: dict[str, int] = getattr(handler, "_applypilot_prev_levels", {})
    for name in _CONTINUOUS_LOG_LOGGER_NAMES:
        logger_ = logging.getLogger(name)
        logger_.removeHandler(handler)
        if name in prev_levels:
            logger_.setLevel(prev_levels[name])
    handler.close()


def _log_decision(logger_: logging.Logger, event: str, **fields) -> None:
    """Emit one machine-parseable decision record.

    Format: "DECISION <event> <json>" -- easy to grep (`grep "DECISION
    claude_availability"`) and easy to parse later (split on the first two
    spaces, json.loads the rest) without a logging framework or schema.
    Reserved for the specific decision points an operator needs to
    reconstruct after the fact (Claude state, ready-queue size, planned
    work, throughput, stage results) -- routine narration uses plain
    log.info() calls instead.
    """
    try:
        payload = json.dumps(fields, default=str, sort_keys=True)
    except TypeError:
        payload = str(fields)
    logger_.info("DECISION %s %s", event, payload)


@dataclass
class SchedulerConfig:
    ready_buffer: int = config.DEFAULTS["ready_buffer"]
    ready_buffer_unknown: int = config.DEFAULTS["ready_buffer_unknown"]
    poll_interval: int = config.DEFAULTS["poll_interval"]
    discover_interval: int = config.DEFAULTS["scheduler_discover_interval"]
    cache_max_age: float = config.DEFAULTS["scheduler_cache_max_age"]
    max_batch: int = config.DEFAULTS["scheduler_max_batch"]
    safety_margin: float = config.DEFAULTS["scheduler_safety_margin"]
    no_continuous_apply: bool = False
    min_score: int = config.DEFAULTS["min_score"]
    max_age_days: int = config.DEFAULTS["max_job_age_days"]
    doc_format: str = "docx"
    throughput_window_minutes: int = 60


# ---------------------------------------------------------------------------
# Single-instance protection (reuses the run-state/ directory + PID-file
# pattern already used by pipeline.py's --stream lock; a distinct file name
# so the two don't collide -- they protect different kinds of processes).
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not handle:
            return False

        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            ):
                return False

            return exit_code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    except Exception:
        log.debug("Failed to check whether PID %s is alive", pid, exc_info=True)
        return False


def acquire_scheduler_lock() -> None:
    """Claim the single-instance lock, or raise SchedulerAlreadyRunning.

    A PID file whose process is no longer alive is treated as stale and
    silently reclaimed (same convention as apply's own stale in_progress
    lock handling in acquire_job). `run_continuous` calls this exactly once
    at startup -- there's no legitimate re-entrant same-process call, so an
    existing live PID always blocks, even if it happens to be this process's
    own (which would only happen from a genuine double-invocation bug).
    """
    _RUN_STATE_DIR.mkdir(parents=True, exist_ok=True)
    if _SCHEDULER_PID_FILE.exists():
        try:
            existing_pid = int(_SCHEDULER_PID_FILE.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            existing_pid = None
        if existing_pid is not None and _pid_alive(existing_pid):
            raise SchedulerAlreadyRunning(
                f"A run-continuous scheduler is already running (pid {existing_pid}). "
                f"Stop that process first, or delete {_SCHEDULER_PID_FILE} if you're "
                f"certain it's stale."
            )
    _SCHEDULER_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def release_scheduler_lock() -> None:
    try:
        _SCHEDULER_PID_FILE.unlink(missing_ok=True)
    except Exception:
        log.debug("Failed to clear scheduler pid file", exc_info=True)


# ---------------------------------------------------------------------------
# Pure planning logic
# ---------------------------------------------------------------------------


def count_ready_to_apply(conn=None) -> int:
    from applypilot.database import get_connection

    if conn is None:
        conn = get_connection()
    row = conn.execute("SELECT COUNT(*) FROM jobs WHERE state = 'ready_to_apply'").fetchone()
    return row[0] if row else 0


def _rate_for(throughput: dict, stage: str) -> float:
    if throughput.get("count", 0) >= MIN_OBSERVATIONS_FOR_MEASURED_RATE:
        return throughput["rate_per_minute"]
    return CONSERVATIVE_FALLBACK_RATES[stage]


def plan_upstream_work(
    *,
    ready: int,
    target: int,
    minutes_available: float | None,
    max_batch: int,
    safety_margin: float,
    tailor_throughput: dict,
    cover_throughput: dict,
) -> dict:
    """How much tailor/cover work to produce this cycle.

    Bounded by (a) how many candidates are actually needed to reach
    `target`, (b) estimated throughput capacity within `minutes_available`
    (None = not time-bounded, e.g. Claude currently AVAILABLE or reset time
    unknown), and (c) `max_batch` as a hard ceiling regardless of how much
    time or backlog exists. Never produces work merely because capacity or
    time exists -- `need` always wins when it's the smallest bound.
    """
    need = max(0, target - ready)
    if need <= 0:
        return {"tailor": 0, "cover": 0, "need": 0}

    tailor_rate = _rate_for(tailor_throughput, "tailor")
    cover_rate = _rate_for(cover_throughput, "cover")

    if minutes_available is None:
        tailor_capacity = max_batch
        cover_capacity = max_batch
    else:
        minutes_available = max(0.0, minutes_available)
        tailor_capacity = int(minutes_available * tailor_rate * safety_margin)
        cover_capacity = int(minutes_available * cover_rate * safety_margin)

    planned = max(0, min(need, max_batch, tailor_capacity, cover_capacity))
    return {"tailor": planned, "cover": planned, "need": need}


def _target_for_state(state: str, cfg: SchedulerConfig) -> int:
    return cfg.ready_buffer if state == AVAILABLE else cfg.ready_buffer_unknown


def _api_capacity_available() -> bool:
    """Cheap, read-only probe: does the tailor/cover LLM fallback chain
    currently have at least one non-exhausted entry? Used only to decide
    whether it's worth attempting bounded upstream work while Claude itself
    is unavailable -- Claude is reserved away from this chain by default
    (APPLYPILOT_RESERVE_CLAUDE_FOR_APPLY), so this is really asking about
    Gemini/OpenAI, not Claude.
    """
    try:
        from applypilot.llm import get_stage_client

        client = get_stage_client("tailor", quality=True)
        return client.has_available_model()
    except Exception:
        log.debug("API capacity probe failed; assuming available", exc_info=True)
        return True


# ---------------------------------------------------------------------------
# One planning + execution cycle
# ---------------------------------------------------------------------------


def run_once(
    cfg: SchedulerConfig,
    *,
    conn=None,
    run_pipeline_fn=None,
    availability_fn=None,
    api_capacity_fn=None,
    now: datetime | None = None,
    last_discover_at: datetime | None = None,
) -> dict:
    """Run exactly one planning+execution cycle. Never loops or sleeps.

    All external calls are injectable (run_pipeline_fn/availability_fn/
    api_capacity_fn) so this is fully testable without hitting real APIs,
    the real ~/.claude.json, or hardware.
    """
    from applypilot import pipeline as pipeline_mod
    from applypilot.database import get_connection, get_recent_stage_throughput

    run_pipeline_fn = run_pipeline_fn or pipeline_mod.run_pipeline
    availability_fn = availability_fn or check_claude_availability
    api_capacity_fn = api_capacity_fn or _api_capacity_available
    now = now or datetime.now(UTC)

    discover_due = last_discover_at is None or (now - last_discover_at).total_seconds() >= cfg.discover_interval

    _log_decision(log, "cycle_start", now=now.isoformat())

    avail: ClaudeAvailability = availability_fn(
        cache_max_age_seconds=cfg.cache_max_age,
        now=now,
    )
    gate.set(avail)
    _log_decision(
        log,
        "claude_availability",
        state=avail.state,
        detail=avail.detail,
        reset_estimate=(avail.reset_estimate.isoformat() if avail.reset_estimate else None),
        reset_source=avail.reset_source,
        binding_window=avail.binding_window,
        cache_age_seconds=avail.cache_age_seconds,
    )
    _log_decision(
        log,
        "apply_gate",
        paused=gate.is_paused(),
        state=avail.state,
    )

    if conn is None:
        conn = get_connection()

    ready = count_ready_to_apply(conn)

    # Discovery is deliberately throttled because a full crawl can take hours.
    discover_result = None
    discovery_ran = False

    if discover_due:
        discover_result = run_pipeline_fn(stages=["discover"])
        discovery_ran = True
        last_discover_at = now
        _log_decision(log, "discover_result", result=discover_result)
    else:
        _log_decision(
            log,
            "discover_skipped",
            reason="interval",
            last_discover_at=(last_discover_at.isoformat() if last_discover_at is not None else None),
            interval_seconds=cfg.discover_interval,
        )

    enrich_result = run_pipeline_fn(stages=["enrich"])
    _log_decision(log, "enrich_result", result=enrich_result)

    pending_score = pipeline_mod._count_pending(
        "score",
        min_score=cfg.min_score,
        max_age_days=cfg.max_age_days,
    )
    score_limit = min(cfg.max_batch, pending_score)
    _log_decision(
        log,
        "score_planned",
        limit=score_limit,
        pending=pending_score,
    )

    if score_limit > 0:
        score_result = run_pipeline_fn(
            stages=["score"],
            limit=score_limit,
            min_score=cfg.min_score,
            max_age_days=cfg.max_age_days,
        )
        _log_decision(
            log,
            "score_result",
            limit=score_limit,
            result=score_result,
        )

    target = _target_for_state(avail.state, cfg)
    minutes_available = None

    if avail.state == EXHAUSTED_KNOWN_RESET and avail.reset_estimate is not None:
        minutes_available = max(
            0.0,
            (avail.reset_estimate - now).total_seconds() / 60.0,
        )

    _log_decision(
        log,
        "ready_queue",
        ready=ready,
        target=target,
        minutes_available=minutes_available,
    )

    tailor_throughput = get_recent_stage_throughput(
        "tailor",
        cfg.throughput_window_minutes,
        conn=conn,
    )
    cover_throughput = get_recent_stage_throughput(
        "cover",
        cfg.throughput_window_minutes,
        conn=conn,
    )

    # AVAILABLE / EXHAUSTED_KNOWN_RESET: proceed normally.
    # Anything else (unknown reset, auth failure, transient error):
    # only bother with extra tailor/cover work if the non-Claude
    # providers actually have capacity.
    do_upstream = avail.state in (AVAILABLE, EXHAUSTED_KNOWN_RESET) or api_capacity_fn()

    if do_upstream:
        plan = plan_upstream_work(
            ready=ready,
            target=target,
            minutes_available=minutes_available,
            max_batch=cfg.max_batch,
            safety_margin=cfg.safety_margin,
            tailor_throughput=tailor_throughput,
            cover_throughput=cover_throughput,
        )
    else:
        plan = {
            "tailor": 0,
            "cover": 0,
            "need": max(0, target - ready),
        }

    _log_decision(
        log,
        "upstream_plan",
        do_upstream=do_upstream,
        ready=ready,
        target=target,
        minutes_available=minutes_available,
        plan=plan,
        tailor_throughput=tailor_throughput,
        cover_throughput=cover_throughput,
    )

    if plan["tailor"] > 0:
        tailor_result = run_pipeline_fn(
            stages=["tailor"],
            limit=plan["tailor"],
            min_score=cfg.min_score,
            max_age_days=cfg.max_age_days,
            doc_format=cfg.doc_format,
        )
        _log_decision(
            log,
            "tailor_result",
            limit=plan["tailor"],
            result=tailor_result,
        )

    if plan["cover"] > 0:
        cover_result = run_pipeline_fn(
            stages=["cover"],
            limit=plan["cover"],
            min_score=cfg.min_score,
            max_age_days=cfg.max_age_days,
            doc_format=cfg.doc_format,
        )
        _log_decision(
            log,
            "cover_result",
            limit=plan["cover"],
            result=cover_result,
        )

    result = {
        "availability": avail,
        "ready": ready,
        "target": target,
        "last_discover_at": last_discover_at,
        "discovery_ran": discovery_ran,
        "minutes_available": minutes_available,
        "plan": plan,
        "score_limit": score_limit,
        "pending_score": pending_score,
        "tailor_throughput": tailor_throughput,
        "cover_throughput": cover_throughput,
    }

    _log_decision(
        log,
        "cycle_end",
        ready=ready,
        target=target,
        plan=plan,
        claude_state=avail.state,
    )
    return result


def format_status_block(result: dict) -> str:
    avail: ClaudeAvailability = result["availability"]
    lines = [f"Claude status: {avail.state}"]

    if avail.state == EXHAUSTED_KNOWN_RESET and avail.reset_estimate is not None:
        mins = result.get("minutes_available")
        mins_str = f"{mins:.0f}m" if mins is not None else "?"
        age = avail.cache_age_seconds
        age_str = f"fetched {age / 60:.0f}m ago" if age is not None else "fetch time unknown"
        lines.append(
            f"Estimated reset: {mins_str} (source: cached utilization, {age_str} -- best-effort, not authoritative)"
        )
        lines.append(f"Binding window: {avail.binding_window}")
    elif avail.state == EXHAUSTED_UNKNOWN_RESET:
        lines.append("Reset estimate: unavailable")
        lines.append(f"Reason: {avail.detail}")
    elif avail.state in (AUTH_FAILURE, TRANSIENT_ERROR):
        lines.append(f"Reason: {avail.detail}")

    lines.append(f"Ready queue: {result['ready']} / target {result['target']}")
    tt, ct = result["tailor_throughput"], result["cover_throughput"]
    lines.append(
        f"Recent throughput: tailor {tt['rate_per_minute']:.2f}/min (n={tt['count']}), "
        f"cover {ct['rate_per_minute']:.2f}/min (n={ct['count']})"
    )
    plan = result["plan"]
    lines.append(
        f"Planned work: score up to {result['score_limit']} "
        f"(pending {result['pending_score']}), "
        f"tailor up to {plan['tailor']}, cover up to {plan['cover']}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The continuous loop
# ---------------------------------------------------------------------------


class _UnknownResetBackoff:
    def __init__(self) -> None:
        self._idx = 0

    def seconds(self) -> int:
        return _UNKNOWN_RESET_BACKOFF_STEPS[min(self._idx, len(_UNKNOWN_RESET_BACKOFF_STEPS) - 1)]

    def advance(self) -> None:
        self._idx = min(self._idx + 1, len(_UNKNOWN_RESET_BACKOFF_STEPS) - 1)

    def reset(self) -> None:
        self._idx = 0


def _next_sleep_seconds(result: dict, cfg: SchedulerConfig, backoff: _UnknownResetBackoff) -> float:
    state = result["availability"].state
    if state == AVAILABLE:
        backoff.reset()
        return cfg.poll_interval
    if state == EXHAUSTED_KNOWN_RESET:
        backoff.reset()
        mins = result.get("minutes_available")
        if mins is not None and mins > 0:
            return min(cfg.poll_interval, mins * 60)
        return cfg.poll_interval
    if state == EXHAUSTED_UNKNOWN_RESET:
        seconds = backoff.seconds()
        backoff.advance()
        return seconds
    if state == TRANSIENT_ERROR:
        return min(cfg.poll_interval, _TRANSIENT_ERROR_BACKOFF_SECONDS)
    # AUTH_FAILURE: claude auth status is already internally rate-limited,
    # so hammering isn't possible either way -- just use the normal interval.
    return cfg.poll_interval


def run_continuous(cfg: SchedulerConfig) -> None:
    """Run the scheduler loop until Ctrl+C. Single-process design: also
    starts the continuous apply worker (unless cfg.no_continuous_apply) in
    a background thread, gated by `claude_status.gate` on every cycle.
    """
    acquire_scheduler_lock()
    log_handler: logging.FileHandler | None = None
    apply_thread: threading.Thread | None = None
    try:
        log_handler = _setup_continuous_file_logging()
        log.info("run-continuous starting (pid=%s) | config=%s", os.getpid(), asdict(cfg))

        from applypilot.apply.launcher import _stop_event as apply_stop_event

        apply_stop_event.clear()

        if not cfg.no_continuous_apply:
            from applypilot.apply.orchestrator import worker_loop

            apply_thread = threading.Thread(
                target=worker_loop,
                kwargs={"worker_id": 0, "limit": 0, "min_score": cfg.min_score, "max_age_days": cfg.max_age_days},
                daemon=True,
                name="scheduler-apply-worker",
            )
            apply_thread.start()
            log.info(
                "Continuous apply worker thread started (worker_id=0). Per-job detail: %s",
                config.LOG_DIR / "worker-0.log",
            )
        else:
            log.info("Continuous apply worker NOT started (--no-continuous-apply)")

        backoff = _UnknownResetBackoff()
        last_discover_at = None
        cycle = 0
        while True:
            cycle += 1
            log.info("Cycle %d starting", cycle)
            result = run_once(
                cfg,
                last_discover_at=last_discover_at,
            )
            last_discover_at = result["last_discover_at"]
            log.info("Cycle %d finished\n%s", cycle, format_status_block(result))
            sleep_s = _next_sleep_seconds(result, cfg, backoff)
            log.info("Cycle %d: sleeping %.0fs until next cycle", cycle, sleep_s)
            time.sleep(max(1.0, sleep_s))
    except KeyboardInterrupt:
        log.info("run-continuous: stopped by user (KeyboardInterrupt)")
    except Exception:
        log.exception("run-continuous: crashed with an unhandled exception")
        raise
    finally:
        from applypilot.apply.launcher import _stop_event as apply_stop_event

        apply_stop_event.set()
        if apply_thread is not None:
            log.info("Waiting for apply worker thread to stop...")
            apply_thread.join(timeout=30)
            log.info(
                "Apply worker thread stopped: %s", "alive (join timed out)" if apply_thread.is_alive() else "clean"
            )
        release_scheduler_lock()
        if log_handler is not None:
            log.info("run-continuous: shutdown complete (pid=%s)", os.getpid())
            _teardown_continuous_file_logging(log_handler)
