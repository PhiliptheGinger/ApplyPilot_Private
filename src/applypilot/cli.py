"""ApplyPilot CLI — the main entry point."""

from __future__ import annotations

import logging
import os
import re

import typer
from rich.console import Console
from rich.table import Table

from applypilot import __version__, config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
# The handler basicConfig() just attached to the root logger -- kept so
# --quiet can raise its level without touching the root logger's own
# level (which the per-run FileHandler in pipeline.py relies on staying
# at INFO so the full detail still lands in ~/.applypilot/logs/).
_console_handler = logging.getLogger().handlers[0]

# httpx/httpcore/openai's HTTP client logs every single request at INFO
# ("HTTP Request: POST ... 200 OK"), which drowns out our own per-job
# progress lines on any run of more than a few dozen jobs. These libraries
# don't carry information we act on -- our own request/retry logging
# (llm.py) already covers failures -- so keep them at WARNING.
for _noisy_logger in ("httpx", "httpcore", "openai"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

app = typer.Typer(
    name="applypilot",
    help="AI-powered end-to-end job application pipeline.",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "score", "tailor", "cover", "pdf")

DEFAULT_TITLE_REJECT_PATTERNS = (
    r"\bsenior\b",
    r"\bsr\.?\b",
    r"\bstaff\b",
    r"\bprincipal\b",
    r"\blead\b",
    r"\bmanager\b",
    r"\bhead\b",
    r"\bdirector\b",
    r"\b(?:vp|vice president)\b",
    r"\bchief\b",
    r"\barchitect\b",
    r"\bdistinguished\b",
    r"\bfellow\b",
)


def _resolve_title_reject_patterns(
    *,
    use_defaults: bool,
    use_config: bool,
    cli_patterns: list[str] | None,
) -> list[str]:
    """Build title-reject regex pattern list from defaults, config, and CLI.

    Priority order: defaults -> searches.yaml -> CLI overrides (append).
    Duplicates are removed while preserving order.
    """
    patterns: list[str] = []
    if use_defaults:
        patterns.extend(DEFAULT_TITLE_REJECT_PATTERNS)

    if use_config:
        cfg = config.load_search_config() or {}
        from_cfg = cfg.get("title_reject_patterns")
        if from_cfg is None and isinstance(cfg.get("filters"), dict):
            from_cfg = cfg["filters"].get("title_reject_patterns")
        if isinstance(from_cfg, list):
            patterns.extend(str(p) for p in from_cfg if str(p).strip())
        exclude_titles = cfg.get("exclude_titles")
        if isinstance(exclude_titles, list):
            patterns.extend(re.escape(str(p).strip()) for p in exclude_titles if str(p).strip())

    if cli_patterns:
        patterns.extend(p for p in cli_patterns if (p or "").strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for p in patterns:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applypilot.config import ensure_dirs, load_env
    from applypilot.database import init_db

    load_env()
    ensure_dirs()
    init_db()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applypilot[/bold] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyPilot — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applypilot.wizard.init import run_wizard

    run_wizard()


@app.command()
def run(
    stages: list[str] | None = typer.Argument(
        None,
        help=(f"Pipeline stages to run. Valid: {', '.join(VALID_STAGES)}, all. Defaults to 'all' if omitted."),
    ),
    min_score: int = typer.Option(
        config.DEFAULTS["min_score"],
        "--min-score",
        help=f"Minimum fit score for tailor/cover stages (default: {config.DEFAULTS['min_score']}).",
    ),
    max_age_days: int = typer.Option(
        config.DEFAULTS["max_job_age_days"],
        "--max-age-days",
        help=(
            "Skip jobs whose discovered_at is older than this many days. "
            "0 = no age filter. "
            f"Default: {config.DEFAULTS['max_job_age_days']}."
        ),
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Max jobs per stage (score/tailor/cover). Default: 20."
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Parallel threads for Workday/smart-extract stages. (JobSpy runs sequentially regardless.)",
    ),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    doc_format: str = typer.Option(
        "docx", "--doc-format", help="Document format for resumes/cover letters: docx (default) or pdf."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    source: list[str] | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Discovery source(s) to run. Repeatable: --source hn --source jobspy. "
        "Aliases: hn=hackernews, smart=smartextract. Only affects the discover stage.",
    ),
    list_sources: bool = typer.Option(
        False,
        "--list-sources",
        help="List available discovery sources and exit.",
    ),
    auto_reject_titles: bool = typer.Option(
        False,
        "--auto-reject-titles",
        help="Before running stages, bulk-archive title-pattern matches (senior/staff/lead/etc).",
    ),
    reject_pattern: list[str] | None = typer.Option(
        None,
        "--reject-pattern",
        help="Regex title reject pattern. Repeatable. Used with --auto-reject-titles.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help=(
            "Suppress per-job/HTTP console logging (stage banners and the final "
            "summary still print). Full detail still goes to the per-run log "
            "file under ~/.applypilot/logs/ -- use that for after-the-fact review."
        ),
    ),
    local_first: bool = typer.Option(
        False,
        "--local-first",
        help=(
            "Before the cloud tailoring call, ask the local model "
            "(APPLYPILOT_LOCAL_LLM_URL) for a cheap tailoring plan that guides "
            "the cloud model's output.  Requires APPLYPILOT_LOCAL_LLM_URL to be set."
        ),
    ),
) -> None:
    """Run pipeline stages: discover, enrich, score, tailor, cover, pdf."""
    if quiet:
        _console_handler.setLevel(logging.WARNING)
    if local_first:
        # Signal to run_tailoring / _tailor_one_job via env var so the flag
        # threads through the pipeline without changing any function signatures.
        os.environ["APPLYPILOT_LOCAL_PLAN"] = "1"
    # Handle --list-sources before bootstrap (no DB/env needed)
    if list_sources:
        from applypilot.pipeline import _SOURCE_ALIASES, DISCOVERY_SOURCES

        console.print("\n[bold]Available discovery sources:[/bold]\n")
        for name, desc in DISCOVERY_SOURCES.items():
            aliases = [a for a, canon in _SOURCE_ALIASES.items() if canon == name]
            alias_str = f"  (alias: {', '.join(aliases)})" if aliases else ""
            console.print(f"  [cyan]{name:<14s}[/cyan] {desc}{alias_str}")
        console.print("\nUsage: applypilot run discover --source hn --source jobspy")
        raise typer.Exit()

    _bootstrap()

    from applypilot.pipeline import resolve_source_names, run_pipeline

    stage_list = stages if stages else ["all"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(f"[red]Unknown stage:[/red] '{s}'. Valid stages: {', '.join(VALID_STAGES)}, all")
            raise typer.Exit(code=1)

    # Resolve --source aliases
    resolved_sources: list[str] | None = None
    if source:
        try:
            resolved_sources = resolve_source_names(source)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(code=1)

    if auto_reject_titles:
        from applypilot.database import reject_jobs_by_title_patterns

        patterns = _resolve_title_reject_patterns(
            use_defaults=True,
            use_config=True,
            cli_patterns=reject_pattern,
        )
        if not patterns:
            console.print("[yellow]Title reject pass requested but no patterns were configured.[/yellow]")
        else:
            res = reject_jobs_by_title_patterns(patterns, dry_run=dry_run, sample_limit=5)
            verb = "would archive" if dry_run else "archived"
            console.print(f"[cyan]Title reject pass:[/cyan] {verb} {res['updated']} / {res['matched']} matched job(s)")

    # Validate --doc-format
    from applypilot.scoring.pdf import VALID_DOC_FORMATS

    if doc_format not in VALID_DOC_FORMATS:
        console.print(
            f"[red]Invalid --doc-format:[/red] '{doc_format}'. Must be one of: {', '.join(VALID_DOC_FORMATS)}"
        )
        raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applypilot.config import check_tier

        check_tier(2, "AI scoring/tailoring")

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        max_age_days=max_age_days,
        limit=limit,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
        sources=resolved_sources,
        doc_format=doc_format,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command("test-local")
def test_local_cmd() -> None:
    """Test connectivity to the configured local LLM (APPLYPILOT_LOCAL_LLM_URL).

    Probes the endpoint and sends a minimal chat request.
    Exit code 0 = working, 1 = not configured or unreachable.
    """
    _bootstrap()
    from applypilot.llm import (
        LLMClient,
        ModelEntry,
        is_local_configured,
        local_available,
        local_openai_base_url,
    )

    if not is_local_configured():
        console.print(
            "[yellow]No local LLM configured.[/yellow]  "
            "Set [bold]APPLYPILOT_LOCAL_LLM_URL[/bold] in "
            "[dim]~/.applypilot/.env[/dim] or the environment.\n"
            "Example (Ollama default): APPLYPILOT_LOCAL_LLM_URL=http://localhost:11434 "
            "(with or without a trailing /v1 -- both work)"
        )
        raise typer.Exit(code=1)

    url = os.environ.get("APPLYPILOT_LOCAL_LLM_URL", "").rstrip("/")
    model = os.environ.get("APPLYPILOT_LOCAL_LLM_MODEL", "llama3.2")
    console.print(f"  URL:   [cyan]{url}[/cyan]")
    console.print(f"  Model: [cyan]{model}[/cyan]")

    if not local_available():
        console.print(f"[red]Endpoint unreachable:[/red] {url}")
        raise typer.Exit(code=1)

    console.print("[green]Endpoint reachable.[/green]  Sending chat probe...")
    try:
        # Empty api_key is intentional — Ollama ignores auth
        client = LLMClient(url, model, "", quality=False)
        # This command's whole purpose is to test THIS local endpoint --
        # pin the chat probe to only the local entry so a configured cloud
        # key (e.g. GEMINI_API_KEY) can never mask a broken/unreachable
        # local model behind a "working" cloud fallback response.
        #
        # LLMClient is OpenAI-compatible-only (always posts to
        # {base_url}/chat/completions), so the entry's base_url must be
        # the /v1 root regardless of whether the user configured the bare
        # server root or already included /v1 -- see local_openai_base_url.
        client._fallback_chain = [ModelEntry(model, "local", local_openai_base_url(url), "")]
        reply = client.chat(
            [{"role": "user", "content": 'Reply with exactly: {"status":"ok"}'}],
            max_tokens=30,
        )
        console.print(f"[dim]Response: {reply[:80]}[/dim]")
        console.print("[green]Local LLM is working.[/green]")
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic probe of an unreliable local LLM endpoint; must degrade to a friendly error + exit code, not an unhandled traceback
        console.print(f"[yellow]Chat probe failed (endpoint reachable but model may not be loaded): {exc}[/yellow]")
        raise typer.Exit(code=1)


@app.command("debug-local-plan")
def debug_local_plan_cmd(
    url: str | None = typer.Option(None, "--url", help="Job URL (or a partial match) to debug."),
    job_id: int | None = typer.Option(
        None,
        "--job-id",
        help="Job's database row id (see the leftmost column in e.g. `applypilot status`-adjacent "
        "SQL, or note it down as jobs are discovered) -- lets you debug a stored job without "
        "retrieving its original URL.",
    ),
) -> None:
    """Show the local model's tailoring plan for one job -- WITHOUT
    generating or modifying any resume.

    Displays the deterministic evidence-retrieval trace (which profile
    items matched, why, and the auto-extracted requirement lines) alongside
    the local model's structured plan, so you can judge whether the local
    stage is useful before spending a cloud tailoring call on it.

    Exactly one of --url or --job-id must be given. Requires
    APPLYPILOT_LOCAL_LLM_URL to be set.
    """
    # Validate args before any bootstrap/IO so a bad invocation fails fast.
    if (url is None) == (job_id is None):
        console.print(
            "[red]Provide exactly one of[/red] [bold]--url[/bold] [red]or[/red] "
            "[bold]--job-id[/bold] [red](not both, not neither).[/red]"
        )
        raise typer.Exit(code=1)

    _bootstrap()
    from applypilot.config import load_profile
    from applypilot.database import get_connection
    from applypilot.llm import is_local_configured
    from applypilot.scoring.local_tailor import debug_plan_for_job

    if not is_local_configured():
        console.print(
            "[yellow]No local LLM configured.[/yellow]  "
            "Set [bold]APPLYPILOT_LOCAL_LLM_URL[/bold] first (see [bold]applypilot test-local[/bold])."
        )
        raise typer.Exit(code=1)

    conn = get_connection()
    select_cols = "rowid, url, title, site, application_url, full_description, fit_score, company"
    if job_id is not None:
        row = conn.execute(
            f"SELECT {select_cols} FROM jobs WHERE rowid = ?",
            (job_id,),
        ).fetchone()
        not_found_desc = f"--job-id {job_id}"
    else:
        like = f"%{url.split('?')[0].rstrip('/')}%"
        row = conn.execute(
            f"SELECT {select_cols} FROM jobs WHERE url = ? OR url LIKE ? ORDER BY discovered_at DESC LIMIT 1",
            (url, like),
        ).fetchone()
        not_found_desc = f"--url {url}"
    if not row:
        console.print(f"[red]No job found matching:[/red] {not_found_desc}")
        raise typer.Exit(code=1)
    job = dict(row)

    profile = load_profile()
    console.print(
        f"\n[bold blue]Local plan debug:[/bold blue] {job.get('title')} @ {job.get('site')} "
        f"[dim](--job-id {job.get('rowid')})[/dim]"
    )
    console.print(f"[dim]{job.get('url')}[/dim]\n")

    result = debug_plan_for_job(job, profile)
    if result is None:
        console.print(
            "[red]Local planning failed[/red] -- model unreachable, timed out, empty "
            "response, or output wasn't parseable JSON (see the warning logged above "
            "for the specific cause). Try [bold]applypilot test-local[/bold] to check "
            "connectivity, or raise [bold]APPLYPILOT_LOCAL_LLM_TIMEOUT[/bold] if the "
            "model is just slow."
        )
        raise typer.Exit(code=1)

    plan = result["plan"]
    evidence = result["evidence"]
    req_lines = result["requirement_lines"]

    if req_lines:
        t = Table(title="Auto-extracted requirement lines (deterministic)", show_header=True, header_style="bold cyan")
        t.add_column("Importance")
        t.add_column("Text", overflow="fold")
        for l in req_lines:
            t.add_row(l["importance"], l["text"])
        console.print(t)
    else:
        console.print("[dim]No bullet/numbered requirement lines detected in the description.[/dim]")

    if evidence:
        t = Table(
            title="Retrieved profile evidence (deterministic, no LLM)", show_header=True, header_style="bold cyan"
        )
        t.add_column("Type")
        t.add_column("Item")
        t.add_column("Score", justify="right")
        t.add_column("Matched job terms/categories", overflow="fold")
        for e in evidence:
            t.add_row(e["type"], e["name"], str(e["score"]), ", ".join(e["matched_terms"]))
        console.print(t)
    else:
        console.print("[dim]No profile evidence (experience/project/skill) matched this job.[/dim]")

    if plan["requirements"]:
        t = Table(title="Requirements (from local model)", show_header=True, header_style="bold cyan")
        t.add_column("Importance")
        t.add_column("Requirement", overflow="fold")
        t.add_column("Supported")
        t.add_column("Evidence", overflow="fold")
        for r in plan["requirements"]:
            t.add_row(
                r["importance"],
                r["requirement"],
                "yes" if r["supported"] else "no",
                "; ".join(r["resume_evidence"]),
            )
        console.print(t)

    if plan["unsupported_requirements"]:
        console.print("\n[yellow]Unsupported requirements (no fabricated evidence):[/yellow]")
        for u in plan["unsupported_requirements"]:
            console.print(f"  - {u}")

    if plan["bullets_to_prioritize"]:
        console.print("\n[green]Bullets to prioritize:[/green]")
        for b in plan["bullets_to_prioritize"]:
            console.print(f"  - {b}")

    if plan["bullets_to_deemphasize"]:
        console.print("\n[dim]Bullets to de-emphasize:[/dim]")
        for b in plan["bullets_to_deemphasize"]:
            console.print(f"  - {b}")

    if plan["safe_rewrites"]:
        console.print("\n[cyan]Proposed safe rewrites:[/cyan]")
        for rw in plan["safe_rewrites"]:
            console.print(f'  - "{rw["original"]}" -> "{rw["suggested"]}"')

    if plan["_warnings"]:
        console.print("\n[dim]Dropped by grounding validation:[/dim]")
        for w in plan["_warnings"]:
            console.print(f"  [dim]- {w}[/dim]")


@app.command()
def revalidate_seniority(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report matches without archiving anything.",
    ),
) -> None:
    """Re-check every non-archived, non-applied job against the current
    seniority hard-disqualifier (applypilot.eligibility) and archive any
    match, regardless of a previously stored fit_score.

    Safe to run repeatedly whenever the seniority rule changes -- jobs that
    don't match are never touched, and already-archived matches are a
    no-op on a second run. No job record is deleted or has its fit_score/
    tailored_resume_path/cover_letter_path overwritten; only state and
    apply_category/apply_error change, so the prior score is preserved for
    audit.
    """
    _bootstrap()

    from applypilot.eligibility import revalidate_seniority as _revalidate

    result = _revalidate(dry_run=dry_run)
    verb = "Would archive" if dry_run else "Archived"
    console.print(
        f"[cyan]Seniority revalidation:[/cyan] {verb} {result['updated']} / {result['matched']} matched job(s)."
    )
    if result["sample"]:
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Title")
        t.add_column("URL", overflow="fold")
        for row in result["sample"][:20]:
            t.add_row(row["title"], row["url"])
        console.print(t)
        if len(result["sample"]) > 20:
            console.print(f"[dim]...and {len(result['sample']) - 20} more.[/dim]")


@app.command()
def revalidate_stale_scores(
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        help="ISO timestamp (e.g. a scoring-rubric-fix commit's date). Jobs scored before this are archived.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report matches without archiving anything.",
    ),
) -> None:
    """Archive jobs whose fit_score was assigned before a given cutoff, so a
    scoring-rubric fix doesn't leave stale-scored rows stranded in the live
    queue (see applypilot.eligibility.revalidate_stale_scores).

    Safe to run repeatedly -- jobs scored on/after the cutoff are never
    touched, already-archived matches are a no-op on a second run, and no
    job record is deleted or has its fit_score/tailored_resume_path/
    cover_letter_path overwritten; only state changes, so the prior score is
    preserved for audit.
    """
    _bootstrap()

    from applypilot.eligibility import revalidate_stale_scores as _revalidate

    result = _revalidate(cutoff=cutoff, dry_run=dry_run)
    verb = "Would archive" if dry_run else "Archived"
    console.print(
        f"[cyan]Stale-score revalidation (cutoff {cutoff}):[/cyan] "
        f"{verb} {result['updated']} / {result['matched']} matched job(s)."
    )
    if result["sample"]:
        t = Table(show_header=True, header_style="bold cyan")
        t.add_column("Title")
        t.add_column("State")
        t.add_column("Score")
        t.add_column("Scored At")
        t.add_column("URL", overflow="fold")
        for row in result["sample"][:20]:
            t.add_row(row["title"], row["state"], str(row["fit_score"]), row["scored_at"], row["url"])
        console.print(t)
        if len(result["sample"]) > 20:
            console.print(f"[dim]...and {len(result['sample']) - 20} more.[/dim]")


@app.command("run-continuous")
def run_continuous(
    ready_buffer: int = typer.Option(
        config.DEFAULTS["ready_buffer"],
        "--ready-buffer",
        help="Target ready_to_apply queue size while Claude is available.",
    ),
    ready_buffer_unknown: int = typer.Option(
        config.DEFAULTS["ready_buffer_unknown"],
        "--ready-buffer-unknown",
        help="Smaller target used while Claude's reset time is unknown or it's unavailable for auth/transient reasons.",
    ),
    poll_interval: int = typer.Option(
        config.DEFAULTS["poll_interval"],
        "--poll-interval",
        help="Seconds between planning cycles when Claude is available.",
    ),
    cache_max_age: float = typer.Option(
        config.DEFAULTS["scheduler_cache_max_age"],
        "--cache-max-age",
        help="Seconds -- ~/.claude.json's cached usage state older than this is treated as unusable.",
    ),
    max_batch: int = typer.Option(
        config.DEFAULTS["scheduler_max_batch"],
        "--max-batch",
        help="Hard per-cycle ceiling on tailor/cover work regardless of estimated capacity.",
    ),
    safety_margin: float = typer.Option(
        config.DEFAULTS["scheduler_safety_margin"],
        "--safety-margin",
        help="Discount applied to measured tailor/cover throughput when estimating capacity before a known reset.",
    ),
    min_score: int = typer.Option(
        config.DEFAULTS["min_score"],
        "--min-score",
        help=f"Minimum fit score for tailor/cover/apply (default: {config.DEFAULTS['min_score']}).",
    ),
    max_age_days: int = typer.Option(
        config.DEFAULTS["max_job_age_days"],
        "--max-age-days",
        help=f"Skip jobs older than this many days (default: {config.DEFAULTS['max_job_age_days']}).",
    ),
    doc_format: str = typer.Option(
        "docx", "--doc-format", help="Document format for resumes/cover letters: docx (default) or pdf."
    ),
    no_continuous_apply: bool = typer.Option(
        False,
        "--no-continuous-apply",
        help="Run the upstream scheduler only -- do not also run the continuous apply worker in this process.",
    ),
) -> None:
    """Run discover/enrich/score/tailor/cover/apply continuously, sizing
    upstream (tailor/cover) work to the ready_to_apply queue and pacing it
    around Claude Code CLI availability for auto-apply.

    Re-queries the database every cycle -- a newly discovered high-scoring
    job enters the very next batch naturally. Single instance only (a
    second concurrent `run-continuous` refuses to start). Stop with Ctrl+C.
    """
    _bootstrap()

    from applypilot.config import check_tier

    check_tier(2, "AI scoring/tailoring")

    from applypilot.scoring.pdf import VALID_DOC_FORMATS

    if doc_format not in VALID_DOC_FORMATS:
        console.print(
            f"[red]Invalid --doc-format:[/red] '{doc_format}'. Must be one of: {', '.join(VALID_DOC_FORMATS)}"
        )
        raise typer.Exit(code=1)

    from applypilot.scheduler import SchedulerAlreadyRunning, SchedulerConfig
    from applypilot.scheduler import run_continuous as _run_continuous

    cfg = SchedulerConfig(
        ready_buffer=ready_buffer,
        ready_buffer_unknown=ready_buffer_unknown,
        poll_interval=poll_interval,
        cache_max_age=cache_max_age,
        max_batch=max_batch,
        safety_margin=safety_margin,
        no_continuous_apply=no_continuous_apply,
        min_score=min_score,
        max_age_days=max_age_days,
        doc_format=doc_format,
    )

    console.print("\n[bold blue]Launching continuous scheduler[/bold blue]")
    console.print(f"  Ready buffer:      {ready_buffer} (unknown-reset: {ready_buffer_unknown})")
    console.print(f"  Poll interval:     {poll_interval}s")
    console.print(f"  Max batch:         {max_batch}")
    console.print(f"  Continuous apply:  {'no' if no_continuous_apply else 'yes'}")
    console.print("[dim]Ctrl+C to stop.[/dim]\n")

    try:
        _run_continuous(cfg)
    except SchedulerAlreadyRunning as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def stop(
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stop an active `applypilot run --stream` process."),
    signal_process: bool = typer.Option(
        True, "--signal/--no-signal", help="Also send an OS interrupt signal to the active stream PID."
    ),
) -> None:
    """Request graceful shutdown of long-running pipeline processes."""
    from applypilot.config import ensure_dirs, load_env

    load_env()
    ensure_dirs()

    from applypilot.pipeline import (
        get_stream_pid,
        request_stream_interrupt_signal,
        request_stream_stop,
    )

    pid = get_stream_pid()
    request_stream_stop()
    console.print("[green]Stop requested for stream pipeline.[/green]")

    if pid:
        console.print(f"  Recorded PID: {pid}")
    else:
        console.print("  [dim]No recorded stream PID found. A stale stop request was still written.[/dim]")

    # Windows-specific fallback: if the stream pid is stale or missing, terminate
    # any active python run process so the discovery pipeline is not left running.
    if signal_process and not pid:
        import subprocess

        if os.name == "nt":
            res = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'python.*applypilot run' } | ForEach-Object { taskkill /PID $_.ProcessId /T /F }",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                console.print("  [green]Killed active applypilot run process tree.[/green]")
            else:
                console.print("  [yellow]No active applypilot run process tree was found.[/yellow]")

    if signal_process:
        signaled = request_stream_interrupt_signal()
        if signaled:
            console.print("  [green]Interrupt signal sent to stream process.[/green]")
        else:
            console.print("  [yellow]Could not send interrupt signal (PID missing or process already exited).[/yellow]")


@app.command("reject-titles")
def reject_titles(
    pattern: list[str] | None = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Regex title pattern to reject. Repeatable. Example: -p '\\bsenior\\b'",
    ),
    use_defaults: bool = typer.Option(
        True,
        "--defaults/--no-defaults",
        help="Include built-in senior-role reject patterns.",
    ),
    use_config: bool = typer.Option(
        True,
        "--from-config/--no-from-config",
        help="Include patterns from searches.yaml: title_reject_patterns (or filters.title_reject_patterns).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview matching jobs without updating DB.",
    ),
    sample: int = typer.Option(20, "--sample", help="How many sample matches to print."),
) -> None:
    """Bulk archive jobs whose title matches reject patterns."""
    _bootstrap()

    from applypilot.database import reject_jobs_by_title_patterns

    patterns = _resolve_title_reject_patterns(
        use_defaults=use_defaults,
        use_config=use_config,
        cli_patterns=pattern,
    )

    if not patterns:
        console.print("[red]No patterns specified.[/red] Use --pattern or --defaults.")
        raise typer.Exit(code=1)

    result = reject_jobs_by_title_patterns(
        patterns,
        dry_run=dry_run,
        sample_limit=max(1, sample),
    )

    action = "Would reject" if dry_run else "Rejected"
    console.print(f"[bold]{action}[/bold] {result['matched']} title-matched job(s).")
    if not dry_run:
        console.print(f"Updated: {result['updated']}")

    sample_rows = result.get("sample") or []
    if sample_rows:
        t = Table(title="Title Reject Sample", show_header=True, header_style="bold cyan")
        t.add_column("Pattern")
        t.add_column("Title", max_width=60)
        t.add_column("URL", max_width=80)
        for row in sample_rows:
            t.add_row(str(row.get("pattern", "")), str(row.get("title", "")), str(row.get("url", "")))
        console.print(t)


@app.command()
def apply(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Max applications to submit."),
    workers: int = typer.Option(5, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(
        config.DEFAULTS["min_score"],
        "--min-score",
        help=f"Minimum fit score for job selection (default: {config.DEFAULTS['min_score']}).",
    ),
    max_age_days: int = typer.Option(
        config.DEFAULTS["max_job_age_days"],
        "--max-age-days",
        help=(
            "Skip jobs whose discovered_at is older than this many days. "
            "0 = no age filter. "
            f"Default: {config.DEFAULTS['max_job_age_days']}."
        ),
    ),
    max_score: int | None = typer.Option(
        None, "--max-score", help="Maximum fit score for job selection (useful for testing on lower-score jobs)."
    ),
    model: str = typer.Option("sonnet", "--model", "-m", help="Claude model name (sonnet | haiku | opus)."),
    apply_engine: str = typer.Option(
        "claude",
        "--apply-engine",
        help="Apply engine: claude (current) or deterministic (Playwright rules).",
    ),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without submitting."),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    url: str | None = typer.Option(None, "--url", help="Apply to a specific job URL."),
    doc_format: str = typer.Option(
        "docx", "--doc-format", help="Document format for resumes/cover letters: docx (default) or pdf."
    ),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: str | None = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied."),
    fresh_sessions: bool = typer.Option(
        False, "--fresh-sessions", help="Refresh Chrome session cookies from your real profile before launching."
    ),
    no_hitl: bool = typer.Option(
        False, "--no-hitl", help="Skip HITL waits: park needs_human jobs and move on. Use for overnight runs."
    ),
    no_focus: bool = typer.Option(
        False,
        "--no-focus",
        help="Prevent Chrome windows from stealing keyboard focus (Linux/GNOME only). Windows stay visible but won't interrupt your active app.",
    ),
    mark_failed: str | None = typer.Option(
        None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."
    ),
    fail_reason: str | None = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
    reset_category: str | None = typer.Option(
        None, "--reset-category", help="Reset all jobs in a category for retry (e.g., blocked_technical)."
    ),
    sessions: bool = typer.Option(False, "--sessions", help="List saved ATS sessions."),
    clear_session: str | None = typer.Option(
        None, "--clear-session", help="Clear a saved ATS session (e.g., workday)."
    ),
) -> None:
    """Launch auto-apply to submit job applications."""
    _bootstrap()

    if apply_engine not in ("claude", "deterministic"):
        console.print("[red]Invalid --apply-engine:[/red] choose 'claude' or 'deterministic'.")
        raise typer.Exit(code=1)

    from applypilot.config import PROFILE_PATH as _profile_path
    from applypilot.config import check_tier
    from applypilot.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applypilot.apply.launcher import mark_job

        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applypilot.apply.launcher import mark_job

        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applypilot.apply.launcher import reset_failed as do_reset

        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    if reset_category:
        from applypilot.database import reset_by_category

        count = reset_by_category(reset_category)
        console.print(f"[green]Reset {count} job(s) in category '{reset_category}' for retry.[/green]")
        return

    if sessions:
        from applypilot.apply.chrome import list_ats_sessions

        ats_sessions = list_ats_sessions()
        if not ats_sessions:
            console.print("[dim]No saved ATS sessions.[/dim]")
            return
        from rich.table import Table

        t = Table(title="Saved ATS Sessions", show_header=True, header_style="bold cyan")
        t.add_column("ATS")
        t.add_column("Cookies")
        t.add_column("Age")
        for s in ats_sessions:
            age_str = f"{s['age_hours']:.1f}h" if s["age_hours"] is not None else "n/a"
            t.add_row(s["slug"], "yes" if s["has_cookies"] else "no", age_str)
        console.print(t)
        return

    if clear_session:
        from applypilot.apply.chrome import clear_ats_session

        if clear_ats_session(clear_session):
            console.print(f"[green]Cleared ATS session: {clear_session}[/green]")
        else:
            console.print(f"[yellow]No session found for: {clear_session}[/yellow]")
        return

    # --- Full apply mode ---

    # Validate --doc-format
    from applypilot.scoring.pdf import VALID_DOC_FORMATS as _valid_fmts

    if doc_format not in _valid_fmts:
        console.print(f"[red]Invalid --doc-format:[/red] '{doc_format}'. Must be one of: {', '.join(_valid_fmts)}")
        raise typer.Exit(code=1)

    # Set doc format for apply workers
    from applypilot.apply.launcher import set_doc_format

    set_doc_format(doc_format)

    # Check 1: Engine prerequisites
    if apply_engine == "claude":
        # Tier 3 requires Claude Code CLI + Chrome
        check_tier(3, "auto-apply")
    else:
        # Deterministic engine still requires Chrome, but not Claude CLI.
        from applypilot.config import get_chrome_path

        try:
            get_chrome_path()
        except FileNotFoundError:
            console.print(
                "[red]Deterministic apply engine requires Chrome/Chromium.[/red]\nInstall Chrome or set CHROME_PATH."
            )
            raise typer.Exit(code=1)

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print("[red]Profile not found.[/red]\nRun [bold]applypilot init[/bold] to create your profile first.")
        raise typer.Exit(code=1)

    # Check 3: Tailored resumes exist in the real ready_to_apply queue.
    # Skipped when --url targets a specific job -- acquire_job's own
    # state != 'archived' check is the real gate for that path.
    if not gen and not url:
        conn = get_connection()
        ready = conn.execute("SELECT COUNT(*) FROM jobs WHERE state = 'ready_to_apply'").fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No jobs in the ready_to_apply queue.[/red]\n"
                "Run [bold]applypilot run score tailor cover[/bold] first to prepare applications."
            )
            raise typer.Exit(code=1)

    if gen:
        from applypilot.apply.launcher import gen_prompt

        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, max_score=max_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print("\n[bold]Run manually:[/bold]")
        console.print(
            f"  claude --model {model} -p --mcp-config {mcp_path} --permission-mode bypassPermissions < {prompt_file}"
        )
        return

    from applypilot.apply.launcher import main as apply_main

    effective_limit = limit if limit is not None else 0

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    console.print(f"  Model:    {model}")
    console.print(f"  Engine:   {apply_engine}")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    if fresh_sessions:
        console.print("  Sessions: [yellow]refreshing from real Chrome profile[/yellow]")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        min_score=min_score,
        max_age_days=max_age_days,
        max_score=max_score,
        headless=headless,
        model=model,
        dry_run=dry_run,
        continuous=continuous,
        workers=workers,
        fresh_sessions=fresh_sessions,
        no_hitl=no_hitl,
        no_focus=no_focus,
        apply_engine=apply_engine,
    )


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applypilot.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyPilot Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Tailor auto-approved", str(stats.get("tailor_auto_approved", 0)))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))
    summary.add_row("Title-pattern rejected", str(stats.get("title_rejected", 0)))
    if stats.get("needs_human", 0) > 0:
        summary.add_row("Needs human review", str(stats["needs_human"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # Score funnel: per-score breakdown of pipeline stages
    funnel = stats.get("score_funnel", [])
    if funnel:
        funnel_table = Table(
            title="\nPipeline Funnel by Score",
            show_header=True,
            header_style="bold magenta",
        )
        funnel_table.add_column("Score", justify="center", style="bold")
        funnel_table.add_column("Cover Ready", justify="right", style="green")
        funnel_table.add_column("Tailored", justify="right", style="cyan")
        funnel_table.add_column("Needs Tailor", justify="right", style="yellow")
        funnel_table.add_column("Applied", justify="right", style="dim green")
        funnel_table.add_column("Errors", justify="right", style="red")

        for row in funnel:
            score = row["score"]
            score_str = f"[bold {'green' if score >= 9 else 'yellow' if score >= 7 else 'white'}]{score}[/]"
            funnel_table.add_row(
                score_str,
                str(row["cover_ready"]) if row["cover_ready"] else "[dim]—[/]",
                str(row["tailored"]) if row["tailored"] else "[dim]—[/]",
                str(row["needs_tailor"]) if row["needs_tailor"] else "[dim]—[/]",
                str(row["applied"]) if row["applied"] else "[dim]—[/]",
                str(row["errors"]) if row["errors"] else "[dim]—[/]",
            )

        console.print(funnel_table)

    # Apply categories with per-score breakdown
    by_cat = stats.get("by_category", {})
    if by_cat:
        cat_table = Table(title="\nApply Categories", show_header=True, header_style="bold blue")
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Total", justify="right")
        cat_table.add_column("10", justify="right", style="bold green")
        cat_table.add_column("9", justify="right", style="green")
        cat_table.add_column("8", justify="right", style="yellow")
        cat_table.add_column("7", justify="right", style="yellow")
        cat_table.add_column("6", justify="right", style="dim")
        cat_table.add_column("<6", justify="right", style="dim")
        cat_table.add_column("Action")

        cat_display = {
            "applied": ("green", "Done"),
            "pending": ("white", "In queue"),
            "in_progress": ("cyan", "Running now"),
            "needs_human": ("magenta", "applypilot human-review"),
            "blocked_auth": ("yellow", "Needs persistent sessions / HITL"),
            "blocked_technical": ("yellow", "Retryable: applypilot reset-category blocked_technical"),
            "archived_ineligible": ("dim", "Location/salary/type mismatch"),
            "archived_expired": ("dim", "Job no longer available"),
            "archived_platform": ("red", "Unsupported platform"),
            "archived_no_url": ("dim", "No application URL"),
            "manual_only": ("dim", "Manual ATS (no automation)"),
        }

        def _score_cell(d: dict, key: str) -> str:
            """Format a score count cell — blank if zero."""
            v = d.get(key, 0) if isinstance(d, dict) else 0
            return str(v) if v else "[dim]—[/dim]"

        order = [
            "applied",
            "pending",
            "in_progress",
            "needs_human",
            "blocked_auth",
            "blocked_technical",
            "archived_ineligible",
            "archived_expired",
            "archived_platform",
            "archived_no_url",
            "manual_only",
        ]
        for cat in order:
            d = by_cat.get(cat)
            if not d:
                continue
            total = d["total"] if isinstance(d, dict) else d
            color, action = cat_display.get(cat, ("white", ""))
            cat_table.add_row(
                f"[{color}]{cat}[/{color}]",
                str(total),
                _score_cell(d, "10"),
                _score_cell(d, "9"),
                _score_cell(d, "8"),
                _score_cell(d, "7"),
                _score_cell(d, "6"),
                _score_cell(d, "<6"),
                action,
            )

        # Any unknown categories
        for cat, d in sorted(by_cat.items(), key=lambda x: -(x[1]["total"] if isinstance(x[1], dict) else x[1])):
            if cat not in cat_display:
                total = d["total"] if isinstance(d, dict) else d
                if total > 0:
                    cat_table.add_row(
                        cat,
                        str(total),
                        _score_cell(d, "10"),
                        _score_cell(d, "9"),
                        _score_cell(d, "8"),
                        _score_cell(d, "7"),
                        _score_cell(d, "6"),
                        _score_cell(d, "<6"),
                        "",
                    )

        console.print(cat_table)

        # Retry hint for high-score retryable jobs
        retryable_tech = by_cat.get("blocked_technical", {})
        if isinstance(retryable_tech, dict):
            high_score_retryable = retryable_tech.get("10", 0) + retryable_tech.get("9", 0)
            if high_score_retryable > 0:
                console.print(
                    f"[bold yellow]  ↳ {high_score_retryable} score 9-10 jobs in blocked_technical are retryable.[/bold yellow]"
                    f" Run: [bold]applypilot reset-category blocked_technical[/bold]"
                )

    # By site (group all HN: * sites under "HackerNews")
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        grouped: dict[str, int] = {}
        for site, count in stats["by_site"]:
            key = "HackerNews" if (site or "").startswith("HN:") else (site or "Unknown")
            grouped[key] = grouped.get(key, 0) + count
        for site, count in sorted(grouped.items(), key=lambda x: -x[1]):
            site_table.add_row(site, str(count))

        console.print(site_table)

    # ── Funnel diagnostics ────────────────────────────────────────
    if stats.get("skipped_stale"):
        max_age = config.DEFAULTS["max_job_age_days"]
        console.print(f"\n[dim]Skipped as stale (>{max_age}d old): {stats['skipped_stale']} ready-to-apply jobs[/dim]")

    bbc = stats.get("blocked_by_cap") or {}
    if bbc.get("count"):
        console.print(f"\n[yellow]Blocked by company cap:[/yellow] {bbc['count']} company/companies")
        if bbc.get("companies"):
            preview = ", ".join(bbc["companies"][:10])
            more = f" (+{bbc['count'] - 10} more)" if bbc["count"] > 10 else ""
            console.print(f"  [dim]{preview}{more}[/dim]")

    # ── Pipeline state distribution (2026-04-24 state machine) ────
    from applypilot.database import get_connection as _get_conn

    _sc_conn = _get_conn()
    state_rows = _sc_conn.execute(
        "SELECT state, COUNT(*) FROM jobs WHERE state IS NOT NULL GROUP BY state ORDER BY COUNT(*) DESC"
    ).fetchall()
    if state_rows:
        state_table = Table(
            title="\nPipeline State Distribution",
            show_header=True,
            header_style="bold cyan",
        )
        state_table.add_column("State", style="bold")
        state_table.add_column("Count", justify="right")
        # Color-code by lifecycle phase.
        TERMINAL_OK = {"applied", "responded", "interview", "offer"}
        TERMINAL_BAD = {
            "rejected",
            "ghosted",
            "archived",
            "apply_failed",
            "enrich_failed",
            "score_failed",
            "tailor_failed",
            "cover_failed",
            "low_score",
        }
        ACTIVE = {"applying", "tailoring", "cover_writing", "ready_to_apply", "needs_human"}
        for r in state_rows:
            st = r[0]
            n = r[1]
            if st in TERMINAL_OK:
                color = "green"
            elif st in TERMINAL_BAD:
                color = "dim"
            elif st in ACTIVE:
                color = "yellow"
            else:
                color = "cyan"
            state_table.add_row(f"[{color}]{st}[/{color}]", str(n))
        console.print(state_table)

    console.print()


@app.command()
def track(
    days: int = typer.Option(14, "--days", "-d", help="Email look-back period in days."),
    setup: bool = typer.Option(False, "--setup", help="Verify Gmail MCP connectivity."),
    actions: bool = typer.Option(False, "--actions", "-a", help="Show pending action items."),
    ghosted_days: int = typer.Option(7, "--ghosted-days", help="Days before marking as ghosted."),
    limit: int = typer.Option(100, "--limit", "-l", help="Max emails to fetch."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Fetch + classify but don't write DB/files."),
    relabel: bool = typer.Option(
        False, "--relabel", help="Apply 'ap-track' label to all emails already in the DB (backfill)."
    ),
    remap_stubs: bool = typer.Option(
        False, "--remap-stubs", help="Re-match emails under multi-company stubs to correct per-company jobs."
    ),
) -> None:
    """Track application responses from Gmail."""
    _bootstrap()

    from applypilot.config import check_tier

    check_tier(2, "application tracking")

    if setup:
        import asyncio

        from applypilot.tracking.gmail_client import check_gmail_setup, verify_connection

        ok, msg = check_gmail_setup()
        if not ok:
            console.print(f"[red]{msg}[/red]")
            raise typer.Exit(code=1)

        console.print("[dim]Testing Gmail MCP connection...[/dim]")
        connected = asyncio.run(verify_connection())
        if connected:
            console.print("[green]Gmail MCP connected successfully.[/green]")
        else:
            console.print("[red]Gmail MCP connection failed.[/red]")
            console.print("[dim]Check that gcp-oauth.keys.json is valid and OAuth is authorized.[/dim]")
            raise typer.Exit(code=1)
        return

    if actions:
        from applypilot.tracking import show_action_items

        show_action_items()
        return

    if relabel:
        from applypilot.tracking import relabel_all_tracked

        relabel_all_tracked()
        return

    if remap_stubs:
        from applypilot.tracking import remap_stubs as _remap_stubs

        _remap_stubs()
        return

    from applypilot.tracking import run_tracking

    result = run_tracking(
        days=days,
        ghosted_days=ghosted_days,
        limit=limit,
        dry_run=dry_run,
    )

    if result.get("errors", 0) > 0:
        raise typer.Exit(code=1)


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from applypilot.view import open_dashboard

    open_dashboard()


# `applypilot human-review` was deleted in plan 5 of the apply UX overhaul.
# The standalone HITL server (port 7373) duplicated the in-pipeline HITL
# flow (port 7380+wid), so it was removed. Jobs that get parked as
# `needs_human` are now picked up automatically by the next
# `applypilot apply` run via _run_hitl's launcher-restart path.


# ---------------------------------------------------------------------------
# Q&A knowledge base commands
# ---------------------------------------------------------------------------

qa_app = typer.Typer(name="qa", help="Manage the Q&A knowledge base for screening questions.")
app.add_typer(qa_app)


@qa_app.command("list")
def qa_list(
    limit: int = typer.Option(50, "--limit", "-l", help="Max Q&A pairs to show."),
) -> None:
    """Show stored Q&A pairs from past applications."""
    _bootstrap()

    from applypilot.database import get_all_qa

    pairs = get_all_qa()
    if not pairs:
        console.print("[dim]No Q&A pairs stored yet. Apply to jobs to build the knowledge base.[/dim]")
        return

    t = Table(title=f"Q&A Knowledge Base ({len(pairs)} total)", show_header=True, header_style="bold cyan")
    t.add_column("Question", max_width=50)
    t.add_column("Answer", max_width=30)
    t.add_column("Source")
    t.add_column("Outcome")
    t.add_column("Type")

    for qa in pairs[:limit]:
        outcome_color = {"accepted": "green", "rejected": "red"}.get(qa["outcome"], "dim")
        t.add_row(
            qa["question_text"][:50],
            qa["answer_text"][:30],
            qa["answer_source"],
            f"[{outcome_color}]{qa['outcome']}[/{outcome_color}]",
            qa.get("field_type") or "",
        )

    console.print(t)


@qa_app.command("stats")
def qa_stats() -> None:
    """Show Q&A knowledge base statistics."""
    _bootstrap()

    from applypilot.database import get_qa_stats

    stats = get_qa_stats()
    if stats["total"] == 0:
        console.print("[dim]No Q&A pairs stored yet.[/dim]")
        return

    console.print("\n[bold]Q&A Knowledge Base Stats[/bold]\n")
    console.print(f"  Total pairs:    {stats['total']}")
    console.print(f"  Unique Qs:      {stats['unique_questions']}")

    if stats["by_source"]:
        console.print("\n  [bold]By source:[/bold]")
        for src, cnt in stats["by_source"].items():
            console.print(f"    {src}: {cnt}")

    if stats["by_outcome"]:
        console.print("\n  [bold]By outcome:[/bold]")
        for out, cnt in stats["by_outcome"].items():
            color = {"accepted": "green", "rejected": "red"}.get(out, "dim")
            console.print(f"    [{color}]{out}[/{color}]: {cnt}")

    if stats.get("by_ats"):
        console.print("\n  [bold]By ATS:[/bold]")
        for ats, cnt in stats["by_ats"].items():
            console.print(f"    {ats or 'unknown'}: {cnt}")

    console.print()


@qa_app.command("export")
def qa_export(
    output: str = typer.Option("qa_export.yaml", "--output", "-o", help="Output YAML file path."),
) -> None:
    """Export Q&A pairs to YAML for review and editing."""
    _bootstrap()

    from applypilot.database import export_qa_yaml

    content = export_qa_yaml()
    if not content:
        console.print("[dim]No Q&A pairs to export.[/dim]")
        return

    from pathlib import Path

    out_path = Path(output)
    out_path.write_text(content, encoding="utf-8")
    console.print(f"[green]Exported Q&A to:[/green] {out_path.resolve()}")


@qa_app.command("import")
def qa_import(
    file: str = typer.Argument(..., help="YAML file with Q&A pairs to import."),
) -> None:
    """Import Q&A pairs from a YAML file."""
    _bootstrap()

    from pathlib import Path

    import yaml

    from applypilot.database import store_qa

    path = Path(file)
    if not path.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(code=1)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        console.print("[red]Expected a YAML list of Q&A objects.[/red]")
        raise typer.Exit(code=1)

    count = 0
    for item in data:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            store_qa(
                q,
                a,
                source=item.get("source", "human"),
                field_type=item.get("field_type"),
                ats_slug=item.get("ats"),
            )
            count += 1

    console.print(f"[green]Imported {count} Q&A pair(s).[/green]")


# ---------------------------------------------------------------------------
# creds — site credential CRUD
# ---------------------------------------------------------------------------

creds_app = typer.Typer(name="creds", help="Manage saved site credentials (usernames / passwords).")
app.add_typer(creds_app)


@creds_app.command("list")
def creds_list(
    show: bool = typer.Option(False, "--show", "-s", help="Show passwords in plaintext."),
) -> None:
    """List all saved site credentials."""
    _bootstrap()

    from applypilot.database import get_all_accounts

    rows = get_all_accounts()
    if not rows:
        console.print("[dim]No credentials saved yet. Use [bold]applypilot creds add[/bold] to add one.[/dim]")
        return

    t = Table(title=f"Site Credentials ({len(rows)} entries)", show_header=True, header_style="bold cyan")
    t.add_column("Domain", min_width=28)
    t.add_column("Site", min_width=12)
    t.add_column("Email", min_width=24)
    t.add_column("Password", min_width=16)
    t.add_column("Notes", max_width=30)
    t.add_column("Saved", min_width=10)

    for row in rows:
        pwd = row["password"] or ""
        pwd_display = pwd if show else (("*" * min(len(pwd), 8)) if pwd else "[dim]—[/dim]")
        saved_date = (row["created_at"] or "")[:10]
        t.add_row(
            row["domain"],
            row["site"] or "",
            row["email"],
            pwd_display,
            row["notes"] or "",
            saved_date,
        )

    console.print(t)
    if not show:
        console.print("[dim]Passwords are masked. Use [bold]--show[/bold] to reveal them.[/dim]")


@creds_app.command("show")
def creds_show(
    domain: str = typer.Argument(..., help="Domain to show credentials for (e.g. linkedin.com)."),
) -> None:
    """Show full (unmasked) credentials for a single domain."""
    _bootstrap()

    from applypilot.database import get_all_accounts

    rows = [r for r in get_all_accounts() if r["domain"] == domain]
    if not rows:
        console.print(f"[red]No credentials found for domain:[/red] {domain}")
        raise typer.Exit(code=1)

    row = rows[0]
    console.print(f"\n  [bold]Domain:[/bold]   {row['domain']}")
    console.print(f"  [bold]Site:[/bold]     {row['site'] or ''}")
    console.print(f"  [bold]Email:[/bold]    {row['email']}")
    console.print(f"  [bold]Password:[/bold] {row['password'] or '[dim](none)[/dim]'}")
    if row["notes"]:
        console.print(f"  [bold]Notes:[/bold]    {row['notes']}")
    if row["job_url"]:
        console.print(f"  [bold]Job URL:[/bold]  {row['job_url']}")
    console.print(f"  [bold]Saved:[/bold]    {(row['created_at'] or '')[:19]}\n")


@creds_app.command("add")
def creds_add(
    domain: str = typer.Argument(..., help="Domain key (e.g. linkedin.com, myworkdayjobs.com)."),
    email: str = typer.Option(..., "--email", "-e", help="Login email / username."),
    password: str = typer.Option(None, "--password", "-p", help="Password (prompted if omitted)."),
    site: str = typer.Option(None, "--site", "-s", help="Human-readable site name."),
    notes: str = typer.Option(None, "--notes", "-n", help="Optional notes."),
) -> None:
    """Add or update credentials for a site."""
    _bootstrap()

    from applypilot.database import upsert_account

    if password is None:
        password = typer.prompt(f"Password for {domain}", hide_input=True, confirmation_prompt=True)

    action = upsert_account(domain, email, password, site=site, notes=notes)
    verb = "[green]Created[/green]" if action == "created" else "[yellow]Updated[/yellow]"
    console.print(f"{verb} credentials for [bold]{domain}[/bold] ({email})")


@creds_app.command("set")
def creds_set(
    domain: str = typer.Argument(..., help="Domain to update."),
    email: str = typer.Option(None, "--email", "-e", help="New email / username."),
    password: str = typer.Option(None, "--password", "-p", help="New password (prompted if --email not given either)."),
    notes: str = typer.Option(None, "--notes", "-n", help="Update notes."),
) -> None:
    """Update one or more fields for an existing credential entry."""
    _bootstrap()

    from applypilot.database import get_all_accounts, upsert_account

    existing = next((r for r in get_all_accounts() if r["domain"] == domain), None)
    if not existing:
        console.print(f"[red]No credentials found for:[/red] {domain}  (use [bold]add[/bold] to create one)")
        raise typer.Exit(code=1)

    new_email = email or existing["email"]
    new_password = password or existing["password"]
    new_notes = notes if notes is not None else existing["notes"]

    if not email and not password and notes is None:
        # Nothing specified — prompt for password at minimum
        new_password = typer.prompt(f"New password for {domain}", hide_input=True, confirmation_prompt=True)

    upsert_account(domain, new_email, new_password, site=existing["site"], notes=new_notes)
    console.print(f"[green]Updated[/green] credentials for [bold]{domain}[/bold]")


@creds_app.command("import-logs")
def creds_import_logs(
    log_dir: str = typer.Option(None, "--log-dir", help="Directory of apply logs. Defaults to ~/.applypilot/logs/."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be imported without writing."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip per-entry confirmation for free-form entries."),
) -> None:
    """Scan apply logs for account credentials and import into the DB.

    Handles both structured ACCOUNT_CREATED: lines (current format) and
    older free-form log entries where the agent wrote the password as prose.
    Already-known domains are skipped unless --yes is passed.
    """
    _bootstrap()

    import os

    from applypilot.database import get_all_accounts, mine_accounts_from_logs, upsert_account

    if log_dir is None:
        log_dir = os.path.expanduser("~/.applypilot/logs")

    console.print(f"[dim]Scanning logs in {log_dir}…[/dim]")
    found = mine_accounts_from_logs(log_dir)

    if not found:
        console.print("[dim]No credential hints found in logs.[/dim]")
        return

    existing_domains = {r["domain"] for r in get_all_accounts()}

    imported = skipped = already = 0
    for entry in found:
        domain = entry["domain"]
        email = entry["email"]
        password = entry.get("password", "")
        site = entry.get("site", "")
        source = entry.get("source", "")
        src_file = entry.get("source_file", "")
        is_new = domain not in existing_domains

        status_tag = "[green]NEW[/green]" if is_new else "[dim]EXISTS[/dim]"
        src_tag = "[cyan]structured[/cyan]" if source == "structured" else "[yellow]free-form[/yellow]"
        console.print(
            f"  {status_tag} {src_tag}  {domain}  {email or '[dim](no email)[/dim]'}"
            f"  pw={'***' if password else '[dim]none[/dim]'}  ({src_file})"
        )

        if not is_new:
            already += 1
            continue

        if dry_run:
            imported += 1
            continue

        # Free-form entries ask for confirmation (may be less reliable)
        if source == "free-form" and not yes:
            if not email or not password:
                console.print(
                    f"    [dim]Skipping — missing email or password. Add manually with: "
                    f"applypilot creds add {domain}[/dim]"
                )
                skipped += 1
                continue
            confirmed = typer.confirm(f"    Import {domain} ({email} / {password[:4]}***)?")
            if not confirmed:
                skipped += 1
                continue

        if email and password:
            upsert_account(domain, email, password, site=site or None, notes=f"auto-imported from log: {src_file}")
            existing_domains.add(domain)
            imported += 1
        else:
            console.print(
                f"    [dim]Skipping — missing {'email' if not email else 'password'}. "
                f"Add manually: applypilot creds add {domain}[/dim]"
            )
            skipped += 1

    suffix = " [dim](dry run — nothing written)[/dim]" if dry_run else ""
    console.print(
        f"\n[green]Imported {imported}[/green]  [dim]already-known {already}  skipped {skipped}[/dim]{suffix}"
    )


@creds_app.command("delete")
def creds_delete(
    domain: str = typer.Argument(..., help="Domain whose credentials to delete."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete all saved credentials for a domain."""
    _bootstrap()

    from applypilot.database import delete_account, get_all_accounts

    existing = [r for r in get_all_accounts() if r["domain"] == domain]
    if not existing:
        console.print(f"[red]No credentials found for:[/red] {domain}")
        raise typer.Exit(code=1)

    row = existing[0]
    if not yes:
        confirmed = typer.confirm(f"Delete credentials for {domain} ({row['email']})?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit()

    deleted = delete_account(domain)
    console.print(f"[green]Deleted {deleted} credential row(s) for[/green] [bold]{domain}[/bold]")


if __name__ == "__main__":
    app()
