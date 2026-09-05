"""Result-output parsing, classification, and bookkeeping.

Extracted from launcher.py. Pure-ish utilities — no Chrome, no agent
invocation, no HITL banners. These functions read the agent's stdout/text
output and write to the database, append to the failed-actions log, append
to the manual-actions markdown, or update a per-worker history list.

`_record_job_history` lazy-imports `launcher` to read the shared
`_worker_state` dict; everything else is self-contained.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

from applypilot import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result/output parsers
# ---------------------------------------------------------------------------


def _parse_account_created(output: str, job_url: str | None = None) -> None:
    """Parse ACCOUNT_CREATED lines from agent output and save to DB."""
    from applypilot.database import get_connection, store_account

    for line in output.split("\n"):
        if "ACCOUNT_CREATED:" not in line:
            continue
        try:
            json_str = line.split("ACCOUNT_CREATED:", 1)[1].strip()
            account = json.loads(json_str)
            conn = get_connection()
            store_account(conn, account, job_url=job_url)
            logger.info("Saved new account: %s @ %s", account.get("email"), account.get("domain"))
        except Exception as e:  # noqa: BLE001 - processing independent output lines; one malformed ACCOUNT_CREATED line must not abort parsing the rest (also: Exception already subsumes the JSONDecodeError/IndexError this used to list separately)
            logger.warning("Failed to parse ACCOUNT_CREATED line: %s", e)


def _parse_qa_lines(output: str, job_url: str | None = None, ats_slug: str | None = None) -> int:
    """Parse QA: lines from agent output and store in qa_knowledge DB.

    Format: QA:{question}|{answer}|{field_type}

    Returns count of Q&A pairs stored.
    """
    from applypilot.database import store_qa

    count = 0
    for line in output.split("\n"):
        if not line.strip().startswith("QA:"):
            continue
        try:
            payload = line.split("QA:", 1)[1].strip()
            parts = payload.split("|")
            if len(parts) < 2:
                continue
            question = parts[0].strip()
            answer = parts[1].strip()
            field_type = parts[2].strip() if len(parts) > 2 else None
            if question and answer:
                store_qa(question, answer, source="agent", field_type=field_type, ats_slug=ats_slug, job_url=job_url)
                count += 1
        except Exception as e:  # noqa: BLE001 - processing independent output lines; one malformed QA line must not abort parsing the rest
            logger.debug("Failed to parse QA line: %s", e)
    if count:
        logger.info("Stored %d Q&A pair(s) from agent output", count)
    return count



# 2026-09-04, found via the architecture review (this function was flagged
# as "closer to reinventing structured output with duct tape" -- checking
# it against realistic examples surfaced a real, concrete false-positive
# rather than just a theoretical one): a negated sentence describing a
# FAILURE can still contain a literal success phrase as a substring, e.g.
# "I was unable to get the application submitted successfully due to a
# CAPTCHA" contains "application submitted successfully" verbatim. Without
# a guard, that's misread as "applied" -- a real failed application could
# get marked applied in the DB and never retried. Deliberately narrow: a
# small, explicit list of common negation/failure cues checked in the text
# immediately preceding the matched phrase (same discipline as this
# codebase's other curated-list checks, not a general sentiment classifier).
_NEGATION_CUES = (
    "unable to",
    "couldn't",
    "could not",
    "did not",
    "didn't",
    "failed to",
    "was not",
    "wasn't",
    "not get",
    "not able",
    "never got",
    "cannot",
    "can't",
)
_NEGATION_WINDOW = 40


def _is_negated_context(lower_text: str, phrase_start: int) -> bool:
    """True if a negation cue appears shortly before the matched phrase,
    within the same clause -- checked on the text BEFORE the match only
    (a negation appearing only after the phrase, in an unrelated later
    clause, shouldn't suppress an otherwise-genuine positive)."""
    window = lower_text[max(0, phrase_start - _NEGATION_WINDOW) : phrase_start]
    return any(cue in window for cue in _NEGATION_CUES)


def _infer_result_from_output(output: str) -> str | None:
    """Infer a result from agent output when no RESULT line was emitted.

    Scans for common phrases that indicate success or a specific failure mode.
    Returns 'applied' for detected successful submissions, a failure reason
    string for failures, or None if nothing can be inferred.
    """
    lower = output.lower()

    # Check for successful application first — agent submitted but forgot RESULT:APPLIED
    success_phrases = [
        "application submitted successfully",
        "application was sent to",
        "application has been received",
        "successfully submitted",
        "thank you for applying",
        "your application was sent",
        "application sent",
        "application received",
        "application submitted",
    ]
    # Strong indicators alone are enough
    strong_success = [
        "application submitted successfully",
        "your application was sent to",
        "thank you for applying",
        "successfully submitted",
    ]
    for phrase in strong_success:
        idx = lower.find(phrase)
        if idx != -1 and not _is_negated_context(lower, idx):
            return "applied"
    # Weaker indicators need 2+ matches, each individually not negated
    success_count = 0
    for p in success_phrases:
        idx = lower.find(p)
        if idx != -1 and not _is_negated_context(lower, idx):
            success_count += 1
    if success_count >= 2:
        return "applied"

    # Order matters — check most specific patterns first
    patterns: list[tuple[str, list[str]]] = [
        (
            "login_issue",
            [
                "password reset requires email",
                "cannot log in",
                "login failed permanently",
                "cannot access external email",
                "session has now expired",
                "credentials between session",
            ],
        ),
        (
            "account_required",
            [
                "account was successfully created",
                "password from the original account",
                "password was not stored",
            ],
        ),
        (
            "captcha",
            [
                "blocked by captcha",
                "captcha cannot be solved",
                "unsolvable captcha",
            ],
        ),
        (
            "already_applied",
            [
                "you've already applied",
                "you have already applied",
                "already applied to this",
                "application already submitted",
                "duplicate application",
                "already applied for this role",
                "application already exists",
            ],
        ),
        (
            "expired",
            [
                "no longer accepting",
                "job has been closed",
                "position has been filled",
                "listing is closed",
                "listing has expired",
            ],
        ),
        (
            "not_eligible_location",
            [
                "not eligible for this location",
                "onsite only",
                "outside your area",
                "cannot relocate",
            ],
        ),
        (
            "stuck",
            [
                "cannot be completed through browser automation",
                "must be completed manually",
                "sandbox environment",
                "sandboxed environment",
                "non-sandboxed environment",
                "technical blocker",
                "cannot satisfy",
                "blocked_by_environment",
                "infrastructure-level",
            ],
        ),
        (
            "browser_unavailable",
            [
                "browser automation service is not responding",
                "connection refused on localhost",
                "browser connection issue",
            ],
        ),
    ]
    for reason, phrases in patterns:
        for phrase in phrases:
            if phrase in lower:
                return reason
    return None


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

PERMANENT_FAILURES: set[str] = {
    "expired",
    "form_interaction_error",
    "browser_unavailable",
    "not_eligible_location",
    "not_eligible_salary",
    "already_applied",
    "not_a_job_application",
    "unsafe_permissions",
    "unsafe_verification",
    "contract_only",
    "site_blocked",
    "cloudflare_blocked",
    "blocked_by_cloudflare",
    "credits_exhausted",
    "file_upload_blocked",
    "account_creation_broken",
    "page_error",
    "application_limit_exceeded",
}

# Errors that should pause and wait for human intervention via HITL banner
# instead of being marked as permanent failures.
HITL_AUTO_ROUTE: frozenset[str] = frozenset(
    {
        "captcha",
        "login_issue",
        "account_required",
        "sso_required",
        "email_verification",
        "resume_upload_blocked",
        "stuck",
    }
)

# login_required is retryable — user logs in manually, then retry succeeds
RETRYABLE_AUTH_FAILURES: set[str] = {"login_required"}

PERMANENT_PREFIXES: tuple[str, ...] = ("site_blocked", "cloudflare", "blocked_by")


def _is_permanent_failure(result: str) -> bool:
    """Determine if a failure should never be retried."""
    reason = result.split(":", 1)[-1] if ":" in result else result
    # login_required is explicitly retryable
    if result in RETRYABLE_AUTH_FAILURES or reason in RETRYABLE_AUTH_FAILURES:
        return False
    return (
        result in PERMANENT_FAILURES
        or reason in PERMANENT_FAILURES
        or any(reason.startswith(p) for p in PERMANENT_PREFIXES)
    )


# ---------------------------------------------------------------------------
# Action logs (file-backed)
# ---------------------------------------------------------------------------

_NEXT_STEPS: dict[str, str] = {
    "login_required": "Log in manually in the Chrome worker window, then re-run. The session will persist.",
    "login_issue": "Login failed permanently. Check if the site requires SSO or a different account. May need to apply manually.",
    "sso_required": "Site requires Google/Microsoft/SSO login. Apply manually via browser.",
    "captcha": "Blocked by unsolvable CAPTCHA. Try again later or apply manually.",
    "expired": "Job listing is closed/expired. No action needed — remove from queue.",
    "not_eligible_location": "Job is onsite-only outside your area. No action needed.",
    "not_eligible_salary": "Salary below floor. No action needed.",
    "not_a_job_application": "Site is a profile builder / talent marketplace, not a job application. No action needed.",
    "unsafe_permissions": "Site requested camera/mic/screen permissions. Apply manually if interested.",
    "unsafe_verification": "Site requires video/biometric verification. Apply manually if interested.",
    "already_applied": "Already applied to this job. No action needed.",
    "stuck": "Agent got stuck on the page after 3 attempts. Check the worker log for details, then try manually.",
    "page_error": "Page was broken (500 error, blank page). Try again later.",
    "timeout": "Agent timed out. Job may be complex (multi-page form). Try with longer timeout or apply manually.",
    "no_result_line": "Agent finished but didn't output a result code. Check worker log for what happened.",
}

_FAILED_LOG = config.LOG_DIR / "failed_actions.log"
_MANUAL_LOG = config.APP_DIR / "manual_actions.md"


def _log_failed_attempt(job: dict, reason: str, worker_id: int, duration_ms: int, permanent: bool) -> None:
    """Append a structured entry to the failed actions log.

    Each entry includes the job, failure reason, whether it's retryable,
    and a human-readable next step so the user knows what to do.
    """
    config.ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    url = job.get("application_url") or job["url"]
    title = job.get("title", "Unknown")
    company = job.get("site", "Unknown")
    score = job.get("fit_score", "?")
    duration_s = duration_ms / 1000 if duration_ms else 0

    # Look up the next step for this failure reason
    next_step = _NEXT_STEPS.get(reason, "Check the worker log for details. May need to apply manually.")

    retryable = "NO (permanent)" if permanent else "YES (will retry automatically)"

    entry = (
        f"\n{'─' * 70}\n"
        f"[{ts}]  {title} @ {company}  (score: {score}/10)\n"
        f"URL:      {url}\n"
        f"Reason:   {reason}\n"
        f"Duration: {duration_s:.0f}s  |  Worker: {worker_id}  |  Retryable: {retryable}\n"
        f"Action:   {next_step}\n"
    )

    try:
        with open(_FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(entry)
    except OSError:
        logger.debug("Could not write to failed actions log", exc_info=True)


def _log_manual_action(job: dict, reason: str, instructions: str) -> None:
    """Append a human-action-required entry to ~/.applypilot/manual_actions.md."""
    config.ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    url = job.get("application_url") or job["url"]
    title = job.get("title", "Unknown")
    company = job.get("site", "Unknown")
    score = job.get("fit_score", "?")

    entry = (
        f"\n## {title} @ {company}\n"
        f"- **When**: {ts}\n"
        f"- **Score**: {score}/10\n"
        f"- **URL**: {url}\n"
        f"- **Reason**: {reason}\n"
        f"- **Action needed**: {instructions}\n"
        f"- **Retry**: `applypilot apply --url '{url}'`\n"
    )

    try:
        with open(_MANUAL_LOG, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("# Manual Actions Required\n\nJobs that need human intervention before retrying.\n")
            f.write(entry)
    except OSError:
        logger.debug("Could not write to manual actions log", exc_info=True)


# ---------------------------------------------------------------------------
# Worker history (depends on launcher's shared _worker_state)
# ---------------------------------------------------------------------------


def _record_job_history(worker_id: int, job: dict, result: str, duration_ms: int) -> None:
    """Append a completed job entry to the worker's in-memory history list.

    Shown on the per-worker homepage (http://localhost:{7380+worker_id}/).
    Keeps the last 50 entries; oldest are dropped automatically.
    """
    # Lazy import to avoid circular dependency: launcher.py owns the
    # _worker_state dict (also touched by the always-on HTTP server and by
    # the orchestrator), and launcher.py re-exports this function.
    from applypilot.apply import launcher

    with launcher._worker_state_lock:
        ws = launcher._worker_state.get(worker_id)
    if ws is None:
        return
    history: list = ws.setdefault("history", [])
    # Classify result into a display category
    if "applied" in result.lower():
        outcome = "applied"
    elif "expired" in result.lower():
        outcome = "expired"
    elif "already_applied" in result.lower():
        outcome = "already_applied"
    elif "needs_human" in result.lower():
        outcome = "needs_human"
    else:
        outcome = "failed"
    history.append(
        {
            "ts": time.time(),
            "title": job.get("title", ""),
            "company": job.get("company") or job.get("site", ""),
            "url": job.get("application_url") or job.get("url", ""),
            "score": job.get("fit_score", 0),
            "result": result,
            "outcome": outcome,
            "duration_s": round(duration_ms / 1000) if duration_ms else 0,
        }
    )
    if len(history) > 50:
        del history[:-50]
