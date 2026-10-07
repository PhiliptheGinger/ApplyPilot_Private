"""Email notifications for events that need the candidate's attention (FW53).

2026-10-07. Jobs sat paused on needs_human for hours, and long Claude
usage-limit stalls went unnoticed, because the only alerts were a terminal
line and a desktop popup (which does nothing on Windows). This sends a
short email through the already-configured Gmail integration
(tracking.gmail_client.send_email, decision #184).

Off by default. Turn it on in ~/.applypilot/.env:

    APPLYPILOT_NOTIFY_EMAIL=self              # send to profile.json personal.email
    APPLYPILOT_NOTIFY_EMAIL=me@example.com    # or any address

Rules:
- Never raises and never blocks the caller: sending happens on a daemon
  thread, and any failure is only logged.
- Each event has a key; the same key is sent at most once per
  `min_interval_seconds` (default 30 minutes), remembered across restarts
  in ~/.applypilot/notify_state.json, so a retry loop can't flood the inbox.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MIN_INTERVAL_SECONDS = 30 * 60
_ENABLE_WORDS = {"self", "1", "true", "yes", "on"}
_DISABLE_WORDS = {"", "0", "false", "no", "off"}

_state_lock = threading.Lock()


def _state_path() -> Path:
    from applypilot import config

    return Path(config.APP_DIR) / "notify_state.json"


def resolve_recipient() -> str | None:
    """The address to notify, or None when notifications are off."""
    raw = (os.environ.get("APPLYPILOT_NOTIFY_EMAIL") or "").strip()
    if raw.lower() in _DISABLE_WORDS:
        return None
    if raw.lower() in _ENABLE_WORDS:
        try:
            from applypilot.config import load_profile

            email = ((load_profile().get("personal") or {}).get("email") or "").strip()
        except Exception:  # noqa: BLE001 - a missing profile just means "can't notify"
            email = ""
        return email or None
    return raw if "@" in raw else None


def _claim_slot(key: str, min_interval_seconds: float, now: float) -> bool:
    """Record that `key` is being sent now, unless it was sent too recently."""
    path = _state_path()
    with _state_lock:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                state = {}
        except (OSError, ValueError):
            state = {}
        last = state.get(key)
        if isinstance(last, (int, float)) and now - last < min_interval_seconds:
            return False
        state[key] = now
        # Keep the file small: drop entries older than a day.
        state = {k: v for k, v in state.items() if isinstance(v, (int, float)) and now - v < 86400}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state), encoding="utf-8")
        except OSError:
            log.debug("could not persist notify state", exc_info=True)
        return True


def _send(recipient: str, subject: str, body: str) -> None:
    try:
        import asyncio

        from applypilot.tracking.gmail_client import check_gmail_setup, send_email

        ready, why = check_gmail_setup()
        if not ready:
            log.warning("Notification not sent (Gmail not set up): %s", why)
            return
        ok, detail = asyncio.run(send_email([recipient], subject, body))
        if ok:
            log.info("Notification sent: %s", subject)
        else:
            log.warning("Notification failed: %s", str(detail)[:200])
    except Exception:  # noqa: BLE001 - notifications must never break the pipeline
        log.warning("Notification failed", exc_info=True)


def notify(
    key: str,
    subject: str,
    body: str,
    *,
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    wait: bool = False,
) -> bool:
    """Email the candidate about an event. Returns True if a send was started.

    `wait=True` sends on the calling thread (used by tests and the CLI test
    command); otherwise the send runs on a daemon thread.
    """
    recipient = resolve_recipient()
    if not recipient:
        return False
    if not _claim_slot(key, min_interval_seconds, time.time()):
        log.debug("Notification %r suppressed (sent recently)", key)
        return False
    subject = f"[ApplyPilot] {subject}"
    if wait:
        _send(recipient, subject, body)
    else:
        threading.Thread(target=_send, args=(recipient, subject, body), daemon=True, name="notify").start()
    return True


# -- Event helpers -------------------------------------------------------------


def notify_needs_human(job: dict, reason: str, stuck_url: str, instructions: str = "") -> bool:
    title = job.get("title") or "Unknown job"
    company = job.get("company") or job.get("site") or ""
    url = job.get("url") or stuck_url or ""
    body = (
        "An application is paused and waiting for you.\n\n"
        f"Job: {title}" + (f" @ {company}" if company else "") + "\n"
        f"Reason: {reason}\n"
        f"Page: {stuck_url}\n"
        + (f"What to do: {instructions}\n" if instructions else "")
        + "\nOpen the worker's Chrome window, finish the step, then click Done in the ApplyPilot banner."
    )
    return notify(f"needs_human:{url}", f"Needs you: {title}", body)


def notify_claude_exhausted(reason: str) -> bool:
    body = (
        "Auto-apply paused: Claude Code reported a usage limit "
        f"({reason}). Interactive Claude and the apply agent share the same plan.\n\n"
        "The run keeps waiting and retries about every 30 minutes on its own. "
        "You'll get another email when it's working again."
    )
    return notify("claude_exhausted", "Auto-apply paused (Claude usage limit)", body)


def notify_claude_recovered() -> bool:
    body = "Claude Code is answering again, so auto-apply has resumed."
    return notify("claude_recovered", "Auto-apply resumed", body)
