"""Twilio SMS relay client for reading verification codes sent to a dedicated number.

Uses Twilio's REST API directly via httpx (Basic Auth: Account SID + Auth
Token) rather than the twilio SDK, to avoid a new heavy dependency --
mirrors gmail_client.py's shape (check_setup/verify_connection/list/get)
but over a plain REST API instead of MCP/stdio.

Deliberately relay-only (CLAUDE.md Future Work item 44/decision #197):
this module can read an SMS once one has arrived at the dedicated Twilio
number, but nothing here fills in phone numbers or submits codes into a
login/verification form. Wiring this into the apply-agent's own flow is
explicitly out of scope until the bot-detection-risk research flagged in
Future Work item 43 has actually been done.
"""

import logging
import os
import re
from datetime import datetime, timedelta, timezone

import httpx

log = logging.getLogger(__name__)

TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"

# 4-8 digit run, the same shape real verification codes take (mirrors how
# Gmail-OTP codes are read out of email bodies elsewhere in this codebase).
_CODE_RE = re.compile(r"\b(\d{4,8})\b")


def _credentials() -> tuple[str, str, str] | None:
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    number = os.environ.get("TWILIO_PHONE_NUMBER")
    if not (sid and token and number):
        return None
    return sid, token, number


def check_sms_setup() -> tuple[bool, str]:
    """Verify Twilio SMS relay prerequisites are in place.

    Returns:
        (ok, message) -- ok is True if all three required env vars are set.
    """
    if _credentials() is None:
        return False, (
            "Twilio SMS relay is not configured.\n\n"
            "Setup steps:\n"
            "  1. Sign up at https://www.twilio.com/try-twilio\n"
            "  2. Buy or claim a phone number that supports SMS\n"
            "     (Console -> Phone Numbers -> Buy a number)\n"
            "  3. From the Console dashboard, copy your Account SID and Auth Token\n"
            "  4. Add to ~/.applypilot/.env:\n"
            "       TWILIO_ACCOUNT_SID=...\n"
            "       TWILIO_AUTH_TOKEN=...\n"
            "       TWILIO_PHONE_NUMBER=+1XXXXXXXXXX\n"
            "  5. Run: applypilot sms --setup"
        )
    return True, "Twilio credentials found."


def verify_connection() -> bool:
    """Test Twilio API connectivity by fetching the account resource.

    Returns True if the credentials are valid and the API responds.
    """
    creds = _credentials()
    if creds is None:
        return False
    sid, token, _number = creds
    try:
        resp = httpx.get(f"{TWILIO_API_BASE}/Accounts/{sid}.json", auth=(sid, token), timeout=15)
    except httpx.HTTPError as e:
        log.error("Twilio connection failed: %s", e)
        return False
    if resp.status_code == 200:
        return True
    log.error("Twilio account check failed: HTTP %d %s", resp.status_code, resp.text[:200])
    return False


def _parse_twilio_date(raw: str | None) -> datetime | None:
    """Parse Twilio's RFC-2822-style date_sent field, e.g. 'Thu, 24 Sep 2026 20:56:32 +0000'."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %z")
    except ValueError:
        return None


def list_recent_messages(since_minutes: int = 10, limit: int = 20) -> list[dict]:
    """List SMS messages received on the relay number in the last `since_minutes`.

    Filtering is done client-side against the parsed `date_sent` field --
    Twilio's own `DateSent>` query filter is day-granularity only, too
    coarse for a "did a code just arrive" check.

    Returns:
        List of dicts with keys: sid, from_, body, date_sent -- newest first.
    """
    creds = _credentials()
    if creds is None:
        return []
    sid, token, number = creds

    try:
        resp = httpx.get(
            f"{TWILIO_API_BASE}/Accounts/{sid}/Messages.json",
            auth=(sid, token),
            params={"To": number, "PageSize": limit},
            timeout=15,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        log.error("Twilio message list failed: %s", e)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    messages = []
    for msg in resp.json().get("messages", []):
        date_sent_raw = msg.get("date_sent")
        date_sent = _parse_twilio_date(date_sent_raw)
        if date_sent is not None and date_sent < cutoff:
            continue
        messages.append(
            {
                "sid": msg.get("sid"),
                "from_": msg.get("from"),
                "body": msg.get("body") or "",
                "date_sent": date_sent_raw,
                "_parsed_date": date_sent,
            }
        )

    messages.sort(key=lambda m: m["_parsed_date"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    for m in messages:
        del m["_parsed_date"]
    return messages


def get_latest_verification_code(since_minutes: int = 5) -> str | None:
    """Return the most recent numeric code (4-8 digits) from a recent SMS, if any.

    Deliberately narrow: just extracts a number-shaped token from the most
    recent matching message body. Does not attempt to enter it anywhere --
    see this module's own docstring for why.
    """
    for msg in list_recent_messages(since_minutes=since_minutes, limit=5):
        match = _CODE_RE.search(msg["body"])
        if match:
            return match.group(1)
    return None
