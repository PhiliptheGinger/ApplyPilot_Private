"""Read SMS codes forwarded from Google Voice into Gmail — a free alternative
to the Twilio relay (tracking/sms_client.py, CLAUDE.md decision #197).

Google Voice numbers are free for US residents and can forward every
inbound text to the account's Gmail inbox (a manual, one-time toggle at
voice.google.com -> Settings -> Messages -> "Forward messages to email").
This reads those forwarded messages through the SAME Gmail MCP
integration gmail_client.py already uses -- no new credentials, no
per-message cost, no A2P 10DLC registration (Twilio's real ongoing cost).

Confirmed real detail: forwarded texts arrive from an @txt.voice.google.com
address. The exact subject-line/body layout was NOT independently verified
against a real forwarded message at build time (no live Google Voice
account existed yet) -- extraction is deliberately permissive (same
4-8-digit-run approach as sms_client.py) rather than assuming a rigid
format. Verify against a real forwarded text before fully trusting this
in production.

Deliberately relay-only, same boundary as sms_client.py: nothing here
fills in a phone number or submits a code anywhere.
"""

import logging
import re

from applypilot.tracking.gmail_client import _call_tool_raw, _create_mcp_client, _normalize_email, _parse_search_results

log = logging.getLogger(__name__)

_CODE_RE = re.compile(r"\b(\d{4,8})\b")

# Gmail's own search operator, day-granularity only (same limitation Twilio's
# DateSent> filter has) -- fine-grained minute-level filtering isn't
# attempted here since it would need real-format verification of Gmail's
# Date header first (not yet available, see module docstring).
_SEARCH_QUERY = "newer_than:1d from:txt.voice.google.com"


async def search_recent_texts(limit: int = 5) -> list[dict]:
    """Search Gmail for Google-Voice-forwarded texts from roughly the last day.

    Returns normalized email dicts with body text populated, newest first
    (Gmail's own default search ordering).
    """
    from mcp import ClientSession

    try:
        async with await _create_mcp_client() as (read, write), ClientSession(read, write) as session:
            import asyncio

            await asyncio.wait_for(session.initialize(), timeout=30)
            raw_text = await _call_tool_raw(session, "search_emails", {"query": _SEARCH_QUERY, "maxResults": limit})

            results = _parse_search_results(raw_text)
            emails: list[dict] = []
            for r in results:
                msg_id = r.get("id")
                if not msg_id:
                    continue
                full_text = await _call_tool_raw(session, "read_email", {"messageId": msg_id})
                from applypilot.tracking.gmail_client import _parse_read_result

                full = _parse_read_result(full_text, msg_id)
                emails.append(_normalize_email(full))
            return emails
    except Exception as e:  # noqa: BLE001 - relay read is best-effort, must degrade to empty list not crash
        log.warning("Google Voice text search failed: %s", e)
        return []


async def get_latest_verification_code() -> str | None:
    """Return the most recent numeric code (4-8 digits) from a recent Google-Voice-forwarded text."""
    for email in await search_recent_texts(limit=5):
        match = _CODE_RE.search(email.get("body", ""))
        if match:
            return match.group(1)
    return None
