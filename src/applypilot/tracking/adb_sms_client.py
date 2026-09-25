"""Read SMS verification codes directly off a connected Android phone via ADB.

Generalizes the technique from a prior personal project (`~/Projects/Haywood`,
`scripts/extract_haywood.py`, 2026-08) which queried `content://sms` over ADB
to pull messages matching a search term. Here it's narrowed to "read the most
recent received text and pull a verification code out of it."

Free, no third-party account, uses the phone's real number directly -- the
real tradeoff versus sms_client.py (Twilio) / google_voice_client.py (Google
Voice) is that this only works while the phone is connected via USB (or
wireless ADB on the same network); it is not an always-on relay. Android
only -- there is no equivalent interface on iOS.

Deliberately relay-only, same boundary as the other two relay clients:
nothing here fills in a phone number or submits a code anywhere.

No hardcoded personal paths: unlike the original Haywood script (which
hardcoded one user's own adb.exe/output paths), this resolves adb via PATH
or an explicit APPLYPILOT_ADB_PATH override, so it works for any candidate's
machine/setup, not just this one.

Gotcha (see CLAUDE.md's Known Technical Gotchas #9): any call here starts
a persistent background `adb` server process if one isn't already
running, and it stays running afterward -- it can hold the USB connection
open and block Windows from safely ejecting the phone. `adb kill-server`
releases it.
"""

import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

_CODE_RE = re.compile(r"\b(\d{4,8})\b")

_ROW_RE = re.compile(r"Row:\s*\d+\s+(.*)")
_ID_RE = re.compile(r"(?:^|,\s*)_id=([^,]*?)\s*,\s*address=")
_ADDRESS_RE = re.compile(r"(?:^|,\s*)address=(.*?)\s*,\s*date=")
_DATE_RE = re.compile(r"(?:^|,\s*)date=(.*?)\s*,\s*type=")
_TYPE_RE = re.compile(r"(?:^|,\s*)type=(.*?)\s*,\s*body=")
_BODY_RE = re.compile(r"(?:^|,\s*)body=(.*)", re.DOTALL)


def _find_adb() -> str | None:
    override = os.environ.get("APPLYPILOT_ADB_PATH")
    if override and os.path.exists(override):
        return override
    return shutil.which("adb")


def search_common_install_locations() -> str | None:
    """Look for adb.exe in common Windows install spots when it's not on PATH.

    Real, live example this was built for (2026-09-24): a user with adb
    genuinely installed (at ~/Downloads/platform-tools/adb.exe, a common
    real download location -- the Android SDK platform-tools zip extracts
    there by default) but never added to PATH, so shutil.which found
    nothing and the setup flow had no way to help beyond "add it to PATH
    yourself." This lets the wizard/CLI offer to save a found path to
    .env automatically instead of leaving the user to hunt for it.

    Does NOT set APPLYPILOT_ADB_PATH itself -- callers decide whether to
    persist it (with the user's confirmation).
    """
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Downloads", "platform-tools", "adb.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk", "platform-tools", "adb.exe"),
        r"C:\platform-tools\adb.exe",
        r"C:\Android\platform-tools\adb.exe",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def check_adb_setup() -> tuple[bool, str]:
    """Verify adb is on PATH (or APPLYPILOT_ADB_PATH) and a device is connected.

    Returns:
        (ok, message) -- ok is True only if a device is authorized and ready.
    """
    adb = _find_adb()
    if adb is None:
        return False, (
            "adb (Android Debug Bridge) was not found.\n\n"
            "Setup steps:\n"
            "  1. Install Android platform-tools: https://developer.android.com/tools/releases/platform-tools\n"
            "  2. On your phone: Settings -> About phone -> tap 'Build number' 7 times to enable Developer options\n"
            "  3. Settings -> Developer options -> turn on 'USB debugging'\n"
            "  4. Connect the phone via USB and accept the 'Allow USB debugging?' prompt on the phone\n"
            "  5. Either add adb to your PATH, or set APPLYPILOT_ADB_PATH=<full path to adb.exe> in ~/.applypilot/.env"
        )

    try:
        result = subprocess.run([adb, "devices"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"Could not run adb: {e}"

    devices = [
        line for line in result.stdout.splitlines()[1:] if line.strip() and not line.startswith("*")
    ]
    ready = [line for line in devices if line.endswith("\tdevice")]
    if ready:
        return True, f"{len(ready)} device(s) ready."
    if devices:
        return False, "A device is connected but not authorized yet -- accept the 'Allow USB debugging?' prompt on the phone."
    return False, "No device connected. Plug in your phone via USB (or connect wireless ADB) and try again."


def _parse_sms_rows(raw_output: str) -> list[dict]:
    """Parse `adb shell content query --uri content://sms` output into row dicts.

    Same parsing shape as the original Haywood script's pull_from_phone(),
    generalized to not filter by any search term here -- filtering by
    recency/direction happens in list_recent_texts.
    """
    records: list[dict] = []
    current: dict | None = None

    for line in raw_output.splitlines():
        row_match = _ROW_RE.match(line)
        if row_match:
            if current is not None:
                records.append(current)
            current = {"id": "", "address": "", "date": "", "type": "", "body": ""}
            row = row_match.group(1)

            if m := _ID_RE.search(row):
                current["id"] = m.group(1).strip()
            if m := _ADDRESS_RE.search(row):
                current["address"] = m.group(1).strip()
            if m := _DATE_RE.search(row):
                current["date"] = m.group(1).strip()
            if m := _TYPE_RE.search(row):
                current["type"] = m.group(1).strip()
            if m := _BODY_RE.search(row):
                current["body"] = m.group(1)
        elif current is not None:
            current["body"] += "\n" + line

    if current is not None:
        records.append(current)
    return records


def list_recent_texts(since_minutes: int = 10, limit: int = 20) -> list[dict]:
    """List received SMS from the connected phone in the last `since_minutes`.

    Returns:
        List of dicts with keys: address, body, date -- newest first.
    """
    adb = _find_adb()
    if adb is None:
        return []

    command = [
        adb, "shell", "content", "query",
        "--uri", "content://sms",
        "--projection", "_id:address:date:type:body",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        log.error("adb SMS query failed: %s", e)
        return []

    if result.returncode != 0:
        log.error("adb SMS query returned an error: %s", result.stderr[:300])
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    messages = []
    for row in _parse_sms_rows(result.stdout):
        if row.get("type", "").rstrip(",").strip() != "1":  # 1 = received, 2 = sent
            continue
        try:
            ts = int(row["date"].rstrip(",").strip()) / 1000
            sent_at = datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            continue
        if sent_at < cutoff:
            continue
        messages.append({"address": row["address"].rstrip(",").strip(), "body": row["body"].strip(), "date": sent_at})

    messages.sort(key=lambda m: m["date"], reverse=True)
    return messages[:limit]


def get_latest_verification_code(since_minutes: int = 5) -> str | None:
    """Return the most recent numeric code (4-8 digits) from a recent received text."""
    for msg in list_recent_texts(since_minutes=since_minutes, limit=5):
        match = _CODE_RE.search(msg["body"])
        if match:
            return match.group(1)
    return None
