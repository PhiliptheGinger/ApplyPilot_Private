"""FW51 (2026-10-07): the extension declines cookie banners.

Behavior was checked in headless Chromium against six fixture pages:
OneTrust (clicks 'Reject All'), a generic cookie notice ('Only
necessary'), a banner injected 2s after load ('Decline'), and three that
must NOT be touched -- an EEO form with 'Decline to self-identify', an
accept-only banner, and a page that only links a cookie policy.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parent.parent / "src" / "applypilot" / "apply" / "extension"


def test_registered_as_content_script():
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    scripts = [js for cs in manifest["content_scripts"] for js in cs.get("js", [])]
    assert "cookie-decline.js" in scripts


def test_safety_gates_present():
    js = (EXT / "cookie-decline.js").read_text(encoding="utf-8")
    assert "/accept|agree|allow/i.test(label)" in js  # never clicks an accept-ish label
    assert '[id*="cookie" i], [class*="cookie" i]' in js  # generic path only inside cookie containers
    assert "autoDeclineCookies" in js  # can be turned off


def test_script_parses():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    result = subprocess.run([node, "--check", str(EXT / "cookie-decline.js")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
