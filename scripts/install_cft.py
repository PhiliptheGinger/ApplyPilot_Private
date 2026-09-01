#!/usr/bin/env python3
"""Download Chrome for Testing 148 to ~/.applypilot/chrome-for-testing/.

Idempotent: skips download if binary already present and version matches latest 148.x.
"""
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

CFT_DIR = Path.home() / ".applypilot" / "chrome-for-testing"
MAJOR = "148"


def _platform_spec() -> tuple[str, str]:
    """Return (feed_platform, extracted_dir)."""
    system = platform.system()
    if system == "Windows":
        return ("win64", "chrome-win64")
    if system == "Darwin":
        # CfT feed uses "mac-x64" / "mac-arm64".
        machine = platform.machine().lower()
        feed = "mac-arm64" if "arm" in machine or "aarch" in machine else "mac-x64"
        return (feed, "chrome-mac-x64" if feed == "mac-x64" else "chrome-mac-arm64")
    return ("linux64", "chrome-linux64")


def _target_bin() -> Path:
    system = platform.system()
    if system == "Windows":
        return CFT_DIR / EXTRACTED_DIR / "chrome.exe"
    if system == "Darwin":
        return CFT_DIR / EXTRACTED_DIR / "Google Chrome for Testing.app" / "Contents" / "MacOS" / "Google Chrome for Testing"
    return CFT_DIR / EXTRACTED_DIR / "chrome"


PLATFORM, EXTRACTED_DIR = _platform_spec()
TARGET_BIN = _target_bin()


def latest_148_url() -> tuple[str, str]:
    """Return (version, download_url) for the latest CfT 148.x build for this platform."""
    feed = "https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"
    with urllib.request.urlopen(feed) as r:
        data = json.load(r)
    candidates = [v for v in data["versions"] if v["version"].startswith(f"{MAJOR}.")]
    if not candidates:
        sys.exit(f"No CfT {MAJOR}.x found in feed {feed}")
    latest = candidates[-1]
    chrome_dl = next(d for d in latest["downloads"]["chrome"] if d["platform"] == PLATFORM)
    return latest["version"], chrome_dl["url"]


def main() -> None:
    version, dl_url = latest_148_url()
    if TARGET_BIN.exists():
        try:
            cur = subprocess.check_output([str(TARGET_BIN), "--version"], text=True).strip()
        except subprocess.CalledProcessError:
            cur = ""
        if version in cur:
            print(f"CfT {version} already installed at {TARGET_BIN}")
            return
        print(f"Replacing existing CfT install (was: {cur!r}, want: {version})")
        shutil.rmtree(CFT_DIR / EXTRACTED_DIR, ignore_errors=True)

    CFT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CFT_DIR / f"{EXTRACTED_DIR}.zip"
    print(f"Downloading CfT {version} from {dl_url}...")
    urllib.request.urlretrieve(dl_url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(CFT_DIR)
    zip_path.unlink()

    if not TARGET_BIN.exists():
        sys.exit(f"Extraction failed: {TARGET_BIN} not found after unzip")

    # zipfile.extractall() does not preserve POSIX permissions.
    # Restore execute bits on POSIX platforms only.
    if platform.system() != "Windows":
        extracted = CFT_DIR / EXTRACTED_DIR
        for name in ("chrome", "chrome_crashpad_handler", "chrome-wrapper", "chrome_sandbox"):
            p = extracted / name
            if p.exists():
                p.chmod(0o755)

    out = subprocess.check_output([str(TARGET_BIN), "--version"], text=True).strip()
    print(f"Installed: {out}")


if __name__ == "__main__":
    main()
