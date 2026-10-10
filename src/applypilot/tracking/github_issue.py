"""File a GitHub issue from the in-banner bug-report icon (FW47a).

Off unless APPLYPILOT_GITHUB_TOKEN is set (~/.applypilot/.env or the repo-
root .env, per config.load_env()'s two-location search). The target repo
is read from the git 'origin' remote -- this is a one-person tool filing
issues against its own repo, not a general-purpose GitHub client.

Scope note (FW47's own open question, "what does capturing logs mean"):
v1 attaches whatever small `context` dict the caller passes (current job
title/url/status), NOT a raw log-file dump -- a full log tail risks
including large/sensitive content (API responses, profile data) for a
feature whose whole point is a quick one-click report. Revisit if that
turns out to be too little context in practice.
"""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger(__name__)


def _resolve_repo() -> str | None:
    """"owner/repo" parsed from the git 'origin' remote URL, or None."""
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - git not available/no remote degrades to "can't file", not a crash
        return None
    # Handles both "https://github.com/owner/repo.git" and "git@github.com:owner/repo.git"
    tail = out.split("github.com")[-1].lstrip(":/").removesuffix(".git")
    return tail if "/" in tail else None


def file_bug_report(description: str, context: dict | None = None) -> tuple[bool, str]:
    """POST a new GitHub issue. Returns (ok, detail) -- detail is the issue
    URL on success, or a human-readable reason on failure. Never raises."""
    token = (os.environ.get("APPLYPILOT_GITHUB_TOKEN") or "").strip()
    if not token:
        return False, "APPLYPILOT_GITHUB_TOKEN not set"
    repo = _resolve_repo()
    if not repo:
        return False, "could not resolve a GitHub repo from the git 'origin' remote"

    body_lines = [description.strip() or "(no description provided)"]
    if context:
        body_lines.append("\n---")
        for k, v in context.items():
            if v:
                body_lines.append(f"**{k}:** {v}")

    try:
        import httpx

        resp = httpx.post(
            f"https://api.github.com/repos/{repo}/issues",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "title": f"[bug report] {(description.strip() or 'untitled')[:80]}",
                "body": "\n".join(body_lines),
                "labels": ["bug-report"],
            },
            timeout=15,
        )
    except Exception as e:  # noqa: BLE001 - network/httpx failure must degrade to a reported failure, not crash the banner handler
        log.warning("GitHub issue filing failed", exc_info=True)
        return False, str(e)[:200]

    if resp.status_code == 201:
        url = resp.json().get("html_url", "")
        log.info("Bug report filed: %s", url)
        return True, url
    log.warning("GitHub issue filing failed: %s %s", resp.status_code, resp.text[:200])
    return False, f"GitHub API {resp.status_code}: {resp.text[:200]}"
