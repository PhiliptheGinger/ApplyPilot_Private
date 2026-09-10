"""GitHub public-repo ingestion for candidate profile evidence.

CLAUDE.md decision #84 (2026-09-09): a first exploration found the
candidate's real GitHub account and two repos not yet in `profile.json`'s
`project_inventory`, but pinned a hard requirement before any of this ships
as a real pipeline: every candidate repo needs an explicit reputational-
visibility review before being used as resume evidence -- named examples
were a repo that scrapes resumes/automates job applications (ApplyPilot
itself), anything "semantically vulgar," or anything "pragmatically against
authority" (e.g. piracy tooling). This module is that gate.

Design, mirroring this codebase's established "regex for what's mechanical,
LLM only for what's genuinely semantic" split (scoring/deterministic_fallback.py
decision #76 and onward): a small deterministic keyword pre-filter catches
the most unambiguous, literal cases cheaply (profanity, piracy/cheating
terms, gambling, drug-culture slang) without needing a model call; a single
narrow LLM classification call per repo catches the genuinely semantic
categories a keyword list can't reliably see (anti-corporate sentiment,
political content, discriminatory content, automation/scraping tools that
could read as ToS-violating, etc.). Flags from either source are unioned
and shown to the user -- nothing is ever silently auto-excluded OR
auto-included; the user makes the final call on every single repo,
per the "show what was found, let them exclude anything" principle
(Future Work item 10c).

Sparse/stub repos (no real content yet) are also flagged and defaulted to
excluded, per the real "greensboro-data-coop" precedent from the same
session -- a repo whose own README said "Initial draft" with no real
functionality described.

NOT built here (see CLAUDE.md Future Work item 10): the categorization
step that turns raw repo/README data into schema-quality project_inventory
content is deliberately simple/deterministic for now (name + description +
top languages), not an LLM extraction pass -- that's a separate, harder
quality problem than the gating this module exists to solve, and a draft
this simple is easy for the user to spot-check and edit before it's ever
used to write real resume content.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
_HEADERS = {
    "User-Agent": "ApplyPilot/1.0 (profile-evidence-ingestion)",
    "Accept": "application/vnd.github+json",
}

# A repo with less real README content than this is treated as sparse/stub
# -- greensboro-data-coop (the real case that established this rule) had a
# README of well under 200 chars, explicitly saying "Initial draft."
MIN_README_LEN = 200

_STUB_MARKER_RE = re.compile(
    r"\binitial draft\b|\bwork in progress\b|\bwip\b|\bcoming soon\b|\bplaceholder\b|\bnothing here yet\b",
    re.IGNORECASE,
)


def _fetch_json(url: str, timeout: float = 20.0) -> dict | list | None:
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (404, 403):
            return None
        raise
    except (urllib.error.URLError, TimeoutError):
        log.warning("GitHub API request failed: %s", url)
        return None


def fetch_public_repos(username: str) -> list[dict]:
    """Public, non-fork repos for a GitHub username, most-recently-updated first."""
    data = _fetch_json(f"{GITHUB_API}/users/{username}/repos?per_page=100&sort=updated")
    if not data:
        return []
    return [r for r in data if not r.get("fork") and not r.get("private")]


def fetch_readme_text(username: str, repo_name: str) -> str:
    data = _fetch_json(f"{GITHUB_API}/repos/{username}/{repo_name}/readme")
    if not data or "content" not in data:
        return ""
    try:
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except (ValueError, UnicodeDecodeError):
        return ""


def fetch_languages(username: str, repo_name: str) -> dict:
    return _fetch_json(f"{GITHUB_API}/repos/{username}/{repo_name}/languages") or {}


def is_sparse_repo(readme: str) -> bool:
    """See MIN_README_LEN's docstring note -- the greensboro-data-coop precedent."""
    if len(readme.strip()) < MIN_README_LEN:
        return True
    return bool(_STUB_MARKER_RE.search(readme[:500]))


# ── Reputational flagging ────────────────────────────────────────────────

REPUTATIONAL_FLAG_CATEGORIES: dict[str, str] = {
    "explicit_language": "Profanity or vulgar language in the repo name, description, or README.",
    "adult_content": "Sexual or adult-oriented content or theming.",
    "hate_or_discriminatory": "Hate speech, slurs, or content targeting a protected group.",
    "violence_or_weapons": (
        "Content glorifying violence, or weapons-related tooling outside a clear professional/defense context."
    ),
    "illegal_or_circumvention": (
        "Piracy, DRM circumvention, exam-cheating tools, or other tools whose primary purpose is bypassing "
        "a legal or technical restriction."
    ),
    "anti_corporate_or_establishment": (
        "Content expressing hostility toward employers, corporations, or institutions in general -- including "
        "a personal grievance rant against a specific past employer."
    ),
    "political_or_controversial": (
        "Partisan political activism or a hot-button controversial topic, presented in a way likely to "
        "polarize a reader regardless of the candidate's actual position."
    ),
    "automation_or_scraping_tool": (
        "A bot, scraper, or automation tool whose purpose could read as circumventing a platform's terms of "
        "service -- e.g. automating job applications, scraping personal data, social-media bots."
    ),
    "privacy_invasive": "Surveillance, tracking, or stalkerware-adjacent tooling.",
    "substance_related": "Content centered on recreational drug use/culture, distinct from legitimate health/pharma work.",
    "gambling_related": "Gambling tools, betting bots, or casino-adjacent content.",
    "financial_manipulation": (
        "Market-manipulation-style tooling (e.g. pump-and-dump bots, wash-trading), distinct from legitimate "
        "fintech/trading-analysis projects."
    ),
}

# Deliberately narrow, high-confidence patterns for the categories that
# actually have reliable literal-keyword signals -- mirrors scorer.py's
# _TS_SCI_PATTERN/_CLEARANCE_REQUIRED_PATTERN design (deterministic where
# possible, never a blanket net). The remaining categories (anti-corporate
# sentiment, political content, automation/scraping intent, etc.) have no
# reliable keyword shape and are left entirely to the LLM pass below.
_OBVIOUS_FLAG_PATTERNS: dict[str, re.Pattern] = {
    "explicit_language": re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|cunt\w*|asshole\w*)\b", re.IGNORECASE),
    "illegal_or_circumvention": re.compile(
        r"\b(keygen|warez|crack(?:ed|z)?|drm[-_]?bypass|exam[-_]?(?:answers|solver|cheat)\w*|"
        r"chegg[-_]?(?:solver|answers)\w*|homework[-_]?cheat\w*)\b",
        re.IGNORECASE,
    ),
    "gambling_related": re.compile(r"\b(casino|sportsbook|betting[-_]?bot|slot[-_]?machine)\b", re.IGNORECASE),
    "substance_related": re.compile(r"\b(drug[-_]?deal\w*|weed[-_]?grow\w*|marijuana[-_]?grow\w*)\b", re.IGNORECASE),
}

_REPUTATIONAL_SYSTEM = """You are screening a candidate's PERSONAL GitHub repository to see if \
anything about it could hurt them in front of a recruiter or hiring manager, before it's used as \
resume/portfolio evidence in a job-application tool. Judge only the repo name, description, and \
README text given below -- ignore code you can't see.

Flag ANY of the following categories that GENUINELY apply. Be conservative -- only flag a real, \
non-trivial concern, not a stretch or a generic tech topic:

{categories}

First write ONE short sentence explaining your reasoning. Then, on its own final line, output \
either "FLAGS: none" or "FLAGS: category_one, category_two" using ONLY the exact category names \
listed above."""


def _repo_text(repo: dict, readme: str) -> str:
    return f"NAME: {repo.get('name')}\nDESCRIPTION: {repo.get('description') or '(none)'}\n\nREADME:\n{readme[:3000]}"


def classify_reputational_flags(client, repo_text: str) -> list[str]:
    """One narrow LLM call per repo -- mirrors scoring/deterministic_fallback.py's
    classify_family in shape and conservatism. Returns [] on any failure
    (never blocks the review flow; an LLM-classification failure just means
    the deterministic pre-filter is all that ran for this repo)."""
    categories = "\n".join(f"- {name}: {desc}" for name, desc in REPUTATIONAL_FLAG_CATEGORIES.items())
    messages = [
        {"role": "system", "content": _REPUTATIONAL_SYSTEM.format(categories=categories)},
        {"role": "user", "content": repo_text},
    ]
    try:
        resp = client.chat(messages, max_tokens=400, temperature=0.2)
    except Exception:  # noqa: BLE001
        return []
    m = re.search(r"FLAGS:\s*(.+)", resp or "", re.IGNORECASE)
    if not m or m.group(1).strip().lower().startswith("none"):
        return []
    found = [c.strip().lower() for c in m.group(1).split(",")]
    return sorted({c for c in found if c in REPUTATIONAL_FLAG_CATEGORIES})


def flag_repo(client, repo: dict, readme: str) -> list[str]:
    """Union of the deterministic pre-filter and the LLM pass."""
    text = _repo_text(repo, readme)
    flags = {cat for cat, pattern in _OBVIOUS_FLAG_PATTERNS.items() if pattern.search(text)}
    flags.update(classify_reputational_flags(client, text))
    return sorted(flags)


# ── Draft project_inventory entry ────────────────────────────────────────


def build_draft_project_entry(repo: dict, readme: str, languages: dict) -> dict:
    """Deliberately simple and deterministic (see module docstring) --
    a starting point for the user to review/edit, not a finished entry."""
    top_langs = sorted(languages, key=languages.get, reverse=True)[:3] if languages else []
    concepts = []
    if repo.get("description"):
        concepts.append(repo["description"])
    if top_langs:
        concepts.append("/".join(top_langs))
    if repo.get("name", "").endswith(".github.io"):
        concepts.append("Deployed via GitHub Pages")
    return {
        "name": repo.get("name", ""),
        "status": "web_project" if "HTML" in languages else "software_project",
        "resume_allowed": True,
        "relevance_categories": [lang.lower() for lang in top_langs[:2]] or ["software"],
        "evidence_level": "project experience",
        "factual_concepts": concepts,
        "constraints": [
            "Do not describe the candidate as an experienced professional engineer based solely on this project."
        ],
        "source": {"github_url": repo.get("html_url", "")},
    }


# ── Orchestration ────────────────────────────────────────────────────────


def gather_repo_review_items(username: str, client) -> list[dict]:
    """Fetch every public repo and compute its sparse/flag status, without
    prompting -- kept separate from the interactive review so both the
    wizard step and tests can drive the review loop independently."""
    items = []
    for repo in fetch_public_repos(username):
        readme = fetch_readme_text(username, repo["name"])
        languages = fetch_languages(username, repo["name"])
        items.append(
            {
                "repo": repo,
                "readme": readme,
                "languages": languages,
                "sparse": is_sparse_repo(readme),
                "flags": flag_repo(client, repo, readme),
            }
        )
    return items


def review_repos_interactively(items: list[dict]) -> list[dict]:
    """Prints each repo with its sparse/flag status and asks the user
    whether to keep it. Returns the subset the user chose to keep. Nothing
    is ever silently auto-excluded or auto-included -- every repo gets an
    explicit yes/no, just with a sensible default (per decision #84's
    review principle)."""
    from rich.console import Console
    from rich.prompt import Confirm

    console = Console()
    kept = []
    for item in items:
        repo = item["repo"]
        console.print(f"\n[bold]{repo['name']}[/bold] — {repo.get('description') or '(no description)'}")
        if item["sparse"]:
            console.print("[yellow]  Looks unfinished (thin README) — excluded by default.[/yellow]")
        for flag in item["flags"]:
            console.print(f"[red]  FLAGGED ({flag}):[/red] {REPUTATIONAL_FLAG_CATEGORIES[flag]}")
        default_keep = not item["sparse"] and not item["flags"]
        if Confirm.ask(f"  Include {repo['name']} as resume/portfolio evidence?", default=default_keep):
            kept.append(item)
    return kept


def import_github_projects(username: str, client=None) -> list[dict]:
    """Fetch, flag, and interactively review a candidate's public GitHub
    repos. Returns draft project_inventory entries for whatever the user
    chose to keep -- does NOT write to profile.json itself; the caller
    (wizard step or CLI command) decides where/whether to persist them."""
    if client is None:
        from applypilot.llm import get_client

        client = get_client(quality=False)

    items = gather_repo_review_items(username, client)
    kept = review_repos_interactively(items)
    return [build_draft_project_entry(item["repo"], item["readme"], item["languages"]) for item in kept]
