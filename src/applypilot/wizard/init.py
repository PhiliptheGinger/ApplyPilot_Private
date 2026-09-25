"""ApplyPilot first-time setup wizard.

Interactive flow that creates ~/.applypilot/ with:
  - resume.txt (and optionally resume.pdf)
  - profile.json
  - searches.yaml
  - .env (LLM API key)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from applypilot.config import (
    APP_DIR,
    ENV_PATH,
    FILES_DIR,
    PROFILE_PATH,
    RESUME_PATH,
    RESUME_PDF_PATH,
    SEARCH_CONFIG_PATH,
    ensure_dirs,
)

console = Console()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def _setup_resume() -> None:
    """Prompt for resume file and copy into APP_DIR."""
    console.print(Panel("[bold]Step 1: Resume[/bold]\nPoint to your master resume file (.txt or .pdf)."))

    while True:
        path_str = Prompt.ask("Resume file path")
        src = Path(path_str.strip().strip('"').strip("'")).expanduser().resolve()

        if not src.exists():
            console.print(f"[red]File not found:[/red] {src}")
            continue

        suffix = src.suffix.lower()
        if suffix not in (".txt", ".pdf"):
            console.print("[red]Unsupported format.[/red] Provide a .txt or .pdf file.")
            continue

        if suffix == ".txt":
            shutil.copy2(src, RESUME_PATH)
            console.print(f"[green]Copied to {RESUME_PATH}[/green]")
        elif suffix == ".pdf":
            shutil.copy2(src, RESUME_PDF_PATH)
            console.print(f"[green]Copied to {RESUME_PDF_PATH}[/green]")

            # Also ask for a plain-text version for LLM consumption
            txt_path_str = Prompt.ask(
                "Plain-text version of your resume (.txt)",
                default="",
            )
            if txt_path_str.strip():
                txt_src = Path(txt_path_str.strip().strip('"').strip("'")).expanduser().resolve()
                if txt_src.exists():
                    shutil.copy2(txt_src, RESUME_PATH)
                    console.print(f"[green]Copied to {RESUME_PATH}[/green]")
                else:
                    console.print("[yellow]File not found, skipping plain-text copy.[/yellow]")
        break


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def _setup_profile() -> dict:
    """Walk through profile questions and return a nested profile dict."""
    console.print(
        Panel(
            "[bold]Step 2: Profile[/bold]\nTell ApplyPilot about yourself. This powers scoring, tailoring, and auto-fill."
        )
    )

    profile: dict = {}

    # -- Personal --
    console.print("\n[bold cyan]Personal Information[/bold cyan]")
    profile["personal"] = {
        "full_name": Prompt.ask("Full name"),
        "email": Prompt.ask("Email address"),
        "phone": Prompt.ask("Phone number", default=""),
        "city": Prompt.ask("City"),
        "country": Prompt.ask("Country"),
        "linkedin_url": Prompt.ask("LinkedIn URL", default=""),
        "password": Prompt.ask("Job site password (used for login walls during auto-apply)", password=True, default=""),
    }

    # -- Work Authorization --
    console.print("\n[bold cyan]Work Authorization[/bold cyan]")
    profile["work_authorization"] = {
        "legally_authorized": Confirm.ask("Are you legally authorized to work in your target country?"),
        "needs_sponsorship": Confirm.ask("Will you now or in the future need sponsorship?"),
    }

    # -- Compensation --
    console.print("\n[bold cyan]Compensation[/bold cyan]")
    salary = Prompt.ask("Expected annual salary (number)", default="")
    salary_currency = Prompt.ask("Currency", default="USD")
    salary_range = Prompt.ask("Acceptable range (e.g. 80000-120000)", default="")
    range_parts = salary_range.split("-") if "-" in salary_range else [salary, salary]
    profile["compensation"] = {
        "salary_expectation": salary,
        "salary_currency": salary_currency,
        "salary_range_min": range_parts[0].strip(),
        "salary_range_max": range_parts[1].strip() if len(range_parts) > 1 else range_parts[0].strip(),
    }

    # -- Experience --
    console.print("\n[bold cyan]Experience[/bold cyan]")
    profile["experience"] = {
        "years_of_experience_total": Prompt.ask("Years of professional experience", default=""),
        "education_level": Prompt.ask("Highest education (e.g. Bachelor's, Master's, PhD, Self-taught)", default=""),
        "current_title": Prompt.ask("Current/most recent job title", default=""),
    }

    # -- Skills Boundary --
    console.print("\n[bold cyan]Skills[/bold cyan] (comma-separated)")
    langs = Prompt.ask("Programming languages", default="")
    frameworks = Prompt.ask("Frameworks & libraries", default="")
    tools = Prompt.ask("Tools & platforms (e.g. Docker, AWS, Git)", default="")
    profile["skills_boundary"] = {
        "programming_languages": [s.strip() for s in langs.split(",") if s.strip()],
        "frameworks": [s.strip() for s in frameworks.split(",") if s.strip()],
        "tools": [s.strip() for s in tools.split(",") if s.strip()],
    }

    # -- Resume Facts (preserved truths for tailoring) --
    console.print("\n[bold cyan]Resume Facts[/bold cyan]")
    console.print("[dim]These are preserved exactly during resume tailoring — the AI will never change them.[/dim]")
    companies = Prompt.ask("Companies to always keep (comma-separated)", default="")
    projects = Prompt.ask("Projects to always keep (comma-separated)", default="")
    school = Prompt.ask("School name(s) to preserve", default="")
    metrics = Prompt.ask("Real metrics to preserve (e.g. '99.9% uptime, 50k users')", default="")
    profile["resume_facts"] = {
        "preserved_companies": [s.strip() for s in companies.split(",") if s.strip()],
        "preserved_projects": [s.strip() for s in projects.split(",") if s.strip()],
        "preserved_school": school.strip(),
        "real_metrics": [s.strip() for s in metrics.split(",") if s.strip()],
    }

    # -- EEO Voluntary (defaults) --
    profile["eeo_voluntary"] = {
        "gender": "Decline to self-identify",
        "ethnicity": "Decline to self-identify",
        "veteran_status": "Decline to self-identify",
        "disability_status": "Decline to self-identify",
    }

    # -- Availability --
    profile["availability"] = {
        "earliest_start_date": Prompt.ask("Earliest start date", default="Immediately"),
    }

    # Save
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]Profile saved to {PROFILE_PATH}[/green]")

    _verify_email(profile)

    return profile


def _verify_email(profile: dict) -> None:
    """Best-effort: send a real test email to prove the collected address is reachable.

    2026-09-24 (CLAUDE.md decision #197 follow-up, Future Work item 57):
    catches typos in the candidate's email before anything downstream
    (tracking, HITL notifications) silently relies on a bad address.
    Requires Gmail MCP to already be set up (decision #173) -- if it
    isn't, this skips gracefully rather than forcing that setup here.
    """
    email = profile.get("personal", {}).get("email", "").strip()
    if not email:
        return

    from applypilot.tracking.gmail_client import check_gmail_setup

    ok, _msg = check_gmail_setup()
    if not ok:
        console.print(
            "[dim]Gmail integration isn't set up yet, so email can't be verified right now. "
            "Set it up later (see scripts/gmail_oauth.py) and re-run [bold]applypilot init[/bold] to verify.[/dim]"
        )
        return

    if not Confirm.ask(f"Send a test email to {email} to confirm it's correct?", default=True):
        return

    import asyncio

    from applypilot.tracking.gmail_client import send_email

    sent_ok, detail = asyncio.run(
        send_email(
            to=[email],
            subject="ApplyPilot setup — email verification",
            body="This confirms ApplyPilot can send to this address. No action needed.",
        )
    )
    if not sent_ok:
        console.print(f"[yellow]Could not send test email: {detail}[/yellow]")
        return

    if Confirm.ask("Check your inbox — did it arrive?", default=True):
        console.print("[green]Email confirmed.[/green]")
    else:
        console.print(f"[yellow]Double-check the address in {PROFILE_PATH} and re-run [bold]applypilot init[/bold].[/yellow]")


# ---------------------------------------------------------------------------
# Search config
# ---------------------------------------------------------------------------


def _setup_searches() -> None:
    """Generate a searches.yaml from user input."""
    console.print(Panel("[bold]Step 3: Job Search Config[/bold]\nDefine what you're looking for."))

    location = Prompt.ask("Target location (e.g. 'Remote', 'Canada', 'New York, NY')", default="Remote")
    distance_str = Prompt.ask("Search radius in miles (0 for remote-only)", default="0")
    try:
        distance = int(distance_str)
    except ValueError:
        distance = 0

    roles_raw = Prompt.ask("Target job titles (comma-separated, e.g. 'Backend Engineer, Full Stack Developer')")
    roles = [r.strip() for r in roles_raw.split(",") if r.strip()]

    if not roles:
        console.print("[yellow]No roles provided. Using a default set.[/yellow]")
        roles = ["Software Engineer"]

    # Build YAML content
    lines = [
        "# ApplyPilot search configuration",
        "# Edit this file to refine your job search queries.",
        "",
        "defaults:",
        f'  location: "{location}"',
        f"  distance: {distance}",
        "  hours_old: 72",
        "  results_per_site: 50",
        "",
        "locations:",
        f'  - location: "{location}"',
        f"    remote: {str(distance == 0).lower()}",
        "",
        "queries:",
    ]
    for i, role in enumerate(roles):
        lines.append(f'  - query: "{role}"')
        lines.append(f"    tier: {min(i + 1, 3)}")

    SEARCH_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Search config saved to {SEARCH_CONFIG_PATH}[/green]")


# ---------------------------------------------------------------------------
# AI Features
# ---------------------------------------------------------------------------


def _setup_ai_features() -> None:
    """Ask about AI scoring/tailoring — optional LLM configuration."""
    console.print(
        Panel(
            "[bold]Step 4: AI Features (optional)[/bold]\n"
            "An LLM powers job scoring, resume tailoring, and cover letters.\n"
            "Without this, you can still discover and enrich jobs."
        )
    )

    if not Confirm.ask("Enable AI scoring and resume tailoring?", default=True):
        console.print("[dim]Discovery-only mode. You can configure AI later with [bold]applypilot init[/bold].[/dim]")
        return

    console.print("Supported providers: [bold]Gemini[/bold] (recommended, free tier), OpenAI, local (Ollama/llama.cpp)")
    provider = Prompt.ask(
        "Provider",
        choices=["gemini", "openai", "local"],
        default="gemini",
    )

    env_lines = ["# ApplyPilot configuration", ""]

    if provider == "gemini":
        api_key = Prompt.ask("Gemini API key (from aistudio.google.com)")
        model = Prompt.ask("Model", default="gemini-3.6-flash")
        env_lines.append(f"GEMINI_API_KEY={api_key}")
        env_lines.append(f"LLM_MODEL={model}")
    elif provider == "openai":
        api_key = Prompt.ask("OpenAI API key")
        model = Prompt.ask("Model", default="gpt-4o-mini")
        env_lines.append(f"OPENAI_API_KEY={api_key}")
        env_lines.append(f"LLM_MODEL={model}")
    elif provider == "local":
        url = Prompt.ask("Local LLM endpoint URL", default="http://localhost:8080/v1")
        model = Prompt.ask("Model name", default="local-model")
        env_lines.append(f"LLM_URL={url}")
        env_lines.append(f"LLM_MODEL={model}")

    env_lines.append("")
    ENV_PATH.write_text("\n".join(env_lines), encoding="utf-8")
    console.print(f"[green]AI configuration saved to {ENV_PATH}[/green]")


# ---------------------------------------------------------------------------
# Auto-Apply
# ---------------------------------------------------------------------------


def _setup_auto_apply() -> None:
    """Configure autonomous job application (requires Claude Code CLI)."""
    console.print(
        Panel(
            "[bold]Step 5: Auto-Apply (optional)[/bold]\n"
            "ApplyPilot can autonomously fill and submit job applications\n"
            "using Claude Code as the browser agent."
        )
    )

    if not Confirm.ask("Enable autonomous job applications?", default=True):
        console.print("[dim]You can apply manually using the tailored resumes ApplyPilot generates.[/dim]")
        return

    # Check for Claude Code CLI
    if shutil.which("claude"):
        console.print("[green]Claude Code CLI detected.[/green]")
    else:
        console.print(
            "[yellow]Claude Code CLI not found on PATH.[/yellow]\n"
            "Install it from: [bold]https://claude.ai/code[/bold]\n"
            "Auto-apply won't work until Claude Code is installed."
        )

    # Optional: CapSolver for CAPTCHAs
    console.print("\n[dim]Some job sites use CAPTCHAs. CapSolver can handle them automatically.[/dim]")
    if Confirm.ask("Configure CapSolver API key? (optional)", default=False):
        capsolver_key = Prompt.ask("CapSolver API key")
        # Append to existing .env or create
        if ENV_PATH.exists():
            existing = ENV_PATH.read_text(encoding="utf-8")
            if "CAPSOLVER_API_KEY" not in existing:
                ENV_PATH.write_text(
                    existing.rstrip() + f"\nCAPSOLVER_API_KEY={capsolver_key}\n",
                    encoding="utf-8",
                )
        else:
            ENV_PATH.write_text(f"# ApplyPilot configuration\nCAPSOLVER_API_KEY={capsolver_key}\n", encoding="utf-8")
        console.print("[green]CapSolver key saved.[/green]")
    else:
        console.print("[dim]Skipped. Add CAPSOLVER_API_KEY to .env later if needed.[/dim]")


# ---------------------------------------------------------------------------
# SMS relay (optional)
# ---------------------------------------------------------------------------


def _setup_sms_relay(profile: dict) -> None:
    """SMS relay for verification codes (decisions #197/#199/#200) -- dispatches
    to whichever provider the user picks.

    Relay-only in every case, matching sms_client.py/google_voice_client.py/
    adb_sms_client.py's own scope: this collects and verifies a way to READ a
    code once one arrives, but does not wire anything into automatic
    login/2FA submission -- that stays gated on the bot-detection-risk
    research in CLAUDE.md Future Work item 43.

    Any credentials collected go in ~/.applypilot/.env only (never
    profile.json, never the git repo), same as every other API key in this
    wizard.
    """
    console.print(
        Panel(
            "[bold]Step 6: SMS Relay (optional)[/bold]\n"
            "Some job applications require SMS/phone verification the agent can't complete on "
            "its own -- it will always pause and hand these to you. This step lets it also *read* "
            "a code once one arrives, so you can enter it without leaving your workflow.\n\n"
            "[dim]Three options: [bold]Google Voice[/bold] (free for US residents, forwards texts "
            "into the Gmail integration you may already have set up), [bold]ADB[/bold] (free, reads "
            "SMS directly off an Android phone connected via USB -- only works while it's connected), "
            "or [bold]Twilio[/bold] (a paid, dedicated API number — a few dollars a month).[/dim]"
        )
    )

    choice = Prompt.ask(
        "Which method?",
        choices=["google-voice", "adb", "twilio", "skip"],
        default="google-voice",
    )
    if choice == "skip":
        console.print("[dim]Skipped. Run [bold]applypilot sms --setup[/bold] later, or come back to this step.[/dim]")
        return
    if choice == "google-voice":
        _setup_google_voice_relay(profile)
        return
    if choice == "adb":
        _setup_adb_relay(profile)
        return
    _setup_twilio_relay(profile)


def _setup_adb_relay(profile: dict) -> None:
    """Free SMS relay via ADB, reading SMS directly off a connected Android phone.

    Generalized from a prior personal project (~/Projects/Haywood) that used
    the same `content://sms` ADB query technique. No credentials, no service
    account -- but only works while the phone is connected via USB (or
    wireless ADB on the same network), unlike the always-on Google
    Voice/Twilio relays.
    """
    console.print(
        "\n[bold cyan]ADB setup[/bold cyan] (free, Android only, phone must be connected when reading):\n"
        "  1. On your phone: Settings -> About phone -> tap 'Build number' 7 times to enable Developer options\n"
        "  2. Settings -> Developer options -> turn on [bold]USB debugging[/bold]\n"
        "  3. Connect the phone via USB and accept the \"Allow USB debugging?\" prompt on the phone\n"
    )

    from applypilot.tracking.adb_sms_client import (
        _find_adb,
        check_adb_setup,
        get_latest_verification_code,
        search_common_install_locations,
    )

    if not Confirm.ask("Is the phone connected now with USB debugging enabled?", default=False):
        console.print("[dim]Come back to this later — re-run [bold]applypilot init[/bold] or [bold]applypilot sms --setup[/bold] once it's connected.[/dim]")
        return

    # adb genuinely installed but not on PATH is a real, common case (2026-09-24
    # live incident) -- offer to save a found copy rather than just saying "not found".
    if _find_adb() is None:
        found = search_common_install_locations()
        if found and Confirm.ask(f"adb isn't on your PATH, but found a copy at {found} — save this so ApplyPilot can use it?", default=True):
            env_block = f"\nAPPLYPILOT_ADB_PATH={found}\n"
            if ENV_PATH.exists():
                existing = ENV_PATH.read_text(encoding="utf-8")
                if "APPLYPILOT_ADB_PATH" not in existing:
                    ENV_PATH.write_text(existing.rstrip() + "\n" + env_block, encoding="utf-8")
            else:
                ENV_PATH.write_text("# ApplyPilot configuration\n" + env_block, encoding="utf-8")
            import os

            os.environ["APPLYPILOT_ADB_PATH"] = found
            console.print(f"[green]Saved to {ENV_PATH}[/green]")

    ok, msg = check_adb_setup()
    if not ok:
        console.print(f"[red]{msg}[/red]")
        return
    console.print(f"[green]{msg}[/green]")

    if not Confirm.ask("Text your phone now from another number, then press Enter here to check?", default=True):
        return

    console.print("[dim]Checking recent SMS on the connected phone...[/dim]")
    code = get_latest_verification_code()
    if code:
        console.print(f"[green]Found it — read a recent code ({code}) directly from the phone. ADB relay confirmed working.[/green]")
    else:
        console.print("[yellow]Didn't find a recent matching text — try again, or double-check the connection.[/yellow]")


def _setup_google_voice_relay(profile: dict) -> None:
    """Free SMS relay via a Google Voice number forwarding texts into Gmail.

    No credentials to collect -- this just reads through the existing
    Gmail integration (gmail_client.py), so setup is two manual steps on
    Google's side plus a live confirmation, not an API key.
    """
    console.print(
        "\n[bold cyan]Google Voice setup[/bold cyan] (free, no code required on your end):\n"
        "  1. If you don't have one, get a free number at [bold]voice.google.com[/bold]\n"
        "  2. Go to [bold]voice.google.com[/bold] -> Settings (gear icon) -> Messages -> "
        "turn on [bold]\"Forward messages to email\"[/bold]\n"
    )

    from applypilot.tracking.gmail_client import check_gmail_setup

    ok, msg = check_gmail_setup()
    if not ok:
        console.print(
            f"[yellow]Gmail integration isn't set up yet, so Google Voice forwarding has nowhere to land: {msg}[/yellow]\n"
            "[dim]Set up Gmail first (see scripts/gmail_oauth.py), then re-run [bold]applypilot init[/bold].[/dim]"
        )
        return

    if not Confirm.ask("Have you completed both steps above?", default=False):
        console.print("[dim]Come back to this later — re-run [bold]applypilot init[/bold] once forwarding is on.[/dim]")
        return

    if not Confirm.ask("Text your Google Voice number now from another phone, then press Enter here to check?", default=True):
        console.print("[dim]Skipped the live check — this should work once forwarding is on.[/dim]")
        return

    import asyncio

    from applypilot.tracking.google_voice_client import get_latest_verification_code

    console.print("[dim]Checking Gmail for a recent forwarded text...[/dim]")
    code = asyncio.run(get_latest_verification_code())
    if code:
        console.print(f"[green]Found it — read a recent code ({code}) from a forwarded text. Google Voice relay confirmed working.[/green]")
    else:
        console.print(
            "[yellow]Didn't find a recent forwarded text. This can take a minute to arrive — "
            "try [bold]applypilot sms --setup[/bold] again shortly, or double-check the forwarding toggle.[/yellow]"
        )


def _setup_twilio_relay(profile: dict) -> None:
    """Paid SMS relay via a dedicated Twilio number (decision #197)."""
    sid = Prompt.ask("Twilio Account SID (from the Console dashboard)").strip()
    token = Prompt.ask("Twilio Auth Token", password=True).strip()
    relay_number = Prompt.ask("Twilio phone number (e.g. +15551234567)").strip()

    if not (sid and token and relay_number):
        console.print("[yellow]Incomplete — skipping SMS relay setup.[/yellow]")
        return

    env_block = f"\nTWILIO_ACCOUNT_SID={sid}\nTWILIO_AUTH_TOKEN={token}\nTWILIO_PHONE_NUMBER={relay_number}\n"
    if ENV_PATH.exists():
        existing = ENV_PATH.read_text(encoding="utf-8")
        if "TWILIO_ACCOUNT_SID" not in existing:
            ENV_PATH.write_text(existing.rstrip() + "\n" + env_block, encoding="utf-8")
    else:
        ENV_PATH.write_text("# ApplyPilot configuration\n" + env_block, encoding="utf-8")
    console.print(f"[green]Twilio credentials saved to {ENV_PATH}[/green]")

    # Re-load so the freshly-written keys are visible to sms_client this run.
    import os

    os.environ["TWILIO_ACCOUNT_SID"] = sid
    os.environ["TWILIO_AUTH_TOKEN"] = token
    os.environ["TWILIO_PHONE_NUMBER"] = relay_number

    from applypilot.tracking.sms_client import send_test_message, verify_connection

    console.print("[dim]Testing Twilio connection...[/dim]")
    if not verify_connection():
        console.print("[red]Connection failed — check the Account SID / Auth Token and try again.[/red]")
        return
    console.print("[green]Connected.[/green]")

    personal_phone = profile.get("personal", {}).get("phone", "").strip()
    if not personal_phone:
        console.print("[dim]No personal phone number on file to send a test message to — skipping that check.[/dim]")
        return

    if not Confirm.ask(f"Send a real test text to {personal_phone} to confirm the relay works end-to-end?", default=True):
        return

    sent_ok, detail = send_test_message(personal_phone)
    if not sent_ok:
        console.print(f"[yellow]Could not send test message: {detail}[/yellow]")
        return

    if Confirm.ask("Check your phone — did it arrive?", default=True):
        console.print("[green]SMS relay confirmed working end-to-end.[/green]")
    else:
        console.print("[yellow]Double-check the phone number and Twilio number's SMS capability, then re-run [bold]applypilot sms --setup[/bold].[/yellow]")


# ---------------------------------------------------------------------------
# Optional documents
# ---------------------------------------------------------------------------

_OPTIONAL_FILE_KEYS = [
    ("profile_photo", "Profile photo / headshot", [".jpg", ".jpeg", ".png", ".webp"]),
    ("id_document", "Government-issued ID scan", [".pdf", ".jpg", ".jpeg", ".png"]),
    ("passport", "Passport scan", [".pdf", ".jpg", ".jpeg", ".png"]),
]


def _setup_optional_files(profile: dict) -> None:
    """Optionally copy documents into ~/.applypilot/files/ and record paths in profile."""
    console.print(
        Panel(
            "[bold]Step 7: Optional Documents (skip if not needed)[/bold]\n"
            "Profile photo, ID, passport, certificates — some applications ask for these.\n"
            "Files are copied to [cyan]~/.applypilot/files/[/cyan] for use by the apply agent."
        )
    )

    if not Confirm.ask("Do you have any optional documents to add?", default=False):
        console.print("[dim]Skipped. Add files to ~/.applypilot/files/ and update profile.json later.[/dim]")
        return

    files: dict[str, str] = profile.get("files", {})
    FILES_DIR.mkdir(parents=True, exist_ok=True)

    # Known file types
    for key, label, allowed_exts in _OPTIONAL_FILE_KEYS:
        if not Confirm.ask(f"Add {label}?", default=False):
            continue
        while True:
            path_str = Prompt.ask(f"{label} file path")
            src = Path(path_str.strip().strip('"').strip("'")).expanduser().resolve()
            if not src.exists():
                console.print(f"[red]File not found:[/red] {src}")
                continue
            if src.suffix.lower() not in allowed_exts:
                console.print(f"[red]Unsupported format.[/red] Use one of: {', '.join(allowed_exts)}")
                continue
            dest = FILES_DIR / f"{key}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            files[key] = f"~/.applypilot/files/{dest.name}"
            console.print(f"[green]Copied to {dest}[/green]")
            break

    # Free-form: certificates / other documents
    while Confirm.ask("Add another document (certificate, portfolio, etc.)?", default=False):
        doc_label = Prompt.ask("Short label for this document (e.g. 'aws_cert', 'portfolio')")
        key = doc_label.strip().lower().replace(" ", "_")
        while True:
            path_str = Prompt.ask("File path")
            src = Path(path_str.strip().strip('"').strip("'")).expanduser().resolve()
            if not src.exists():
                console.print(f"[red]File not found:[/red] {src}")
                continue
            dest = FILES_DIR / f"{key}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            files[key] = f"~/.applypilot/files/{dest.name}"
            console.print(f"[green]Copied to {dest}[/green]")
            break

    if files:
        profile["files"] = files
        PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print("[green]Document paths saved to profile.json[/green]")


# ---------------------------------------------------------------------------
# GitHub project import (optional)
# ---------------------------------------------------------------------------


def _setup_github_import(profile: dict) -> None:
    """Optionally pull public GitHub repos as project_inventory evidence.

    2026-09-09 (CLAUDE.md decision #84/85): opt-in, off by default -- this
    step makes a network call and (for reputational flagging) an LLM call
    per repo, and every repo gets an explicit human review before it's
    added, per the "show what was found, let them exclude anything"
    principle. See discovery/github_profile.py for the fetch/flag/review
    implementation.
    """
    console.print(
        Panel(
            "[bold]Step 8: Import GitHub Projects (optional)[/bold]\n"
            "Pull your public GitHub repos as candidate project evidence. Each repo is "
            "screened for reputational concerns (vulgar language, anti-corporate/political "
            "content, automation tools that could look ToS-violating, etc.) and shown to you "
            "before anything is added -- nothing is included automatically."
        )
    )

    if not Confirm.ask("Import projects from a public GitHub account?", default=False):
        console.print("[dim]Skipped. You can add projects to profile.json manually later.[/dim]")
        return

    username = Prompt.ask("GitHub username").strip()
    if not username:
        return

    from applypilot.discovery.github_profile import import_github_projects

    try:
        entries = import_github_projects(username)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]GitHub import failed: {e}[/red]")
        return

    if not entries:
        console.print("[dim]No projects were added.[/dim]")
        return

    profile.setdefault("project_inventory", []).extend(entries)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Added {len(entries)} project(s) to profile.json[/green]")
    console.print("[dim]Review the generated entries in profile.json -- they're a starting point, not final copy.[/dim]")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_wizard() -> None:
    """Run the full interactive setup wizard."""
    console.print()
    console.print(
        Panel.fit(
            "[bold green]ApplyPilot Setup Wizard[/bold green]\n\n"
            "This will create your configuration at:\n"
            f"  [cyan]{APP_DIR}[/cyan]\n\n"
            "You can re-run this anytime with [bold]applypilot init[/bold].",
            border_style="green",
        )
    )

    ensure_dirs()
    console.print(f"[dim]Created {APP_DIR}[/dim]\n")

    # Step 1: Resume
    _setup_resume()
    console.print()

    # Step 2: Profile
    profile = _setup_profile()
    console.print()

    # Step 3: Search config
    _setup_searches()
    console.print()

    # Step 4: AI features (optional LLM)
    _setup_ai_features()
    console.print()

    # Step 5: Auto-apply (Claude Code detection)
    _setup_auto_apply()
    console.print()

    # Step 6: SMS relay (optional)
    _setup_sms_relay(profile)
    console.print()

    # Step 7: Optional documents (profile photo, ID, certs)
    _setup_optional_files(profile)
    console.print()

    # Step 8: GitHub project import (optional)
    _setup_github_import(profile)
    console.print()

    # Done — show tier status
    from applypilot.config import TIER_COMMANDS, TIER_LABELS, get_tier

    tier = get_tier()

    tier_lines: list[str] = []
    for t in range(1, 4):
        label = TIER_LABELS[t]
        cmds = ", ".join(f"[bold]{c}[/bold]" for c in TIER_COMMANDS[t])
        if t <= tier:
            tier_lines.append(f"  [green]✓ Tier {t} — {label}[/green]  ({cmds})")
        elif t == tier + 1:
            tier_lines.append(f"  [yellow]→ Tier {t} — {label}[/yellow]  ({cmds})")
        else:
            tier_lines.append(f"  [dim]✗ Tier {t} — {label}  ({cmds})[/dim]")

    unlock_hint = ""
    if tier == 1:
        unlock_hint = "\n[dim]To unlock Tier 2: configure an LLM API key (re-run [bold]applypilot init[/bold]).[/dim]"
    elif tier == 2:
        unlock_hint = "\n[dim]To unlock Tier 3: install Claude Code CLI + Chrome.[/dim]"

    console.print(
        Panel.fit(
            "[bold green]Setup complete![/bold green]\n\n"
            f"[bold]Your tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]\n\n" + "\n".join(tier_lines) + unlock_hint,
            border_style="green",
        )
    )
