"""Deterministic apply engine (Option 1).

This engine uses direct Playwright actions instead of spawning a Claude
subprocess. It is intentionally conservative:
- Supports a minimal Greenhouse flow first.
- Escalates CAPTCHAs to human intervention immediately.
- Uses profile/application_profile facts only (no hallucinated answers).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from applypilot import config
from applypilot.apply.chrome import detect_ats

logger = logging.getLogger(__name__)


def _get_profile_fields() -> dict[str, str]:
    profile = config.load_profile()
    personal = profile.get("personal", {})
    app_prof = profile.get("application_profile", {})
    online = app_prof.get("online_profiles", {})

    full_name = str(personal.get("full_name") or "").strip()
    parts = full_name.split()
    first = parts[0] if parts else ""
    last = parts[-1] if len(parts) > 1 else ""

    return {
        "first_name": first,
        "last_name": last,
        "email": str(personal.get("email") or "").strip(),
        "phone": str(personal.get("phone") or "").strip(),
        "city": str(personal.get("city") or "").strip(),
        "linkedin": str(online.get("linkedin") or personal.get("linkedin_url") or "").strip(),
        "website": str(online.get("website") or personal.get("website_url") or personal.get("portfolio_url") or "").strip(),
    }


def _safe_fill(page, selectors: list[str], value: str) -> bool:
    if not value:
        return False
    for sel in selectors:
        locator = page.locator(sel)
        if locator.count() > 0:
            try:
                locator.first.fill(value, timeout=1200)
                return True
            except Exception:
                continue
    return False


def _has_captcha(page) -> bool:
    return bool(
        page.locator(
            "iframe[src*='recaptcha'], .g-recaptcha, iframe[src*='hcaptcha'], .h-captcha, iframe[src*='turnstile'], [data-sitekey]"
        ).count()
    )


def _upload_resume_if_present(page, job: dict) -> bool:
    resume_path = job.get("tailored_resume_path")
    if not resume_path:
        return False

    src = Path(resume_path)
    if not src.exists():
        return False

    upload_targets = [
        "input[type='file'][name*='resume']",
        "input[type='file'][id*='resume']",
        "input[type='file']",
    ]
    for sel in upload_targets:
        locator = page.locator(sel)
        if locator.count() > 0:
            try:
                locator.first.set_input_files(str(src), timeout=3000)
                return True
            except Exception:
                continue
    return False


def _run_greenhouse(page, job: dict, dry_run: bool) -> tuple[str, int, list[dict]]:
    started = time.time()
    fields = _get_profile_fields()

    _safe_fill(page, [
        "input[name='first_name']",
        "input[name='job_application[first_name]']",
        "#first_name",
    ], fields["first_name"])
    _safe_fill(page, [
        "input[name='last_name']",
        "input[name='job_application[last_name]']",
        "#last_name",
    ], fields["last_name"])
    _safe_fill(page, [
        "input[name='email']",
        "input[name='job_application[email]']",
        "#email",
    ], fields["email"])
    _safe_fill(page, [
        "input[name='phone']",
        "input[name='job_application[phone]']",
        "input[type='tel']",
        "#phone",
    ], fields["phone"])
    _safe_fill(page, [
        "input[name='location']",
        "input[name='job_application[location]']",
        "#location",
    ], fields["city"])
    _safe_fill(page, [
        "input[name='linkedin']",
        "input[name='job_application[linkedin]']",
        "input[name*='linkedin']",
    ], fields["linkedin"])
    _safe_fill(page, [
        "input[name='website']",
        "input[name='job_application[website]']",
        "input[name*='portfolio']",
        "input[name*='url']",
    ], fields["website"])

    _upload_resume_if_present(page, job)

    if _has_captcha(page):
        dur = int((time.time() - started) * 1000)
        return f"needs_human:captcha:{page.url}", dur, []

    if dry_run:
        dur = int((time.time() - started) * 1000)
        return "applied", dur, []

    submit_buttons = [
        "button[type='submit']",
        "#submit_app",
        "button:has-text('Submit')",
        "button:has-text('Apply')",
    ]
    for sel in submit_buttons:
        btn = page.locator(sel)
        if btn.count() > 0:
            try:
                btn.first.click(timeout=2500)
                page.wait_for_timeout(1500)
                dur = int((time.time() - started) * 1000)
                return "applied", dur, []
            except Exception:
                continue

    dur = int((time.time() - started) * 1000)
    return f"needs_human:review_required:{page.url}", dur, []


def run_job_deterministic(
    job: dict,
    port: int,
    worker_id: int = 0,
    dry_run: bool = False,
    skip_tab_reset: bool = False,
    extra_context: str | None = None,
) -> tuple[str, int, list[dict]]:
    """Run one job using deterministic browser actions.

    Returns the same status contract as launcher.run_job.
    """
    del worker_id, skip_tab_reset, extra_context  # kept for signature parity

    started = time.time()
    apply_url = job.get("application_url") or job.get("url") or ""
    if not apply_url:
        return "failed:no_application_url", 0, []

    ats = detect_ats(apply_url)
    if ats != "greenhouse":
        return f"needs_human:unsupported_ats:{apply_url}", 0, []

    try:
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            if browser.contexts:
                context = browser.contexts[0]
            else:
                context = browser.new_context()

            page = context.pages[0] if context.pages else context.new_page()
            page.goto(apply_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(1200)

            if _has_captcha(page):
                dur = int((time.time() - started) * 1000)
                return f"needs_human:captcha:{page.url}", dur, []

            return _run_greenhouse(page, job, dry_run=dry_run)

    except PlaywrightTimeoutError:
        dur = int((time.time() - started) * 1000)
        return "failed:timeout", dur, []
    except Exception as exc:
        logger.exception("deterministic engine failed")
        dur = int((time.time() - started) * 1000)
        return f"failed:deterministic_error:{str(exc)[:80]}", dur, []
