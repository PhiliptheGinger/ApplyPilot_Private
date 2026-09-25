"""Deterministic apply engine (Option 1).

This engine uses direct Playwright actions instead of spawning a Claude
subprocess. It is intentionally conservative:
- Supports a minimal Greenhouse flow, and a personal-info-page-only
  Workday flow (2026-09-19 -- see _run_workday's docstring for scope).
- Escalates CAPTCHAs to human intervention immediately.
- Uses profile/application_profile facts only (no hallucinated answers).

2026-09-19 speed work: field selectors for each supported ATS live in
`_FIELD_SELECTORS`, a per-ATS "semantic field -> candidate CSS selectors"
map, rather than one-off calls scattered through each `_run_*` function.
This is the concrete version of the "flexible map" idea discussed with
the user: a new ATS mostly means adding one more entry to this dict, not
writing a new fill routine. `_fill_known_fields` is the single generic
consumer. Screening-question answering (`_answer_known_screening_
questions`) is shared across every ATS for the same reason -- it reuses
the existing `qa_knowledge` table (already fed by the Claude engine's own
successful runs), so a question this candidate has answered before (on
ANY ATS) gets filled here without ever invoking Claude for it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from applypilot import config
from applypilot.apply.chrome import detect_ats
from applypilot.database import get_qa

logger = logging.getLogger(__name__)

# Per-ATS "semantic field -> candidate CSS selectors" map. Selectors are
# tried in order; the first that matches AND accepts the fill wins (see
# _fill_known_fields). Greenhouse's entries are the exact selectors the
# original _run_greenhouse used, just relocated here so both ATSes share
# one lookup mechanism. Workday's entries use its well-known, tenant-
# consistent `data-automation-id` convention (Workday's own frontend
# framework reuses these IDs across every employer's tenant) -- NOT yet
# verified against a real, live Workday page in this session; the engine
# is designed to fail safe (escalate to needs_human, never guess or
# submit on a low fill-rate) if these turn out to need adjustment once
# tried for real. See _run_workday's docstring for the current scope.
_FIELD_SELECTORS: dict[str, dict[str, list[str]]] = {
    "greenhouse": {
        "first_name": [
            "input[name='first_name']",
            "input[name='job_application[first_name]']",
            "#first_name",
        ],
        "last_name": [
            "input[name='last_name']",
            "input[name='job_application[last_name]']",
            "#last_name",
        ],
        "email": [
            "input[name='email']",
            "input[name='job_application[email]']",
            "#email",
        ],
        "phone": [
            "input[name='phone']",
            "input[name='job_application[phone]']",
            "input[type='tel']",
            "#phone",
        ],
        "city": [
            "input[name='location']",
            "input[name='job_application[location]']",
            "#location",
        ],
        "linkedin": [
            "input[name='linkedin']",
            "input[name='job_application[linkedin]']",
            "input[name*='linkedin']",
        ],
        "website": [
            "input[name='website']",
            "input[name='job_application[website]']",
            "input[name*='portfolio']",
            "input[name*='url']",
        ],
    },
    "workday": {
        "first_name": [
            "input[data-automation-id='legalNameSection_firstName']",
            "input[data-automation-id*='firstName']",
        ],
        "last_name": [
            "input[data-automation-id='legalNameSection_lastName']",
            "input[data-automation-id*='lastName']",
        ],
        "email": [
            "input[data-automation-id='email']",
            "input[type='email']",
        ],
        "phone": [
            "input[data-automation-id='phone-number']",
            "input[data-automation-id*='phoneNumber']",
        ],
        "city": [
            "input[data-automation-id='addressSection_city']",
            "input[data-automation-id*='city' i]",
        ],
        "linkedin": [
            "input[data-automation-id*='linkedIn' i]",
            "input[data-automation-id*='socialNetwork' i]",
        ],
        "website": [
            "input[data-automation-id*='website' i]",
        ],
    },
}

# Vendor-agnostic fallback, tried AFTER an ATS's own `_FIELD_SELECTORS`
# entry for a field comes up empty. Built from the HTML `autocomplete`
# attribute -- a real, standardized, cross-vendor signal (it's what
# enables the browser's OWN autofill, so most ATS vendors populate it
# correctly regardless of their internal `name`/`id`/`data-*` conventions)
# -- plus `type=` as a second-tier fallback for fields with a reliable
# input type. Scoped 2026-09-25 per the apply-page-schema discussion
# (Future Work items 29/42/30): this is the safe, additive slice of that
# idea -- it improves resilience for ATSes ALREADY in `_FIELD_SELECTORS`
# (a vendor DOM change that breaks one hand-authored selector can still
# be caught here) without touching which ATSes `run_job_deterministic`
# will even attempt (see its own `ats in ("greenhouse", "workday")` gate)
# -- that's a separate, bigger decision, not made here.
_GENERIC_FIELD_SELECTORS: dict[str, list[str]] = {
    "first_name": ["input[autocomplete='given-name']"],
    "last_name": ["input[autocomplete='family-name']"],
    "email": ["input[autocomplete='email']", "input[type='email']"],
    "phone": ["input[autocomplete='tel']", "input[type='tel']"],
    "city": ["input[autocomplete='address-level2']"],
    "linkedin": ["input[autocomplete*='linkedin' i]"],
    "website": ["input[autocomplete='url']"],
}

# A submit is only attempted when at least this fraction of the
# ATS's known fields were actually filled -- a low fill rate means the
# page's selectors likely drifted from `_FIELD_SELECTORS` (or this isn't
# really the page we think it is), and submitting a mostly-empty form is
# worse than escalating to a human. Greenhouse's existing behavior (no
# gate at all, shipped and used before this refactor) is left unchanged;
# this only applies to the new Workday path -- see _run_workday.
_MIN_FILL_RATE_TO_SUBMIT = 0.5


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
        "website": str(
            online.get("website") or personal.get("website_url") or personal.get("portfolio_url") or ""
        ).strip(),
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
            except Exception:  # noqa: BLE001, S112 - try each candidate selector; a Playwright element-interaction failure means try the next one, not abort the whole fill/upload attempt
                continue
    return False


def _fill_known_fields(page, ats_slug: str, fields: dict[str, str]) -> tuple[int, int]:
    """Fill every field this ATS (or the generic fallback) knows a selector for.

    Returns (filled_count, known_count) -- known_count is how many fields
    have a candidate value AND at least one candidate selector, vendor or
    generic (an empty profile value never counts against the fill rate,
    since there's nothing this engine could have filled either way).

    Tries the vendor's own `_FIELD_SELECTORS[ats_slug]` entry for a field
    first, then falls back to `_GENERIC_FIELD_SELECTORS` -- both for
    fields the vendor map doesn't list at all, and as a second attempt
    when the vendor's own selectors are present but don't match (e.g. a
    DOM change since those were authored).
    """
    vendor_map = _FIELD_SELECTORS.get(ats_slug, {})
    field_names = set(vendor_map) | set(_GENERIC_FIELD_SELECTORS)
    filled = 0
    known = 0
    for field_name in field_names:
        value = fields.get(field_name, "")
        if not value:
            continue
        selectors = vendor_map.get(field_name, []) + _GENERIC_FIELD_SELECTORS.get(field_name, [])
        if not selectors:
            continue
        known += 1
        if _safe_fill(page, selectors, value):
            filled += 1
    return filled, known


# Generic question-label selectors, ATS-agnostic on purpose: every ATS
# renders a screening question as *some* label/legend followed by an
# input/select/textarea, just with different exact markup. This looks at
# the visible text, not the DOM shape, which is what qa_knowledge's own
# question_key normalization already keys on.
_QUESTION_LABEL_SELECTORS = "label, legend, [class*='question' i] > *:first-child"


def _answer_known_screening_questions(page, doc_format: str | None = None) -> int:
    """Fill any screening question this candidate has answered before,
    on ANY ats (qa_knowledge is not ATS-scoped for lookup purposes),
    without ever guessing an answer that wasn't already known.

    Deliberately conservative: only text/textarea/select inputs are
    touched (radio/checkbox groups need per-option matching this doesn't
    attempt yet), and a question with no `qa_knowledge` row is left
    completely alone for the human/Claude engine to handle.

    Returns the number of questions answered.
    """
    answered = 0
    try:
        labels = page.locator(_QUESTION_LABEL_SELECTORS)
        count = min(labels.count(), 60)  # bound: a runaway match set must not hang the apply attempt
    except Exception:  # noqa: BLE001 - a selector-engine failure here must not abort the whole apply attempt
        return 0

    for i in range(count):
        try:
            label = labels.nth(i)
            question_text = (label.inner_text(timeout=500) or "").strip()
            if not question_text or len(question_text) < 4:
                continue
            answer = get_qa(question_text, doc_format=doc_format)
            if not answer:
                continue

            # The input is usually the label's `for` target or the next
            # form control in DOM order -- try both, cheaply.
            control = None
            input_id = label.get_attribute("for", timeout=300)
            if input_id:
                candidate = page.locator(f"#{input_id}")
                if candidate.count() > 0:
                    control = candidate.first
            if control is None:
                sibling = label.locator(
                    "xpath=following::input[1] | xpath=following::textarea[1] | xpath=following::select[1]"
                )
                if sibling.count() > 0:
                    control = sibling.first
            if control is None:
                continue

            tag = (control.evaluate("el => el.tagName") or "").lower()
            if tag == "select":
                control.select_option(label=answer, timeout=1000)
            else:
                control.fill(answer, timeout=1000)
            answered += 1
        except Exception:  # noqa: BLE001, S112 - one question's markup being unexpected must not abort scanning the rest
            continue
    return answered


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
            except Exception:  # noqa: BLE001, S112 - try each candidate selector; a Playwright element-interaction failure means try the next one, not abort the whole fill/upload attempt
                continue
    return False


def _run_greenhouse(page, job: dict, dry_run: bool) -> tuple[str, int, list[dict]]:
    started = time.time()
    fields = _get_profile_fields()

    _fill_known_fields(page, "greenhouse", fields)
    _upload_resume_if_present(page, job)
    _answer_known_screening_questions(page, doc_format=_doc_suffix(job))

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
            except Exception:  # noqa: BLE001, S112 - try each candidate selector; a Playwright element-interaction failure means try the next one, not abort the whole fill/upload attempt
                continue

    dur = int((time.time() - started) * 1000)
    return f"needs_human:review_required:{page.url}", dur, []


def _doc_suffix(job: dict) -> str | None:
    """Return 'pdf'/'docx' from the tailored resume's own extension, for
    get_qa's stale-answer-format filtering (see database.get_qa)."""
    resume_path = job.get("tailored_resume_path")
    if not resume_path:
        return None
    suffix = Path(resume_path).suffix.lstrip(".")
    return suffix or None


def _run_workday(page, job: dict, dry_run: bool) -> tuple[str, int, list[dict]]:
    """Fill Workday's personal-info page only, then stop.

    2026-09-19 scope (deliberately narrow, per the same "start with one
    slice, not the whole wizard" plan already outlined for Workday in
    CLAUDE.md's Future Work): Workday's real apply flow is a multi-page
    wizard (often account creation -> personal info -> experience ->
    voluntary disclosures -> review), and this function only ever
    understands the personal-info shape (name/email/phone/city/resume
    upload, `_FIELD_SELECTORS["workday"]`) plus whatever screening
    questions happen to appear on that same page. It intentionally never
    attempts a page beyond that -- clicking "Next"/"Save and Continue" is
    the last action this function takes; everything after that hands off
    to a human via `needs_human`, exactly like an unhandled captcha.

    Never submits a final application -- Workday's actual submit step is
    on a later review page this function doesn't reach, so `dry_run` has
    no effect here (there is no submit action to skip).
    """
    started = time.time()
    fields = _get_profile_fields()

    filled, known = _fill_known_fields(page, "workday", fields)
    _upload_resume_if_present(page, job)
    _answer_known_screening_questions(page, doc_format=_doc_suffix(job))

    if _has_captcha(page):
        dur = int((time.time() - started) * 1000)
        return f"needs_human:captcha:{page.url}", dur, []

    # A low fill rate means the real page's markup doesn't match
    # _FIELD_SELECTORS["workday"] closely enough to trust -- escalate
    # rather than click "Next" on a form that's still mostly blank.
    if known > 0 and (filled / known) < _MIN_FILL_RATE_TO_SUBMIT:
        dur = int((time.time() - started) * 1000)
        return f"needs_human:workday_fields_unmatched:{page.url}", dur, []

    next_buttons = [
        "button[data-automation-id='bottom-navigation-next-button']",
        "button:has-text('Save and Continue')",
        "button:has-text('Next')",
    ]
    for sel in next_buttons:
        btn = page.locator(sel)
        if btn.count() > 0:
            try:
                btn.first.click(timeout=2500)
                page.wait_for_timeout(1500)
                break
            except Exception:  # noqa: BLE001, S112 - try each candidate selector; a Playwright element-interaction failure means try the next one, not abort this attempt
                continue

    dur = int((time.time() - started) * 1000)
    return f"needs_human:workday_wizard_incomplete:{page.url}", dur, []


def _run_generic(page, job: dict, dry_run: bool, ats_slug: str) -> tuple[str, int, list[dict]]:
    """Conservative fill-only path for any ATS other than the two verified
    ones (greenhouse: submits; workday: fills one page then escalates).

    2026-09-25 (apply-page schema Stage 3, Future Work item 62): widens
    `run_job_deterministic`'s dispatch from a hardcoded `ats in
    ("greenhouse", "workday")` allowlist to attempting ANY detected (or
    undetected) ATS -- using `_FIELD_SELECTORS[ats_slug]` if this session
    has real, hand-verified selectors for it, falling back to
    `_GENERIC_FIELD_SELECTORS`'s autocomplete-attribute map otherwise.

    Deliberately never submits a final application, regardless of fill
    rate -- unlike `_run_greenhouse` (a verified, trusted vendor) and
    matching `_run_workday`'s own precedent (an unverified vendor that
    only fills then hands off), no ATS reaching this function has been
    confirmed live in this session. `dry_run` has no effect here for the
    same reason `_run_workday` documents: there is no submit action to
    skip. Always ends in `needs_human` so a real human reviews and
    completes the actual submission -- this is strictly an assistive
    fill, not a new auto-apply path, until a specific vendor is proven
    out and promoted to its own verified `_run_*` function.
    """
    started = time.time()
    fields = _get_profile_fields()

    filled, known = _fill_known_fields(page, ats_slug, fields)
    _upload_resume_if_present(page, job)
    _answer_known_screening_questions(page, doc_format=_doc_suffix(job))

    if _has_captcha(page):
        dur = int((time.time() - started) * 1000)
        return f"needs_human:captcha:{page.url}", dur, []

    if known == 0:
        # Nothing this engine could even attempt to fill (no candidate
        # value AND selector for any field) -- distinct from a low fill
        # rate below, so a human reviewing the log can tell "we tried and
        # missed" from "we had nothing to try".
        dur = int((time.time() - started) * 1000)
        return f"needs_human:generic_ats_no_known_fields:{page.url}", dur, []

    if (filled / known) < _MIN_FILL_RATE_TO_SUBMIT:
        dur = int((time.time() - started) * 1000)
        return f"needs_human:generic_ats_fields_unmatched:{page.url}", dur, []

    dur = int((time.time() - started) * 1000)
    return f"needs_human:generic_ats_review:{page.url}", dur, []


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

            if ats == "workday":
                return _run_workday(page, job, dry_run=dry_run)
            if ats == "greenhouse":
                return _run_greenhouse(page, job, dry_run=dry_run)
            # Stage 3: any other detected-or-undetected ATS gets the
            # conservative, never-submits generic path instead of an
            # immediate needs_human:unsupported_ats bailout.
            return _run_generic(page, job, dry_run=dry_run, ats_slug=ats or "unknown")

    except PlaywrightTimeoutError:
        dur = int((time.time() - started) * 1000)
        return "failed:timeout", dur, []
    except Exception as exc:
        logger.exception("deterministic engine failed")
        dur = int((time.time() - started) * 1000)
        return f"failed:deterministic_error:{str(exc)[:80]}", dur, []
