# Smaller and longer-horizon items

Scoped 2026-10-07. One short plan per item; full write-ups stay in
`docs/future_work.md`.

## FW62 — apply-page schema layer, remaining stages

Stages 1 (autocomplete fallback) and 3 (generic ATS dispatch, never submits)
shipped. Stage 2 needs **real DOM captures**: have the worker save
`page.content()` plus an accessibility snapshot of each application page it
reaches (redacting filled values) to `~/.applypilot/page_samples/{ats}/`.
After ~20 samples per ATS, derive selectors from the samples with tests
against the saved HTML. Stages 4–5 (page-shape classifier, per-field
confidence) build on the same corpus. **First step is the capture hook**:
small, cloud-buildable, then runs passively on the real machine.

## FW65 — remaining piece

Per-ATS field hints in the Claude prompt (e.g. "on Workday the first-name
input is `data-automation-id=legalNameSection_firstName`"), generated from
`_FIELD_SELECTORS` only for selectors verified by FW62 Stage 2 samples.
Depends on FW62 Stage 2.

## FW9 — build the experience/project inventory in the wizard

Today every `experience_inventory` entry is hand-written. Plan: parse the
resume into roles (the PDF parser already extracts title/subtitle/bullets),
show each role and ask the FW26 follow-up questions, write entries with
`evidence_level` and empty `constraints` for the user to fill, then offer
to run `expand-bank`. The GitHub import (decision #85) already produces
project drafts. Cloud-buildable; needs one real walk-through.

## FW33 — wizard walkthroughs for external setup

Start with Gmail OAuth, the one with known snags: auto-detect a
`client_secret_*.json` in Downloads and offer to move/rename it to
`gcp-oauth.keys.json`; check the OAuth client type; print the exact
console URL for each step. A static decision tree first; a local-model
helper only if the tree proves insufficient.

## FW47 — bug-report and flag-job buttons

- **Flag this job:** a banner button that POSTs to the listener, which
  writes a `job_flags` row (url, reason, free text). Cheap and useful for
  scoring audits.
- **Report a bug:** collect the job row, the latest worker log tail, and
  the user's note into a local Markdown file, then open a prefilled
  GitHub "new issue" URL in a new tab (no token needed; the user submits).
  Avoids storing a GitHub token on the machine.

## FW50 — document downloads and dedup

Save every document the agent downloads under `files/{job_hash}/`, hash
with SHA-256, and keep one canonical copy per hash with per-job links.
Text-similarity dedup (same notice, different PDF render) only if exact
hashing proves insufficient.

## FW67 — company reputation

Cache one reputation summary per employer (not per posting): public
sources only (news, NLRB filings, B Corp registry, Consumer Rights Wiki).
Output a small bonus/penalty like `labor_signals.py`, never a hard gate,
and always show the cited sources. Needs a design conversation on sources
and bias; per the user's own framing, not a purity test.

## FW68 — keyword store, questionnaire, wizard scope, multi-user

Four separate conversations. The cheapest real step: move the filter keyword
lists (ethical exclusions, labor signals, scam signals) into one YAML file
with a shared loader, so new categories don't each invent a format.

## FW52 and FW39 — not recommended without an explicit decision

- **FW52 (deliberately imperfect assessment answers):** shaping answers to
  look less machine-like on a hiring assessment is a misrepresentation
  question, not an engineering one. Recommendation: don't build; if
  assessments are a problem, route them to the human.
- **FW39 (proxy/fingerprint hardening for LinkedIn):** escalates scraping
  against LinkedIn's terms. FW64's ATS-matching route gets the same jobs
  without it. Recommendation: do FW64 instead.

## FW58, FW59, FW66, D1

Long horizon, unchanged: companion app (FW53 email alerts now cover the
urgent part), research ideas, SMS gateway app (after FW44), and custom-ATS
scrapers (after FW71 decides how locality data is sourced).
