"""Cover letter generation: LLM-powered, profile-driven, with validation.

Generates concise, engineering-voice cover letters tailored to specific job
postings. All personal data (name, skills, achievements) comes from the user's
profile at runtime. No hardcoded personal information.
"""

import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from applypilot.config import COVER_LETTER_DIR, load_profile
from applypilot.llm import get_stage_client, get_token_limit
from applypilot.scoring.resume_router import (
    is_communication_role,
    load_resume_text_for_job,
)
from applypilot.scoring.validator import (
    sanitize_text,
    validate_cover_letter,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up


def get_client(quality: bool = True):
    """Compatibility wrapper used by tests and call sites.

    Internally routes to the stage-aware client selection logic.
    """
    return get_stage_client("cover", quality=quality)


DEFAULT_COMMUNICATION_DIFFERENTIATOR = (
    "Not something I normally put on my resume, but I believe this is a "
    "legitimate differentiator for roles that require clear communication, "
    "where one might need to take in rejection or sincere feedback."
)


# ── Prompt Builder (profile-driven) ──────────────────────────────────────


def _build_cover_letter_prompt(profile: dict, job: dict | None = None) -> str:
    """Build the cover letter system prompt from the user's profile.

    All personal data, skills, and sign-off name come from the profile.
    """
    personal = profile.get("personal", {})
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Preferred name for the sign-off (falls back to full name)
    sign_off_name = personal.get("preferred_name") or personal.get("full_name", "")

    # Flatten all allowed skills
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "the tools listed in the resume"

    # Real metrics from resume_facts
    real_metrics = resume_facts.get("real_metrics", [])
    preserved_projects = resume_facts.get("preserved_projects", [])

    # Build achievement examples for the prompt
    projects_hint = ""
    if preserved_projects:
        projects_hint = f"\nKnown projects to reference: {', '.join(preserved_projects)}"

    metrics_hint = ""
    if real_metrics:
        metrics_hint = f"\nReal metrics to use: {', '.join(real_metrics)}"

    communication_mode = bool(job and is_communication_role(job))
    cl_cfg = profile.get("cover_letter", {}) if isinstance(profile, dict) else {}
    differentiator = cl_cfg.get("communication_differentiator_note") or DEFAULT_COMMUNICATION_DIFFERENTIATOR

    communication_block = ""
    if communication_mode:
        communication_block = (
            "\nCOMMUNICATION-ROLE REQUIREMENT:\n"
            "Include one concise sentence (your own wording; close paraphrase allowed) "
            "that conveys this idea:\n"
            f'"{differentiator}"\n'
            "Keep it sincere and concrete, not performative."
        )

    # For communication-heavy roles, allow "I believe" because the
    # differentiator sentence may legitimately use it.
    banned_i_believe = '"I believe", ' if not communication_mode else ""

    return f"""Write a cover letter for {sign_off_name}. The goal is to get an interview.

STRUCTURE: 4 paragraphs. TARGET 300-400 words; MINIMUM 260 words (Jobscan 3.4x interview-rate sweet spot is 250-400). Letters under 260 words get rejected automatically.

PARAGRAPH 1 — HOOK (4-6 sentences, ~80 words): Open with a specific thing YOU built that solves THEIR problem. Identify the problem they're hiring to solve (infer from the job description) and name the work you've done that directly addresses it. Include enough context that the reader understands the scope and impact. Not "I'm excited about this role." Not "This role aligns with my experience." Start with the work.

PARAGRAPH 2 — EVIDENCE (4-6 sentences, ~120 words): Pick 2 achievements from the resume that are MOST relevant to THIS job. For each, name the problem, the concrete action you took (specific tools, architecture decisions), and the quantified outcome. Use numbers. Frame each as solving their problem, not listing your accomplishments.{projects_hint}{metrics_hint}

PARAGRAPH 3 — COMPANY FIT (3-4 sentences, ~70 words): Reference one specific thing about the company from the job description (a product, a technical challenge, a team structure). Connect it to your experience with a concrete parallel, not a generic nod. Show you've read the posting and that you've solved a similar shape of problem.
If COMPANY is marked unknown, infer the employer from the job description. If the description doesn't name it either, write about the team's product and problem domain WITHOUT naming any company. NEVER treat the job board the listing came from (LinkedIn, Indeed, etc.) as the employer.

PARAGRAPH 4 — CLOSE (2 sentences, ~30 words): Offer to go deeper on one or two SPECIFIC topics from your evidence paragraph, naming the actual system, migration, or metric. Write the offer in your own words; do not use stock closers ("Happy to walk through...", "I'd welcome the chance to discuss..."). Then sign off.

BANNED WORDS/PHRASES (using ANY of these = instant rejection):
"resonated", "aligns with", "passionate", "eager", "eager to", "excited to apply", "I am confident",
{banned_i_believe}"proven track record", "strong track record", "cutting-edge", "innovative", "innovative solutions",
"leverage", "leveraging", "robust", "driven", "dedicated", "committed to",
"I look forward to hearing from you", "great fit", "unique opportunity",
"commitment to excellence", "dynamic team", "fast-paced environment",
"I am writing to express", "caught my eye", "caught my attention"

BANNED PUNCTUATION: No em dashes. Use commas or periods.

VOICE:
- Write like a real engineer emailing someone they respect. Not formal, not casual. Just direct.
- NEVER narrate or explain what you're doing. BAD: "This demonstrates my commitment to X." GOOD: Just state the fact and move on.
- NEVER hedge. BAD: "might address some of your challenges." GOOD: "solves the same problem your team is facing."
- NEVER use "Also," to start a sentence. NEVER use "Furthermore," or "Additionally,".
- Every sentence should contain either a number, a tool name, or a specific outcome. If it doesn't, cut it.
- Read it out loud. If it sounds like a robot wrote it, rewrite it.

ADDITIONAL BANNED PHRASES:
"This demonstrates", "This reflects", "This showcases", "This shows",
"This experience translates", "which aligns with", "which is relevant to",
"as demonstrated by", "showing experience with", "reflecting the need for",
"which directly addresses", "I have experience with",
"Also,", "Furthermore,", "Additionally,", "Moreover,",
"Happy to walk through", "demonstrate" (any form: demonstrates, demonstrated, demonstrating),
"resonate" (any form), "align with" (any form: aligns with, aligned with)

{communication_block}

FABRICATION = INSTANT REJECTION:
The candidate's real tools are ONLY: {skills_str}.
Do NOT mention ANY tool not in this list. If the job asks for tools not listed, talk about the work you did, not the tools.
Never mention ApplyPilot or any private/internal project by name.
Never describe the candidate as an engineer, architect, developer, or other professional technical role unless their actual employment history supports that title.

Sign off: just "{sign_off_name}"

Output ONLY the letter. Start with "Dear Hiring Manager," end with the name."""


# ── Core Generation ──────────────────────────────────────────────────────


def generate_cover_letter(resume_text: str, job: dict, profile: dict, max_retries: int = 3) -> tuple[str, dict]:
    """Generate a cover letter with fresh context on each retry + auto-sanitize.

    Same design as tailor_resume: fresh conversation per attempt, issues noted
    in the prompt, no conversation history stacking.

    Args:
        resume_text: The candidate's resume text (base or tailored).
        job: Job dict with title, site, company, location, full_description.
        profile: User profile dict.
        max_retries: Maximum retry attempts.

    Returns:
        (letter, validation) — the best attempt and its validate_cover_letter
        verdict. Callers must check validation["passed"] before shipping.
    """
    from applypilot.scoring.tailor import display_company

    company = display_company(job)
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {company or 'unknown (aggregator listing; employer may be named in the description)'}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    # Same deterministic, LLM-free per-job schema representation tailor.py
    # uses -- cached per job (scoring/schemas.py), so if this job was already
    # tailored in this run, this reuses that computation instead of
    # redoing it. Cover letters previously got zero structured evidence
    # grounding at all (just the raw description dump above); this gives the
    # cover-letter model the same requirement/evidence/schema mapping.
    # Failure here must never block cover-letter generation -- it's guidance.
    schema_guidance = ""
    try:
        from applypilot.scoring.schemas import format_schema_guidance, get_or_build_job_schema

        job_schema = get_or_build_job_schema(job, profile)
        schema_guidance = format_schema_guidance(job_schema)
    except Exception:
        log.debug("Job schema computation failed for %s", job.get("title", "")[:40], exc_info=True)

    avoid_notes: list[str] = []
    letter = ""
    validation: dict = {"passed": False, "errors": ["no attempts"], "warnings": []}
    client = get_client(quality=True)
    cl_prompt_base = _build_cover_letter_prompt(profile, job)

    for attempt in range(max_retries + 1):
        # Fresh conversation every attempt
        prompt = cl_prompt_base
        if avoid_notes:
            prompt += "\n\n## AVOID THESE ISSUES:\n" + "\n".join(f"- {n}" for n in avoid_notes[-5:])

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    (f"{schema_guidance}\n\n---\n\n" if schema_guidance else "") + f"RESUME:\n{resume_text}\n\n---\n\n"
                    f"TARGET JOB:\n{job_text}\n\n"
                    "Write the cover letter:"
                ),
            },
        ]

        # Higher ceiling helps thinking-models avoid truncation while
        # still letting prompts enforce concise output.
        letter = client.chat(
            messages,
            max_tokens=get_token_limit("cover", 8192),
            temperature=0.7,
        )
        letter = sanitize_text(letter)  # auto-fix em dashes, smart quotes

        validation = validate_cover_letter(letter, profile)
        if validation["passed"]:
            return letter, validation

        avoid_notes.extend(validation["errors"])
        # The model chronically undershoots length; a generic "too short"
        # error doesn't fix it. Give explicit per-paragraph expansion targets.
        words = len(letter.split())
        if words < 260:
            avoid_notes.append(
                f"Your previous draft was only {words} words. The minimum is 260; "
                "target 300-400. Expand the hook to ~80 words and the evidence "
                "paragraph to ~120 words with additional concrete details, tools, "
                "and numbers from the resume. Do not pad with filler."
            )
        log.debug(
            "Cover letter attempt %d/%d failed: %s",
            attempt + 1,
            max_retries + 1,
            validation["errors"],
        )

    return letter, validation  # last attempt, validation["passed"] is False


# ── Batch Entry Point ────────────────────────────────────────────────────


def _cover_one_job(job: dict, resume_text: str | None, profile: dict, doc_format: str = "docx") -> dict:
    """Generate cover letter for a single job. Safe to call from multiple threads."""
    from applypilot.scoring.tailor import _extract_keywords, _name_parts

    if resume_text is None:
        resume_text, _ = load_resume_text_for_job(job)
    letter, validation = generate_cover_letter(resume_text, job, profile)

    # Filename: FirstName_LastName_JobTitle_hash_CL.{ext} (Jobscan §3).
    first, last = _name_parts(profile)
    safe_title = re.sub(r"[^\w\s-]", "", job.get("title") or "untitled")[:50].strip().replace(" ", "_")
    url_hash = hashlib.md5(job["url"].encode()).hexdigest()[:8]
    if first and last:
        prefix = f"{first}_{last}_{safe_title}_{url_hash}"
    elif first:
        prefix = f"{first}_{safe_title}_{url_hash}"
    else:
        safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
        prefix = f"{safe_site}_{safe_title}_{url_hash}"

    if not validation["passed"]:
        # Don't ship a letter that failed validation. Keep the rejected draft
        # on disk for inspection, return no path so _mark_cover_result parks
        # the job as cover_failed (retried until MAX_ATTEMPTS via pending_cover).
        reason = "; ".join(validation["errors"])
        rejected_path = COVER_LETTER_DIR / f"{prefix}_CL_rejected.txt"
        rejected_path.write_text(letter, encoding="utf-8")
        log.info("Cover letter rejected for %s: %s", (job.get("title") or "")[:40], reason)
        return {
            "url": job["url"],
            "path": None,
            "pdf_path": None,
            "title": job["title"],
            "site": job["site"],
            "error": f"validation failed: {reason}",
        }

    cl_path = COVER_LETTER_DIR / f"{prefix}_CL.txt"
    cl_path.write_text(letter, encoding="utf-8")

    doc_path = None
    try:
        from applypilot.scoring.pdf import convert_to_pdf

        personal = profile.get("personal", {})
        full_name = personal.get("full_name") or personal.get("preferred_name") or ""
        job_title = (job.get("title") or "").strip()[:150]
        site = (job.get("site") or "").strip()[:80]
        cl_metadata = {
            "title": f"Cover Letter — {full_name} for {job_title}" if full_name else f"Cover Letter — {job_title}",
            "subject": job_title,
            "author": full_name,
            "category": "Cover Letter",
            "keywords": _extract_keywords(job, profile),
            "comments": (
                f"Cover letter for: {job_title}\nSource: {site}\nDate: {datetime.now(UTC).strftime('%Y-%m-%d')}"
            ),
        }
        doc_path = str(convert_to_pdf(cl_path, doc_format=doc_format, metadata=cl_metadata, content_type="cover_letter"))
    except Exception:
        log.debug("Document generation failed for %s", cl_path, exc_info=True)

    return {
        "url": job["url"],
        "path": str(cl_path),
        "pdf_path": doc_path,
        "title": job["title"],
        "site": job["site"],
    }


def _mark_cover_result(
    conn,
    url: str,
    path: str | None,
    *,
    error: str | None = None,
    now: str | None = None,
) -> None:
    """Persist one cover letter result and emit a state transition.

    Extracted from ``_flush_cover_results`` so tests can call it directly.
    Transitions to ``ready_to_apply`` on success, ``cover_failed`` on failure.

    Phase 3 state-machine hardening (2026-08-27): the job must still be in
    'cover_writing' -- the in-flight state run_cover_letters claims it
    into before starting the LLM call -- for this completion write to
    happen at all. The success path is the most dangerous stale-completion
    case found in the investigation: it lands directly on 'ready_to_apply'
    with no intermediate failure state, so a stale worker's write here
    would otherwise resurrect an archived job one step from being applied
    to.
    """
    if now is None:
        now = datetime.now(UTC).isoformat()

    from applypilot.database import current_state, transition_state

    current = current_state(conn, url)
    if current != "cover_writing":
        log.warning(
            "Skipping stale cover-letter completion for %s -- expected state 'cover_writing', "
            "found %r (job was likely archived or otherwise reassigned by another process while "
            "this cover-letter call was in flight)",
            url[:80],
            current,
        )
        return

    if path:
        conn.execute(
            "UPDATE jobs SET cover_letter_path=?, cover_letter_at=?, "
            "cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
            (path, now, url),
        )
        transition_state(
            conn,
            url,
            "ready_to_apply",
            reason="cover letter done",
            metadata={"path": path},
        )
    else:
        conn.execute(
            "UPDATE jobs SET cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
            (url,),
        )
        transition_state(
            conn,
            url,
            "cover_failed",
            reason="cover generation failed",
            metadata={"error": error},
        )


def run_cover_letters(
    min_score: int | None = None,
    limit: int = 20,
    workers: int = 1,
    doc_format: str = "docx",
    max_age_days: int | None = None,
    job_ids: list[str] | None = None,
) -> dict:
    """Generate cover letters for high-scoring jobs that have tailored resumes.

    Args:
        min_score: Minimum fit_score threshold (default from config).
        limit: Maximum jobs to process. Ignored when job_ids is given -- see below.
        workers: Parallel LLM threads (default 1 = sequential).
        doc_format: Output document format — "docx" (default) or "pdf".
        max_age_days: Skip jobs older than this (default from config).
        job_ids: If given (e.g. by a sequential pipeline run carrying
            forward the exact batch a preceding `tailor` stage produced --
            see run_tailoring's job_urls / pipeline._run_sequential),
            restrict candidates to exactly this set of job URLs instead of
            independently querying for any eligible job. Still subject to
            the normal pending_cover eligibility conditions (min_score,
            has a tailored resume, no cover letter yet, attempts < 5,
            eligibility) and the per-company cap below, so only the
            eligible subset of this batch is actually processed -- an
            ineligible job in the batch is just skipped, never backfilled
            with an unrelated already-eligible job from an earlier run to
            make up the count. An empty list restricts to nothing. None
            (the default) preserves the original standalone behavior:
            select up to `limit` eligible jobs with no batch restriction.

    Returns:
        {"generated": int, "errors": int, "elapsed": float}
    """
    from applypilot.config import DEFAULTS
    from applypilot.database import get_jobs_by_stage

    if min_score is None:
        min_score = DEFAULTS["min_score"]

    from applypilot.database import (
        commit_with_retry,
        get_connection,
        recover_stale_claims,
        transition_state,
        write_with_retry,
    )

    profile = load_profile()
    conn = get_connection()

    # Recover jobs stranded in 'cover_writing' by a worker that died
    # mid-call (see database.recover_stale_claims) before selecting new
    # candidates, so a just-recovered cover_failed job is immediately
    # eligible for pending_cover below instead of waiting for a later run.
    recovered = recover_stale_claims(conn, "cover_writing", "cover_failed", "cover_attempts")
    if recovered:
        commit_with_retry(conn)
        log.info("Recovered %d stale 'cover_writing' claim(s) -> cover_failed", len(recovered))

    # Note: get_jobs_by_stage applies a 14-day discovered_at filter by default
    # (config.DEFAULTS["max_job_age_days"]). Pass max_age_days=0 to disable.
    jobs = get_jobs_by_stage(
        conn=conn, stage="pending_cover", min_score=min_score, max_age_days=max_age_days, limit=limit, urls=job_ids
    )

    if job_ids is not None:
        log.info(
            "Cover letters restricted to a carried batch of %d job(s) from the preceding "
            "tailor stage (%d eligible after the pending_cover filter).",
            len(job_ids),
            len(jobs),
        )

    # Per-company cover-letter cap (mirrors tailor cap in tailor.py).
    # Keys resolved from `company`, with `site` fallback for direct-employer
    # scrapers. Aggregator sites (LinkedIn, Indeed, etc.) are exempt.
    from applypilot.scoring.tailor import resolve_company_key

    cap = DEFAULTS["max_tailored_per_company"]

    existing_rows = conn.execute(
        """
        SELECT LOWER(company) AS key, COUNT(*) AS n
        FROM jobs
        WHERE cover_letter_path IS NOT NULL
          AND company IS NOT NULL AND TRIM(company) != ''
          AND discovered_at > datetime('now', ?)
        GROUP BY key
        UNION ALL
        SELECT LOWER(site) AS key, COUNT(*) AS n
        FROM jobs
        WHERE cover_letter_path IS NOT NULL
          AND (company IS NULL OR TRIM(company) = '')
          AND strategy IN ('greenhouse_api', 'workday_api', 'lever_api',
                           'ashby_api', 'amazon_jobs', 'microsoft_careers',
                           'apple_jobs', 'google_careers')
          AND site IS NOT NULL AND TRIM(site) != ''
          AND discovered_at > datetime('now', ?)
        GROUP BY key
    """,
        (
            f"-{max_age_days or DEFAULTS['max_job_age_days']} days",
            f"-{max_age_days or DEFAULTS['max_job_age_days']} days",
        ),
    ).fetchall()
    existing: dict[str, int] = {}
    for r in existing_rows:
        existing[r["key"]] = existing.get(r["key"], 0) + r["n"]

    added_per_company: dict[str, int] = {}
    capped_jobs: list[dict] = []
    skipped_by_cap = 0
    for job in jobs:
        key = resolve_company_key(job)
        if key is None:
            capped_jobs.append(job)
            continue
        already = existing.get(key, 0) + added_per_company.get(key, 0)
        if already >= cap:
            skipped_by_cap += 1
            continue
        capped_jobs.append(job)
        added_per_company[key] = added_per_company.get(key, 0) + 1
    if skipped_by_cap:
        log.info("Cover cap: skipped %d job(s) where company is at/over %d covers.", skipped_by_cap, cap)
    jobs = capped_jobs

    # Phase 3 state-machine hardening (2026-08-27): claim each candidate
    # into the 'cover_writing' in-flight state before starting the LLM
    # call -- mirrors run_tailoring's identical claim step. Without this,
    # a job archived by a concurrent process while cover generation was
    # mid-flight could have its SUCCESS completion land straight on
    # 'ready_to_apply' (see _mark_cover_result), the most direct
    # resurrection path found in the investigation. transition_state
    # without force=True only succeeds if the job's current state is
    # still 'tailored' or 'cover_failed' -- the only two VALID_TRANSITIONS
    # sources for 'cover_writing' -- so a job that lost eligibility
    # between selection and this claim is dropped here.
    claimed_jobs = []
    for job in jobs:
        if transition_state(conn, job["url"], "cover_writing", reason="claimed for cover writing"):
            claimed_jobs.append(job)
        else:
            log.info(
                "Skipping cover candidate no longer claimable (state changed since selection): %s",
                job["url"][:80],
            )
    jobs = claimed_jobs

    conn.commit()  # Close read transaction before long LLM phase

    if not jobs:
        log.info("No jobs needing cover letters (score >= %d; after per-company cap).", min_score)
        return {"generated": 0, "errors": 0, "elapsed": 0.0}

    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        "Generating cover letters for %d jobs (score >= %d, workers=%d)...",
        len(jobs),
        min_score,
        workers,
    )
    t0 = time.time()
    completed = 0
    results: list[dict] = []
    error_count = 0

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_cover_one_job, job, None, profile, doc_format): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "url": job["url"],
                        "title": job.get("title") or "",
                        "site": job["site"],
                        "path": None,
                        "pdf_path": None,
                        "error": str(exc),
                    }
                    error_count += 1
                    log.exception("[ERROR] %s -- exception during future.result()", (job.get("title") or "")[:40])

                results.append(result)
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                status = "OK" if result.get("path") else "ERR"
                log.info(
                    "%d/%d [%s] | %.1f jobs/min | %s",
                    completed,
                    len(jobs),
                    status,
                    rate * 60,
                    (result.get("title") or "")[:40],
                )
    else:
        for job in jobs:
            completed += 1
            try:
                result = _cover_one_job(job, None, profile, doc_format)
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                status = "OK" if result.get("path") else "REJ"
                log.info(
                    "%d/%d [%s] | %.1f jobs/min | %s",
                    completed,
                    len(jobs),
                    status,
                    rate * 60,
                    (result.get("title") or "")[:40],
                )
            except Exception as exc:
                result = {
                    "url": job["url"],
                    "title": job.get("title") or "",
                    "site": job["site"],
                    "path": None,
                    "pdf_path": None,
                    "error": str(exc),
                }
                error_count += 1
                log.exception(
                    "%d/%d [ERROR] %s -- exception generating cover",
                    completed,
                    len(jobs),
                    (job.get("title") or "")[:40],
                )
            results.append(result)

    # Persist to DB: increment attempt counter for ALL, save path only for successes
    now = datetime.now(UTC).isoformat()
    saved = sum(1 for r in results if r.get("path"))

    def _flush_cover_results(conn, results, now):
        for r in results:
            # Prefer the generated DOCX/PDF path; fall back to text path
            # if conversion failed (apply layer will flag as invalid).
            stored_path = r.get("pdf_path") or r.get("path")
            _mark_cover_result(
                conn,
                r["url"],
                stored_path,
                error=r.get("error"),
                now=now,
            )

    try:
        write_with_retry(conn, _flush_cover_results, conn, results, now)
    except Exception:
        log.exception("DB flush failed for cover letter batch")

    elapsed = time.time() - t0
    rejected = sum(1 for r in results if not r.get("path") and str(r.get("error", "")).startswith("validation failed"))
    log.info(
        "Cover letters done in %.1fs: %d generated, %d rejected by validation, %d errors",
        elapsed,
        saved,
        rejected,
        error_count,
    )

    return {
        "generated": saved,
        "rejected": rejected,
        "errors": error_count,
        "elapsed": elapsed,
    }
