"""Statistical test of the fixed validator's fabrication-detection recall,
across a representative random sample of REAL jobs from the live DB.

Design: rather than running an LLM N times (slow, and non-reproducible),
this injects CONTROLLED, RANDOMIZED-BUT-REALISTIC fabrications into a
resume built from real profile evidence, varying the specific fabrication
per job (drawing on that job's own real description text) so each trial
is a genuinely different instance, not the same hand-picked example
repeated. This is what makes N independent Bernoulli trials meaningful
for a confidence interval -- a deterministic check run on the identical
input N times would have zero sampling variance and tell us nothing.

Five conditions per sampled job:
  clean            -- no fabrication (measures false-positive rate)
  skill            -- one fabricated skill drawn from the job's own vocabulary
  date             -- one entry's date shifted outside its real range
  bare_header      -- a fabricated prior job using the target's own title/company
  keyword_injection-- job vocabulary grafted onto an unrelated bullet

The "clean" baseline is built PROGRAMMATICALLY from profile.json's own
authoritative experience_inventory (not any pre-existing generated resume
file), specifically so it has none of the pre-existing formatting quirks
(placeholder dates, non-authoritative titles) found in real historical
artifacts during this investigation -- those would confound a false-
positive measurement that has nothing to do with THIS test's question.

No production code modified. No DB writes.
"""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\phili\Projects\resume-agent")
sys.path.insert(0, str(REPO_ROOT / "src"))

from applypilot import config  # noqa: E402
from applypilot import database  # noqa: E402
from applypilot.scoring.validator import (  # noqa: E402
    _distinctive_words,
    _build_skills_set,
    _SKILL_CLAIM_STOPWORDS,
    validate_json_fields,
)

OUT_DIR = Path(__file__).resolve().parent
SEED = 20260901
SAMPLE_SIZE = 200

random.seed(SEED)


def wilson_ci(successes: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval -- more accurate than the normal (Wald)
    approximation for proportions near 0 or 1, which is exactly the
    regime a well-functioning validator's catch rate should sit in."""
    if n == 0:
        return (0.0, 0.0)
    z = {0.95: 1.959963985, 0.99: 2.575829304}[confidence]
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def build_clean_resume(profile: dict) -> dict:
    """A resume built directly and only from profile.json's own
    authoritative experience_inventory -- guaranteed no placeholder dates,
    no title mismatches, nothing that would fail validation for reasons
    unrelated to this test."""
    preserved = [c.lower() for c in (profile.get("resume_facts") or {}).get("preserved_companies", [])]
    experience = []
    for item in profile.get("experience_inventory") or []:
        if item.get("resume_allowed") is False:
            continue
        name_lower = str(item.get("name") or "").lower()
        # Only entries the profile itself lists as a preserved/known
        # employer -- otherwise the baseline would legitimately (and
        # correctly) trip the unrecognized-employer warning on every
        # single trial regardless of injected fabrication, contaminating
        # the false-positive measurement with a signal that has nothing
        # to do with what this test is checking.
        if preserved and not any(p in name_lower or name_lower in p for p in preserved):
            continue
        role_title = item.get("role_title") or item.get("role_type") or ""
        name = item.get("name") or ""
        start = item.get("start_date") or ""
        end = item.get("end_date") or "Present"
        subtitle = f"{start} - {end}" if start else ""
        bullets = list(item.get("responsibilities") or [])
        if not bullets and item.get("description"):
            bullets = [str(item["description"])]
        if not bullets:
            continue
        experience.append({"header": f"{role_title} at {name}", "subtitle": subtitle, "bullets": bullets[:4]})

    edu = profile.get("education") or []
    edu_text = ""
    if edu:
        e = edu[0]
        edu_text = f"{e.get('institution', '')}\n{e.get('official_degree', '')}"

    return {
        "title": "Technical Support & Customer Service",
        "summary": "Customer-facing technical background with hands-on troubleshooting experience.",
        "skills": {"Languages": "Python", "Other": "Customer service, Outreach"},
        "experience": experience,
        "projects": [],
        "education": edu_text,
    }


def pick_job_only_word(job: dict, profile: dict) -> str | None:
    """A distinctive word from the job's own description that isn't
    anywhere in the candidate's declared skills -- a job-specific, varying
    fabricated-skill candidate rather than a fixed hand-picked example."""
    allowed = _build_skills_set(profile)
    words = sorted(_distinctive_words((job.get("full_description") or "") + " " + (job.get("title") or "")))
    random.shuffle(words)
    for w in words:
        # Exclude generic category/qualifier words (the validator's own
        # _SKILL_CLAIM_STOPWORDS) -- a real fabrication is a plausible
        # TOOL/TECH NAME ("Databricks"), not a bare word like "Language"
        # or "Proficient"; picking those would test whether the check
        # ignores noise, not whether it catches genuine fabrication.
        if w in _SKILL_CLAIM_STOPWORDS:
            continue
        if not any(w in a or a in w for a in allowed):
            return w
    return None


def make_skill_variant(clean: dict, job: dict, profile: dict) -> dict | None:
    word = pick_job_only_word(job, profile)
    if not word:
        return None
    variant = copy.deepcopy(clean)
    variant["skills"]["Fabricated"] = word.title()
    return variant


def make_date_variant(clean: dict, profile: dict) -> dict | None:
    dated_entries = [
        item for item in profile.get("experience_inventory") or [] if item.get("start_date") and item.get("end_date")
    ]
    if not dated_entries:
        return None
    item = random.choice(dated_entries)
    real_start_year = int(item["start_date"][:4])
    real_end_year = int(item["end_date"][:4])
    offset = random.choice([-1, 1]) * random.randint(3, 12)
    fake_year = real_start_year + offset
    fake_year2 = fake_year + max(0, real_end_year - real_start_year)

    variant = copy.deepcopy(clean)
    header_needle = f"{item.get('role_title') or item.get('role_type')} at {item['name']}"
    for entry in variant["experience"]:
        if entry["header"] == header_needle:
            entry["subtitle"] = f"{fake_year} - {fake_year2}"
            break
    else:
        return None
    return variant


def make_bare_header_variant(clean: dict, job: dict) -> dict:
    variant = copy.deepcopy(clean)
    company = job.get("company") or job.get("site") or "Unknown Employer"
    desc_words = (job.get("full_description") or "")[:200]
    variant["experience"].insert(
        0,
        {
            "header": f"{job.get('title', 'Employee')} at {company}",
            "subtitle": "",
            "bullets": [f"Handled responsibilities including {desc_words}".strip()],
        },
    )
    return variant


def make_keyword_injection_variant(clean: dict, job: dict) -> dict | None:
    job_words = sorted(_distinctive_words((job.get("full_description") or "") + " " + (job.get("title") or "")))
    if len(job_words) < 3:
        return None
    random.shuffle(job_words)
    picked = job_words[:3]
    variant = copy.deepcopy(clean)
    if not variant["experience"]:
        return None
    entry = random.choice(variant["experience"])
    if not entry["bullets"]:
        return None
    idx = random.randrange(len(entry["bullets"]))
    entry["bullets"][idx] = (
        entry["bullets"][idx].rstrip(".") + f", utilizing {picked[0]} and {picked[1]} to support {picked[2]}."
    )
    return variant


def get_representative_job_sample(conn, n: int) -> list[dict]:
    rows = conn.execute(
        "SELECT rowid, url, title, company, site, location, full_description "
        "FROM jobs WHERE full_description IS NOT NULL AND length(full_description) > 500"
    ).fetchall()
    all_jobs = [dict(r) for r in rows]
    print(f"Population of qualifying jobs (full_description > 500 chars): {len(all_jobs)}")
    if len(all_jobs) <= n:
        return all_jobs
    return random.sample(all_jobs, n)


_DETECTION_MARKERS = {
    "skill": ("fabricated skill", "not grounded in profile", "unsupported technical skill"),
    "date": ("fabricated date(s)",),
    "bare_header": ("unrecognized employer", "could not be matched"),
    "keyword_injection": ("graft job-posting language",),
}


def _detected(condition: str, result: dict) -> bool:
    """Whether THIS SPECIFIC fabrication mechanism's own signal fired --
    not just whether validation failed for some unrelated coincidental
    reason. bare_header/keyword_injection are deliberately WARNING-tier
    (advisory, not blocking), so their detection lives in warnings, not
    errors -- "passed" alone would never count them."""
    markers = _DETECTION_MARKERS[condition]
    haystack = " ".join(result["errors"] + result["warnings"]).lower()
    return any(m in haystack for m in markers)


def _false_positive(clean_result: dict) -> bool:
    """Whether the CLEAN control incorrectly tripped any of the four
    fabrication-specific signals (not unrelated noise like the
    optional-projects-field warning)."""
    haystack = " ".join(clean_result["errors"] + clean_result["warnings"]).lower()
    return any(m in haystack for markers in _DETECTION_MARKERS.values() for m in markers)


def main() -> None:
    conn = database.get_connection()
    profile = config.load_profile()
    clean = build_clean_resume(profile)

    # Sanity-check the clean baseline actually passes before running the
    # full sweep -- if it doesn't, every trial would be confounded.
    sanity = validate_json_fields(copy.deepcopy(clean), profile, standup_decision="INCLUDE")
    print("Clean baseline sanity check:", "PASS" if sanity["passed"] else "FAIL")
    if not sanity["passed"]:
        print("Errors:", sanity["errors"])
        print("ABORTING -- fix the clean baseline before running the statistical sweep.")
        return

    jobs = get_representative_job_sample(conn, SAMPLE_SIZE)
    print(f"Sampled {len(jobs)} jobs (seed={SEED})")

    results: dict[str, list[int]] = {"clean_false_positive": [], "skill": [], "date": [], "bare_header": [], "keyword_injection": []}
    misses: dict[str, list] = {"skill": [], "keyword_injection": []}
    skipped: dict[str, int] = {"skill": 0, "date": 0, "keyword_injection": 0}

    for i, job in enumerate(jobs):
        # Clean control -- should NOT fail (measures false positives).
        v = validate_json_fields(copy.deepcopy(clean), profile, standup_decision="INCLUDE", job=job)
        results["clean_false_positive"].append(1 if _false_positive(v) else 0)

        skill_variant = make_skill_variant(clean, job, profile)
        if skill_variant is None:
            skipped["skill"] += 1
        else:
            v = validate_json_fields(skill_variant, profile, standup_decision="INCLUDE", job=job)
            hit = _detected("skill", v)
            results["skill"].append(1 if hit else 0)
            if not hit:
                misses["skill"].append({"job": job.get("title"), "word": skill_variant["skills"].get("Fabricated")})

        date_variant = make_date_variant(clean, profile)
        if date_variant is None:
            skipped["date"] += 1
        else:
            v = validate_json_fields(date_variant, profile, standup_decision="INCLUDE", job=job)
            results["date"].append(1 if _detected("date", v) else 0)

        bare_variant = make_bare_header_variant(clean, job)
        v = validate_json_fields(bare_variant, profile, standup_decision="INCLUDE", job=job)
        results["bare_header"].append(1 if _detected("bare_header", v) else 0)

        kw_variant = make_keyword_injection_variant(clean, job)
        if kw_variant is None:
            skipped["keyword_injection"] += 1
        else:
            v = validate_json_fields(kw_variant, profile, standup_decision="INCLUDE", job=job)
            hit = _detected("keyword_injection", v)
            results["keyword_injection"].append(1 if hit else 0)
            if not hit:
                misses["keyword_injection"].append({"job": job.get("title"), "variant": kw_variant["experience"]})

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(jobs)}")

    summary = {"seed": SEED, "sample_size": len(jobs), "skipped": skipped, "conditions": {}}
    for condition, outcomes in results.items():
        n = len(outcomes)
        x = sum(outcomes)
        p = x / n if n else 0.0
        ci95 = wilson_ci(x, n, 0.95)
        ci99 = wilson_ci(x, n, 0.99)
        label = "false_positive_rate" if condition == "clean_false_positive" else "catch_rate"
        summary["conditions"][condition] = {
            "n": n,
            "hits": x,
            label: round(p, 4),
            "ci_95": [round(ci95[0], 4), round(ci95[1], 4)],
            "ci_99": [round(ci99[0], 4), round(ci99[1], 4)],
        }
        print(f"\n{condition}: {x}/{n} = {p:.1%}  ({label})")
        print(f"  95% CI: [{ci95[0]:.1%}, {ci95[1]:.1%}]")
        print(f"  99% CI: [{ci99[0]:.1%}, {ci99[1]:.1%}]")

    summary["misses"] = misses
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
