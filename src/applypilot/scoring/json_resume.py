"""JSON Resume (jsonresume.org) interop for ApplyPilot.

Two-way conversion between our hand-authored ``resume.txt`` master format
and the open `JSON Resume schema <https://jsonresume.org>`_, with schema
validation and a round-trip audit that surfaces gaps + lossy fields.

Entry points:

* :func:`to_json_resume(text)` — parse master text → JSON Resume ``dict``
* :func:`from_json_resume(jr)` — JSON Resume ``dict`` → master text
* :func:`validate(jr)` — return list of jsonschema validation errors
* :func:`audit(text)` — round-trip and return ``{json, errors, gaps,
  lossy, schema_unused, suggestions}`` for review

The bundled schema lives at ``jsonresume_schema.json`` next to this
module; updates can drop in a new copy from the upstream repo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Reuse the existing text parser so we share canonical section semantics.
from applypilot.scoring.pdf import parse_resume, parse_skills, parse_entries

_SCHEMA_PATH = Path(__file__).with_name("jsonresume_schema.json")
_SCHEMA = json.loads(_SCHEMA_PATH.read_text())

# ── Date-range parsing ───────────────────────────────────────────────────

_MONTHS = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}

_DATE_RANGE_RE = re.compile(
    r"""
    ^
    \s*
    (?P<start_mon>[A-Za-z]+)?\s*(?P<start_year>\d{4})
    \s*[-–—]\s*
    (?P<end>(?:Present|Current|(?P<end_mon>[A-Za-z]+)?\s*(?P<end_year>\d{4})))
    \s*
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _to_iso(month: str | None, year: str | None) -> str | None:
    """Return ``YYYY-MM`` or ``YYYY`` matching the schema's iso8601 pattern."""
    if not year:
        return None
    if month and month.lower() in _MONTHS:
        return f"{year}-{_MONTHS[month.lower()]}"
    return year


def _parse_date_range(s: str) -> tuple[str | None, str | None]:
    """Parse human date range → (startDate, endDate). 'Present' → (start, None)."""
    if not s:
        return None, None
    m = _DATE_RANGE_RE.match(s.strip())
    if not m:
        return None, None
    start = _to_iso(m.group("start_mon"), m.group("start_year"))
    end_text = (m.group("end") or "").strip()
    if end_text.lower() in ("present", "current"):
        return start, None
    end = _to_iso(m.group("end_mon"), m.group("end_year"))
    return start, end


# ── Title-line parsing ───────────────────────────────────────────────────

_PAREN_URL_RE = re.compile(r"\(([a-z0-9.-]+\.[a-z]{2,})\)", re.IGNORECASE)


def _split_title_pipes(title: str) -> dict:
    """Decompose a pipe-delimited entry title into JSON Resume work fields.

    Examples handled::

        "Uber | Developer Advocate | Cadence Workflows | Core Platform | Seattle, WA"
        "Uber (engineering.uber.com) | Developer Advocate | ... | Seattle, WA"
        "El Ninja | Founder & Principal Engineer | Seattle, WA"
    """
    parts = [p.strip() for p in title.split("|") if p.strip()]
    out = {"name": None, "url": None, "position": None,
           "description": None, "location": None}
    if not parts:
        return out

    # First piece: company, optionally with "(domain.tld)" annotation.
    first = parts[0]
    url_m = _PAREN_URL_RE.search(first)
    if url_m:
        out["url"] = f"https://{url_m.group(1)}"
        first = _PAREN_URL_RE.sub("", first).strip()
    out["name"] = first

    # Last piece: location if it looks like one ("City, ST" or contains "Remote").
    if len(parts) > 1:
        last = parts[-1]
        if (re.match(r"^[A-Z][A-Za-z .-]+,\s*[A-Z]{2}\b", last)
                or "Remote" in last or "Seattle" in last):
            out["location"] = last
            parts = parts[:-1]

    # Middle pieces: position is parts[1] (if present), the rest is description
    # — we keep parenthetical URLs in description so the original text round-trips.
    if len(parts) > 1:
        out["position"] = parts[1]
    if len(parts) > 2:
        out["description"] = " | ".join(parts[2:])

    return out


def _join_title_pipes(work: dict) -> str:
    """Inverse of _split_title_pipes."""
    name = work.get("name") or ""
    if work.get("url"):
        host = re.sub(r"^https?://", "", work["url"]).rstrip("/")
        name = f"{name} ({host})" if name else f"({host})"
    parts = [p for p in (
        name,
        work.get("position"),
        work.get("description"),
        work.get("location"),
    ) if p]
    return " | ".join(parts)


# ── Contact line parsing ─────────────────────────────────────────────────

_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/([A-Za-z0-9_-]+)", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9_-]+)", re.IGNORECASE)
_LOC_RE = re.compile(r"^([A-Za-z .'-]+),\s*([A-Z]{2})\b")


def _parse_contact(text: str) -> tuple[dict, list[dict]]:
    """Return (basics-fragment, profiles-list) extracted from the contact line."""
    basics: dict = {}
    profiles: list[dict] = []

    if e := _EMAIL_RE.search(text):
        basics["email"] = e.group(0)
    if p := _PHONE_RE.search(text):
        basics["phone"] = p.group(0)

    if li := _LINKEDIN_RE.search(text):
        username = li.group(1)
        profiles.append({
            "network": "LinkedIn",
            "username": username,
            "url": f"https://linkedin.com/in/{username}",
        })
    if gh := _GITHUB_RE.search(text):
        username = gh.group(1)
        profiles.append({
            "network": "GitHub",
            "username": username,
            "url": f"https://github.com/{username}",
        })
    return basics, profiles


def _parse_location(loc_text: str) -> dict:
    """Parse 'Seattle, WA' → {city, region, countryCode}."""
    m = _LOC_RE.match((loc_text or "").strip())
    if not m:
        return {"address": loc_text} if loc_text else {}
    return {"city": m.group(1).strip(), "region": m.group(2),
            "countryCode": "US"}


# ── Earlier-experience one-liner parsing ─────────────────────────────────

# "Underground Elephant (ue.co) | Software Engineer (Apr 2016 - Jan 2017): Led ZipQuote..."
# Title (with optional parenthetical URL) | Position (Date - Date): Summary
_EARLIER_RE = re.compile(
    r"""
    ^
    (?P<title>[^|]+?)\s*\|\s*
    (?P<position>.+?)\s*
    \((?P<dates>(?:[A-Za-z]+\s+)?\d{4}\s*[-–—]\s*(?:[A-Za-z]+\s+)?\d{4})\)
    :\s*
    (?P<summary>.+)
    $
    """,
    re.VERBOSE,
)


def _looks_like_earlier(line: str) -> bool:
    return bool(_EARLIER_RE.match(line.strip()))


def _parse_earlier_line(line: str) -> dict | None:
    m = _EARLIER_RE.match(line.strip())
    if not m:
        return None
    title = m.group("title").strip()
    # Reuse paren-URL extraction by feeding "Foo (bar.com) |" — the trailing
    # pipe makes _split_title_pipes treat the first piece as the company.
    name_url = _split_title_pipes(title + " |")
    start, end = _parse_date_range(m.group("dates"))
    out = {
        "name": name_url["name"],
        "url": name_url["url"],
        "position": m.group("position").strip(),
        "startDate": start,
        "endDate": end,
        "summary": m.group("summary").strip(),
        # Custom marker so from_json_resume() can render this as a one-line
        # EARLIER EXPERIENCE entry instead of a full PROFESSIONAL block.
        "_earlier": True,
    }
    return {k: v for k, v in out.items() if v}


# ── Languages line parsing ───────────────────────────────────────────────

_LANG_LINE_RE = re.compile(r"\bLanguages:\s*([^|]+)", re.IGNORECASE)
_LANG_ENTRY_RE = re.compile(r"([A-Za-z][A-Za-z -]+?)\s*(?:\(([^)]+)\))?\s*(?:,|$)")


# ── Project-bullet detection (El Ninja Founder section) ─────────────────

# A "project bullet" looks like:  "Name [(...annotation...)]: description text"
# where Name is a short capitalized identifier (Pursuit, Meridian, ApplyPilot,
# Lomito, Elminarete, BMW Motorrad diagnostics, OpenAI Parameter Golf, etc.).
_PROJECT_BULLET_RE = re.compile(
    r"""
    ^
    (?P<name>[A-Za-z][A-Za-z0-9 &+./'-]+?)        # project name, allow words/spaces
    \s*
    (?:\(\s*(?P<annotation>[^)]{2,200}?)\s*\))?   # optional (annotation)
    \s*[:—-]\s*                                   # separator
    (?P<rest>.+)                                  # description
    $
    """,
    re.VERBOSE,
)


def _extract_project(bullet: str) -> dict | None:
    """Try to interpret a bullet as a self-contained project description.

    Returns a JSON Resume project entry, or None if the bullet doesn't
    fit the pattern.
    """
    m = _PROJECT_BULLET_RE.match(bullet.strip())
    if not m:
        return None

    name = m.group("name").strip()
    annotation = (m.group("annotation") or "").strip()
    rest = m.group("rest").strip()

    # Project names should look like proper nouns — short, no leading verbs.
    if len(name) > 60 or " " in name and name.lower().startswith(
        ("provisioned", "migrated", "rebuilt", "developed", "enhanced",
         "designed", "built", "led", "managed", "implemented", "contributed",
         "architected", "reduced", "improved", "engineered", "established",
         "founded", "debugged", "deployed", "continued", "reverse-engineered")
    ):
        return None

    project: dict = {"name": name, "description": rest}

    # Pull a URL out of the annotation if present (e.g. "(elminarete.com)").
    # Non-URL annotations (e.g. "(co-founder, AI Scalathon Seattle 2026)")
    # go into the dedicated `_annotation` field — `entity` is reserved for
    # the parent company and gets populated by the caller.
    if annotation:
        url_m = re.search(r"([a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?)", annotation, re.I)
        if url_m and "." in url_m.group(0) and "http" not in annotation:
            project["url"] = "https://" + url_m.group(0).strip("/.")
        else:
            # Use a custom field; JSON Resume schema is open (additionalProperties).
            project["_annotation"] = annotation

    # Heuristic keyword extraction: pick out tech tokens we recognize.
    keyword_pool = {
        "TypeScript", "JavaScript", "Python", "Kotlin", "Java", "Go", "Rust", "PHP",
        "Astro", "Next.js", "React", "React Native", "Vue.js", "Node.js", "FastAPI",
        "Django", "Laravel", "Micronaut", "Spring Boot",
        "PostgreSQL", "Postgres", "PostGIS", "Supabase", "MySQL", "MongoDB", "Redis",
        "SQLite", "Cassandra", "CockroachDB",
        "Kubernetes", "Docker", "AWS", "GCP", "Cloudflare", "Terraform",
        "Cadence", "Temporal", "Apache Kafka", "gRPC",
        "MCP", "Playwright", "Patchright", "Chrome DevTools Protocol", "Claude Code",
        "Browser-Use", "Stripe", "Mapbox GL", "Mapbox",
        "Ghidra", "OpenOCD", "SWD", "ARM Cortex-M", "ARM Cortex-M4", "FTDI",
        "KWP2000", "BEST/2", "NMEA 2000", "AIS", "OpenCPN",
        "RunPod", "GPTQ", "Triton", "H100", "8xH100", "OpenAI",
        "DALL·E", "Flux", "Gemini", "Nano Banana",
    }
    text = f"{annotation} {rest}"
    keywords = sorted({kw for kw in keyword_pool if kw.lower() in text.lower()})
    if keywords:
        project["keywords"] = keywords

    return project


def _parse_languages(text: str) -> list[dict]:
    """'English (Native), Spanish (Native)' → list of {language, fluency}."""
    out: list[dict] = []
    seen: set[str] = set()
    for m in _LANG_ENTRY_RE.finditer(text):
        lang = m.group(1).strip()
        if not lang or lang.lower() in seen:
            continue
        seen.add(lang.lower())
        entry: dict = {"language": lang}
        if fluency := m.group(2):
            entry["fluency"] = fluency.strip()
        out.append(entry)
    return out


# ── Certifications + Education line ──────────────────────────────────────

# "Computer Science | Southwestern Community College | Certifications: A+, GCP ..."
_EDU_RE = re.compile(
    r"""
    ^
    (?P<area>[^|]+)\s*\|\s*
    (?P<institution>[^|]+?)\s*
    (?:\|\s*Certifications?:\s*(?P<certs>.+))?
    $
    """,
    re.VERBOSE,
)


def _parse_education_line(line: str) -> tuple[dict, list[dict]]:
    """Return (education-entry, certificates[]) from the master's EDUCATION line."""
    m = _EDU_RE.match(line.strip())
    if not m:
        return ({"institution": line.strip()}, [])
    edu = {
        "institution": m.group("institution").strip(),
        "area": m.group("area").strip(),
    }
    certs: list[dict] = []
    if cert_text := m.group("certs"):
        for c in cert_text.split(","):
            c = c.strip()
            if c:
                certs.append({"name": c})
    return edu, certs


# ── Public conversion API ────────────────────────────────────────────────

def to_json_resume(text: str, *, normalize_sections: bool = True) -> dict:
    """Convert the master ``resume.txt`` format into a JSON Resume document.

    Args:
        text: Full text of the master resume.
        normalize_sections: If True (default), apply the same section-name
            normalization the DOCX renderer uses (PROFESSIONAL EXPERIENCE →
            EXPERIENCE, EDUCATION & CERTIFICATIONS → EDUCATION, etc.).

    Returns:
        A dict matching the JSON Resume schema. May validate cleanly via
        :func:`validate`; missing optional fields are simply omitted rather
        than left empty.
    """
    if normalize_sections:
        # Mirror the same one-shot normalization used elsewhere in this
        # module. Keep it in sync with /tmp/render_master.py logic.
        lines = text.strip().split("\n")
        if not any(ln.strip() == "SUMMARY" for ln in lines):
            try:
                rest_idx = next(
                    i for i, ln in enumerate(lines[3:], start=3) if ln.strip()
                )
                headline = lines[rest_idx].strip()
                summary_text = lines[rest_idx + 1].strip()
                body_start = next(
                    i for i, ln in enumerate(lines[rest_idx + 2:], start=rest_idx + 2)
                    if ln.strip()
                )
                location_line = lines[1].strip()
                contact_line = lines[2].strip()
                location = "Seattle, WA"
                rest_contact = location_line.replace(f"{location} | ", "")
                contact_combined = f"{rest_contact} | {contact_line}"
                body = "\n".join(lines[body_start:])
                body = body.replace("PROFESSIONAL EXPERIENCE", "EXPERIENCE")
                body = body.replace("\nEARLIER EXPERIENCE\n", "\n")
                body = body.replace("EDUCATION & CERTIFICATIONS", "EDUCATION")
                text = (
                    f"{lines[0].strip()}\n"
                    f"{headline}\n"
                    f"{location}\n"
                    f"{contact_combined}\n\n"
                    f"SUMMARY\n{summary_text}\n\n"
                    f"{body}\n"
                )
            except (StopIteration, IndexError):
                pass  # leave text alone — parser will do its best

    parsed = parse_resume(text)

    # ── basics ──────────────────────────────────────────────────────────
    basics: dict = {"name": parsed["name"].strip()}
    if parsed.get("title"):
        basics["label"] = parsed["title"].strip()
    if parsed.get("location"):
        loc = _parse_location(parsed["location"])
        if loc:
            basics["location"] = loc

    contact_basics, profiles = _parse_contact(parsed.get("contact", ""))
    basics.update(contact_basics)
    if profiles:
        basics["profiles"] = profiles

    sections = parsed.get("sections", {})

    if "SUMMARY" in sections:
        basics["summary"] = sections["SUMMARY"].strip()

    jr: dict = {
        "$schema": "https://raw.githubusercontent.com/jsonresume/resume-schema/v1.0.0/schema.json",
        "basics": basics,
    }

    # ── work ────────────────────────────────────────────────────────────
    work: list[dict] = []
    if "EXPERIENCE" in sections:
        for entry in parse_entries(sections["EXPERIENCE"]):
            title = entry.get("title", "").strip()
            subtitle = entry.get("subtitle", "").strip()

            # Earlier-experience single-liners: full job in one line. They
            # have the shape "Company (...) | Position (Year - Year): ...".
            # When two consecutive one-liners appear, parse_entries collapses
            # the second into the first's `subtitle` slot — split it back out.
            if _looks_like_earlier(title):
                if e := _parse_earlier_line(title):
                    work.append(e)
                if subtitle and _looks_like_earlier(subtitle):
                    if e := _parse_earlier_line(subtitle):
                        work.append(e)
                continue

            split = _split_title_pipes(title)
            start, end = _parse_date_range(subtitle)
            w: dict = {}
            if split["name"]:        w["name"] = split["name"]
            if split["location"]:    w["location"] = split["location"]
            if split["description"]: w["description"] = split["description"]
            if split["position"]:    w["position"] = split["position"]
            if split["url"]:         w["url"] = split["url"]
            if start:                w["startDate"] = start
            if end:                  w["endDate"] = end
            if entry.get("bullets"): w["highlights"] = list(entry["bullets"])
            work.append(w)
    if work:
        jr["work"] = work

    # ── projects ───────────────────────────────────────────────────────
    # The El Ninja Founder section's highlights read as discrete projects
    # (Pursuit, Meridian, ApplyPilot, Lomito, Elminarete, BMW Motorrad,
    # firmware RE, OpenAI Parameter Golf, pintxo-marine-hub) rather than
    # ordinary job accomplishments. Promote any work entry whose role is
    # Founder/Co-Founder/Principal and whose highlights match the
    # "Name (...): description" pattern.
    projects: list[dict] = []
    for w in jr.get("work", []) or []:
        position = (w.get("position") or "").lower()
        is_founder = any(role in position for role in (
            "founder", "co-founder", "principal", "cto", "ceo",
        ))
        if not is_founder:
            continue
        kept_highlights: list[str] = []
        for bullet in (w.get("highlights") or []):
            project = _extract_project(bullet)
            if project:
                project.setdefault("entity", w.get("name") or "")
                project.setdefault("startDate", w.get("startDate"))
                project.setdefault("endDate", w.get("endDate"))
                # Drop empty default fields so validation stays clean.
                project = {k: v for k, v in project.items() if v not in (None, "", [])}
                projects.append(project)
            else:
                kept_highlights.append(bullet)
        # If we promoted ALL bullets, keep just an overall summary so the
        # work entry still anchors the role; otherwise keep the leftovers.
        if not kept_highlights and projects:
            w["summary"] = (
                "Operating El Ninja as a solo founder/principal engineer; "
                "shipping the projects listed below across AI tooling, "
                "civic platforms, embedded reverse-engineering, and "
                "infrastructure research."
            )
            w.pop("highlights", None)
        else:
            if kept_highlights:
                w["highlights"] = kept_highlights
            elif "highlights" in w:
                del w["highlights"]
    if projects:
        jr["projects"] = projects

    # ── skills ──────────────────────────────────────────────────────────
    # parse_skills doesn't strip the leading `* ` bullet marker that the
    # master uses, so do it here.
    skills: list[dict] = []
    if "TECHNICAL SKILLS" in sections:
        for cat, val in parse_skills(sections["TECHNICAL SKILLS"]):
            cat = re.sub(r"^[\*\-•]\s*", "", cat).strip()
            keywords = [k.strip() for k in val.split(",") if k.strip()]
            skills.append({"name": cat, "keywords": keywords})
    if skills:
        jr["skills"] = skills

    # ── education / certificates / languages (from EDUCATION section) ──
    education: list[dict] = []
    certificates: list[dict] = []
    languages: list[dict] = []

    if "EDUCATION" in sections:
        edu_text = sections["EDUCATION"].strip()
        for line in edu_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if _LANG_LINE_RE.search(line):
                lang_match = _LANG_LINE_RE.search(line)
                if lang_match:
                    languages = _parse_languages(lang_match.group(1))
                continue
            edu, certs = _parse_education_line(line)
            education.append(edu)
            certificates.extend(certs)

    if education:
        jr["education"] = education
    if certificates:
        jr["certificates"] = certificates
    if languages:
        jr["languages"] = languages

    return jr


def from_json_resume(jr: dict) -> str:
    """Render a JSON Resume dict back to the master ``resume.txt`` format.

    Section names match the master format (PROFESSIONAL EXPERIENCE,
    EARLIER EXPERIENCE, EDUCATION & CERTIFICATIONS) so output can drop in
    as the master directly.
    """
    out: list[str] = []
    basics = jr.get("basics", {}) or {}
    name = basics.get("name", "").strip()
    out.append(name)

    # Reconstruct the contact line from email/phone/location/profiles.
    loc = basics.get("location") or {}
    loc_str = ""
    if loc.get("city") and loc.get("region"):
        loc_str = f"{loc['city']}, {loc['region']}"
    elif loc.get("address"):
        loc_str = loc["address"]
    contact_bits = []
    if loc_str:
        contact_bits.append(loc_str)
    if basics.get("phone"):
        contact_bits.append(basics["phone"])
    if basics.get("email"):
        contact_bits.append(basics["email"])
    out.append(" | ".join(contact_bits))

    profile_bits = []
    for p in basics.get("profiles", []) or []:
        net = (p.get("network") or "").strip()
        url = p.get("url") or ""
        host = re.sub(r"^https?://", "", url).rstrip("/")
        if net and host:
            profile_bits.append(f"{net}: {host}")
    if profile_bits:
        out.append(" | ".join(profile_bits))

    out.append("")
    out.append("")

    if basics.get("label"):
        out.append(basics["label"])
    if basics.get("summary"):
        out.append(basics["summary"])

    # Skills
    skills = jr.get("skills") or []
    if skills:
        out.append("TECHNICAL SKILLS")
        for sk in skills:
            keywords = sk.get("keywords") or []
            line = f"* {sk.get('name', '')}: {', '.join(keywords)}"
            out.append(line)

    # Work — split into PROFESSIONAL EXPERIENCE (full multi-line entries)
    # and EARLIER EXPERIENCE (compact one-liners). The `_earlier` marker
    # is set by `_parse_earlier_line`; everything else renders professional
    # (even when its highlights got promoted to projects[] and replaced
    # with a summary).
    work = jr.get("work") or []
    early = [w for w in work if w.get("_earlier")]
    pro   = [w for w in work if not w.get("_earlier")]

    def _fmt_dates(w: dict) -> str:
        start = w.get("startDate") or ""
        end = w.get("endDate") or "Present"
        # Convert YYYY-MM → "Mon YYYY" for human format.
        def _h(s: str) -> str:
            if not s or s == "Present":
                return s
            parts = s.split("-")
            if len(parts) >= 2:
                month_idx = int(parts[1])
                month_name = ["", "January", "February", "March", "April", "May",
                              "June", "July", "August", "September", "October",
                              "November", "December"][month_idx]
                return f"{month_name} {parts[0]}"
            return s
        return f"{_h(start)} - {_h(end)}".strip(" -")

    # When projects[] is populated, fold each project back into the host
    # work entry's highlights when re-rendering text — so resume.txt round-
    # trips to the same shape it started in. Group by `entity` (parent
    # company); use the URL-as-host when no other annotation exists.
    projects = jr.get("projects") or []
    project_bullets_by_entity: dict[str, list[str]] = {}
    for p in projects:
        entity = p.get("entity") or ""
        name = p.get("name", "")
        ann = p.get("_annotation")
        if not ann and p.get("url"):
            ann = re.sub(r"^https?://", "", p["url"]).rstrip("/")
        prefix = name + (f" ({ann})" if ann else "")
        bullet = f"{prefix}: {p.get('description', '')}".strip()
        project_bullets_by_entity.setdefault(entity, []).append(bullet)

    pro = list(pro)  # avoid mutating the cached list above
    if pro:
        out.append("PROFESSIONAL EXPERIENCE")
        for w in pro:
            out.append(_join_title_pipes(w))
            d = _fmt_dates(w)
            if d:
                out.append(d)
            entity_name = w.get("name") or ""
            highlights = list(w.get("highlights") or [])
            highlights = highlights + project_bullets_by_entity.pop(entity_name, [])
            for h in highlights:
                out.append(f"* {h}")

    if early:
        out.append("EARLIER EXPERIENCE")
        for w in early:
            title = _join_title_pipes(w)
            dates = _fmt_dates(w)
            summary = w.get("summary", "")
            line = f"{title} ({dates}): {summary}".strip()
            out.append(line)

    # Education + certifications + languages
    edu = jr.get("education") or []
    certs = jr.get("certificates") or []
    langs = jr.get("languages") or []
    if edu or certs or langs:
        out.append("EDUCATION & CERTIFICATIONS")
        for e in edu:
            bits = [b for b in (e.get("area"), e.get("institution")) if b]
            line = " | ".join(bits)
            if certs:
                line += " | Certifications: " + ", ".join(c.get("name", "") for c in certs)
                certs = []  # only attach to the first edu entry
            out.append(line)
        if langs:
            lang_bits = [
                f"{l.get('language', '')}" + (f" ({l['fluency']})" if l.get("fluency") else "")
                for l in langs
            ]
            out.append("Languages: " + ", ".join(lang_bits))

    return "\n".join(out) + "\n"


# ── Validation ───────────────────────────────────────────────────────────

def validate(jr: dict) -> list[str]:
    """Return list of validation-error messages (empty list = valid)."""
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema package not installed"]

    validator = jsonschema.Draft7Validator(_SCHEMA)
    errors = sorted(validator.iter_errors(jr), key=lambda e: e.path)
    return [f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]


# ── Audit ────────────────────────────────────────────────────────────────

def _flatten(text: str) -> str:
    """Normalize whitespace + lowercase for fuzzy diffs."""
    return re.sub(r"\s+", " ", text).strip().lower()


def audit(text: str) -> dict:
    """Round-trip the master text through JSON Resume and report findings.

    Returns a dict with::

        {
          "json": <JSON Resume dict>,
          "errors": list[str],         # schema violations
          "schema_unused": list[str],  # top-level schema sections not populated
          "lossy": list[str],          # parts of the original lost in round-trip
          "suggestions": list[str],    # opportunities to enrich the data
        }
    """
    jr = to_json_resume(text)
    errors = validate(jr)

    schema_top = list(_SCHEMA.get("properties", {}).keys())
    populated = {k for k, v in jr.items() if v}
    schema_unused = [k for k in schema_top if k not in populated and not k.startswith("$")]

    # Round-trip check.
    re_text = from_json_resume(jr)

    lossy: list[str] = []
    flat_orig = _flatten(text)
    flat_round = _flatten(re_text)
    # Any 60+-char chunks in the original that don't appear after the round-trip
    # are likely lost.
    for chunk in re.findall(r"[^\n]{60,}", text):
        if _flatten(chunk) not in flat_round:
            lossy.append(chunk[:200])

    # Opportunities to enrich.
    suggestions: list[str] = []
    basics = jr.get("basics", {})
    if "url" not in basics:
        suggestions.append(
            "basics.url — homepage / portfolio. Resume currently has no top-level personal site."
        )
    if "image" not in basics:
        suggestions.append(
            "basics.image — headshot URL. Optional but well-supported by JSON Resume themes."
        )
    if not jr.get("projects"):
        suggestions.append(
            "projects — the El Ninja Founder bullets (Pursuit, Meridian, ApplyPilot, "
            "Lomito, Elminarete, BMW Motorrad diagnostics, OpenAI Parameter Golf, "
            "pintxo-marine-hub) read as discrete projects rather than one job's "
            "highlights — would render better as JSON Resume `projects[]` entries "
            "with their own dates, urls, and keywords."
        )
    if not jr.get("publications") and "OpenAI Parameter Golf" in text:
        suggestions.append(
            "publications — `OpenAI Parameter Golf` leaderboard entries could "
            "surface as publications[] with the leaderboard URL as `url`."
        )
    if not jr.get("awards"):
        suggestions.append(
            "awards — none currently. If Pursuit's $500K pre-seed counts as a "
            "milestone, it could land here with a date + summary."
        )
    if "Citizenship" in text and not any(
        "citizen" in str(v).lower() for v in jr.get("basics", {}).values()
    ):
        suggestions.append(
            "Citizenship: Dual US-Mexico — JSON Resume has no native field for "
            "this. Convention is to put it in basics.summary or a custom property "
            "(schema is open via additionalProperties)."
        )
    if "meta" not in jr:
        suggestions.append(
            "meta — populate `version`/`canonical`/`lastModified` for resume "
            "consumers that surface those fields."
        )

    return {
        "json": jr,
        "errors": errors,
        "schema_unused": schema_unused,
        "lossy": lossy,
        "suggestions": suggestions,
    }
