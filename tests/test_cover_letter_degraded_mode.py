"""Tests for the 2026-09-12 cover-letter degraded-mode fallback.

Covers:
- local_tailor's new template-building helpers (_pick_supported_requirements,
  _clean_snippet, _gather_evidence_sentences, _build_degraded_cover_paragraphs)
  and the compose_degraded_cover_letter orchestrator -- all deterministic,
  zero LLM calls except through the already-tested select_and_edit_bank_
  bullets pipeline (mocked here).
- cover_letter.generate_cover_letter's new degraded-mode redirect: the
  steady-state fast-path check and the "discovery" RuntimeError branch,
  both gated on is_local_configured().
- A real, pre-existing (not introduced by this change) validator bug found
  while testing the above against a real job/profile: validate_cover_
  letter's education-field integrity check used to fire whenever BOTH
  official_degree/field_of_study were populated on the profile, regardless
  of whether the letter mentioned education at all -- an unconditional
  "must always restate your degree field" requirement rather than the
  anti-fabrication guard ("if you claim the degree, get the field right")
  its own sibling check two lines below already correctly modeled.
"""

from unittest.mock import Mock

import pytest

from applypilot.scoring import local_tailor
from applypilot.scoring.validator import validate_cover_letter

# ── Fixtures ──────────────────────────────────────────────────────────────


def _stub_client(reply: str = "Understood, thanks for reaching out.") -> Mock:
    """A client whose .chat() returns a real, benign string -- needed since
    2026-09-13's filler-polish feature means compose_degraded_cover_letter
    now genuinely calls client.chat() (for the hook-closer/close polish),
    not just the mocked-away bank-editor path. A bare Mock()'s .chat()
    returns another Mock, which crashes deep inside check_banned_patterns's
    re.search() -- a real mocking gap, not a production bug (a real
    LLMClient.chat() always returns str)."""
    client = Mock()
    client.chat.return_value = reply
    return client

PROFILE = {
    "personal": {"full_name": "Jordan Alexander Lee", "preferred_name": "Jordan"},
    "skills_boundary": {"languages": ["Python"]},
    "resume_facts": {},
    "experience_inventory": [
        {
            "name": "Mavis",
            "role_title": "Automotive Technician",
            "responsibilities": [
                "Diagnosed and repaired vehicle alignment issues for walk-in customers every shift.",
                "Explained repair options and cost tradeoffs directly to customers before starting work.",
            ],
        },
        {
            "name": "Waffle House",
            "role_title": "Server",
            "responsibilities": [
                "Took orders and resolved complaints from customers during high-volume rush periods.",
                "Trained new servers on the register and the floor during their first two weeks.",
            ],
        },
    ],
}

JOB = {
    "url": "https://example.com/job/1",
    "title": "IT Support Technician",
    "company": "Acme Corp",
    "location": "Remote",
    "full_description": "Acme Corp needs someone who can support end users directly.",
}


def _req(text, supported=True, evidence=("Mavis",), tier="prototype"):
    return {
        "requirement": text,
        "importance": "required",
        "supported": supported,
        "resume_evidence": list(evidence) if supported else [],
        "category_tier": tier if supported else None,
    }


RICH_JOB_SCHEMA = {
    "job_url": JOB["url"],
    "requirements": [
        _req(
            "Three or more years of direct, hands-on experience diagnosing customer equipment problems",
            tier="prototype",
        ),
        _req(
            "Comfortable explaining technical repair options directly to walk-in customers every day",
            tier="near_prototype",
        ),
        _req(
            "Willing to resolve customer complaints calmly during high-volume, high-pressure periods",
            evidence=("Waffle House",),
            tier="prototype",
        ),
        _req(
            "Available for shift work including some evenings and weekends as needed",
            evidence=("Waffle House",),
            tier="peripheral",
        ),
        _req(
            "Bachelor's degree in mechanical or electrical engineering preferred but not required",
            tier="near_prototype",
        ),
        _req("Experience with cloud infrastructure and container orchestration", supported=False),
    ],
    "viewpoint": "support",
    "summary_schema": "generic",
    "evidence_considered": 2,
}

_BANK_SENTENCES = {
    "Mavis": [
        "I diagnosed and repaired vehicle alignment issues for walk-in customers on every shift I "
        "worked, without a second technician double-checking my work, even during the busiest weeks.",
        "I explained repair options and real cost tradeoffs directly to customers before any work "
        "began, so nobody was surprised by the final bill or felt talked into anything.",
    ],
    "Waffle House": [
        "I took orders and resolved customer complaints on my own during the busiest rush periods, "
        "when the whole floor was full and short-staffed and everyone else was already stretched thin.",
        "I trained new servers on the register and the floor during their first two weeks on shift.",
    ],
}


def _patch_bank(monkeypatch, fully_covered=True):
    monkeypatch.setattr(
        local_tailor,
        "select_and_edit_bank_bullets",
        lambda client, job_schema, profile, **_kwargs: (dict(_BANK_SENTENCES), fully_covered),
    )


# ── _pick_supported_requirements ────────────────────────────────────────────


def test_pick_supported_requirements_orders_by_tier_and_filters_unsupported():
    picked = local_tailor._pick_supported_requirements(RICH_JOB_SCHEMA, limit=10)
    assert all(r["supported"] for r in picked)
    tiers = [r["category_tier"] for r in picked]
    assert tiers.index("prototype") < tiers.index("near_prototype") < tiers.index("peripheral")
    assert "cloud infrastructure" not in " ".join(r["requirement"] for r in picked)


def test_pick_supported_requirements_respects_limit():
    picked = local_tailor._pick_supported_requirements(RICH_JOB_SCHEMA, limit=2)
    assert len(picked) == 2


def test_pick_supported_requirements_skips_requirement_with_no_evidence():
    schema = {"requirements": [_req("Something supported but with no evidence tag", evidence=())]}
    schema["requirements"][0]["resume_evidence"] = []
    assert local_tailor._pick_supported_requirements(schema) == []


# ── _clean_snippet ──────────────────────────────────────────────────────────


def test_clean_snippet_collapses_whitespace_and_adds_period():
    assert local_tailor._clean_snippet("  multiple   spaces  here  ") == "multiple spaces here."


def test_clean_snippet_strips_trailing_punctuation_before_adding_period():
    assert local_tailor._clean_snippet("already punctuated,") == "already punctuated."


def test_clean_snippet_truncates_to_word_boundary():
    long_text = "one two three four five six seven eight nine ten"
    cleaned = local_tailor._clean_snippet(long_text, max_len=20)
    assert cleaned.endswith(".")
    assert len(cleaned) <= 21
    assert " " not in cleaned[-1:]  # never cuts mid-word


def test_clean_snippet_empty_input_returns_empty_string():
    assert local_tailor._clean_snippet("   ") == ""
    assert local_tailor._clean_snippet(None) == ""


# ── _ensure_first_person ─────────────────────────────────────────────────


def test_ensure_first_person_prepends_pronoun_to_bare_resume_bullet():
    """Real bug found by testing against a real job/profile: a resume-style
    bullet with an implied subject read as a broken fragment once dropped
    into cover-letter prose ("In practice, Develop and execute...")."""
    assert (
        local_tailor._ensure_first_person("Develop and execute targeted outreach.")
        == "I develop and execute targeted outreach."
    )


def test_ensure_first_person_preserves_verb_tense():
    assert (
        local_tailor._ensure_first_person("Diagnosed and repaired vehicle alignment issues.")
        == "I diagnosed and repaired vehicle alignment issues."
    )


def test_ensure_first_person_does_not_double_up_existing_pronoun():
    original = "I already diagnosed and repaired the issue."
    assert local_tailor._ensure_first_person(original) == original


def test_ensure_first_person_empty_input():
    assert local_tailor._ensure_first_person("") == ""


# ── _display_company_capitalized ─────────────────────────────────────────


def test_display_company_capitalized_titles_a_lowercase_slug():
    job = dict(JOB, company=None, application_url="https://jobs.ashbyhq.com/ramp/abc")
    assert local_tailor._display_company_capitalized(job) == "Ramp"


def test_display_company_capitalized_preserves_deliberate_mixed_case():
    job = dict(JOB, company="eBay")
    assert local_tailor._display_company_capitalized(job) == "eBay"


def test_display_company_capitalized_preserves_original_company_column():
    assert local_tailor._display_company_capitalized(JOB) == "Acme Corp"


# ── _gather_evidence_sentences ───────────────────────────────────────────


def test_gather_evidence_sentences_uses_bank_when_available(monkeypatch):
    _patch_bank(monkeypatch, fully_covered=True)
    sentences, names, fully_covered = local_tailor._gather_evidence_sentences(
        Mock(), RICH_JOB_SCHEMA, PROFILE, limit=4
    )
    assert fully_covered is True
    assert sentences  # got real bank content
    assert all(s.endswith(".") for s in sentences)
    assert set(names) <= {"Mavis", "Waffle House"}


def test_gather_evidence_sentences_falls_back_to_source_facts_when_no_bank(monkeypatch):
    monkeypatch.setattr(local_tailor, "select_and_edit_bank_bullets", lambda client, job_schema, profile, **_kwargs: ({}, False))
    sentences, names, fully_covered = local_tailor._gather_evidence_sentences(
        Mock(), RICH_JOB_SCHEMA, PROFILE, limit=4
    )
    assert fully_covered is False
    assert sentences  # still got something -- real, verbatim (plus a subject pronoun) responsibilities text
    # Every returned sentence must be traceable to real profile text (no
    # fabrication) once the added "I " subject (_ensure_first_person) is
    # accounted for -- the fact itself is never altered, only a pronoun.
    all_real_facts = " ".join(
        r for item in PROFILE["experience_inventory"] for r in item["responsibilities"]
    ).lower()
    for s in sentences:
        without_pronoun = s.rstrip(".")
        if without_pronoun.lower().startswith("i "):
            without_pronoun = without_pronoun[2:]
        assert without_pronoun.lower() in all_real_facts


def test_gather_evidence_sentences_empty_when_nothing_available(monkeypatch):
    monkeypatch.setattr(local_tailor, "select_and_edit_bank_bullets", lambda client, job_schema, profile, **_kwargs: ({}, False))
    empty_schema = {"requirements": [_req("no matching evidence here", evidence=("Unknown",))]}
    sentences, names, fully_covered = local_tailor._gather_evidence_sentences(Mock(), empty_schema, PROFILE)
    assert sentences == []
    assert names == []
    assert fully_covered is False


def test_gather_evidence_sentences_passes_cl_banned_patterns_to_bank_selector(monkeypatch):
    """Real bug found live, twice: the editor's own rewording introduced a
    cover-letter-banned style word ("demonstrate"/"align with") it has no
    knowledge of. Confirms _gather_evidence_sentences opts the editor in to
    checking for it, via select_and_edit_bank_bullets's banned_patterns."""
    from applypilot.scoring.validator import CL_BANNED_PATTERNS

    captured = {}

    def _fake_select(client, job_schema, profile, **kwargs):
        captured["banned_patterns"] = kwargs.get("banned_patterns")
        return dict(_BANK_SENTENCES), True

    monkeypatch.setattr(local_tailor, "select_and_edit_bank_bullets", _fake_select)
    local_tailor._gather_evidence_sentences(Mock(), RICH_JOB_SCHEMA, PROFILE, limit=4)
    assert captured["banned_patterns"] == CL_BANNED_PATTERNS


def test_gather_evidence_sentences_drops_a_bank_sentence_containing_a_banned_phrase(monkeypatch):
    """Belt-and-suspenders: even if select_and_edit_bank_bullets somehow
    returns a sentence containing a banned phrase (e.g. the editor's
    fallback-to-original path doesn't itself re-check the original),
    _gather_evidence_sentences must never ship it -- it should be dropped,
    not patched, and a clean alternative used instead."""
    monkeypatch.setattr(
        local_tailor,
        "select_and_edit_bank_bullets",
        lambda client, job_schema, profile, **_kwargs: (
            {
                "Mavis": [
                    "I demonstrate strong diagnostic ability with alignment equipment.",  # banned
                    "I diagnosed and repaired vehicle alignment issues for walk-in customers.",  # clean
                ]
            },
            True,
        ),
    )
    sentences, _names, _covered = local_tailor._gather_evidence_sentences(Mock(), RICH_JOB_SCHEMA, PROFILE, limit=4)
    assert not any("demonstrate" in s.lower() for s in sentences)
    assert any("diagnosed and repaired" in s.lower() for s in sentences)


def test_gather_evidence_sentences_drops_a_fallback_sentence_containing_a_banned_phrase(monkeypatch):
    """Same guarantee for the raw source_facts fallback path (no bank at
    all) -- a real profile responsibility line could in principle contain
    one of these words too."""
    monkeypatch.setattr(local_tailor, "select_and_edit_bank_bullets", lambda client, job_schema, profile, **_kwargs: ({}, False))
    profile_with_banned_fact = {
        **PROFILE,
        "experience_inventory": [
            {
                "name": "Mavis",
                "role_title": "Automotive Technician",
                "responsibilities": [
                    "Demonstrates strong diagnostic ability with alignment equipment.",  # banned
                    "Repaired vehicle alignment issues for walk-in customers every shift.",  # clean
                ],
            },
        ],
    }
    sentences, _names, _covered = local_tailor._gather_evidence_sentences(
        Mock(), RICH_JOB_SCHEMA, profile_with_banned_fact, limit=4
    )
    assert not any("demonstrate" in s.lower() for s in sentences)
    assert any("repaired vehicle alignment" in s.lower() for s in sentences)


# ── _build_degraded_cover_paragraphs ─────────────────────────────────────


def test_build_paragraphs_returns_exactly_four_single_block_paragraphs():
    reqs = local_tailor._pick_supported_requirements(RICH_JOB_SCHEMA)
    evidence = [
        "I diagnosed and repaired vehicle alignment issues for walk-in customers on every shift I worked.",
        "I explained repair options and real cost tradeoffs directly to customers before any work began.",
    ]
    paragraphs = local_tailor._build_degraded_cover_paragraphs(JOB, RICH_JOB_SCHEMA, reqs, evidence)
    assert len(paragraphs) == 4
    for p in paragraphs:
        assert "\n" not in p


def test_build_paragraphs_never_invents_content_beyond_inputs():
    reqs = local_tailor._pick_supported_requirements(RICH_JOB_SCHEMA)
    evidence = ["I diagnosed and repaired vehicle alignment issues for walk-in customers on every shift I worked."]
    paragraphs = local_tailor._build_degraded_cover_paragraphs(JOB, RICH_JOB_SCHEMA, reqs, evidence)
    hook = paragraphs[0]
    # The requirement text quoted in the hook must come verbatim (modulo
    # trailing period stripping) from the real requirement text supplied.
    assert reqs[0]["requirement"][:30].lower() in hook.lower()


def test_build_paragraphs_uses_job_schemas_precomputed_viewpoint():
    """Real bug caught by inspecting actual output before shipping: an
    earlier version recomputed the viewpoint from `job` directly instead of
    reading job_schema's own already-computed value, silently ignoring it."""
    from applypilot.scoring.schemas import VIEWPOINT_EMPHASIS

    schema = dict(RICH_JOB_SCHEMA, viewpoint="support")
    reqs = local_tailor._pick_supported_requirements(schema)
    paragraphs = local_tailor._build_degraded_cover_paragraphs(JOB, schema, reqs, [])
    assert VIEWPOINT_EMPHASIS["support"] in paragraphs[2]
    assert VIEWPOINT_EMPHASIS["general"] not in paragraphs[2]


def test_build_paragraphs_does_not_repeat_the_same_sentence_across_paragraphs():
    """Real bug caught by inspecting actual output before shipping: an
    earlier version quoted evidence_sentences[0] verbatim in BOTH the hook
    and the opening of the evidence paragraph."""
    reqs = local_tailor._pick_supported_requirements(RICH_JOB_SCHEMA)
    evidence = ["I diagnosed and repaired vehicle alignment issues for walk-in customers on every shift I worked."]
    hook, evidence_para, _fit, _close = local_tailor._build_degraded_cover_paragraphs(
        JOB, RICH_JOB_SCHEMA, reqs, evidence
    )
    assert evidence[0] not in hook
    assert evidence[0] in evidence_para


def test_build_paragraphs_handles_zero_requirements_and_zero_evidence_without_crashing():
    paragraphs = local_tailor._build_degraded_cover_paragraphs(JOB, RICH_JOB_SCHEMA, [], [])
    assert len(paragraphs) == 4
    assert all(p.strip() for p in paragraphs)


# 2026-09-13: a job with only 1 supported requirement -- traced (per the
# user's direct request) to be the typical real case for this candidate's
# actual job mix, and previously left FIT with zero per-job content since
# req_texts[2:5] is empty whenever there's only 1-2 supported requirements.
THIN_JOB_SCHEMA = {
    "job_url": "https://example.com/job/thin",
    "requirements": [_req("Diagnose and repair customer equipment issues", tier="prototype")],
    "viewpoint": "support",
}

_FOUR_EVIDENCE_SENTENCES = [
    "I diagnosed and repaired vehicle alignment issues for walk-in customers on every shift I worked.",
    "I explained repair options and real cost tradeoffs directly to customers before any work began.",
    "I took orders and resolved customer complaints on my own during the busiest rush periods.",
    "I trained new servers on the register and the floor during their first two weeks on shift.",
]


def test_build_paragraphs_fit_uses_spare_evidence_when_no_leftover_requirements():
    reqs = local_tailor._pick_supported_requirements(THIN_JOB_SCHEMA)
    _hook, evidence_para, fit, _close = local_tailor._build_degraded_cover_paragraphs(
        JOB, THIN_JOB_SCHEMA, reqs, _FOUR_EVIDENCE_SENTENCES
    )
    # EVIDENCE only ever uses the first 3 -- the 4th is genuinely spare,
    # not silently discarded.
    assert _FOUR_EVIDENCE_SENTENCES[3] not in evidence_para
    assert _FOUR_EVIDENCE_SENTENCES[3] in fit


def test_build_paragraphs_fit_stays_generic_when_no_spare_evidence_available():
    """2026-09-13: the FIT opener is now one of several deterministic
    variants (see _FIT_OPENER_VARIANTS), so this no longer asserts one
    variant's exact wording -- only that the paragraph is a real,
    non-empty opener and that nothing spare was fabricated in."""
    reqs = local_tailor._pick_supported_requirements(THIN_JOB_SCHEMA)
    _hook, _evidence_para, fit, _close = local_tailor._build_degraded_cover_paragraphs(
        JOB, THIN_JOB_SCHEMA, reqs, _FOUR_EVIDENCE_SENTENCES[:1]
    )
    assert fit.strip()
    assert "In a similar vein" not in fit  # nothing spare to add -- never fabricated


def test_build_paragraphs_fit_prefers_leftover_requirements_over_spare_evidence():
    """When a job HAS enough supported requirements, FIT must keep using
    requirement text (unchanged behavior) rather than reaching for spare
    evidence sentences even when some exist."""
    reqs = local_tailor._pick_supported_requirements(RICH_JOB_SCHEMA)
    _hook, _evidence_para, fit, _close = local_tailor._build_degraded_cover_paragraphs(
        JOB, RICH_JOB_SCHEMA, reqs, _FOUR_EVIDENCE_SENTENCES
    )
    assert "In a similar vein" not in fit
    assert "The posting also calls out" in fit


# ── compose_degraded_cover_letter (integration) ──────────────────────────


def test_compose_degraded_cover_letter_passes_real_validation(monkeypatch):
    _patch_bank(monkeypatch, fully_covered=True)
    letter, meta = local_tailor.compose_degraded_cover_letter(_stub_client(), JOB, PROFILE, RICH_JOB_SCHEMA)

    assert letter.startswith("Dear Hiring Manager,")
    # 2026-09-13: sign-off uses the FULL name for a formal closing signature
    # (not preferred_name, which is only for resume-header display).
    assert letter.rstrip().endswith("Jordan Alexander Lee")
    assert meta["llm_called"] is False
    assert meta["tier"] == "degraded_template"
    assert meta["bank_covered"] is True

    validation = validate_cover_letter(letter, PROFILE)
    assert validation["passed"], validation["errors"]


def test_compose_degraded_cover_letter_evidence_gathering_itself_makes_no_calls(monkeypatch):
    """2026-09-13: previously this asserted client.chat() was NEVER called
    at all -- no longer true by design, since the filler-polish feature
    deliberately calls it for the hook-closer/close polish (see
    _build_degraded_cover_paragraphs). What's still true, and worth
    pinning: evidence GATHERING itself (select_and_edit_bank_bullets,
    mocked here) doesn't independently call the client too -- every real
    call comes from the filler-polish layer alone, not duplicated work."""
    _patch_bank(monkeypatch, fully_covered=True)
    client = _stub_client()
    local_tailor.compose_degraded_cover_letter(client, JOB, PROFILE, RICH_JOB_SCHEMA)
    assert client.chat.called  # the filler-polish layer does call it now
    for call in client.chat.call_args_list:
        messages = call.args[0]
        system_prompt = messages[0]["content"]
        assert "cover letter" in system_prompt.lower()  # a filler-polish call, not a bank-editor one


def test_compose_degraded_cover_letter_no_supported_requirements_does_not_crash(monkeypatch):
    monkeypatch.setattr(local_tailor, "select_and_edit_bank_bullets", lambda client, job_schema, profile, **_kwargs: ({}, False))
    thin_schema = {"requirements": [_req("unrelated", supported=False)], "viewpoint": "general"}
    letter, meta = local_tailor.compose_degraded_cover_letter(_stub_client(), JOB, PROFILE, thin_schema)
    assert letter.startswith("Dear Hiring Manager,")
    assert meta["requirements_used"] == 0
    assert meta["evidence_used"] == []


def test_compose_degraded_cover_letter_builds_job_schema_when_not_given(monkeypatch):
    monkeypatch.setattr(local_tailor, "select_and_edit_bank_bullets", lambda client, job_schema, profile, **_kwargs: ({}, False))
    called = {}

    def _fake_get_or_build(job, profile):
        called["yes"] = True
        return {"requirements": [], "viewpoint": "general"}

    monkeypatch.setattr("applypilot.scoring.schemas.get_or_build_job_schema", _fake_get_or_build)
    local_tailor.compose_degraded_cover_letter(_stub_client(), JOB, PROFILE, job_schema=None)
    assert called.get("yes") is True


def test_compose_degraded_cover_letter_filler_polish_attempted_flag(monkeypatch):
    """meta["filler_polish_attempted"] is distinct from llm_called (which
    tracks the heavy full-writer fallback cover letters don't have) -- it
    reflects whether a client was available to attempt the lighter,
    optional filler-polish pass."""
    monkeypatch.setattr(local_tailor, "select_and_edit_bank_bullets", lambda client, job_schema, profile, **_kwargs: ({}, False))
    thin_schema = {"requirements": [], "viewpoint": "general"}
    _letter, meta = local_tailor.compose_degraded_cover_letter(_stub_client(), JOB, PROFILE, thin_schema)
    assert meta["filler_polish_attempted"] is True
    assert meta["llm_called"] is False


# ── cover_letter.py wiring ────────────────────────────────────────────────


def test_generate_cover_letter_redirects_to_degraded_mode_on_runtime_error(monkeypatch):
    from applypilot.scoring import cover_letter as cl

    monkeypatch.setenv("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")

    stub = Mock()
    stub.chat.side_effect = RuntimeError("All models exhausted")
    monkeypatch.setattr(cl, "get_client", lambda quality=False: stub)

    fake_letter = "Dear Hiring Manager,\n\nBody one is here with enough words to count as real.\n\nJordan"
    monkeypatch.setattr(
        "applypilot.scoring.local_tailor.compose_degraded_cover_letter",
        lambda client, job, profile, job_schema: (fake_letter, {"bank_covered": False, "requirements_used": 0, "evidence_used": [], "word_count": 12}),
    )

    letter, validation = cl.generate_cover_letter("RESUME", JOB, PROFILE, max_retries=3)
    assert letter.startswith("Dear Hiring Manager,")
    assert stub.chat.call_count == 1  # never retried the cloud call after exhaustion
    assert not validation["passed"]  # this particular fake letter is too short -- still a real, honest validation result


def test_generate_cover_letter_skips_cloud_entirely_when_no_cloud_available(monkeypatch):
    from applypilot.scoring import cover_letter as cl

    monkeypatch.setenv("APPLYPILOT_LOCAL_LLM_URL", "http://localhost:11434")

    stub = Mock()
    stub.has_cloud_available = Mock(return_value=False)
    monkeypatch.setattr(cl, "get_client", lambda quality=False: stub)

    fake_letter = "Dear Hiring Manager,\n\nBody.\n\nJordan"
    monkeypatch.setattr(
        "applypilot.scoring.local_tailor.compose_degraded_cover_letter",
        lambda client, job, profile, job_schema: (fake_letter, {"bank_covered": False, "requirements_used": 0, "evidence_used": [], "word_count": 3}),
    )

    cl.generate_cover_letter("RESUME", JOB, PROFILE, max_retries=3)
    stub.chat.assert_not_called()  # never attempted the heavy cloud call at all


def test_generate_cover_letter_still_fails_fast_when_local_not_configured(monkeypatch):
    """Existing behavior, unchanged: with no local fallback configured at
    all, cloud exhaustion still fails immediately with no degraded attempt."""
    from applypilot.scoring import cover_letter as cl

    monkeypatch.delenv("APPLYPILOT_LOCAL_LLM_URL", raising=False)

    stub = Mock()
    stub.chat.side_effect = RuntimeError("All models exhausted")
    monkeypatch.setattr(cl, "get_client", lambda quality=False: stub)

    letter, validation = cl.generate_cover_letter("RESUME", JOB, PROFILE, max_retries=3)
    assert letter == ""
    assert not validation["passed"]
    assert "cloud_cover_generation_exhausted" in validation["errors"][0]


# ── validate_cover_letter's education-field check (real bug, fixed) ─────

# Real candidate data ("Bachelor of Arts in Media Studies" / "Media
# Studies") isn't useful for testing the mismatch/anti-fabrication branch
# specifically, since the real official_degree string already contains the
# real field as a substring -- any letter mentioning the degree title
# automatically also mentions the field. A synthetic profile with two
# genuinely disjoint strings isolates that branch instead.
_EDUCATION_PROFILE = dict(
    PROFILE,
    education=[{"official_degree": "Bachelor of Arts in Media Studies", "field_of_study": "Media Studies"}],
)
_DISJOINT_EDUCATION_PROFILE = dict(
    PROFILE,
    education=[{"official_degree": "Some Official Degree Title", "field_of_study": "Underwater Basket Weaving"}],
)


def test_education_field_check_does_not_fire_when_letter_never_mentions_education():
    """The actual bug: this used to fire even though the letter never
    brings up school at all, purely because both profile fields exist."""
    letter = (
        "Dear Hiring Manager,\n\n"
        "This letter is entirely about work experience and never brings up school at all, "
        "on purpose, the same way most real cover letters skip education entirely.\n\n"
        "Jordan"
    )
    result = validate_cover_letter(letter, _EDUCATION_PROFILE)
    assert not any("Official education field" in e for e in result["errors"])


def test_education_field_check_does_not_fire_when_degree_and_field_both_present():
    letter = "Dear Hiring Manager,\n\nI hold a Bachelor of Arts in Media Studies from a real university.\n\nJordan"
    result = validate_cover_letter(letter, _EDUCATION_PROFILE)
    assert not any("Official education field" in e for e in result["errors"])


def test_education_field_check_still_fires_when_degree_named_but_field_absent():
    """The anti-fabrication guard this check exists for still works: naming
    the degree title without ever stating the real field is exactly the
    "changed field" shape it should still catch."""
    letter = (
        "Dear Hiring Manager,\n\n"
        "I hold a degree described in my materials as Some Official Degree Title, though this "
        "letter never mentions what it was actually in anywhere in its own body text.\n\n"
        "Jordan"
    )
    result = validate_cover_letter(letter, _DISJOINT_EDUCATION_PROFILE)
    assert any("Official education field" in e for e in result["errors"])


def test_education_field_check_does_not_fire_when_neither_degree_nor_field_mentioned():
    letter = "Dear Hiring Manager,\n\nThis letter is only about hands-on work experience.\n\nJordan"
    result = validate_cover_letter(letter, _DISJOINT_EDUCATION_PROFILE)
    assert not any("Official education field" in e for e in result["errors"])


def test_real_previously_shipped_letter_no_longer_blocked_by_education_check():
    """Direct regression pin for the real bug: a genuine cover letter that
    never mentions education used to fail validation purely because the
    profile has both education fields populated."""
    real_shape_letter = (
        "Dear Hiring Manager,\n\n"
        "I created a tooling platform that processes documents with review workflows and "
        "machine-learning experimentation to improve accuracy, addressing a problem similar to "
        "what your team is solving. The system handles complex processing pipelines end to end.\n\n"
        "The evidence paragraph here talks entirely about hands-on engineering work, never once "
        "touching on school, transcripts, coursework, or any degree program by name.\n\n"
        "Your team's own posting focuses on developer education and advocacy, and the parallel "
        "to explaining complex technical systems clearly is one I have lived directly.\n\n"
        "I would be glad to walk through any of this in more depth.\n\n"
        "Jordan"
    )
    result = validate_cover_letter(real_shape_letter, _EDUCATION_PROFILE)
    assert not any("Official education field" in e for e in result["errors"])
