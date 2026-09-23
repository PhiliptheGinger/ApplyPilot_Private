"""Tests for the human-first Hand Off button's bot-wall guard (CLAUDE.md
decision #190 follow-up).

Real, live-reported incident (2026-09-23): the user accidentally clicked
"Hand Off" on the LinkedIn page itself, before being redirected to the
company's own ATS. Had that click's captured URL actually been a
linkedin.com page (it happened to already be off LinkedIn that time), it
would have handed the automation a LinkedIn URL as the application_url --
scripted interaction with a known bot-detection-sensitive site, exactly
the exposure decision #168 already avoids everywhere else. The user then
asked for this to generalize beyond just linkedin.com to "ANY botting
wall," not a single hardcoded domain.

These tests cover `_human_first_blocked_domains` (the Python-side list
embedded into the banner JS at build time) directly -- the in-page JS
guard itself (alert/confirm dialogs) isn't exercised here since it needs
a real browser; `test_hitl_stdin_eof_fallback.py`'s node --check-based
verification is the precedent for validating generated JS syntax
separately, done manually during development rather than in this suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply.human_review import _build_human_first_banner_js, _human_first_blocked_domains


class TestHumanFirstBlockedDomains:
    def test_linkedin_always_included(self):
        assert "linkedin.com" in _human_first_blocked_domains()

    def test_confirmed_403_sources_included(self):
        """indeed.com/ziprecruiter.com/glassdoor.com aren't a guess -- all
        three returned real HTTP 403s from JobSpy's own discovery scraper
        the same day this guard was added."""
        domains = _human_first_blocked_domains()
        assert "indeed.com" in domains
        assert "ziprecruiter.com" in domains
        assert "glassdoor.com" in domains

    def test_no_duplicates(self):
        domains = _human_first_blocked_domains()
        assert len(domains) == len(set(domains))

    def test_config_load_failure_does_not_lose_the_hardcoded_list(self, monkeypatch):
        """If sites.yaml can't be read for any reason, the guard must
        degrade to its hardcoded safety net, not silently disable itself
        by propagating the exception."""
        import applypilot.config as config_mod

        def _raise(*a, **k):
            raise RuntimeError("sites.yaml unreadable")

        monkeypatch.setattr(config_mod, "load_blocked_sites", _raise)

        domains = _human_first_blocked_domains()

        assert "linkedin.com" in domains
        assert "indeed.com" in domains

    def test_banner_js_embeds_the_blocked_domains_list(self):
        js = _build_human_first_banner_js("hash1", "Some Job", "acme", 7380)

        assert "BLOCKED_DOMAINS" in js
        assert "linkedin.com" in js
        assert "_onBlockedDomain" in js
