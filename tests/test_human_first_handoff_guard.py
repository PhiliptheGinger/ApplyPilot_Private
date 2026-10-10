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

from applypilot.apply.human_review import (
    _build_human_first_banner_js,
    _human_first_blocked_domains,
    _inject_human_first_banner,
)


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


class TestReadableHandoffConfirmation:
    """FW48 (2026-10-07): the Hand Off confirmation leads with the site name,
    not the raw URL. Verified once in headless Chromium against a long ADP
    URL (shows 'workforcenow.adp.com', full link folded under a toggle)."""

    def _js(self):
        from applypilot.apply.human_review import _build_human_first_banner_js

        return _build_human_first_banner_js("abc123", "Help Desk", "Acme", 8800)

    def test_confirmation_uses_site_name_builder(self):
        js = self._js()
        assert "_showConfirmPanel(\n        _handoffConfirmContent()," in js
        assert "Show full link" in js
        assert "replace(/^www\\./, '')" in js
        # The old raw-URL-first message is gone.
        assert "Hand off automation on this page? ' + window.location.href" not in js

    def test_generated_js_parses(self, tmp_path):
        import shutil
        import subprocess

        import pytest

        node = shutil.which("node")
        if not node:
            pytest.skip("node not installed")
        path = tmp_path / "banner.js"
        path.write_text(self._js(), encoding="utf-8")
        result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


class TestBannerProgressAndGuidance:
    def test_handoff_starts_progress_display(self):
        """FW49 (2026-10-07): verified once in headless Chromium -- button reads
        'Automation working 0:22' and the subline shows the worker's status."""
        from applypilot.apply.human_review import _build_human_first_banner_js

        js = _build_human_first_banner_js("abc123", "Help Desk", "Acme", 8800)
        assert "_startHandoffProgress();" in js
        assert "'Automation working ' + clock" in js
        assert "/api/status" in js

    def test_pause_banner_says_what_to_do_next(self):
        """FW45 (2026-10-07)."""
        from applypilot.apply.human_review import _build_banner_js

        js = _build_banner_js("abc123", "Help Desk", "Acme", 9, "Solve the CAPTCHA", 7373)
        assert "Do the step in this tab, then click Done." in js

    def test_pause_banner_has_flag_and_bug_buttons(self):
        """FW47: pipeline-logic flag icon + tool bug-report icon, both wired."""
        from applypilot.apply.human_review import _build_banner_js

        js = _build_banner_js(
            "abc123", "Help Desk", "Acme", 9, "Solve the CAPTCHA", 7373, job_url="https://example.com/job/1"
        )
        assert "topRow.appendChild(btnFlag)" in js
        assert "topRow.appendChild(btnBug)" in js
        assert "/api/flag-job" in js
        assert "/api/bug-report" in js
        assert "JOB_URL = 'https://example.com/job/1'" in js

    def test_pause_banner_js_parses(self, tmp_path):
        import shutil
        import subprocess

        import pytest

        from applypilot.apply.human_review import _build_banner_js

        node = shutil.which("node")
        if not node:
            pytest.skip("node not installed")
        path = tmp_path / "pause.js"
        path.write_text(_build_banner_js("abc123", "Help Desk", "Acme", 9, "Solve it", 7373), encoding="utf-8")
        result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


class TestHumanFirstDocumentLinks:
    """Real gap found live (2026-10-06): doing a manual LinkedIn Easy Apply
    left no way to grab the job's own already-tailored resume/cover letter
    to upload -- the banner only ever offered Apply/Hand Off. Fixed by
    serving the files from the same per-worker listener the banner already
    talks to (launcher._handle_human_first_document) and only showing the
    link when the job actually has that document.
    """

    def test_banner_js_omits_links_by_default(self):
        js = _build_human_first_banner_js("hash1", "Some Job", "acme", 7380)

        assert "HAS_RESUME = false" in js
        assert "HAS_COVER = false" in js

    def test_banner_js_includes_links_when_requested(self):
        js = _build_human_first_banner_js("hash1", "Some Job", "acme", 7380, has_resume=True, has_cover=True)

        assert "HAS_RESUME = true" in js
        assert "HAS_COVER = true" in js
        assert "_makeDocLink('&#128196; Resume', 'resume')" in js
        assert "_makeDocLink('&#128196; Cover Letter', 'cover-letter')" in js

    def test_inject_derives_has_resume_has_cover_from_job_dict(self, monkeypatch):
        """The actual Python-side wiring: whether the banner offers a
        document link must follow the real job row, not be hardcoded."""
        import applypilot.apply.human_review as human_review

        captured = {}

        def _fake_build(hash_, title, company, server_port, *, has_resume=False, has_cover=False, job_url=""):
            captured["has_resume"] = has_resume
            captured["has_cover"] = has_cover
            return "// fake js"

        monkeypatch.setattr(human_review, "_build_human_first_banner_js", _fake_build)
        monkeypatch.setattr(
            human_review.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"returncode": 0, "stderr": ""})(),
        )

        job_with_both = {"url": "https://www.linkedin.com/jobs/view/1", "tailored_resume_path": "r.pdf", "cover_letter_path": "c.pdf"}
        _inject_human_first_banner(9222, job_with_both, server_port=7380)
        assert captured == {"has_resume": True, "has_cover": True}

        job_with_neither = {"url": "https://www.linkedin.com/jobs/view/2"}
        _inject_human_first_banner(9222, job_with_neither, server_port=7380)
        assert captured == {"has_resume": False, "has_cover": False}
