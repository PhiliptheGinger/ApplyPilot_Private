"""Tests for the 2026-09-19 deterministic-apply-engine speed work:
- `_FIELD_SELECTORS` / `_fill_known_fields`: the shared per-ATS field map
  (Greenhouse's selectors relocated, Workday's added).
- `_answer_known_screening_questions`: shared qa_knowledge-backed
  screening-question filling, usable by any ATS.
- `_run_workday`: personal-info-page-only Workday flow -- never submits a
  final application, escalates to needs_human on a low fill rate or once
  the page is done, exactly like an unhandled captcha.
- `run_job_deterministic`'s ATS dispatch now accepting "workday".

These use a minimal hand-rolled Playwright-locator fake rather than the
real library, since the goal is to pin this module's OWN control flow
(which selector list gets tried, what triggers needs_human, the fill-rate
gate), not Playwright's selector engine itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.apply import engine_deterministic as ed


class _FakeLocator:
    def __init__(self, count: int = 0, text: str = "", attrs: dict | None = None, tag: str = "input"):
        self._count = count
        self._text = text
        self._attrs = attrs or {}
        self._tag = tag
        self.fill_calls: list[str] = []
        self.click_calls = 0
        self.select_calls: list[str] = []

    def count(self) -> int:
        return self._count

    @property
    def first(self):
        return self

    def fill(self, value, timeout=None):
        self.fill_calls.append(value)

    def click(self, timeout=None):
        self.click_calls += 1

    def set_input_files(self, path, timeout=None):
        self.fill_calls.append(path)

    def inner_text(self, timeout=None):
        return self._text

    def get_attribute(self, name, timeout=None):
        return self._attrs.get(name)

    def evaluate(self, script, timeout=None):
        return self._tag

    def select_option(self, label=None, timeout=None):
        self.select_calls.append(label)

    def locator(self, selector):
        return _FakeLocator(count=0)


class _FakePage:
    """Selector-string-keyed fake: `locator_map` is exactly what the code
    under test is expected to ask for."""

    def __init__(self, locator_map: dict[str, _FakeLocator] | None = None, url: str = "https://example.com/apply"):
        self._map = locator_map or {}
        self.url = url

    def locator(self, selector: str):
        return self._map.get(selector, _FakeLocator(count=0))

    def wait_for_timeout(self, ms):
        pass

    def goto(self, url, wait_until=None, timeout=None):
        pass


class TestFillKnownFields:
    def test_fills_every_matched_field_and_counts_known(self, monkeypatch):
        page = _FakePage(
            {
                "input[data-automation-id='legalNameSection_firstName']": _FakeLocator(count=1),
                "input[data-automation-id='email']": _FakeLocator(count=1),
            }
        )
        fields = {"first_name": "Philip", "last_name": "", "email": "p@example.com", "city": ""}

        filled, known = ed._fill_known_fields(page, "workday", fields)

        # first_name and email have both a value AND a matching selector -> known=2, filled=2.
        # last_name/city have no value -> never counted against the rate.
        assert known == 2
        assert filled == 2

    def test_does_not_count_fields_with_no_profile_value(self):
        page = _FakePage({})
        fields = {"first_name": "", "email": ""}

        filled, known = ed._fill_known_fields(page, "workday", fields)

        assert (filled, known) == (0, 0)

    def test_unmatched_selector_counts_as_known_but_not_filled(self):
        page = _FakePage({})  # nothing matches any selector
        fields = {"first_name": "Philip"}

        filled, known = ed._fill_known_fields(page, "workday", fields)

        assert known == 1
        assert filled == 0

    def test_generic_autocomplete_fallback_catches_field_vendor_selectors_miss(self):
        """Vendor selector list is present but doesn't match this page's real
        DOM (e.g. drift) -- the generic autocomplete-based fallback should
        still catch it, scoped 2026-09-25 per the apply-page-schema work."""
        page = _FakePage({"input[autocomplete='email']": _FakeLocator(count=1)})
        fields = {"email": "p@example.com"}

        filled, known = ed._fill_known_fields(page, "workday", fields)

        assert known == 1
        assert filled == 1

    def test_generic_fallback_works_for_an_ats_with_no_vendor_entry_at_all(self):
        """A field with no vendor-specific selectors at all for this ats_slug
        (or an unknown ats_slug) can still be filled via the generic map."""
        page = _FakePage({"input[autocomplete='tel']": _FakeLocator(count=1)})
        fields = {"phone": "555-0100"}

        filled, known = ed._fill_known_fields(page, "some_unlisted_ats", fields)

        assert known == 1
        assert filled == 1

    def test_vendor_selector_is_tried_before_generic_fallback(self):
        """When BOTH the vendor selector and the generic fallback would match,
        the vendor-specific one (more precise, hand-verified) should be used."""
        vendor_locator = _FakeLocator(count=1)
        page = _FakePage(
            {
                "input[data-automation-id='email']": vendor_locator,
                "input[type='email']": vendor_locator,
                "input[autocomplete='email']": _FakeLocator(count=1),
            }
        )
        fields = {"email": "p@example.com"}

        filled, known = ed._fill_known_fields(page, "workday", fields)

        assert known == 1
        assert filled == 1
        assert vendor_locator.fill_calls == ["p@example.com"]


class TestAnswerKnownScreeningQuestions:
    def test_fills_a_text_answer_found_in_qa_knowledge(self, monkeypatch):
        input_locator = _FakeLocator(count=1, tag="input")
        label = _FakeLocator(count=1, text="Are you legally authorized to work in the US?", attrs={"for": "auth"})

        class _Labels:
            def count(self):
                return 1

            def nth(self, i):
                return label

        page = MagicMock()
        page.locator.side_effect = lambda sel: _Labels() if sel == ed._QUESTION_LABEL_SELECTORS else input_locator

        monkeypatch.setattr(ed, "get_qa", lambda q, doc_format=None: "Yes" if "authorized" in q else None)

        answered = ed._answer_known_screening_questions(page)

        assert answered == 1
        assert input_locator.fill_calls == ["Yes"]

    def test_skips_questions_with_no_known_answer(self, monkeypatch):
        label = _FakeLocator(count=1, text="Describe a challenge you overcame", attrs={"for": "essay"})

        class _Labels:
            def count(self):
                return 1

            def nth(self, i):
                return label

        page = MagicMock()
        page.locator.side_effect = lambda sel: _Labels() if sel == ed._QUESTION_LABEL_SELECTORS else _FakeLocator(count=1)
        monkeypatch.setattr(ed, "get_qa", lambda q, doc_format=None: None)

        answered = ed._answer_known_screening_questions(page)

        assert answered == 0


class TestRunWorkday:
    def _job(self):
        return {"tailored_resume_path": None}

    def test_escalates_on_low_fill_rate_without_clicking_next(self, monkeypatch):
        monkeypatch.setattr(ed, "_fill_known_fields", lambda page, ats, fields: (1, 5))  # 20% fill rate
        monkeypatch.setattr(ed, "_upload_resume_if_present", lambda *a, **k: False)
        monkeypatch.setattr(ed, "_answer_known_screening_questions", lambda *a, **k: 0)
        monkeypatch.setattr(ed, "_has_captcha", lambda page: False)
        page = _FakePage(url="https://acme.wd1.myworkdayjobs.com/job/1")

        result, _dur, _steps = ed._run_workday(page, self._job(), dry_run=False)

        assert result.startswith("needs_human:workday_fields_unmatched:")

    def test_clicks_next_then_hands_off_to_needs_human_on_good_fill_rate(self, monkeypatch):
        monkeypatch.setattr(ed, "_fill_known_fields", lambda page, ats, fields: (4, 5))  # 80% fill rate
        monkeypatch.setattr(ed, "_upload_resume_if_present", lambda *a, **k: True)
        monkeypatch.setattr(ed, "_answer_known_screening_questions", lambda *a, **k: 0)
        monkeypatch.setattr(ed, "_has_captcha", lambda page: False)
        next_btn = _FakeLocator(count=1)
        page = _FakePage(
            {"button[data-automation-id='bottom-navigation-next-button']": next_btn},
            url="https://acme.wd1.myworkdayjobs.com/job/1",
        )

        result, _dur, _steps = ed._run_workday(page, self._job(), dry_run=False)

        assert next_btn.click_calls == 1
        assert result.startswith("needs_human:workday_wizard_incomplete:")

    def test_never_returns_applied(self, monkeypatch):
        """Workday's real submit step is on a page this function never
        reaches -- it must never claim success."""
        monkeypatch.setattr(ed, "_fill_known_fields", lambda page, ats, fields: (5, 5))
        monkeypatch.setattr(ed, "_upload_resume_if_present", lambda *a, **k: True)
        monkeypatch.setattr(ed, "_answer_known_screening_questions", lambda *a, **k: 0)
        monkeypatch.setattr(ed, "_has_captcha", lambda page: False)
        page = _FakePage(url="https://acme.wd1.myworkdayjobs.com/job/1")

        result, _dur, _steps = ed._run_workday(page, self._job(), dry_run=True)

        assert result != "applied"
        assert "needs_human" in result

    def test_escalates_immediately_on_captcha(self, monkeypatch):
        monkeypatch.setattr(ed, "_fill_known_fields", lambda page, ats, fields: (5, 5))
        monkeypatch.setattr(ed, "_upload_resume_if_present", lambda *a, **k: True)
        monkeypatch.setattr(ed, "_answer_known_screening_questions", lambda *a, **k: 0)
        monkeypatch.setattr(ed, "_has_captcha", lambda page: True)
        page = _FakePage(url="https://acme.wd1.myworkdayjobs.com/job/1")

        result, _dur, _steps = ed._run_workday(page, self._job(), dry_run=False)

        assert result.startswith("needs_human:captcha:")


class TestRunGreenhouseRegression:
    """The 2026-09-19 refactor moved Greenhouse's selectors into
    `_FIELD_SELECTORS` and routed them through `_fill_known_fields` --
    this pins that the observable behavior (fill, upload, submit, return
    "applied") is unchanged."""

    def test_fills_via_shared_map_and_submits(self, monkeypatch):
        submit_btn = _FakeLocator(count=1)
        first_name = _FakeLocator(count=1)
        page = _FakePage(
            {
                "input[name='first_name']": first_name,
                "button[type='submit']": submit_btn,
            }
        )
        monkeypatch.setattr(ed, "_get_profile_fields", lambda: {"first_name": "Philip", "last_name": ""})
        monkeypatch.setattr(ed, "_upload_resume_if_present", lambda *a, **k: True)
        monkeypatch.setattr(ed, "_answer_known_screening_questions", lambda *a, **k: 0)
        monkeypatch.setattr(ed, "_has_captcha", lambda page: False)

        result, _dur, _steps = ed._run_greenhouse(page, {}, dry_run=False)

        assert first_name.fill_calls == ["Philip"]
        assert submit_btn.click_calls == 1
        assert result == "applied"

    def test_dry_run_skips_submit(self, monkeypatch):
        page = _FakePage({})
        monkeypatch.setattr(ed, "_get_profile_fields", lambda: {"first_name": "Philip"})
        monkeypatch.setattr(ed, "_upload_resume_if_present", lambda *a, **k: True)
        monkeypatch.setattr(ed, "_answer_known_screening_questions", lambda *a, **k: 0)
        monkeypatch.setattr(ed, "_has_captcha", lambda page: False)

        result, _dur, _steps = ed._run_greenhouse(page, {}, dry_run=True)

        assert result == "applied"


class TestRunJobDeterministicDispatch:
    def test_unsupported_ats_returns_needs_human_without_touching_playwright(self, monkeypatch):
        monkeypatch.setattr(ed, "detect_ats", lambda url: "ashby")

        result, dur, steps = ed.run_job_deterministic(
            {"application_url": "https://jobs.ashbyhq.com/acme/1"}, port=9222
        )

        assert result.startswith("needs_human:unsupported_ats:")
        assert dur == 0

    def test_workday_ats_dispatches_to_run_workday(self, monkeypatch):
        monkeypatch.setattr(ed, "detect_ats", lambda url: "workday")
        monkeypatch.setattr(ed, "_run_workday", lambda page, job, dry_run: ("needs_human:workday_wizard_incomplete:x", 100, []))

        fake_page = _FakePage()
        fake_context = MagicMock(pages=[fake_page])
        fake_browser = MagicMock(contexts=[fake_context])
        fake_pw = MagicMock()
        fake_pw.chromium.connect_over_cdp.return_value = fake_browser

        monkeypatch.setattr(ed, "sync_playwright", lambda: MagicMock(__enter__=lambda s: fake_pw, __exit__=lambda *a: None))
        monkeypatch.setattr(ed, "_has_captcha", lambda page: False)

        result, _dur, _steps = ed.run_job_deterministic(
            {"application_url": "https://acme.wd1.myworkdayjobs.com/job/1"}, port=9222
        )

        assert result == "needs_human:workday_wizard_incomplete:x"
