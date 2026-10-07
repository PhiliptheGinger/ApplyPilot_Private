"""FW53 (2026-10-07): email notifications for needs_human pauses and Claude exhaustion."""

import threading

import pytest

from applypilot import claude_status, notify


@pytest.fixture
def sent(monkeypatch, tmp_path):
    """Capture sends instead of calling Gmail; isolate the rate-limit state file."""
    from applypilot import config

    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    class _Calls(list):
        done = threading.Event()

    calls = _Calls()
    done = calls.done = threading.Event()

    def fake_send(recipient, subject, body):
        calls.append((recipient, subject, body))
        done.set()

    monkeypatch.setattr(notify, "_send", fake_send)
    return calls


class TestRecipient:
    def test_off_by_default(self, monkeypatch, sent):
        monkeypatch.delenv("APPLYPILOT_NOTIFY_EMAIL", raising=False)
        assert notify.resolve_recipient() is None
        assert notify.notify("k", "s", "b", wait=True) is False
        assert sent == []

    @pytest.mark.parametrize("value", ["0", "off", "false", ""])
    def test_disable_words(self, monkeypatch, value):
        monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", value)
        assert notify.resolve_recipient() is None

    def test_explicit_address(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "me@example.com")
        assert notify.resolve_recipient() == "me@example.com"

    def test_self_uses_profile_email(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "self")
        monkeypatch.setattr("applypilot.config.load_profile", lambda: {"personal": {"email": "cand@example.com"}})
        assert notify.resolve_recipient() == "cand@example.com"

    def test_garbage_value_is_off(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "maybe")
        assert notify.resolve_recipient() is None


class TestRateLimit:
    def test_same_key_suppressed_within_interval(self, monkeypatch, sent):
        monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "me@example.com")
        assert notify.notify("k", "first", "b", wait=True) is True
        assert notify.notify("k", "second", "b", wait=True) is False
        assert [s for _, s, _ in sent] == ["[ApplyPilot] first"]

    def test_different_keys_both_sent(self, monkeypatch, sent):
        monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "me@example.com")
        assert notify.notify("a", "s", "b", wait=True)
        assert notify.notify("b", "s", "b", wait=True)
        assert len(sent) == 2

    def test_suppression_survives_restart(self, monkeypatch, sent):
        monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "me@example.com")
        notify.notify("k", "s", "b", wait=True)
        assert notify._state_path().exists()
        # A fresh process reads the same file.
        assert notify._claim_slot("k", notify.DEFAULT_MIN_INTERVAL_SECONDS, __import__("time").time()) is False

    def test_zero_interval_always_sends(self, monkeypatch, sent):
        monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "me@example.com")
        assert notify.notify("k", "s", "b", min_interval_seconds=0, wait=True)
        assert notify.notify("k", "s", "b", min_interval_seconds=0, wait=True)


def test_default_send_is_non_blocking(monkeypatch, sent):
    monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "me@example.com")
    assert notify.notify("k", "s", "b") is True
    assert sent.done.wait(5)


def test_send_failure_never_raises(monkeypatch, tmp_path):
    from applypilot import config

    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "me@example.com")
    monkeypatch.setattr("applypilot.tracking.gmail_client.check_gmail_setup", lambda: (True, ""))

    async def boom(*a, **k):
        raise RuntimeError("gmail down")

    monkeypatch.setattr("applypilot.tracking.gmail_client.send_email", boom)
    assert notify.notify("k", "s", "b", wait=True) is True  # logged, not raised


def test_needs_human_email_names_job_and_next_step(monkeypatch, sent):
    monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "me@example.com")
    job = {"title": "Help Desk Analyst", "company": "Acme", "url": "https://example.com/j/1"}
    assert notify.notify_needs_human(job, "captcha", "https://ats.example.com/apply", "Solve the CAPTCHA")
    _, subject, body = sent[0]
    assert "Help Desk Analyst" in subject
    assert "captcha" in body and "Solve the CAPTCHA" in body and "click Done" in body
    # Same job pausing again right away doesn't email twice.
    assert notify.notify_needs_human(job, "captcha", "https://ats.example.com/apply") is False


class TestClaudeStatusTransitions:
    @pytest.fixture(autouse=True)
    def _clean(self):
        claude_status.reset_apply_signal_for_tests()
        yield
        claude_status.reset_apply_signal_for_tests()

    def test_emails_once_on_exhaustion_and_once_on_recovery(self, monkeypatch, sent):
        monkeypatch.setenv("APPLYPILOT_NOTIFY_EMAIL", "me@example.com")
        monkeypatch.setattr(notify, "notify", lambda key, subject, body, **kw: sent.append((key, subject, body)) or True)

        claude_status.record_apply_exhaustion("session_limit")
        claude_status.record_apply_exhaustion("session_limit")  # still exhausted: no new email
        claude_status.record_apply_success()
        claude_status.record_apply_success()  # already fine: no email

        assert [k for k, _, _ in sent] == ["claude_exhausted", "claude_recovered"]

    def test_notification_error_does_not_break_the_signal(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("nope")

        monkeypatch.setattr(notify, "notify_claude_exhausted", boom)
        claude_status.record_apply_exhaustion("session_limit")
        assert claude_status.is_exhausted() == (True, "session_limit")
