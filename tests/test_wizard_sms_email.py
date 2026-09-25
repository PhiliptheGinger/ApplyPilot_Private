"""Tests for the wizard's SMS-relay and email-verification steps (decision #197 follow-up).

Mocks rich.prompt.Confirm/Prompt (no real terminal input) and the external
Gmail/Twilio clients (no real network calls) -- mirrors this codebase's
existing convention of mocking at the client-module boundary.
"""

from unittest.mock import AsyncMock, patch

import pytest

from applypilot.wizard import init as wizard_init


@pytest.fixture(autouse=True)
def _isolated_env_path(tmp_path, monkeypatch):
    """Point ENV_PATH at a throwaway file so tests never touch the real .env."""
    fake_env = tmp_path / ".env"
    monkeypatch.setattr(wizard_init, "ENV_PATH", fake_env)
    return fake_env


class TestSetupSmsRelay:
    def test_skips_cleanly_when_declined(self, capsys):
        with patch("applypilot.wizard.init.Confirm.ask", return_value=False):
            wizard_init._setup_sms_relay({"personal": {"phone": "+15555550111"}})
        out = capsys.readouterr().out
        assert "Skipped" in out

    def test_writes_env_and_verifies_connection(self, _isolated_env_path):
        answers = iter([False])  # Confirm.ask for the test-message send, declined
        with (
            patch("applypilot.wizard.init.Confirm.ask", side_effect=[True, False]),
            patch(
                "applypilot.wizard.init.Prompt.ask",
                side_effect=["ACfake", "tokenfake", "+15555550100"],
            ),
            patch("applypilot.tracking.sms_client.verify_connection", return_value=True),
        ):
            wizard_init._setup_sms_relay({"personal": {"phone": "+15555550111"}})

        content = _isolated_env_path.read_text()
        assert "TWILIO_ACCOUNT_SID=ACfake" in content
        assert "TWILIO_AUTH_TOKEN=tokenfake" in content
        assert "TWILIO_PHONE_NUMBER=+15555550100" in content

    def test_reports_failure_when_connection_check_fails(self, capsys):
        with (
            patch("applypilot.wizard.init.Confirm.ask", return_value=True),
            patch(
                "applypilot.wizard.init.Prompt.ask",
                side_effect=["ACfake", "tokenfake", "+15555550100"],
            ),
            patch("applypilot.tracking.sms_client.verify_connection", return_value=False),
        ):
            wizard_init._setup_sms_relay({"personal": {"phone": "+15555550111"}})
        out = capsys.readouterr().out
        assert "Connection failed" in out

    def test_sends_and_confirms_test_message_end_to_end(self, capsys):
        with (
            patch("applypilot.wizard.init.Confirm.ask", side_effect=[True, True, True]),
            patch(
                "applypilot.wizard.init.Prompt.ask",
                side_effect=["ACfake", "tokenfake", "+15555550100"],
            ),
            patch("applypilot.tracking.sms_client.verify_connection", return_value=True),
            patch("applypilot.tracking.sms_client.send_test_message", return_value=(True, "Sent.")),
        ):
            wizard_init._setup_sms_relay({"personal": {"phone": "+15555550111"}})
        out = capsys.readouterr().out
        assert "confirmed working end-to-end" in out

    def test_skips_test_message_when_no_personal_phone_on_file(self, capsys):
        with (
            patch("applypilot.wizard.init.Confirm.ask", return_value=True),
            patch(
                "applypilot.wizard.init.Prompt.ask",
                side_effect=["ACfake", "tokenfake", "+15555550100"],
            ),
            patch("applypilot.tracking.sms_client.verify_connection", return_value=True),
        ):
            wizard_init._setup_sms_relay({"personal": {"phone": ""}})
        out = capsys.readouterr().out
        assert "No personal phone number on file" in out

    def test_does_not_write_real_personal_info_to_env(self, _isolated_env_path):
        """Only the Twilio-specific values go into .env -- never profile.json contents."""
        with (
            patch("applypilot.wizard.init.Confirm.ask", side_effect=[True, False]),
            patch(
                "applypilot.wizard.init.Prompt.ask",
                side_effect=["ACfake", "tokenfake", "+15555550100"],
            ),
            patch("applypilot.tracking.sms_client.verify_connection", return_value=True),
        ):
            wizard_init._setup_sms_relay({"personal": {"phone": "+15555550111", "full_name": "Real Name"}})
        content = _isolated_env_path.read_text()
        assert "Real Name" not in content
        assert "+15555550111" not in content


class TestVerifyEmail:
    def test_skips_when_no_email(self, capsys):
        wizard_init._verify_email({"personal": {"email": ""}})
        # No prompts should fire; nothing to assert beyond "doesn't crash".

    def test_skips_gracefully_when_gmail_not_configured(self, capsys):
        with patch(
            "applypilot.tracking.gmail_client.check_gmail_setup",
            return_value=(False, "not set up"),
        ):
            wizard_init._verify_email({"personal": {"email": "candidate@example.com"}})
        out = capsys.readouterr().out
        assert "Gmail integration isn't set up" in out

    def test_sends_and_confirms(self, capsys):
        with (
            patch("applypilot.tracking.gmail_client.check_gmail_setup", return_value=(True, "ok")),
            patch("applypilot.wizard.init.Confirm.ask", side_effect=[True, True]),
            patch("applypilot.tracking.gmail_client.send_email", new_callable=AsyncMock, return_value=(True, "ok")),
        ):
            wizard_init._verify_email({"personal": {"email": "candidate@example.com"}})
        out = capsys.readouterr().out
        assert "Email confirmed" in out

    def test_declining_to_send_skips_cleanly(self, capsys):
        with (
            patch("applypilot.tracking.gmail_client.check_gmail_setup", return_value=(True, "ok")),
            patch("applypilot.wizard.init.Confirm.ask", return_value=False),
        ):
            wizard_init._verify_email({"personal": {"email": "candidate@example.com"}})
        # Declining the send prompt should short-circuit before any send_email call.
