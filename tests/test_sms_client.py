"""Tests for the Twilio SMS relay client (decision #197).

Mirrors the httpx-mocking style already used in test_apply_precheck_expired.py
(unittest.mock.patch("httpx.get", ...)) rather than a network-hitting client.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx
import pytest

from applypilot.tracking import sms_client


def _twilio_date(dt: datetime) -> str:
    """Format a datetime the way Twilio's API returns date_sent."""
    return dt.strftime("%a, %d %b %Y %H:%M:%S %z")


def _fake_response(status_code: int, json_data: dict | None = None, text: str = ""):
    resp = httpx.Response(
        status_code=status_code,
        json=json_data,
        text=text if json_data is None else None,
        request=httpx.Request("GET", "https://api.twilio.com/fake"),
    )
    return resp


@pytest.fixture(autouse=True)
def _twilio_env(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACfake")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tokenfake")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15555550100")


class TestCheckSmsSetup:
    def test_ok_when_all_three_env_vars_present(self):
        ok, msg = sms_client.check_sms_setup()
        assert ok is True
        assert "found" in msg.lower()

    def test_fails_with_setup_instructions_when_missing(self, monkeypatch):
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        ok, msg = sms_client.check_sms_setup()
        assert ok is False
        assert "twilio.com" in msg
        assert "TWILIO_ACCOUNT_SID" in msg

    def test_fails_if_only_partial_credentials_present(self, monkeypatch):
        monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)
        ok, _msg = sms_client.check_sms_setup()
        assert ok is False


class TestVerifyConnection:
    def test_true_on_200(self):
        with patch("httpx.get", return_value=_fake_response(200, json_data={"sid": "ACfake"})):
            assert sms_client.verify_connection() is True

    def test_false_on_auth_failure(self):
        with patch("httpx.get", return_value=_fake_response(401, text="Authentication Error")):
            assert sms_client.verify_connection() is False

    def test_false_on_network_error(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
            assert sms_client.verify_connection() is False

    def test_false_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        assert sms_client.verify_connection() is False


class TestListRecentMessages:
    def test_returns_messages_within_window(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=2)
        payload = {
            "messages": [
                {"sid": "SM1", "from": "+15555550199", "body": "Your code is 481920", "date_sent": _twilio_date(recent)}
            ]
        }
        with patch("httpx.get", return_value=_fake_response(200, json_data=payload)):
            result = sms_client.list_recent_messages(since_minutes=10)
        assert len(result) == 1
        assert result[0]["body"] == "Your code is 481920"

    def test_excludes_messages_outside_window(self):
        old = datetime.now(timezone.utc) - timedelta(minutes=30)
        payload = {
            "messages": [
                {"sid": "SM1", "from": "+15555550199", "body": "Old code 111111", "date_sent": _twilio_date(old)}
            ]
        }
        with patch("httpx.get", return_value=_fake_response(200, json_data=payload)):
            result = sms_client.list_recent_messages(since_minutes=10)
        assert result == []

    def test_returns_empty_list_on_http_error(self):
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            assert sms_client.list_recent_messages() == []

    def test_returns_empty_list_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        assert sms_client.list_recent_messages() == []


class TestGetLatestVerificationCode:
    def test_extracts_first_numeric_run(self):
        with patch(
            "applypilot.tracking.sms_client.list_recent_messages",
            return_value=[{"sid": "SM1", "from_": "x", "body": "Your Lumen code is 296312", "date_sent": "now"}],
        ):
            assert sms_client.get_latest_verification_code() == "296312"

    def test_returns_none_when_no_digits_present(self):
        with patch(
            "applypilot.tracking.sms_client.list_recent_messages",
            return_value=[{"sid": "SM1", "from_": "x", "body": "Thanks for signing up!", "date_sent": "now"}],
        ):
            assert sms_client.get_latest_verification_code() is None

    def test_returns_none_when_no_messages(self):
        with patch("applypilot.tracking.sms_client.list_recent_messages", return_value=[]):
            assert sms_client.get_latest_verification_code() is None


class TestSendTestMessage:
    def test_ok_on_201(self):
        with patch("httpx.post", return_value=_fake_response(201, json_data={"sid": "SM1"})):
            ok, detail = sms_client.send_test_message("+15555550199")
        assert ok is True
        assert detail == "Sent."

    def test_fails_with_detail_on_error_status(self):
        with patch("httpx.post", return_value=_fake_response(400, text="Invalid 'To' Phone Number")):
            ok, detail = sms_client.send_test_message("+1invalid")
        assert ok is False
        assert "400" in detail

    def test_fails_on_network_error(self):
        with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
            ok, detail = sms_client.send_test_message("+15555550199")
        assert ok is False

    def test_fails_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        ok, detail = sms_client.send_test_message("+15555550199")
        assert ok is False
        assert "not configured" in detail
