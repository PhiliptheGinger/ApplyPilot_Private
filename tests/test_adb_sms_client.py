"""Tests for the ADB direct-from-phone SMS relay client (decision #200),
generalized from the prior Haywood personal project's content://sms technique.

Mocks subprocess.run and shutil.which -- no real adb binary or connected
device required.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from applypilot.tracking import adb_sms_client


def _fake_run(returncode=0, stdout="", stderr=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class TestFindAdb:
    def test_prefers_env_override_when_it_exists(self, monkeypatch, tmp_path):
        fake_adb = tmp_path / "adb.exe"
        fake_adb.write_text("")
        monkeypatch.setenv("APPLYPILOT_ADB_PATH", str(fake_adb))
        assert adb_sms_client._find_adb() == str(fake_adb)

    def test_falls_back_to_path_when_override_missing(self, monkeypatch):
        monkeypatch.delenv("APPLYPILOT_ADB_PATH", raising=False)
        with patch("shutil.which", return_value="/usr/bin/adb"):
            assert adb_sms_client._find_adb() == "/usr/bin/adb"

    def test_ignores_override_pointing_at_nonexistent_file(self, monkeypatch):
        monkeypatch.setenv("APPLYPILOT_ADB_PATH", "C:/nonexistent/adb.exe")
        with patch("shutil.which", return_value=None):
            assert adb_sms_client._find_adb() is None


class TestCheckAdbSetup:
    def test_fails_with_instructions_when_adb_not_found(self):
        with patch("applypilot.tracking.adb_sms_client._find_adb", return_value=None):
            ok, msg = adb_sms_client.check_adb_setup()
        assert ok is False
        assert "platform-tools" in msg

    def test_ok_when_device_ready(self):
        stdout = "List of devices attached\nABCD1234\tdevice\n"
        with (
            patch("applypilot.tracking.adb_sms_client._find_adb", return_value="adb"),
            patch("subprocess.run", return_value=_fake_run(stdout=stdout)),
        ):
            ok, msg = adb_sms_client.check_adb_setup()
        assert ok is True

    def test_fails_when_device_unauthorized(self):
        stdout = "List of devices attached\nABCD1234\tunauthorized\n"
        with (
            patch("applypilot.tracking.adb_sms_client._find_adb", return_value="adb"),
            patch("subprocess.run", return_value=_fake_run(stdout=stdout)),
        ):
            ok, msg = adb_sms_client.check_adb_setup()
        assert ok is False
        assert "authorized" in msg

    def test_fails_when_no_device_connected(self):
        stdout = "List of devices attached\n"
        with (
            patch("applypilot.tracking.adb_sms_client._find_adb", return_value="adb"),
            patch("subprocess.run", return_value=_fake_run(stdout=stdout)),
        ):
            ok, msg = adb_sms_client.check_adb_setup()
        assert ok is False
        assert "No device" in msg


class TestListRecentTexts:
    def _row(self, msg_id, address, dt: datetime, body, direction="1"):
        epoch_ms = int(dt.timestamp() * 1000)
        return f"Row: 0 _id={msg_id}, address={address}, date={epoch_ms}, type={direction}, body={body}"

    def test_returns_recent_received_messages(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=2)
        stdout = self._row("1", "12025550123", recent, "Your code is 481920")
        with (
            patch("applypilot.tracking.adb_sms_client._find_adb", return_value="adb"),
            patch("subprocess.run", return_value=_fake_run(stdout=stdout)),
        ):
            result = adb_sms_client.list_recent_texts(since_minutes=10)
        assert len(result) == 1
        assert "481920" in result[0]["body"]

    def test_excludes_sent_messages(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=2)
        stdout = self._row("1", "12025550123", recent, "My reply", direction="2")
        with (
            patch("applypilot.tracking.adb_sms_client._find_adb", return_value="adb"),
            patch("subprocess.run", return_value=_fake_run(stdout=stdout)),
        ):
            result = adb_sms_client.list_recent_texts(since_minutes=10)
        assert result == []

    def test_excludes_messages_outside_window(self):
        old = datetime.now(timezone.utc) - timedelta(minutes=30)
        stdout = self._row("1", "12025550123", old, "Old code 111111")
        with (
            patch("applypilot.tracking.adb_sms_client._find_adb", return_value="adb"),
            patch("subprocess.run", return_value=_fake_run(stdout=stdout)),
        ):
            result = adb_sms_client.list_recent_texts(since_minutes=10)
        assert result == []

    def test_returns_empty_list_when_adb_not_found(self):
        with patch("applypilot.tracking.adb_sms_client._find_adb", return_value=None):
            assert adb_sms_client.list_recent_texts() == []

    def test_returns_empty_list_on_nonzero_return_code(self):
        with (
            patch("applypilot.tracking.adb_sms_client._find_adb", return_value="adb"),
            patch("subprocess.run", return_value=_fake_run(returncode=1, stderr="permission denied")),
        ):
            assert adb_sms_client.list_recent_texts() == []

    def test_returns_empty_list_on_subprocess_error(self):
        with (
            patch("applypilot.tracking.adb_sms_client._find_adb", return_value="adb"),
            patch("subprocess.run", side_effect=OSError("adb crashed")),
        ):
            assert adb_sms_client.list_recent_texts() == []

    def test_handles_multiline_message_body(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=1)
        epoch_ms = int(recent.timestamp() * 1000)
        stdout = f"Row: 0 _id=1, address=12025550123, date={epoch_ms}, type=1, body=Line one\nLine two 296312"
        with (
            patch("applypilot.tracking.adb_sms_client._find_adb", return_value="adb"),
            patch("subprocess.run", return_value=_fake_run(stdout=stdout)),
        ):
            result = adb_sms_client.list_recent_texts(since_minutes=10)
        assert len(result) == 1
        assert "296312" in result[0]["body"]


class TestGetLatestVerificationCode:
    def test_extracts_first_numeric_run(self):
        with patch(
            "applypilot.tracking.adb_sms_client.list_recent_texts",
            return_value=[{"address": "x", "body": "Your Lumen code is 296312", "date": None}],
        ):
            assert adb_sms_client.get_latest_verification_code() == "296312"

    def test_returns_none_when_no_digits_present(self):
        with patch(
            "applypilot.tracking.adb_sms_client.list_recent_texts",
            return_value=[{"address": "x", "body": "Hey, how's it going?", "date": None}],
        ):
            assert adb_sms_client.get_latest_verification_code() is None

    def test_returns_none_when_no_texts(self):
        with patch("applypilot.tracking.adb_sms_client.list_recent_texts", return_value=[]):
            assert adb_sms_client.get_latest_verification_code() is None
