"""Tests for the Google Voice SMS relay client (decision #199) -- reads texts
Google Voice forwards into Gmail, using the existing gmail_client MCP plumbing.

get_latest_verification_code is tested against the extraction logic alone
(mocking search_recent_texts), matching how sms_client.get_latest_verification_code
is tested. search_recent_texts is tested against a minimal fake MCP
session/context-manager, since no prior test file in this codebase mocks
gmail_client's raw MCP layer to mirror. Tests stay plain `def` (not
`async def`) and drive the async code via `asyncio.run(...)`, since
pytest-asyncio isn't a project dependency.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from applypilot.tracking import google_voice_client


class TestGetLatestVerificationCode:
    def test_extracts_first_numeric_run(self):
        with patch(
            "applypilot.tracking.google_voice_client.search_recent_texts",
            new=AsyncMock(return_value=[{"id": "m1", "body": "Your Lumen code is 296312"}]),
        ):
            assert asyncio.run(google_voice_client.get_latest_verification_code()) == "296312"

    def test_returns_none_when_no_digits_present(self):
        with patch(
            "applypilot.tracking.google_voice_client.search_recent_texts",
            new=AsyncMock(return_value=[{"id": "m1", "body": "Thanks for signing up!"}]),
        ):
            assert asyncio.run(google_voice_client.get_latest_verification_code()) is None

    def test_returns_none_when_no_texts(self):
        with patch(
            "applypilot.tracking.google_voice_client.search_recent_texts",
            new=AsyncMock(return_value=[]),
        ):
            assert asyncio.run(google_voice_client.get_latest_verification_code()) is None


class _FakeSession:
    def __init__(self, search_response: str, read_responses: dict[str, str]):
        self._search_response = search_response
        self._read_responses = read_responses

    async def initialize(self):
        return None

    async def call_tool(self, tool_name, arguments):
        class _Block:
            def __init__(self, text):
                self.text = text

        class _Result:
            def __init__(self, text):
                self.content = [_Block(text)]

        if tool_name == "search_emails":
            return _Result(self._search_response)
        if tool_name == "read_email":
            return _Result(self._read_responses.get(arguments["messageId"], "Error: not found"))
        raise AssertionError(f"unexpected tool call: {tool_name}")


class _FakeClientCtx:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


class _FakeStdioCtx:
    async def __aenter__(self):
        return (None, None)

    async def __aexit__(self, *exc):
        return False


class TestSearchRecentTexts:
    def test_returns_normalized_emails_with_bodies(self):
        search_response = (
            "ID: gv1\nSubject: New text message\nFrom: Voice <voice-noreply@google.com>\n"
            "Date: Thu, 24 Sep 2026 20:00:00 +0000"
        )
        read_response = (
            "Thread ID: t1\nSubject: New text message\nFrom: (555) 010-0100 <15550100100.txt.voice.google.com>\n"
            "To: candidate@gmail.com\nDate: Thu, 24 Sep 2026 20:00:00 +0000\n\nYour code is 481920"
        )
        fake_session = _FakeSession(search_response, {"gv1": read_response})

        with (
            patch(
                "applypilot.tracking.google_voice_client._create_mcp_client",
                new=AsyncMock(return_value=_FakeStdioCtx()),
            ),
            patch("mcp.ClientSession", return_value=_FakeClientCtx(fake_session)),
        ):
            result = asyncio.run(google_voice_client.search_recent_texts(limit=5))

        assert len(result) == 1
        assert "481920" in result[0]["body"]

    def test_returns_empty_list_on_connection_failure(self):
        with patch(
            "applypilot.tracking.google_voice_client._create_mcp_client",
            new=AsyncMock(side_effect=RuntimeError("mcp connect failed")),
        ):
            assert asyncio.run(google_voice_client.search_recent_texts()) == []

    def test_returns_empty_list_when_no_results(self):
        fake_session = _FakeSession("Error: no results", {})
        with (
            patch(
                "applypilot.tracking.google_voice_client._create_mcp_client",
                new=AsyncMock(return_value=_FakeStdioCtx()),
            ),
            patch("mcp.ClientSession", return_value=_FakeClientCtx(fake_session)),
        ):
            assert asyncio.run(google_voice_client.search_recent_texts()) == []
