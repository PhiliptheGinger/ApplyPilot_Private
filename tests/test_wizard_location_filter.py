"""E9 (2026-10-07): applypilot init writes a location filter into searches.yaml."""

from unittest.mock import patch

import yaml

from applypilot.discovery.location_filter import load_location_filter, suggest_accept_patterns
from applypilot.wizard import init as wizard


class TestSuggestAcceptPatterns:
    def test_city_and_state_code(self):
        assert suggest_accept_patterns("Greensboro, NC") == ["Greensboro", "NC", "North Carolina", "Remote"]

    def test_state_name_adds_code(self):
        assert suggest_accept_patterns("Austin, Texas") == ["Austin", "Texas", "TX", "Remote"]

    def test_remote_only(self):
        assert suggest_accept_patterns("Greensboro, NC", remote_only=True) == ["Remote"]
        assert suggest_accept_patterns("Remote") == ["Remote"]

    def test_non_us_location_kept_verbatim(self):
        assert suggest_accept_patterns("Toronto, Ontario") == ["Toronto", "Ontario", "Remote"]

    def test_never_adds_a_country(self):
        assert "United States" not in suggest_accept_patterns("Greensboro, NC")
        assert "US" not in suggest_accept_patterns("Greensboro, NC")


def _run_setup_searches(tmp_path, answers):
    path = tmp_path / "searches.yaml"
    with (
        patch.object(wizard, "SEARCH_CONFIG_PATH", path),
        patch.object(wizard.Prompt, "ask", side_effect=answers),
    ):
        wizard._setup_searches()
    return path


def test_wizard_writes_accept_and_reject_patterns(tmp_path):
    path = _run_setup_searches(
        tmp_path,
        [
            "Greensboro, NC",  # target location
            "25",  # radius
            "IT Support Specialist, Help Desk",  # roles
            "Greensboro, NC, North Carolina, Winston-Salem, Remote",  # keep
            "Raleigh",  # skip
        ],
    )
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["location"]["accept_patterns"] == ["Greensboro", "NC", "North Carolina", "Winston-Salem", "Remote"]
    assert cfg["location"]["reject_patterns"] == ["Raleigh"]
    assert load_location_filter(cfg) == (
        ["Greensboro", "NC", "North Carolina", "Winston-Salem", "Remote"],
        ["Raleigh"],
    )
    assert [q["query"] for q in cfg["queries"]] == ["IT Support Specialist", "Help Desk"]


def test_wizard_offers_suggestions_as_default(tmp_path):
    seen = {}

    def fake_ask(prompt, default=None, **kwargs):
        seen[prompt] = default
        answers = {
            "Target location": "Greensboro, NC",
            "Search radius": "25",
            "Target job titles": "Help Desk",
        }
        for key, value in answers.items():
            if prompt.startswith(key):
                return value
        return default or ""

    path = tmp_path / "searches.yaml"
    with patch.object(wizard, "SEARCH_CONFIG_PATH", path), patch.object(wizard.Prompt, "ask", side_effect=fake_ask):
        wizard._setup_searches()

    assert seen["Locations to keep (comma-separated)"] == "Greensboro, NC, North Carolina, Remote"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["location"] == {"accept_patterns": ["Greensboro", "NC", "North Carolina", "Remote"], "reject_patterns": []}


def test_values_with_quotes_round_trip(tmp_path):
    path = _run_setup_searches(tmp_path, ["Remote", "0", "Help Desk", 'Remote, "Quoted" Town', ""])
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["location"]["accept_patterns"] == ["Remote", '"Quoted" Town']


def test_ask_choice_names_the_default_once():
    """FW60 (2026-10-07): Rich printed the default twice."""
    from rich.prompt import Prompt

    seen = {}

    def fake_get_input(console, prompt, password, stream=None):
        seen["prompt"] = prompt.plain
        return ""

    with patch.object(Prompt, "get_input", side_effect=fake_get_input):
        result = wizard.ask_choice("Which SMS relay?", ["google-voice", "adb", "twilio"], "google-voice")
    assert result == "google-voice"
    assert seen["prompt"] == "Which SMS relay? (Enter for google-voice) [google-voice/adb/twilio]: "
