"""FW26 (2026-10-07): expand-bank asks for more facts when an entry is too thin."""

import json
from unittest.mock import patch

from typer.testing import CliRunner

from applypilot import cli, config
from applypilot.scoring import thin_entry

THIN = {
    "name": "Freelance Photography",
    "resume_allowed": True,
    "factual_concepts": ["Photographed local events."],
}


def _profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"project_inventory": [dict(THIN)], "experience_inventory": []}), encoding="utf-8")
    return path


class TestThinEntryHelpers:
    def test_answers_become_fact_sentences_and_blanks_are_skipped(self):
        answers = iter(["shot weddings and portraits", "", "local clients, 2019-2021", "  "])
        facts = thin_entry.collect_followup_facts("X", lambda q: next(answers))
        assert facts == ["Shot weddings and portraits.", "Local clients, 2019-2021."]

    def test_project_entry_without_responsibilities_gets_factual_concepts(self, tmp_path):
        path = _profile(tmp_path)
        item = thin_entry.add_facts_to_profile(path, "Freelance Photography", ["Edited photos in Lightroom."])
        assert item["factual_concepts"] == ["Photographed local events.", "Edited photos in Lightroom."]
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["project_inventory"][0]["factual_concepts"][-1] == "Edited photos in Lightroom."

    def test_entry_with_responsibilities_gets_responsibilities(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(
            json.dumps({"experience_inventory": [{"name": "Shop", "responsibilities": ["Ran the register."]}]}),
            encoding="utf-8",
        )
        item = thin_entry.add_facts_to_profile(path, "Shop", ["Counted the drawer nightly.", "Ran the register."])
        assert item["responsibilities"] == ["Ran the register.", "Counted the drawer nightly."]

    def test_unknown_entry_raises(self, tmp_path):
        try:
            thin_entry.add_facts_to_profile(_profile(tmp_path), "Nope", ["x."])
        except KeyError:
            return
        raise AssertionError("expected KeyError")


def _run(tmp_path, monkeypatch, banks, prompt_answers, confirm=True, extra_args=("--ask",)):
    path = _profile(tmp_path)
    monkeypatch.setattr(config, "PROFILE_PATH", path)
    monkeypatch.setattr(cli, "_bootstrap", lambda: None)
    calls = []
    saved = {}
    banks = iter(banks)

    def fake_build(item, client, profile):
        calls.append(item)
        return next(banks)

    answers = iter(prompt_answers)
    with (
        patch("applypilot.config.load_profile", lambda: json.loads(path.read_text(encoding="utf-8"))),
        patch("applypilot.llm.get_stage_client", lambda *a, **k: object()),
        patch("applypilot.scoring.local_tailor.build_phrase_bank", side_effect=fake_build),
        patch("applypilot.scoring.phrase_bank.load_bank", return_value=None),
        patch("applypilot.scoring.phrase_bank.save_bank", side_effect=lambda n, b, h: saved.update({n: b})),
        patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **k: next(answers)),
        patch("rich.prompt.Confirm.ask", return_value=confirm),
    ):
        result = CliRunner().invoke(cli.app, ["expand-bank", *extra_args])
    return result, calls, saved, path


def test_thin_entry_gets_questions_saved_facts_and_a_retry(tmp_path, monkeypatch):
    bank = {"Photographed local events.": ["Covered local events as a photographer."]}
    result, calls, saved, path = _run(
        tmp_path, monkeypatch, banks=[{}, bank], prompt_answers=["shot weddings", "", "", ""]
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 2
    assert "Shot weddings." in calls[1]["factual_concepts"]
    assert saved == {"Freelance Photography": bank}
    profile = json.loads(path.read_text(encoding="utf-8"))
    assert "Shot weddings." in profile["project_inventory"][0]["factual_concepts"]


def test_declining_to_save_changes_nothing(tmp_path, monkeypatch):
    result, calls, saved, path = _run(
        tmp_path, monkeypatch, banks=[{}], prompt_answers=["shot weddings", "", "", ""], confirm=False
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and saved == {}
    assert json.loads(path.read_text(encoding="utf-8"))["project_inventory"][0] == THIN


def test_no_ask_keeps_old_behavior(tmp_path, monkeypatch):
    result, calls, saved, _ = _run(tmp_path, monkeypatch, banks=[{}], prompt_answers=[], extra_args=("--no-ask",))
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and saved == {}
