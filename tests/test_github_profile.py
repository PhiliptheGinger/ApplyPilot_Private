"""Tests for the GitHub public-repo ingestion gate (CLAUDE.md decision
#84/85): the deterministic + LLM reputational-flagging pipeline, the
sparse-repo filter, the interactive review loop, and the draft
project_inventory entry builder. Network calls (fetch_public_repos,
fetch_readme_text, fetch_languages) are mocked throughout -- these tests
cover the gating logic itself, not GitHub's actual API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from applypilot.discovery.github_profile import (
    REPUTATIONAL_FLAG_CATEGORIES,
    build_draft_project_entry,
    classify_reputational_flags,
    flag_repo,
    gather_repo_review_items,
    import_github_projects,
    is_sparse_repo,
    review_repos_interactively,
)


class TestIsSparseRepo:
    def test_short_readme_is_sparse(self):
        assert is_sparse_repo("Just a quick script.")

    def test_stub_marker_is_sparse_even_if_long_enough(self):
        """The real greensboro-data-coop case: a README padded past
        MIN_README_LEN but explicitly saying it's an unfinished draft."""
        text = "Initial draft of the project website. " + "Filler text to pad length. " * 10
        assert is_sparse_repo(text)

    def test_substantial_readme_not_sparse(self):
        text = (
            "# Real Project\n\nThis project scrapes and digitizes handwritten standup notes "
            "using OCR, then stores structured entries in a local database for later review. "
            "Built with Python, Tesseract, and SQLite. Includes a CLI for batch processing."
        )
        assert not is_sparse_repo(text)

    def test_empty_readme_is_sparse(self):
        assert is_sparse_repo("")


class TestFlagRepoDeterministic:
    def test_explicit_language_detected_without_llm(self):
        repo = {"name": "some-project", "description": "a normal project"}
        with patch(
            "applypilot.discovery.github_profile.classify_reputational_flags", return_value=[]
        ):
            flags = flag_repo(client=object(), repo=repo, readme="This is a fucking great tool.")
        assert "explicit_language" in flags

    def test_piracy_keyword_detected_without_llm(self):
        repo = {"name": "movie-tool", "description": "utility"}
        with patch(
            "applypilot.discovery.github_profile.classify_reputational_flags", return_value=[]
        ):
            flags = flag_repo(client=object(), repo=repo, readme="Includes a keygen for the app.")
        assert "illegal_or_circumvention" in flags

    def test_clean_repo_no_deterministic_flags(self):
        repo = {"name": "cap-predictor", "description": "A prosperity prediction tool."}
        with patch(
            "applypilot.discovery.github_profile.classify_reputational_flags", return_value=[]
        ):
            flags = flag_repo(client=object(), repo=repo, readme="Analyzes public data to predict prosperity trends.")
        assert flags == []

    def test_llm_flags_are_unioned_with_deterministic(self):
        repo = {"name": "resume-bot", "description": "automates job applications"}
        with patch(
            "applypilot.discovery.github_profile.classify_reputational_flags",
            return_value=["automation_or_scraping_tool"],
        ):
            flags = flag_repo(client=object(), repo=repo, readme="A tool that scrapes and auto-applies to jobs.")
        assert flags == ["automation_or_scraping_tool"]


class TestClassifyReputationalFlags:
    def _client(self, response_text):
        client = type("C", (), {"chat": lambda self, *a, **k: response_text})()
        return client

    def test_none_response_returns_empty(self):
        client = self._client("This is a normal utility repo.\nFLAGS: none")
        assert classify_reputational_flags(client, "some repo text") == []

    def test_single_flag_parsed(self):
        client = self._client("Expresses hostility toward a past employer.\nFLAGS: anti_corporate_or_establishment")
        result = classify_reputational_flags(client, "some repo text")
        assert result == ["anti_corporate_or_establishment"]

    def test_multiple_flags_parsed(self):
        client = self._client("Multiple concerns present.\nFLAGS: political_or_controversial, hate_or_discriminatory")
        result = classify_reputational_flags(client, "some repo text")
        assert result == ["hate_or_discriminatory", "political_or_controversial"]

    def test_unknown_category_name_is_dropped(self):
        client = self._client("FLAGS: not_a_real_category, adult_content")
        result = classify_reputational_flags(client, "some repo text")
        assert result == ["adult_content"]

    def test_malformed_response_returns_empty(self):
        client = self._client("The model rambled without a FLAGS line at all.")
        assert classify_reputational_flags(client, "some repo text") == []

    def test_llm_exception_returns_empty_not_raise(self):
        client = type("C", (), {"chat": lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))})()
        assert classify_reputational_flags(client, "some repo text") == []


class TestBuildDraftProjectEntry:
    def test_basic_shape(self):
        repo = {"name": "cap-predictor", "description": "Predicts prosperity trends.", "html_url": "https://github.com/x/cap-predictor"}
        entry = build_draft_project_entry(repo, readme="...", languages={"Python": 5000})
        assert entry["name"] == "cap-predictor"
        assert entry["status"] == "software_project"
        assert "Python" in entry["relevance_categories"] or "python" in entry["relevance_categories"]
        assert "Predicts prosperity trends." in entry["factual_concepts"]
        assert entry["constraints"]

    def test_html_repo_is_web_project(self):
        repo = {"name": "you-power-you", "description": "site", "html_url": ""}
        entry = build_draft_project_entry(repo, readme="...", languages={"HTML": 100, "CSS": 50})
        assert entry["status"] == "web_project"

    def test_github_io_repo_notes_pages_deployment(self):
        repo = {"name": "PhiliptheGinger.github.io", "description": "portfolio", "html_url": ""}
        entry = build_draft_project_entry(repo, readme="...", languages={"HTML": 100})
        assert any("GitHub Pages" in c for c in entry["factual_concepts"])

    def test_no_languages_defaults_to_software(self):
        repo = {"name": "empty-repo", "description": None, "html_url": ""}
        entry = build_draft_project_entry(repo, readme="", languages={})
        assert entry["relevance_categories"] == ["software"]


class TestReviewReposInteractively:
    def _item(self, name="repo", sparse=False, flags=None):
        return {
            "repo": {"name": name, "description": "desc"},
            "readme": "readme text",
            "languages": {"Python": 100},
            "sparse": sparse,
            "flags": flags or [],
        }

    def test_clean_repo_defaults_to_kept_on_enter(self):
        items = [self._item()]
        with patch("rich.prompt.Confirm.ask", return_value=True) as mock_ask:
            kept = review_repos_interactively(items)
        assert len(kept) == 1
        # Default should be True (keep) for a clean, non-sparse repo.
        assert mock_ask.call_args.kwargs.get("default") is True

    def test_sparse_repo_defaults_to_excluded(self):
        items = [self._item(sparse=True)]
        with patch("rich.prompt.Confirm.ask", return_value=False) as mock_ask:
            review_repos_interactively(items)
        assert mock_ask.call_args.kwargs.get("default") is False

    def test_flagged_repo_defaults_to_excluded(self):
        items = [self._item(flags=["explicit_language"])]
        with patch("rich.prompt.Confirm.ask", return_value=False) as mock_ask:
            review_repos_interactively(items)
        assert mock_ask.call_args.kwargs.get("default") is False

    def test_user_can_override_default_and_include_flagged_repo(self):
        """The user always has final say -- a flagged repo the user
        explicitly confirms must still be kept, not silently dropped."""
        items = [self._item(flags=["political_or_controversial"])]
        with patch("rich.prompt.Confirm.ask", return_value=True):
            kept = review_repos_interactively(items)
        assert len(kept) == 1

    def test_user_can_decline_a_clean_repo(self):
        items = [self._item()]
        with patch("rich.prompt.Confirm.ask", return_value=False):
            kept = review_repos_interactively(items)
        assert kept == []


class TestGatherRepoReviewItems:
    def test_fetches_and_computes_status_per_repo(self):
        repos = [{"name": "repo-a", "description": "d", "fork": False}]
        with (
            patch("applypilot.discovery.github_profile.fetch_public_repos", return_value=repos),
            patch("applypilot.discovery.github_profile.fetch_readme_text", return_value="A real, substantial README describing genuine functionality in enough detail to not look like a stub. It explains the project's purpose, how it works internally, and how to set it up and run it locally, with real command examples included below."),
            patch("applypilot.discovery.github_profile.fetch_languages", return_value={"Python": 100}),
            patch("applypilot.discovery.github_profile.flag_repo", return_value=[]),
        ):
            items = gather_repo_review_items("someuser", client=object())
        assert len(items) == 1
        assert items[0]["sparse"] is False
        assert items[0]["flags"] == []


class TestImportGithubProjects:
    def test_end_to_end_with_mocked_network_and_llm(self):
        repos = [
            {"name": "good-repo", "description": "A real project.", "html_url": "https://github.com/u/good-repo", "fork": False},
            {"name": "stub-repo", "description": "wip", "html_url": "https://github.com/u/stub-repo", "fork": False},
        ]

        def fake_readme(username, name):
            return "" if name == "stub-repo" else "A real, substantial README describing genuine functionality in enough detail to not look like a stub. It explains the project's purpose, how it works internally, and how to set it up and run it locally, with real command examples included below."

        with (
            patch("applypilot.discovery.github_profile.fetch_public_repos", return_value=repos),
            patch("applypilot.discovery.github_profile.fetch_readme_text", side_effect=fake_readme),
            patch("applypilot.discovery.github_profile.fetch_languages", return_value={"Python": 100}),
            patch("applypilot.discovery.github_profile.classify_reputational_flags", return_value=[]),
            patch("rich.prompt.Confirm.ask", side_effect=[True, False]),
        ):
            entries = import_github_projects("someuser", client=object())

        assert len(entries) == 1
        assert entries[0]["name"] == "good-repo"

    def test_no_repos_kept_returns_empty_list(self):
        repos = [{"name": "repo-a", "description": "d", "html_url": "", "fork": False}]
        with (
            patch("applypilot.discovery.github_profile.fetch_public_repos", return_value=repos),
            patch("applypilot.discovery.github_profile.fetch_readme_text", return_value="A real, substantial README describing genuine functionality in enough detail to not look like a stub. It explains the project's purpose, how it works internally, and how to set it up and run it locally, with real command examples included below."),
            patch("applypilot.discovery.github_profile.fetch_languages", return_value={}),
            patch("applypilot.discovery.github_profile.classify_reputational_flags", return_value=[]),
            patch("rich.prompt.Confirm.ask", return_value=False),
        ):
            entries = import_github_projects("someuser", client=object())
        assert entries == []
