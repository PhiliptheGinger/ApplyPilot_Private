"""Tests for the per-ATS successful-path memoization helper."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def tmp_paths_dir(tmp_path, monkeypatch):
    """Redirect PATHS_DIR to a tmp directory so tests don't pollute ~/.applypilot."""
    from applypilot.apply import successful_paths
    monkeypatch.setattr(successful_paths, "PATHS_DIR", tmp_path / "successful_paths")
    return tmp_path / "successful_paths"


def test_save_then_load_roundtrip(tmp_paths_dir):
    from applypilot.apply.successful_paths import save_path, load_path
    steps = [
        {"tool": "browser_navigate", "summary": "browser_navigate https://x"},
        {"tool": "browser_snapshot", "summary": "browser_snapshot"},
        {"tool": "browser_click",    "summary": "browser_click Apply"},
    ]
    out = save_path("greenhouse", steps, job_url="https://x", duration_ms=240_000)
    assert out is not None and out.exists()

    loaded = load_path("greenhouse")
    assert loaded is not None
    assert loaded["ats_slug"] == "greenhouse"
    assert loaded["job_url"] == "https://x"
    assert loaded["duration_ms"] == 240_000
    assert len(loaded["steps"]) == 3
    assert loaded["steps"][0]["tool"] == "browser_navigate"


def test_save_caps_to_max_steps(tmp_paths_dir):
    from applypilot.apply.successful_paths import save_path, load_path, MAX_STEPS
    steps = [{"tool": "x", "summary": f"step {i}"} for i in range(MAX_STEPS + 50)]
    save_path("workday", steps)
    loaded = load_path("workday")
    assert len(loaded["steps"]) == MAX_STEPS
    # Tail-preserved (most recent steps are the form-fill phase)
    assert loaded["steps"][-1]["summary"] == f"step {MAX_STEPS + 49}"


def test_save_empty_steps_is_noop(tmp_paths_dir):
    from applypilot.apply.successful_paths import save_path, load_path
    assert save_path("greenhouse", []) is None
    assert load_path("greenhouse") is None


def test_save_blank_slug_is_noop(tmp_paths_dir):
    from applypilot.apply.successful_paths import save_path
    assert save_path("", [{"tool": "x"}]) is None
    assert save_path(None, [{"tool": "x"}]) is None  # type: ignore[arg-type]


def test_load_missing_slug_returns_none(tmp_paths_dir):
    from applypilot.apply.successful_paths import load_path
    assert load_path("nonexistent") is None
    assert load_path("") is None


def test_format_for_prompt_renders_section(tmp_paths_dir):
    from applypilot.apply.successful_paths import save_path, load_path, format_path_for_prompt
    save_path("ashby", [
        {"tool": "browser_navigate", "summary": "browser_navigate https://ashby/x"},
        {"tool": "browser_click",    "summary": "browser_click Apply"},
    ], duration_ms=180_000)
    rendered = format_path_for_prompt(load_path("ashby"))
    assert rendered is not None
    assert "PRIOR SUCCESSFUL PATH (ashby)" in rendered
    assert "completed in 180s" in rendered
    assert "browser_click Apply" in rendered
    # Hint framing — explicit "guide, not a script"
    assert "guide" in rendered.lower()


def test_format_for_prompt_handles_none(tmp_paths_dir):
    from applypilot.apply.successful_paths import format_path_for_prompt
    assert format_path_for_prompt(None) is None
    assert format_path_for_prompt({}) is None
    assert format_path_for_prompt({"steps": []}) is None


def test_overwrite_replaces_prior(tmp_paths_dir):
    from applypilot.apply.successful_paths import save_path, load_path
    save_path("lever", [{"tool": "old", "summary": "old run"}])
    save_path("lever", [{"tool": "new", "summary": "new run"}])
    loaded = load_path("lever")
    assert len(loaded["steps"]) == 1
    assert loaded["steps"][0]["summary"] == "new run"
