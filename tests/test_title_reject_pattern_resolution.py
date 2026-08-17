from applypilot import cli


def test_resolve_title_reject_patterns_dedupes_and_merges(monkeypatch):
    monkeypatch.setattr(
        cli.config,
        "load_search_config",
        lambda: {
            "title_reject_patterns": [r"\bsenior\b", r"\barchitect\b"],
            "filters": {"title_reject_patterns": [r"\bignored\b"]},
        },
    )

    got = cli._resolve_title_reject_patterns(
        use_defaults=True,
        use_config=True,
        cli_patterns=[r"\barchitect\b", r"\bprincipal\b"],
    )

    assert r"\bsenior\b" in got
    assert r"\barchitect\b" in got
    assert got.count(r"\barchitect\b") == 1
    assert r"\bprincipal\b" in got


def test_resolve_title_reject_patterns_supports_filters_fallback(monkeypatch):
    monkeypatch.setattr(
        cli.config,
        "load_search_config",
        lambda: {"filters": {"title_reject_patterns": [r"\bstaff\b"]}},
    )

    got = cli._resolve_title_reject_patterns(
        use_defaults=False,
        use_config=True,
        cli_patterns=None,
    )

    assert got == [r"\bstaff\b"]
