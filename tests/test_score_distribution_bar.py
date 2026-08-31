"""Regression tests for the 2026-08-30 score-distribution bar-chart fix.

`applypilot status`'s "Score Distribution" table used a raw linear scale
(count/max_count*30) -- against this candidate's real, heavily skewed
distribution (score 2: 5,935 jobs vs. score 9: 8 jobs), every small-but-real
population rounded to a 0-length bar, making it look like those scores had
zero jobs even though the adjacent Count column showed the true number.
`_score_bar_length` replaces the linear scale with a log1p scale so small
nonzero populations stay visible without letting the dominant low scores
swallow the chart -- the underlying counts/scoring logic are untouched;
this is presentation-only.
"""

from __future__ import annotations

from applypilot.cli import _score_bar_length


def test_zero_count_is_zero_length():
    """Must never manufacture a bar for a genuinely empty score bucket."""
    assert _score_bar_length(0, 5935) == 0


def test_max_count_gets_full_width():
    assert _score_bar_length(5935, 5935, width=30) == 30


def test_small_nonzero_population_is_visible_against_a_dominant_max():
    """The exact real-data case this fix targets: 8 score-9 jobs against a
    5,935-job max score-2 population. The old linear scale rounded this to
    0 (int(8/5935*30) == 0) -- the bug this test guards against."""
    bar_len = _score_bar_length(8, 5935, width=30)
    assert bar_len > 0


def test_dominant_population_does_not_completely_swallow_smaller_ones():
    """A log scale must compress the gap between the biggest and smallest
    real populations relative to a linear scale -- smaller-but-real values
    stay in the same visible ballpark as the max, not orders of magnitude
    smaller on screen."""
    max_bar = _score_bar_length(5935, 5935, width=30)
    small_bar = _score_bar_length(8, 5935, width=30)
    # Linear scale would put this at 0; log scale keeps it within a
    # readable fraction of the max bar rather than negligible.
    assert small_bar >= max_bar / 5


def test_relative_ordering_is_preserved():
    """A strictly larger count must never produce a shorter bar."""
    counts = [8, 19, 24, 42, 362, 441, 940, 2016, 5935]
    max_count = max(counts)
    bars = [_score_bar_length(c, max_count) for c in counts]
    assert bars == sorted(bars)


def test_negative_or_missing_counts_are_safe():
    assert _score_bar_length(-1, 100) == 0
    assert _score_bar_length(5, 0) == 0
