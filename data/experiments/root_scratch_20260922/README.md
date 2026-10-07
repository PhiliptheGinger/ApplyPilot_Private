# Root-level scratch files (moved 2026-10-07)

These one-off experiment scripts and output dumps used to sit at the repo
root. They were moved here unchanged during the October 2026 cleanup so the
root only holds project files. Nothing in `src/` or `tests/` imports them;
a few code comments still mention `arbitration_test.py` and
`semantic_retrieval_experiment.py` by name -- this is where they are.

`lint_remaining.txt` (stale ruff output) and an empty file named `=` were
deleted rather than moved; both are recoverable from git history.
