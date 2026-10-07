# Archive

Nothing in this project is deleted outright: decisions, Future Work items,
notes and stray files that are no longer current get moved here (or into the
archive files below) so they can be looked up later.

| What | Where |
|------|-------|
| Full text of every numbered decision | `docs/decisions_archive.md` |
| Every Future Work item, including closed ones (struck through, never removed) | `docs/future_work.md` |
| One-off experiment scripts and their output | `data/experiments/` (one dated folder per experiment) |
| Root-level scratch files moved during the October 2026 cleanup | `data/experiments/root_scratch_20260922/` |
| Stray files that were going to be deleted | this folder |

## Contents

- `lint_remaining_2026-09-22.txt` -- ruff output that sat at the repo root
  (2,781 lines, from an earlier, wider rule set). Restored here on 2026-10-07
  after the cleanup first deleted it. Lint is now configured in
  `pyproject.toml` (`[tool.ruff.lint]`).

An empty file named `=` (0 bytes, almost certainly a mistyped shell
redirect) was also removed from the repo root on 2026-10-07. It had no
content to keep.
