"""One-off script: split CLAUDE.md's giant "Security Decisions" table into
(1) a compact 2-column index that stays in CLAUDE.md (number + one-line gist,
which is already what column 2 of each row contains) and (2) a full archive
file (docs/decisions_archive.md) preserving every row's complete original
text verbatim, nothing lost.

Run once, by hand, not part of the pipeline. Idempotent-unsafe by design --
don't re-run against an already-compressed CLAUDE.md.
"""

import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
CLAUDE_MD = os.path.join(ROOT, "CLAUDE.md")
ARCHIVE_MD = os.path.join(ROOT, "docs", "decisions_archive.md")


def split_row(line: str) -> list[str]:
    """Split a markdown table row on bare (unescaped) pipes, treating \\|
    as a literal escaped pipe character within a cell. Returns cell
    contents with the leading/trailing empty strings (from the outer
    pipes) still included at index 0 and -1."""
    cells: list[str] = []
    current: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == "\\" and i + 1 < n and line[i + 1] == "|":
            current.append("\\|")
            i += 2
            continue
        if ch == "|":
            cells.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    cells.append("".join(current))
    return cells


def main() -> None:
    with open(CLAUDE_MD, encoding="utf-8", newline="") as f:
        content = f.read()

    marker_start = "## Security Decisions"
    marker_end = "## Known Technical Gotchas"
    start_idx = content.index(marker_start)
    end_idx = content.index(marker_end)

    before = content[:start_idx]
    section = content[start_idx:end_idx]
    after = content[end_idx:]

    lines = section.split("\n")
    header_line = None
    sep_line = None
    data_rows = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| #"):
            header_line = line
        elif stripped.startswith("|---") or stripped.startswith("| ---"):
            sep_line = line
        elif stripped.startswith("|"):
            data_rows.append(line)

    assert header_line is not None and sep_line is not None, "table header not found"
    # Verified independently (PowerShell cross-check) that this section has
    # exactly 142 real data rows, not one-per-integer-0-152 -- some decision
    # numbers were apparently never given their own row across the many
    # sessions this file was built up over. 142 is the real count, not a
    # bug in this parser.
    assert len(data_rows) == 142, f"expected 142 rows (verified count), found {len(data_rows)}"

    # Build the full archive (verbatim rows, unchanged).
    archive_lines = [
        "# ApplyPilot Decision Archive",
        "",
        "Full text of every numbered decision from CLAUDE.md's Security Decisions",
        "table. CLAUDE.md itself keeps only a compact index (number + one-line gist)",
        "to keep the auto-loaded instructions file small -- this file has the",
        "complete rationale/detail for every entry, verbatim, nothing shortened.",
        "",
        "Compressed out of CLAUDE.md on 2026-09-17 once the decisions table alone",
        "reached ~457,000 characters (87% of the whole file's size) and was being",
        "reloaded in full on every turn regardless of relevance to the task at hand.",
        "",
        header_line,
        sep_line,
    ]
    archive_lines.extend(data_rows)
    archive_text = "\n".join(archive_lines) + "\n"

    # Build the compact index (columns 1-2 only) to replace the table in CLAUDE.md.
    compact_header = "| # | Decision |"
    compact_sep = "|---|----------|"
    compact_rows = []
    for row in data_rows:
        cells = split_row(row)
        # cells[0] is empty (before leading |), cells[1]=number, cells[2]=gist,
        # cells[3]=rationale (dropped), cells[-1] empty (after trailing |).
        num = cells[1].strip()
        gist = cells[2].strip()
        compact_rows.append(f"| {num} | {gist} |")

    compact_section = (
        "## Security Decisions\n\n"
        "Full rationale/detail for every decision below has been moved to\n"
        "`docs/decisions_archive.md` to keep this auto-loaded file small -- this\n"
        "table is a compact index only (number + one-line gist). Look up a\n"
        "decision by number in the archive file for the complete real-data\n"
        "verification, root cause, and fix detail.\n\n"
        f"{compact_header}\n{compact_sep}\n" + "\n".join(compact_rows) + "\n\n---\n\n"
    )

    new_content = before + compact_section + after

    with open(ARCHIVE_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(archive_text)

    with open(CLAUDE_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)

    print(f"Archive written: {ARCHIVE_MD} ({len(archive_text)} chars, {len(data_rows)} rows)")
    print(f"CLAUDE.md rewritten: {len(new_content)} chars (was {len(content)})")


if __name__ == "__main__":
    main()
