# Live checks for the next session on the real machine

Written 2026-10-07 for the remote-control session on the user's machine.
Everything below was built and tested in a cloud session; these steps confirm
it against the real DB, Chrome and logs. Pull branch
`claude/adoring-dijkstra-xu2p07` (or main, once merged) first.

## 1. searches.yaml (user's call, nothing changed yet)

- `location.accept_patterns` includes the country-wide entries
  "United States", "US" and "USA". With whole-word matching (decision #217)
  they accept any onsite US posting that names the country
  ("Denver, CO, United States"), so the local filter is effectively
  nationwide. Remote jobs are kept regardless, so removing the three entries
  keeps remote coverage. Ask the user before editing.
- The legacy `location_accept` / `location_reject_non_remote` keys are YAML
  aliases of the new lists; they are merged and deduplicated, so they are
  harmless. Removing them is optional.
- "Winston" is redundant next to "Winston-Salem" but harmless.
- The reject list now applies to Greenhouse, Lever and Ashby too.

## 2. E3: did the transaction-leak fix end the lock errors? (#216, #183)

Run one overnight `--stream` on the new code, then compare counts of
`giving up after 8 attempts` and `database is locked` per log against the
nights before (logs in `~/.applypilot/logs/`). Zero or near-zero gave-ups
closes #183; otherwise redesign.

## 3. Other checks

- FW17: query the corpus for jobs newly rejected by the Canadian-province
  pattern and confirm none are real US locations.
- FW51: one `applypilot apply --dry-run --url URL` on a site with a cookie
  banner; confirm the banner is declined, never accepted.
- FW23/FW24: the reviews listed in `docs/backlog.md`.
- FW53: `applypilot notify --test` sends a real email.
- FW36: confirm Claude Code accepts the HTTP MCP entry under
  `--strict-mcp-config` on Windows (see `session_architecture.md`).
