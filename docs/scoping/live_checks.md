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

Baseline from the user's logs (all before the #216 fix): zero 'giving up after 8 attempts' in every single-stage or partial run, and in every run before 2026-09-21. Every give-up is in the full `discover+4more` --stream runs that started 2026-09-21: 473 give-ups over ~160 run-hours (~2.9/hour; 2026-09-28 to 10-06 alone: 168 over ~86 hours, ~1.9/hour). Run hours are first-to-last log write, so idle time inflates them. From ~2026-09-26 the 'database is locked' count equals the give-up count; before that, thousands of per-attempt lines. Pass: one overnight `discover+4more` run on the fixed code with zero or near-zero give-ups.

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

## 4. Persistent Playwright MCP (decision #235, now the default)

On the first apply run, `~/.applypilot/logs/playwright-mcp-w{N}.log` should
show the server listening, the worker log should show no
`Playwright MCP failed to connect` retries, and `browser_tool_unavailable`
failures should stop. If Windows rejects the HTTP entry, each job restarts
the server once and then falls back to the old per-job server on its own;
look for `-- retry 3 after Playwright MCP connect failure --` in the worker
log as the sign of that.

## 5. FW28 validation (decision #236)

```
python scripts/validate_requirement_framing.py
```

Read-only. Passes at >= 98% agreement with the Claude-direct scores; then
add `APPLYPILOT_FRAMING_CLASSIFIER=on` to `~/.applypilot/.env`. Paste the
disagreement list either way.
