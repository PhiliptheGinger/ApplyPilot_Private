# Session / worker architecture (FW36 → FW69 → FW46 → FW41)

Scoped 2026-10-07. Covers four backlog items that turned out to be one
problem: **a worker's identity, its Chrome instance, its CDP port, its
profile directory and its Playwright MCP server are all the same thing
today**, so anything that needs one of them to outlive a job (a paused
HITL session, a reusable MCP connection, a human hand-off) has to block
the worker.

## What is true today

- `launch_chrome(worker_id)` keys the CDP port (`BASE_CDP_PORT + id`), the
  profile dir (`chrome-workers/worker-{id}`), the extension copy and the
  HITL listener port off the worker id. Chrome is killed and relaunched per
  job (`cleanup_worker` in the per-job `finally`).
- `launcher._make_mcp_config` writes a **stdio** MCP entry
  (`npx @playwright/mcp@0.0.75 --cdp-endpoint ...`) per job, so every job
  spawns a fresh MCP server and pays the connect handshake. The
  connect race of decisions #165–176 lives in that handshake.
- Non-blocking HITL (decisions #207/#208) already proves the resources can
  be keyed by a synthetic id instead of a live thread.

## FW36 answered (experiment, 2026-10-07)

`@playwright/mcp@0.0.75` (the pinned version) supports a long-lived HTTP
server: `--port <n>` (streamable HTTP at `/mcp`, plus SSE) and
`--shared-browser-context`. Tested here: Chromium launched with a CDP
port, one `playwright-mcp --port 8931 --cdp-endpoint http://127.0.0.1:9333`
process, then **two separate client sessions in sequence**:

| Session | Connect + navigate | Result |
|---|---|---|
| 1 | 0.45 s | navigated, 23 tools |
| 2 | 0.20 s | navigated, same Chrome |

Both drove the same browser; after both, Chrome had one tab (session 2
reused it). Claude Code accepts an HTTP MCP entry
(`{"type": "http", "url": "http://localhost:8931/mcp"}`) in place of
`command`/`args`. **Not yet verified:** that Claude Code CLI 's
`--strict-mcp-config` + HTTP entry behaves identically on the user's
Windows install; one dry-run apply answers it.

## Design

**Session pool** (new `apply/session_pool.py` grows into this; today it
only allocates synthetic ids):

```
Session = {id, chrome_proc, cdp_port, profile_dir, listener_port, mcp_proc, mcp_url, state}
state: idle -> leased(worker) -> paused(needs_human) -> ready(after human) -> leased(...) -> retired
```

1. **Stage A — persistent MCP per session (fixes the connect race).**
   When a session's Chrome starts, also start `playwright-mcp --port P
   --cdp-endpoint http://127.0.0.1:{cdp}` and wait for `/mcp` to answer.
   `_make_mcp_config` emits the HTTP entry. Jobs reuse the server; restart
   it only if a health probe fails. *Done when:* 20 consecutive dry-run
   jobs on one worker show zero `browser_tool_unavailable`, and median
   time-to-first-tool drops (today ~5–30 s of handshake per job).
2. **Stage B — sessions outlive jobs.** Stop killing Chrome per job; reset
   tabs between jobs (`_reset_browser_tabs` already exists) and relaunch
   only on crash or every N jobs. *Done when:* a 10-job batch launches
   Chrome once per worker.
3. **Stage C — pause without blocking (generalized FW69).** On
   `needs_human`, the worker returns its session to the pool as `paused`
   and leases a fresh `idle` one. When the human clicks Done, the session
   becomes `ready`; **any** free worker leases it and resumes with the
   existing `skip_tab_reset=True` resume prompt (decision #187's
   mechanic). *Done when:* a CAPTCHA pause on job 1 doesn't delay job 2,
   and job 1 completes after Done without re-login.
4. **Stage D — coordinator vs. executors (FW46)** falls out of C: the
   human-first acquire loop becomes a "coordinator" that leases sessions
   and hands them off; executors only run automation. Name it
   coordinator/executor, not server/cook.
5. **Stage E — queue UI (FW41)**: the pool's state table is exactly what a
   live queue view needs; serve it from the existing listener.

## Risks

- **Memory:** 12 GB machine (decision #131). Each idle Chrome is a few
  hundred MB; cap the pool (e.g. workers + 2) and retire idle sessions.
- **Cross-job leakage:** cookies/logins persist within a session. That is
  the point for Stage C, but sessions should be scoped per employer
  domain when reused for a *different* job.
- **Pinned MCP version:** stay on 0.0.75 until Stage A is proven, then
  consider the latest (0.0.83 has the same flags).

## Where it runs

Stage A can be built and unit-tested in a cloud session; every stage needs
a supervised dry-run batch on the real machine to sign off.

## Decision needed

- OK to change the default from per-job Chrome/MCP to per-worker
  (behind a flag first, e.g. `--persistent-sessions`)?
