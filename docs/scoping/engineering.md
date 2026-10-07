# Engineering health: E1 (Python 3.12+) and E4 (splitting big files)

Scoped 2026-10-07.

## E1 — the numpy pin that blocks Python 3.12+

Investigated, **blocked upstream; recommendation: stay on 3.11.**

- `python-jobspy` 1.1.45–1.1.46 hard-pins `numpy==1.24.2` (no 3.12 wheels).
- Every later jobspy release (1.1.47 through 1.1.82) also requires
  `markdownify<0.14`.
- `pyproject.toml` requires `markdownify>=0.14.1` **as a security fix**:
  GHSA-7mpr-5m44-h73r / PYSEC-2026-1604 (oversized heading tags like
  `<h9999999>` in untrusted HTML). jobspy feeds LinkedIn/Indeed HTML through
  markdownify, so this is reachable input. ApplyPilot's own code never
  imports markdownify.
- So the only ways to Python 3.12+ are: accept the vulnerable markdownify,
  vendor jobspy (MIT) with its pins relaxed, or install with `uv` and
  `override-dependencies`. None is worth it while 3.11 works; CI and the
  security workflow are pinned to 3.11 with comments pointing here.

Revisit when jobspy relaxes its markdownify cap, or if 3.11 support ends
(October 2027).

## E4 — splitting the largest files

| File | Lines | Shape |
|---|---|---|
| `scoring/local_tailor.py` | ~4,400 | requirement extraction, evidence ranking, planning, editor, phrase bank, degraded cover |
| `apply/launcher.py` | ~3,300 | HTTP listener, acquire/lock, run_job, MCP config, Q&A parsing |
| `database.py` | ~3,170 | connection/retry, schema, state machine, backfills, stats, accounts, tracking, Q&A |
| `cli.py` | ~2,800 | one Typer command per function |

### Why not done mechanically

Tests patch functions by module path. If `database.get_connection` moves to
`database/connection.py` and the old module only re-exports it, a test that
patches `applypilot.database.X` no longer affects calls made *inside* the
new module, and can keep passing while testing nothing.

### Measured risk for `database.py` (start here)

Only five distinct patch targets point into `applypilot.database` across the
whole suite: `DB_PATH` (conftest + `tmp_db`), `get_accounts_for_prompt`,
`commit_with_retry`, `repair_enriched_without_description`,
`get_connection`. That makes it the safest first split.

### Plan (one PR per step, suite green and grep-clean at each)

1. Convert `database.py` to a package `database/` whose `__init__.py`
   re-exports the full current public surface (keeps every import working).
2. Move one cohesive group per PR, in this order:
   `qa.py` (normalize_question … export_qa_yaml), `tracking.py`
   (create_stub_job … get_tracking_stats), `accounts.py`, `company.py`
   (extract_company, backfill_companies, ATS tables), `stats.py`.
   Leave connection/retry, schema and the state machine in `__init__`
   until last, because `DB_PATH` lives there.
3. For each moved name, grep `tests/` and `src/` for patches of the old path
   and repoint them in the same PR.
4. Add a test that fails if any module defines a function that is also
   re-exported from `database/__init__` under a different object (catches
   accidental duplicate definitions).

`local_tailor.py` follows the same recipe; it has many more patch targets
(check with `grep -rhoE "local_tailor\.[A-Za-z_]+" tests | sort | uniq -c`),
so plan it after `database/` proves the process.

**Where:** entirely cloud-ok. **Size:** ~1 session for `database/`.
