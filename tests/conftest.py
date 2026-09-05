"""Shared pytest fixtures for applypilot tests.

`tmp_db` yields a factory returning a fresh, schema-initialized SQLite
connection rooted at a tmp path. Each call is isolated — tests that need
multiple DBs get multiple calls.

Adaptation notes vs. original plan:
- database uses `_local` (threading.local), not `_thread_local`
- connections are cached in `_local.connections` dict keyed by path string
- init_db(db_path) accepts a path arg and returns the connection directly
"""

import sqlite3
from datetime import UTC

import pytest

# Module-level counter so repeated seed_job calls get unique URLs by default.
_seed_counter = [0]


@pytest.fixture(autouse=True)
def _isolate_llm_exhaustion_state(tmp_path, monkeypatch):
    """Every test in the suite gets its own isolated LLM exhaustion-state
    file. LLMClient._mark_exhausted (2026-09-02) persists exhaustion
    timestamps to ~/.applypilot/llm_exhaustion_state.json by default so a
    restarted process doesn't re-discover known-exhausted providers via a
    real call -- but any test that builds a real LLMClient and drives it
    through the real 429-handling path in chat() (not just tests that poke
    client._exhausted directly, which stays in-memory) writes into that
    REAL file unless this is patched first.

    2026-09-04: found this leaking from tests/test_local_llm.py::
    TestDailyExhaustionNotReset (never isolated) into the actual user's
    ~/.applypilot/llm_exhaustion_state.json -- fake provider names
    "cloud-0"/"cloud-1" ended up permanently marked exhausted on a real
    machine, and a later run of the SAME test then self-poisoned by
    loading its own prior run's leftover state back in and failing before
    it reached the behavior under test. tests/test_llm_cascade.py had
    already discovered and fixed this exact failure mode for itself
    (2026-09-02, see git blame) with a fixture of this same shape, but
    file-local -- promoted here to conftest.py so every test file gets it
    automatically instead of each one needing to remember to opt in.
    tmp_path is function-scoped (fresh per test) and monkeypatch
    auto-reverts, so no manual setUp/tearDown is needed anywhere."""
    import applypilot.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_EXHAUSTION_STATE_PATH", tmp_path / "llm_exhaustion_state.json")


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Yield a factory that returns a fresh sqlite3.Connection backed by a tmp file.

    Also monkeypatches applypilot.config.DB_PATH and APP_DIR so that
    `applypilot.database.get_connection()` returns the same connection.
    """
    from applypilot import config, database

    db_file = tmp_path / "applypilot.db"
    monkeypatch.setattr(config, "DB_PATH", db_file)
    monkeypatch.setattr(config, "APP_DIR", tmp_path)
    # database.DB_PATH is imported by-name at module load, so patching only
    # config.DB_PATH leaves database.DB_PATH pointing at the real DB.
    monkeypatch.setattr(database, "DB_PATH", db_file)

    # Reset the module-level thread-local connection cache so get_connection
    # opens fresh against the new tmp path.  _local.connections is a dict
    # keyed by path string; clear it entirely to avoid any stale handle.
    if hasattr(database._local, "connections"):
        for conn in database._local.connections.values():
            try:
                conn.close()
            except Exception:  # noqa: BLE001, S110 - fixture teardown; closing an already-closed/broken connection is harmless, must not fail the test run
                pass
        database._local.connections.clear()

    def _factory() -> sqlite3.Connection:
        # init_db(db_path) creates the schema and returns the connection.
        return database.init_db(db_file)

    yield _factory

    # Cleanup: close the connection opened for the tmp db_file
    if hasattr(database._local, "connections"):
        path_key = str(db_file)
        conn = database._local.connections.pop(path_key, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001, S110 - fixture teardown; closing an already-closed/broken connection is harmless, must not fail the test run
                pass


@pytest.fixture
def seed_job():
    """Return a callable that inserts a minimally-valid job row into a connection.

    Returns the full row dict so callers can assert on any field.
    The URL is available at ``row["url"]``.

    ``url_suffix`` is an optional keyword-only override (stripped before INSERT)
    that customises the URL path segment.  When omitted an auto-incrementing
    suffix is used so that repeated calls never collide on the UNIQUE ``url``
    constraint.
    """
    from datetime import datetime

    def _seed(conn: sqlite3.Connection, **overrides) -> dict:
        default_suffix = f"auto-{_seed_counter[0]}"
        _seed_counter[0] += 1
        suffix = overrides.get("url_suffix", default_suffix)
        now = datetime.now(UTC).isoformat()
        row = {
            "url": f"https://example.com/job/{suffix}",
            "title": "Software Engineer",
            "description": "A job.",
            "full_description": "A full description.",
            "location": "Remote (US)",
            "site": "linkedin",
            "company": "acme",
            "application_url": "https://boards.greenhouse.io/acme/jobs/1",
            "fit_score": 9,
            "tailored_resume_path": "/tmp/resume.pdf",
            "cover_letter_path": "/tmp/cover.pdf",
            "discovered_at": now,
            "apply_status": None,
            "apply_attempts": 0,
        }
        row.update({k: v for k, v in overrides.items() if k != "url_suffix"})
        cols = ", ".join(row.keys())
        qs = ", ".join("?" * len(row))
        conn.execute(f"INSERT INTO jobs ({cols}) VALUES ({qs})", tuple(row.values()))
        conn.commit()
        return row

    return _seed
