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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_profile: asserts facts from the candidate's real profile.json; skipped when none is present",
    )


def pytest_collection_modifyitems(config, items):
    from pathlib import Path

    import applypilot.config as app_config

    if Path(app_config.PROFILE_PATH).exists():
        return
    skip = pytest.mark.skip(reason="needs the candidate's real profile.json (not present in this checkout)")
    for item in items:
        if "real_profile" in item.keywords:
            item.add_marker(skip)


# Module-level counter so repeated seed_job calls get unique URLs by default.
_seed_counter = [0]


@pytest.fixture(autouse=True)
def _isolate_environ():
    """Restore the real process os.environ after every test.

    2026-09-06: found via a real full-suite regression. cli.py's
    _bootstrap() calls config.load_env(), which loads ~/.applypilot/.env
    into the REAL process os.environ via python-dotenv -- with no cleanup,
    unlike monkeypatch-based patches (which self-revert). Any test that
    invokes a CLI command (e.g. the test-local probe tests) mutates
    os.environ for the rest of the pytest process. This was latent until
    APPLYPILOT_LOCAL_OLLAMA_NATIVE was added to this machine's real
    ~/.applypilot/.env the same session (decision #74) -- its mere presence
    changes LLMClient's local-provider call routing, and it leaked from one
    test into ~11 unrelated tests later in suite order that never touch
    that var themselves, each asserting the (correct, but now violated)
    OpenAI-compat-only behavior. Same failure shape decision #67d already
    fixed once for ~/.applypilot/llm_exhaustion_state.json specifically;
    generalized here to os.environ itself so any future env-var addition to
    a real dotenv file can't cause the same class of cross-test leak.
    monkeypatch's own setenv/delenv already self-revert and are unaffected
    by this -- this only cleans up mutations made outside monkeypatch."""
    import os

    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


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


@pytest.fixture(autouse=True)
def _never_send_real_notifications(monkeypatch):
    """Tests must never email anyone (FW53 notifications, 2026-10-07).

    A CLI test's _bootstrap() can load the real ~/.applypilot/.env, which may
    set APPLYPILOT_NOTIFY_EMAIL. test_notify.py replaces _send itself.
    """
    import applypilot.notify as notify

    monkeypatch.delenv("APPLYPILOT_NOTIFY_EMAIL", raising=False)
    monkeypatch.setattr(notify, "_send", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _never_touch_real_db(tmp_path_factory, monkeypatch):
    """Point the default database path at a throwaway file for every test.

    2026-10-07: test_apply_mcp_connect_retry's success path reached code that
    opens the default DB (config.DB_PATH, bound by name into database.py at
    import). On a dev machine that is the candidate's REAL
    ~/.applypilot/applypilot.db; in CI the directory doesn't exist and the
    test failed with "unable to open database file". Tests that need a real
    schema use the tmp_db fixture, which overrides this again.
    """
    import applypilot.config as config
    import applypilot.database as database

    db_file = tmp_path_factory.mktemp("default_db") / "applypilot.db"
    monkeypatch.setattr(config, "DB_PATH", db_file)
    monkeypatch.setattr(database, "DB_PATH", db_file)


@pytest.fixture(autouse=True)
def _fallback_profile_and_resume(tmp_path_factory, monkeypatch):
    """Give the suite a synthetic profile/resume when the real ones are absent.

    2026-10-07: 35+ tests called config.load_profile() or read
    resume.txt and only passed on a machine that has a real
    ~/.applypilot/profile.json (or data/profile.json) and resume.txt. In a
    clean checkout -- CI, a cloud session, a new contributor -- they failed
    with FileNotFoundError. When the real files exist this fixture does
    nothing, so tests that intentionally exercise the real profile behave
    exactly as before.
    """
    import json
    from pathlib import Path

    import applypilot.config as config
    import applypilot.scoring.resume_router as resume_router

    root = tmp_path_factory.getbasetemp() / "fallback_profile"
    root.mkdir(exist_ok=True)

    if not Path(config.PROFILE_PATH).exists():
        example = Path(__file__).resolve().parent.parent / "profile.example.json"
        profile_path = root / "profile.json"
        if not profile_path.exists():
            profile_path.write_text(json.dumps(json.loads(example.read_text(encoding="utf-8"))), encoding="utf-8")
        monkeypatch.setattr(config, "PROFILE_PATH", profile_path)

    if not Path(config.RESUME_PATH).exists():
        resume_path = root / "resume.txt"
        if not resume_path.exists():
            resume_path.write_text(
                "Alex Example\nalex@example.com\n\nEXPERIENCE\nIT Support Technician, Example Co\n"
                "- Resolved hardware and software tickets for 200 users\n\nSKILLS\nPython, SQL, Windows\n",
                encoding="utf-8",
            )
        monkeypatch.setattr(config, "RESUME_PATH", resume_path)
        monkeypatch.setattr(resume_router, "RESUME_PATH", resume_path)


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
