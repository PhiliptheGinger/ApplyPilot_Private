"""Long-lived Playwright MCP server per apply worker (session architecture Stage A).

Before this, every job's Claude subprocess spawned its own stdio Playwright
MCP server via npx and had to finish the connect handshake inside Claude
Code's 30s MCP budget. That handshake is where the intermittent
"playwright: failed" race of decisions #165-#176 lived, and every lost race
cost a Claude attempt.

Now each worker starts one `playwright-mcp --port P` HTTP server, waits until
it answers, and every job connects to it with an HTTP MCP entry. The server
outlives Chrome: tested 2026-10-07, a server kept serving new sessions after
its Chrome was killed and relaunched on the same CDP port (each session
reconnects over CDP; a call made while Chrome is down returns a tool error,
not a hang). If the server can't be started, callers fall back to the old
per-job stdio entry, so this never makes a job worse off than before.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from applypilot import config

logger = logging.getLogger(__name__)

MCP_BASE_PORT = 8931
# Pinned (was @latest). Bump deliberately after a release has soaked ~2 weeks;
# check `npm view @playwright/mcp@<v> dist.attestations.url`.
PLAYWRIGHT_MCP_PACKAGE = "@playwright/mcp@0.0.75"
# First run may download the pinned package; later runs hit the npx cache.
_READY_TIMEOUT_S = 60.0


@dataclass
class _Server:
    proc: subprocess.Popen
    port: int
    cdp_port: int


_servers: dict[int, _Server] = {}
_lock = threading.Lock()


def mcp_port(worker_id: int) -> int:
    return MCP_BASE_PORT + worker_id


def mcp_url(port: int) -> str:
    # `localhost`, not 127.0.0.1: playwright-mcp checks the Host header and
    # rejected 127.0.0.1 requests in testing.
    return f"http://localhost:{port}/mcp"


def server_args(port: int, cdp_port: int, viewport: tuple[int, int], user_agent: str) -> list[str]:
    """playwright-mcp arguments; same browser flags the per-job stdio entry uses."""
    return [
        "--prefer-offline",
        PLAYWRIGHT_MCP_PACKAGE,
        f"--port={port}",
        f"--cdp-endpoint=http://localhost:{cdp_port}",
        f"--viewport-size={viewport[0]}x{viewport[1]}",
        f"--user-agent={user_agent}",
    ]


def _is_answering(port: int, timeout: float = 2.0) -> bool:
    """True if an HTTP server answers on /mcp (any status code counts)."""
    try:
        urllib.request.urlopen(mcp_url(port), timeout=timeout)  # noqa: S310 - fixed localhost URL
        return True
    except urllib.error.HTTPError:
        return True  # a 400/405/406 still means the server is up
    except Exception:  # noqa: BLE001 - connection refused / timeout both mean "not up"
        return False


def _wait_ready(srv: _Server, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if srv.proc.poll() is not None:
            return False
        if _is_answering(srv.port):
            return True
        time.sleep(0.25)
    return False


def _kill(srv: _Server) -> None:
    from applypilot.apply.chrome import _kill_process_tree

    if srv.proc.poll() is None:
        _kill_process_tree(srv.proc.pid)


def _start(worker_id: int, cdp_port: int) -> _Server | None:
    from applypilot.apply.chrome import _get_real_user_agent, _kill_on_port, get_worker_viewport

    port = mcp_port(worker_id)
    # A server left on this port by a crashed earlier run may point at the
    # wrong Chrome; never adopt it.
    if _is_answering(port, timeout=0.5):
        _kill_on_port(port)
    npx = shutil.which("npx") or "npx"
    cmd = [npx, *server_args(port, cdp_port, get_worker_viewport(worker_id), _get_real_user_agent())]
    config.ensure_dirs()
    log_path = config.LOG_DIR / f"playwright-mcp-w{worker_id}.log"
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as e:
        logger.warning("[W%d] Could not start persistent Playwright MCP: %s", worker_id, e)
        return None
    srv = _Server(proc=proc, port=port, cdp_port=cdp_port)
    if _wait_ready(srv, _READY_TIMEOUT_S):
        logger.info("[W%d] Persistent Playwright MCP ready on port %d", worker_id, port)
        return srv
    logger.warning("[W%d] Persistent Playwright MCP did not come up (see %s)", worker_id, log_path)
    _kill(srv)
    return None


def ensure_playwright_mcp(worker_id: int, cdp_port: int, restart: bool = False) -> str | None:
    """Return the worker's MCP URL, starting or restarting the server as needed.

    Returns None when the server can't be started; the caller then uses the
    per-job stdio entry instead.
    """
    with _lock:
        srv = _servers.get(worker_id)
        healthy = (
            srv is not None
            and not restart
            and srv.cdp_port == cdp_port
            and srv.proc.poll() is None
            and _is_answering(srv.port)
        )
        if healthy:
            return mcp_url(srv.port)
        if srv is not None:
            _kill(srv)
            _servers.pop(worker_id, None)
        srv = _start(worker_id, cdp_port)
        if srv is None:
            return None
        _servers[worker_id] = srv
        return mcp_url(srv.port)


def stop_playwright_mcp(worker_id: int) -> None:
    with _lock:
        srv = _servers.pop(worker_id, None)
    if srv is not None:
        _kill(srv)


def stop_all() -> None:
    with _lock:
        servers = list(_servers.values())
        _servers.clear()
    for srv in servers:
        _kill(srv)


atexit.register(stop_all)
