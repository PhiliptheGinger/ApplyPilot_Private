<#
.SYNOPSIS
    Minimal restart-on-exit supervisor for `applypilot run-continuous`.

.DESCRIPTION
    This is NOT a second scheduler -- run-continuous already implements the
    full discover/enrich/score/tailor/cover/apply loop and its own
    single-instance PID lock (~/.applypilot/run-state/). This script only
    solves the one remaining operational gap: nothing currently restarts
    run-continuous if it exits (crash, transient error, manual Ctrl+C during
    a terminal session, etc.) or keeps it running unattended.

    Deliberately distinct from the repo's watchdog.py, which is a bounded-run
    RAM/disk emergency safety net (can kill ApplyPilot AND sleep the whole
    machine under sustained resource pressure, then exits -- it never
    restarts what it supervises). Reusing watchdog.py's restart-on-exit
    behavior for an always-on service would also inherit its emergency
    machine-sleep action, which is not appropriate for unattended operation.
    This script does none of that -- it only restarts the same command.

    Safe against duplicate instances: if this script is accidentally started
    twice (or a manual `run-continuous` is already active), the SECOND
    process will hit run-continuous's own existing PID lock, print its
    refusal message, and exit immediately -- this loop will just keep
    retrying every $RestartDelaySeconds harmlessly until the first instance
    stops.

.PARAMETER RunContinuousArgs
    Extra arguments forwarded verbatim to `applypilot run-continuous`
    (e.g. -RunContinuousArgs "--no-continuous-apply","--no-discovery").
    Empty by default -- runs the full discover/enrich/score/tailor/cover/
    apply loop with default settings.

.PARAMETER RestartDelaySeconds
    Seconds to wait before relaunching after an exit. Default 30.

.EXAMPLE
    # Full loop, restart forever, in this terminal (Ctrl+C to stop):
    .\scripts\run_continuous_supervisor.ps1

.EXAMPLE
    # Work the existing backlog only, no new discovery, no auto-apply yet:
    .\scripts\run_continuous_supervisor.ps1 -RunContinuousArgs "--no-discovery","--no-continuous-apply"
#>

param(
    [string[]]$RunContinuousArgs = @(),
    [int]$RestartDelaySeconds = 30
)

$ErrorActionPreference = "Continue"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectDir "venv\Scripts\python.exe"
$LogDir = Join-Path $env:USERPROFILE ".applypilot\logs"
$SupervisorLog = Join-Path $LogDir "run-continuous-supervisor.log"

if (-not (Test-Path $Python)) {
    Write-Error "venv python not found at $Python -- run this from the repo's own venv."
    exit 1
}
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

function Write-SupervisorLog {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Output $line
    Add-Content -Path $SupervisorLog -Value $line
}

# Self-lock (2026-08-31): the Windows Scheduled Task that launches this
# script uses a recurring (e.g. every 5 minutes) trigger rather than
# at-logon/at-startup -- both of those trigger types are blocked by policy
# on this machine (confirmed: schtasks /SC ONLOGON and /SC ONSTART return
# "Access is denied" even for a trivial cmd.exe target, while /SC MINUTE
# succeeds). A recurring trigger means this script could in principle be
# launched again while a previous instance's restart-loop is still active.
# run-continuous itself already refuses a second concurrent instance (its
# own scheduler.pid lock), so a duplicate supervisor could never cause a
# second real pipeline to run -- but it would waste a PowerShell process
# endlessly retrying against that lock. This lock file prevents even that:
# a supervisor that finds a live one already running exits immediately,
# harmlessly, leaving the existing one as the sole active instance.
$SupervisorLockFile = Join-Path (Join-Path $env:USERPROFILE ".applypilot\run-state") "supervisor.pid"
$LockDir = Split-Path -Parent $SupervisorLockFile
if (-not (Test-Path $LockDir)) { New-Item -ItemType Directory -Force -Path $LockDir | Out-Null }

if (Test-Path $SupervisorLockFile) {
    $existingPid = Get-Content $SupervisorLockFile -ErrorAction SilentlyContinue
    if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        Write-SupervisorLog "Another supervisor instance is already running (pid=$existingPid) -- exiting, not starting a duplicate."
        exit 0
    }
    Write-SupervisorLog "Found a stale lock file (pid=$existingPid, not running) -- taking over."
}
Set-Content -Path $SupervisorLockFile -Value $PID

Write-SupervisorLog "=== supervisor starting (pid=$PID) ==="
Write-SupervisorLog "Command: $Python -m applypilot run-continuous $($RunContinuousArgs -join ' ')"
Write-SupervisorLog "Per-run ApplyPilot logs remain under ~/.applypilot/logs/ as usual; this file only records supervisor start/restart events."

try {
    while ($true) {
        Write-SupervisorLog "Launching run-continuous..."
        & $Python -m applypilot run-continuous @RunContinuousArgs
        $exitCode = $LASTEXITCODE
        Write-SupervisorLog "run-continuous exited with code $exitCode -- restarting in ${RestartDelaySeconds}s (Ctrl+C to stop the supervisor)"
        Start-Sleep -Seconds $RestartDelaySeconds
    }
} finally {
    Remove-Item -Path $SupervisorLockFile -ErrorAction SilentlyContinue
    Write-SupervisorLog "=== supervisor stopped ==="
}
