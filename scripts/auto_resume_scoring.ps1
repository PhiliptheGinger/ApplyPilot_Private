# One-shot auto-resume: fires once Gemini's daily quota is expected to have
# reset (see CLAUDE.md decision #81 / 2026-09-08 session), so the clean
# labeled ground-truth pool keeps growing without manual intervention.
# Registered as a one-time Windows Scheduled Task -- see that decision for
# the exact trigger-time calculation and reasoning (deliberately NOT a
# continuously-polling scheduler, which would burn through jobs' own
# exponential-backoff retry budgets before the real quota reset arrived).

$ErrorActionPreference = "Continue"
Set-Location "C:\Users\phili\Projects\resume-agent"

$logDir = "C:\Users\phili\.applypilot\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logFile = Join-Path $logDir ("auto_resume_scoring_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

"=== auto_resume_scoring started $(Get-Date -Format o) ===" | Out-File -FilePath $logFile -Encoding utf8
& "C:\Users\phili\Projects\resume-agent\venv\Scripts\applypilot.exe" run score --limit 5000 *>> $logFile
"=== auto_resume_scoring finished $(Get-Date -Format o) (exit code $LASTEXITCODE) ===" | Out-File -FilePath $logFile -Append -Encoding utf8
