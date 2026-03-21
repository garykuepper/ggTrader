#!/usr/bin/env pwsh
<#
.SYNOPSIS
Run the trading strategy pipeline in the background with proper logging.

.DESCRIPTION
Launches the full pipeline as a background process with output redirected to log files.
Automatically finds and displays the pipeline status using pipeline_status.py.

.PARAMETER Watch
If specified, automatically launch pipeline_status.py --watch in a new terminal
to monitor progress in real-time.

.PARAMETER DryRun
If specified, run with --dry-run flag (3 coins, 1 chunk, 2 folds for testing).

.EXAMPLE
# Start pipeline in background
.\scripts\run_pipeline_bg.ps1

# Start pipeline and automatically watch status
.\scripts\run_pipeline_bg.ps1 -Watch

# Start dry-run mode
.\scripts\run_pipeline_bg.ps1 -DryRun

.NOTES
Requires: Python 3.8+, PowerShell 5.0+
#>

param(
    [switch]$Watch,
    [switch]$DryRun
)

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptPath

# Build arguments
$PythonArgs = @("$ScriptPath/run_full_pipeline.py")
if ($DryRun) {
    $PythonArgs += "--dry-run"
}

# Determine log file path
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputLog = Join-Path $LogDir "pipeline_$Timestamp.log"
$ErrorLog = Join-Path $LogDir "pipeline_$Timestamp_err.log"

Write-Host "Starting pipeline in background..." -ForegroundColor Green
Write-Host "Output log: $OutputLog"
Write-Host "Error log:  $ErrorLog"
Write-Host ""

# Start the process
$Process = Start-Process -FilePath python `
    -ArgumentList $PythonArgs `
    -RedirectStandardOutput $OutputLog `
    -RedirectStandardError $ErrorLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Pipeline started with PID: $($Process.Id)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitor status with:" -ForegroundColor Yellow
Write-Host "  python scripts/pipeline_status.py --watch" -ForegroundColor White
Write-Host ""
Write-Host "View logs with:" -ForegroundColor Yellow
Write-Host "  Get-Content '$OutputLog' -Wait" -ForegroundColor White
Write-Host ""

# Optionally start status watcher in new PowerShell window
if ($Watch) {
    Write-Host "Launching status monitor..." -ForegroundColor Green
    Start-Process -FilePath pwsh -ArgumentList "-Command", "cd '$ProjectRoot'; python scripts/pipeline_status.py --watch"
}

# Wait for process to complete (don't block the terminal, just report when done)
Write-Host "Waiting for pipeline to complete..."
$Process.WaitForExit()

$ExitCode = $Process.ExitCode
if ($ExitCode -eq 0) {
    Write-Host "Pipeline completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Pipeline exited with code: $ExitCode" -ForegroundColor Red
    if (Test-Path $ErrorLog) {
        Write-Host "Errors logged to: $ErrorLog" -ForegroundColor Yellow
    }
}

# Show location of results
$LatestResult = Get-ChildItem "$ProjectRoot/results" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($LatestResult) {
    Write-Host "Results saved to: $($LatestResult.FullName)" -ForegroundColor Cyan
    $ReportFile = Join-Path $LatestResult.FullName "pipeline_report.md"
    if (Test-Path $ReportFile) {
        Write-Host "Report: $ReportFile" -ForegroundColor White
    }
}
