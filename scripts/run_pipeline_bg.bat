@echo off
REM Run the trading strategy pipeline in the background with proper logging
REM Usage:
REM   run_pipeline_bg.bat              - Run full pipeline
REM   run_pipeline_bg.bat --dry-run    - Run dry-run mode
REM   run_pipeline_bg.bat --watch      - Run and watch status

setlocal enabledelayedexpansion

cd /d "%~dp0.."

REM Parse arguments
set "args="
set "watch="
if "%1"=="--watch" (
    set "watch=1"
    set "args=%2%"
) else (
    set "args=%*"
)

REM Create logs directory
if not exist logs mkdir logs

REM Generate timestamp
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set "datevar=%%c%%a%%b"
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do set "timevar=%%a%%b"
set "timestamp=!datevar!_!timevar!"

set "logfile=logs\pipeline_!timestamp!.log"
set "errfile=logs\pipeline_!timestamp!_err.log"

echo Starting pipeline in background...
echo Output log: %logfile%
echo Error log:  %errfile%
echo.

REM Set environment for background run
set "PYTHONUNBUFFERED=1"

REM Run pipeline in background with --no-progress flag
start /B python scripts/run_full_pipeline.py --no-progress %args% > %logfile% 2> %errfile%

echo Pipeline started. PID: %ERRORLEVEL%
echo.
echo Monitor status with:
echo   python scripts/pipeline_status.py --watch
echo.
echo View logs with:
echo   type %logfile%
echo.

REM Optionally start status watcher
if "%watch%"=="1" (
    echo Launching status monitor...
    start python scripts/pipeline_status.py --watch
)
