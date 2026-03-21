# Pipeline Execution Guide

This guide explains how to run the trading strategy pipeline with proper background execution and monitoring.

## Quick Start

### PowerShell (Recommended)

```powershell
# Run full pipeline in background
.\scripts\run_pipeline_bg.ps1

# Run and automatically watch status in real-time
.\scripts\run_pipeline_bg.ps1 -Watch

# Run dry-run mode (3 coins, 1 chunk, 2 folds for testing)
.\scripts\run_pipeline_bg.ps1 -DryRun
```

### Batch (Alternative for CMD)

```cmd
# Run full pipeline
scripts\run_pipeline_bg.bat

# Run with dry-run mode
scripts\run_pipeline_bg.bat --dry-run

# Run and watch status
scripts\run_pipeline_bg.bat --watch
```

## Monitoring

### Option 1: Watch Status Script (Recommended)
```powershell
python scripts/pipeline_status.py --watch
```
This automatically finds the latest pipeline run and displays live updates with 30-second refresh intervals.

### Option 2: View Logs Directly
```powershell
Get-Content logs\pipeline_<timestamp>.log -Wait
```

### Option 3: Check Status File
The pipeline writes live status to `results/pipeline_<timestamp>/status.txt`:
```powershell
Get-Content results/pipeline_<timestamp>/status.txt -Wait
```

## Features

✓ **Reliable Background Execution**: Uses proper OS-level background process handling instead of shell redirection
✓ **Automatic Logging**: Outputs separated into `.log` and `_err.log` files with timestamps
✓ **Status Monitoring**: Real-time progress tracking with ETAs
✓ **Dry-Run Mode**: Test the pipeline with minimal data (3 coins, 1 chunk, 2 folds)
✓ **Results Organization**: Automatically displays results directory path when complete

## Why These Scripts?

The naive `python scripts/run_full_pipeline.py > pipeline.log 2>&1` approach has issues in PowerShell:
- Output buffering causes delays in log updates
- Process may hang or appear frozen
- Logs don't update while process is running

The `run_pipeline_bg.ps1` script solves these by:
- Using `Start-Process` with explicit log redirection
- Properly handling Python output buffering
- Supporting real-time monitoring with `pipeline_status.py`

## Log Locations

- **Pipeline logs**: `logs/pipeline_<timestamp>.log`
- **Pipeline errors**: `logs/pipeline_<timestamp>_err.log`
- **Status updates**: `results/pipeline_<timestamp>/status.txt`
- **Final report**: `results/pipeline_<timestamp>/pipeline_report.md`

## Troubleshooting

### Script won't execute
If you get "cannot be loaded because running scripts is disabled", enable script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Pipeline not starting
1. Check that you're in the project root directory
2. Verify Python is installed: `python --version`
3. Ensure dependencies are installed: `pip install -e .`

### Logs not updating in real-time
Use the `--watch` flag to automatically monitor:
```powershell
.\scripts\run_pipeline_bg.ps1 -Watch
```

Or manually check status:
```powershell
python scripts/pipeline_status.py --watch
```
