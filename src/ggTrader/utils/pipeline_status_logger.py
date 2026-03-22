"""Timestamped pipeline status logging to stdout and a status file."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path


class StatusLogger:
    """Writes timestamped status lines to both stdout and a status.txt file."""

    def __init__(self, status_path: Path) -> None:
        self.path = status_path
        self.pipeline_start = time.time()
        self.phase_start = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Monitor this file for live progress.\n\n")

    def _elapsed_str(self, seconds: float) -> str:
        return str(timedelta(seconds=int(seconds)))

    def update(self, message: str, start_phase: bool = False) -> None:
        if start_phase:
            self.phase_start = time.time()
        total_elapsed = time.time() - self.pipeline_start
        line = f"[{self._elapsed_str(total_elapsed):>10}] {message}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def phase_done(self, phase_name: str) -> None:
        phase_elapsed = time.time() - self.phase_start
        total_elapsed = time.time() - self.pipeline_start
        line = (
            f"[{self._elapsed_str(total_elapsed):>10}] "
            f"{phase_name} COMPLETE ({self._elapsed_str(phase_elapsed)})"
        )
        print(line)
        print()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n\n")
