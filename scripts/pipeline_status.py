#!/usr/bin/env python
"""Display status of the most recently started pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ggTrader.utils.paths import find_project_root


def get_latest_status_file(results_dir: Path) -> Path | None:
    """Find the most recently modified pipeline status.txt file."""
    if not results_dir.exists():
        return None

    status_files = list(results_dir.glob("pipeline_*/status.txt"))
    if not status_files:
        return None

    return max(status_files, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """Print the latest pipeline status (optionally watch for updates)."""
    parser = argparse.ArgumentParser(
        description="Display status of the most recently started pipeline",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch the status file and refresh every 30 seconds until completion",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing pipeline_* folders (default: <project>/results)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        metavar="SEC",
        help="Seconds between full status refreshes when using --watch (default: 30)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = find_project_root() / "results"
    results_dir = results_dir.resolve()

    status_file = get_latest_status_file(results_dir)

    if not status_file:
        print("No active pipeline found. Check results/ directory.")
        sys.exit(1)

    pipeline_name = status_file.parent.name
    print(f"Pipeline: {pipeline_name}")
    try:
        rel_path = status_file.relative_to(Path.cwd())
    except ValueError:
        rel_path = status_file
    print(f"Status file: {rel_path}")
    print("=" * 100)
    print()

    if not args.watch:
        print(status_file.read_text(encoding="utf-8"))
    else:
        print("Watching pipeline status (Ctrl+C to stop)...")
        print()

        try:
            while True:
                content = status_file.read_text(encoding="utf-8")
                print(content)

                if "COMPLETE" in content or "FAILED" in content:
                    print("\nPipeline is complete.")
                    break

                interval = max(2, min(args.interval, 600))
                print(f"\n--- Next update in {interval} seconds (Ctrl+C to stop) ---\n")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nStopped watching pipeline.")


if __name__ == "__main__":
    main()
