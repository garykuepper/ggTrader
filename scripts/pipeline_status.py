#!/usr/bin/env python
"""Display status of the most recently started pipeline."""

import sys
import time
from pathlib import Path


def get_latest_status_file():
    """Find the most recently modified pipeline status.txt file."""
    results = Path("results")
    if not results.exists():
        return None
    
    status_files = list(results.glob("pipeline_*/status.txt"))
    if not status_files:
        return None
    
    # Sort by modification time, get the latest
    return max(status_files, key=lambda p: p.stat().st_mtime)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Display status of the most recently started pipeline"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch the status file and refresh every 30 seconds until completion"
    )
    args = parser.parse_args()
    
    status_file = get_latest_status_file()
    
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
        # Single display
        print(status_file.read_text())
    else:
        # Watch mode
        print("Watching pipeline status (Ctrl+C to stop)...")
        print()
        
        try:
            while True:
                content = status_file.read_text()
                print(content)
                
                # Check if pipeline is complete
                if "COMPLETE" in content or "FAILED" in content:
                    print("\nPipeline is complete.")
                    break
                
                print("\n--- Next update in 30 seconds (Ctrl+C to stop) ---\n")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n\nStopped watching pipeline.")


if __name__ == "__main__":
    main()
