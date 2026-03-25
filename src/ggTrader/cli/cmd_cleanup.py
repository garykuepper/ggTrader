import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path


def register_cleanup_parser(subparsers: argparse._SubParsersAction):
    """Registers the 'cleanup' subcommand."""
    parser = subparsers.add_parser(
        "cleanup", help="Remove old results, logs, and legacy code to streamline the project"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=10,
        help="Number of recent research runs to keep (default: 10)",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also clear the archive/ folder of old legacy code",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Purge historical asset pool JSON files in data/",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicit confirmation required to delete files",
    )


def run_cleanup(args: argparse.Namespace):
    """Executes project cleanup logic."""
    if not args.confirm:
        print("ERROR: Safety check failed. Please run with --confirm to delete files.")
        return

    root = Path(os.getcwd())
    results_dir = root / "results"

    print(f"\n[{datetime.now()}] Project Cleanup Initiated...")

    # 1. Results Folder Cleanup
    if results_dir.exists():
        # Get all subdirectories in results/
        folders = [f for f in results_dir.iterdir() if f.is_dir()]
        # Sort by modification time (descending)
        folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        keep_count = args.keep
        to_delete = folders[keep_count:]

        print(f"  > Managing results/: {len(folders)} total. Keeping latest {keep_count}.")

        for folder in to_delete:
            try:
                shutil.rmtree(folder)
                # print(f"    - Deleted: {folder.name}")
            except Exception as e:
                print(f"    - Error deleting {folder.name}: {e}")

        # Cleanup root log/txt files in results (older than 1h to avoid deleting active worker logs)
        one_hour_ago = datetime.now().timestamp() - 3600
        for f in results_dir.glob("*.log"):
            if f.stat().st_mtime < one_hour_ago:
                f.unlink()
        for f in results_dir.glob("*.txt"):
            if f.name != "wfo_isolated_status.txt" and f.stat().st_mtime < one_hour_ago:
                f.unlink()

    # 2. Root Log and Tmp File Cleanup
    print("  > Purging root logs and temporary scripts...")
    patterns = ["*.log", "tmp_*.py", "lint_errors.txt", "test_cache.py", "update_coin_names.py"]
    for pattern in patterns:
        for f in root.glob(pattern):
            try:
                f.unlink()
            except Exception as e:
                print(f"    - Error deleting {f.name}: {e}")

    # 3. logs/ and tmp/ Directory Cleanup
    for d_name in ["logs", "tmp"]:
        d_path = root / d_name
        if d_path.exists() and d_path.is_dir():
            print(f"  > Clearing {d_name}/ folder content...")
            for item in d_path.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    print(f"    - Error deleting {item.name}: {e}")

    # 4. Data/ Folder Cleanup
    if args.data:
        data_dir = root / "data"
        if data_dir.exists():
            print("  > Purging historical asset pools in data/...")
            # Keep top_50_ccxt_volume.json and other system files
            keep_list = [".processed_dirs.json", "top_50_ccxt_volume.json"]
            for f in data_dir.glob("*.json"):
                if f.name not in keep_list:
                    try:
                        f.unlink()
                    except Exception as e:
                        print(f"    - Error deleting {f.name}: {e}")
        else:
            print("  ! Data directory not found. Skipping.")

    # 5. Archive Cleanup (Legacy Code)
    if args.archive:
        archive_dir = root / "archive"
        if archive_dir.exists():
            print("  > Clearing archive/ folder as requested...")
            for item in archive_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    print(f"    - Error deleting {item.name}: {e}")
        else:
            print("  ! Archive directory not found. Skipping.")

    print(f"[{datetime.now()}] Cleanup complete.")
