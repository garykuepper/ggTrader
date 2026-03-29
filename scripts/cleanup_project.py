import os
import shutil
from datetime import datetime
from pathlib import Path


def cleanup_project():
    root = Path(os.getcwd())
    results_dir = root / "results"
    archive_dir = root / "archive"

    print(f"Starting project cleanup in {root}...")

    # 1. Clean results directory - keep only the 10 most recent folders
    if results_dir.exists():
        all_folders = [f for f in results_dir.iterdir() if f.is_dir()]
        all_folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Keep the latest research folders and a few others
        keep_count = 10
        to_delete = all_folders[keep_count:]

        print(
            f"Found {len(all_folders)} result folders. Keeping {keep_count}, "
            f"deleting {len(to_delete)}..."
        )
        for folder in to_delete:
            try:
                shutil.rmtree(folder)
                # print(f"  Deleted: {folder.name}")
            except Exception as e:
                print(f"  Error deleting {folder.name}: {e}")

        # Also delete old log files in the root results/
        for log_file in results_dir.glob("*.log"):
            if log_file.stat().st_mtime < datetime.now().timestamp() - 3600:  # Older than 1h
                log_file.unlink()
                print(f"  Deleted old log: {log_file.name}")
        for txt_file in results_dir.glob("*.txt"):
            one_hour_ago = datetime.now().timestamp() - 3600
            is_status = txt_file.name == "wfo_isolated_status.txt"
            if not is_status and txt_file.stat().st_mtime < one_hour_ago:
                txt_file.unlink()
                print(f"  Deleted old txt: {txt_file.name}")

    # 2. Clean archive directory - delete items that are truly legacy
    # (Actually, let's keep the archive folder but clear its contents if the user confirmed)
    if archive_dir.exists():
        print(f"Clearing archive directory {archive_dir.name}...")
        for item in archive_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"  Error deleting archive item {item.name}: {e}")

    # 3. Final summary
    print("Cleanup complete.")


if __name__ == "__main__":
    # Safety check: don't run automatically without being called
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--confirm":
        cleanup_project()
    else:
        print("Run with --confirm to execute cleanup.")
