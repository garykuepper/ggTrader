"""
Optimized Export:
- Fixes Windows rsync pathing issues
- Uses Parallel Chunking (-j)
- Directory Format (-Fd)
- Rsync for progress-tracked transfer
- Memory-efficient streaming output
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

# Ensure we can find internal modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

try:
    from ggTrader.utils.config import get_db_connection_string
except ImportError:

    def get_db_connection_string():
        return "postgresql://ggtrader:ggtrader@localhost:5432/ggtrader"


DEFAULT_DUMP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "exports")


def parse_connection_string(conn_str: str) -> dict:
    parsed = urlparse(conn_str)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/"),
    }


def find_pg_dump() -> str:
    pg_dump = shutil.which("pg_dump")
    if pg_dump:
        return pg_dump
    common_paths = [
        r"C:\Program Files\PostgreSQL\16\bin\pg_dump.exe",
        r"C:\Program Files\PostgreSQL\15\bin\pg_dump.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("pg_dump not found.")


def run_parallel_pg_dump(db_params: dict, output_dir: str, jobs: int = 4) -> str:
    pg_dump = find_pg_dump()
    cmd = [
        pg_dump,
        "-h",
        db_params["host"],
        "-p",
        db_params["port"],
        "-U",
        db_params["user"],
        "-Fd",
        "-j",
        str(jobs),
        "-Z",
        "0",
        "--verbose",
        "-f",
        output_dir,
        db_params["dbname"],
    ]
    env = os.environ.copy()
    if db_params["password"]:
        env["PGPASSWORD"] = db_params["password"]

    print(f"\n--- Starting Parallel Dump ({jobs} threads) ---")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] pg_dump failed.")
        sys.exit(1)
    return output_dir


def transfer_via_rsync(local_path: str, remote_host: str, remote_path: str) -> None:
    """Transfer via Rsync using relative paths to avoid Windows drive-letter colon issues."""

    # Get current working directory and find the relative path to the dump
    # This turns 'C:\Users\...\dump' into '.\data\exports\dump'
    cwd = os.getcwd()
    rel_path = os.path.relpath(local_path, cwd)

    # Use -r for recursive directory transfer
    rsync_cmd = ["rsync", "-razP", rel_path, f"{remote_host}:{remote_path}"]

    print(f"\n--- Syncing to {remote_host} ---")
    print(f"  Local path converted to: {rel_path}")

    start = time.time()
    result = subprocess.run(rsync_cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print("\n[ERROR] Rsync failed. Falling back to SCP...")
        scp_cmd = ["scp", "-r", local_path, f"{remote_host}:{remote_path}"]
        subprocess.run(scp_cmd)
    else:
        print(f"✓ Transfer completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", "-j", type=int, default=4)
    parser.add_argument("--remote", "-r", type=str)
    parser.add_argument("--remote-path", default="/home/flynn/ggtraderdb/")
    args = parser.parse_args()

    conn_str = get_db_connection_string()
    db_params = parse_connection_string(conn_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.abspath(os.path.join(DEFAULT_DUMP_DIR, f"ggtrader_dir_{timestamp}"))
    os.makedirs(output_path, exist_ok=True)

    run_parallel_pg_dump(db_params, output_path, jobs=args.jobs)

    if args.remote:
        transfer_via_rsync(output_path, args.remote, args.remote_path)

    dump_folder_name = os.path.basename(output_path)
    print(f"\n[DONE] Database successfully exported.")
    print(f"\n--- Remote Restore Instructions ---")
    print(f"1. docker cp {args.remote_path}{dump_folder_name} ggtrader_db:/tmp/")
    print(
        f"2. docker exec -it ggtrader_db pg_restore -U {db_params['user']} -d {db_params['dbname']} -j {args.jobs} /tmp/{dump_folder_name}"
    )


if __name__ == "__main__":
    main()
