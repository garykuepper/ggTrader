"""Export the local TimescaleDB database and optionally transfer it to a remote server."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.utils.config import get_db_connection_string

# Defaults
DEFAULT_DUMP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "exports"
)
DEFAULT_DUMP_FORMAT = "custom"  # pg_dump -Fc (compressed, best for pg_restore)


def parse_connection_string(conn_str: str) -> dict:
    """Extract host, port, user, and dbname from a SQLAlchemy connection string."""
    parsed = urlparse(conn_str)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/"),
    }


def find_pg_dump() -> str:
    """Locate pg_dump, checking PATH and common WSL locations."""
    pg_dump = shutil.which("pg_dump")
    if pg_dump:
        return pg_dump

    # Common Windows PostgreSQL install paths
    common_paths = [
        r"C:\Program Files\PostgreSQL\16\bin\pg_dump.exe",
        r"C:\Program Files\PostgreSQL\15\bin\pg_dump.exe",
        r"C:\Program Files\PostgreSQL\14\bin\pg_dump.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "pg_dump not found. Install PostgreSQL client tools or add them to PATH.\n"
        "  Windows: https://www.postgresql.org/download/windows/\n"
        "  Or install just the client: choco install postgresql --params '/Password:\"\"'"
    )


def run_pg_dump(db_params: dict, output_path: str, dump_format: str = "custom") -> str:
    """Run pg_dump and return the output file path."""
    pg_dump = find_pg_dump()

    format_flag = {"custom": "-Fc", "directory": "-Fd", "plain": "-Fp"}
    fmt = format_flag.get(dump_format, "-Fc")

    cmd = [
        pg_dump,
        "-h",
        db_params["host"],
        "-p",
        db_params["port"],
        "-U",
        db_params["user"],
        fmt,
        "--verbose",
        "-f",
        output_path,
        db_params["dbname"],
    ]

    env = os.environ.copy()
    if db_params["password"]:
        env["PGPASSWORD"] = db_params["password"]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output:  {output_path}")
    print("  This may take several minutes for large databases...\n")

    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  [ERROR] pg_dump failed (exit code {result.returncode}):")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Dump completed in {elapsed:.1f}s — {size_mb:.1f} MB")
    return output_path


def transfer_to_remote(
    local_path: str,
    remote_host: str,
    remote_path: str,
    ssh_key: str | None = None,
    ssh_port: int = 22,
) -> None:
    """Transfer dump file to remote server via SCP."""
    scp_cmd = ["scp", "-P", str(ssh_port)]

    if ssh_key:
        scp_cmd.extend(["-i", ssh_key])

    scp_cmd.extend([local_path, f"{remote_host}:{remote_path}"])

    print(f"\n--- Transferring to {remote_host} ---")
    print(f"  Command: {' '.join(scp_cmd)}")

    start = time.time()
    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  [ERROR] SCP transfer failed:")
        print(f"  stderr: {result.stderr}")
        sys.exit(1)

    print(f"  ✓ Transfer completed in {elapsed:.1f}s")


def print_restore_instructions(
    db_params: dict, remote_host: str | None, remote_path: str
) -> None:
    """Print the commands needed to restore on the remote server."""
    dbname = db_params["dbname"]
    dump_file = remote_path if remote_host else os.path.basename(remote_path)

    print("\n--- Restore Instructions ---")
    if remote_host:
        print(f"  SSH into your server:  ssh {remote_host}")
    print(f"\n  # 1. Create database and enable TimescaleDB")
    print(f"  sudo -u postgres createdb {dbname}")
    print(
        f"  sudo -u postgres psql -d {dbname} "
        f'-c "CREATE EXTENSION IF NOT EXISTS timescaledb;"'
    )
    print(f"\n  # 2. Restore the dump")
    print(
        f"  sudo -u postgres pg_restore -d {dbname} "
        f"--no-owner --verbose {dump_file}"
    )
    print(f"\n  # 3. (Optional) Re-enable compression after restore")
    print(
        f"  sudo -u postgres psql -d {dbname} "
        f"-c \"SELECT add_compression_policy('ohlcv', "
        f"INTERVAL '30 days');\""
    )


def main() -> None:
    """Orchestrate database export and optional remote transfer."""
    parser = argparse.ArgumentParser(
        description="Export ggTrader TimescaleDB database and optionally "
        "transfer to a remote Linux server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dump only (saves to data/exports/)\n"
            "  python export_db.py\n\n"
            "  # Dump and transfer to remote server\n"
            "  python export_db.py --remote user@192.168.1.100\n\n"
            "  # Dump and transfer with SSH key\n"
            "  python export_db.py --remote user@server --key ~/.ssh/id_rsa\n\n"
            "  # Custom output path\n"
            "  python export_db.py --output /tmp/ggtrader.dump\n"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for the dump file. Default: data/exports/ggtrader_YYYYMMDD_HHMMSS.dump",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["custom", "plain", "directory"],
        default=DEFAULT_DUMP_FORMAT,
        help="pg_dump format: custom (default, compressed), plain (SQL), directory",
    )
    parser.add_argument(
        "--remote",
        "-r",
        type=str,
        default=None,
        help="Remote destination as user@host (e.g., deploy@192.168.1.100)",
    )
    parser.add_argument(
        "--remote-path",
        type=str,
        default="/tmp/ggtrader.dump",
        help="Path on the remote server to store the dump. Default: /tmp/ggtrader.dump",
    )
    parser.add_argument(
        "--key",
        "-k",
        type=str,
        default=None,
        help="Path to SSH private key for SCP transfer",
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=22,
        help="SSH port for remote transfer. Default: 22",
    )
    parser.add_argument(
        "--skip-dump",
        action="store_true",
        help="Skip the dump step; only transfer an existing file (requires --output)",
    )

    args = parser.parse_args()

    # Parse DB connection
    conn_str = get_db_connection_string()
    db_params = parse_connection_string(conn_str)

    print("=" * 50)
    print("  ggTrader Database Export")
    print("=" * 50)
    print(f"  Database: {db_params['dbname']}")
    print(f"  Host:     {db_params['host']}:{db_params['port']}")
    print(f"  User:     {db_params['user']}")

    # Resolve output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".dump" if args.format == "custom" else ".sql"
        dump_dir = os.path.abspath(DEFAULT_DUMP_DIR)
        os.makedirs(dump_dir, exist_ok=True)
        output_path = os.path.join(dump_dir, f"ggtrader_{timestamp}{ext}")

    # Step 1: Dump
    if not args.skip_dump:
        print(f"\n--- Dumping database ---")
        run_pg_dump(db_params, output_path, args.format)
    else:
        if not os.path.exists(output_path):
            print(f"  [ERROR] --skip-dump used but file not found: {output_path}")
            sys.exit(1)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(
            f"\n  Skipping dump, using existing file: {output_path} ({size_mb:.1f} MB)"
        )

    # Step 2: Transfer (optional)
    if args.remote:
        transfer_to_remote(
            local_path=output_path,
            remote_host=args.remote,
            remote_path=args.remote_path,
            ssh_key=args.key,
            ssh_port=args.ssh_port,
        )

    # Step 3: Print restore instructions
    print_restore_instructions(db_params, args.remote, args.remote_path)

    print("\n✓ Export complete!")


if __name__ == "__main__":
    main()
