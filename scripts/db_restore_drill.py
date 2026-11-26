#!/usr/bin/env python3
"""Verify a database dump in a disposable TimescaleDB container."""

import argparse
import json
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psycopg2

from ggTrader.data.kaggle_import.guards import verify_sha256
from ggTrader.utils.db_snapshot import snapshot

SCRATCH_ROOT = Path("/media/thesix/ggTrader-data/db-restore-scratch")


def container_name(token: str) -> str:
    """Return the dedicated container name."""
    return f"ggt_restore_drill_{token}"


def refuse_unsafe(name: str, scratch_dir: Path, live_container: str = "ggtrader_db") -> None:
    """Reject targets outside the dedicated disposable namespace."""
    root = SCRATCH_ROOT.resolve()
    folder = scratch_dir.resolve()
    if (
        not name.startswith("ggt_restore_drill_")
        or name == live_container
        or folder == root
        or not folder.is_relative_to(root)
    ):
        raise ValueError("Unsafe restore drill target")


def compare(expected: dict, actual: dict) -> dict:
    """Compare all counts and digests, including missing entries."""
    result: dict[str, Any] = {}
    for source, target in (("counts", "tables"), ("digests", "digests")):
        left, right = expected[source], actual[source]
        result[target] = {
            name: {
                "expected": left.get(name),
                "actual": right.get(name),
                "ok": name in left and name in right and left[name] == right[name],
            }
            for name in sorted(left.keys() | right.keys())
        }
    result["ok"] = all(item["ok"] for group in result.values() for item in group.values())
    return result


def main(
    argv: list[str] | None = None,
    *,
    connect: Callable = psycopg2.connect,
    runner: Callable = subprocess.run,
) -> int:
    """Restore and compare a dump using an isolated container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--image", default="timescale/timescaledb:latest-pg16")
    parser.add_argument("--sums", type=Path)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)
    name = container_name(secrets.token_hex(16))
    folder = SCRATCH_ROOT / name
    created = False
    folder_created = False
    conn = None
    status = 1
    try:
        refuse_unsafe(name, folder)
        verify_sha256(args.dump, args.sums or args.dump.parent / "SHA256SUMS")
        record = json.loads(args.counts.read_text())
        version = record["timescaledb"]
        expected = record["after"]
        if not isinstance(version, str) or not version:
            raise ValueError("Missing TimescaleDB version")
        SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        folder.mkdir(mode=0o700)
        folder_created = True
        password = secrets.token_urlsafe(32)
        runner(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-p",
                "127.0.0.1::5432",
                "-v",
                f"{folder}:/var/lib/postgresql/data",
                "-e",
                f"POSTGRES_PASSWORD={password}",
                args.image,
            ],
            check=True,
            capture_output=True,
        )
        created = True
        port_result = runner(
            ["docker", "port", name, "5432/tcp"], check=True, capture_output=True, text=True
        )
        address = port_result.stdout.strip()
        host, port = address.rsplit(":", 1)
        if host != "127.0.0.1" or not port.isdecimal() or not 1 <= int(port) <= 65535:
            raise ValueError("Invalid scratch container port")
        for attempt in range(60):
            try:
                conn = connect(
                    host="127.0.0.1",
                    port=int(port),
                    dbname="postgres",
                    user="postgres",
                    password=password,
                    connect_timeout=2,
                )
                break
            except psycopg2.OperationalError:
                if attempt == 59:
                    raise
                time.sleep(1)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE DATABASE ggtrader")
        conn.close()
        conn = connect(
            host="127.0.0.1",
            port=int(port),
            dbname="ggtrader",
            user="postgres",
            password=password,
            connect_timeout=2,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION timescaledb")
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
            row = cur.fetchone()
            if row is None or row[0] != version:
                raise ValueError("TimescaleDB extension version mismatch")
            cur.execute("SELECT timescaledb_pre_restore()")
        conn.close()
        conn = None
        with args.dump.open("rb") as source:
            runner(
                [
                    "docker",
                    "exec",
                    "-i",
                    name,
                    "pg_restore",
                    "--exit-on-error",
                    "--jobs",
                    "1",
                    "--no-owner",
                    "-U",
                    "postgres",
                    "-d",
                    "ggtrader",
                ],
                stdin=source,
                check=True,
                capture_output=True,
            )
        conn = connect(
            host="127.0.0.1",
            port=int(port),
            dbname="ggtrader",
            user="postgres",
            password=password,
            connect_timeout=2,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT timescaledb_post_restore()")
            cur.execute("ANALYZE")
        report = compare(expected, snapshot(conn))
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        status = 0 if report["ok"] else 1
    except Exception as exc:
        # Exceptions from subprocess and drivers can contain credentials or connection strings.
        print(f"Restore drill failed ({type(exc).__name__})", file=sys.stderr)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                print("Restore drill connection cleanup failed", file=sys.stderr)
                status = 1
        if not args.keep:
            container_removed = not created
            if created:
                try:
                    refuse_unsafe(name, folder)
                    runner(["docker", "rm", "-f", name], check=True, capture_output=True)
                    container_removed = True
                except Exception:
                    print("Restore drill container cleanup failed", file=sys.stderr)
                    status = 1
            if folder_created and container_removed:
                try:
                    refuse_unsafe(name, folder)
                    shutil.rmtree(folder)
                except Exception:
                    print("Restore drill folder cleanup failed", file=sys.stderr)
                    status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
