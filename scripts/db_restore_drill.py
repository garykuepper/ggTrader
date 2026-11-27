#!/usr/bin/env python3
"""Verify a database dump in a disposable TimescaleDB container."""

import argparse
import errno
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import sql

from ggTrader.data.kaggle_import.guards import verify_sha256
from ggTrader.utils.db_snapshot import snapshot

SCRATCH_ROOT = Path("/media/thesix/ggTrader-data/db-restore-scratch")
REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRA_ROLES = ("universe_reader", "universe_loader")


def container_name(token: str) -> str:
    """Return the dedicated container name."""
    return f"ggt_restore_drill_{token}"


def space_path(root: Path) -> Path:
    """Return the nearest existing parent, whose drive holds the scratch folder."""
    path = root.resolve()
    while not path.exists():
        path = path.parent
    return path


def refuse_unsafe(
    name: str, scratch_dir: Path, live_container: str = "ggtrader_db", root: Path | None = None
) -> None:
    """Reject targets outside the dedicated disposable namespace."""
    root = (root or SCRATCH_ROOT).resolve()
    folder = scratch_dir.resolve()
    if (
        not name.startswith("ggt_restore_drill_")
        or name == live_container
        or root == Path(root.anchor)
        or root.is_relative_to(REPO_ROOT)
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
    disk_usage: Callable = shutil.disk_usage,
) -> int:
    """Restore and compare a dump using an isolated container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--image", default="timescale/timescaledb:latest-pg16")
    parser.add_argument("--sums", type=Path)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="folder for the temporary copy (default: the thesix drive)",
    )
    args = parser.parse_args(argv)
    name = container_name(secrets.token_hex(16))
    root = args.scratch_root or SCRATCH_ROOT
    folder = root / name
    attempted = False
    started = False
    stage = "verify"
    report: dict = {"ok": False, "stage": stage, "error": "UnknownError"}
    folder_created = False
    report_notes: dict = {}
    conn = None
    status = 1
    try:
        refuse_unsafe(name, folder, root=root)
        verify_sha256(args.dump, args.sums or args.dump.parent / "SHA256SUMS")
        record = json.loads(args.counts.read_text())
        version = record["timescaledb"]
        expected = record["after"]
        if not isinstance(version, str) or not version:
            raise ValueError("Missing TimescaleDB version")
        if disk_usage(space_path(root)).free < 3 * args.dump.stat().st_size + 20 * 10**9:
            print("Insufficient scratch space", file=sys.stderr)
            status = 2
            raise ValueError("Insufficient scratch space")
        stage = "start"
        root.mkdir(parents=True, exist_ok=True)
        folder.mkdir(mode=0o700)
        folder_created = True
        password = secrets.token_urlsafe(32)
        attempted = True
        env = os.environ.copy()
        env["POSTGRES_PASSWORD"] = password
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
                "POSTGRES_PASSWORD",
                "-e",
                "POSTGRES_USER=ggtrader",
                "-e",
                "POSTGRES_DB=postgres",
                args.image,
            ],
            check=True,
            capture_output=True,
            env=env,
        )
        started = True
        port_result = runner(
            ["docker", "port", name, "5432/tcp"], check=True, capture_output=True, text=True
        )
        address = port_result.stdout.strip()
        host, port = address.rsplit(":", 1)
        if host != "127.0.0.1" or not port.isdecimal() or not 1 <= int(port) <= 65535:
            raise ValueError("Invalid scratch container port")
        stage = "connect"
        for attempt in range(60):
            try:
                conn = connect(
                    host="127.0.0.1",
                    port=int(port),
                    dbname="postgres",
                    user="ggtrader",
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
            user="ggtrader",
            password=password,
            connect_timeout=2,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
            row = cur.fetchone()
            if row is None or row[0] != version:
                raise ValueError("TimescaleDB extension version mismatch")
            for role in EXTRA_ROLES:
                cur.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role)))
            cur.execute("SELECT timescaledb_pre_restore()")
        conn.close()
        conn = None
        stage = "restore"
        with args.dump.open("rb") as source:
            try:
                restored = runner(
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
                        "ggtrader",
                        "-d",
                        "ggtrader",
                    ],
                    stdin=source,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr or b""
                args.out.with_suffix(".restore.log").write_text(
                    "\n".join(stderr.decode("utf-8", errors="replace").splitlines()[-200:])
                    + ("\n" if stderr else "")
                )
                raise
            stderr = getattr(restored, "stderr", None) or b""
            args.out.with_suffix(".restore.log").write_text(
                "\n".join(stderr.decode("utf-8", errors="replace").splitlines()[-200:])
                + ("\n" if stderr else "")
            )
        stage = "reconnect"
        conn = connect(
            host="127.0.0.1",
            port=int(port),
            dbname="ggtrader",
            user="ggtrader",
            password=password,
            connect_timeout=2,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            stage = "post-restore"
            # The restored catalog keeps the version the database was first created with, which
            # can trail the installed extension after an in-place upgrade; align it so the check
            # inside timescaledb_post_restore compares the code actually running.
            cur.execute(
                "UPDATE _timescaledb_catalog.metadata SET value = %s "
                "WHERE key = 'timescaledb_version' AND value <> %s",
                (version, version),
            )
            report_notes["catalog_version_aligned"] = cur.rowcount == 1
            cur.execute("SELECT timescaledb_post_restore()")
            stage = "analyze"
            cur.execute("ANALYZE")
        stage = "compare"
        report = compare(expected, snapshot(conn))
        report.update(report_notes)
        if report["ok"]:
            status = 0
        else:
            report.update(stage=stage, error="SnapshotMismatch")
    except Exception as exc:
        # Exceptions from subprocess and drivers can contain credentials or connection strings.
        print(f"Restore drill failed ({type(exc).__name__})", file=sys.stderr)
        report = {"ok": False, "stage": stage, "error": type(exc).__name__}
        # The server's own message never carries the password, unlike the driver's full text.
        primary = getattr(getattr(exc, "diag", None), "message_primary", None)
        if isinstance(primary, str) and primary:
            report["server_message"] = primary[:500]
    finally:

        def cleanup_failure(exc: Exception) -> None:
            nonlocal report, status
            if status == 0:
                report = {"ok": False, "stage": "cleanup", "error": type(exc).__name__}
            else:
                report.setdefault("cleanup_error", type(exc).__name__)
            status = 1

        if conn is not None:
            try:
                conn.close()
            except Exception as exc:
                cleanup_failure(exc)
        if not args.keep:
            container_removed = not attempted
            if attempted:
                try:
                    refuse_unsafe(name, folder)
                    runner(["docker", "rm", "-f", name], check=True, capture_output=True)
                    container_removed = True
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr or b""
                    if not started and b"no such container" in stderr.lower():
                        container_removed = True
                    else:
                        cleanup_failure(exc)
                except Exception as exc:
                    cleanup_failure(exc)
            if folder_created and container_removed:
                try:
                    refuse_unsafe(name, folder)
                    try:
                        folder.rmdir()
                    except OSError as exc:
                        if exc.errno not in {
                            errno.ENOTEMPTY,
                            errno.EEXIST,
                            errno.EACCES,
                            errno.EPERM,
                        }:
                            raise
                        runner(
                            [
                                "docker",
                                "run",
                                "--rm",
                                "-u",
                                "0",
                                "-v",
                                f"{folder}:/scratch",
                                "--entrypoint",
                                "sh",
                                args.image,
                                "-c",
                                "rm -rf /scratch/* /scratch/.[!.]* /scratch/..?*",
                            ],
                            check=True,
                            capture_output=True,
                        )
                        folder.rmdir()
                except Exception as exc:
                    cleanup_failure(exc)
        try:
            args.out.write_text(json.dumps(report, indent=2) + "\n")
        except Exception:
            print("Restore drill report write failed", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
