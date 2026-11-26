#!/usr/bin/env python3
"""Write a dated database dump and its verification record."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import psycopg2

from ggTrader.data.kaggle_import.guards import safe_error
from ggTrader.utils.db_snapshot import snapshot

THESIX = Path("/media/thesix")
DUMP_PATTERN = re.compile(r"^ggtrader_(\d{8})\.dump$")
#: Tries per day: a dump during which a table changed is retaken once before giving up.
ATTEMPTS = 2
MISMATCH_PATTERN = re.compile(r"^ggtrader_(\d{8})\.dump\.mismatch$")


def dump_filename(day: date) -> str:
    return f"ggtrader_{day:%Y%m%d}.dump"


def build_dump_command(container: str = "ggtrader_db") -> list[str]:
    return [
        "docker",
        "exec",
        container,
        "pg_dump",
        "-U",
        "ggtrader",
        "-Fc",
        "-Z",
        "1",
        "--no-owner",
        "ggtrader",
    ]


def prune(folder: Path, keep: int) -> list[Path]:
    dumps = sorted(p for p in folder.iterdir() if p.is_file() and DUMP_PATTERN.fullmatch(p.name))
    removed = []
    for path in dumps[: -max(1, keep)]:
        day = DUMP_PATTERN.fullmatch(path.name).group(1)
        path.unlink()
        removed.append(path)
        counts = folder / f"counts_{day}.json"
        if counts.is_file():
            counts.unlink()
            removed.append(counts)
    # A discarded dump from an earlier day is no use to anyone; keep only today's for inspection.
    for path in folder.iterdir():
        match = MISMATCH_PATTERN.fullmatch(path.name)
        if path.is_file() and match and match.group(1) != f"{date.today():%Y%m%d}":
            path.unlink()
            removed.append(path)
    return removed


def main(
    argv: list[str] | None = None,
    *,
    connect: Callable = psycopg2.connect,
    runner: Callable = subprocess.run,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--db-url-env", default="GGTRADER_DB_URL")
    args = parser.parse_args(argv)
    folder = args.out_dir.resolve()
    if (
        not folder.is_dir()
        or not folder.is_relative_to(THESIX.resolve())
        or folder == THESIX.resolve()
    ):
        print("Output directory must exist under thesix", file=sys.stderr)
        return 2
    owned_part: Path | None = None
    try:
        today = date.today()
        name = dump_filename(today)
        earlier = sorted(
            p
            for p in folder.iterdir()
            if DUMP_PATTERN.fullmatch(p.name) and p.name < name and p.is_file()
        )
        required = max(30 * 10**9, 2 * earlier[-1].stat().st_size if earlier else 0)
        if shutil.disk_usage(folder).free < required:
            print("Insufficient free space", file=sys.stderr)
            return 2
        final = folder / name
        part = folder / (name + ".part")
        if final.exists() or part.exists() or part.with_suffix(".mismatch").exists():
            raise FileExistsError("A dump or unfinished dump already exists for today")
        url = os.environ.get(args.db_url_env)
        if not url:
            from ggTrader.utils.config import get_db_connection_string

            if args.db_url_env != "GGTRADER_DB_URL":
                raise ValueError(f"Environment variable {args.db_url_env} is not set")
            url = get_db_connection_string()
        conn = connect(url.replace("postgresql+psycopg2://", "postgresql://", 1))
        start = time.monotonic()
        try:
            conn.autocommit = True
            before = snapshot(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                version = cur.fetchone()
                if version is None:
                    raise ValueError("TimescaleDB extension not found")
        finally:
            conn.close()
        for attempt in range(1, ATTEMPTS + 1):
            with part.open("xb") as output:
                owned_part = part
                runner(build_dump_command(), stdout=output, check=True)
            if part.stat().st_size < 1024 * 1024:
                raise ValueError("Dump is smaller than 1 MiB")
            conn = connect(url.replace("postgresql+psycopg2://", "postgresql://", 1))
            try:
                conn.autocommit = True
                after = snapshot(conn)
            finally:
                conn.close()
            if before == after:
                break
            # The dump itself is one consistent snapshot, but the restore drill compares it with
            # the counts taken before it, so a dump taken while something wrote is not usable.
            mismatch = part.with_suffix(".mismatch")
            part.rename(mismatch)
            owned_part = None
            print(f"Snapshot mismatch on attempt {attempt}: {mismatch}", file=sys.stderr)
            if attempt == ATTEMPTS:
                return 3
            mismatch.unlink()
            conn = connect(url.replace("postgresql+psycopg2://", "postgresql://", 1))
            try:
                conn.autocommit = True
                before = snapshot(conn)
            finally:
                conn.close()
        with part.open("rb") as dump_file:
            digest = hashlib.file_digest(dump_file, "sha256").hexdigest()
        size = part.stat().st_size
        # Checksum first: a crash between the two steps leaves a stray checksum line, never a
        # dump the restore drill would refuse for lack of one.
        with (folder / "SHA256SUMS").open("a") as sums:
            sums.write(f"{digest}  {name}\n")
        part.rename(final)
        owned_part = None
        record = {
            "day": today.isoformat(),
            "dump_bytes": size,
            "seconds": time.monotonic() - start,
            "timescaledb": version[0],
            "before": before,
            "after": after,
        }
        (folder / f"counts_{today:%Y%m%d}.json").write_text(json.dumps(record, indent=2) + "\n")
        prune(folder, args.keep)
        return 0
    except Exception as exc:
        if owned_part is not None:
            try:
                owned_part.unlink(missing_ok=True)
            except OSError:
                print("Failed to remove partial dump", file=sys.stderr)
        print(f"Dump failed: {safe_error(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
