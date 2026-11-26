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

from ggTrader.utils.db_snapshot import snapshot

THESIX = Path("/media/thesix")
DUMP_PATTERN = re.compile(r"^ggtrader_(\d{8})\.dump$")


def dump_filename(day: date) -> str:
    return f"ggtrader_{day:%Y%m%d}.dump"


def build_dump_command(container: str = "ggtrader_db") -> list[str]:
    return [
        "docker",
        "exec",
        "-u",
        "postgres",
        container,
        "pg_dump",
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
        with part.open("xb") as output:
            runner(build_dump_command(), stdout=output, check=True)
        if part.stat().st_size < 1024 * 1024:
            raise ValueError("Dump is smaller than 1 MiB")
        conn = connect(url.replace("postgresql+psycopg2://", "postgresql://", 1))
        try:
            conn.autocommit = True
            after = snapshot(conn)
        finally:
            conn.close()
        if before != after:
            mismatch = part.with_suffix(".mismatch")
            part.rename(mismatch)
            print(f"Snapshot mismatch: {mismatch}", file=sys.stderr)
            return 3
        with part.open("rb") as dump_file:
            digest = hashlib.file_digest(dump_file, "sha256").hexdigest()
        size = part.stat().st_size
        part.rename(final)
        with (folder / "SHA256SUMS").open("a") as sums:
            sums.write(f"{digest}  {name}\n")
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
        print(f"Dump failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
