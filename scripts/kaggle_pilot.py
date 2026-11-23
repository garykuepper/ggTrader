#!/usr/bin/env python3
"""Measure CRSP loading and compression in a disposable database."""

import argparse
import json
import resource
import shutil
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg2
import pyarrow.compute as pc
import pyarrow.parquet as pq
from psycopg2 import sql

from ggTrader.data.kaggle_import.crsp import compress_kaggle_chunks, load_crsp
from ggTrader.data.kaggle_import.ddl import apply_ddl

DATA = Path("/media/thesix/ggTrader-data/market-data/crsp-kaggle")
MARGIN = 40 * 1024**3


def rows_by_year(path: Path) -> dict[int, int]:
    """Count years without reading other columns or the whole file."""
    parquet = pq.ParquetFile(path)
    counts: dict[int, int] = {}
    for i in range(parquet.num_row_groups):
        values = pc.value_counts(parquet.read_row_group(i, columns=["year"]).column("year"))
        for entry in values.to_pylist():
            if entry["values"] is None:
                raise ValueError("Null year in input")
            year = int(entry["values"])
            counts[year] = counts.get(year, 0) + entry["counts"]
    return counts


def project(  # noqa: PLR0913 - explicit measurements are the public projection inputs
    rows_by_year: dict[int, int],
    year: int,
    bytes_uncompressed: int,
    bytes_compressed: int,
    rows_loaded: int,
    free_before: int,
    margin: int = MARGIN,
) -> dict:
    """Scale measured table sizes to the full file and estimate reload space."""
    if rows_loaded <= 0 or year not in rows_by_year:
        raise ValueError("A measured year and positive row count are required")
    total = sum(rows_by_year.values())
    uncompressed = bytes_uncompressed * total // rows_loaded
    compressed = bytes_compressed * total // rows_loaded
    peak = (compressed + uncompressed) * 11 // 10
    full_free = free_before - compressed
    peak_free = free_before - peak
    return dict(
        full_uncompressed_bytes=uncompressed,
        full_compressed_bytes=compressed,
        reload_peak_bytes=peak,
        free_after_full_load=full_free,
        free_after_reload_peak=peak_free,
        margin_ok=full_free >= margin and peak_free >= margin,
    )


def measure(conn, started: float) -> dict:
    """Record the table size, rows and local process and disk measurements."""
    with conn.cursor() as cur:
        cur.execute("SELECT hypertable_size('kaggle_crsp.daily'), count(*) FROM kaggle_crsp.daily")
        size, count = cur.fetchone()
    conn.rollback()
    return dict(
        hypertable_bytes=size,
        rows=count,
        free_bytes=shutil.disk_usage("/").free,
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )


def run(args: argparse.Namespace) -> dict:
    """Run the four measurements and always remove the disposable database."""
    counts = rows_by_year(args.file)
    year = args.year if args.year is not None else max(counts, key=counts.get)
    if year not in counts:
        raise ValueError("Requested year is absent")
    free = shutil.disk_usage("/").free
    name = "ggt_kaggle_pilot_" + uuid.uuid4().hex[:8]
    admin = psycopg2.connect(args.scratch_url)
    admin.autocommit = True
    created = False
    try:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created = True
        target = urlunparse(urlparse(args.scratch_url)._replace(path="/" + name))
        conn = psycopg2.connect(target)
        try:
            apply_ddl(conn)
            steps = {}
            for label, action in (
                (
                    "A",
                    lambda: load_crsp(
                        conn,
                        args.file,
                        args.sums,
                        backup_dir=None,
                        year=year,
                        db_free_path=Path("/"),
                    ),
                ),
                ("B", lambda: compress_kaggle_chunks(conn)),
                (
                    "C",
                    lambda: load_crsp(
                        conn,
                        args.file,
                        args.sums,
                        backup_dir=None,
                        year=year,
                        parser_suffix="v2",
                        db_free_path=Path("/"),
                    ),
                ),
                ("D", lambda: compress_kaggle_chunks(conn)),
            ):
                start = time.monotonic()
                action()
                steps[label] = measure(conn, start)
        finally:
            conn.close()
    finally:
        try:
            if created:
                with admin.cursor() as cur:
                    cur.execute(
                        sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                            sql.Identifier(name)
                        )
                    )
        finally:
            admin.close()
    inputs = dict(
        rows_by_year=counts,
        year=year,
        bytes_uncompressed=steps["A"]["hypertable_bytes"],
        bytes_compressed=steps["B"]["hypertable_bytes"],
        rows_loaded=steps["A"]["rows"],
        free_before=free,
        margin=MARGIN,
    )
    return dict(inputs=inputs, steps=steps, projection=project(**inputs))


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and write the pilot report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int)
    parser.add_argument("--scratch-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sums", type=Path, default=DATA / "SHA256SUMS.txt")
    parser.add_argument(
        "--file", type=Path, default=DATA / "cleaned/crsp_1968_2024_daily_cleaned.parquet"
    )
    args = parser.parse_args(argv)
    try:
        report = run(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001 - avoid exposing database credentials
        print(f"Pilot failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
