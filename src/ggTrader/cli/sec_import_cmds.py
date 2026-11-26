"""Commands for loading existing SEC research files."""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from psycopg2 import Error as PgError

from ggTrader.data.kaggle_import.guards import BackupStale, assert_connected_database, safe_error
from ggTrader.data.kaggle_import.loadlog import LoadFailed
from ggTrader.data.sec_import.cache import (
    CacheConflict,
    CacheCorrupt,
    load_cache,
    merge_caches,
    scan_cache,
    stray_files,
)
from ggTrader.data.sec_import.headers import load_headers
from ggTrader.data.sec_import.index import load_index, quarter_of
from ggTrader.data.sec_import.labels import (
    LabelConflict,
    LabelMismatch,
    load_labels,
    read_early_labels,
    read_payout_labels,
)
from ggTrader.data.universe_db import connect, to_psycopg_url


def register_sec_parsers(subs: argparse._SubParsersAction) -> list[argparse.ArgumentParser]:
    """Register SEC source and backup arguments."""
    commands = []
    for name in ("sec-cache", "sec-index", "sec-headers", "sec-labels"):
        command = subs.add_parser(name, help=f"Import {name}")
        if name == "sec-labels":
            command.add_argument("--study", choices=("early", "payouts"), required=True)
            command.add_argument("--file", type=Path, required=True)
        else:
            command.add_argument("--folder", type=Path, action="append", required=True)
        if name == "sec-index":
            command.add_argument("--only-quarter", action="append")
        command.add_argument("--backup-dir", type=Path, required=True)
        command.add_argument("--db-free-path", type=Path, default=Path("/"))
        commands.append(command)
    return commands


def _inspect(args: argparse.Namespace) -> Any:
    """Inspect the inputs without writing or opening a database."""
    if args.import_command == "sec-labels":
        reader = read_early_labels if args.study == "early" else read_payout_labels
        rows = reader(args.file)
        print(f"{len(rows)} label rows: {dict(Counter(row.label for row in rows))}")
        return rows
    groups = []
    # Resolved here so the paths recorded by the loaders do not depend on the working folder.
    args.folder = [Path(folder).resolve() for folder in args.folder]
    for folder in args.folder:
        group = scan_cache(folder)
        groups.append(group)
        counts = dict(Counter(entry.kind for entry in group))
        print(f"{folder}: {len(group)} pairs; kinds: {counts}")
        print(f"Other files: {stray_files(folder)}")
    entries = merge_caches(*groups)
    if args.import_command == "sec-index":
        quarters = sorted(quarter_of(e.url) for e in entries if e.kind == "quarterly_index")
        if args.only_quarter is not None:
            missing = sorted(set(args.only_quarter) - set(quarters))
            if missing:
                raise ValueError(f"Requested quarters not in the cache: {', '.join(missing)}")
            quarters = [q for q in quarters if q in args.only_quarter]
        print(f"Quarters: {quarters}")
    return entries


def _load(conn: Any, args: argparse.Namespace, source: Any) -> int:
    """Call the appropriate loader with the inspected source."""
    options = {"backup_dir": args.backup_dir, "db_free_path": args.db_free_path}
    if args.import_command == "sec-cache":
        return load_cache(conn, args.folder, **options)
    if args.import_command == "sec-index":
        return load_index(conn, source, only_quarters=args.only_quarter, **options)
    if args.import_command == "sec-headers":
        return load_headers(conn, source, **options)
    return load_labels(conn, source, source_file=args.file, release_id=1, **options)


def run_sec_import(args: argparse.Namespace, url: str) -> None:
    """Inspect inputs and load only after confirmation."""
    try:
        source = _inspect(args)
        if not args.yes:
            print(f"Would import {args.import_command}; pass --yes to proceed.")
            return
        db_url = to_psycopg_url(url)
        try:
            conn = connect(db_url)
        except Exception:  # noqa: BLE001 — connection errors may contain credentials
            sys.exit("Database connection failed")
        try:
            assert_connected_database(conn, unquote(urlparse(db_url).path.lstrip("/")))
            print(f"Load {_load(conn, args, source)}")
        finally:
            conn.close()
    except LabelMismatch as exc:
        for row in exc.unmatched:
            print(f"Unmatched: PERMNO {row.permno}, last date {row.last_date}, label {row.label}")
        sys.exit("Labels do not match the stock list")
    except (
        CacheCorrupt,
        CacheConflict,
        LabelConflict,
        LoadFailed,
        BackupStale,
        OSError,
        ValueError,
        PgError,
    ) as exc:
        sys.exit(f"SEC import failed: {safe_error(exc)}")
