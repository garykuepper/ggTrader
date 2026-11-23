#!/usr/bin/env python3
"""Read-only evidence checks for a published combined-data release."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from datetime import date
from pathlib import Path
from unittest.mock import patch

NAMES = (
    "Source row counts",
    "Ten-stock monthly returns",
    "McGill and CRSP returns",
    "Existing Yahoo bars",
    "Named stocks and identity",
    "Database and free space",
    "UTC and Pacific sessions",
    "Indexes and read timings",
    "Link tests",
    "Splice tests",
    "Spell tests",
    "Release tests",
    "Lab entry-point tests",
)
TEST_FILES = {
    9: ["tests/data/kaggle_import/test_links.py", "tests/data/kaggle_import/test_columns.py"],
    10: ["tests/data/kaggle_import/test_reader.py", "tests/lab/test_combined_smoke.py"],
    11: ["tests/data/kaggle_import/test_spells.py", "tests/data/kaggle_import/test_derive.py"],
    12: ["tests/data/kaggle_import/test_release.py", "tests/engine/test_provenance_release.py"],
    13: ["tests/lab/test_cli.py", "tests/lab/test_blend.py", "tests/data/test_bars_combined.py"],
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command line; the database target must be selected explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--db-url-env", metavar="NAME")
    target.add_argument("--use-configured-db", action="store_true")
    parser.add_argument("--release-id", type=int)
    parser.add_argument("--data-volume", type=Path, default=Path("/"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--label")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--save-baseline", type=Path)
    baseline.add_argument("--compare-baseline", type=Path)
    return parser


def default_output(day: date, label: str | None = None) -> Path:
    """A dated report name, with a safe optional label."""
    if label and not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("label must contain only letters, digits, underscores or hyphens")
    return Path("results/kaggle_checks") / f"{day.isoformat()}{'-' + label if label else ''}.json"


def initial_checks() -> list[dict]:
    """List all checks, including the test suites that must be run separately."""
    import ast

    root = Path(__file__).resolve().parents[1]
    checks = []
    for number, name in enumerate(NAMES, 1):
        files = TEST_FILES.get(number, [])
        test_names = []
        for file in files:
            tree = ast.parse((root / file).read_text())
            test_names.extend(
                f"{file}::{node.name}"
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
        checks.append(
            {
                "id": f"check_{number:02}",
                "name": name,
                "status": "covered_by_tests" if files else "skipped",
                "measured": {
                    "test_files": files,
                    "test_names": sorted(test_names),
                    "test_count": len(test_names),
                }
                if files
                else {},
                "tolerance": None,
                "elapsed_seconds": 0.0,
                "reason": "run the listed test suites separately" if files else "not assessed",
            }
        )
    return checks


def connect_target(url: str):
    """Open the same server-enforced read-only connection used by the data layer."""
    from ggTrader.data.bars import connect_readonly

    return connect_readonly(url)


def _symbols(conn, release_id: int) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol FROM market.stock_spans WHERE release_id=%s ORDER BY symbol",
            (release_id,),
        )
        return [row[0] for row in cur.fetchall()]


def _load(url: str, names: list[str], source: str, release_id: int | None = None, *, start=None):
    import ggTrader.data.bars as bars

    # Every connection made by load_bars receives the explicit target, without changing config.
    original = bars.connect_readonly
    with patch.object(bars, "connect_readonly", lambda: original(url)):
        return bars.load_bars(names, start=start, source=source, release_id=release_id)


def _set(check: dict, status: str, measured: dict, tolerance, reason: str = "") -> None:
    check.update(status=status, measured=measured, tolerance=tolerance, reason=reason)


def _baseline(conn, url: str, *, save: Path | None = None, compare: Path | None = None) -> dict:
    """Fingerprint all Yahoo symbols in fixed batches through the normal lab reader."""
    import json

    from ggTrader.utils.io import write_json

    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT symbol FROM public.ohlcv "
            "WHERE venue='yfinance' AND interval='1d' ORDER BY symbol"
        )
        names = [row[0] for row in cur.fetchall()]
    batches = []
    for offset in range(0, len(names), 120):
        symbols = names[offset : offset + 120]
        bars = _load(url, symbols, "db")
        batches.append(
            {
                "symbols": symbols,
                "shape": list(bars.close.shape),
                "bars_sha256": bars.provenance["bars_sha256"],
            }
        )
    baseline = {"source": "db", "batches": batches}
    if save is not None:
        write_json(save, baseline)
    return {
        "batches": len(batches),
        "symbols": len(names),
        "matches": baseline == json.loads(compare.read_text()) if compare else None,
    }


def _check_counts(conn, release_id: int) -> tuple[str, dict, str]:
    """Reconcile load records, database rows, chunk counts and Parquet metadata."""
    import pyarrow.parquet as pq

    with conn.cursor() as cur:
        cur.execute(
            "SELECT l.load_id,l.file_path,l.checks FROM ref.release_loads r "
            "JOIN ref.loads l ON l.load_id=r.load_id WHERE r.release_id=%s AND r.role='crsp-bars'",
            (release_id,),
        )
        derived = cur.fetchone()
        if not derived or not (derived[1] or "").startswith("load:"):
            return "skipped", {}, "derived load does not identify its source load"
        source_id = int(derived[1].split(":", 1)[1])
        cur.execute(
            "SELECT role,load_id FROM ref.release_loads WHERE release_id=%s AND role='mcgill'",
            (release_id,),
        )
        sources = [("crsp", source_id)] + [(role, load) for role, load in cur.fetchall()]
        measured = {}
        complete = True
        matches = True
        for kind, load in sources:
            cur.execute(
                "SELECT file_path,rows_parsed,rows_accepted,rows_quarantined "
                "FROM ref.loads WHERE load_id=%s",
                (load,),
            )
            record = cur.fetchone()
            if record is None:
                return "fail", measured, f"{kind} source load is missing"
            path, parsed, accepted, quarantined = record
            table = "kaggle_crsp.daily" if kind == "crsp" else "kaggle_chars.mcgill"
            cur.execute(f"SELECT count(*) FROM {table} WHERE load_id=%s", (load,))
            db_rows = cur.fetchone()[0]
            cur.execute(
                "SELECT coalesce(sum(rows),0) FROM ref.load_chunks WHERE load_id=%s", (load,)
            )
            chunk_rows = cur.fetchone()[0]
            item = {
                "load_id": load,
                "parsed": parsed,
                "accepted": accepted,
                "quarantined": quarantined,
                "database_rows": db_rows,
                "chunk_rows": chunk_rows,
            }
            if kind == "crsp":
                if path and Path(path).is_file():
                    item["parquet_rows"] = pq.ParquetFile(path).metadata.num_rows
                else:
                    item["parquet_rows"] = None
                    complete = False
            measured[kind] = item
            if None in (parsed, accepted, quarantined):
                complete = False
            else:
                matches &= parsed == accepted + quarantined and accepted == db_rows == chunk_rows
                if kind == "crsp" and item["parquet_rows"] is not None:
                    matches &= parsed == item["parquet_rows"]
        if not any(kind == "mcgill" for kind, _ in sources):
            complete = False
        if derived[2] and "source_rows" in derived[2]:
            measured["derived_source_rows"] = derived[2]["source_rows"]
            if measured["crsp"]["accepted"] is not None:
                matches &= derived[2]["source_rows"] == measured["crsp"]["accepted"]
        return (
            ("fail" if not matches else "pass" if complete else "skipped"),
            measured,
            ("" if complete else "missing source metadata or McGill release load"),
        )


def run_checks(
    url: str,
    release_id: int | None,
    data_volume: Path,
    *,
    save_baseline: Path | None = None,
    compare_baseline: Path | None = None,
) -> dict:
    """Measure implemented checks; label unimplemented evidence rather than guessing."""
    import random

    checks = initial_checks()
    conn = connect_target(url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT release_id FROM ref.releases WHERE status='published' "
                "AND (%s IS NULL OR release_id=%s) "
                "ORDER BY created_at DESC, release_id DESC LIMIT 1",
                (release_id, release_id),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError("no requested published release is available")
            release_id = row[0]
            cur.execute("SELECT pg_database_size(current_database())")
            db_size = cur.fetchone()[0]
        symbols = _symbols(conn, release_id)
        started = time.perf_counter()
        status, measured, reason = _check_counts(conn, release_id)
        _set(checks[0], status, measured, "exact row counts", reason)
        checks[0]["elapsed_seconds"] = time.perf_counter() - started
        _set(
            checks[1],
            "skipped",
            {},
            {"correlation_min": 0.9998},
            "combined linked columns expose CRSP only before Yahoo starts; "
            "the requested 2005–2024 comparison cannot use them",
        )
        _set(
            checks[2],
            "skipped",
            {},
            {"correlation_min": 0.9999},
            "McGill versus CRSP comparison is not implemented",
        )
        if save_baseline or compare_baseline:
            baseline = _baseline(conn, url, save=save_baseline, compare=compare_baseline)
            status = "fail" if baseline["matches"] is False else "skipped"
            reason = "Yahoo key comparison not implemented"
            if save_baseline:
                reason += "; baseline saved but not compared"
            _set(checks[3], status, baseline, "identical batch hashes and Yahoo keys", reason)
        else:
            _set(
                checks[3], "skipped", {}, None, "baseline and Yahoo key comparison not implemented"
            )
        _set(
            checks[4],
            "skipped",
            {},
            None,
            "named-stock traces and double-name audit not implemented",
        )
        started = time.perf_counter()
        free = shutil.disk_usage(data_volume).free
        with conn.cursor() as cur:
            cur.execute(
                "SELECT hypertable_schema,hypertable_name FROM timescaledb_information.hypertables "
                "WHERE (hypertable_schema,hypertable_name) IN "
                "(('public','ohlcv'),('market','daily_bars'),('kaggle_crsp','daily'))"
            )
            names = cur.fetchall()
            sizes = {}
            for schema, table in names:
                cur.execute("SELECT hypertable_size(%s::regclass)", (f"{schema}.{table}",))
                sizes[f"{schema}.{table}"] = cur.fetchone()[0]
        _set(
            checks[5],
            "pass" if free >= 40_000_000_000 else "fail",
            {"database_bytes": db_size, "free_bytes": free, "hypertable_bytes": sizes},
            {"minimum_free_bytes": 40_000_000_000},
        )
        checks[5]["elapsed_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        sample = random.Random(42).sample(symbols, min(20, len(symbols)))
        if sample:
            prior = os.environ.get("PGTZ")
            try:
                os.environ["PGTZ"] = "UTC"
                utc = _load(url, sample, "combined", release_id, start="2000-01-03")
                os.environ["PGTZ"] = "America/Los_Angeles"
                pacific = _load(url, sample, "combined", release_id, start="2000-01-03")
            finally:
                if prior is None:
                    os.environ.pop("PGTZ", None)
                else:
                    os.environ["PGTZ"] = prior
            a, b = utc.provenance["bars_sha256"], pacific.provenance["bars_sha256"]
            _set(
                checks[6],
                "skipped" if utc.close.empty else "pass" if a == b else "fail",
                {"symbols": sample, "utc_hash": a, "pacific_hash": b},
                "hashes equal",
            )
        else:
            _set(checks[6], "skipped", {}, "hashes equal", "release has no symbols")
        checks[6]["elapsed_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT schemaname,tablename,indexname,indexdef FROM pg_indexes "
                "WHERE (schemaname,tablename) IN "
                "(('ref','release_loads'),('ref','symbol_map'),('market','daily_bars'),"
                "('kaggle_crsp','daily')) ORDER BY schemaname,tablename,indexname"
            )
            indexes = [
                dict(zip(("schema", "table", "name", "definition"), row, strict=True))
                for row in cur.fetchall()
            ]
            cur.execute(
                "SELECT symbol FROM market.stock_spans WHERE release_id=%s "
                "AND kind IN ('yahoo','linked_yahoo') ORDER BY symbol",
                (release_id,),
            )
            yahoo = [row[0] for row in cur.fetchall()]
        index_ok = any(index["name"] == "release_loads_role_unique" for index in indexes)
        bars_key = any(
            index["table"] == "daily_bars"
            and all(col in index["definition"].lower() for col in ("load_id", "permno", "date"))
            for index in indexes
        )
        symbol_key = any(
            index["table"] == "symbol_map"
            and "symbol" in index["definition"].lower()
            and (
                "valid_from" in index["definition"].lower()
                or "valid_to" in index["definition"].lower()
            )
            for index in indexes
        )
        index_ok = index_ok and bars_key and symbol_key
        measurements = {
            "indexes": indexes,
            "available_yahoo_symbols": len(yahoo),
            "required_indexes_present": index_ok,
        }
        if len(yahoo) >= 100:
            picked = random.Random(42).sample(yahoo, 100)
            for label, source, names in (
                ("combined_one", "combined", picked[:1]),
                ("combined_hundred", "combined", picked),
                ("db_one", "db", picked[:1]),
                ("db_hundred", "db", picked),
            ):
                start = time.perf_counter()
                _load(url, names, source, release_id if source == "combined" else None)
                measurements[label + "_seconds"] = time.perf_counter() - start
            good = (
                index_ok
                and measurements["combined_one_seconds"] < 5
                and measurements["combined_hundred_seconds"] < 60
                and measurements["db_one_seconds"] < 5
                and measurements["db_hundred_seconds"] < 60
            )
            _set(
                checks[7],
                "pass" if good else "fail",
                measurements,
                {"combined_one_seconds_lt": 5, "combined_hundred_seconds_lt": 60},
            )
        else:
            _set(
                checks[7],
                "skipped" if index_ok else "fail",
                measurements,
                {"combined_one_seconds_lt": 5, "combined_hundred_seconds_lt": 60},
                "fewer than 100 Yahoo symbols",
            )
        checks[7]["elapsed_seconds"] = time.perf_counter() - started
        return {
            "release_id": release_id,
            "database_bytes": db_size,
            "free_bytes": free,
            "checks": checks,
        }
    finally:
        conn.close()


def main() -> int:
    """Run the checks and save a report without printing the connection string."""
    args = build_parser().parse_args()
    from ggTrader.utils.io import write_json

    if args.use_configured_db:
        from ggTrader.utils.config import get_db_connection_string

        url = get_db_connection_string()
    else:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.db_url_env):
            raise SystemExit("invalid database environment variable name")
        url = os.environ.get(args.db_url_env)
        if not url:
            raise SystemExit("database environment variable is unset")
    try:
        report = run_checks(
            url,
            args.release_id,
            args.data_volume,
            save_baseline=args.save_baseline,
            compare_baseline=args.compare_baseline,
        )
    except Exception:
        raise SystemExit("Closing checks could not read the selected database") from None
    out = args.out or default_output(date.today(), args.label)
    write_json(out, report)
    for item in report["checks"]:
        print(f"{item['id']}  {item['status']:<16} {item['name']}")
    print(f"Report: {out}")
    return 1 if any(item["status"] in ("fail", "skipped") for item in report["checks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
