#!/usr/bin/env python3
"""Read-only evidence checks for a published combined-data release."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
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
READ_TIMING_TOLERANCE = {
    "combined_one_seconds_lt": 10,
    "combined_hundred_seconds_lt": 60,
    "db_one_seconds_lt": 10,
    "db_hundred_seconds_lt": 60,
}
BASELINE_REQUIRED_REASON = (
    "no baseline given: take one with --save-baseline on the target BEFORE "
    "the first load, compare after"
)


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
    parser.add_argument(
        "--nested-plans", type=Path, help="JSON containing captured Yahoo and CRSP nested plans"
    )
    parser.add_argument(
        "--allow-synthetic-skips",
        action="store_true",
        help="Allow checks without sufficient synthetic sample data to be skipped.",
    )
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--save-baseline", type=Path)
    baseline.add_argument("--compare-baseline", type=Path)
    parser.add_argument("--baseline-origin", help="Describe where and how the baseline was taken.")
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
    from urllib.parse import unquote, urlsplit

    from ggTrader.data.bars import connect_readonly
    from ggTrader.data.kaggle_import.guards import assert_connected_database, checked_pg_url

    url = checked_pg_url(url.replace("postgresql+psycopg2://", "postgresql://", 1))
    conn = connect_readonly(url)
    try:
        assert_connected_database(conn, unquote(urlsplit(url).path.lstrip("/")))
        return conn
    except Exception:
        conn.close()
        raise


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
    from urllib.parse import unquote, urlsplit

    from ggTrader.data.kaggle_import.guards import assert_connected_database, checked_pg_url

    url = checked_pg_url(url.replace("postgresql+psycopg2://", "postgresql://", 1))

    def checked_connection():
        conn = original(url)
        try:
            assert_connected_database(conn, unquote(urlsplit(url).path.lstrip("/")))
            return conn
        except Exception:
            conn.close()
            raise

    with patch.object(bars, "connect_readonly", checked_connection):
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


def _check_yahoo_keys(conn, url: str, release_id: int, names: list[str]) -> tuple[str, dict, str]:
    """Compare Yahoo rows and ensure all CRSP additions precede their Yahoo splice."""
    import numpy as np
    import pandas as pd

    from ggTrader.data.bars import BAR_FIELDS

    combined = _load(url, names, "combined", release_id)
    db = _load(url, names, "db")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT timestamp,symbol FROM public.ohlcv WHERE symbol=ANY(%s) "
            "AND venue='yfinance' AND interval='1d'",
            (names,),
        )
        db_keys = {(pd.Timestamp(stamp, tz="UTC"), symbol) for stamp, symbol in cur.fetchall()}
        cur.execute(
            "SELECT symbol,first_yf_date FROM market.stock_spans "
            "WHERE release_id=%s AND symbol=ANY(%s)",
            (release_id, names),
        )
        first_yahoo = dict(cur.fetchall())
    cutoff = pd.Timestamp("2000-01-03", tz="UTC")
    combo_keys = set()
    before_2000 = after_2000 = 0
    invalid_crsp = []
    invalid_source = []
    for (stamp, symbol), source in combined.flags["source"].items():
        symbol = str(symbol)
        if source == "yfinance":
            combo_keys.add((stamp, symbol))
        elif source == "crsp":
            if stamp < cutoff:
                before_2000 += 1
            else:
                after_2000 += 1
            first = first_yahoo.get(symbol)
            if first is None or stamp.date() >= first:
                invalid_crsp.append({"symbol": symbol, "date": str(stamp.date())})
        else:
            invalid_source.append(
                {"symbol": symbol, "date": str(stamp.date()), "source": str(source)}
            )
    missing, extra = db_keys - combo_keys, combo_keys - db_keys
    different = []
    for symbol in names:
        dates = sorted(ts for ts, ticker in db_keys & combo_keys if ticker == symbol)
        if not dates:
            continue
        for field in BAR_FIELDS:
            a = combined.frame(field).reindex(index=dates, columns=[symbol]).to_numpy(dtype=float)
            b = db.frame(field).reindex(index=dates, columns=[symbol]).to_numpy(dtype=float)
            if not np.array_equal(a, b, equal_nan=True):
                different.append({"symbol": symbol, "field": field})
    measured = {
        "symbols": names,
        "compared_rows": len(db_keys & combo_keys),
        "missing_rows": len(missing),
        "unexpected_rows": len(extra),
        "crsp_rows_before_2000": before_2000,
        "crsp_rows_from_2000": after_2000,
        "invalid_crsp_rows": len(invalid_crsp),
        "invalid_crsp_examples": invalid_crsp[:50],
        "invalid_source_rows": len(invalid_source),
        "invalid_source_examples": invalid_source[:50],
        "different_fields": different,
    }
    return (
        ("fail" if missing or extra or different or invalid_crsp or invalid_source else "pass"),
        measured,
        "",
    )


def duplicate_stock_days(flags, identities: dict[str, int]) -> list[dict]:
    """Report stock-days read under multiple CRSP-backed names."""
    import pandas as pd

    if flags.empty:
        return []
    frame = flags.index.to_frame(index=False)
    frame["permno"] = frame["symbol"].map(identities)
    frame = frame.dropna(subset=["permno"])
    frame = frame.loc[frame.duplicated(["timestamp", "permno"], keep=False)]
    result = []
    for (day, permno), rows in frame.groupby(["timestamp", "permno"], sort=False):
        if rows.symbol.nunique() > 1:
            result.append(
                {
                    "date": str(pd.Timestamp(day).date()),
                    "permno": int(permno),
                    "symbols": sorted(rows.symbol.astype(str).unique().tolist()),
                }
            )
    return result


def classify_identity_records(records):
    """Classify candidate matches separately from overlapping CRSP-backed series."""
    by_permno = {}
    for row in records:
        symbol, permno, kind, first, last, flagged = row[:6]
        if permno is not None:
            by_permno.setdefault(permno, []).append((symbol, kind, first, last, bool(flagged)))
    overlap_groups, flag_issues, different_companies = [], [], []
    checked_pairs = 0
    for permno, rows in by_permno.items():
        backed = [row for row in rows if row[1] in ("linked_yahoo", "placeholder", "spell")]
        for symbol, kind, _, _, flagged in rows:
            if kind != "yahoo" or not backed:
                continue
            linked = sorted(
                row[0] for row in backed if row[1] == "linked_yahoo" and row[0] != symbol
            )
            if linked:
                different_companies.append(
                    {"yahoo_symbol": symbol, "linked_symbols": linked, "permno": permno}
                )
                continue
            for name, _, _, _, crsp_flagged in backed:
                checked_pairs += 1
                if not flagged or not crsp_flagged:
                    flag_issues.append(
                        {"yahoo_symbol": symbol, "crsp_symbol": name, "permno": permno}
                    )
        if len(backed) > 1:
            overlap_groups.append((permno, [row[0] for row in backed]))
    return overlap_groups, flag_issues, different_companies, checked_pairs


def pack_identity_groups(groups, *, limit: int = 120):
    """Pack complete identity groups into bounded symbol batches."""
    batch = []
    size = 0
    for group in groups:
        count = len(group[1])
        if count > limit:
            raise ValueError(f"identity has more than {limit} columns")
        if batch and size + count > limit:
            yield batch
            batch, size = [], 0
        batch.append(group)
        size += count
    if batch:
        yield batch


def read_identity_overlaps(url: str, release_id: int, groups) -> list[dict]:
    """Read packed groups and report every shared CRSP-backed stock-day."""
    violations = []
    for batch in pack_identity_groups(groups):
        names = [name for _, columns in batch for name in columns]
        batch_bars = _load(url, names, "combined", release_id)
        identities = {name: permno for permno, columns in batch for name in columns}
        violations.extend(duplicate_stock_days(batch_bars.flags, identities))
    return violations


def validate_named_frame(bars, span: dict) -> dict:
    """Check one named column against its published span and splice boundaries."""
    import pandas as pd

    name = span["symbol"]
    subset = bars.flags[bars.flags.index.get_level_values("symbol") == name]
    dates = subset.index.get_level_values("timestamp")
    source = subset["source"].astype(str)
    crsp = dates[source == "crsp"]
    yahoo = dates[source == "yfinance"]
    errors = []
    if subset.empty:
        errors.append("no rows")
    else:
        if span["first_date"] is None or dates.min().date() != span["first_date"]:
            errors.append("first date differs from span")
        if span["last_date"] is None or dates.max().date() != span["last_date"]:
            errors.append("last date differs from span")
        if not subset.possible_double.eq(bool(span["possible_double"])).all():
            errors.append("possible_double differs from span")
    kind = span["kind"]
    if kind in ("yahoo", "linked_yahoo"):
        if span["first_yf_date"] is None or len(yahoo) == 0:
            errors.append("Yahoo start date or rows are missing")
        elif yahoo.min().date() != span["first_yf_date"]:
            errors.append("first Yahoo date differs from span")
    if kind == "linked_yahoo":
        first = span["first_yf_date"]
        end = span["crsp_until"]
        if first is None or end is None:
            errors.append("linked span has no splice date")
        elif any(ts.date() >= first or ts.date() >= end for ts in crsp) or any(
            ts.date() < first for ts in yahoo
        ):
            errors.append("rows cross the splice boundary")
        if set(source) - {"crsp", "yfinance"}:
            errors.append("unknown row source")
    elif kind == "yahoo":
        if set(source) - {"yfinance"}:
            errors.append("Yahoo column contains non-Yahoo rows")
    elif kind in ("placeholder", "spell"):
        if set(source) - {"crsp"}:
            errors.append("CRSP column contains non-CRSP rows")
    else:
        errors.append("unknown span kind")
    result = {
        "status": "fail" if errors else "pass",
        "errors": errors,
        "kind": kind,
        "possible_double": bool(span["possible_double"]),
        "rows": len(subset),
        "first": str(dates.min().date()) if len(dates) else None,
        "last": str(dates.max().date()) if len(dates) else None,
        "first_yf_date": str(span["first_yf_date"]) if span["first_yf_date"] else None,
        "crsp_until": str(span["crsp_until"]) if span["crsp_until"] else None,
        "crsp_first": str(crsp.min().date()) if len(crsp) else None,
        "crsp_last": str(crsp.max().date()) if len(crsp) else None,
        "sources": source.value_counts().to_dict(),
    }
    if kind == "linked_yahoo" and len(crsp) and len(yahoo):
        close = bars.close[name]
        before, after = close.loc[crsp.max()], close.loc[yahoo.min()]
        if pd.isna(before) or pd.isna(after) or before == 0:
            errors.append("join prices missing or zero")
        else:
            result["join_return"] = float(after / before - 1)
            if abs(result["join_return"]) > 0.5:
                errors.append("join return exceeds 50%")
        if errors:
            result["status"] = "fail"
    return result


def _check_named_stocks(conn, url: str, release_id: int) -> tuple[str, dict, str]:
    """Inspect named traces and check overlapping identity columns in bounded batches."""
    import pandas as pd

    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol,permno,kind,first_date,last_date,possible_double, "
            "crsp_until,first_yf_date "
            "FROM market.stock_spans WHERE release_id=%s ORDER BY symbol",
            (release_id,),
        )
        records = cur.fetchall()
        cur.execute(
            "SELECT DISTINCT permno FROM ref.symbol_map WHERE release_id=%s "
            "AND symbol='GM' AND permno IS NOT NULL ORDER BY permno",
            (release_id,),
        )
        gm_permnos = [row[0] for row in cur.fetchall()]
    catalog = {row[0]: row for row in records}
    cases = {}
    names = [
        "AAPL",
        "TTWO",
        "CMVT",
        "SMCI",
        "IBKR",
        "VC",
        "PERMNO-10128",
        "GM",
        "PERMNO-10128~1",
        "PERMNO-88319~1",
    ]
    related = {
        row[0]
        for row in records
        if row[0].startswith("PERMNO-10128~")
        or row[0].startswith("PERMNO-88319")
        or row[0] == "PERMNO-12079"
    }
    related.update(row[0] for row in records if row[1] in {*gm_permnos, 12079, 12369})
    selected = [name for name in names if name in catalog] + sorted(related)
    selected = list(dict.fromkeys(selected))
    for name in names:
        if name not in catalog:
            cases[name] = {"status": "skipped", "reason": "name not in release"}
    for offset in range(0, len(selected), 120):
        batch = selected[offset : offset + 120]
        bars = _load(url, batch, "combined", release_id)
        for name in batch:
            row = catalog[name]
            span = dict(
                zip(
                    (
                        "symbol",
                        "permno",
                        "kind",
                        "first_date",
                        "last_date",
                        "possible_double",
                        "crsp_until",
                        "first_yf_date",
                    ),
                    row,
                    strict=True,
                )
            )
            cases[name] = validate_named_frame(bars, span)
    placeholders = [
        row
        for row in records
        if row[2] == "placeholder"
        and row[3] is not None
        and row[3] < pd.Timestamp("2000-01-01").date()
    ]
    if placeholders:
        placeholder = placeholders[0][0]
        historic = _load(url, [placeholder], "combined", release_id, start="1900-01-01")
        pre2000 = int(
            historic.close.loc[historic.close.index < pd.Timestamp("2000-01-01", tz="UTC")]
            .notna()
            .sum()
            .sum()
        )
    else:
        placeholder, pre2000 = None, 0
    requested_missing = ["PERMNO-84761~7", "PERMNO-1~7"]
    if "AAPL" in catalog and catalog["AAPL"][2] == "linked_yahoo":
        requested_missing.append("PERMNO-14593")
    if "TTWO" in catalog and catalog["TTWO"][2] == "linked_yahoo":
        requested_missing.extend(["PERMNO-84761", "PERMNO-84761~1"])
    missing = _load(url, requested_missing, "combined", release_id)
    missing_names = list(missing.missing_symbols)
    candidates, flag_issues, different_companies, checked_pairs = classify_identity_records(records)
    try:
        violations = read_identity_overlaps(url, release_id, candidates)
    except ValueError as exc:
        if "identity has more than 120 columns" not in str(exc):
            raise
        return "fail", {"cases": cases}, str(exc)
    measured = {
        "cases": cases,
        "placeholder": placeholder,
        "pre_2000_rows": pre2000,
        "gm_permnos": gm_permnos,
        "missing_symbols": missing_names,
        "overlap_groups": len(candidates),
        "crsp_backed_overlapping_stock_days": len(violations),
        "crsp_backed_overlap_examples": violations[:10],
        "yahoo_candidates_missing_double_flags": len(flag_issues),
        "yahoo_candidate_and_crsp_pairs_checked": checked_pairs,
        "yahoo_candidate_and_crsp_pairs_both_flagged": checked_pairs - len(flag_issues),
        "yahoo_candidate_flag_exceptions": flag_issues,
        "yahoo_candidates_matched_to_different_linked_company": len(different_companies),
        "different_company_candidate_examples": different_companies[:50],
    }
    errors = []
    if set(missing_names) != set(requested_missing):
        errors.append("an unavailable placeholder or spell was returned")
    if pre2000 == 0:
        errors.append("no pre-2000 placeholder bars")
    aapl = cases.get("AAPL", {})
    if aapl.get("status") != "skipped" and aapl.get("kind") == "linked_yahoo":
        if (
            aapl.get("crsp_first") != "1980-12-15"
            or (aapl.get("crsp_last") or "9999") > "1999-12-31"
        ):
            errors.append("AAPL CRSP dates differ from the fixed trace")
    ttwo = cases.get("TTWO", {})
    if ttwo.get("kind") == "linked_yahoo" and (ttwo.get("crsp_last") or "9999") > "1999-12-31":
        errors.append("TTWO CRSP history extends past 1999")
    vc = cases.get("VC", {})
    if vc.get("status") != "skipped":
        if vc.get("kind") != "yahoo" or vc.get("first") != "2010-10-05":
            errors.append("VC Yahoo trace differs")
        old = cases.get("PERMNO-88319~1")
        if old is None:
            cases["PERMNO-88319~1"] = {"status": "skipped", "reason": "name not in release"}
        elif old.get("status") != "skipped" and (old.get("last") or "9999") > "2009-03-05":
            errors.append("VC earlier spell ends after 2009-03-05")
    if "GM" in catalog and catalog["GM"][1] not in (None, 12369):
        errors.append("GM names the wrong stock")
    gm_columns = {row[1]: row[0] for row in records if row[1] in (12079, 12369)}
    measured["gm_columns"] = gm_columns
    if len(gm_columns) == 2 and len(set(gm_columns.values())) != 2:
        errors.append("old and new GM share a column")
    if 12079 in gm_columns and not any(
        row[1] == 12079 and row[2] == "placeholder" and row[0].startswith("PERMNO-12079")
        for row in records
    ):
        errors.append("old GM is not a placeholder")
    if len(gm_columns) < 2:
        cases["GM_pair"] = {"status": "skipped", "reason": "name not in release"}
    spell_names = [name for name in selected if catalog[name][2] == "spell"]
    for name in selected:
        if name.startswith("PERMNO-10128~") and catalog[name][2] != "spell":
            errors.append(f"{name} is not a CRSP spell")
    measured["trace_errors"] = errors
    if (
        violations
        or flag_issues
        or errors
        or any(case["status"] == "fail" for case in cases.values())
    ):
        return "fail", measured, "a named trace or duplicate-stock check failed"
    if "AAPL" not in catalog or not spell_names:
        return "skipped", measured, "name not in release"
    return "pass", measured, ""


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
            chunk_rows = int(cur.fetchone()[0])
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


TEN_STOCKS = {
    "AAPL": 14593,
    "MSFT": 10107,
    "IBM": 12490,
    "GE": 12060,
    "XOM": 11850,
    "KO": 11308,
    "WMT": 55976,
    "INTC": 59328,
    "PG": 18163,
    "JPM": 47896,
}


def return_statistics(left, right) -> dict:
    """Correlation and absolute differences in percentage points for paired returns."""
    import numpy as np

    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2:
        return {
            "count": len(a),
            "correlation": None,
            "mean_abs_points": None,
            "max_abs_points": None,
        }
    difference = np.abs(a - b) * 100
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
    return {
        "count": len(a),
        "correlation": correlation,
        "mean_abs_points": float(difference.mean()),
        "max_abs_points": float(difference.max()),
    }


def _release_loads(conn, release_id: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.role,l.load_id,l.file_path FROM ref.release_loads r "
            "JOIN ref.loads l ON l.load_id=r.load_id "
            "WHERE r.release_id=%s",
            (release_id,),
        )
        return {role: (load, path) for role, load, path in cur.fetchall()}


def select_timezone_symbols(placeholders: list[str], all_symbols: list[str]) -> list[str]:
    """Choose five pre-2000 placeholders and fifteen distinct other columns."""
    import random

    rng = random.Random(42)
    old = sorted(set(placeholders))
    if len(old) < 5:
        return []
    selected = rng.sample(old, 5)
    remaining = sorted(set(all_symbols) - set(selected))
    if len(remaining) < 15:
        return []
    return selected + rng.sample(remaining, 15)


def symbol_filter_in_plan(plan) -> bool:
    """Find a symbol condition on a scan node, including decompression scans."""
    if isinstance(plan, list):
        return any(symbol_filter_in_plan(item) for item in plan)
    if not isinstance(plan, dict):
        return False
    scan = "scan" in str(plan.get("Node Type", "")).lower()
    if scan and any(
        "symbol" in str(value).lower()
        for key, value in plan.items()
        if "cond" in key.lower() or "filter" in key.lower()
    ):
        return True
    return any(
        symbol_filter_in_plan(value) for value in plan.values() if isinstance(value, (dict, list))
    )


def nested_plan_evidence(path: Path | None) -> dict:
    """Check externally captured nested plans for symbol and date restrictions."""
    if path is None:
        return {"status": "collected_by_orchestrator", "plans": None}
    import json

    try:
        plans = json.loads(path.read_text())
        valid = all(
            isinstance(plans.get(name), str)
            and re.search(
                r"(?i)(?:filter|cond)[^\n]*\bsymbol\b[^\n]*(?:=|\bany\b|\bin\b)", plans[name]
            )
            and re.search(
                r'(?i)(?<![a-z0-9_])"?(?:timestamp|date|_ts_meta_(?:min|max)_1)"?\s*'
                r"(?:>=|<=|>|<|=|\bbetween\b)",
                plans[name],
            )
            for name in ("yahoo", "crsp")
        )
    except (OSError, ValueError, AttributeError, TypeError):
        return {"status": "fail", "plans": None}
    return {"status": "pass" if valid else "fail", "plans": plans}


def check8_status(good: bool, enough_symbols: bool, nested: str, synthetic: bool) -> str:
    """Combine measured timings with the separately collected nested-plan evidence."""
    if not good or nested == "fail":
        return "fail"
    if synthetic and (not enough_symbols or nested == "collected_by_orchestrator"):
        return "skipped"
    if nested == "collected_by_orchestrator":
        return "partial"
    return "pass" if enough_symbols else "skipped"


def read_timings_pass(measurements: dict) -> bool:
    """Apply the recorded limits to each measured read duration."""
    return all(
        measurements[key.removesuffix("_lt")] < limit
        for key, limit in READ_TIMING_TOLERANCE.items()
    )


def ten_stock_passes(value: dict, *, minimum_months: int) -> bool:
    """Require paired coverage and all three return-comparison thresholds."""
    return (
        value.get("count", 0) >= minimum_months
        and value.get("correlation") is not None
        and value["correlation"] >= 0.9998
        and value["mean_abs_points"] <= 0.02
        and value["max_abs_points"] <= 1.6
    )


def _stock_comparison(conn, release_id: int, mapping: dict | None = None) -> tuple[str, dict, str]:
    """Compare month-end returns from derived close and adjusted Yahoo close."""
    import pandas as pd

    minimum_months = 239 if mapping is None else 2
    mapping = TEN_STOCKS if mapping is None else mapping
    loads = _release_loads(conn, release_id)
    if "crsp-bars" not in loads:
        return "fail", {}, "release has no derived bars load"
    source_path = loads["crsp-bars"][1] or ""
    if not source_path.startswith("load:"):
        return "fail", {}, "derived bars do not identify the original CRSP load"
    raw_load = int(source_path.split(":", 1)[1])
    with conn.cursor() as cur:
        cur.execute(
            "SELECT permno,count(*) FROM kaggle_crsp.daily "
            "WHERE load_id=%s AND permno=ANY(%s) GROUP BY permno",
            (raw_load, list(mapping.values())),
        )
        raw_counts = dict(cur.fetchall())
        dated_tickers = {}
        if "mcgill" in loads:
            cur.execute(
                "SELECT stock_ticker,permno,min(date),max(date) FROM kaggle_chars.mcgill "
                "WHERE load_id=%s AND stock_ticker=ANY(%s) AND permno IS NOT NULL "
                "GROUP BY stock_ticker,permno ORDER BY stock_ticker,permno",
                (loads["mcgill"][0], list(mapping)),
            )
            for ticker, permno, first, last in cur.fetchall():
                dated_tickers.setdefault(ticker, []).append(
                    {"permno": permno, "first_date": str(first), "last_date": str(last)}
                )
        cur.execute(
            "SELECT permno,date,spell,close FROM market.daily_bars "
            "WHERE load_id=%s AND permno=ANY(%s) AND date >= DATE '2005-01-01' "
            "AND date <= DATE '2024-12-31' ORDER BY permno,date",
            (loads["crsp-bars"][0], list(mapping.values())),
        )
        derived = pd.DataFrame(cur.fetchall(), columns=["permno", "date", "spell", "close"])
        cur.execute(
            "SELECT symbol,timestamp,close FROM public.ohlcv WHERE symbol=ANY(%s) "
            "AND interval='1d' AND venue='yfinance' AND timestamp >= TIMESTAMP '2005-01-01' "
            "AND timestamp < TIMESTAMP '2025-01-01' ORDER BY symbol,timestamp",
            (list(mapping),),
        )
        yahoo = pd.DataFrame(cur.fetchall(), columns=["symbol", "date", "close"])
    measured = {}
    for ticker, permno in mapping.items():
        x = derived.loc[derived.permno == permno].copy()
        y = yahoo.loc[yahoo.symbol == ticker].copy()
        if x.empty or y.empty:
            measured[ticker] = {"permno": permno, "count": 0, "reason": "missing data"}
            continue
        x["month"] = pd.to_datetime(x.date).dt.to_period("M")
        y["month"] = pd.to_datetime(y.date).dt.to_period("M")
        x = x.groupby("month", sort=True).last()
        y = y.groupby("month", sort=True).last()
        x = x.reindex(pd.period_range(x.index.min(), x.index.max(), freq="M"))
        y = y.reindex(pd.period_range(y.index.min(), y.index.max(), freq="M"))
        xr = x.close.pct_change(fill_method=None).where(x.spell.eq(x.spell.shift()))
        yr = y.close.pct_change(fill_method=None)
        paired = (
            pd.concat([xr.rename("derived"), yr.rename("yahoo")], axis=1)
            .loc["2005-02":"2024-12"]
            .dropna()
        )
        measured[ticker] = {
            "permno": permno,
            "raw_rows": raw_counts.get(permno, 0),
            "mcgill_dated_tickers": dated_tickers.get(ticker, []),
            "mcgill_disagrees": bool(dated_tickers.get(ticker))
            and not any(item["permno"] == permno for item in dated_tickers[ticker]),
            "mapping_source": "validate_crsp.py",
            **return_statistics(paired.derived, paired.yahoo),
        }

    passed = all(
        row.get("raw_rows", 0) > 0 and ten_stock_passes(row, minimum_months=minimum_months)
        for row in measured.values()
    )
    return ("pass" if passed else "fail"), measured, ""


def _mcgill_comparison(
    conn, raw_id: int, mcgill_id: int, *, count: int = 300
) -> tuple[str, dict, str]:
    """Compare McGill excess return plus risk-free rate to same-month compounded CRSP return."""
    import random

    import pandas as pd

    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT permno FROM kaggle_chars.mcgill WHERE load_id=%s "
            "AND permno IS NOT NULL ORDER BY permno",
            (mcgill_id,),
        )
        available = [row[0] for row in cur.fetchall()]
        if not available:
            return "fail", {}, "McGill load has no permnos"
        sample = random.Random(42).sample(available, min(count, len(available)))
        if len(sample) < count:
            return (
                "skipped",
                {"sample_permnos": sample, "seed": 42},
                "fewer than 300 McGill permnos",
            )
        cur.execute(
            "SELECT permno,date_trunc('month',date)::date, "
            "CASE WHEN bool_or(ret = -1) THEN -1 "
            "ELSE exp(sum(CASE WHEN ret > -1 THEN ln(1+ret) END)) - 1 END "
            "FROM kaggle_crsp.daily WHERE load_id=%s AND permno=ANY(%s) "
            "AND ret IS NOT NULL AND ret >= -1 GROUP BY permno,2",
            (raw_id, sample),
        )
        crsp = pd.DataFrame(cur.fetchall(), columns=["permno", "month", "crsp"])
        cur.execute(
            "SELECT permno,date_trunc('month',ret_eom)::date,stock_exret + rf "
            "FROM kaggle_chars.mcgill WHERE load_id=%s AND permno=ANY(%s) "
            "AND ret_eom IS NOT NULL AND stock_exret IS NOT NULL AND rf IS NOT NULL",
            (mcgill_id, sample),
        )
        mcgill = pd.DataFrame(cur.fetchall(), columns=["permno", "month", "mcgill"])
    paired = mcgill.merge(crsp, on=["permno", "month"], how="inner", validate="one_to_one")
    values = return_statistics(paired.mcgill, paired.crsp)
    values.update(
        {
            "compared_permnos": int(paired.permno.nunique()),
            "sample_permnos": sample,
            "seed": 42,
            "alignment": "ret_eom calendar month; stock_exret + rf against compounded daily ret",
        }
    )
    ok = (
        values["compared_permnos"] == len(sample)
        and values["correlation"] is not None
        and values["correlation"] >= 0.9999
    )
    return ("pass" if ok else "fail"), values, ""


def _finish_check(check: dict, started: float) -> None:
    """Record and print elapsed time as each check finishes."""
    check["elapsed_seconds"] = time.perf_counter() - started
    print(
        f"{check['id']} {check['name']}: {check['elapsed_seconds']:.2f}s",
        file=sys.stderr,
        flush=True,
    )


def run_checks(
    url: str,
    release_id: int | None,
    data_volume: Path,
    *,
    save_baseline: Path | None = None,
    compare_baseline: Path | None = None,
    baseline_origin: str | None = None,
    allow_synthetic_skips: bool = False,
    nested_plans: Path | None = None,
) -> dict:
    """Measure implemented checks; label unimplemented evidence rather than guessing."""
    import random

    checks = initial_checks()
    conn = connect_target(url)
    try:
        if save_baseline is not None and compare_baseline is None:
            from ggTrader.data.kaggle_import.link_manifest import file_sha256

            started = time.perf_counter()
            baseline = _baseline(conn, url, save=save_baseline)
            measured = {
                "baseline": baseline,
                "baseline_file": str(save_baseline),
                "baseline_sha256": file_sha256(save_baseline),
                "baseline_origin": baseline_origin,
            }
            for check in checks[:8]:
                _set(check, "skipped", {}, None, "baseline only")
            _set(
                checks[3],
                "skipped" if allow_synthetic_skips else "fail",
                measured,
                "baseline comparison required",
                "no baseline given" if allow_synthetic_skips else BASELINE_REQUIRED_REASON,
            )
            _finish_check(checks[3], started)
            with conn.cursor() as cur:
                cur.execute("SELECT pg_database_size(current_database())")
                db_size = cur.fetchone()[0]
                cur.execute(
                    "SELECT setting::bigint FROM pg_settings WHERE name='statement_timeout'"
                )
                statement_timeout_ms = cur.fetchone()[0]
            return {
                "release_id": None,
                "database_bytes": db_size,
                "free_bytes": shutil.disk_usage(data_volume).free,
                "statement_timeout_ms": statement_timeout_ms,
                "synthetic_skips_allowed": allow_synthetic_skips,
                "checks": checks,
            }
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
            cur.execute("SELECT setting::bigint FROM pg_settings WHERE name='statement_timeout'")
            statement_timeout_ms = cur.fetchone()[0]
        symbols = _symbols(conn, release_id)
        started = time.perf_counter()
        status, measured, reason = _check_counts(conn, release_id)
        _set(checks[0], status, measured, "exact row counts", reason)
        _finish_check(checks[0], started)
        started = time.perf_counter()
        status, measured, reason = _stock_comparison(conn, release_id)
        if (
            allow_synthetic_skips
            and status == "fail"
            and all(
                row.get("count", 0) < 2 or ten_stock_passes(row, minimum_months=2)
                for row in measured.values()
            )
        ):
            status, reason = "skipped", "synthetic data"
        _set(
            checks[1],
            status,
            measured,
            {"correlation_min": 0.9998, "mean_abs_points_max": 0.02, "max_abs_points_max": 1.6},
            reason,
        )
        _finish_check(checks[1], started)
        started = time.perf_counter()
        loads = _release_loads(conn, release_id)
        source_path = loads.get("crsp-bars", (None, None))[1] or ""
        if "mcgill" not in loads or not source_path.startswith("load:"):
            _set(
                checks[2],
                "fail",
                {},
                {"correlation_min": 0.9999},
                "release lacks McGill or identifiable CRSP source load",
            )
        else:
            status, measured, reason = _mcgill_comparison(
                conn, int(source_path.split(":", 1)[1]), loads["mcgill"][0]
            )
            if allow_synthetic_skips and status == "skipped":
                reason = "synthetic data"
            _set(checks[2], status, measured, {"correlation_min": 0.9999}, reason)
        _finish_check(checks[2], started)
        started = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol FROM market.stock_spans WHERE release_id=%s "
                "AND kind IN ('yahoo','linked_yahoo') ORDER BY symbol",
                (release_id,),
            )
            yahoo_symbols = [row[0] for row in cur.fetchall()]
        chosen = random.Random(42).sample(yahoo_symbols, min(40, len(yahoo_symbols)))
        if chosen:
            status, measured, reason = _check_yahoo_keys(conn, url, release_id, chosen)
            if status == "pass" and len(chosen) < 40:
                status, reason = (
                    "skipped",
                    (
                        "synthetic data"
                        if allow_synthetic_skips
                        else "fewer than 40 Yahoo symbols in release"
                    ),
                )
        else:
            status, measured, reason = (
                "skipped",
                {},
                ("synthetic data" if allow_synthetic_skips else "no Yahoo symbols in release"),
            )
        if save_baseline or compare_baseline:
            baseline = _baseline(conn, url, save=save_baseline, compare=compare_baseline)
            measured["baseline"] = baseline
            if baseline["matches"] is False:
                status, reason = "fail", "baseline hashes differ"
        baseline_path = compare_baseline or save_baseline
        from ggTrader.data.kaggle_import.link_manifest import file_sha256

        measured["baseline_file"] = str(baseline_path) if baseline_path else None
        measured["baseline_sha256"] = file_sha256(baseline_path) if baseline_path else None
        measured["baseline_origin"] = baseline_origin
        if compare_baseline is None:
            if allow_synthetic_skips:
                if status == "pass":
                    status, reason = "skipped", "no baseline given"
            else:
                status, reason = "fail", BASELINE_REQUIRED_REASON
        _set(checks[3], status, measured, "identical row keys and all five fields", reason)
        _finish_check(checks[3], started)
        started = time.perf_counter()
        status, measured, reason = _check_named_stocks(conn, url, release_id)
        _set(checks[4], status, measured, {"absolute_join_return_lt": 0.5}, reason)
        _finish_check(checks[4], started)
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
        _finish_check(checks[5], started)
        started = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol FROM market.stock_spans WHERE release_id=%s "
                "AND kind='placeholder' AND first_date < DATE '2000-01-01' ORDER BY symbol",
                (release_id,),
            )
            placeholders = [row[0] for row in cur.fetchall()]
        sample = select_timezone_symbols(placeholders, symbols)
        if sample:
            prior = os.environ.get("PGTZ")
            try:
                os.environ["PGTZ"] = "UTC"
                utc = _load(url, sample, "combined", release_id, start="1968-01-01")
                os.environ["PGTZ"] = "America/Los_Angeles"
                pacific = _load(url, sample, "combined", release_id, start="1968-01-01")
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
            _set(
                checks[6],
                "skipped",
                {},
                "hashes equal",
                "synthetic data"
                if allow_synthetic_skips
                else "fewer than five pre-2000 placeholders or twenty total symbols",
            )
        _finish_check(checks[6], started)
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
            and index["name"] == "symbol_map_release_symbol_dates_idx"
            and all(col in index["definition"].lower() for col in ("symbol", "valid_from"))
            for index in indexes
        )
        index_ok = index_ok and bars_key and symbol_key
        with conn.cursor() as cur:
            probe_symbols = yahoo[:1] or ["__NO_SYMBOL__"]
            cur.execute(
                "EXPLAIN (FORMAT JSON) SELECT timestamp,symbol,open,high,low,close,volume "
                "FROM public.ohlcv WHERE symbol=ANY(%s) AND interval='1d' "
                "AND venue='yfinance' AND timestamp >= TIMESTAMP '2000-01-03' "
                "AND timestamp <= TIMESTAMP '2024-12-31'",
                (probe_symbols,),
            )
            plan = cur.fetchone()[0]
        plan_ok = symbol_filter_in_plan(plan)
        nested = nested_plan_evidence(nested_plans)
        measurements = {
            "indexes": indexes,
            "baseline_direct_probe_plan": plan,
            "nested_function_plan": nested["plans"],
            "nested_function_plan_status": nested["status"],
            "symbol_filter_in_scan": plan_ok,
            "available_yahoo_symbols": len(yahoo),
            "required_indexes_present": index_ok,
            "timing_note": "A combined read has a fixed cost of a few seconds per call.",
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
            good = index_ok and plan_ok and read_timings_pass(measurements)
            status = check8_status(good, True, nested["status"], allow_synthetic_skips)
            _set(
                checks[7],
                status,
                measurements,
                READ_TIMING_TOLERANCE,
                "synthetic data"
                if status == "skipped"
                else "nested plans not supplied"
                if status == "partial"
                else "",
            )
        else:
            status = check8_status(
                index_ok and plan_ok, False, nested["status"], allow_synthetic_skips
            )
            _set(
                checks[7],
                status,
                measurements,
                READ_TIMING_TOLERANCE,
                "synthetic data" if allow_synthetic_skips else "fewer than 100 Yahoo symbols",
            )
        _finish_check(checks[7], started)
        for check in checks[8:]:
            _finish_check(check, time.perf_counter())
        return {
            "release_id": release_id,
            "database_bytes": db_size,
            "free_bytes": free,
            "statement_timeout_ms": statement_timeout_ms,
            "synthetic_skips_allowed": allow_synthetic_skips,
            "checks": checks,
        }
    finally:
        conn.close()


def checks_succeeded(checks: list[dict], *, allow_synthetic_skips: bool = False) -> bool:
    """Require every check to pass or have one explicitly permitted absence."""
    allowed_skips = {"name not in release"}
    if allow_synthetic_skips:
        allowed_skips.update(("synthetic data", "no baseline given"))
    return all(
        item["status"] in ("pass", "covered_by_tests")
        or (item["status"] == "skipped" and item.get("reason") in allowed_skips)
        for item in checks
    )


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
            baseline_origin=args.baseline_origin,
            allow_synthetic_skips=args.allow_synthetic_skips,
            nested_plans=args.nested_plans,
        )
    except Exception as exc:
        from ggTrader.data.kaggle_import.guards import safe_error

        raise SystemExit(f"Closing checks failed: {safe_error(exc)}") from None
    out = args.out or default_output(date.today(), args.label)
    write_json(out, report)
    for item in report["checks"]:
        print(f"{item['id']}  {item['status']:<16} {item['name']}")
    print(f"Report: {out}")
    return (
        0
        if checks_succeeded(report["checks"], allow_synthetic_skips=args.allow_synthetic_skips)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
