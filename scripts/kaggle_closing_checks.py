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


def _check_yahoo_keys(conn, url: str, release_id: int, names: list[str]) -> tuple[str, dict, str]:
    """Compare post-2000 row keys and all fields; count added older rows."""
    import numpy as np
    import pandas as pd

    from ggTrader.data.bars import BAR_FIELDS

    combined = _load(url, names, "combined", release_id)
    db = _load(url, names, "db", start="2000-01-03")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT timestamp,symbol FROM public.ohlcv WHERE symbol=ANY(%s) "
            "AND venue='yfinance' AND interval='1d' AND timestamp >= TIMESTAMP '2000-01-03'",
            (names,),
        )
        db_keys = {(pd.Timestamp(stamp, tz="UTC"), symbol) for stamp, symbol in cur.fetchall()}
    cutoff = pd.Timestamp("2000-01-03", tz="UTC")
    flag_index = combined.flags.index
    combo_keys = {(ts, str(symbol)) for ts, symbol in flag_index if ts >= cutoff}
    older = sum(ts < cutoff for ts, _ in flag_index)
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
        "earlier_rows_added": older,
        "different_fields": different,
    }
    return ("fail" if missing or extra or different else "pass"), measured, ""


def duplicate_stock_days(flags, identities: dict[str, int]) -> list[dict]:
    """Report stock-days read under multiple names unless every row is flagged."""
    import pandas as pd

    if flags.empty:
        return []
    frame = flags[["possible_double"]].reset_index()
    frame["permno"] = frame["symbol"].map(identities)
    frame = frame.dropna(subset=["permno"])
    result = []
    for (day, permno), rows in frame.groupby(["timestamp", "permno"], sort=False):
        if rows.symbol.nunique() > 1 and not rows.possible_double.fillna(False).all():
            result.append(
                {
                    "date": str(pd.Timestamp(day).date()),
                    "permno": int(permno),
                    "symbols": sorted(rows.symbol.astype(str).unique().tolist()),
                }
            )
    return result


def _check_named_stocks(conn, url: str, release_id: int) -> tuple[str, dict, str]:
    """Inspect named traces and check overlapping identity columns in bounded batches."""
    import pandas as pd

    with conn.cursor() as cur:
        cur.execute(
            "SELECT symbol,permno,kind,first_date,last_date,possible_double "
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
    names = ["AAPL", "TTWO", "CMVT", "SMCI", "IBKR", "VC", "PERMNO-10128", "GM"]
    related = {row[0] for row in records if row[0].startswith("PERMNO-10128~")}
    related.update(row[0] for row in records if row[1] in gm_permnos)
    selected = [name for name in names if name in catalog] + sorted(related)
    selected = list(dict.fromkeys(selected))
    for name in names:
        if name not in catalog:
            cases[name] = {"status": "skipped", "reason": "name not in release"}
    if selected:
        bars = _load(url, selected, "combined", release_id)
        for name in selected:
            subset = bars.flags[bars.flags.index.get_level_values("symbol") == name]
            sources = subset["source"].astype(str).value_counts().to_dict()
            close = bars.close[name].dropna() if name in bars.close else pd.Series(dtype=float)
            cases[name] = {
                "status": "pass" if len(close) else "fail",
                "rows": int(len(subset)),
                "first": str(close.index[0].date()) if len(close) else None,
                "last": str(close.index[-1].date()) if len(close) else None,
                "sources": sources,
                "kind": catalog[name][2],
                "possible_double": bool(catalog[name][5]),
            }
            if catalog[name][2] == "linked_yahoo" and len(subset):
                yahoo = subset[subset.source == "yfinance"]
                earlier = subset[subset.source == "crsp"]
                if not yahoo.empty and not earlier.empty:
                    start = yahoo.index.get_level_values("timestamp").min()
                    previous = close.loc[close.index < start]
                    if not previous.empty and start in close.index:
                        jump = float(close.loc[start] / previous.iloc[-1] - 1)
                        cases[name]["join_return"] = jump
                        if abs(jump) >= 0.5:
                            cases[name]["status"] = "fail"
            if name == "AAPL":
                before = subset.index.get_level_values("timestamp") < pd.Timestamp(
                    "2000-01-03", tz="UTC"
                )
                if not ((subset.source == "crsp") & before).any():
                    cases[name]["status"] = "fail"
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
        pre2000 = int((historic.close.index < pd.Timestamp("2000-01-01", tz="UTC")).sum())
    else:
        placeholder, pre2000 = None, 0
    missing = _load(url, ["PERMNO-84761", "PERMNO-1~7"], "combined", release_id)
    missing_names = list(missing.missing_symbols)
    groups = {}
    for symbol, permno, _, first, last, _ in records:
        if permno is not None and first is not None and last is not None:
            groups.setdefault(permno, []).append((symbol, first, last))
    candidates = []
    for permno, rows in groups.items():
        if any(a[1] <= b[2] and b[1] <= a[2] for i, a in enumerate(rows) for b in rows[i + 1 :]):
            candidates.append((permno, [row[0] for row in rows]))
    violations = []
    for permno, columns in candidates:
        if len(columns) > 120:
            return "fail", {"cases": cases, "permno": permno}, "identity has more than 120 columns"
        group_bars = _load(url, columns, "combined", release_id)
        violations.extend(
            duplicate_stock_days(group_bars.flags, {name: permno for name in columns})
        )
    measured = {
        "cases": cases,
        "placeholder": placeholder,
        "pre_2000_rows": pre2000,
        "gm_permnos": gm_permnos,
        "missing_symbols": missing_names,
        "overlap_groups": len(candidates),
        "unflagged_duplicate_days": len(violations),
        "unflagged_examples": violations[:10],
    }
    if violations or any(case["status"] == "fail" for case in cases.values()):
        return "fail", measured, "a named trace or duplicate-stock check failed"
    return "skipped", measured, "exact named-stock traces are not yet fully verified"


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


def _stock_comparison(conn, release_id: int, mapping: dict | None = None) -> tuple[str, dict, str]:
    """Compare month-end returns from derived close and adjusted Yahoo close."""
    import pandas as pd

    mapping = TEN_STOCKS if mapping is None else mapping
    loads = _release_loads(conn, release_id)
    if "crsp-bars" not in loads:
        return "fail", {}, "release has no derived bars load"
    with conn.cursor() as cur:
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
        measured[ticker] = {"permno": permno, **return_statistics(paired.derived, paired.yahoo)}

    def good(value):
        return (
            value.get("count", 0) >= 2
            and value.get("correlation") is not None
            and value["correlation"] >= 0.9998
            and value["mean_abs_points"] <= 0.02
            and value["max_abs_points"] <= 1.6
        )

    return ("pass" if all(good(row) for row in measured.values()) else "fail"), measured, ""


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
            "sample_permnos": sample,
            "seed": 42,
            "alignment": "ret_eom calendar month; stock_exret + rf against compounded daily ret",
        }
    )
    ok = values["correlation"] is not None and values["correlation"] >= 0.9999
    return ("pass" if ok else "fail"), values, ""


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
        started = time.perf_counter()
        status, measured, reason = _stock_comparison(conn, release_id)
        _set(
            checks[1],
            status,
            measured,
            {"correlation_min": 0.9998, "mean_abs_points_max": 0.02, "max_abs_points_max": 1.6},
            reason,
        )
        checks[1]["elapsed_seconds"] = time.perf_counter() - started
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
            _set(checks[2], status, measured, {"correlation_min": 0.9999}, reason)
        checks[2]["elapsed_seconds"] = time.perf_counter() - started
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
                status, reason = "skipped", "fewer than 40 Yahoo symbols in release"
        else:
            status, measured, reason = "skipped", {}, "no Yahoo symbols in release"
        if save_baseline or compare_baseline:
            baseline = _baseline(conn, url, save=save_baseline, compare=compare_baseline)
            measured["baseline"] = baseline
            if baseline["matches"] is False:
                status, reason = "fail", "baseline hashes differ"
            elif status == "pass" and compare_baseline is None:
                status, reason = "skipped", "no baseline given"
        elif status == "pass":
            status, reason = "skipped", "no baseline given"
        _set(checks[3], status, measured, "identical row keys and all five fields", reason)
        checks[3]["elapsed_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        status, measured, reason = _check_named_stocks(conn, url, release_id)
        _set(checks[4], status, measured, {"absolute_join_return_lt": 0.5}, reason)
        checks[4]["elapsed_seconds"] = time.perf_counter() - started
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
                "fewer than five pre-2000 placeholders or twenty total symbols",
            )
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
        measurements = {
            "indexes": indexes,
            "plan": plan,
            "symbol_filter_in_scan": plan_ok,
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
                and plan_ok
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
                "skipped" if index_ok and plan_ok else "fail",
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
