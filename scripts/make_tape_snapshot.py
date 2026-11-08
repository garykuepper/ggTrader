"""Export a hashed, read-only Parquet snapshot of the daily equity OHLCV tape.

The snapshot is the frozen input for every vectorbt-engine reference run (spec
docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md, section 7.3). It is
read straight from the ``ohlcv`` table (venue='yfinance', interval='1d') with
plain SQL on a read-only session: no loader, no network, no DDL, no INSERT.

Stored layout (long, sorted by symbol then timestamp):
``timestamp`` (naive UTC midnight, exactly as the DB column), ``symbol``,
``open``, ``high``, ``low``, ``close``, ``volume`` (float64).
:func:`load_snapshot_wide` rebuilds the frame ``ggTrader.lab.data.load_ohlcv``
returns: tz-aware UTC index, ``(symbol, field)`` MultiIndex columns, fields sorted
close/high/low/open/volume, ``collapse_daily_duplicates`` applied.

Usage:
    PYTHONPATH=/path/with/.env/src python scripts/make_tape_snapshot.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
MANIFEST_NAME = "MANIFEST.json"

#: Window the reference runs need (spec P0 task 3). The start covers the 365-day live
#: window and the 252-bar warm-up before the first eval day; the end is the last
#: complete session before the vbt12 branch was cut.
DEFAULT_START = "2019-03-01"
DEFAULT_END = "2026-10-30"
DEFAULT_EVAL_START = "2021-01-31"
DEFAULT_EVAL_END = "2026-04-30"
DEFAULT_UNIVERSE = "sp500"
#: Symbols outside the SP500 PIT universe that reference runs read: the benchmark and the
#: cross_asset_sleeve legs (SPY core, TLT/GLD/PDBC sleeve, BIL cash).
DEFAULT_EXTRAS = ("SPY", "TLT", "GLD", "PDBC", "BIL")

FIELDS = ("open", "high", "low", "close", "volume")
WIDE_FIELD_ORDER = ("close", "high", "low", "open", "volume")  # alphabetical, as sort_index gives
VENUE = "yfinance"
INTERVAL = "1d"
STATEMENT_TIMEOUT_MS = 900_000

#: Sanity thresholds (spec 7.3: price jumps > 10x, long runs of identical closes).
JUMP_RATIO = 10.0
IDENTICAL_CLOSE_RUN = 20


# --------------------------------------------------------------------------- DB (read-only)


def connect_readonly() -> Any:
    """Open a read-only psycopg2 connection (server-enforced) using the project config."""
    import psycopg2

    from ggTrader.utils.config import get_db_connection_string

    conn_str = os.environ.get("GGTRADER_DB_URL") or get_db_connection_string()
    conn_str = conn_str.replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(
        conn_str,
        options=f"-c default_transaction_read_only=on -c statement_timeout={STATEMENT_TIMEOUT_MS}",
    )
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("show default_transaction_read_only")
        if cur.fetchone()[0] != "on":
            conn.close()
            raise RuntimeError("database session is not read-only; refusing to continue")
    return conn


def session_timezone(conn: Any) -> str:
    """The server session time zone (it decides how the DB compares naive stamps to UTC bounds)."""
    with conn.cursor() as cur:
        cur.execute("show timezone")
        return str(cur.fetchone()[0])


def read_long(conn: Any, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Read daily bars for ``symbols`` as long rows, with the loader's own column handling."""
    query = """
        SELECT timestamp, symbol, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = ANY(%s) AND interval = %s AND venue = %s
          AND timestamp >= %s AND timestamp <= %s
        ORDER BY symbol ASC, timestamp ASC;
    """
    with conn.cursor() as cur:
        cur.execute(query, (symbols, INTERVAL, VENUE, start, end))
        rows = cur.fetchall()
    flat = pd.DataFrame(rows, columns=["timestamp", "symbol", *FIELDS])
    flat["timestamp"] = pd.to_datetime(flat["timestamp"])
    for col in FIELDS:
        flat[col] = pd.to_numeric(flat[col]).astype("float64")
    return flat


# --------------------------------------------------------------------------- universe


def build_symbol_list(
    universe: str, eval_start: str, eval_end: str, extras: Iterable[str], membership: Any = None
) -> list[str]:
    """PIT universe members at any point in the eval span (yfinance-normalised) plus extras.

    ``membership`` is the S&P 500 list source (``None``: ``GGT_MEMBERSHIP``, else an error).
    """
    from ggTrader.lab.data import equity_universe_between

    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")
    members = equity_universe_between(es, ee, universe=universe, membership=membership)
    return sorted(set(members) | set(extras))


# --------------------------------------------------------------------------- frame conventions


def collapse_daily_duplicates(wide: pd.DataFrame) -> pd.DataFrame:
    """Same rule as ``ggTrader.lab.data.collapse_daily_duplicates`` (kept local to be import-light).

    The legacy function is imported when available and used instead, so this copy only
    serves environments where ``ggTrader.lab`` cannot be imported.
    """
    try:
        from ggTrader.lab.data import collapse_daily_duplicates as legacy

        return legacy(wide)
    except ImportError:  # pragma: no cover - only without the legacy lab package
        pass
    if wide.empty:
        return wide
    idx = pd.DatetimeIndex(wide.index)
    tods = pd.Index(idx.strftime("%H:%M"))
    if tods.nunique() <= 1 or not pd.Series(idx.normalize()).duplicated().any():
        return wide
    coverage = wide.notna().mean(axis=1).groupby(tods).mean()
    keep = set(coverage[coverage >= coverage.max() * 0.5].index)
    return wide[tods.isin(keep)]


def long_to_wide(flat: pd.DataFrame) -> pd.DataFrame:
    """Pivot long rows exactly as ``CachedYFinanceLoader._db_fetch`` does (tz-aware UTC index)."""
    if flat.empty:
        return pd.DataFrame()
    flat = flat.copy()
    idx = pd.DatetimeIndex(flat["timestamp"])
    flat["timestamp"] = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    wide = flat.pivot(index="timestamp", columns="symbol", values=list(FIELDS))
    wide.columns = pd.MultiIndex.from_tuples([(sym, metric) for metric, sym in wide.columns])
    wide.sort_index(axis=1, inplace=True)
    wide.columns.names = ["symbol", "field"]
    return wide


def load_snapshot_wide(
    path: str | Path,
    symbols: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    session_tz: Optional[str] = None,
) -> pd.DataFrame:
    """Rebuild the ``load_ohlcv`` frame for ``symbols`` in [start, end] from a snapshot.

    ``session_tz`` reproduces the database's own boundary rule: a naive stamp compared with a
    tz-aware bound is read in the server session zone, so with America/Los_Angeles the bar
    stamped ``end`` 00:00 sits at 07:00 UTC and is excluded from an ``end`` 00:00 UTC bound.
    ``None`` uses the zone recorded in the manifest next to the file.
    """
    path = Path(path)
    if session_tz is None:
        manifest = json.loads((path.parent / MANIFEST_NAME).read_text())
        entry = next((e for e in manifest["snapshots"] if e["path"].endswith(path.name)), None)
        session_tz = entry["db_session_timezone"] if entry else "UTC"
    flat = pd.read_parquet(path)
    if symbols is not None:
        flat = flat[flat["symbol"].isin(set(symbols))]
    stamps = pd.DatetimeIndex(flat["timestamp"])
    as_db = stamps.tz_localize(session_tz).tz_convert("UTC")
    keep = np.ones(len(flat), dtype=bool)
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    if start_ts is not None:
        keep &= np.asarray(as_db >= start_ts)
    if end_ts is not None:
        keep &= np.asarray(as_db <= end_ts)
    wide = long_to_wide(flat[keep])
    if wide.empty:
        return wide
    # fetch_stock_ohlcv's own post-filter on the (tz-aware UTC) index
    if start_ts is not None:
        wide = wide[wide.index >= start_ts]
    if end_ts is not None:
        wide = wide[wide.index <= end_ts]
    wide.columns.names = ["symbol", "field"]
    return collapse_daily_duplicates(wide)


# --------------------------------------------------------------------------- hashing / checks


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_sha256(flat: pd.DataFrame) -> str:
    """Writer-independent hash of the data (sorted rows, little-endian float64 bytes)."""
    flat = flat.sort_values(["symbol", "timestamp"], kind="mergesort")
    h = hashlib.sha256()
    h.update("\n".join(flat["symbol"].astype(str)).encode())
    h.update(flat["timestamp"].to_numpy(dtype="datetime64[ns]").astype("<i8").tobytes())
    for col in FIELDS:
        h.update(flat[col].to_numpy(dtype="<f8").tobytes())
    return h.hexdigest()


def sanity_flags(flat: pd.DataFrame) -> dict[str, Any]:
    """Tape sanity: jumps > 10x, long identical-close runs, dups, weekday and midnight stamps."""
    out: dict[str, Any] = {"jump_symbols": {}, "identical_close_symbols": {}}
    stamps = pd.DatetimeIndex(flat["timestamp"])
    out["non_midnight_rows"] = int((stamps != stamps.normalize()).sum())
    out["weekend_rows"] = int((stamps.dayofweek >= 5).sum())
    out["duplicate_symbol_timestamp_rows"] = int(flat.duplicated(["symbol", "timestamp"]).sum())
    out["null_close_rows"] = int(flat["close"].isna().sum())
    for sym, grp in flat.groupby("symbol", sort=True):
        close = grp["close"].to_numpy(dtype=float)
        if len(close) < 2:
            continue
        ratio = close[1:] / close[:-1]
        bad = np.isfinite(ratio) & ((ratio > JUMP_RATIO) | (ratio < 1.0 / JUMP_RATIO))
        if bad.any():
            first = int(np.argmax(bad)) + 1
            out["jump_symbols"][str(sym)] = {
                "n": int(bad.sum()),
                "first": str(grp["timestamp"].iloc[first].date()),
            }
        same = np.concatenate([[False], close[1:] == close[:-1]])
        run = best = 0
        for flag in same:
            run = run + 1 if flag else 0
            best = max(best, run)
        if best >= IDENTICAL_CLOSE_RUN:
            out["identical_close_symbols"][str(sym)] = best
    return out


def verify_against_db(conn: Any, path: Path, sample: int, seed: int = 0) -> dict[str, Any]:
    """Re-read ``sample`` random symbols from the DB and compare with the file, bit for bit."""
    snap = pd.read_parquet(path)
    syms = sorted(snap["symbol"].unique())
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(syms, size=min(sample, len(syms)), replace=False).tolist())
    lo, hi = str(snap["timestamp"].min().date()), str(snap["timestamp"].max().date())
    fresh = read_long(conn, pick, lo, hi + " 23:59:59")
    mine = snap[snap["symbol"].isin(pick)].reset_index(drop=True)
    fresh = fresh.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    ok = len(mine) == len(fresh) and bool(
        (
            mine[["symbol", "timestamp"]].to_numpy() == fresh[["symbol", "timestamp"]].to_numpy()
        ).all()
    )
    if ok:
        for col in FIELDS:
            a, b = mine[col].to_numpy(), fresh[col].to_numpy()
            ok &= bool(np.array_equal(a, b, equal_nan=True))
    return {"symbols_checked": pick, "rows": int(len(mine)), "identical": ok}


# --------------------------------------------------------------------------- manifest


def update_manifest(entry: dict[str, Any], manifest_path: Path) -> None:
    """Insert or replace ``entry`` (keyed by path) in the tracked manifest."""
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "description": (
                "Frozen read-only daily OHLCV snapshots. Parquet files are git-ignored; this "
                "manifest is tracked. Rebuild with the command recorded per entry."
            ),
            "snapshots": [],
        }
    manifest["snapshots"] = [e for e in manifest["snapshots"] if e["path"] != entry["path"]]
    manifest["snapshots"].append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- main


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description="Export a hashed read-only Parquet snapshot of the daily equity OHLCV tape.",
    )
    p.add_argument("--start", default=DEFAULT_START, help="first bar date (default %(default)s)")
    p.add_argument("--end", default=DEFAULT_END, help="last bar date (default %(default)s)")
    p.add_argument("--eval-start", default=DEFAULT_EVAL_START, help="PIT universe span start")
    p.add_argument("--eval-end", default=DEFAULT_EVAL_END, help="PIT universe span end")
    p.add_argument("--universe", default=DEFAULT_UNIVERSE, help="index universe (default sp500)")
    p.add_argument(
        "--extras", default=",".join(DEFAULT_EXTRAS), help="extra symbols, comma separated"
    )
    p.add_argument("--tag", default=date.today().strftime("%Y%m%d"), help="file tag (YYYYMMDD)")
    p.add_argument("--out-dir", default=str(SNAPSHOT_DIR), help="snapshot directory")
    p.add_argument("--verify-sample", type=int, default=60, help="symbols re-read for the check")
    p.add_argument("--verify-only", metavar="PARQUET", help="only verify an existing file vs DB")
    # Imported here, not at module level: legacy_reference_runs loads this file under the legacy
    # interpreter, where ggTrader.data.membership does not exist.
    from ggTrader.data.membership import ORACLE_SPEC, add_membership_argument

    add_membership_argument(p, default=ORACLE_SPEC)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Export (or verify) the snapshot and update the manifest."""
    args = parse_args(argv)
    t0 = time.time()
    conn = connect_readonly()
    if args.verify_only:
        res = verify_against_db(conn, Path(args.verify_only), args.verify_sample)
        print(json.dumps(res, indent=2))
        return 0 if res["identical"] else 1

    from ggTrader.data.membership import source_from_args

    membership = source_from_args(args)
    extras = [s for s in args.extras.split(",") if s]
    symbols = build_symbol_list(args.universe, args.eval_start, args.eval_end, extras, membership)
    print(f"[snapshot] {len(symbols)} requested symbols", flush=True)
    tz_name = session_timezone(conn)
    flat = read_long(conn, symbols, args.start, args.end + " 23:59:59")
    have = sorted(flat["symbol"].unique())
    missing = sorted(set(symbols) - set(have))
    print(f"[snapshot] {len(flat)} rows, {len(have)} symbols, {len(missing)} not in DB", flush=True)

    # collapse_daily_duplicates is a no-op on the current tape; record that rather than assume it.
    wide = long_to_wide(flat)
    collapsed = collapse_daily_duplicates(wide)
    dropped = int(len(wide) - len(collapsed))
    if dropped:
        keep = pd.DatetimeIndex(collapsed.index).tz_convert(None)
        flat = flat[pd.DatetimeIndex(flat["timestamp"]).isin(keep)]
    flat = flat.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ohlcv_{args.tag}.parquet"
    out.unlink(missing_ok=True)  # a previous snapshot of the same tag is read-only
    flat.to_parquet(out, index=False, compression="zstd", engine="pyarrow")
    out.chmod(0o444)
    flags = sanity_flags(flat)
    check = verify_against_db(conn, out, args.verify_sample)
    conn.close()

    rel = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    entry = {
        "path": str(rel),
        "sha256": sha256_file(out),
        "content_sha256": content_sha256(flat),
        "bytes": out.stat().st_size,
        "rows": int(len(flat)),
        "symbols": int(flat["symbol"].nunique()),
        "requested_symbols": len(symbols),
        "symbols_not_in_db": missing,
        "date_min": str(flat["timestamp"].min().date()),
        "date_max": str(flat["timestamp"].max().date()),
        "trading_days": int(flat["timestamp"].nunique()),
        "columns": ["timestamp", "symbol", *FIELDS],
        "timestamp_convention": "naive UTC midnight (DB column as stored); loader localises to UTC",
        "venue": VENUE,
        "interval": INTERVAL,
        "universe": args.universe,
        "universe_span": [args.eval_start, args.eval_end],
        # New keys: which S&P 500 list chose the symbols (spec 3.5).
        **membership.provenance(date.fromisoformat(args.eval_end)),
        "extras": extras,
        "db_session_timezone": tz_name,
        "collapse_daily_duplicates_rows_dropped": dropped,
        "sanity": flags,
        "verify_against_db": check,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "creation_command": "python " + " ".join(shlex.quote(a) for a in sys.argv),
        "wall_seconds": round(time.time() - t0, 1),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    update_manifest(entry, out_dir / MANIFEST_NAME)
    print(json.dumps({k: entry[k] for k in ("path", "sha256", "rows", "symbols")}, indent=2))
    return 0 if check["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
