#!/usr/bin/env python
"""Fill the ``ohlcv`` price table for current index members that have no bars or stale bars.

Found by the 2026-11-09 membership check (research/programme/membership-database/
2026-11-09-open-checks-closed-with-press-releases.md): 13 of 503 members had no bars (new
listings) or bars that stopped months ago. This tool adds only what is missing:

* INSERT ONLY. Bars after the symbol's last stored bar (or its whole history when it has none)
  are inserted with ``ON CONFLICT DO NOTHING``; no existing row is changed or deleted.
* ``--refresh-last`` additionally rewrites the symbol's LAST stored bar when yfinance's complete bar
  for that day differs by more than ``--tolerance`` (a bar captured mid-session is partial). The old
  row is saved to a CSV first so the change can be undone.
* Same convention as ``CachedYFinanceLoader``: yfinance ``auto_adjust=True``, daily bars, venue
  ``yfinance``, ``timestamp`` as NAIVE UTC midnight (a tz-aware value once shifted the whole table
  one day early), a row without a close is skipped, volume as given.
* Dry run by default: prints the plan and writes nothing. ``--commit`` writes one transaction per
  symbol. Bars after ``--through`` (a pinned date, never "today": today's bar is incomplete) are
  never written.

Usage:
    python scripts/backfill_missing_members.py --symbols BE,FERG --through 2026-11-05
    python scripts/backfill_missing_members.py --symbols MRVL,P --through 2026-11-05 \
        --refresh-last --commit
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd

from ggTrader.data.live.cached_yfinance_loader import STOCK_VENUE
from ggTrader.utils.paths import REPO_ROOT

INTERVAL = "1d"
DEFAULT_START = "2000-01-01"
DEFAULT_TOLERANCE = 0.0005
BACKUP_DIRNAME = "ohlcv_backfill"
COLUMNS = ("open", "high", "low", "close", "volume")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-][A-Z0-9]{1,2})?$")


@dataclass
class SymbolPlan:
    """What would be written for one symbol."""

    symbol: str
    last_stored: date | None
    inserts: pd.DataFrame
    refresh: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""

    @property
    def n_insert(self) -> int:
        """Rows that would be inserted."""
        return len(self.inserts)

    @property
    def n_refresh(self) -> int:
        """Rows that would be rewritten (0 or 1)."""
        return len(self.refresh)


def clean_frame(frame: pd.DataFrame, through: date) -> pd.DataFrame:
    """Daily bars with a close, naive-UTC dates, columns open..volume, none after ``through``."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    out = frame.rename(columns=str.lower)[list(COLUMNS)].copy()
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_convert("UTC") if idx.tz is not None else idx
    out.index = pd.DatetimeIndex(idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    out = out[out["close"].notna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[out.index <= pd.Timestamp(through)]


def plan_symbol(
    symbol: str,
    stored_last: tuple[date, dict[str, float]] | None,
    fetched: pd.DataFrame,
    *,
    through: date,
    refresh_last: bool,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SymbolPlan:
    """Pure: which rows to insert and (optionally) which last stored bar to refresh.

    ``stored_last`` is ``(date, {open,high,low,close,volume})`` of the newest stored bar, or None.
    """
    frame = clean_frame(fetched, through)
    if frame.empty:
        return SymbolPlan(symbol, stored_last[0] if stored_last else None, frame, note="no data")
    if stored_last is None:
        return SymbolPlan(symbol, None, frame, note="whole history")
    last_day, stored = stored_last
    inserts = frame[frame.index > pd.Timestamp(last_day)]
    refresh = pd.DataFrame(columns=list(COLUMNS))
    note = ""
    stamp = pd.Timestamp(last_day)
    if refresh_last and stamp in frame.index:
        fresh = frame.loc[[stamp]]
        close_gap = abs(float(fresh["close"].iloc[0]) / float(stored["close"]) - 1.0)
        if close_gap > tolerance:
            refresh = fresh
            note = f"last bar {last_day} differs by {close_gap:.4%}"
    return SymbolPlan(symbol, last_day, inserts, refresh, note)


def rows_for_db(symbol: str, frame: pd.DataFrame) -> list[tuple]:
    """Tuples for the INSERT; ``timestamp`` is a naive ``datetime`` (never tz-aware)."""
    records = []
    for ts, row in frame.iterrows():
        vol = None if pd.isna(row["volume"]) else float(row["volume"])
        records.append(
            (
                ts.to_pydatetime().replace(tzinfo=None),
                symbol,
                INTERVAL,
                STOCK_VENUE,
                None if pd.isna(row["open"]) else float(row["open"]),
                None if pd.isna(row["high"]) else float(row["high"]),
                None if pd.isna(row["low"]) else float(row["low"]),
                float(row["close"]),
                vol,
                0,
            )
        )
    return records


def read_last_bars(cur, symbols: list[str]) -> dict[str, tuple[date, dict[str, float]]]:
    """Newest stored daily bar per symbol (read-only SELECT)."""
    out: dict[str, tuple[date, dict[str, float]]] = {}
    for symbol in symbols:
        cur.execute(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol = %s AND interval = %s AND venue = %s ORDER BY timestamp DESC LIMIT 1",
            (symbol, INTERVAL, STOCK_VENUE),
        )
        row = cur.fetchone()
        if row is not None:
            out[symbol] = (
                row[0].date(),
                dict(zip(COLUMNS, (None if v is None else float(v) for v in row[1:]), strict=True)),
            )
    return out


def fetch_history(symbols: list[str], start: str, through: date) -> dict[str, pd.DataFrame]:
    """yfinance daily bars (auto-adjusted, as the project's loader does), one frame per symbol."""
    import yfinance as yf

    end = (pd.Timestamp(through) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            out[symbol] = raw[symbol] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            out[symbol] = pd.DataFrame()
    return out


def write_backup(plans: list[SymbolPlan], stored: dict, directory: Path, stamp: str) -> Path | None:
    """Save every row that will be rewritten, and the keys of every row that will be added."""
    if not any(p.n_refresh or p.n_insert for p in plans):
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"backfill_{stamp}.csv"
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "symbol", "timestamp", *COLUMNS])
        for plan in plans:
            for ts in plan.inserts.index:
                writer.writerow(["insert", plan.symbol, ts.date().isoformat()])
            if plan.n_refresh and plan.symbol in stored:
                day, old = stored[plan.symbol]
                writer.writerow(
                    ["old_row", plan.symbol, day.isoformat(), *(old[c] for c in COLUMNS)]
                )
    return path


INSERT_SQL = (
    "INSERT INTO ohlcv "
    "(timestamp, symbol, interval, venue, open, high, low, close, volume, trades) "
    "VALUES %s ON CONFLICT (timestamp, symbol, interval, venue) DO NOTHING"
)
REFRESH_SQL = (
    "UPDATE ohlcv SET open = %s, high = %s, low = %s, close = %s, volume = %s "
    "WHERE timestamp = %s AND symbol = %s AND interval = %s AND venue = %s"
)


def commit_plan(conn, plan: SymbolPlan) -> tuple[int, int]:
    """Write one symbol in one transaction; returns ``(inserted, refreshed)``."""
    from psycopg2.extras import execute_values

    inserted = refreshed = 0
    with conn.cursor() as cur:
        if plan.n_insert:
            execute_values(cur, INSERT_SQL, rows_for_db(plan.symbol, plan.inserts), page_size=5000)
            inserted = cur.rowcount
        for ts, row in plan.refresh.iterrows():
            cur.execute(
                REFRESH_SQL,
                (
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    None if pd.isna(row["volume"]) else float(row["volume"]),
                    ts.to_pydatetime().replace(tzinfo=None),
                    plan.symbol,
                    INTERVAL,
                    STOCK_VENUE,
                ),
            )
            refreshed += cur.rowcount
    conn.commit()
    return inserted, refreshed


def parse_symbols(text: str) -> list[str]:
    """Validated, de-duplicated, upper-case symbols from a comma list."""
    symbols = []
    for part in text.split(","):
        symbol = part.strip().upper()
        if not _SYMBOL.match(symbol):
            raise argparse.ArgumentTypeError(f"not a ticker: {part!r}")
        if symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise argparse.ArgumentTypeError("no symbols given")
    return symbols


def build_parser() -> argparse.ArgumentParser:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--symbols", required=True, type=parse_symbols, help="comma-separated tickers")
    p.add_argument(
        "--through", required=True, type=date.fromisoformat, help="last bar date to write"
    )
    p.add_argument("--start", default=DEFAULT_START, help="history start for a symbol with no bars")
    p.add_argument("--refresh-last", action="store_true", help="also fix a partial last stored bar")
    p.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE, help="close gap to refresh"
    )
    p.add_argument("--commit", action="store_true", help="write (default: dry run, no writes)")
    p.add_argument(
        "--backup-dir", type=Path, default=REPO_ROOT / "data" / "exports" / BACKUP_DIRNAME
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Plan (and with ``--commit`` write) the backfill; exit 0 done, 1 a symbol failed."""
    args = build_parser().parse_args(argv)
    if args.through >= datetime.now(timezone.utc).date():
        print("--through must be before today (today's bar is incomplete)", file=sys.stderr)
        return 2
    from ggTrader.data.universe_db import connect

    reader = connect(read_only=True)
    try:
        with reader.cursor() as cur:
            stored = read_last_bars(cur, args.symbols)
    finally:
        reader.close()
    # one download from the earliest start any symbol needs (a symbol with bars needs ~a week back)
    fetched = fetch_history(args.symbols, args.start, args.through)
    plans = [
        plan_symbol(
            s,
            stored.get(s),
            fetched.get(s, pd.DataFrame()),
            through=args.through,
            refresh_last=args.refresh_last,
            tolerance=args.tolerance,
        )
        for s in args.symbols
    ]
    for p in plans:
        span = (
            f"{p.inserts.index.min().date()}..{p.inserts.index.max().date()}" if p.n_insert else "-"
        )
        print(
            f"{p.symbol:6} last stored {str(p.last_stored or 'none'):>10}  "
            f"insert {p.n_insert:5} ({span})"
            f"  refresh {p.n_refresh}  {p.note}"
        )
    if not args.commit:
        print("dry run: nothing written (add --commit)")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = write_backup(plans, stored, args.backup_dir, stamp)
    print(f"backup of changed rows and inserted keys: {backup}")
    failed = 0
    conn = connect()
    try:
        for p in plans:
            try:
                inserted, refreshed = commit_plan(conn, p)
                print(f"{p.symbol}: inserted {inserted}, refreshed {refreshed}")
            except Exception as exc:  # noqa: BLE001 - log and continue with the other symbols
                conn.rollback()
                failed += 1
                print(f"{p.symbol}: FAILED {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
    finally:
        conn.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
