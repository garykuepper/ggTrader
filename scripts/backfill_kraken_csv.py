"""Backfill the ohlcv table from Kraken's downloadable OHLCVT CSV archive.

Kraken's CCXT OHLC endpoint silently ignores `since` and only returns the
most recent ~720 bars per call, so the live trader's daily incremental
fetches can never reach back further than they were initially seeded. As a
result, some symbols (notably ADA-USD and AAVE-USD) are missing 6+ months
of pre-2024 history plus 90-day-class internal gaps that align across
symbols (ingest pipeline outages, not Kraken delistings). This script fills
those holes from Kraken's authoritative quarterly CSV archive.

Archive layout (mounted into the container at /kraken_archive):
    /kraken_archive/Kraken_OHLCVT_Q{1..4}_{YYYY}/<PAIR>USD_<MIN>.csv
        columns: epoch, open, high, low, close, volume, trades
        timeframes: 1, 5, 15, 60, 720, 1440 (no 240, hence resampling)

Resample 60-min → 4h: open=first, high=max, low=min, close=last,
volume=sum, trades=sum. `ON CONFLICT (timestamp, symbol, interval) DO
NOTHING` so re-runs are idempotent and we never overwrite live-trader-fetched
bars (which carry real fee/trade-count data).

Usage (host):
    docker exec ggtrader_live python /app/scripts/backfill_kraken_csv.py \
        --symbols ADA-USD,AAVE-USD,DOT-USD

    # All symbols already in ohlcv:
    docker exec ggtrader_live python /app/scripts/backfill_kraken_csv.py --all
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

ARCHIVE_ROOT = Path("/kraken_archive")
TARGET_INTERVAL = "4h"
TARGET_RULE = "4h"
# Native 4h CSV files exist from Q1 2024 onward; for 2023 we resample 60-min.
NATIVE_4H_MIN = 240
RESAMPLE_SOURCE_MIN = 60


def _load_symbols_in_db(conn_str: str) -> list[str]:
    conn = psycopg2.connect(conn_str)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT symbol FROM ohlcv WHERE interval = %s ORDER BY symbol",
                (TARGET_INTERVAL,),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def _archive_pair_candidates(symbol: str) -> list[str]:
    """Candidate archive filenames (without the _<min>.csv suffix) for ``symbol``.

    The DB stores modern names (BTC-USD, DOGE-USD); the existing
    ``kraken_map`` in ``data.core.constants`` already normalizes legacy ->
    modern when CSVs are ingested via ``postgres_ingestor``. The downloadable
    archive itself, however, still uses the Kraken-internal codes for some
    pairs (XBTUSD, XDGUSD). We invert ``kraken_map`` to discover the legacy
    code for any modern base and try both forms — modern first, then legacy.
    """
    from ggTrader.data.core.constants import kraken_map

    base = symbol.split("-")[0]
    candidates = [f"{base}USD"]
    # Invert kraken_map (legacy -> modern) and find any legacy code that maps
    # to this modern base. Multiple legacy codes can point at the same modern
    # base (e.g. both XBT and XXBT -> BTC), so collect all of them.
    for legacy, modern in kraken_map.items():
        if modern == base and f"{legacy}USD" not in candidates:
            candidates.append(f"{legacy}USD")
    return candidates


def _read_csv_files(files: list[str]) -> pd.DataFrame:
    frames = []
    for f in files:
        df = pd.read_csv(
            f,
            header=None,
            names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
            dtype={"trades": "int64"},
        )
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return out.set_index("timestamp")


def _load_csv_for_symbol(symbol: str) -> pd.DataFrame:
    """Build a 4h DataFrame for ``symbol`` from the archive.

    Uses native 240-min CSVs where available (Q1 2024+) and resamples 60-min
    CSVs for the older quarters. Returns a UTC-indexed DataFrame, never both
    sources for the same period (240-min wins on overlap to avoid drift).
    """
    native_files: list[str] = []
    sixty_files: list[str] = []
    for pair in _archive_pair_candidates(symbol):
        native_files.extend(sorted(glob(str(ARCHIVE_ROOT / f"Kraken_OHLCVT_Q*_*/{pair}_{NATIVE_4H_MIN}.csv"))))
        sixty_files.extend(sorted(glob(str(ARCHIVE_ROOT / f"Kraken_OHLCVT_Q*_*/{pair}_{RESAMPLE_SOURCE_MIN}.csv"))))

    df_native = _read_csv_files(native_files)
    if not df_native.empty:
        df_native = df_native[~df_native.index.duplicated(keep="last")]
        native_start = df_native.index.min()
    else:
        native_start = None

    # For periods earlier than the native coverage, resample 60-min.
    df_resampled = pd.DataFrame()
    if sixty_files:
        df_60 = _read_csv_files(sixty_files)
        if native_start is not None:
            df_60 = df_60[df_60.index < native_start]
        if not df_60.empty:
            df_resampled = df_60.resample(TARGET_RULE, label="left", closed="left").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "trades": "sum",
                }
            )
            df_resampled = df_resampled.dropna(subset=["open", "high", "low", "close"])

    if df_native.empty and df_resampled.empty:
        return pd.DataFrame()
    if df_native.empty:
        return df_resampled
    if df_resampled.empty:
        return df_native
    out = pd.concat([df_resampled, df_native]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _upsert_bars(conn_str: str, symbol: str, df: pd.DataFrame) -> tuple[int, int]:
    """Insert with ON CONFLICT DO NOTHING. Returns (attempted, actually_inserted).

    DO NOTHING preserves any existing row — including live-trader rows whose
    `trades` column has the true count from CCXT rather than the archive's
    pre-aggregated count.
    """
    if df.empty:
        return (0, 0)

    # IMPORTANT: the `ohlcv.timestamp` column is `timestamp without time zone`.
    # If we pass a tz-aware datetime through psycopg2, it converts to the
    # connection's local TZ (America/Los_Angeles in this container) and
    # strips the tz, shifting every bar by 7-8h depending on DST. To store
    # actual UTC time labels we must convert to tz-naive UTC first.
    records = [
        (
            ts.tz_convert("UTC").tz_localize(None).to_pydatetime(),
            symbol,
            TARGET_INTERVAL,
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]) if not pd.isna(row["volume"]) else None,
            int(row["trades"]) if not pd.isna(row["trades"]) else 0,
        )
        for ts, row in df.iterrows()
    ]
    if not records:
        return (0, 0)

    conn = psycopg2.connect(conn_str)
    conn.autocommit = True
    inserted = 0
    try:
        with conn.cursor() as cur:
            # Use a temp table + INSERT ... SELECT to count actual inserts.
            execute_values(
                cur,
                """
                INSERT INTO ohlcv (timestamp, symbol, interval, open, high, low, close, volume, trades)
                VALUES %s
                ON CONFLICT (timestamp, symbol, interval) DO NOTHING
                """,
                records,
            )
            inserted = cur.rowcount
    finally:
        conn.close()
    return (len(records), inserted)


def _coverage(conn_str: str, symbol: str) -> tuple[int, str | None, str | None]:
    conn = psycopg2.connect(conn_str)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*), MIN(timestamp)::text, MAX(timestamp)::text
                FROM ohlcv WHERE symbol = %s AND interval = %s
                """,
                (symbol, TARGET_INTERVAL),
            )
            return cur.fetchone()
    finally:
        conn.close()


def _resolve_conn_str() -> str:
    from ggTrader.utils.config import get_db_connection_string

    return get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--symbols", help="Comma-separated list, e.g. ADA-USD,AAVE-USD")
    grp.add_argument("--all", action="store_true", help="Backfill every symbol already in ohlcv")
    args = ap.parse_args()

    if not ARCHIVE_ROOT.exists():
        print(f"ERROR: archive not found at {ARCHIVE_ROOT} — mount /media/thesix/Kraken into the container.")
        return 2

    conn_str = _resolve_conn_str()

    if args.all:
        symbols = _load_symbols_in_db(conn_str)
    else:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    print(f"Backfilling {len(symbols)} symbol(s) from {ARCHIVE_ROOT} into interval={TARGET_INTERVAL}.\n")

    grand_total_in = 0
    grand_skipped: list[str] = []
    for sym in symbols:
        before = _coverage(conn_str, sym)
        print(f"[{sym}] before: {before[0]} bars, range {before[1]} → {before[2]}")
        df_4h = _load_csv_for_symbol(sym)
        if df_4h.empty:
            print(f"  no archive CSV for {sym} (looked for {_archive_pair_candidates(sym)}) — skipping\n")
            grand_skipped.append(sym)
            continue
        attempted, _ = _upsert_bars(conn_str, sym, df_4h)
        after = _coverage(conn_str, sym)
        print(
            f"  archive 4h bars: {len(df_4h)}  upsert attempted={attempted}"
        )
        print(f"  after:  {after[0]} bars, range {after[1]} → {after[2]}\n")
        grand_total_in += attempted

    print(f"Done. Total bars upserted={grand_total_in}.")
    if grand_skipped:
        print(f"Skipped (no archive CSV): {grand_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
