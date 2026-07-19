"""One-time historical backfill: Google Trends search-interest for the
SP500 universe, 2015-present. One query per symbol (each returns the full
window as monthly-resolution data directly, verified 2026-07-19 -- no
query-stitching needed). Single-threaded with an explicit delay between
requests -- pytrends is an unofficial scraper of Google's own UI backend,
not a supported public API, so this stays conservative for a full-scale
sustained run even though a quick rate-limit spot-check (39/40 succeeded
with zero delay) suggested more headroom than expected.
"""

from __future__ import annotations

import time

import pandas as pd

from ggTrader.lab.data import equity_universe_between
from ggTrader.lab.google_trends_data import cache_symbol_interest, ensure_schema

REQUEST_DELAY_SECONDS = 2.0
EARLIEST_DATE = "2015-01-01"


def main() -> None:
    ensure_schema()
    eval_start = pd.Timestamp(EARLIEST_DATE, tz="UTC")
    eval_end = pd.Timestamp.now(tz="UTC")
    symbols = equity_universe_between(eval_start, eval_end, universe="sp500")
    print(
        f"Backfilling Google Trends data for {len(symbols)} SP500 (ever-member) symbols", flush=True
    )

    end_str = str(eval_end.date())
    total_rows = 0
    errors = 0
    for i, symbol in enumerate(symbols):
        try:
            n = cache_symbol_interest(symbol, EARLIEST_DATE, end_str)
            total_rows += n
            print(f"  [{i + 1}/{len(symbols)}] {symbol}: {n} rows", flush=True)
        except Exception as e:  # noqa: BLE001 -- one bad symbol must not kill the whole backfill
            errors += 1
            print(f"  [{i + 1}/{len(symbols)}] {symbol}: ERROR {e}", flush=True)
        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Done. {total_rows} total rows cached, {errors} errors.", flush=True)


if __name__ == "__main__":
    main()
