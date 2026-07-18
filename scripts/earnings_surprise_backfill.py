"""One-time historical backfill: yfinance earnings-surprise history for the
SP500 (large-cap control -- literature says PEAD should be weak/dead here)
and Russell 2000 (lower-coverage test universe, per the candidate's own
framing) symbol universes.

One yfinance API call per symbol (not per date, unlike short-interest's
FINRA pagination) -- ~1-2s per symbol. Idempotent: cache_symbol_surprises
upserts, so re-running only refreshes data, never duplicates.
"""

from __future__ import annotations

import time

import pandas as pd

from ggTrader.lab.data import equity_universe_between
from ggTrader.lab.earnings_surprise_data import cache_symbol_surprises

LOOKBACK_LIMIT = 100  # yfinance quarters per symbol -- covers back to ~2002 for most


def main() -> None:
    eval_start = pd.Timestamp("2015-01-01", tz="UTC")
    eval_end = pd.Timestamp.now(tz="UTC")

    sp500 = set(equity_universe_between(eval_start, eval_end, universe="sp500"))
    russell = set(equity_universe_between(eval_start, eval_end, universe="russell2000"))
    symbols = sorted(sp500 | russell)
    print(
        f"Backfilling {len(symbols)} symbols (SP500={len(sp500)}, "
        f"Russell2000={len(russell)}, overlap={len(sp500 & russell)})",
        flush=True,
    )

    total = 0
    failures = []
    for i, sym in enumerate(symbols):
        try:
            n = cache_symbol_surprises(sym, limit=LOOKBACK_LIMIT)
        except Exception as e:  # noqa: BLE001 -- one bad symbol must not kill the whole backfill
            failures.append(sym)
            n = 0
            print(f"  [{i + 1}/{len(symbols)}] {sym}: ERROR {e}", flush=True)
        else:
            print(f"  [{i + 1}/{len(symbols)}] {sym}: {n} rows", flush=True)
        total += n
        time.sleep(0.2)

    print(
        f"Done. {total} total rows cached. {len(failures)} symbols failed: {failures}", flush=True
    )


if __name__ == "__main__":
    main()
