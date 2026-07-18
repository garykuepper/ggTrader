"""One-time historical backfill: FINRA consolidated short interest for the
SP500 universe, from the earliest date the free API actually has data
(~April 2020 -- verified empirically, despite FINRA's own metadata claiming
only a rolling one year) through today.

Settlement dates are DISCOVERED via discover_settlement_dates, not assumed
from a fixed "15th and month-end" calendar rule -- an earlier version of
this script guessed those exact dates and silently missed 71 of 151 real
settlement cycles (47%), because FINRA shifts a settlement date to the
nearest preceding business day whenever the 15th/month-end falls on a
weekend or holiday.

Idempotent: cache_settlement_date upserts, so re-running only refreshes
data, never duplicates. Safe to re-run to pick up newly-published cycles.
"""

from __future__ import annotations

import time

import pandas as pd

from ggTrader.lab.data import equity_universe_between
from ggTrader.lab.short_interest_data import cache_settlement_date, discover_settlement_dates

EARLIEST_SETTLEMENT = "2020-04-15"  # empirically verified earliest available record


def main() -> None:
    eval_start = pd.Timestamp(EARLIEST_SETTLEMENT, tz="UTC")
    eval_end = pd.Timestamp.now(tz="UTC")
    symbols = equity_universe_between(eval_start, eval_end, universe="sp500")
    print(f"Backfilling {len(symbols)} SP500 (ever-member) symbols", flush=True)

    dates = discover_settlement_dates(EARLIEST_SETTLEMENT, str(eval_end.date()))
    print(f"{len(dates)} discovered settlement dates: {dates[0]} .. {dates[-1]}", flush=True)

    total = 0
    for i, d in enumerate(dates):
        n = cache_settlement_date(d, symbols=symbols)
        total += n
        print(f"  [{i + 1}/{len(dates)}] {d}: {n} rows", flush=True)
        time.sleep(0.5)  # be a polite citizen of a free public API

    print(f"Done. {total} total rows cached.", flush=True)


if __name__ == "__main__":
    main()
