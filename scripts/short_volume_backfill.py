"""One-time historical backfill: FINRA daily short-sale volume for the
SP500 universe, from the verified CDN retention boundary (2018-08-01)
through present. One file per business day (~2,000+ requests). Market
holidays 404/403 (no file published that day) -- caught and counted as
errors along with any genuine fetch failure, since both raise the same
HTTPError from urllib; there are only ~9 US market holidays/year (~70-80
total over the backfill window) so this doesn't meaningfully pollute the
error count.

Idempotent: cache_daily_file upserts, so re-running only refreshes data,
never duplicates.
"""

from __future__ import annotations

import time

import pandas as pd

from ggTrader.lab.data import equity_universe_between
from ggTrader.lab.short_volume_data import EARLIEST_DATE, cache_daily_file


def main() -> None:
    eval_start = pd.Timestamp(EARLIEST_DATE, tz="UTC")
    eval_end = pd.Timestamp.now(tz="UTC")
    symbols = equity_universe_between(eval_start, eval_end, universe="sp500")
    print(f"Backfilling {len(symbols)} SP500 (ever-member) symbols", flush=True)

    dates = pd.bdate_range(EARLIEST_DATE, str(eval_end.date()))
    print(f"{len(dates)} business days: {dates[0].date()} .. {dates[-1].date()}", flush=True)

    total = 0
    errors = 0
    for i, d in enumerate(dates):
        date_str = d.strftime("%Y-%m-%d")
        try:
            n = cache_daily_file(date_str, symbols=symbols)
            total += n
        except Exception:  # noqa: BLE001 -- one bad date (or a holiday) must not kill the whole backfill
            errors += 1
        if (i + 1) % 100 == 0:
            print(
                f"  [{i + 1}/{len(dates)}] {date_str}: {total} rows so far, {errors} errors",
                flush=True,
            )
        time.sleep(0.3)

    print(
        f"Done. {total} total rows cached, {errors} errors (includes market holidays).", flush=True
    )


if __name__ == "__main__":
    main()
