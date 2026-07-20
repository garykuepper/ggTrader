"""One-time (re-runnable) backfill: scheduled FOMC announcement dates,
2011-present, into a static CSV cache. Keeps the fomc_drift strategy's
WFO runs fast and independent of the Fed's site being reachable, rather
than re-scraping ~11 pages on every process start.
"""

from __future__ import annotations

import pandas as pd

from ggTrader.lab.fomc_calendar import historical_fomc_announcement_dates

OUT_PATH = "data/universe/fomc_meeting_dates.csv"


def main() -> None:
    dates = historical_fomc_announcement_dates(start_year=2011)
    df = pd.DataFrame({"announcement_date": [d.strftime("%Y-%m-%d") for d in dates]})
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} FOMC announcement dates to {OUT_PATH}")


if __name__ == "__main__":
    main()
