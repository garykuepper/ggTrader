"""Sanity check (not WFO): split IBIT returns into intraday vs overnight, weeknight vs weekend."""

import argparse
import json
from pathlib import Path

import numpy as np
import yfinance as yf

OUT = Path(__file__).with_name("ibit_overnight_split.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2024-01-11")
    ap.add_argument("--end", default="2026-10-24")
    a = ap.parse_args()
    # Unadjusted open/close; IBIT pays no dividends and has not split.
    h = yf.Ticker("IBIT").history(start=a.start, end=a.end, auto_adjust=False)
    o, c = h["Open"], h["Close"]
    intraday = np.log(c / o)
    overnight = np.log(o / c.shift(1)).dropna()
    gap_days = h.index.to_series().diff().dt.days.reindex(overnight.index)
    weekend = overnight[gap_days > 1]  # includes Monday-after-holiday gaps
    weeknight = overnight[gap_days == 1]

    def cum(x):
        return float(np.expm1(x.sum()))

    res = {
        "window": [str(h.index[0].date()), str(h.index[-1].date())],
        "n_days": len(h),
        "close_to_close": cum(np.log(c / c.shift(1)).dropna()),
        "intraday_open_to_close": cum(intraday),
        "overnight_all": cum(overnight),
        "overnight_weeknight": cum(weeknight),
        "overnight_weekend_or_holiday": cum(weekend),
        "n_weeknight": int(len(weeknight)),
        "n_weekend": int(len(weekend)),
        "mean_bp": {
            "intraday": float(intraday.mean() * 1e4),
            "weeknight": float(weeknight.mean() * 1e4),
            "weekend": float(weekend.mean() * 1e4),
        },
        "t_stat": {
            k: float(v.mean() / v.std() * np.sqrt(len(v)))
            for k, v in {
                "intraday": intraday.dropna(),
                "weeknight": weeknight,
                "weekend": weekend,
            }.items()
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
