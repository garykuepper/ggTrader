"""Sanity check (NOT a WFO): static SPY/SSO mixes vs SPY, plus two slow SSO-sleeve timing rules.

Pre-tax, dividend-adjusted closes from yfinance, signal on close t, trade at close t+1.
Intent: is static ~1.5x or a slow leverage-sleeve switch even in the running before taxes?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path(__file__).with_suffix(".json")


def stats(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {
        "label": label,
        "cagr": round(float(cagr), 4),
        "vol": round(float(r.std() * np.sqrt(252)), 4),
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3),
        "max_dd": round(float(dd), 4),
        "end_wealth": round(float(eq.iloc[-1]), 3),
    }


def switches(w: pd.Series) -> float:
    """Sleeve state changes per year."""
    return round(float((w.diff().abs() > 0).sum() / (len(w) / 252)), 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2006-07-01")
    p.add_argument("--end", default="2026-09-29")
    a = p.parse_args()
    px = yf.download(
        ["SPY", "SSO", "BIL"], start="2006-01-01", end=a.end, auto_adjust=True, progress=False
    )["Close"]
    px = px.dropna(subset=["SPY", "SSO"])
    r = px.pct_change()
    r["BIL"] = r["BIL"].fillna(0.0)  # BIL starts 2007-05; treat pre-inception cash as 0%
    sma200 = px["SPY"].rolling(200).mean()
    m_close = px["SPY"].resample("ME").last()
    sma10m = m_close.rolling(10).mean()
    r = r.loc[a.start :]
    out = []
    # static mixes, daily rebalanced (idealized; band-rebalanced would drift toward the winner)
    for w in (0.0, 0.3, 0.5, 1.0):
        out.append(
            stats(
                (1 - w) * r["SPY"] + w * r["SSO"],
                f"static SPY/SSO {int((1 - w) * 100)}/{int(w * 100)} (~{1 + w:.1f}x)",
            )
        )
    # Rule A: 200d SMA daily, whole book SSO / BIL (Gayed-Bilello style, 2x)
    on = (px["SPY"] > sma200).shift(1).reindex(r.index).fillna(False)
    ra = np.where(on, r["SSO"], r["BIL"])
    d = stats(pd.Series(ra, index=r.index), "200d SMA daily: SSO / BIL (2x on, cash off)")
    d["switches_per_yr"] = switches(on.astype(int))
    out.append(d)
    # Rule B: 200d SMA daily, sleeve only: 50% SPY core + 50% (SSO if on else SPY)
    rb = 0.5 * r["SPY"] + 0.5 * np.where(on, r["SSO"], r["SPY"])
    d = stats(pd.Series(rb, index=r.index), "200d SMA daily, sleeve only: 50 SPY + 50 (SSO|SPY)")
    d["switches_per_yr"] = switches(on.astype(int))
    out.append(d)
    # Rule C: 10-month SMA month-end with 2% buffer, sleeve only 40% (report #2's A2 spec)
    state = pd.Series(index=m_close.index, dtype=float)
    cur = 0.0
    for t in m_close.index:
        if pd.isna(sma10m.loc[t]):
            state.loc[t] = np.nan
            continue
        ratio = m_close.loc[t] / sma10m.loc[t]
        if ratio >= 1.02:
            cur = 1.0
        elif ratio <= 0.98:
            cur = 0.0
        state.loc[t] = cur
    on_m = state.reindex(r.index, method="ffill").shift(1).fillna(0.0)
    rc = 0.6 * r["SPY"] + 0.4 * np.where(on_m > 0, r["SSO"], r["SPY"])
    d = stats(
        pd.Series(rc, index=r.index), "10m SMA +/-2% month-end, sleeve only: 60 SPY + 40 (SSO|SPY)"
    )
    d["switches_per_yr"] = switches(on_m)
    out.append(d)
    # era splits for the two most relevant lines
    eras = {
        "2006-07..2012": ("2006-07-01", "2012-12-31"),
        "2013..2019": ("2013-01-01", "2019-12-31"),
        "2020..2026-09": ("2020-01-01", a.end),
    }
    era_rows = []
    for name, (s, e) in eras.items():
        sl = slice(s, e)
        era_rows.append(
            {
                "era": name,
                "SPY": stats(r["SPY"].loc[sl], "SPY")["cagr"],
                "static_50_50": stats((0.5 * r["SPY"] + 0.5 * r["SSO"]).loc[sl], "s")["cagr"],
                "sleeve_200d": stats(pd.Series(rb, index=r.index).loc[sl], "b")["cagr"],
                "sleeve_10m": stats(pd.Series(rc, index=r.index).loc[sl], "c")["cagr"],
                "SPY_dd": stats(r["SPY"].loc[sl], "SPY")["max_dd"],
                "static_50_50_dd": stats((0.5 * r["SPY"] + 0.5 * r["SSO"]).loc[sl], "s")["max_dd"],
                "sleeve_200d_dd": stats(pd.Series(rb, index=r.index).loc[sl], "b")["max_dd"],
                "sleeve_10m_dd": stats(pd.Series(rc, index=r.index).loc[sl], "c")["max_dd"],
            }
        )
    res = {
        "window": [a.start, a.end],
        "note": "pre-tax, daily-rebalanced static mixes, next-close execution",
        "full": out,
        "eras": era_rows,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(pd.DataFrame(out).to_string(index=False))
    print()
    print(pd.DataFrame(era_rows).to_string(index=False))


if __name__ == "__main__":
    main()
