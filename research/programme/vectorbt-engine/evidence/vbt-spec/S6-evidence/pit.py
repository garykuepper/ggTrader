"""S6 prototype: fully vectorised point-in-time universe mask (dates x symbols, bool).

Replaces sweep.pit_entry_mask (one eligible_at call per bar, O(T^2)). No per-day Python loop:
membership = step function over the constituent-change snapshots (searchsorted + fancy index),
history rule = close.notna().cumsum(), optional alphabetical max_stocks cap and per-sector cap
via cumsum. Semantics copied from lab/data.py:eligible_at and strategies/registry.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ggTrader.data.core.index_constituents import load_sp500_history, normalize_yf_ticker


def membership_mask(index: pd.DatetimeIndex, columns, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """True where the symbol is an S&P 500 member on that bar (latest snapshot <= bar date)."""
    hist = history if history is not None else load_sp500_history()
    idx = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    if len(idx) and idx[0] < hist.index[0]:
        raise ValueError(f"index starts {idx[0].date()} before constituent history {hist.index[0].date()}")
    pos = hist.index.searchsorted(idx, side="right") - 1          # snapshot row in force per bar
    used = np.unique(pos)
    col_pos = {c: i for i, c in enumerate(columns)}
    snap = np.zeros((len(hist), len(columns)), dtype=bool)         # only rows in `used` are filled
    for p in used:
        for t in hist["tickers"].iloc[p]:
            j = col_pos.get(normalize_yf_ticker(t))
            if j is not None:
                snap[p, j] = True
    return pd.DataFrame(snap[pos], index=index, columns=list(columns))


def pit_mask(close: pd.DataFrame, min_history_bars: int = 400, max_stocks: int | None = None,
             max_sector_count: int | None = None, sector_map: dict[str, str] | None = None,
             history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily PIT eligibility == eligible_at(day, ohlcv.loc[:day]) for every day, vectorised."""
    cols = sorted(close.columns)                                    # eligible_at sorts for max_stocks
    close = close[cols]
    mask = membership_mask(close.index, cols, history)
    mask &= (close.notna().cumsum() >= min_history_bars).to_numpy()
    if max_stocks is not None:                                      # sorted(eligible)[:max_stocks]
        mask &= (mask.cumsum(axis=1) <= max_stocks).to_numpy()
    if max_sector_count is not None:                                # registry.apply_sector_constraints
        sec = pd.Series([(sector_map or {}).get(c, "Unknown") for c in cols], index=cols)
        keep = pd.DataFrame(False, index=mask.index, columns=cols)
        for _, members in sec.groupby(sec):
            m = mask[members.index]
            keep[members.index] = (m.cumsum(axis=1) <= max_sector_count) & m
        mask = keep
    return mask
