"""Causal cross-sectional IC weighting for the ensemble_ic strategy.

All functions are pure functions of their inputs. forward_returns peeks ahead
by `horizon` bars BY DESIGN; the leak guard lives in ic_weight_schedule, which
never consumes a forward return that is not yet realized at a rebalance date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Forward return close[t+horizon]/close[t] - 1 per (date, symbol)."""
    return close.shift(-horizon) / close - 1.0


def daily_cross_sectional_ic(
    raw: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 10
) -> pd.Series:
    """Per-date Spearman rank IC across symbols between `raw` and `fwd`.

    Spearman = Pearson on per-row ranks. Days with fewer than `min_names`
    jointly-valid symbols return NaN.
    """
    raw, fwd = raw.align(fwd, join="inner")
    valid = raw.notna() & fwd.notna()
    n = valid.sum(axis=1)

    rr = raw.where(valid).rank(axis=1)
    fr = fwd.where(valid).rank(axis=1)
    rm = rr.sub(rr.mean(axis=1), axis=0)
    fm = fr.sub(fr.mean(axis=1), axis=0)
    cov = (rm * fm).sum(axis=1)
    denom = np.sqrt((rm**2).sum(axis=1) * (fm**2).sum(axis=1))
    ic = cov / denom.replace(0, np.nan)
    ic[n < min_names] = np.nan
    return ic
