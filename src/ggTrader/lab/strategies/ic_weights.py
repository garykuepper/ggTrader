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


def ic_weight_schedule(
    raw_by_voter: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    *,
    lookback_months: int,
    horizon: int = 3,
    rebalance: str = "QE",
    min_names: int = 10,
) -> pd.DataFrame:
    """Causal (time x voter) weight schedule, recomputed each rebalance date.

    At each rebalance date t_k: take the trailing `lookback_months` window,
    DROP its last `horizon` bars (their forward returns are not realized by
    t_k -> the leak guard), average each voter's daily IC over the window, and
    set w_j = max(0, IC_j) / sum_k max(0, IC_k). Warmup and all-non-positive
    windows fall back to equal weights. Weights forward-fill until the next
    rebalance date.
    """
    voters = list(raw_by_voter)
    eq = 1.0 / len(voters)
    fwd = forward_returns(close, horizon)
    daily_ic = pd.DataFrame(
        {j: daily_cross_sectional_ic(raw_by_voter[j], fwd, min_names) for j in voters}
    )

    index = close.index
    rebal_dates = pd.Series(index, index=index).resample(rebalance).last().dropna()
    lookback = pd.DateOffset(months=lookback_months)

    weights = pd.DataFrame(index=index, columns=voters, dtype=float)
    for t_k in rebal_dates:
        window_start = t_k - lookback
        cutoff = index[index <= t_k]
        # leak guard: last usable bar is `horizon` bars before t_k
        usable_end = cutoff[-(horizon + 1)] if len(cutoff) > horizon else None
        if usable_end is None or window_start < index[0]:
            w = pd.Series(eq, index=voters)  # warmup: not a full window yet
        else:
            win = daily_ic.loc[(daily_ic.index > window_start) & (daily_ic.index <= usable_end)]
            ic = win.mean()
            pos = ic.clip(lower=0.0)
            total = pos.sum()
            w = pos / total if total > 0 else pd.Series(eq, index=voters)
        weights.loc[t_k:] = w.values

    return weights.fillna(eq)  # pre-first-rebalance rows -> equal weight
