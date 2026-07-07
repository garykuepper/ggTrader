"""Cross-sectional idiosyncratic-volatility strategy (weight-based)."""

from __future__ import annotations

import pandas as pd


def idiosyncratic_variance(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Rolling residual variance of each symbol vs. a single market-factor return.

    beta_t = Cov(r_i, r_m)_t / Var(r_m)_t (rolling, causal as of bar t);
    resid_t = r_i,t - beta_t * r_m,t; output = rolling variance of resid.
    A near-zero rolling market variance (e.g. a flat/constant market factor)
    produces beta = inf/NaN by division, which is expected and handled by the
    caller (NaN residual variance sorts last / is dropped, never selected).
    """
    market_returns = market_returns.reindex(returns.index)
    market_var = market_returns.rolling(window=window, min_periods=window).var()

    resid_var = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        cov = returns[col].rolling(window=window, min_periods=window).cov(market_returns)
        beta = cov / market_var
        resid = returns[col] - beta * market_returns
        resid_var[col] = resid.rolling(window=window, min_periods=window).var()
    return resid_var
