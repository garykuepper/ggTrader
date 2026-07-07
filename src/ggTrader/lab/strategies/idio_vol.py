"""Cross-sectional idiosyncratic-volatility strategy (weight-based)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import eligible_symbols, extract_close
from ggTrader.lab.strategy import LabConfig, Plan


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


class IdioVolStrategy:
    """Long-only defensive-premium sleeve: equal-weight the lowest-idiosyncratic-
    variance quintile of the eligible universe, rebalanced monthly.

    Market factor is the eligible universe's own equal-weighted return (the
    same convention ggTrader.lab.simulate.compute_vol_scalar already uses for
    vol targeting) rather than an external benchmark series, since the
    Strategy protocol's select()/to_targets() only receive the strategy's own
    OHLCV frame.
    """

    name = "idio_vol"
    target_kind = "weights"

    def __init__(self, cfg: LabConfig, reg_window: int = 20, quintile: int = 5) -> None:
        self.cfg = cfg
        self.reg_window = reg_window
        self.quintile = quintile

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "reg_window": [20, 40, 60],
            "quintile": [4, 5],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if len(elig) < self.quintile:
            return []

        close = extract_close(data, elig)
        returns = close.pct_change()
        market_returns = returns.mean(axis=1)
        resid_var = idiosyncratic_variance(returns, market_returns, self.reg_window)

        latest = resid_var.iloc[-1].dropna()
        if len(latest) < self.quintile:
            return []

        ranked = latest.sort_values()  # ascending: lowest residual variance first
        bucket_size = max(1, len(ranked) // self.quintile)
        bottom = ranked.index[:bucket_size].tolist()
        if not bottom:
            return []

        weight = 1.0 / len(bottom)
        return [{"symbol": s, "weight": weight} for s in bottom]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0  # default: exit anything not re-selected
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets
