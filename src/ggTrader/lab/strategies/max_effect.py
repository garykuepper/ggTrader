"""MAX effect / lottery-demand anomaly (weight-based)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import eligible_symbols, extract_close
from ggTrader.lab.strategy import LabConfig, Plan


def trailing_max_return(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling max of daily returns over a trailing window, per symbol --
    Bali/Cakici/Whitelaw's MAX: the single highest daily return in the
    trailing month. Causal (min_periods=window): warmup bars are NaN, never
    ranked/selected."""
    return returns.rolling(window=window, min_periods=window).max()


class MaxEffectStrategy:
    """Long-only defensive/behavioral sleeve: equal-weight the LOWEST-MAX
    quintile of the eligible universe (avoids the high-MAX "lottery" decile
    that Bali, Cakici & Whitelaw 2011 find underperforms), rebalanced
    monthly. Portfolio-construction filter in spirit, run standalone here
    like idio_vol.py's equivalent quintile-bucket sleeve.
    """

    name = "max_effect"
    target_kind = "weights"

    def __init__(self, cfg: LabConfig, window: int = 21, quintile: int = 5) -> None:
        self.cfg = cfg
        self.window = window
        self.quintile = quintile

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "window": [21, 42, 63],
            "quintile": [4, 5],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if len(elig) < self.quintile:
            return []

        close = extract_close(data, elig)
        returns = close.pct_change()
        max_ret = trailing_max_return(returns, self.window)

        latest = max_ret.iloc[-1].dropna()
        if len(latest) < self.quintile:
            return []

        ranked = latest.sort_values()  # ascending: lowest MAX (least lottery-like) first
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
