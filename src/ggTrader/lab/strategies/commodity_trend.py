"""Commodity medium-term trend -- candidate A3 from WEB_RESEARCH_CANDIDATES.md's
2026-07-19 cross-asset register. Cross-sectional 12-1 month momentum across
a liquid single-commodity ETF universe, with a volatility-regime filter to
avoid crash periods.

Source: Bloomberg Professional Services, "Capturing curve, carry and trend
premia in commodity markets" (Feb 2026) -- practitioner-grade commentary
for Bloomberg's BERY index, not independent academic research.

Mechanism split from the register's original combined "carry+trend+basis
reversal" idea (see A2/A4 for the other two, untested here) -- this is the
trend leg alone. Reuses this lab's established 12-1 cross-sectional
momentum pattern (momentum.py's CrossSectionalMomentum), applied to a
fixed commodity-ETF universe instead of an equity index, plus a
market-wide realized-volatility regime filter: when the commodity
universe's average trailing realized vol is itself running unusually high
relative to its own recent history (z-score above a threshold), skip the
rebalance entirely (go to cash) rather than chase momentum into a
volatility spike.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import eligible_symbols, extract_close
from ggTrader.lab.strategy import LabConfig, Plan

#: Liquid single-commodity ETFs: metals, energy, agriculture. Broad
#: multi-commodity baskets (DBC/GSG) deliberately excluded -- they'd
#: dilute the cross-sectional ranking, not add a distinct commodity.
COMMODITY_TREND_UNIVERSE = [
    "GLD",
    "SLV",
    "USO",
    "UNG",
    "DBA",
    "CORN",
    "WEAT",
    "SOYB",
    "CANE",
    "CPER",
    "PALL",
    "PPLT",
    "UGA",
    "DBB",
]


def cross_sectional_momentum_scores(close: pd.DataFrame, lookback: int, skip: int) -> pd.Series:
    """Latest per-symbol 12-1-style momentum score: return from
    (lookback+1) bars ago to (skip+1) bars ago. Symbols with insufficient
    history are simply absent from the result, not zero-filled."""
    scores: Dict[str, float] = {}
    for sym in close.columns:
        closes = close[sym].dropna()
        if len(closes) < lookback + 1:
            continue
        past = float(closes.iloc[-(lookback + 1)])
        recent = float(closes.iloc[-(skip + 1)]) if skip > 0 else float(closes.iloc[-1])
        if past <= 0.0 or not np.isfinite(past) or not np.isfinite(recent):
            continue
        scores[sym] = recent / past - 1.0
    return pd.Series(scores, dtype=float)


def regime_vol_zscore(returns: pd.DataFrame, vol_lookback: int, zscore_window: int) -> float:
    """Latest z-score of the commodity universe's average trailing
    realized volatility against its own recent history. Positive = the
    market is running hotter than usual; not a per-symbol filter."""
    per_symbol_vol = returns.rolling(vol_lookback, min_periods=vol_lookback).std()
    market_vol = per_symbol_vol.mean(axis=1)
    hist = market_vol.dropna()
    if len(hist) < max(zscore_window // 3, 5):
        return 0.0
    window = hist.iloc[-zscore_window:]
    mean, std = window.mean(), window.std()
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float((window.iloc[-1] - mean) / std)


class CommodityTrendStrategy:
    """Long-only weights sleeve: equal-weight the top-N commodity ETFs by
    12-1 cross-sectional momentum, rebalanced monthly, skipping the
    rebalance entirely (cash) when the universe's realized-vol regime is
    running unusually hot."""

    name = "commodity_trend"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        top_n: int = 5,
        vol_lookback: int = 20,
        zscore_window: int = 252,
        vol_z_threshold: float = 2.0,
    ) -> None:
        self.cfg = cfg
        self.top_n = top_n
        self.vol_lookback = vol_lookback
        self.zscore_window = zscore_window
        self.vol_z_threshold = vol_z_threshold

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "top_n": [3, 5, 7],
            "vol_z_threshold": [1.5, 2.0, 3.0],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if len(elig) < self.top_n:
            return []

        close = extract_close(data, elig)
        returns = close.pct_change(fill_method=None)
        vol_z = regime_vol_zscore(returns, self.vol_lookback, self.zscore_window)
        if vol_z > self.vol_z_threshold:
            return []

        scores = cross_sectional_momentum_scores(close, self.cfg.lookback, self.cfg.skip)
        if scores.empty:
            return []
        ranked = scores.sort_values(ascending=False).head(self.top_n)
        if ranked.empty:
            return []

        weight = 1.0 / len(ranked)
        return [{"symbol": s, "weight": weight} for s in ranked.index]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets
