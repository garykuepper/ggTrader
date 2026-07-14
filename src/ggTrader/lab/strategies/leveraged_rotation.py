"""Breadth-driven leveraged/inverse index rotation (weight-based, monthly).

Rotates between a universe's leveraged-long ETF, its inverse ETF, and cash,
driven by the breadth of the existing validated EnsembleSignal across that
universe's own constituent stocks -- a repurposing of a stock-picking signal
as an index-timing feature. See
docs/superpowers/specs/2026-07-14-leveraged-index-rotation-design.md.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig, Plan


def compute_breadth(entries: pd.DataFrame) -> pd.Series:
    """Fraction of columns with an active (True) entry signal per row.

    Fixed denominator = entries.shape[1] (the full breadth-universe size),
    not the count of symbols currently past warmup -- see spec for why.
    """
    if entries.shape[1] == 0:
        return pd.Series(0.0, index=entries.index)
    return entries.sum(axis=1) / entries.shape[1]


def rotate_positions(
    breadth: pd.Series,
    upper_threshold: float,
    lower_threshold: float,
    min_hold_months: int,
) -> pd.Series:
    """Per-date state in {"long", "inverse", "cash"} with hysteresis.

    A state change only takes effect once the new raw signal (breadth vs.
    thresholds) has held for min_hold_months consecutive dates in the input
    series. The first date's state takes effect immediately.
    """
    raw = pd.Series("cash", index=breadth.index, dtype=object)
    raw[breadth > upper_threshold] = "long"
    raw[breadth < lower_threshold] = "inverse"

    states: list[str] = []
    current: str | None = None
    streak_value: str | None = None
    streak_len = 0
    for val in raw:
        if val == streak_value:
            streak_len += 1
        else:
            streak_value = val
            streak_len = 1
        if current is None:
            current = val
        elif val != current and streak_len >= min_hold_months:
            current = val
        states.append(current)
    return pd.Series(states, index=breadth.index)


class _LeveragedRotationBase:
    """Rotates between a leveraged-long ETF, an inverse ETF, and cash,
    driven by monthly breadth of EnsembleSignal across the universe's own
    constituent stocks. Subclasses fix PAIR_3X/PAIR_2X (see Task 3)."""

    name: str
    target_kind = "weights"
    PAIR_3X: tuple[str, str]
    PAIR_2X: tuple[str, str]

    def __init__(
        self,
        cfg: LabConfig,
        upper_threshold: float = 0.6,
        lower_threshold: float = 0.4,
        min_hold_months: int = 1,
        leverage_tier: str = "3x",
    ) -> None:
        self.cfg = cfg
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.min_hold_months = min_hold_months
        self.leverage_tier = leverage_tier

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "upper_threshold": [0.55, 0.60, 0.65],
            "lower_threshold": [0.35, 0.40, 0.45],
            "min_hold_months": [1, 2, 3],
            "leverage_tier": ["2x", "3x"],
        }

    def _pair(self) -> tuple[str, str]:
        return self.PAIR_3X if self.leverage_tier == "3x" else self.PAIR_2X

    def _all_etf_tickers(self) -> set[str]:
        return set(self.PAIR_3X) | set(self.PAIR_2X)

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        long_t, inv_t = self._pair()
        return [{"symbol": s, "weight": 0.0} for s in (long_t, inv_t) if s in eligible]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        long_t, inv_t = self._pair()
        rebalance_dates = sorted(plans.keys())
        if not rebalance_dates or data.empty:
            return pd.DataFrame(columns=[long_t, inv_t])

        have = set(data.columns.get_level_values(0).unique())
        breadth_symbols = sorted(have - self._all_etf_tickers())

        ensemble = EnsembleSignal(self.cfg)
        placeholder = [{"symbol": s, "weight": 0.0} for s in breadth_symbols]
        signal_targets = ensemble.to_targets({data.index[0]: placeholder}, data)
        breadth = compute_breadth(signal_targets.entries)

        monthly_breadth = breadth.reindex(rebalance_dates)
        states = rotate_positions(
            monthly_breadth, self.upper_threshold, self.lower_threshold, self.min_hold_months
        )

        targets = pd.DataFrame(np.nan, index=data.index, columns=[long_t, inv_t])
        for asof in rebalance_dates:
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            state = states.loc[asof]
            targets.loc[bar, long_t] = 1.0 if state == "long" else 0.0
            targets.loc[bar, inv_t] = 1.0 if state == "inverse" else 0.0
        return targets
