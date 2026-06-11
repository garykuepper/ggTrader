"""Pluggable monthly strategies for the honest walk-forward harness.

A MonthlyStrategy maps (data <= T, point-in-time eligible universe) to
next-month selections, then simulates the forward month with those frozen
selections. The select/simulate split is what makes the generic leak check
possible: ``select`` must be a pure function of data <= T.

The harness (research/monthly_walkforward.py) guarantees ``select`` only ever
receives data truncated to <= asof.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

import numpy as np
import pandas as pd


class MonthlyStrategy(Protocol):
    """Contract for a strategy runnable by run_monthly_walkforward."""

    name: str

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        """JSON-able selection records (each with at least "symbol"); data <= asof."""
        ...

    def simulate(
        self,
        ohlcv: pd.DataFrame,
        selections: List[Dict[str, Any]],
        asof: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Daily portfolio returns for (asof, month_end] plus diagnostics."""
        ...


def _portfolio_exposure(pf) -> pd.Series:
    """Fraction of capital deployed per bar: 1 - cash/value (grouped portfolio)."""
    cash, value = pf.cash(), pf.value()
    if isinstance(cash, pd.DataFrame):
        cash = cash.iloc[:, 0]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return 1.0 - cash / value


class CrossSectionalMomentum:
    """12-1 cross-sectional momentum: top-N by trailing return, equal weight."""

    name = "xs_momentum"

    def __init__(
        self,
        cfg,
        base_config: Dict[str, Any],
        lookback: int = 252,
        skip: int = 21,
    ) -> None:
        self.cfg = cfg
        self.base_config = base_config
        self.lookback = lookback
        self.skip = skip

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        for sym in eligible:
            closes = ohlcv[sym]["close"].dropna()
            if len(closes) < self.lookback + 1:
                continue
            past = float(closes.iloc[-(self.lookback + 1)])
            recent = float(closes.iloc[-(self.skip + 1)])
            if past <= 0.0 or not np.isfinite(past) or not np.isfinite(recent):
                continue
            scores[sym] = recent / past - 1.0
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.cfg.top_n]
        if not ranked:
            return []
        weight = 1.0 / len(ranked)
        return [{"symbol": s, "weight": weight, "momentum": m} for s, m in ranked]

    def simulate(
        self,
        ohlcv: pd.DataFrame,
        selections: List[Dict[str, Any]],
        asof: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        weights = {s["symbol"]: float(s["weight"]) for s in selections}
        return simulate_hold_weights(ohlcv, weights, asof, month_end, self.base_config)


def simulate_hold_weights(ohlcv, weights, asof, month_end, base_config):
    raise NotImplementedError  # implemented in Task 2


class DualMomentum(CrossSectionalMomentum):
    name = "dual_momentum"  # behavior added in Task 3
