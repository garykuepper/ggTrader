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
import vectorbt as vbt


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


def simulate_hold_weights(
    ohlcv: pd.DataFrame,
    weights: Dict[str, float],
    asof: pd.Timestamp,
    month_end: pd.Timestamp,
    base_config: Dict[str, Any],
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Buy target weights at the first bar after ``asof``, hold to ``month_end``.

    Symbols with no price at the first forward bar are dropped (their weight
    stays in cash). Mid-month gaps are forward-filled.
    """
    empty = pd.Series(dtype=float)
    month_mask = (ohlcv.index > asof) & (ohlcv.index <= month_end)
    if not month_mask.any() or not weights:
        return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

    have = set(ohlcv.columns.get_level_values(0))
    close = (
        pd.concat({s: ohlcv[s]["close"] for s in weights if s in have}, axis=1)
        .loc[month_mask]
        .ffill()
    )
    close = close.dropna(axis=1)  # NaN after ffill == no price at month start
    if close.shape[1] == 0:
        return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

    size = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    size.iloc[0] = [weights[s] for s in close.columns]
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=size,
        size_type="targetpercent",
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
        cash_sharing=True,
        group_by=np.full(close.shape[1], 0),
        call_seq="auto",
    ).copy()

    returns = pf.returns()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    diags = {
        "n_positions": int(close.shape[1]),
        "n_trades": int(pf.trades.count().sum()),
        "avg_exposure": float(_portfolio_exposure(pf).mean()),
        "month_return_pct": float((1.0 + returns).prod() - 1.0) * 100,
    }
    return returns, diags


class DualMomentum(CrossSectionalMomentum):
    name = "dual_momentum"  # behavior added in Task 3
