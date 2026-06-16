# src/ggTrader/lab/strategies/signals.py
"""Signal-based lab strategies: entry/exit boolean signals via from_signals."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


class EmaCrossSignal:
    """EMA crossover signal strategy.

    select() returns all eligible symbols with fixed EMA params.
    to_targets() computes whole-window entry/exit signals using pandas ewm.
    """

    name = "ema_cross"
    target_kind = "signals"

    def __init__(self, cfg: LabConfig, ema_fast: int = 20, ema_slow: int = 50) -> None:
        self.cfg = cfg
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        """All eligible symbols with enough history — fixed EMA params."""
        data = data.loc[:asof]
        have = set(data.columns.get_level_values(0).unique())
        return [
            {
                "symbol": s,
                "weight": 0.0,
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
            }
            for s in eligible
            if s in have and len(data[s]["close"].dropna()) >= self.cfg.min_history_bars
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        """Compute EMA cross signals over the full window for all selected symbols."""
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )

        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()

        entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
        exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)

        return SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))


_SIGNAL_REGISTRY = {
    "ema_cross": EmaCrossSignal,
}

SIGNAL_STRATEGY_NAMES = tuple(_SIGNAL_REGISTRY)


def build_signal_strategy(name: str, cfg: LabConfig) -> EmaCrossSignal:
    if name not in _SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal strategy {name!r}. Available: {SIGNAL_STRATEGY_NAMES}")
    return _SIGNAL_REGISTRY[name](cfg)
