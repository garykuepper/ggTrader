"""Signal ensemble: enter when N-of-M sub-signals agree on the same bar+symbol."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ggTrader.lab.strategies.indicators import bb_signals, ema_signals, rsi_signals
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


class EnsembleSignal:
    """Majority-vote ensemble: enter when >= min_agree sub-signals fire together.

    Sub-signals: bb_reversion, rsi_reversion, ema_cross.
    Exit: when >= min_agree sub-signals fire an exit.
    """

    name = "ensemble"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        min_agree: int = 2,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_exit: int = 50,
        ema_fast: int = 20,
        ema_slow: int = 50,
    ) -> None:
        self.cfg = cfg
        self.min_agree = min_agree
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_exit = rsi_exit
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "min_agree": [2, 3],
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "rsi_period": [7, 14],
            "rsi_oversold": [25, 30],
            "ema_fast": [10, 20],
            "ema_slow": [50, 100],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        have = set(data.columns.get_level_values(0).unique())
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible
            if s in have and len(data[s]["close"].dropna()) >= self.cfg.min_history_bars
        ]

    def _generate_signals(self, close: pd.DataFrame) -> SignalTargets:
        """Run all 3 sub-signals, sum entry/exit votes, threshold at min_agree."""
        bb_ent, bb_ext = bb_signals(close, self.bb_period, self.bb_std)
        rsi_ent, rsi_ext = rsi_signals(close, self.rsi_period, self.rsi_oversold, self.rsi_exit)
        ema_ent, ema_ext = ema_signals(close, self.ema_fast, self.ema_slow)

        entry_votes = bb_ent.astype(int) + rsi_ent.astype(int) + ema_ent.astype(int)
        exit_votes = bb_ext.astype(int) + rsi_ext.astype(int) + ema_ext.astype(int)

        entries = (entry_votes >= self.min_agree).astype(bool)
        exits = (exit_votes >= self.min_agree).astype(bool)
        return SignalTargets(entries=entries, exits=exits)

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        return self._generate_signals(close)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            strat = EnsembleSignal(
                self.cfg,
                min_agree=int(combo.get("min_agree", self.min_agree)),
                bb_period=int(combo.get("bb_period", self.bb_period)),
                bb_std=float(combo.get("bb_std", self.bb_std)),
                rsi_period=int(combo.get("rsi_period", self.rsi_period)),
                rsi_oversold=int(combo.get("rsi_oversold", self.rsi_oversold)),
                rsi_exit=int(combo.get("rsi_exit", self.rsi_exit)),
                ema_fast=int(combo.get("ema_fast", self.ema_fast)),
                ema_slow=int(combo.get("ema_slow", self.ema_slow)),
            )
            targets = strat._generate_signals(close)
            key = combo_name(self.name, combo)
            result[key] = targets
        return result
