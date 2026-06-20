"""Conviction-weighted signal strategies: size proportional to signal strength."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import bb_signals, eligible_symbols, extract_close
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


class ConvictionBBSignal:
    """Bollinger Band reversion with conviction-weighted sizing.

    Position size scales with how far price is below the lower band:
    deeper oversold = larger position (up to max_size).
    """

    name = "conviction_bb"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        bb_period: int = 20,
        bb_std: float = 2.0,
        min_size: float = 0.01,
        max_size: float = 0.05,
    ) -> None:
        self.cfg = cfg
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.min_size = min_size
        self.max_size = max_size

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "bb_period": [10, 15, 20, 30],
            "bb_std": [1.5, 2.0, 2.5],
            "min_size": [0.01],
            "max_size": [0.03, 0.05, 0.08],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def _compute_conviction_sizes(self, close: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
        """Size = linear interpolation based on distance below lower band."""
        sma = close.rolling(window=self.bb_period, min_periods=self.bb_period).mean()
        rolling_std = close.rolling(window=self.bb_period, min_periods=self.bb_period).std()
        lower = sma - self.bb_std * rolling_std
        band_width = self.bb_std * rolling_std
        depth = ((lower - close) / band_width.replace(0, np.nan)).clip(lower=0.0, upper=1.0)
        sizes = self.min_size + depth * (self.max_size - self.min_size)
        sizes = sizes.where(entries, np.nan)
        return sizes

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        entries, exits = bb_signals(close, self.bb_period, self.bb_std)
        sizes = self._compute_conviction_sizes(close, entries)
        return SignalTargets(entries=entries, exits=exits, sizes=sizes)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            period = int(combo["bb_period"])
            std = float(combo["bb_std"])
            min_s = float(combo.get("min_size", self.min_size))
            max_s = float(combo.get("max_size", self.max_size))
            entries, exits = bb_signals(close, period, std)
            strat = ConvictionBBSignal(
                self.cfg, bb_period=period, bb_std=std, min_size=min_s, max_size=max_s
            )
            sizes = strat._compute_conviction_sizes(close, entries)
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries, exits=exits, sizes=sizes)
        return result
