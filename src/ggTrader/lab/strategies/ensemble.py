"""Signal ensemble: enter when N-of-M sub-signals agree on the same bar+symbol."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import (
    bb_signals,
    bb_strength,
    eligible_symbols,
    ema_signals,
    ema_strength,
    extract_close,
    extract_volume,
    macd_signals,
    macd_strength,
    mtf_signals,
    mtf_strength,
    rsi_signals,
    rsi_strength,
    volume_bb_signals,
    volume_bb_strength,
)
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


class EnsembleSignal:
    """Majority-vote ensemble: enter when >= min_agree sub-signals fire together.

    Sub-signals: bb_reversion, rsi_reversion, ema_cross, macd_divergence,
    volume_bb_reversion, mtf_reversion.
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
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        divergence_window: int = 20,
        vol_period: int = 20,
        vol_mult: float = 2.0,
        weekly_rsi_period: int = 14,
        weekly_rsi_oversold: int = 30,
        weekly_rsi_exit: int = 50,
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
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.divergence_window = divergence_window
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.weekly_rsi_period = weekly_rsi_period
        self.weekly_rsi_oversold = weekly_rsi_oversold
        self.weekly_rsi_exit = weekly_rsi_exit

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "min_agree": [2, 3, 4],
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "rsi_period": [7, 14],
            "rsi_oversold": [25, 30],
            "ema_fast": [10, 20],
            "ema_slow": [50, 100],
            "macd_fast": [8, 12],
            "macd_slow": [21, 26],
            "macd_signal": [9],
            "divergence_window": [10, 20],
            "vol_period": [20],
            "vol_mult": [1.5, 2.0],
            "weekly_rsi_period": [7, 14],
            "weekly_rsi_oversold": [30, 35],
            "weekly_rsi_exit": [50],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def _generate_signals(self, close: pd.DataFrame, volume: pd.DataFrame) -> SignalTargets:
        """Run all 6 sub-signals, sum entry/exit votes, threshold at min_agree."""
        bb_ent, bb_ext = bb_signals(close, self.bb_period, self.bb_std)
        rsi_ent, rsi_ext = rsi_signals(close, self.rsi_period, self.rsi_oversold, self.rsi_exit)
        ema_ent, ema_ext = ema_signals(close, self.ema_fast, self.ema_slow)
        macd_ent, macd_ext = macd_signals(
            close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window
        )
        vbb_ent, vbb_ext = volume_bb_signals(
            close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult
        )
        mtf_ent, mtf_ext = mtf_signals(
            close,
            self.weekly_rsi_period,
            self.weekly_rsi_oversold,
            self.weekly_rsi_exit,
            self.bb_period,
            self.bb_std,
        )

        entry_votes = (
            bb_ent.astype(int)
            + rsi_ent.astype(int)
            + ema_ent.astype(int)
            + macd_ent.astype(int)
            + vbb_ent.astype(int)
            + mtf_ent.astype(int)
        )
        exit_votes = (
            bb_ext.astype(int)
            + rsi_ext.astype(int)
            + ema_ext.astype(int)
            + macd_ext.astype(int)
            + vbb_ext.astype(int)
            + mtf_ext.astype(int)
        )

        entries = (entry_votes >= self.min_agree).astype(bool)
        exits = (exit_votes >= self.min_agree).astype(bool)
        return SignalTargets(entries=entries, exits=exits)

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        return self._generate_signals(close, volume)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
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
                macd_fast=int(combo.get("macd_fast", self.macd_fast)),
                macd_slow=int(combo.get("macd_slow", self.macd_slow)),
                macd_signal=int(combo.get("macd_signal", self.macd_signal)),
                divergence_window=int(combo.get("divergence_window", self.divergence_window)),
                vol_period=int(combo.get("vol_period", self.vol_period)),
                vol_mult=float(combo.get("vol_mult", self.vol_mult)),
                weekly_rsi_period=int(combo.get("weekly_rsi_period", self.weekly_rsi_period)),
                weekly_rsi_oversold=int(combo.get("weekly_rsi_oversold", self.weekly_rsi_oversold)),
                weekly_rsi_exit=int(combo.get("weekly_rsi_exit", self.weekly_rsi_exit)),
            )
            targets = strat._generate_signals(close, volume)
            key = combo_name(self.name, combo)
            result[key] = targets
        return result


class EnsembleConvictionSignal:
    """Majority-vote ensemble with conviction-weighted position sizing.

    Same entry/exit logic as EnsembleSignal, but sizes positions by the
    average strength of the agreeing sub-signals on each entry bar.
    """

    name = "ensemble_conviction"
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
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        divergence_window: int = 20,
        vol_period: int = 20,
        vol_mult: float = 2.0,
        weekly_rsi_period: int = 14,
        weekly_rsi_oversold: int = 30,
        weekly_rsi_exit: int = 50,
        min_size: float = 0.01,
        max_size: float = 0.04,
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
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.divergence_window = divergence_window
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.weekly_rsi_period = weekly_rsi_period
        self.weekly_rsi_oversold = weekly_rsi_oversold
        self.weekly_rsi_exit = weekly_rsi_exit
        self.min_size = min_size
        self.max_size = max_size

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "min_agree": [2, 3, 4],
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "rsi_period": [7, 14],
            "rsi_oversold": [25, 30],
            "ema_fast": [10, 20],
            "ema_slow": [50, 100],
            "macd_fast": [8, 12],
            "macd_slow": [21, 26],
            "macd_signal": [9],
            "divergence_window": [10, 20],
            "vol_period": [20],
            "vol_mult": [1.5, 2.0],
            "weekly_rsi_period": [7, 14],
            "weekly_rsi_oversold": [30, 35],
            "weekly_rsi_exit": [50],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def _generate_signals_with_sizes(
        self, close: pd.DataFrame, volume: pd.DataFrame
    ) -> SignalTargets:
        """Entry/exit via majority vote + conviction-weighted sizes."""
        bb_ent, bb_ext = bb_signals(close, self.bb_period, self.bb_std)
        rsi_ent, rsi_ext = rsi_signals(close, self.rsi_period, self.rsi_oversold, self.rsi_exit)
        ema_ent, ema_ext = ema_signals(close, self.ema_fast, self.ema_slow)
        macd_ent, macd_ext = macd_signals(
            close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window
        )
        vbb_ent, vbb_ext = volume_bb_signals(
            close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult
        )
        mtf_ent, mtf_ext = mtf_signals(
            close,
            self.weekly_rsi_period,
            self.weekly_rsi_oversold,
            self.weekly_rsi_exit,
            self.bb_period,
            self.bb_std,
        )

        entry_votes = (
            bb_ent.astype(int)
            + rsi_ent.astype(int)
            + ema_ent.astype(int)
            + macd_ent.astype(int)
            + vbb_ent.astype(int)
            + mtf_ent.astype(int)
        )
        exit_votes = (
            bb_ext.astype(int)
            + rsi_ext.astype(int)
            + ema_ext.astype(int)
            + macd_ext.astype(int)
            + vbb_ext.astype(int)
            + mtf_ext.astype(int)
        )

        entries = (entry_votes >= self.min_agree).astype(bool)
        exits = (exit_votes >= self.min_agree).astype(bool)

        bb_str = bb_strength(close, self.bb_period, self.bb_std)
        rsi_str = rsi_strength(close, self.rsi_period, self.rsi_oversold)
        ema_str = ema_strength(close, self.ema_fast, self.ema_slow)
        macd_str = macd_strength(close, self.macd_fast, self.macd_slow, self.macd_signal)
        vbb_str = volume_bb_strength(close, volume, self.bb_period, self.bb_std, self.vol_period)
        mtf_str = mtf_strength(
            close, self.weekly_rsi_period, self.weekly_rsi_oversold, self.bb_period, self.bb_std
        )

        strength_sum = (
            bb_str.where(bb_ent, 0.0)
            + rsi_str.where(rsi_ent, 0.0)
            + ema_str.where(ema_ent, 0.0)
            + macd_str.where(macd_ent, 0.0)
            + vbb_str.where(vbb_ent, 0.0)
            + mtf_str.where(mtf_ent, 0.0)
        )
        conviction = strength_sum / entry_votes.replace(0, np.nan)

        sizes = self.min_size + conviction * (self.max_size - self.min_size)
        sizes = sizes.where(entries, np.nan)

        return SignalTargets(entries=entries, exits=exits, sizes=sizes)

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        return self._generate_signals_with_sizes(close, volume)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            strat = EnsembleConvictionSignal(
                self.cfg,
                min_agree=int(combo.get("min_agree", self.min_agree)),
                bb_period=int(combo.get("bb_period", self.bb_period)),
                bb_std=float(combo.get("bb_std", self.bb_std)),
                rsi_period=int(combo.get("rsi_period", self.rsi_period)),
                rsi_oversold=int(combo.get("rsi_oversold", self.rsi_oversold)),
                rsi_exit=int(combo.get("rsi_exit", self.rsi_exit)),
                ema_fast=int(combo.get("ema_fast", self.ema_fast)),
                ema_slow=int(combo.get("ema_slow", self.ema_slow)),
                macd_fast=int(combo.get("macd_fast", self.macd_fast)),
                macd_slow=int(combo.get("macd_slow", self.macd_slow)),
                macd_signal=int(combo.get("macd_signal", self.macd_signal)),
                divergence_window=int(combo.get("divergence_window", self.divergence_window)),
                vol_period=int(combo.get("vol_period", self.vol_period)),
                vol_mult=float(combo.get("vol_mult", self.vol_mult)),
                weekly_rsi_period=int(combo.get("weekly_rsi_period", self.weekly_rsi_period)),
                weekly_rsi_oversold=int(combo.get("weekly_rsi_oversold", self.weekly_rsi_oversold)),
                weekly_rsi_exit=int(combo.get("weekly_rsi_exit", self.weekly_rsi_exit)),
                min_size=self.min_size,
                max_size=self.max_size,
            )
            targets = strat._generate_signals_with_sizes(close, volume)
            key = combo_name(self.name, combo)
            result[key] = targets
        return result
