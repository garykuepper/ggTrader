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

#: All available sub-signal voters.
ALL_VOTERS: tuple[str, ...] = ("bb", "rsi", "ema", "macd", "vbb", "mtf")
#: The validated 5-voter production target (2026-06-24 ablation under fixed NDH
#: gate + ATR exits): the full set MINUS MTF. MTF is the one consistently
#: harmful voter (`core+mtf` Sharpe 0.49 vs core 0.68); MACD and VolBB *add*
#: edge (`core+macd+vbb` Sharpe 0.89 / DD -10.5% / 14-of-17 gate-validated, the
#: best config). The earlier "MACD/VolBB dilute" call was a broken-gate artifact.
FIVE_VOTERS: tuple[str, ...] = ("bb", "rsi", "ema", "macd", "vbb")
#: Default for live + lab + gate training: the ablation-validated 5-voter.
DEFAULT_VOTERS: tuple[str, ...] = FIVE_VOTERS
#: The 3-voter core (BB reversion + RSI + EMA trend) — defensive, solid but
#: beaten by the 5-voter (Sharpe 0.68 / DD -20.5%).
THREE_VOTERS: tuple[str, ...] = ("bb", "rsi", "ema")


def _validate_voters(voters: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate a voter selection and return it as a tuple."""
    if not voters:
        raise ValueError("voters must be non-empty")
    unknown = [v for v in voters if v not in ALL_VOTERS]
    if unknown:
        raise ValueError(f"unknown voter(s) {unknown}; valid: {ALL_VOTERS}")
    return tuple(voters)


class EnsembleSignal:
    """Majority-vote ensemble: enter when >= min_agree sub-signals fire together.

    Sub-signals (configurable via ``voters``): bb_reversion, rsi_reversion,
    ema_cross, macd_divergence, volume_bb_reversion, mtf_reversion.
    Exit: RSI fires independently (when active); other exits require
    >= min_agree_exit votes.
    """

    name = "ensemble"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        min_agree: int = 2,
        min_agree_exit: int | None = None,
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
        voters: tuple[str, ...] | list[str] = DEFAULT_VOTERS,
    ) -> None:
        self.voters = _validate_voters(voters)
        self.cfg = cfg
        self.min_agree = min_agree
        self.min_agree_exit = min_agree_exit if min_agree_exit is not None else min_agree
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
        # Core ensemble params swept; new signal params pinned to defaults.
        # Use --sweep-param to widen individual axes.
        return {
            "min_agree": [2, 3, 4],
            "min_agree_exit": [1, 2],
            "bb_period": [20],
            "bb_std": [2.0, 2.5],
            "rsi_period": [14],
            "rsi_oversold": [25, 30],
            "ema_fast": [10, 20],
            "ema_slow": [50],
            "macd_fast": [12],
            "macd_slow": [26],
            "macd_signal": [9],
            "divergence_window": [10],
            "vol_period": [20],
            "vol_mult": [1.5],
            "weekly_rsi_period": [14],
            "weekly_rsi_oversold": [30],
            "weekly_rsi_exit": [50],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def _generate_signals(self, close: pd.DataFrame, volume: pd.DataFrame) -> SignalTargets:
        """Run the active sub-signals, sum entry/exit votes, threshold at min_agree."""
        ent: Dict[str, pd.DataFrame] = {}
        ext: Dict[str, pd.DataFrame] = {}
        if "bb" in self.voters:
            ent["bb"], ext["bb"] = bb_signals(close, self.bb_period, self.bb_std)
        if "rsi" in self.voters:
            ent["rsi"], ext["rsi"] = rsi_signals(
                close, self.rsi_period, self.rsi_oversold, self.rsi_exit
            )
        if "ema" in self.voters:
            ent["ema"], ext["ema"] = ema_signals(close, self.ema_fast, self.ema_slow)
        if "macd" in self.voters:
            ent["macd"], ext["macd"] = macd_signals(
                close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window
            )
        if "vbb" in self.voters:
            ent["vbb"], ext["vbb"] = volume_bb_signals(
                close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult
            )
        if "mtf" in self.voters:
            ent["mtf"], ext["mtf"] = mtf_signals(
                close,
                self.weekly_rsi_period,
                self.weekly_rsi_oversold,
                self.weekly_rsi_exit,
                self.bb_period,
                self.bb_std,
            )

        entry_votes = sum(df.astype(int) for df in ent.values())
        exit_votes = sum(df.astype(int) for df in ext.values())

        entries = (entry_votes >= self.min_agree).astype(bool)
        # RSI exit fires independently when RSI is an active voter.
        independent_exit = ext["rsi"] if "rsi" in ext else False
        exits = independent_exit | (exit_votes >= self.min_agree_exit)
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
                min_agree_exit=int(combo.get("min_agree_exit", self.min_agree_exit)),
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
                voters=self.voters,
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
        min_agree_exit: int | None = None,
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
        voters: tuple[str, ...] | list[str] = DEFAULT_VOTERS,
    ) -> None:
        self.voters = _validate_voters(voters)
        self.cfg = cfg
        self.min_agree = min_agree
        self.min_agree_exit = min_agree_exit if min_agree_exit is not None else min_agree
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
            "min_agree_exit": [1, 2],
            "bb_period": [20],
            "bb_std": [2.0, 2.5],
            "rsi_period": [14],
            "rsi_oversold": [25, 30],
            "ema_fast": [10, 20],
            "ema_slow": [50],
            "macd_fast": [12],
            "macd_slow": [26],
            "macd_signal": [9],
            "divergence_window": [10],
            "vol_period": [20],
            "vol_mult": [1.5],
            "weekly_rsi_period": [14],
            "weekly_rsi_oversold": [30],
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
        ent: Dict[str, pd.DataFrame] = {}
        ext: Dict[str, pd.DataFrame] = {}
        strengths: Dict[str, pd.DataFrame] = {}

        if "bb" in self.voters:
            ent["bb"], ext["bb"] = bb_signals(close, self.bb_period, self.bb_std)
            strengths["bb"] = bb_strength(close, self.bb_period, self.bb_std)
        if "rsi" in self.voters:
            ent["rsi"], ext["rsi"] = rsi_signals(
                close, self.rsi_period, self.rsi_oversold, self.rsi_exit
            )
            strengths["rsi"] = rsi_strength(close, self.rsi_period, self.rsi_oversold)
        if "ema" in self.voters:
            ent["ema"], ext["ema"] = ema_signals(close, self.ema_fast, self.ema_slow)
            strengths["ema"] = ema_strength(close, self.ema_fast, self.ema_slow)
        if "macd" in self.voters:
            ent["macd"], ext["macd"] = macd_signals(
                close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window
            )
            strengths["macd"] = macd_strength(
                close, self.macd_fast, self.macd_slow, self.macd_signal
            )
        if "vbb" in self.voters:
            ent["vbb"], ext["vbb"] = volume_bb_signals(
                close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult
            )
            strengths["vbb"] = volume_bb_strength(
                close, volume, self.bb_period, self.bb_std, self.vol_period
            )
        if "mtf" in self.voters:
            ent["mtf"], ext["mtf"] = mtf_signals(
                close,
                self.weekly_rsi_period,
                self.weekly_rsi_oversold,
                self.weekly_rsi_exit,
                self.bb_period,
                self.bb_std,
            )
            strengths["mtf"] = mtf_strength(
                close,
                self.weekly_rsi_period,
                self.weekly_rsi_oversold,
                self.bb_period,
                self.bb_std,
            )

        entry_votes = sum(df.astype(int) for df in ent.values())
        exit_votes = sum(df.astype(int) for df in ext.values())

        entries = (entry_votes >= self.min_agree).astype(bool)
        independent_exit = ext["rsi"] if "rsi" in ext else False
        exits = independent_exit | (exit_votes >= self.min_agree_exit)

        # Conviction average strength of agreeing sub-signals
        strength_sum = sum(strengths[k].where(ent[k], 0.0) for k in ent)
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
                min_agree_exit=int(combo.get("min_agree_exit", self.min_agree_exit)),
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
                voters=self.voters,
            )
            targets = strat._generate_signals_with_sizes(close, volume)
            key = combo_name(self.name, combo)
            result[key] = targets
        return result
