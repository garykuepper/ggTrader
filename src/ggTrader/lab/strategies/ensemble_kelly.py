"""Kelly-criterion-sized ensemble: same entries/exits as EnsembleSignal, but
position size scales with a pooled, causal, expanding Kelly fraction
estimated from the strategy's own closed-trade history (see
ggTrader.lab.kelly). See docs/superpowers/specs/2026-07-06-kelly-position-sizing-design.md.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ggTrader.lab.kelly import kelly_sizes
from ggTrader.lab.strategies.ensemble import DEFAULT_VOTERS, EnsembleSignal, _validate_voters
from ggTrader.lab.strategies.indicators import eligible_symbols, extract_close, extract_volume
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


class EnsembleKellySignal:
    """EnsembleSignal entries/exits with Kelly-criterion position sizing.

    Falls back to `base_size` (the deployed flat-3% baseline) whenever
    there isn't yet a positive measurable edge; always capped at `max_size`.
    """

    name = "ensemble_kelly"
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
        kelly_multiplier: float = 0.5,
        base_size: float = 0.03,
        max_size: float = 0.05,
        min_trades: int = 10,
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
        self.kelly_multiplier = kelly_multiplier
        self.base_size = base_size
        self.max_size = max_size
        self.min_trades = min_trades

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        # Only the Kelly multiplier is swept; ensemble params pinned at baseline.
        return {"kelly_multiplier": [0.25, 0.5, 1.0]}

    def _base_ensemble(self) -> EnsembleSignal:
        return EnsembleSignal(
            self.cfg,
            min_agree=self.min_agree,
            min_agree_exit=self.min_agree_exit,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            rsi_period=self.rsi_period,
            rsi_oversold=self.rsi_oversold,
            rsi_exit=self.rsi_exit,
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            macd_fast=self.macd_fast,
            macd_slow=self.macd_slow,
            macd_signal=self.macd_signal,
            divergence_window=self.divergence_window,
            vol_period=self.vol_period,
            vol_mult=self.vol_mult,
            weekly_rsi_period=self.weekly_rsi_period,
            weekly_rsi_oversold=self.weekly_rsi_oversold,
            weekly_rsi_exit=self.weekly_rsi_exit,
            voters=self.voters,
        )

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        syms = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        max_sec = self.cfg.max_sector_count
        if max_sec is not None:
            from ggTrader.lab.strategies.registry import apply_sector_constraints

            syms = apply_sector_constraints(syms, max_sec)
        return [{"symbol": s, "weight": 0.0} for s in syms]

    def _generate_signals_with_sizes(
        self, close: pd.DataFrame, volume: pd.DataFrame
    ) -> SignalTargets:
        base_targets = self._base_ensemble()._generate_signals(close, volume)
        sizes = kelly_sizes(
            base_targets.entries,
            base_targets.exits,
            close,
            kelly_multiplier=self.kelly_multiplier,
            base_size=self.base_size,
            max_size=self.max_size,
            min_trades=self.min_trades,
        )
        return SignalTargets(entries=base_targets.entries, exits=base_targets.exits, sizes=sizes)

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        return self._generate_signals_with_sizes(close, volume)

    def sweep_signals(
        self, combos: list[dict], symbols: list[str], data: pd.DataFrame
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            strat = EnsembleKellySignal(
                self.cfg,
                min_agree=self.min_agree,
                min_agree_exit=self.min_agree_exit,
                bb_period=self.bb_period,
                bb_std=self.bb_std,
                rsi_period=self.rsi_period,
                rsi_oversold=self.rsi_oversold,
                rsi_exit=self.rsi_exit,
                ema_fast=self.ema_fast,
                ema_slow=self.ema_slow,
                macd_fast=self.macd_fast,
                macd_slow=self.macd_slow,
                macd_signal=self.macd_signal,
                divergence_window=self.divergence_window,
                vol_period=self.vol_period,
                vol_mult=self.vol_mult,
                weekly_rsi_period=self.weekly_rsi_period,
                weekly_rsi_oversold=self.weekly_rsi_oversold,
                weekly_rsi_exit=self.weekly_rsi_exit,
                kelly_multiplier=float(combo.get("kelly_multiplier", self.kelly_multiplier)),
                base_size=self.base_size,
                max_size=self.max_size,
                min_trades=self.min_trades,
                voters=self.voters,
            )
            result[combo_name(self.name, combo)] = strat._generate_signals_with_sizes(close, volume)
        return result
