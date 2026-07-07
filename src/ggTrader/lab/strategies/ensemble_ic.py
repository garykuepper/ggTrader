"""IC-weighted voting ensemble: weight the 5-voter pool by trailing Spearman IC."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ggTrader.lab.strategies.ensemble import DEFAULT_VOTERS, _validate_voters
from ggTrader.lab.strategies.ic_weights import ic_weight_schedule
from ggTrader.lab.strategies.indicators import (
    bb_raw,
    bb_signals,
    eligible_symbols,
    ema_raw,
    ema_signals,
    extract_close,
    extract_volume,
    macd_raw,
    macd_signals,
    rsi_raw,
    rsi_signals,
    vbb_raw,
    volume_bb_signals,
)
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets

# Tiny epsilon used when comparing the weighted score to consensus_threshold.
# Equal weights (e.g. 5 voters at 0.2 each) summing to exactly the threshold
# can miss the boundary due to floating-point accumulation; epsilon hardens it.
_SCORE_EPS = 1e-9


class EnsembleICSignal:
    """Enter when the IC-weighted sum of voter entries clears a consensus threshold.

    Weights come from a causal trailing-window cross-sectional Spearman IC,
    recomputed quarterly (see ic_weights.ic_weight_schedule). Exits reuse the
    baseline EnsembleSignal logic. The validated EnsembleSignal is untouched.
    """

    name = "ensemble_ic"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        *,
        consensus_threshold: float = 0.4,
        ic_lookback_months: int = 12,
        ic_horizon: int = 3,
        ic_rebalance: str = "QE",
        ic_min_names: int = 10,
        min_agree_exit: int = 2,
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
        td_stop: int | None = None,
        exits_enabled: bool = True,
        voters: tuple[str, ...] | list[str] = DEFAULT_VOTERS,
    ) -> None:
        self.voters = _validate_voters(voters)
        self.cfg = cfg
        self.consensus_threshold = consensus_threshold
        self.ic_lookback_months = ic_lookback_months
        self.ic_horizon = ic_horizon
        self.ic_rebalance = ic_rebalance
        self.ic_min_names = ic_min_names
        self.min_agree_exit = min_agree_exit
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
        self.td_stop = td_stop
        self.exits_enabled = exits_enabled

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        # Only the two IC axes are swept; indicator params pinned at baseline.
        return {
            "consensus_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
            "ic_lookback_months": [3, 6, 12],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        syms = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        max_sec = self.cfg.max_sector_count
        if max_sec is not None:
            from ggTrader.lab.strategies.registry import apply_sector_constraints

            syms = apply_sector_constraints(syms, max_sec)
        return [{"symbol": s, "weight": 0.0} for s in syms]

    def _entries_exits_raw(self, close: pd.DataFrame, volume: pd.DataFrame):
        ent: Dict[str, pd.DataFrame] = {}
        ext: Dict[str, pd.DataFrame] = {}
        raw: Dict[str, pd.DataFrame] = {}
        if "bb" in self.voters:
            ent["bb"], ext["bb"] = bb_signals(close, self.bb_period, self.bb_std)
            raw["bb"] = bb_raw(close, self.bb_period, self.bb_std)
        if "rsi" in self.voters:
            ent["rsi"], ext["rsi"] = rsi_signals(
                close, self.rsi_period, self.rsi_oversold, self.rsi_exit
            )
            raw["rsi"] = rsi_raw(close, self.rsi_period)
        if "ema" in self.voters:
            ent["ema"], ext["ema"] = ema_signals(close, self.ema_fast, self.ema_slow)
            raw["ema"] = ema_raw(close, self.ema_fast, self.ema_slow)
        if "macd" in self.voters:
            ent["macd"], ext["macd"] = macd_signals(
                close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window
            )
            raw["macd"] = macd_raw(close, self.macd_fast, self.macd_slow, self.macd_signal)
        if "vbb" in self.voters:
            ent["vbb"], ext["vbb"] = volume_bb_signals(
                close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult
            )
            raw["vbb"] = vbb_raw(close, volume, self.bb_period, self.bb_std, self.vol_period)
        return ent, ext, raw

    def _apply_time_stop(self, entries: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
        if self.td_stop is None:
            return exits.astype(bool)
        timed = entries.shift(self.td_stop, fill_value=False).astype(bool)
        return (exits.astype(bool) | timed).astype(bool)

    def _generate_signals(self, close: pd.DataFrame, volume: pd.DataFrame) -> SignalTargets:
        ent, ext, raw = self._entries_exits_raw(close, volume)

        weights = ic_weight_schedule(
            raw,
            close,
            lookback_months=self.ic_lookback_months,
            horizon=self.ic_horizon,
            rebalance=self.ic_rebalance,
            min_names=self.ic_min_names,
        )
        # weighted_score[d, s] = sum_j w_j[d] * ent_j[d, s]; rows of w sum to 1.
        score = sum(ent[j].astype(float).mul(weights[j], axis=0) for j in ent)
        entries = (score >= self.consensus_threshold - _SCORE_EPS).astype(bool)

        exit_votes = sum(df.astype(int) for df in ext.values())
        if self.exits_enabled:
            independent_exit = ext["rsi"] if "rsi" in ext else False
            exits = independent_exit | (exit_votes >= self.min_agree_exit)
        else:
            exits = pd.DataFrame(False, index=entries.index, columns=entries.columns)
        exits = self._apply_time_stop(entries, exits)
        return SignalTargets(entries=entries, exits=exits.astype(bool))

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        return self._generate_signals(close, volume)

    def sweep_signals(
        self, combos: list[dict], symbols: list[str], data: pd.DataFrame
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            strat = EnsembleICSignal(
                self.cfg,
                consensus_threshold=float(
                    combo.get("consensus_threshold", self.consensus_threshold)
                ),
                ic_lookback_months=int(combo.get("ic_lookback_months", self.ic_lookback_months)),
                min_agree_exit=int(combo.get("min_agree_exit", self.min_agree_exit)),
                voters=self.voters,
            )
            result[combo_name(self.name, combo)] = strat._generate_signals(close, volume)
        return result
