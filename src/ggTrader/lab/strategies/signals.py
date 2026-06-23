# src/ggTrader/lab/strategies/signals.py
"""Signal-based lab strategies: entry/exit boolean signals via from_signals."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import (
    bb_signals,
    eligible_symbols,
    ema_signals,
    extract_close,
    extract_volume,
    macd_signals,
    mtf_signals,
    rsi_signals,
    volume_bb_signals,
)
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets

_EMA_COMBOS = [
    {"ema_fast": 5, "ema_slow": 20},
    {"ema_fast": 10, "ema_slow": 30},
    {"ema_fast": 20, "ema_slow": 50},
    {"ema_fast": 50, "ema_slow": 200},
]


def _ema_combo_is_sharpe(close_is: pd.DataFrame, ema_fast: int, ema_slow: int) -> float:
    """Equal-weight portfolio IS Sharpe for a single EMA combo."""
    import vectorbt as vbt

    entries, exits = ema_signals(close_is, ema_fast, ema_slow)
    n_syms = close_is.shape[1]
    if n_syms == 0:
        return float("-inf")
    try:
        pf = vbt.Portfolio.from_signals(
            close=close_is,
            entries=entries,
            exits=exits,
            size=1.0 / n_syms,
            size_type="percent",
            init_cash=10000.0,
            fees=0.0,
            freq="1d",
            group_by=np.zeros(n_syms, dtype=int),
            cash_sharing=True,
        ).copy()
        sharpe = pf.sharpe_ratio()
        val = float(sharpe.iloc[0] if hasattr(sharpe, "iloc") else sharpe)
        return val if np.isfinite(val) else float("-inf")
    except Exception:
        return float("-inf")


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

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "ema_fast": [5, 10, 20, 50],
            "ema_slow": [20, 30, 50, 100, 200],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        """All eligible symbols with enough history — fixed EMA params."""
        data = data.loc[:asof]
        return [
            {
                "symbol": s,
                "weight": 0.0,
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
            }
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        """Compute EMA cross signals over the full window for all selected symbols."""
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        entries, exits = ema_signals(close, self.ema_fast, self.ema_slow)
        return SignalTargets(entries=entries, exits=exits)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        """Vectorized signal generation for all (ema_fast, ema_slow) combos at once."""
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        unique_spans = sorted({v for c in combos for v in c.values()})
        emas: dict[int, pd.DataFrame] = {
            span: close.ewm(span=span, adjust=False).mean() for span in unique_spans
        }
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            fast, slow = int(combo["ema_fast"]), int(combo["ema_slow"])
            ema_f, ema_s = emas[fast], emas[slow]
            entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
            exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))
        return result


class WfoTournamentSignal:
    """EMA combo tournament: pick best (fast, slow) params on IS data each rebalance.

    select() evaluates 4 EMA combos on a 70% IS window and picks the best combo
    by equal-weight portfolio Sharpe. to_targets() generates piecewise signals
    using the per-period winning params.
    """

    name = "wfo_tournament"
    target_kind = "signals"

    def __init__(self, cfg: LabConfig, is_fraction: float = 0.7) -> None:
        self.cfg = cfg
        self.is_fraction = is_fraction

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "is_fraction": [0.5, 0.6, 0.7, 0.8],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        syms = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if not syms:
            return []

        close_all = extract_close(data, syms).ffill()
        is_end = max(1, int(len(close_all) * self.is_fraction))
        close_is = close_all.iloc[:is_end].dropna(axis=1, how="all")

        if len(close_is) < self.cfg.min_history_bars or close_is.shape[1] == 0:
            return []

        best_sharpe = float("-inf")
        best_combo: Dict[str, Any] = _EMA_COMBOS[2]  # default: 20/50
        for combo in _EMA_COMBOS:
            sharpe = _ema_combo_is_sharpe(close_is, combo["ema_fast"], combo["ema_slow"])
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_combo = combo

        result = [
            {
                "symbol": s,
                "weight": 0.0,
                "ema_fast": best_combo["ema_fast"],
                "ema_slow": best_combo["ema_slow"],
                "is_sharpe": round(best_sharpe, 6),
            }
            for s in syms
        ]
        return result[: self.cfg.top_n] if self.cfg.top_n else result

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        """Piecewise signals: each period uses the params selected at its rebalance date."""
        all_symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        entries = pd.DataFrame(False, index=data.index, columns=all_symbols)
        exits = pd.DataFrame(False, index=data.index, columns=all_symbols)

        sorted_dates = sorted(plans.keys())
        have = set(data.columns.get_level_values(0).unique())

        for i, asof in enumerate(sorted_dates):
            next_asof = sorted_dates[i + 1] if i + 1 < len(sorted_dates) else data.index[-1]
            period_mask = (data.index > asof) & (data.index <= next_asof)
            period_index = data.index[period_mask]
            if len(period_index) == 0:
                continue

            active = {s["symbol"] for s in plans[asof]}

            # Exit dropped symbols at the start of this period
            if i > 0:
                prev_active = {s["symbol"] for s in plans[sorted_dates[i - 1]]}
                for sym in prev_active - active:
                    if sym in exits.columns and len(period_index) > 0:
                        exits.loc[period_index[0], sym] = True

            # Generate signals for active symbols using this period's params
            for sel in plans[asof]:
                sym = sel["symbol"]
                if sym not in have:
                    continue
                close_sym = data[sym]["close"].dropna().to_frame(sym)
                sym_ent, sym_ext = ema_signals(
                    close_sym, int(sel.get("ema_fast", 20)), int(sel.get("ema_slow", 50))
                )
                entries.loc[period_index, sym] = (
                    sym_ent[sym].reindex(period_index).fillna(False).to_numpy()
                )
                exits.loc[period_index, sym] = (
                    sym_ext[sym].reindex(period_index).fillna(False).to_numpy()
                )

        return SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        """Sweep over is_fraction values -- each gets its own IS/OOS split and tournament."""
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols).ffill()

        result: dict[str, SignalTargets] = {}
        for combo in combos:
            is_frac = float(combo["is_fraction"])
            is_end = max(1, int(len(close) * is_frac))
            close_is = close.iloc[:is_end].dropna(axis=1, how="all")
            best_combo = _EMA_COMBOS[2]  # default 20/50
            best_sharpe = float("-inf")
            for ec in _EMA_COMBOS:
                sharpe = _ema_combo_is_sharpe(close_is, ec["ema_fast"], ec["ema_slow"])
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_combo = ec
            fast, slow = best_combo["ema_fast"], best_combo["ema_slow"]
            entries, exits = ema_signals(close, fast, slow)
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries, exits=exits)
        return result


class BollingerReversionSignal:
    """Bollinger Band mean-reversion: buy below lower band, sell at middle band.

    Entry: close crosses below lower Bollinger Band (oversold).
    Exit: close crosses above the middle band (SMA) or upper band.
    """

    name = "bb_reversion"
    target_kind = "signals"

    def __init__(self, cfg: LabConfig, bb_period: int = 20, bb_std: float = 2.0) -> None:
        self.cfg = cfg
        self.bb_period = bb_period
        self.bb_std = bb_std

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "bb_period": [10, 15, 20, 30],
            "bb_std": [1.5, 2.0, 2.5, 3.0],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {
                "symbol": s,
                "weight": 0.0,
                "bb_period": self.bb_period,
                "bb_std": self.bb_std,
            }
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        entries, exits = bb_signals(close, self.bb_period, self.bb_std)
        return SignalTargets(entries=entries, exits=exits)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        cache: dict[tuple[int, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            period = int(combo["bb_period"])
            std = float(combo["bb_std"])
            key = (period, std)
            if key not in cache:
                cache[key] = bb_signals(close, period, std)
            ent, ext = cache[key]
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result


class RsiReversionSignal:
    """RSI mean-reversion: buy on oversold RSI, sell when RSI returns to neutral.

    Entry: RSI crosses below oversold threshold.
    Exit: RSI crosses above exit threshold (neutral zone).
    """

    name = "rsi_reversion"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_exit: int = 50,
    ) -> None:
        self.cfg = cfg
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_exit = rsi_exit

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "rsi_period": [7, 14, 21],
            "rsi_oversold": [20, 25, 30],
            "rsi_exit": [50, 55, 60],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {
                "symbol": s,
                "weight": 0.0,
                "rsi_period": self.rsi_period,
                "rsi_oversold": self.rsi_oversold,
                "rsi_exit": self.rsi_exit,
            }
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        entries, exits = rsi_signals(close, self.rsi_period, self.rsi_oversold, self.rsi_exit)
        return SignalTargets(entries=entries, exits=exits)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        cache: dict[tuple[int, int, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            period = int(combo["rsi_period"])
            oversold = int(combo["rsi_oversold"])
            exit_level = int(combo["rsi_exit"])
            key = (period, oversold, exit_level)
            if key not in cache:
                cache[key] = rsi_signals(close, period, oversold, exit_level)
            ent, ext = cache[key]
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result


class MACDDivergenceSignal:
    """MACD bullish divergence: price makes lower low, histogram makes higher low."""

    name = "macd_divergence"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        divergence_window: int = 20,
    ) -> None:
        self.cfg = cfg
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.divergence_window = divergence_window

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "macd_fast": [8, 12],
            "macd_slow": [21, 26],
            "macd_signal": [9],
            "divergence_window": [10, 20],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        entries, exits = macd_signals(
            close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window
        )
        return SignalTargets(entries=entries, exits=exits)

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
            ent, ext = macd_signals(
                close,
                int(combo["macd_fast"]),
                int(combo["macd_slow"]),
                int(combo["macd_signal"]),
                int(combo["divergence_window"]),
            )
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result


class VolumeBBReversionSignal:
    """BB reversion gated by a volume spike — capitulation filter."""

    name = "volume_bb_reversion"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        bb_period: int = 20,
        bb_std: float = 2.0,
        vol_period: int = 20,
        vol_mult: float = 2.0,
    ) -> None:
        self.cfg = cfg
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.vol_period = vol_period
        self.vol_mult = vol_mult

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "vol_period": [20],
            "vol_mult": [1.5, 2.0, 2.5],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        entries, exits = volume_bb_signals(
            close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult
        )
        return SignalTargets(entries=entries, exits=exits)

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
            ent, ext = volume_bb_signals(
                close,
                volume,
                int(combo["bb_period"]),
                float(combo["bb_std"]),
                int(combo["vol_period"]),
                float(combo["vol_mult"]),
            )
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result


class MultiTimeframeReversionSignal:
    """Multi-timeframe: weekly RSI oversold confirms daily BB breakdown."""

    name = "mtf_reversion"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        weekly_rsi_period: int = 14,
        weekly_rsi_oversold: int = 30,
        weekly_rsi_exit: int = 50,
        daily_bb_period: int = 20,
        daily_bb_std: float = 2.0,
    ) -> None:
        self.cfg = cfg
        self.weekly_rsi_period = weekly_rsi_period
        self.weekly_rsi_oversold = weekly_rsi_oversold
        self.weekly_rsi_exit = weekly_rsi_exit
        self.daily_bb_period = daily_bb_period
        self.daily_bb_std = daily_bb_std

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "weekly_rsi_period": [7, 14],
            "weekly_rsi_oversold": [30, 35],
            "weekly_rsi_exit": [50, 55],
            "daily_bb_period": [15, 20],
            "daily_bb_std": [2.0, 2.5],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        entries, exits = mtf_signals(
            close,
            self.weekly_rsi_period,
            self.weekly_rsi_oversold,
            self.weekly_rsi_exit,
            self.daily_bb_period,
            self.daily_bb_std,
        )
        return SignalTargets(entries=entries, exits=exits)

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
            ent, ext = mtf_signals(
                close,
                int(combo["weekly_rsi_period"]),
                int(combo["weekly_rsi_oversold"]),
                int(combo["weekly_rsi_exit"]),
                int(combo["daily_bb_period"]),
                float(combo["daily_bb_std"]),
            )
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result


def _build_signal_registry() -> dict[str, Any]:
    from ggTrader.lab.strategies.conviction import ConvictionBBSignal
    from ggTrader.lab.strategies.ensemble import EnsembleConvictionSignal, EnsembleSignal

    return {
        "ema_cross": EmaCrossSignal,
        "wfo_tournament": WfoTournamentSignal,
        "bb_reversion": BollingerReversionSignal,
        "rsi_reversion": RsiReversionSignal,
        "macd_divergence": MACDDivergenceSignal,
        "volume_bb_reversion": VolumeBBReversionSignal,
        "mtf_reversion": MultiTimeframeReversionSignal,
        "ensemble": EnsembleSignal,
        "conviction_bb": ConvictionBBSignal,
        "ensemble_conviction": EnsembleConvictionSignal,
    }


_SIGNAL_REGISTRY: dict[str, Any] | None = None


def _get_registry() -> dict[str, Any]:
    global _SIGNAL_REGISTRY  # noqa: PLW0603
    if _SIGNAL_REGISTRY is None:
        _SIGNAL_REGISTRY = _build_signal_registry()
    return _SIGNAL_REGISTRY


SIGNAL_STRATEGY_NAMES = (
    "ema_cross",
    "wfo_tournament",
    "bb_reversion",
    "rsi_reversion",
    "macd_divergence",
    "volume_bb_reversion",
    "mtf_reversion",
    "ensemble",
    "conviction_bb",
    "ensemble_conviction",
)


def build_signal_strategy(name: str, cfg: LabConfig) -> Any:
    registry = _get_registry()
    if name not in registry:
        raise ValueError(f"Unknown signal strategy {name!r}. Available: {SIGNAL_STRATEGY_NAMES}")
    return registry[name](cfg)
