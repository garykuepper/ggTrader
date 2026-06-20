# src/ggTrader/lab/strategies/signals.py
"""Signal-based lab strategies: entry/exit boolean signals via from_signals."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets

_EMA_COMBOS = [
    {"ema_fast": 5, "ema_slow": 20},
    {"ema_fast": 10, "ema_slow": 30},
    {"ema_fast": 20, "ema_slow": 50},
    {"ema_fast": 50, "ema_slow": 200},
]


def _ema_combo_is_sharpe(close_is: pd.DataFrame, ema_fast: int, ema_slow: int) -> float:
    """Equal-weight portfolio IS Sharpe for a single EMA combo."""
    ema_f = close_is.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close_is.ewm(span=ema_slow, adjust=False).mean()
    entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
    exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)
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

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        """Vectorized signal generation for all (ema_fast, ema_slow) combos at once."""
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
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
        have = set(data.columns.get_level_values(0).unique())
        syms = [
            s
            for s in eligible
            if s in have and len(data[s]["close"].dropna()) >= self.cfg.min_history_bars
        ]
        if not syms:
            return []

        close_all = pd.concat({s: data[s]["close"] for s in syms}, axis=1).ffill()
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
                ema_fast = int(sel.get("ema_fast", 20))
                ema_slow = int(sel.get("ema_slow", 50))

                close = data[sym]["close"].dropna()
                ema_f = close.ewm(span=ema_fast, adjust=False).mean()
                ema_s = close.ewm(span=ema_slow, adjust=False).mean()
                sym_entries = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
                sym_exits = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))

                entries.loc[period_index, sym] = (
                    sym_entries.reindex(period_index).fillna(False).to_numpy()
                )
                exits.loc[period_index, sym] = (
                    sym_exits.reindex(period_index).fillna(False).to_numpy()
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

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        ).ffill()
        unique_spans = sorted({v for combo_list in _EMA_COMBOS for v in combo_list.values()})
        emas: dict[int, pd.DataFrame] = {
            span: close.ewm(span=span, adjust=False).mean() for span in unique_spans
        }

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
            ema_f, ema_s = emas[fast], emas[slow]
            entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
            exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))
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
        have = set(data.columns.get_level_values(0).unique())
        return [
            {
                "symbol": s,
                "weight": 0.0,
                "bb_period": self.bb_period,
                "bb_std": self.bb_std,
            }
            for s in eligible
            if s in have and len(data[s]["close"].dropna()) >= self.cfg.min_history_bars
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        entries, exits = _bb_signals(close, self.bb_period, self.bb_std)
        return SignalTargets(entries=entries, exits=exits)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        cache: dict[tuple[int, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            period = int(combo["bb_period"])
            std = float(combo["bb_std"])
            key = (period, std)
            if key not in cache:
                cache[key] = _bb_signals(close, period, std)
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
        have = set(data.columns.get_level_values(0).unique())
        return [
            {
                "symbol": s,
                "weight": 0.0,
                "rsi_period": self.rsi_period,
                "rsi_oversold": self.rsi_oversold,
                "rsi_exit": self.rsi_exit,
            }
            for s in eligible
            if s in have and len(data[s]["close"].dropna()) >= self.cfg.min_history_bars
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        entries, exits = _rsi_signals(close, self.rsi_period, self.rsi_oversold, self.rsi_exit)
        return SignalTargets(entries=entries, exits=exits)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        cache: dict[tuple[int, int, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            period = int(combo["rsi_period"])
            oversold = int(combo["rsi_oversold"])
            exit_level = int(combo["rsi_exit"])
            key = (period, oversold, exit_level)
            if key not in cache:
                cache[key] = _rsi_signals(close, period, oversold, exit_level)
            ent, ext = cache[key]
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result


def _bb_signals(close: pd.DataFrame, period: int, std: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized Bollinger Band entry/exit signals."""
    sma = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()
    lower = sma - std * rolling_std

    prev_above = close.shift(1) >= lower.shift(1)
    now_below = close < lower
    entries = (prev_above & now_below).fillna(False).astype(bool)

    prev_below = close.shift(1) < sma.shift(1)
    now_above = close >= sma
    exits = (prev_below & now_above).fillna(False).astype(bool)

    return entries, exits


def _rsi_signals(
    close: pd.DataFrame, period: int, oversold: int, exit_level: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized RSI entry/exit signals."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    prev_above = rsi.shift(1) >= oversold
    now_below = rsi < oversold
    entries = (prev_above & now_below).fillna(False).astype(bool)

    prev_below = rsi.shift(1) < exit_level
    now_above = rsi >= exit_level
    exits = (prev_below & now_above).fillna(False).astype(bool)

    return entries, exits


from ggTrader.lab.strategies.ensemble import EnsembleSignal

_SIGNAL_REGISTRY = {
    "ema_cross": EmaCrossSignal,
    "wfo_tournament": WfoTournamentSignal,
    "bb_reversion": BollingerReversionSignal,
    "rsi_reversion": RsiReversionSignal,
    "ensemble": EnsembleSignal,
}

SIGNAL_STRATEGY_NAMES = tuple(_SIGNAL_REGISTRY)


def build_signal_strategy(name: str, cfg: LabConfig) -> Any:
    if name not in _SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal strategy {name!r}. Available: {SIGNAL_STRATEGY_NAMES}")
    return _SIGNAL_REGISTRY[name](cfg)
