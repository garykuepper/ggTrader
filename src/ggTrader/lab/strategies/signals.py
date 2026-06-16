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


_SIGNAL_REGISTRY = {
    "ema_cross": EmaCrossSignal,
    "wfo_tournament": WfoTournamentSignal,
}

SIGNAL_STRATEGY_NAMES = tuple(_SIGNAL_REGISTRY)


def build_signal_strategy(name: str, cfg: LabConfig) -> Any:
    if name not in _SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal strategy {name!r}. Available: {SIGNAL_STRATEGY_NAMES}")
    return _SIGNAL_REGISTRY[name](cfg)
