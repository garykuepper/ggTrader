"""Short-volume-ratio signal -- free-data-only cut of candidate #5
("stealthy shorts": informed liquidity-supplying short sellers).

Goyal, Reed, Smajlbegovic & Soebhag (JFE 2025) decompose short-sale volume
into liquidity-demanding vs. liquidity-supplying components using
transaction-level exchange data this project doesn't have. This free-data
cut uses a coarser proxy instead: trailing average short-volume RATIO
(FINRA daily short volume / total volume) alone, without the
liquidity-demand/supply decomposition -- a real fidelity gap flagged in
WEB_RESEARCH_CANDIDATES.md's own candidate #5 entry, not something this
implementation resolves.

Market-neutral (long lowest quintile, short highest quintile), same
long/short Plan and to_targets pattern as pairs_stat_arb.py -- the second
market-neutral construction in this lab.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.short_volume_data import PUBLISH_LAG_DAYS, available_as_of, load_short_volume
from ggTrader.lab.strategies.indicators import eligible_symbols
from ggTrader.lab.strategy import LabConfig, Plan

#: How far back to pull short-volume history per select() call.
_LOOKBACK_DAYS = 400

SvLoader = Callable[[List[str], str, str], pd.DataFrame]


class ShortVolumeRatioStrategy:
    """Market-neutral: long the lowest-quintile / short the
    highest-quintile of the eligible universe by trailing average
    short-volume ratio, rebalanced monthly. target_kind="weights" --
    verified to support short (negative) weights via simulate_weights.
    """

    name = "short_volume_ratio"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        lookback_days: int = 20,
        quintile: int = 5,
        publish_lag_days: int = PUBLISH_LAG_DAYS,
        _sv_loader: SvLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.lookback_days = lookback_days
        self.quintile = quintile
        self.publish_lag_days = publish_lag_days
        self._sv_loader: SvLoader = _sv_loader or load_short_volume
        self._sv_cache: Dict[tuple, pd.DataFrame] = {}

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "lookback_days": [10, 20, 40],
            "quintile": [4, 5],
        }

    def _load_sv(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        key = (tuple(sorted(symbols)), start, end)
        if key not in self._sv_cache:
            self._sv_cache[key] = self._sv_loader(symbols, start, end)
        return self._sv_cache[key]

    def _avg_ratios(self, sv: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sym, g in sv.sort_values("date").groupby("symbol"):
            window = g.tail(self.lookback_days)["short_volume_ratio"].dropna()
            if window.empty:
                continue
            out[sym] = float(window.mean())
        return out

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if len(elig) < self.quintile:
            return []

        start = (asof - pd.Timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = asof.strftime("%Y-%m-%d")
        sv = self._load_sv(elig, start, end)
        if sv.empty:
            return []

        sv = available_as_of(sv, asof, self.publish_lag_days)
        if sv.empty:
            return []

        ratios = self._avg_ratios(sv)
        if len(ratios) < self.quintile:
            return []

        # Ascending: lowest short-volume ratio first (least shorted -> long
        # leg); highest ratio last (most shorted -> short leg).
        ranked = pd.Series(ratios).sort_values()
        bucket_size = max(1, len(ranked) // self.quintile)
        long_leg = ranked.index[:bucket_size].tolist()
        short_leg = ranked.index[-bucket_size:].tolist()
        long_leg = [s for s in long_leg if s not in short_leg]
        if not long_leg or not short_leg:
            return []

        n = min(len(long_leg), len(short_leg))
        long_leg, short_leg = long_leg[:n], short_leg[:n]
        weight = 1.0 / n
        return [
            {"symbol_long": lo, "symbol_short": sh, "weight": weight}
            for lo, sh in zip(long_leg, short_leg)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted(
            {
                leg
                for plan in plans.values()
                for sel in plan
                for leg in (sel["symbol_long"], sel["symbol_short"])
            }
        )
        if not symbols or len(data.index) == 0:
            return pd.DataFrame(index=data.index, columns=symbols)

        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0  # reset: close any leg not re-selected
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol_long"]] = float(sel["weight"])
                targets.loc[bar, sel["symbol_short"]] = -float(sel["weight"])
        return targets
