"""Breadth-driven leveraged/inverse index rotation (weight-based, monthly).

Rotates between a universe's leveraged-long ETF, its inverse ETF, and cash,
driven by the breadth of the existing validated EnsembleSignal across that
universe's own constituent stocks -- a repurposing of a stock-picking signal
as an index-timing feature. See
docs/superpowers/specs/2026-07-14-leveraged-index-rotation-design.md.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig, Plan


def compute_breadth(entries: pd.DataFrame) -> pd.Series:
    """Fraction of columns with an active (True) entry signal per row.

    Fixed denominator = entries.shape[1] (the full breadth-universe size),
    not the count of symbols currently past warmup -- see spec for why.
    """
    if entries.shape[1] == 0:
        return pd.Series(0.0, index=entries.index)
    return entries.sum(axis=1) / entries.shape[1]


def rotate_positions(
    breadth: pd.Series,
    upper_threshold: float,
    lower_threshold: float,
    min_hold_months: int,
) -> pd.Series:
    """Per-date state in {"long", "inverse", "cash"} with hysteresis.

    A state change only takes effect once the new raw signal (breadth vs.
    thresholds) has held for min_hold_months consecutive dates in the input
    series. The first date's state takes effect immediately.
    """
    raw = pd.Series("cash", index=breadth.index, dtype=object)
    raw[breadth > upper_threshold] = "long"
    raw[breadth < lower_threshold] = "inverse"

    states: list[str] = []
    current: str | None = None
    streak_value: str | None = None
    streak_len = 0
    for val in raw:
        if val == streak_value:
            streak_len += 1
        else:
            streak_value = val
            streak_len = 1
        if current is None:
            current = val
        elif val != current and streak_len >= min_hold_months:
            current = val
        states.append(current)
    return pd.Series(states, index=breadth.index)


#: Memoizes the EnsembleSignal breadth pass, keyed by a content-derived
#: fingerprint of the data window, breadth-universe symbol set, AND the
#: cfg fields EnsembleSignal reads (min_history_bars, max_stocks,
#: max_sector_count). Breadth doesn't depend on the strategy's own swept
#: params (upper_threshold/lower_threshold/min_hold_months/leverage_tier),
#: but the WFO harness constructs a fresh strategy instance per combo and
#: calls to_targets() on the same data window for every one of them --
#: without this cache, the full vectorized EnsembleSignal pass over the
#: whole breadth universe (up to ~500+ stocks) gets redundantly recomputed
#: once per combo, dozens of times per fold. The cfg fields are included
#: defensively: today EnsembleSignal.to_targets()/_generate_signals() never
#: reads them (only its own voter kwargs, always constructed with defaults
#: here), so they're currently invariant across combos on the same window
#: and this widening changes no cache-hit behavior -- but a cache keyed
#: only on the data window would have silently served a stale answer for a
#: different cfg the moment that stopped being true (e.g. min_history_bars
#: ever swept per-combo). Runs in a joblib worker process, like
#: leveraged_trend.py's _underlying_cache and pairs_stat_arb.py's
#: _pair_candidate_cache: the cache is process-local, so each worker still
#: pays for one miss per (window, cfg) it sees -- only redundant recompute
#: *within* a worker process is eliminated.
_breadth_cache: dict[tuple, pd.Series] = {}


def _cached_breadth(cfg: LabConfig, data: pd.DataFrame, breadth_symbols: list[str]) -> pd.Series:
    cfg_key = (cfg.min_history_bars, cfg.max_stocks, cfg.max_sector_count)
    key = (data.index[0], data.index[-1], len(data.index), tuple(breadth_symbols), cfg_key)
    if key not in _breadth_cache:
        ensemble = EnsembleSignal(cfg)
        placeholder = [{"symbol": s, "weight": 0.0} for s in breadth_symbols]
        signal_targets = ensemble.to_targets({data.index[0]: placeholder}, data)
        _breadth_cache[key] = compute_breadth(signal_targets.entries)
    return _breadth_cache[key]


class _LeveragedRotationBase:
    """Rotates between a leveraged-long ETF, an inverse ETF, and cash,
    driven by monthly breadth of EnsembleSignal across the universe's own
    constituent stocks. Subclasses fix PAIR_3X/PAIR_2X (see Task 3)."""

    name: str
    target_kind = "weights"
    PAIR_3X: tuple[str, str]
    PAIR_2X: tuple[str, str]

    def __init__(
        self,
        cfg: LabConfig,
        upper_threshold: float = 0.6,
        lower_threshold: float = 0.4,
        min_hold_months: int = 1,
        leverage_tier: str = "3x",
    ) -> None:
        self.cfg = cfg
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.min_hold_months = min_hold_months
        self.leverage_tier = leverage_tier

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "upper_threshold": [0.55, 0.60, 0.65],
            "lower_threshold": [0.35, 0.40, 0.45],
            "min_hold_months": [1, 2, 3],
            "leverage_tier": ["2x", "3x"],
        }

    def _pair(self) -> tuple[str, str]:
        return self.PAIR_3X if self.leverage_tier == "3x" else self.PAIR_2X

    def _all_etf_tickers(self) -> set[str]:
        return set(self.PAIR_3X) | set(self.PAIR_2X)

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        long_t, inv_t = self._pair()
        return [{"symbol": s, "weight": 0.0} for s in (long_t, inv_t) if s in eligible]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        long_t, inv_t = self._pair()
        rebalance_dates = sorted(plans.keys())
        if not rebalance_dates or data.empty:
            return pd.DataFrame(columns=[long_t, inv_t])

        have = set(data.columns.get_level_values(0).unique())
        breadth_symbols = sorted(have - self._all_etf_tickers())

        breadth = _cached_breadth(self.cfg, data, breadth_symbols)

        monthly_breadth = breadth.reindex(rebalance_dates)
        states = rotate_positions(
            monthly_breadth, self.upper_threshold, self.lower_threshold, self.min_hold_months
        )

        targets = pd.DataFrame(np.nan, index=data.index, columns=[long_t, inv_t])
        for asof in rebalance_dates:
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            state = states.loc[asof]
            targets.loc[bar, long_t] = 1.0 if state == "long" else 0.0
            targets.loc[bar, inv_t] = 1.0 if state == "inverse" else 0.0
        return targets


class LeveragedRotationSp500(_LeveragedRotationBase):
    name = "leveraged_rotation_sp500"
    BREADTH_UNIVERSE = "sp500"
    PAIR_3X = ("UPRO", "SPXU")
    PAIR_2X = ("SSO", "SDS")


class LeveragedRotationNasdaq100(_LeveragedRotationBase):
    name = "leveraged_rotation_nasdaq100"
    BREADTH_UNIVERSE = "nasdaq100"
    PAIR_3X = ("TQQQ", "SQQQ")
    PAIR_2X = ("QLD", "QID")


class LeveragedRotationRussell2000(_LeveragedRotationBase):
    name = "leveraged_rotation_russell2000"
    BREADTH_UNIVERSE = "russell2000"
    PAIR_3X = ("TNA", "TZA")
    PAIR_2X = ("UWM", "TWM")
