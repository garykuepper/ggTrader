"""Long-only leveraged-ETF trend-following strategy.

Holds a universe's leveraged-long ETF (2x or 3x) when its underlying,
unleveraged index is above its trailing SMA; otherwise sits in cash. No
inverse leg -- the breadth-driven long/inverse/cash design in
leveraged_rotation.py was WFO-validated NO-GO (see
docs/research/2026-07-16-leveraged-index-rotation-nogo.md); this is a
deliberately simpler, decoupled arc.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ggTrader.lab.data import load_ohlcv
from ggTrader.lab.strategies.leveraged_rotation import rotate_positions
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets

#: Memoizes the underlying index's close series, keyed by (ticker, start,
#: end) -- already a complete key: those three values fully determine the
#: series returned, so (unlike leveraged_rotation.py's _breadth_cache before
#: its cfg-key fix) there is no wrong-answer risk here, only a performance
#: one. WFO builds a fresh strategy instance per grid combo and calls
#: sweep_signals/to_targets on the same data window for every one of them --
#: without this cache, load_ohlcv would be redundantly called once per combo,
#: mirroring the _breadth_cache rationale in leveraged_rotation.py. Like that
#: cache and pairs_stat_arb.py's _pair_candidate_cache, this dict is
#: process-local: under joblib's per-combo worker parallelism each worker
#: still pays for its own first miss per (ticker, start, end) -- only
#: redundant recompute *within* a worker process is eliminated, not across
#: workers. load_ohlcv itself is DB-cache-backed (not a raw yfinance call),
#: so a cross-worker miss costs a DB round trip, not a network fetch.
_underlying_cache: dict[tuple, pd.Series] = {}


def _cached_underlying_close(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    key = (ticker, start, end)
    if key not in _underlying_cache:
        ohlcv = load_ohlcv([ticker], str(start.date()), str(end.date()), use_negative_cache=True)
        _underlying_cache[key] = ohlcv[ticker]["close"].dropna()
    return _underlying_cache[key]


def _trend_state(
    underlying_close: pd.Series,
    index: pd.DatetimeIndex,
    trend_window: int,
    min_hold_days: int,
) -> pd.Series:
    """Boolean is-long series aligned to ``index``, hysteresis-confirmed via
    rotate_positions. upper_threshold == lower_threshold == 0.5 makes the raw
    signal a strict up/down binary (no dead zone) -- min_hold_days is the
    only source of hysteresis. rotate_positions labels the down state
    "inverse" (it has no concept of "no short leg"); this strategy only ever
    checks state == "long" and treats every other label as cash, so the
    "inverse" name is a harmless artifact of reusing that helper."""
    sma = underlying_close.rolling(trend_window, min_periods=trend_window).mean()
    raw_trend = (underlying_close > sma).reindex(index).ffill().fillna(False)
    states = rotate_positions(
        raw_trend.astype(float),
        upper_threshold=0.5,
        lower_threshold=0.5,
        min_hold_months=min_hold_days,
    )
    return states == "long"


def _entries_exits(is_long: pd.Series, etf: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prev = is_long.shift(1, fill_value=False)
    entries = (is_long & ~prev).to_frame(etf)
    exits = (~is_long & prev).to_frame(etf)
    return entries.astype(bool), exits.astype(bool)


class _LeveragedTrendBase:
    """target_kind="signals": hold ETF_3X/ETF_2X while UNDERLYING trends up."""

    name: str
    target_kind = "signals"
    ETF_3X: str
    ETF_2X: str
    UNDERLYING: str

    def __init__(
        self,
        cfg: LabConfig,
        trend_window: int = 200,
        leverage_tier: str = "3x",
        min_hold_days: int = 5,
        vol_target: float = 0.15,
    ) -> None:
        self.cfg = cfg
        self.trend_window = trend_window
        self.leverage_tier = leverage_tier
        self.min_hold_days = min_hold_days
        self.vol_target = vol_target

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "trend_window": [50, 100, 150, 200],
            "leverage_tier": ["2x", "3x"],
            "min_hold_days": [1, 5, 10],
            "vol_target": [0.10, 0.15, 0.20],
        }

    def _etf(self) -> str:
        return self.ETF_3X if self.leverage_tier == "3x" else self.ETF_2X

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        etf = self._etf()
        return [{"symbol": etf, "weight": 1.0}] if etf in eligible else []

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        etf = self._etf()
        if len(data.index) == 0:
            # NOTE: not `data.empty` -- that's also True for a frame with a
            # real index but zero columns, which to_targets never receives
            # in production (data always carries OHLCV columns) but is easy
            # to construct by accident in a test double.
            return SignalTargets(
                entries=pd.DataFrame(columns=[etf]), exits=pd.DataFrame(columns=[etf])
            )
        underlying_close = _cached_underlying_close(self.UNDERLYING, data.index[0], data.index[-1])
        is_long = _trend_state(underlying_close, data.index, self.trend_window, self.min_hold_days)
        entries, exits = _entries_exits(is_long, etf)
        return SignalTargets(entries=entries, exits=exits)

    def sweep_signals(
        self,
        combos: List[Dict[str, Any]],
        symbols: List[str],
        data: pd.DataFrame,
    ) -> Dict[str, SignalTargets]:
        """Vectorized signal generation for all (trend_window, leverage_tier,
        min_hold_days) combos at once -- required by wfo.py's _sweep_fold,
        which calls this unconditionally for target_kind="signals"
        strategies (see sweep.py's sweep_signal_group). vol_target/
        vol_lookback are split out as OVERLAY_PARAMS before reaching here."""
        from ggTrader.lab.sweep import combo_name

        underlying_close = _cached_underlying_close(self.UNDERLYING, data.index[0], data.index[-1])
        unique_windows = sorted({int(c.get("trend_window", self.trend_window)) for c in combos})
        smas = {w: underlying_close.rolling(w, min_periods=w).mean() for w in unique_windows}

        result: Dict[str, SignalTargets] = {}
        for combo in combos:
            window = int(combo.get("trend_window", self.trend_window))
            tier = combo.get("leverage_tier", self.leverage_tier)
            min_hold = int(combo.get("min_hold_days", self.min_hold_days))
            etf = self.ETF_3X if tier == "3x" else self.ETF_2X

            raw_trend = (underlying_close > smas[window]).reindex(data.index).ffill().fillna(False)
            states = rotate_positions(
                raw_trend.astype(float),
                upper_threshold=0.5,
                lower_threshold=0.5,
                min_hold_months=min_hold,
            )
            is_long = states == "long"
            entries, exits = _entries_exits(is_long, etf)
            result[combo_name(self.name, combo)] = SignalTargets(entries=entries, exits=exits)
        return result


class LeveragedTrendSp500(_LeveragedTrendBase):
    name = "leveraged_trend_sp500"
    ETF_3X, ETF_2X, UNDERLYING = "UPRO", "SSO", "SPY"


class LeveragedTrendNasdaq100(_LeveragedTrendBase):
    name = "leveraged_trend_nasdaq100"
    ETF_3X, ETF_2X, UNDERLYING = "TQQQ", "QLD", "QQQ"


class LeveragedTrendRussell2000(_LeveragedTrendBase):
    name = "leveraged_trend_russell2000"
    ETF_3X, ETF_2X, UNDERLYING = "TNA", "UWM", "IWM"
