"""Treasury term-structure factors -- candidate A5 from
WEB_RESEARCH_CANDIDATES.md's 2026-07-19 cross-asset register, explicitly
labeled an ETF APPROXIMATION, not a replication.

Source: Filipović, Pelger & Ye, "Shrinking the Term Structure," NBER WP
32472 (2024), now accepted at the Review of Finance -- proposes FOUR
investable term-structure factors from a rich cross-section of Treasury
cash flows. A three-ETF implementation (SHY/IEF/TLT) provides only three
broad duration buckets and cannot be presumed to reproduce the paper's
fourth "complexity" factor (tied to recession performance) -- this is an
exploratory approximation, not the paper's actual model.

Mechanism: a curve steepener/flattener regime signal using real FRED
constant-maturity Treasury yields (DGS2/DGS10, free, no API key -- same
fred_data.py module built for candidate A1). When the 10y-2y curve slope
is running unusually steep vs its own trailing history, go all-in on
long duration (TLT) to capture rolldown; when unusually flat/inverted, go
all-in short duration (SHY) to avoid duration risk; in between, hold the
belly (IEF). One-hot regime allocation, not a continuous blend -- kept
simple and testable given the explicit approximation framing.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

from ggTrader.lab.strategies.indicators import extract_close
from ggTrader.lab.strategy import LabConfig, Plan

TREASURY_CURVE_UNIVERSE = ["SHY", "IEF", "TLT"]

LONG_YIELD_SERIES = "DGS10"
SHORT_YIELD_SERIES = "DGS2"

FredLoader = Callable[[str, pd.Timestamp], pd.Series]


def _default_fred_loader(series_id: str, asof: pd.Timestamp) -> pd.Series:
    from ggTrader.lab.fred_data import available_as_of, load_fred_series

    # Yield series publish with only a short reporting lag (a few business
    # days), unlike CPI/policy-rate monthly series -- a conservative fixed
    # lag is still applied for point-in-time safety.
    start = (asof - pd.Timedelta(days=3650)).strftime("%Y-%m-%d")
    end = asof.strftime("%Y-%m-%d")
    df = load_fred_series(series_id, start, end)
    if df.empty:
        return pd.Series(dtype=float)
    df = available_as_of(df, asof, lag_days=5)
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["value"].sort_index()


def curve_slope_zscore(slope: pd.Series, zscore_window: int) -> float:
    """Latest z-score of a yield-curve slope series against its own
    trailing history. 0.0 if there isn't enough history to judge."""
    hist = slope.dropna()
    if len(hist) < max(zscore_window // 5, 20):
        return 0.0
    window = hist.iloc[-zscore_window:]
    mean, std = window.mean(), window.std()
    if std == 0 or not pd.notna(std):
        return 0.0
    return float((window.iloc[-1] - mean) / std)


def curve_regime_weights(
    z: float, steep_threshold: float, flat_threshold: float
) -> Dict[str, float]:
    """One-hot regime allocation: steep curve -> long duration (TLT),
    flat/inverted -> short duration (SHY), neutral -> belly (IEF)."""
    if z > steep_threshold:
        return {"TLT": 1.0, "IEF": 0.0, "SHY": 0.0}
    if z < flat_threshold:
        return {"TLT": 0.0, "IEF": 0.0, "SHY": 1.0}
    return {"TLT": 0.0, "IEF": 1.0, "SHY": 0.0}


class TreasuryCurveStrategy:
    """Long-only weights sleeve: one-hot allocate across SHY/IEF/TLT based
    on a real-yield curve-steepness regime signal, rebalanced monthly."""

    name = "treasury_curve"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        zscore_window: int = 1260,
        steep_threshold: float = 1.0,
        flat_threshold: float = -1.0,
        _fred_loader: FredLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.zscore_window = zscore_window
        self.steep_threshold = steep_threshold
        self.flat_threshold = flat_threshold
        self._fred_loader: FredLoader = _fred_loader or _default_fred_loader

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "steep_threshold": [0.5, 1.0, 1.5],
            "flat_threshold": [-1.5, -1.0, -0.5],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        have = set(data.columns.get_level_values(0).unique())
        if not set(TREASURY_CURVE_UNIVERSE).issubset(have):
            return []

        close = extract_close(data, TREASURY_CURVE_UNIVERSE).dropna()
        if len(close) < self.cfg.min_history_bars:
            return []

        long_yield = self._fred_loader(LONG_YIELD_SERIES, asof)
        short_yield = self._fred_loader(SHORT_YIELD_SERIES, asof)
        if long_yield.empty or short_yield.empty:
            return []

        common_idx = long_yield.index.intersection(short_yield.index)
        if len(common_idx) < 20:
            return []
        slope = (long_yield.loc[common_idx] - short_yield.loc[common_idx]).sort_index()

        z = curve_slope_zscore(slope, self.zscore_window)
        weights = curve_regime_weights(z, self.steep_threshold, self.flat_threshold)

        return [{"symbol": sym, "weight": w} for sym, w in weights.items() if w > 0.0]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        symbols = sorted(
            {s["symbol"] for plan in plans.values() for s in plan} | set(TREASURY_CURVE_UNIVERSE)
        )
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets
