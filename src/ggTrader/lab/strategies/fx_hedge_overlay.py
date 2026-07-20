"""Dynamic FX hedge overlay (carry + PPP-value + trend) -- candidate A1 from
WEB_RESEARCH_CANDIDATES.md's 2026-07-19 cross-asset register, the top-ranked
pick for this project's home-lab/ETF workflow (Castro/Hamill/Harber/Harvey/
Van Hemert, "The Best Strategies for FX Hedging," JPM 2025).

Retail-proxy implementation: for each of a small set of hedged/unhedged ETF
pairs tracking the same underlying international-equity exposure, allocate
between the unhedged and currency-hedged share class based on three signals
per the source paper -- carry (foreign short rate minus US short rate),
PPP-value (real exchange rate deviation from its own trailing history), and
trend (medium-term momentum of the unhedged/hedged price ratio, which
isolates the currency return net of the shared equity beta). This is an
approximation of the paper's forward-based hedge-ratio construction, not a
replication -- see the register entry for the explicit caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import extract_close
from ggTrader.lab.strategy import LabConfig, Plan

US_RATE_SERIES = "TB3MS"
US_CPI_SERIES = "CPIAUCSL"


@dataclass(frozen=True)
class FxHedgePair:
    unhedged: str
    hedged: str
    currency: str
    foreign_rate_series: str
    foreign_cpi_series: str
    spot_fx_series: str
    #: True if spot_fx_series quotes foreign-currency-per-USD (e.g. FRED's
    #: DEXJPUS is JPY per USD) -- must invert to a consistent USD-per-foreign
    #: convention before combining with the CPI ratio, or the real_fx trend's
    #: sign flips relative to pairs quoted the other way (e.g. DEXUSEU).
    invert_spot: bool


#: Only currently-active hedged/unhedged pairs. Several single-country
#: hedged ETFs (HEWG, HEWU) were delisted 2023-2024 -- deliberately excluded.
FX_HEDGE_PAIRS: List[FxHedgePair] = [
    FxHedgePair("EWJ", "DXJ", "JPY", "IRSTCI01JPM156N", "CPALTT01JPM659N", "DEXJPUS", True),
    # CP0000EZ19M086NEST: Eurozone HICP (Eurostat, via FRED) -- the
    # originally-chosen CPALTT01EZM659N does not exist as a FRED series
    # (verified: fredgraph.csv returns HTTP 404 for it, caught only by
    # checking response content, not just line count -- an earlier
    # curl-only check miscounted an HTML error page's lines as CSV rows).
    FxHedgePair("EZU", "HEZU", "EUR", "IRSTCI01EZM156N", "CP0000EZ19M086NEST", "DEXUSEU", False),
]

FredLoader = Callable[[str, pd.Timestamp], pd.Series]


def _default_fred_loader(series_id: str, asof: pd.Timestamp) -> pd.Series:
    from ggTrader.lab.fred_data import PUBLISH_LAG_DAYS, available_as_of, load_fred_series

    start = (asof - pd.Timedelta(days=3650)).strftime("%Y-%m-%d")
    end = asof.strftime("%Y-%m-%d")
    df = load_fred_series(series_id, start, end)
    if df.empty:
        return pd.Series(dtype=float)
    df = available_as_of(df, asof, lag_days=PUBLISH_LAG_DAYS)
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["value"].sort_index()


def rolling_zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / std


def real_fx_index(spot: pd.Series, us_cpi: pd.Series, foreign_cpi: pd.Series) -> pd.Series:
    """PPP-adjusted real exchange rate (USD-per-foreign-currency convention):
    nominal spot scaled by relative CPI growth since the start of the
    aligned window. Falling values mean the foreign currency has gotten
    cheaper in real terms than nominal spot alone would suggest -- a
    mean-reversion (value) signal, not the trend signal (see trend_signal,
    which uses the ETF price ratio instead)."""
    us_idx = us_cpi / us_cpi.iloc[0]
    foreign_idx = foreign_cpi / foreign_cpi.iloc[0]
    return spot * us_idx / foreign_idx


def trend_signal(ratio: pd.Series, lookback: int, skip: int) -> float:
    """12-1 month momentum of a price ratio (unhedged/hedged), matching this
    lab's standard momentum convention (LabConfig.lookback/skip). NaN if
    there isn't enough history."""
    if len(ratio) < lookback + skip:
        return float("nan")
    window = ratio.iloc[-(lookback + skip) :]
    start = window.iloc[0]
    end = window.iloc[-skip - 1] if skip > 0 else window.iloc[-1]
    if pd.isna(start) or pd.isna(end) or start <= 0 or end <= 0:
        return float("nan")
    return float(np.log(end / start))


def unhedged_weight(score: float, k: float = 2.25) -> float:
    """Squash a combined carry+trend+value score into an unhedged
    allocation fraction in (0, 1); score=0 -> 0.5 (no tilt)."""
    return 0.5 * (1.0 + float(np.tanh(score / k)))


class FxHedgeOverlayStrategy:
    """Long-only weights sleeve: allocate each pair's fixed share between
    its unhedged and hedged ETF based on a carry+PPP-value+trend score,
    rebalanced monthly. Pairs not in ``data`` or without enough history are
    skipped (partial coverage degrades gracefully rather than failing the
    whole select() call).
    """

    name = "fx_hedge_overlay"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        k: float = 2.25,
        carry_scale: float = 2.0,
        trend_scale: float = 0.10,
        trend_lookback: int = 252,
        trend_skip: int = 21,
        value_zscore_window: int = 60,
        _fred_loader: FredLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.k = k
        self.carry_scale = carry_scale
        self.trend_scale = trend_scale
        self.trend_lookback = trend_lookback
        self.trend_skip = trend_skip
        self.value_zscore_window = value_zscore_window
        self._fred_loader: FredLoader = _fred_loader or _default_fred_loader

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "k": [1.5, 2.25, 3.0],
            "trend_lookback": [126, 252],
        }

    def _pair_score(self, pair: FxHedgePair, ratio: pd.Series, asof: pd.Timestamp) -> float | None:
        trend_raw = trend_signal(ratio, self.trend_lookback, self.trend_skip)
        if pd.isna(trend_raw):
            return None

        us_rate = self._fred_loader(US_RATE_SERIES, asof)
        foreign_rate = self._fred_loader(pair.foreign_rate_series, asof)
        us_cpi = self._fred_loader(US_CPI_SERIES, asof)
        foreign_cpi = self._fred_loader(pair.foreign_cpi_series, asof)
        spot = self._fred_loader(pair.spot_fx_series, asof)
        if us_rate.empty or foreign_rate.empty or us_cpi.empty or foreign_cpi.empty or spot.empty:
            return None
        if pair.invert_spot:
            spot = 1.0 / spot.replace(0.0, np.nan)

        carry_diff = float(foreign_rate.iloc[-1] - us_rate.iloc[-1])

        common_idx = us_cpi.index.intersection(foreign_cpi.index).intersection(spot.index)
        if len(common_idx) < 2:
            value_z = 0.0
        else:
            aligned_spot = spot.loc[common_idx].sort_index()
            aligned_us_cpi = us_cpi.loc[common_idx].sort_index()
            aligned_foreign_cpi = foreign_cpi.loc[common_idx].sort_index()
            real_fx = real_fx_index(aligned_spot, aligned_us_cpi, aligned_foreign_cpi)
            min_periods = max(6, self.value_zscore_window // 3)
            value_z_series = rolling_zscore(real_fx, self.value_zscore_window, min_periods)
            last = value_z_series.iloc[-1] if len(value_z_series) else float("nan")
            value_z = float(last) if pd.notna(last) else 0.0

        carry_component = float(np.tanh(carry_diff / self.carry_scale))
        trend_component = float(np.tanh(trend_raw / self.trend_scale))
        value_component = float(np.tanh(-value_z))
        return carry_component + trend_component + value_component

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        have = set(data.columns.get_level_values(0).unique())
        per_pair_share = 1.0 / len(FX_HEDGE_PAIRS)

        plan: Plan = []
        for pair in FX_HEDGE_PAIRS:
            if pair.unhedged not in have or pair.hedged not in have:
                continue
            close = extract_close(data, [pair.unhedged, pair.hedged]).dropna()
            if len(close) < self.cfg.min_history_bars:
                continue

            ratio = close[pair.unhedged] / close[pair.hedged]
            score = self._pair_score(pair, ratio, asof)
            if score is None:
                continue

            w_unhedged = unhedged_weight(score, k=self.k)
            plan.append({"symbol": pair.unhedged, "weight": per_pair_share * w_unhedged})
            plan.append({"symbol": pair.hedged, "weight": per_pair_share * (1.0 - w_unhedged)})

        return plan

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
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
