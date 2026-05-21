"""SYNTHETIC feature data — TEMPORARY Phase 3 / Phase 3.5 stand-in.

================================================================================
THIS IS NOT REAL DATA.
================================================================================

Phase 3 validated the new architecture. Real Kraken quarterly-future historical
OHLCV is not yet in TimescaleDB (see ``docs/kraken_futures_backfill_design.md``).
Until it lands, this module emits a deterministic synthetic basis_apr series +
a matching spot series + per-contract futures mid-prices derived from
``spot * (1 + basis_apr * dte/365)``.

When the real backfill lands:
  1. Add ``features/price.py``     reading ``mid_price`` from TimescaleDB.
  2. Add ``features/derivatives.py`` reading ``basis_apr`` from TimescaleDB.
  3. Wire them into a real ``FeatureStore`` (Phase 2 proper).
  4. Delete this file. Strategy + backtest code should not change.

Phase 3.5 changes: the synthetic store now satisfies the widened ``FeatureStore``
Protocol from ``ggTrader.features.base`` and serves ``mid_price`` as a standard
feature (Pricer Callable removed from the backtest engine).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd

from ggTrader.core.instrument import Instrument
from ggTrader.core.universe import Universe
from ggTrader.features.base import (
    FeatureStore,
    InstrumentArg,
    _normalize_instruments,
    pair_column_label,
)


@dataclass(frozen=True)
class SyntheticBasisConfig:
    start: datetime
    end: datetime
    seed: int = 42
    mean_apr: float = 0.05
    long_cycle_amplitude: float = 0.15
    long_cycle_period_days: int = 365
    short_cycle_amplitude: float = 0.03
    short_cycle_period_days: int = 30
    noise_sigma: float = 0.01
    spot_start_price: float = 40_000.0
    spot_drift_annual: float = 0.4
    spot_vol_annual: float = 0.6


def _daily_index(start: datetime, end: datetime) -> pd.DatetimeIndex:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be tz-aware UTC")
    return pd.date_range(start=start, end=end, freq="1D", tz=timezone.utc)


def generate_synthetic_basis(config: SyntheticBasisConfig) -> pd.Series:
    idx = _daily_index(config.start, config.end)
    rng = np.random.default_rng(config.seed)
    t = np.arange(len(idx), dtype=float)
    long_cycle = config.long_cycle_amplitude * np.sin(
        2 * math.pi * t / config.long_cycle_period_days
    )
    short_cycle = config.short_cycle_amplitude * np.sin(
        2 * math.pi * t / config.short_cycle_period_days
    )
    noise = rng.normal(0.0, config.noise_sigma, size=len(idx))
    return pd.Series(
        config.mean_apr + long_cycle + short_cycle + noise, index=idx, name="basis_apr"
    )


def generate_synthetic_spot(config: SyntheticBasisConfig) -> pd.Series:
    idx = _daily_index(config.start, config.end)
    rng = np.random.default_rng(config.seed + 1)
    dt = 1.0 / 365.0
    drift = (config.spot_drift_annual - 0.5 * config.spot_vol_annual**2) * dt
    diffusion = config.spot_vol_annual * math.sqrt(dt) * rng.standard_normal(len(idx))
    log_returns = drift + diffusion
    log_returns[0] = 0.0
    prices = config.spot_start_price * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=idx, name="spot_price")


def _parse_expiry(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class SyntheticFeatureStore:
    """In-memory FeatureStore serving ``mid_price`` (single) and ``basis_apr``
    (pair). Satisfies the widened ``FeatureStore`` Protocol."""

    def __init__(
        self,
        config: SyntheticBasisConfig,
        spot_instrument: Instrument,
        future_instruments: Sequence[Instrument],
    ) -> None:
        self._spot_series = generate_synthetic_spot(config)
        self._basis_series = generate_synthetic_basis(config)
        self._spot_instrument = spot_instrument
        self._future_by_symbol = {f.symbol: f for f in future_instruments}
        self._index = self._spot_series.index

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._index)

    # ------------------------------------------------------------------ get

    def get(
        self,
        feature_name: str,
        instruments: InstrumentArg,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        insts = _normalize_instruments(instruments, start)
        if feature_name == "mid_price":
            return self._mid_price_frame(insts, start, end)
        if feature_name == "basis_apr":
            return self._basis_apr_frame(insts, start, end)
        raise KeyError(f"unknown feature: {feature_name}")

    def get_at(
        self,
        feature_name: str,
        instruments: InstrumentArg,
        ts: datetime,
    ) -> pd.Series:
        if ts.tzinfo is None:
            raise ValueError("ts must be tz-aware UTC")
        df = self.get(feature_name, instruments, ts, ts)
        try:
            row = df.loc[ts]
        except KeyError as exc:
            raise LookupError(f"no value for {feature_name} at {ts.isoformat()}") from exc
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    # ------------------------------------------------------------------ internals

    def _slice(self, series: pd.Series, start: datetime, end: datetime) -> pd.Series:
        return series.loc[(series.index >= start) & (series.index <= end)]

    def _spot_in_window(self, start: datetime, end: datetime) -> pd.Series:
        return self._slice(self._spot_series, start, end)

    def _basis_in_window(self, start: datetime, end: datetime) -> pd.Series:
        return self._slice(self._basis_series, start, end)

    def _mid_price_frame(
        self, instruments: Sequence[Instrument], start: datetime, end: datetime
    ) -> pd.DataFrame:
        spot = self._spot_in_window(start, end)
        basis = self._basis_in_window(start, end)
        cols: dict[str, pd.Series] = {}
        for inst in instruments:
            if inst.symbol == self._spot_instrument.symbol:
                cols[inst.symbol] = spot
                continue
            if inst.symbol not in self._future_by_symbol:
                raise KeyError(f"no mid_price for {inst.symbol}")
            future = self._future_by_symbol[inst.symbol]
            expiry = _parse_expiry(str(future.expiry))
            index_dt = pd.DatetimeIndex(spot.index).to_pydatetime()
            dte_days = np.array(
                [(expiry - ts).total_seconds() / 86400.0 for ts in index_dt],
                dtype=float,
            )
            dte_days = np.maximum(dte_days, 0.0)
            cols[inst.symbol] = pd.Series(
                spot.values * (1.0 + basis.values * (dte_days / 365.0)),
                index=spot.index,
                name=inst.symbol,
            )
        return pd.DataFrame(cols)

    def _basis_apr_frame(
        self, instruments: Sequence[Instrument], start: datetime, end: datetime
    ) -> pd.DataFrame:
        if len(instruments) != 2:
            raise KeyError(
                f"basis_apr is arity='pair'; expected 2 instruments, got {len(instruments)}"
            )
        spot_inst, future_inst = instruments
        if spot_inst.symbol != self._spot_instrument.symbol:
            raise KeyError(f"basis_apr expects spot at index 0; got {spot_inst.symbol}")
        if future_inst.symbol not in self._future_by_symbol:
            raise KeyError(f"unknown future contract: {future_inst.symbol}")
        basis = self._basis_in_window(start, end)
        col = pair_column_label(instruments)
        return pd.DataFrame({col: basis.values}, index=basis.index)


def synthetic_quarterly_expiries(start: datetime, end: datetime) -> list[datetime]:
    """Synthetic last-Friday-of-quarter expiries (kept for reference / docs)."""
    expiries: list[datetime] = []
    year = start.year
    while year <= end.year + 1:
        for month in (3, 6, 9, 12):
            last_day = (
                (datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1))
                if month != 12
                else datetime(year, 12, 31, tzinfo=timezone.utc)
            )
            while last_day.weekday() != 4:
                last_day -= timedelta(days=1)
            expiry = last_day.replace(hour=16, minute=0, second=0, microsecond=0)
            if start <= expiry <= end + timedelta(days=120):
                expiries.append(expiry)
        year += 1
    return expiries


def select_active_future(
    expiries: Sequence[datetime], ts: datetime, roll_buffer_hours: int = 24
) -> datetime:
    threshold = ts + timedelta(hours=roll_buffer_hours)
    for exp in sorted(expiries):
        if exp > threshold:
            return exp
    raise LookupError(f"no active future after {ts.isoformat()}")


# Re-export Protocol so call sites can import from the same module if convenient
__all__ = [
    "FeatureStore",
    "SyntheticBasisConfig",
    "SyntheticFeatureStore",
    "generate_synthetic_basis",
    "generate_synthetic_spot",
    "select_active_future",
    "synthetic_quarterly_expiries",
]


# ------------------------------------------------------------------ legacy shim


class _LegacyUniverse(Universe):
    """Compatibility shim — unused, retained to avoid breaking type imports."""

    def members(self, ts: datetime) -> list[Instrument]:  # pragma: no cover
        return []

    def is_dynamic(self) -> bool:  # pragma: no cover
        return False
