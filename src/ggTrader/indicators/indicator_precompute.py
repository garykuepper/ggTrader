"""Indicator pre-computation and caching layer for vectorized signal generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt


class IndicatorPrecomputer:
    """Computes technical indicators once per parameter range and caches results.

    Eliminates redundant per-combo indicator recalculation by pre-computing
    each indicator (PSAR, ADX, ATR) across all parameter values, then using
    numpy broadcasting to combine them in signal logic.
    """

    def __init__(self, close: np.ndarray, high: np.ndarray, low: np.ndarray):
        """Initialize with OHLCV arrays.

        Args:
            close: Close prices (n_time, n_symbols) or (n_time,).
            high: High prices, same shape as close.
            low: Low prices, same shape as close.
        """
        self.close = self._to_ndarray(close)
        self.high = self._to_ndarray(high)
        self.low = self._to_ndarray(low)

        # Cache: (indicator_name, frozen_param_tuple) -> indicator_result
        self._cache: dict[tuple[str, Any], Any] = {}

    @staticmethod
    def _to_ndarray(x: object) -> np.ndarray:
        """Convert pandas Series/DataFrame to ndarray."""
        if hasattr(x, "values"):
            return np.asarray(x.values, dtype=np.float64)
        return np.asarray(x, dtype=np.float64)

    def _make_cache_key(self, indicator_name: str, params: dict) -> tuple[str, Any]:
        """Create a cache key from indicator name and frozen parameters."""
        frozen_params = tuple(sorted(params.items()))
        return (indicator_name, frozen_params)

    def compute_psar(
        self,
        sar_acceleration_values: list[float] | float = 0.02,
        sar_maximum_values: list[float] | float = 0.2,
    ) -> Any:
        """Pre-compute PSAR across parameter ranges.

        Args:
            sar_acceleration_values: Single value or list of acceleration values.
            sar_maximum_values: Single value or list of maximum values.

        Returns:
            VBT indicator object with param_product structure.
        """
        accel = sar_acceleration_values if isinstance(sar_acceleration_values, list) else [sar_acceleration_values]
        maxim = sar_maximum_values if isinstance(sar_maximum_values, list) else [sar_maximum_values]

        cache_key = self._make_cache_key("psar", {"acceleration": tuple(accel), "maximum": tuple(maxim)})
        if cache_key in self._cache:
            return self._cache[cache_key]

        psar_ind = vbt.IndicatorFactory.from_pandas_ta("psar").run(
            self.high,
            self.low,
            close=self.close,
            acceleration=accel,
            maximum=maxim,
            param_product=True,
        )
        self._cache[cache_key] = psar_ind
        return psar_ind

    def compute_adx(
        self,
        adx_length_values: list[int] | int = 14,
    ) -> Any:
        """Pre-compute ADX across parameter ranges.

        Args:
            adx_length_values: Single value or list of ADX period lengths.

        Returns:
            VBT indicator object with param_product structure.
        """
        lengths = adx_length_values if isinstance(adx_length_values, list) else [adx_length_values]

        cache_key = self._make_cache_key("adx", {"length": tuple(int(l) for l in lengths)})
        if cache_key in self._cache:
            return self._cache[cache_key]

        adx_ind = vbt.IndicatorFactory.from_pandas_ta("adx").run(
            self.high, self.low, self.close, length=[int(l) for l in lengths], param_product=True
        )
        self._cache[cache_key] = adx_ind
        return adx_ind

    def compute_atr(
        self,
        atr_length_values: list[int] | int = 14,
    ) -> Any:
        """Pre-compute ATR across parameter ranges.

        Args:
            atr_length_values: Single value or list of ATR period lengths.

        Returns:
            VBT indicator object with param_product structure.
        """
        lengths = atr_length_values if isinstance(atr_length_values, list) else [atr_length_values]

        cache_key = self._make_cache_key("atr", {"length": tuple(int(l) for l in lengths)})
        if cache_key in self._cache:
            return self._cache[cache_key]

        atr_ind = vbt.IndicatorFactory.from_pandas_ta("atr").run(
            self.high, self.low, self.close, length=[int(l) for l in lengths], param_product=True
        )
        self._cache[cache_key] = atr_ind
        return atr_ind

    def compute_ema(
        self,
        ema_length_values: list[int] | int,
    ) -> Any:
        """Pre-compute EMA across parameter ranges.

        Args:
            ema_length_values: Single value or list of EMA period lengths.

        Returns:
            VBT indicator object with param_product structure.
        """
        lengths = ema_length_values if isinstance(ema_length_values, list) else [ema_length_values]

        cache_key = self._make_cache_key("ema", {"length": tuple(int(l) for l in lengths)})
        if cache_key in self._cache:
            return self._cache[cache_key]

        ema_ind = vbt.IndicatorFactory.from_pandas_ta("ema").run(
            self.close, length=[int(l) for l in lengths], param_product=True
        )
        self._cache[cache_key] = ema_ind
        return ema_ind

    def compute_rsi(
        self,
        rsi_length_values: list[int] | int = 14,
    ) -> Any:
        """Pre-compute RSI across parameter ranges.

        Args:
            rsi_length_values: Single value or list of RSI period lengths.

        Returns:
            VBT indicator object with param_product structure.
        """
        lengths = rsi_length_values if isinstance(rsi_length_values, list) else [rsi_length_values]

        cache_key = self._make_cache_key("rsi", {"length": tuple(int(l) for l in lengths)})
        if cache_key in self._cache:
            return self._cache[cache_key]

        rsi_ind = vbt.IndicatorFactory.from_pandas_ta("rsi").run(
            self.close, length=[int(l) for l in lengths], param_product=True
        )
        self._cache[cache_key] = rsi_ind
        return rsi_ind

    def clear_cache(self) -> None:
        """Clear the indicator cache to free memory."""
        self._cache.clear()

    def get_cache_info(self) -> dict[str, int]:
        """Return cache statistics."""
        return {"cached_indicators": len(self._cache)}
