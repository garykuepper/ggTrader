"""Indicator pre-computation and caching layer for vectorized signal generation."""

from __future__ import annotations

from types import SimpleNamespace
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

        if len(accel) == 1 and len(maxim) == 1:
            psar_ind = vbt.IndicatorFactory.from_pandas_ta("psar").run(
                self.high,
                self.low,
                close=self.close,
                acceleration=float(accel[0]),
                maximum=float(maxim[0]),
            )
        else:
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

        # Single length + param_product breaks some pandas_ta/VBT builds (output split mismatch).
        if len(lengths) == 1:
            adx_ind = vbt.IndicatorFactory.from_pandas_ta("adx").run(
                self.high, self.low, self.close, length=int(lengths[0])
            )
        else:
            adx_ind = vbt.IndicatorFactory.from_pandas_ta("adx").run(
                self.high,
                self.low,
                self.close,
                length=[int(l) for l in lengths],
                param_product=True,
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

        if len(lengths) == 1:
            atr_ind = vbt.IndicatorFactory.from_pandas_ta("atr").run(
                self.high, self.low, self.close, length=int(lengths[0])
            )
        else:
            atr_ind = vbt.IndicatorFactory.from_pandas_ta("atr").run(
                self.high,
                self.low,
                self.close,
                length=[int(l) for l in lengths],
                param_product=True,
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

        if len(lengths) == 1:
            ema_ind = vbt.IndicatorFactory.from_pandas_ta("ema").run(
                self.close, length=int(lengths[0])
            )
        else:
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

        if len(lengths) == 1:
            rsi_ind = vbt.IndicatorFactory.from_pandas_ta("rsi").run(
                self.close, length=int(lengths[0])
            )
        else:
            rsi_ind = vbt.IndicatorFactory.from_pandas_ta("rsi").run(
                self.close, length=[int(l) for l in lengths], param_product=True
            )
        self._cache[cache_key] = rsi_ind
        return rsi_ind

    def compute_macd(
        self,
        fast_values: list[int] | int,
        slow_values: list[int] | int,
        signal_values: list[int] | int,
    ) -> Any:
        """Pre-compute MACD (macd line + signal) across parameter ranges."""
        fasts = fast_values if isinstance(fast_values, list) else [fast_values]
        slows = slow_values if isinstance(slow_values, list) else [slow_values]
        signals = signal_values if isinstance(signal_values, list) else [signal_values]

        cache_key = self._make_cache_key(
            "macd",
            {
                "fast": tuple(int(x) for x in fasts),
                "slow": tuple(int(x) for x in slows),
                "signal": tuple(int(x) for x in signals),
            },
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        if len(fasts) == 1 and len(slows) == 1 and len(signals) == 1:
            macd_ind = vbt.IndicatorFactory.from_pandas_ta("macd").run(
                self.close,
                fast=int(fasts[0]),
                slow=int(slows[0]),
                signal=int(signals[0]),
            )
        else:
            macd_ind = vbt.IndicatorFactory.from_pandas_ta("macd").run(
                self.close,
                fast=[int(x) for x in fasts],
                slow=[int(x) for x in slows],
                signal=[int(x) for x in signals],
                param_product=True,
            )
        self._cache[cache_key] = macd_ind
        return macd_ind

    def compute_bbands(
        self,
        length_values: list[int] | int,
        std: float = 2.0,
    ) -> Any:
        """Pre-compute Bollinger bands across lengths (fixed ``std`` per VBT multi-col limits)."""
        lengths = length_values if isinstance(length_values, list) else [length_values]
        std_f = float(std)

        cache_key = self._make_cache_key(
            "bbands",
            {"length": tuple(int(l) for l in lengths), "std": std_f},
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        if len(lengths) == 1:
            bb_ind = vbt.IndicatorFactory.from_pandas_ta("bbands").run(
                self.close, length=int(lengths[0]), std=std_f
            )
        else:
            bb_ind = vbt.IndicatorFactory.from_pandas_ta("bbands").run(
                self.close,
                length=[int(l) for l in lengths],
                std=std_f,
                param_product=True,
            )
        self._cache[cache_key] = bb_ind
        return bb_ind

    def compute_donchian(
        self,
        length_values: list[int] | int,
    ) -> Any:
        """Pre-compute Donchian channels (symmetric upper/lower length).

        Multiple lengths are stacked per run: VBT ``param_product`` on paired
        lengths does not yield reliable distinct columns for all builds.
        """
        lengths = length_values if isinstance(length_values, list) else [length_values]

        cache_key = self._make_cache_key(
            "donchian", {"length": tuple(int(l) for l in lengths)}
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        if len(lengths) == 1:
            L = int(lengths[0])
            dc_ind = vbt.IndicatorFactory.from_pandas_ta("donchian").run(
                self.high, self.low, lower_length=L, upper_length=L
            )
            self._cache[cache_key] = dc_ind
            return dc_ind

        dcu_stack: list[np.ndarray] = []
        dcl_stack: list[np.ndarray] = []
        for L in lengths:
            Li = int(L)
            dc_i = vbt.IndicatorFactory.from_pandas_ta("donchian").run(
                self.high, self.low, lower_length=Li, upper_length=Li
            )
            dcu = dc_i.dcu.values if hasattr(dc_i.dcu, "values") else dc_i.dcu
            dcl = dc_i.dcl.values if hasattr(dc_i.dcl, "values") else dc_i.dcl
            dcu_stack.append(np.asarray(dcu, dtype=np.float64))
            dcl_stack.append(np.asarray(dcl, dtype=np.float64))
        dcu_3d = np.stack(dcu_stack, axis=1)
        dcl_3d = np.stack(dcl_stack, axis=1)
        wrapped = SimpleNamespace(dcu=dcu_3d, dcl=dcl_3d)
        self._cache[cache_key] = wrapped
        return wrapped

    def compute_supertrend(
        self,
        length_values: list[int] | int,
        multiplier_values: list[float] | float,
    ) -> Any:
        """Pre-compute Supertrend direction ``supertd`` across (length, multiplier).

        Multi-parameter runs stack per (length, mult): VBT ``param_product`` is
        unreliable for ``supertd`` with multiple symbols.
        """
        lengths = length_values if isinstance(length_values, list) else [length_values]
        mults = multiplier_values if isinstance(multiplier_values, list) else [
            multiplier_values
        ]

        cache_key = self._make_cache_key(
            "supertrend",
            {
                "length": tuple(int(l) for l in lengths),
                "mult": tuple(float(m) for m in mults),
            },
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        if len(lengths) == 1 and len(mults) == 1:
            st_ind = vbt.IndicatorFactory.from_pandas_ta("supertrend").run(
                self.high,
                self.low,
                self.close,
                length=int(lengths[0]),
                multiplier=float(mults[0]),
            )
            self._cache[cache_key] = st_ind
            return st_ind

        td_stack: list[np.ndarray] = []
        for L in lengths:
            for m in mults:
                st_i = vbt.IndicatorFactory.from_pandas_ta("supertrend").run(
                    self.high,
                    self.low,
                    self.close,
                    length=int(L),
                    multiplier=float(m),
                )
                td = st_i.supertd.values if hasattr(st_i.supertd, "values") else st_i.supertd
                td_stack.append(np.asarray(td, dtype=np.float64))
        td_3d = np.stack(td_stack, axis=1)
        wrapped = SimpleNamespace(supertd=td_3d)
        self._cache[cache_key] = wrapped
        return wrapped

    def clear_cache(self) -> None:
        """Clear the indicator cache to free memory."""
        self._cache.clear()

    def get_cache_info(self) -> dict[str, int]:
        """Return cache statistics."""
        return {"cached_indicators": len(self._cache)}
