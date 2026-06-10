"""Indicator pre-computation and caching layer for vectorized signal generation."""

import hashlib
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt


def get_data_hash(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> str:
    """Create a unique hash of the input price data.

    Combined bytes of all 3 arrays ensure different assets/windows never collide.
    """
    combined = b"".join([high.tobytes(), low.tobytes(), close.tobytes()])
    return hashlib.sha256(combined).hexdigest()[:16]


class IndicatorPrecomputer:
    """Computes technical indicators once per parameter range and caches results.

    Eliminates redundant per-combo indicator recalculation by pre-computing
    each indicator (PSAR, ADX, ATR, ...) across all parameter values, then
    using numpy broadcasting to combine them in signal logic.

    Caching is two-level and in-memory only (the old gzip-pickle disk layer
    was removed — research re-runs recompute, which is cheap):
    - ``self._cache``: per-instance, cleared by ``clear_cache()``.
    - ``IndicatorPrecomputer._shared_cache``: process-wide, keyed by
      (data_hash, indicator, params), so repeated runs over the same window
      (e.g. one FastBacktest per strategy combo in a WFO sweep) reuse
      indicator arrays across instances. Bounded FIFO to keep memory flat
      in long multi-window loops.
    """

    _shared_cache: dict[tuple[str, str, Any], Any] = {}
    _shared_cache_max_entries: int = 256

    def __init__(self, close: np.ndarray, high: np.ndarray, low: np.ndarray):
        """Initialize with OHLCV arrays."""
        self.close = self._to_ndarray(close)
        self.high = self._to_ndarray(high)
        self.low = self._to_ndarray(low)

        # In-memory cache for the current run
        self._cache: dict[tuple[str, Any], Any] = {}

        self._data_hash = get_data_hash(self.high, self.low, self.close)

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

    def _wrap_indicator(self, ind: Any) -> SimpleNamespace:
        """Extract output arrays from a VBT indicator into a serializable namespace.

        This avoids pickle errors with dynamically created VectorBT classes.
        """
        if isinstance(ind, SimpleNamespace):
            return ind
        outputs = {}
        # VBT indicator classes have an _outputs tuple defined by the factory.
        out_names = getattr(ind.__class__, "_outputs", [])
        if not out_names and hasattr(ind, "wrapper"):
            # Fallback for some vbt objects or custom wrappers
            out_names = [
                k
                for k in dir(ind)
                if not k.startswith("_")
                and isinstance(getattr(ind, k), (pd.Series, pd.DataFrame, np.ndarray))
            ]

        for out_name in out_names:
            val = getattr(ind, out_name, None)
            if val is not None:
                if hasattr(val, "values"):
                    outputs[out_name] = np.asarray(val.values, dtype=np.float64)
                else:
                    outputs[out_name] = np.asarray(val, dtype=np.float64)
        return SimpleNamespace(**outputs)

    def _get_persistent(self, name: str, params: dict) -> Optional[Any]:
        """Check the instance cache, then the process-wide shared cache."""
        key = self._make_cache_key(name, params)
        if key in self._cache:
            return self._cache[key]

        shared_key = (self._data_hash, *key)
        data = IndicatorPrecomputer._shared_cache.get(shared_key)
        if data is not None:
            self._cache[key] = data
        return data

    def _save_persistent(self, name: str, params: dict, data: Any) -> SimpleNamespace:
        """Save to the instance cache and the process-wide shared cache.

        Returns the wrapped (plain-ndarray) form so compute_* methods hand out
        the same object on first call and cache hits alike.
        """
        wrapped = self._wrap_indicator(data)
        key = self._make_cache_key(name, params)
        self._cache[key] = wrapped

        shared = IndicatorPrecomputer._shared_cache
        shared[(self._data_hash, *key)] = wrapped
        while len(shared) > self._shared_cache_max_entries:
            shared.pop(next(iter(shared)))
        return wrapped

    def compute_psar(
        self,
        sar_acceleration_values: list[float] | float = 0.02,
        sar_maximum_values: list[float] | float = 0.2,
    ) -> Any:
        """Pre-compute PSAR across parameter ranges."""
        accel = (
            sar_acceleration_values
            if isinstance(sar_acceleration_values, list)
            else [sar_acceleration_values]
        )
        maxim = sar_maximum_values if isinstance(sar_maximum_values, list) else [sar_maximum_values]

        params = {"acceleration": tuple(accel), "maximum": tuple(maxim)}
        cached = self._get_persistent("psar", params)
        if cached is not None:
            return cached

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
        return self._save_persistent("psar", params, psar_ind)

    def compute_adx(
        self,
        adx_length_values: list[int] | int = 14,
    ) -> Any:
        """Pre-compute ADX across parameter ranges."""
        lengths = adx_length_values if isinstance(adx_length_values, list) else [adx_length_values]
        params = {"length": tuple(int(lo) for lo in lengths)}
        cached = self._get_persistent("adx", params)
        if cached is not None:
            return cached

        # Compute ADX per (length, symbol_col) and assemble as a 2D array.
        # VectorBT's IndicatorFactory has issues with multi-column + multi-param for
        # multi-output indicators; doing it manually is more reliable.
        # self.close is a 2D numpy array (n_time, n_symbols).
        import pandas_ta as ta

        n_time = self.close.shape[0]
        n_symbols = self.close.shape[1] if self.close.ndim == 2 else 1

        adx_cols, dmp_cols, dmn_cols = [], [], []
        for length in lengths:
            le = int(length)
            for sym_idx in range(n_symbols):
                h = self.high[:, sym_idx] if self.high.ndim == 2 else self.high
                lo = self.low[:, sym_idx] if self.low.ndim == 2 else self.low
                c = self.close[:, sym_idx] if self.close.ndim == 2 else self.close
                try:
                    res = ta.adx(pd.Series(h), pd.Series(lo), pd.Series(c), length=le)
                    if res is not None and hasattr(res, "shape") and res.shape[1] >= 3:
                        adx_cols.append(res.iloc[:, 0].values)
                        dmp_cols.append(res.iloc[:, 1].values)
                        dmn_cols.append(res.iloc[:, 2].values)
                    else:
                        raise ValueError("Unexpected ADX shape")
                except Exception:
                    nan_arr = np.full(n_time, np.nan)
                    adx_cols.append(nan_arr)
                    dmp_cols.append(nan_arr)
                    dmn_cols.append(nan_arr)

        adx_arr = np.column_stack(adx_cols) if adx_cols else np.full((n_time, 0), np.nan)
        dmp_arr = np.column_stack(dmp_cols) if dmp_cols else np.full((n_time, 0), np.nan)
        dmn_arr = np.column_stack(dmn_cols) if dmn_cols else np.full((n_time, 0), np.nan)
        adx_ind = SimpleNamespace(adx=adx_arr, dmp=dmp_arr, dmn=dmn_arr)
        return self._save_persistent("adx", params, adx_ind)

    def compute_atr(
        self,
        atr_length_values: list[int] | int = 14,
    ) -> Any:
        """Pre-compute ATR across parameter ranges."""
        lengths = atr_length_values if isinstance(atr_length_values, list) else [atr_length_values]

        params = {"length": tuple(int(lo) for lo in lengths)}
        cached = self._get_persistent("atr", params)
        if cached is not None:
            return cached

        if len(lengths) == 1:
            atr_ind = vbt.IndicatorFactory.from_pandas_ta("atr").run(
                self.high, self.low, self.close, length=int(lengths[0])
            )
        else:
            atr_ind = vbt.IndicatorFactory.from_pandas_ta("atr").run(
                self.high,
                self.low,
                self.close,
                length=[int(lo) for lo in lengths],
                param_product=True,
            )
        return self._save_persistent("atr", params, atr_ind)

    def compute_ema(
        self,
        ema_length_values: list[int] | int,
    ) -> Any:
        """Pre-compute EMA across parameter ranges."""
        lengths = ema_length_values if isinstance(ema_length_values, list) else [ema_length_values]

        params = {"length": tuple(int(lo) for lo in lengths)}
        cached = self._get_persistent("ema", params)
        if cached is not None:
            return cached

        if len(lengths) == 1:
            ema_ind = vbt.IndicatorFactory.from_pandas_ta("ema").run(
                self.close, length=int(lengths[0])
            )
        else:
            ema_ind = vbt.IndicatorFactory.from_pandas_ta("ema").run(
                self.close, length=[int(lo) for lo in lengths], param_product=True
            )
        return self._save_persistent("ema", params, ema_ind)

    def compute_rsi(
        self,
        rsi_length_values: list[int] | int = 14,
    ) -> Any:
        """Pre-compute RSI across parameter ranges."""
        lengths = rsi_length_values if isinstance(rsi_length_values, list) else [rsi_length_values]

        params = {"length": tuple(int(lo) for lo in lengths)}
        cached = self._get_persistent("rsi", params)
        if cached is not None:
            return cached

        if len(lengths) == 1:
            rsi_ind = vbt.IndicatorFactory.from_pandas_ta("rsi").run(
                self.close, length=int(lengths[0])
            )
        else:
            rsi_ind = vbt.IndicatorFactory.from_pandas_ta("rsi").run(
                self.close, length=[int(lo) for lo in lengths], param_product=True
            )
        return self._save_persistent("rsi", params, rsi_ind)

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

        params = {
            "fast": tuple(int(x) for x in fasts),
            "slow": tuple(int(x) for x in slows),
            "signal": tuple(int(x) for x in signals),
        }
        cached = self._get_persistent("macd", params)
        if cached is not None:
            return cached

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
        return self._save_persistent("macd", params, macd_ind)

    def compute_bbands(
        self,
        length_values: list[int] | int,
        std: float = 2.0,
    ) -> Any:
        """Pre-compute Bollinger bands across lengths."""
        lengths = length_values if isinstance(length_values, list) else [length_values]
        std_f = float(std)

        params = {"length": tuple(int(lo) for lo in lengths), "std": std_f}
        cached = self._get_persistent("bbands", params)
        if cached is not None:
            return cached

        if len(lengths) == 1:
            bb_ind = vbt.IndicatorFactory.from_pandas_ta("bbands").run(
                self.close, length=int(lengths[0]), std=std_f
            )
        else:
            bb_ind = vbt.IndicatorFactory.from_pandas_ta("bbands").run(
                self.close,
                length=[int(lo) for lo in lengths],
                std=std_f,
                param_product=True,
            )
        return self._save_persistent("bbands", params, bb_ind)

    def compute_donchian(
        self,
        length_values: list[int] | int,
    ) -> Any:
        """Pre-compute Donchian channels."""
        lengths = length_values if isinstance(length_values, list) else [length_values]

        params = {"length": tuple(int(le) for le in lengths)}
        cached = self._get_persistent("donchian", params)
        if cached is not None:
            return cached

        if len(lengths) == 1:
            le_val = int(lengths[0])
            dc_ind = vbt.IndicatorFactory.from_pandas_ta("donchian").run(
                self.high, self.low, lower_length=le_val, upper_length=le_val
            )
            return self._save_persistent("donchian", params, dc_ind)

        dcu_stack: list[np.ndarray] = []
        dcl_stack: list[np.ndarray] = []
        for le_val in lengths:
            le_i = int(le_val)
            dc_i = vbt.IndicatorFactory.from_pandas_ta("donchian").run(
                self.high, self.low, lower_length=le_i, upper_length=le_i
            )
            dcu = dc_i.dcu.values if hasattr(dc_i.dcu, "values") else dc_i.dcu
            dcl = dc_i.dcl.values if hasattr(dc_i.dcl, "values") else dc_i.dcl
            dcu_stack.append(np.asarray(dcu, dtype=np.float64))
            dcl_stack.append(np.asarray(dcl, dtype=np.float64))
        dcu_3d = np.stack(dcu_stack, axis=1)
        dcl_3d = np.stack(dcl_stack, axis=1)
        wrapped = SimpleNamespace(dcu=dcu_3d, dcl=dcl_3d)
        return self._save_persistent("donchian", params, wrapped)

    def compute_supertrend(
        self,
        length_values: list[int] | int,
        multiplier_values: list[float] | float,
    ) -> Any:
        """Pre-compute Supertrend direction across (length, multiplier)."""
        lengths = length_values if isinstance(length_values, list) else [length_values]
        mults = multiplier_values if isinstance(multiplier_values, list) else [multiplier_values]

        params = {
            "length": tuple(int(le) for le in lengths),
            "mult": tuple(float(m) for m in mults),
        }
        cached = self._get_persistent("supertrend", params)
        if cached is not None:
            return cached

        if len(lengths) == 1 and len(mults) == 1:
            st_ind = vbt.IndicatorFactory.from_pandas_ta("supertrend").run(
                self.high,
                self.low,
                self.close,
                length=int(lengths[0]),
                multiplier=float(mults[0]),
            )
            return self._save_persistent("supertrend", params, st_ind)

        td_stack: list[np.ndarray] = []
        for le_val in lengths:
            for m in mults:
                st_i = vbt.IndicatorFactory.from_pandas_ta("supertrend").run(
                    self.high,
                    self.low,
                    self.close,
                    length=int(le_val),
                    multiplier=float(m),
                )
                td = st_i.supertd.values if hasattr(st_i.supertd, "values") else st_i.supertd
                td_stack.append(np.asarray(td, dtype=np.float64))
        td_3d = np.stack(td_stack, axis=1)
        wrapped = SimpleNamespace(supertd=td_3d)
        return self._save_persistent("supertrend", params, wrapped)

    def compute_stochrsi(
        self,
        rsi_length_values: list[int] | int = 14,
        stoch_length_values: list[int] | int = 14,
        k_smooth: int = 3,
    ) -> Any:
        """Pre-compute Stochastic RSI K-line across parameter ranges.

        Uses per-symbol pandas_ta calls (same pattern as compute_adx) because
        VBT's IndicatorFactory has issues with multi-output multi-param indicators.
        """
        import pandas_ta as ta

        rsi_lengths = (
            rsi_length_values if isinstance(rsi_length_values, list) else [rsi_length_values]
        )
        stoch_lengths = (
            stoch_length_values if isinstance(stoch_length_values, list) else [stoch_length_values]
        )
        k_sm = int(k_smooth)

        params = {
            "rsi_length": tuple(int(x) for x in rsi_lengths),
            "stoch_length": tuple(int(x) for x in stoch_lengths),
            "k_smooth": k_sm,
        }
        cached = self._get_persistent("stochrsi", params)
        if cached is not None:
            return cached

        n_time = self.close.shape[0]
        n_symbols = self.close.shape[1] if self.close.ndim == 2 else 1

        k_cols: list[np.ndarray] = []
        for rsi_len in rsi_lengths:
            for stoch_len in stoch_lengths:
                for sym_idx in range(n_symbols):
                    c = self.close[:, sym_idx] if self.close.ndim == 2 else self.close
                    try:
                        res = ta.stochrsi(
                            pd.Series(c),
                            length=int(rsi_len),
                            rsi_length=int(rsi_len),
                            stoch_length=int(stoch_len),
                            k=k_sm,
                        )
                        if res is not None and res.shape[1] >= 1:
                            k_cols.append(res.iloc[:, 0].values)
                        else:
                            raise ValueError("Unexpected StochRSI shape")
                    except Exception:
                        k_cols.append(np.full(n_time, np.nan))

        k_arr = np.column_stack(k_cols) if k_cols else np.full((n_time, 0), np.nan)
        stochrsi_ind = SimpleNamespace(stochrsi_k=k_arr)
        return self._save_persistent("stochrsi", params, stochrsi_ind)

    def compute_kc(
        self,
        length_values: list[int] | int,
        scalar: float = 1.5,
    ) -> Any:
        """Pre-compute Keltner Channels across lengths for a given scalar (multiplier).

        Called once per multiplier value from the strategy class (same pattern as
        compute_bbands per-std). Uses per-length loops like compute_donchian.
        """
        import pandas_ta as ta

        lengths = length_values if isinstance(length_values, list) else [length_values]
        scalar_f = float(scalar)

        params = {"length": tuple(int(le) for le in lengths), "scalar": scalar_f}
        cached = self._get_persistent("kc", params)
        if cached is not None:
            return cached

        n_time = self.close.shape[0]
        n_symbols = self.close.shape[1] if self.close.ndim == 2 else 1

        kcu_stack: list[np.ndarray] = []
        kcl_stack: list[np.ndarray] = []
        for le_val in lengths:
            le_i = int(le_val)
            u_cols: list[np.ndarray] = []
            l_cols: list[np.ndarray] = []
            for sym_idx in range(n_symbols):
                h = self.high[:, sym_idx] if self.high.ndim == 2 else self.high
                lo = self.low[:, sym_idx] if self.low.ndim == 2 else self.low
                c = self.close[:, sym_idx] if self.close.ndim == 2 else self.close
                try:
                    res = ta.kc(
                        pd.Series(h),
                        pd.Series(lo),
                        pd.Series(c),
                        length=le_i,
                        scalar=scalar_f,
                    )
                    if res is not None and res.shape[1] >= 3:
                        # KC returns: KCLe (lower), KCBe (basis), KCUe (upper)
                        l_cols.append(res.iloc[:, 0].values)
                        u_cols.append(res.iloc[:, 2].values)
                    else:
                        raise ValueError("Unexpected KC shape")
                except Exception:
                    nan_arr = np.full(n_time, np.nan)
                    l_cols.append(nan_arr)
                    u_cols.append(nan_arr)
            kcu_stack.append(np.column_stack(u_cols))
            kcl_stack.append(np.column_stack(l_cols))

        kcu_3d = np.stack(kcu_stack, axis=1)
        kcl_3d = np.stack(kcl_stack, axis=1)
        wrapped = SimpleNamespace(kcu=kcu_3d, kcl=kcl_3d)
        return self._save_persistent("kc", params, wrapped)

    def clear_cache(self) -> None:
        """Clear the indicator cache to free memory."""
        self._cache.clear()

    def get_cache_info(self) -> dict[str, int]:
        """Return cache statistics."""
        return {"cached_indicators": len(self._cache)}
