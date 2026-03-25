"""Indicator pre-computation and caching layer for vectorized signal generation."""

import gzip
import hashlib
import os
import pickle
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt


class PersistentIndicatorCache:
    """Manages disk-based caching of technical indicators to speed up repeats."""

    def __init__(self, cache_dir: str = ".cache/indicators"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def get_data_hash(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> str:
        """Create a unique hash of the input price data."""
        # Use first and last few points + total sum for speed, or full bytes for safety.
        # Combined bytes of all 3 arrays ensures we don't mix up different assets.
        combined = b"".join([high.tobytes(), low.tobytes(), close.tobytes()])
        return hashlib.sha256(combined).hexdigest()[:16]

    def get_cache_path(self, data_hash: str, indicator_name: str, params: tuple) -> str:
        """Build a stable file path for the cached indicator result."""
        # Param hash handles the combinatorial grid variations (length, etc).
        param_str = hashlib.md5(str(params).encode()).hexdigest()[:8]
        filename = f"{data_hash}_{indicator_name}_{param_str}.pkl.gz"
        return os.path.join(self.cache_dir, filename)

    def save(self, data: Any, path: str) -> None:
        """Save indicator result to disk using compressed pickle."""
        try:
            with gzip.open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache to {path}: {e}")

    def load(self, path: str) -> Optional[Any]:
        """Load indicator result from disk if it exists."""
        if not os.path.exists(path):
            return None
        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache from {path}: {e}")
            return None


class IndicatorPrecomputer:
    """Computes technical indicators once per parameter range and caches results.

    Eliminates redundant per-combo indicator recalculation by pre-computing
    each indicator (PSAR, ADX, ATR) across all parameter values, then using
    numpy broadcasting to combine them in signal logic.
    """

    def __init__(self, close: np.ndarray, high: np.ndarray, low: np.ndarray):
        """Initialize with OHLCV arrays and setup disk cache."""
        self.close = self._to_ndarray(close)
        self.high = self._to_ndarray(high)
        self.low = self._to_ndarray(low)

        # In-memory cache for the current run
        self._cache: dict[tuple[str, Any], Any] = {}

        # Persistent disk cache
        self._disk_cache = PersistentIndicatorCache()
        self._data_hash = self._disk_cache.get_data_hash(self.high, self.low, self.close)

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
        """Check both in-memory and disk cache."""
        key = self._make_cache_key(name, params)
        if key in self._cache:
            return self._cache[key]

        path = self._disk_cache.get_cache_path(self._data_hash, name, key[1])
        data = self._disk_cache.load(path)
        if data is not None:
            self._cache[key] = data
        return data

    def _save_persistent(self, name: str, params: dict, data: Any) -> None:
        """Save to both in-memory and disk cache."""
        wrapped = self._wrap_indicator(data)
        key = self._make_cache_key(name, params)
        self._cache[key] = wrapped
        path = self._disk_cache.get_cache_path(self._data_hash, name, key[1])
        self._disk_cache.save(wrapped, path)

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
        self._save_persistent("psar", params, psar_ind)
        return psar_ind

    def compute_adx(
        self,
        adx_length_values: list[int] | int = 14,
    ) -> Any:
        """Pre-compute ADX across parameter ranges."""
        lengths = adx_length_values if isinstance(adx_length_values, list) else [adx_length_values]
        params = {"length": tuple(int(l) for l in lengths)}
        cached = self._get_persistent("adx", params)
        if cached is not None:
            return cached

        # Use own custom factory to avoid issues with from_pandas_ta splitting
        # ADX usually returns (ADX, DMP, DMN)
        ADXFactory = vbt.IndicatorFactory(
            class_name="ADX",
            short_name="adx",
            input_names=["high", "low", "close"],
            param_names=["length"],
            output_names=["adx", "dmp", "dmn"],
        )

        def custom_adx(high, low, close, length):
            import pandas_ta as ta

            h_s = pd.Series(np.asarray(high).flatten())
            l_s = pd.Series(np.asarray(low).flatten())
            c_s = pd.Series(np.asarray(close).flatten())
            try:
                # length might be a list/array with one element if param_product is used
                le = int(length[0] if isinstance(length, (list, np.ndarray)) else length)
                res = ta.adx(h_s, l_s, c_s, length=le)
                if res is not None and res.shape[1] >= 3:
                    return res.iloc[:, 0].values, res.iloc[:, 1].values, res.iloc[:, 2].values
            except Exception:
                pass
            # Fallback for failing ADX or empty data
            nan_arr = np.full(close.shape, np.nan)
            return nan_arr, nan_arr, nan_arr

        factory = ADXFactory.from_apply_func(custom_adx)

        if len(lengths) == 1:
            adx_ind = factory.run(self.high, self.low, self.close, length=int(lengths[0]))
        else:
            adx_ind = factory.run(
                self.high,
                self.low,
                self.close,
                length=[int(l) for l in lengths],
                param_product=True,
            )
        self._save_persistent("adx", params, adx_ind)
        return adx_ind

    def compute_atr(
        self,
        atr_length_values: list[int] | int = 14,
    ) -> Any:
        """Pre-compute ATR across parameter ranges."""
        lengths = atr_length_values if isinstance(atr_length_values, list) else [atr_length_values]

        params = {"length": tuple(int(l) for l in lengths)}
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
                length=[int(l) for l in lengths],
                param_product=True,
            )
        self._save_persistent("atr", params, atr_ind)
        return atr_ind

    def compute_ema(
        self,
        ema_length_values: list[int] | int,
    ) -> Any:
        """Pre-compute EMA across parameter ranges."""
        lengths = ema_length_values if isinstance(ema_length_values, list) else [ema_length_values]

        params = {"length": tuple(int(l) for l in lengths)}
        cached = self._get_persistent("ema", params)
        if cached is not None:
            return cached

        if len(lengths) == 1:
            ema_ind = vbt.IndicatorFactory.from_pandas_ta("ema").run(
                self.close, length=int(lengths[0])
            )
        else:
            ema_ind = vbt.IndicatorFactory.from_pandas_ta("ema").run(
                self.close, length=[int(l) for l in lengths], param_product=True
            )
        self._save_persistent("ema", params, ema_ind)
        return ema_ind

    def compute_rsi(
        self,
        rsi_length_values: list[int] | int = 14,
    ) -> Any:
        """Pre-compute RSI across parameter ranges."""
        lengths = rsi_length_values if isinstance(rsi_length_values, list) else [rsi_length_values]

        params = {"length": tuple(int(l) for l in lengths)}
        cached = self._get_persistent("rsi", params)
        if cached is not None:
            return cached

        if len(lengths) == 1:
            rsi_ind = vbt.IndicatorFactory.from_pandas_ta("rsi").run(
                self.close, length=int(lengths[0])
            )
        else:
            rsi_ind = vbt.IndicatorFactory.from_pandas_ta("rsi").run(
                self.close, length=[int(l) for l in lengths], param_product=True
            )
        self._save_persistent("rsi", params, rsi_ind)
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
        self._save_persistent("macd", params, macd_ind)
        return macd_ind

    def compute_bbands(
        self,
        length_values: list[int] | int,
        std: float = 2.0,
    ) -> Any:
        """Pre-compute Bollinger bands across lengths."""
        lengths = length_values if isinstance(length_values, list) else [length_values]
        std_f = float(std)

        params = {"length": tuple(int(l) for l in lengths), "std": std_f}
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
                length=[int(l) for l in lengths],
                std=std_f,
                param_product=True,
            )
        self._save_persistent("bbands", params, bb_ind)
        return bb_ind

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
            self._save_persistent("donchian", params, dc_ind)
            return dc_ind

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
        self._save_persistent("donchian", params, wrapped)
        return wrapped

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
            self._save_persistent("supertrend", params, st_ind)
            return st_ind

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
        self._save_persistent("supertrend", params, wrapped)
        return wrapped

    def clear_cache(self) -> None:
        """Clear the indicator cache to free memory."""
        self._cache.clear()

    def get_cache_info(self) -> dict[str, int]:
        """Return cache statistics."""
        return {"cached_indicators": len(self._cache)}
