"""WFO result cache for per-coin strategy tournament runs.

Caches (wfo_stats, is_metrics_by_fold) per (symbol, entry, exit, param_grid, config, data_range)
so repeated research runs with identical inputs skip re-running the 6-fold WFO entirely.

Cache is stored in the TimescaleDB ``wfo_cache`` table — keyed by an MD5 hash of the inputs.
Use ``ggt db purge-wfo-cache`` to clear, or bump ``_WFO_CACHE_VERSION`` for a global invalidation.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text

_WFO_CACHE_VERSION = 3  # bump to invalidate all cached entries globally
# v3 (2026-05-10): WFO textbook reset — removed per-cell OOS metrics
#                  (test_metrics_by_fold), rank-based composite scoring

# Config keys whose values affect WFO fold computation.
# Changing any of these causes a cache miss for that (symbol, combo).
_WFO_RELEVANT_CONFIG_KEYS = [
    "N_SPLITS",
    "TEST_RATIO",
    "MIN_FOLD_BARS",
    "MIN_CLOSED_TRADES_TRAIN",
    "MAX_TRAIN_DRAWDOWN_PCT",
    "REJECT_OPEN_END_IF_CLOSED_LT",
    "TRAIN_METRIC",
    "START_CASH",
    "FEES",
    "SLIPPAGE",
    "INTERVAL",
    "BTC_REGIME_FILTER",
    "BTC_REGIME_FILTER_SHORT_EMA",
    "BTC_REGIME_FILTER_MIN_CORRELATION",
    "ALTCOIN_REGIME_FILTER",
    "ALTCOIN_REGIME_FILTER_CORR_MIN",
    "EMA_WARMUP_BARS",
    "MIN_TRADES_PER_TRAIN_FOLD",
    "MIN_TRAIN_FOLD_PASS_COUNT",
    "WFO_BARS_PER_YEAR",
]


def _hash_param_grid(param_grid: Dict[str, Any]) -> str:
    """Deterministic hash of a param grid."""
    safe = {}
    for k, v in sorted(param_grid.items()):
        safe[k] = sorted(v) if isinstance(v, list) else v
    return hashlib.md5(json.dumps(safe, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _hash_config(config: Dict[str, Any]) -> str:
    """Hash only WFO-relevant config keys."""
    relevant = {k: config.get(k) for k in _WFO_RELEVANT_CONFIG_KEYS}
    return hashlib.md5(
        json.dumps(relevant, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _data_fingerprint(ohlcv: pd.DataFrame) -> Tuple[str, str, str]:
    """Return (first_date, last_date, bar_count) for cache key uniqueness."""
    idx = ohlcv.index
    return str(idx[0])[:10], str(idx[-1])[:10], str(len(idx))


def _make_cache_key(
    symbol: str,
    strategy_name: str,
    exit_name: str,
    param_grid: Dict[str, Any],
    config: Dict[str, Any],
    ohlcv: pd.DataFrame,
) -> str:
    """Build a unique 32-char hex cache key for this (symbol, combo, grid, config, data)."""
    date_start, date_end, bar_count = _data_fingerprint(ohlcv)
    raw = "|".join([
        f"v{_WFO_CACHE_VERSION}",
        symbol,
        strategy_name,
        exit_name,
        _hash_param_grid(param_grid),
        _hash_config(config),
        date_start,
        date_end,
        bar_count,
    ])
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _float_safe(v: Any) -> Any:
    """Convert float to JSON-safe value (None for NaN/Inf)."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _float_restore(v: Any) -> float:
    """Restore None → NaN when deserializing."""
    return float("nan") if v is None else float(v)


def _series_to_json(s: pd.Series) -> Dict:
    """Serialize a pd.Series with scalar or tuple index to a JSON dict."""
    index_out = []
    for idx in s.index:
        if isinstance(idx, tuple):
            index_out.append(list(idx))   # tuple → list
        else:
            index_out.append(idx)         # scalar as-is
    values_out = [_float_safe(float(v)) for v in s.to_numpy(dtype=float)]
    return {"index": index_out, "values": values_out}


def _series_from_json(data: Dict) -> pd.Series:
    """Deserialize a pd.Series from a JSON dict produced by _series_to_json."""
    raw_index = data["index"]
    values = [_float_restore(v) for v in data["values"]]
    # Restore lists → tuples, scalars stay as-is
    restored_index = [tuple(x) if isinstance(x, list) else x for x in raw_index]
    return pd.Series(values, index=pd.Index(restored_index, dtype=object))


_SCALAR_FOLD_KEYS = {
    "oos_sharpe", "oos_return", "profit", "start_capital",
    "end_capital", "return_pct", "is_sharpe", "sortino",
}


def _wfo_stats_to_json(wfo_stats: List[Dict]) -> List[Dict]:
    """Strip non-JSON-serializable fields and encode NaN → None."""
    result = []
    for fold in wfo_stats:
        safe: Dict[str, Any] = {}
        for k, v in fold.items():
            if isinstance(v, (int, str, bool, type(None))):
                safe[k] = v
            elif isinstance(v, float):
                safe[k] = _float_safe(v)
            elif isinstance(v, dict):
                # best_params dict (scalar values)
                safe[k] = {dk: _float_safe(dv) if isinstance(dv, float) else dv
                           for dk, dv in v.items()}
            # Skip DataFrames / Series (already popped before this point)
        result.append(safe)
    return result


def _wfo_stats_from_json(data: List[Dict]) -> List[Dict]:
    """Restore NaN floats for known scalar fold keys."""
    result = []
    for fold in data:
        restored: Dict[str, Any] = {}
        for k, v in fold.items():
            if v is None and k in _SCALAR_FOLD_KEYS:
                restored[k] = float("nan")
            else:
                restored[k] = v
        result.append(restored)
    return result


# ---------------------------------------------------------------------------
# WFOCache class
# ---------------------------------------------------------------------------

class WFOCache:
    """TimescaleDB-backed cache for WFO per-coin tournament (wfo_stats, is_metrics_by_fold)."""

    def __init__(self, cache_dir: Optional[Path] = None, db_manager: Any = None):
        # ``cache_dir`` retained for backwards-compatible call sites; the cache
        # itself now lives in the ``wfo_cache`` table in TimescaleDB.
        if db_manager is None:
            from ggTrader.utils.result_db_manager import ResultDBManager
            db_manager = ResultDBManager()
        self.db_manager = db_manager
        self._hits = 0
        self._misses = 0

    def get(
        self,
        symbol: str,
        strategy_name: str,
        exit_name: str,
        param_grid: Dict[str, Any],
        config: Dict[str, Any],
        ohlcv: pd.DataFrame,
    ) -> Optional[Tuple[List[Dict], Dict[int, pd.Series]]]:
        """Return cached (wfo_stats, is_metrics_by_fold) or None on miss."""
        key = _make_cache_key(symbol, strategy_name, exit_name, param_grid, config, ohlcv)
        try:
            with self.db_manager.engine.connect() as conn:
                row = conn.execute(
                    text("SELECT payload FROM wfo_cache WHERE cache_key = :k"),
                    {"k": key},
                ).fetchone()
        except Exception:
            self._misses += 1
            return None
        if row is None or row[0] is None:
            self._misses += 1
            return None
        try:
            data = row[0]
            wfo_stats = _wfo_stats_from_json(data["wfo_stats"])
            is_metrics_by_fold: Dict[int, pd.Series] = {
                int(k): _series_from_json(v)
                for k, v in data["is_metrics_by_fold"].items()
            }
            self._hits += 1
            return wfo_stats, is_metrics_by_fold
        except Exception:
            # Corrupt entry — drop it and treat as miss
            try:
                with self.db_manager.engine.begin() as conn:
                    conn.execute(
                        text("DELETE FROM wfo_cache WHERE cache_key = :k"),
                        {"k": key},
                    )
            except Exception:
                pass
            self._misses += 1
            return None

    def put(
        self,
        symbol: str,
        strategy_name: str,
        exit_name: str,
        param_grid: Dict[str, Any],
        config: Dict[str, Any],
        ohlcv: pd.DataFrame,
        wfo_stats: List[Dict],
        is_metrics_by_fold: Dict[int, pd.Series],
    ) -> None:
        """Persist WFO results to cache (non-fatal on write errors)."""
        key = _make_cache_key(symbol, strategy_name, exit_name, param_grid, config, ohlcv)
        payload = {
            "version": _WFO_CACHE_VERSION,
            "symbol": symbol,
            "strategy": strategy_name,
            "exit": exit_name,
            "wfo_stats": _wfo_stats_to_json(wfo_stats),
            "is_metrics_by_fold": {
                str(k): _series_to_json(v)
                for k, v in is_metrics_by_fold.items()
            },
        }
        try:
            with self.db_manager.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO wfo_cache (cache_key, symbol, strategy_name,
                                               exit_name, cache_version, payload)
                        VALUES (:k, :sym, :strat, :exit, :ver, CAST(:p AS JSONB))
                        ON CONFLICT (cache_key) DO NOTHING
                        """
                    ),
                    {
                        "k": key,
                        "sym": symbol,
                        "strat": strategy_name,
                        "exit": exit_name,
                        "ver": _WFO_CACHE_VERSION,
                        "p": json.dumps(payload),
                    },
                )
        except Exception:
            pass  # cache write failure is non-fatal

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def summary(self) -> str:
        total = self._hits + self._misses
        if total == 0:
            return "WFO cache: 0 lookups"
        pct = 100.0 * self._hits / total
        return f"WFO cache: {self._hits}/{total} hits ({pct:.0f}%)"
