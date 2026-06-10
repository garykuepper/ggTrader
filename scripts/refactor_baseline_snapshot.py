#!/usr/bin/env python3
"""Refactor baseline snapshot — TEMPORARY (deleted when the vbt refactor lands).

Captures deterministic before/after fingerprints of the core engine on synthetic
data so each refactor phase can prove it didn't change behavior:

  S1  vectorized grid run (psar_adx + atr_trailing, composite train metric)
  S2  scalar run on the legacy SignalFactory path (USE_VECTORIZED=False)
  S3  full per-coin WFO loop + robustness scoring (gates enabled), psar_adx
  S3e same WFO loop with ENTRY_STRATEGY=ema_cross (documents the old-path bug fix)

Usage:
    source .venv/bin/activate && python -u scripts/refactor_baseline_snapshot.py [tag]

Writes scratch/refactor_baseline/<tag>.json (tag defaults to "baseline").
Compare runs with:  diff <(jq -S . a.json) <(jq -S . b.json)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ggTrader.core.fast_backtest import FastBacktest  # noqa: E402
from ggTrader.core.metrics import _train_metric_series  # noqa: E402

try:  # post-refactor replacement for the patched pf.trades.profit_factor()
    from ggTrader.core.metrics import _profit_factor_raw
except ImportError:
    _profit_factor_raw = None
from ggTrader.core.orchestrator_utils import _to_native  # noqa: E402
from ggTrader.core.wfo import (  # noqa: E402
    _calculate_oos_robustness,
    _calculate_robustness,
    _execute_wfo_loop,
)

OUT_DIR = PROJECT_ROOT / "scratch" / "refactor_baseline"


def make_ohlcv(n_bars: int = 600, symbols: tuple[str, ...] = ("AAA", "BBB", "CCC")) -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV with trending + choppy regimes."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-03", periods=n_bars, freq="B", tz="UTC")
    frames = {}
    for i, sym in enumerate(symbols):
        t = np.arange(n_bars)
        drift = 0.0008 * np.sin(2 * np.pi * t / 180 + i) + 0.0003
        noise = rng.normal(0, 0.015, n_bars)
        close = 100.0 * (1 + i) * np.exp(np.cumsum(drift + noise))
        spread = np.abs(rng.normal(0, 0.008, n_bars))
        high = close * (1 + spread)
        low = close * (1 - spread)
        open_ = np.empty(n_bars)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        high = np.maximum.reduce([high, close, open_])
        low = np.minimum.reduce([low, close, open_])
        frames[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": 1e6},
            index=idx,
        )
    out = pd.concat(frames, axis=1)
    out.columns.names = ["symbol", "field"]
    return out


def _hash_bool_frame(df: pd.DataFrame) -> str:
    arr = np.ascontiguousarray(df.values.astype(np.bool_))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _jsonable(obj):
    """Best-effort conversion to JSON-serializable structures; drop what isn't."""
    native = _to_native(obj)
    try:
        json.dumps(native)
        return native
    except (TypeError, ValueError):
        if isinstance(native, dict):
            return {k: _jsonable(v) for k, v in native.items() if _is_jsonable(v)}
        return repr(native)


def _is_jsonable(v) -> bool:
    try:
        json.dumps(_to_native(v))
        return True
    except (TypeError, ValueError):
        return False


def _series_record(s: pd.Series, ndigits: int = 10) -> list:
    vals = np.asarray(s, dtype=float).ravel()
    return [None if not np.isfinite(v) else round(float(v), ndigits) for v in vals]


GRID = {
    "sar_acceleration": [0.02, 0.04],
    "sar_maximum": [0.2, 0.3],
    "adx_length": [14],
    "adx_threshold": [20, 25, 30],
    "use_dmp_cross": [True, False],
    "atr_length": [14],
    "atr_multiplier": [2.0, 3.0],
}

BASE_CONFIG = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.0,
    "SLIPPAGE": 0.0005,
    "FREQ": "1d",
    "USE_CASH_SHARING": True,
    "ENTRY_STRATEGY": "psar_adx",
    "EXIT_STRATEGY": "atr_trailing",
    "TRAIN_METRIC": "composite",
}


def s1_grid_vectorized(ohlcv: pd.DataFrame) -> dict:
    config = {**BASE_CONFIG, "USE_VECTORIZED": True}
    engine = FastBacktest(ohlcv, GRID, config=config)
    pf = engine.run(show_progress=False)
    train_metric = _train_metric_series(pf, config)
    if _profit_factor_raw is not None:
        pf_raw = _profit_factor_raw(pf)
    else:
        pf_raw = pf.trades.profit_factor()
    if not isinstance(pf_raw, pd.Series):
        pf_raw = pd.Series([float(pf_raw)])
    return {
        "stats": _jsonable(engine.get_stats()),
        "entries_hash": _hash_bool_frame(engine.entries),
        "exits_hash": _hash_bool_frame(engine.exits),
        "n_signal_cols": int(engine.entries.shape[1]),
        "n_param_combos": len(engine._last_param_combos or []),
        "train_metric_composite": _series_record(train_metric),
        "profit_factor": _series_record(pf_raw),
    }


def s2_scalar_legacy(ohlcv: pd.DataFrame) -> dict:
    config = {**BASE_CONFIG, "USE_VECTORIZED": False}
    params = {
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "adx_length": 14,
        "adx_threshold": 25,
        "use_dmp_cross": True,
        "atr_length": 14,
        "atr_multiplier": 3.0,
    }
    engine = FastBacktest(ohlcv, params, config=config)
    engine.run(show_progress=False)
    return {
        "stats": _jsonable(engine.get_stats()),
        "entries_hash": _hash_bool_frame(engine.entries),
        "exits_hash": _hash_bool_frame(engine.exits),
        "n_signal_cols": int(engine.entries.shape[1]),
    }


def s3_wfo(ohlcv: pd.DataFrame, entry_strategy: str) -> dict:
    sym_ohlcv = ohlcv[["AAA"]]
    config = {
        **BASE_CONFIG,
        "USE_VECTORIZED": True,
        "USE_CASH_SHARING": False,
        "ENTRY_STRATEGY": entry_strategy,
        "MIN_CLOSED_TRADES_TRAIN": 1,
        "MIN_TRADES_PER_TRAIN_FOLD": 2,
        "MAX_TRAIN_DRAWDOWN_PCT": 75,
        "REJECT_OPEN_END_IF_CLOSED_LT": 1,
    }
    if entry_strategy == "psar_adx":
        grid = {
            "sar_acceleration": [0.02, 0.04],
            "sar_maximum": [0.2],
            "adx_length": [14],
            "adx_threshold": [20, 25],
            "use_dmp_cross": [True],
            "atr_length": [14],
            "atr_multiplier": [2.0, 3.0],
        }
    elif entry_strategy == "ema_cross":
        grid = {
            "ema_fast": [9, 12],
            "ema_slow": [50],
            "atr_length": [14],
            "atr_multiplier": [2.0, 3.0],
        }
    else:
        raise ValueError(entry_strategy)

    param_names = list(grid.keys())
    wfo_stats, is_metrics_by_fold, _ = _execute_wfo_loop(
        sym_ohlcv, None, grid, config, param_names, 4, 3.0, False, None
    )

    oos_metrics = {
        i + 1: s.get("oos_sharpe", float("nan"))
        for i, s in enumerate(wfo_stats)
        if not s.get("_skipped_vectorized_failure") and not s.get("_skipped_insufficient_bars")
    }
    oos_bear = {
        i + 1: s.get("oos_is_bear", False)
        for i, s in enumerate(wfo_stats)
        if not s.get("_skipped_vectorized_failure") and not s.get("_skipped_insufficient_bars")
    }
    robust_top_5, best_robust_params = _calculate_robustness(
        is_metrics_by_fold, param_names, grid, debug_metrics=False, config=config
    )
    oos_rob, fold_cons = _calculate_oos_robustness(
        oos_metrics, config=config, oos_bear_by_fold=oos_bear
    )

    folds = []
    for s in wfo_stats:
        folds.append(
            {
                k: _jsonable(v)
                for k, v in s.items()
                if _is_jsonable(v)
                and k
                in (
                    "fold",
                    "params",
                    "best_params",
                    "train_sharpe",
                    "oos_sharpe",
                    "oos_sortino",
                    "oos_return",
                    "oos_is_bear",
                    "profit",
                    "total_trades",
                    "max_drawdown",
                    "_skipped_vectorized_failure",
                    "_skipped_insufficient_bars",
                )
            }
        )
    is_by_fold = {
        str(f): _series_record(s) for f, s in is_metrics_by_fold.items() if isinstance(s, pd.Series)
    }
    return {
        "folds": folds,
        "is_metrics_by_fold": is_by_fold,
        "robust_top_5": _jsonable(robust_top_5),
        "best_robust_params": _jsonable(best_robust_params),
        "oos_robustness": _jsonable(oos_rob),
        "fold_consistency": _jsonable(fold_cons),
    }


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ohlcv = make_ohlcv()

    import vectorbt as vbt

    snapshot = {"vectorbt_version": vbt.__version__, "tag": tag}
    print("S1: vectorized grid run...")
    snapshot["s1_grid_vectorized"] = s1_grid_vectorized(ohlcv)
    print("S2: scalar legacy-path run...")
    snapshot["s2_scalar_legacy"] = s2_scalar_legacy(ohlcv)
    print("S3: WFO psar_adx...")
    snapshot["s3_wfo_psar"] = s3_wfo(ohlcv, "psar_adx")
    print("S3e: WFO ema_cross...")
    snapshot["s3_wfo_ema"] = s3_wfo(ohlcv, "ema_cross")

    out_path = OUT_DIR / f"{tag}.json"
    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    print(f"\nSnapshot written: {out_path}")
    print(f"  S1 trades={snapshot['s1_grid_vectorized']['stats'].get('total_trades')}")
    print(f"  S2 trades={snapshot['s2_scalar_legacy']['stats'].get('total_trades')}")
    print(f"  S3 psar oos_rob={snapshot['s3_wfo_psar']['oos_robustness']}")
    print(f"  S3 ema  oos_rob={snapshot['s3_wfo_ema']['oos_robustness']}")


if __name__ == "__main__":
    main()
