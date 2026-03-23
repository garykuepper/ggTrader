"""Shared default configuration dicts for CLI scripts and orchestrators."""

from __future__ import annotations

from typing import Any

# --- Common date / portfolio keys (many scripts share these) ---
SHARED_DATES_AND_PORTFOLIO: dict[str, Any] = {
    "START_DATE": "2023-01-01",
    "END_DATE": "2025-12-31",
    "INTERVAL": "4h",
    "START_CASH": 1000,
    "PORTFOLIO_SHARE": 0.10,
    "FEES": 0.004,
}

SLIPPAGE_TIGHT = 0.001
SLIPPAGE_STANDARD = 0.003

# Default PSAR+ADX parameters (single-strategy runs)
DEFAULT_PSAR_ADX_PARAMS: dict[str, Any] = {
    "adx_threshold": 25,
    "adx_length": 14,
    "sar_acceleration": 0.02,
    "sar_maximum": 0.2,
    "atr_multiplier": 3.0,
    "atr_length": 14,
    "use_dmp_cross": False,
}


def merge_run_config(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a shallow copy of ``base`` with non-None overrides applied."""
    out = dict(base)
    for key, value in overrides.items():
        if value is not None:
            out[key] = value
    return out


def backtest_script_config() -> dict[str, Any]:
    """Defaults for ``scripts/run_backtest.py`` (orchestrator ``config`` only)."""
    return {
        **SHARED_DATES_AND_PORTFOLIO,
        "SLIPPAGE": SLIPPAGE_STANDARD,
        "SYMBOLS": None,
        "SYMBOLS_FILE": "data/top_25_USD_2023-01-01_2025-12-31.json",
        "USE_MOVERS": 10,
    }


def sensitivity_script_config() -> dict[str, Any]:
    """Defaults for ``scripts/run_sensitivity_analysis.py``."""
    return {
        **SHARED_DATES_AND_PORTFOLIO,
        "SLIPPAGE": SLIPPAGE_TIGHT,
        "SYMBOLS": None,
        "SYMBOLS_FILE": "data/top_10_USD_2023-01-01_2025-12-31.json",
        "USE_MOVERS": 0,
        "MIN_TRADES": 0,
        "MIN_CLOSED_TRADES_TRAIN": 1,
        "CHUNK_SIZE": 1000,
        "ENTRY_STRATEGY": "psar_adx",
        "EXIT_STRATEGY": "atr_trailing",
        "USE_VECTORIZED": False,
    }


def wfo_script_config() -> dict[str, Any]:
    """Defaults for ``scripts/run_walk_forward_optimization.py``."""
    return {
        **SHARED_DATES_AND_PORTFOLIO,
        "SLIPPAGE": SLIPPAGE_STANDARD,
        "SYMBOLS": None,
        "SYMBOLS_FILE": "data/top_25_USD_2023-01-01_2025-12-31.json",
        "USE_MOVERS": 0,
        "N_SPLITS": 4,
        "TEST_RATIO": 2,
        "MIN_TRADES": 0,
        "MIN_CLOSED_TRADES_TRAIN": 1,
        "CHUNK_SIZE": 500,
        "ENTRY_STRATEGY": "psar_adx",
        "EXIT_STRATEGY": "atr_trailing",
        "WFO_MODE": "universal",
        "USE_VECTORIZED": False,
    }


def strategy_comparison_config() -> dict[str, Any]:
    """Defaults for ``scripts/run_strategy_comparison.py``."""
    return {
        **SHARED_DATES_AND_PORTFOLIO,
        "SLIPPAGE": SLIPPAGE_TIGHT,
        "SYMBOLS": None,
        "SYMBOLS_FILE": "data/top_10_USD_2023-01-01_2025-12-31.json",
        "USE_MOVERS": 0,
    }


def full_pipeline_config() -> dict[str, Any]:
    """Defaults for ``scripts/run_full_pipeline.py``."""
    return {
        "SYMBOLS_FILE": "data/top_25_USD_2023-01-01_2025-12-31.json",
        "MAX_SYMBOLS": 5,
        "START_DATE": "2023-01-01",
        "END_DATE": "2025-12-31",
        "INTERVAL": "4h",
        "FREQ": "4h",
        "START_CASH": 1000,
        "PORTFOLIO_SHARE": 0.10,
        "FEES": 0.004,
        "SLIPPAGE": SLIPPAGE_STANDARD,
        # 6 folds × TEST_RATIO=3 keeps ~2190 train bars/fold (same as 4×2) while adding
        # 2 more OOS samples for a more reliable fold-consistency signal in the gate.
        "N_SPLITS": 6,
        "TEST_RATIO": 3,
        "MIN_TRADES": 0,
        "MIN_CLOSED_TRADES_TRAIN": 1,
        # WFO / sensitivity train ranking: composite blends Sharpe, Sortino, Calmar-like (return/|maxDD|).
        "TRAIN_METRIC": "composite",
        "TRAIN_METRIC_COMPOSITE_WEIGHTS": {
            "sharpe": 0.25,
            "sortino": 0.25,
            "calmar": 0.25,
            "profit_factor": 0.25,
        },
        "MAX_TRAIN_DRAWDOWN_PCT": None,
        "CHUNK_SIZE": 500,
        "USE_VECTORIZED": True,
        "USE_VECTORIZED_SENSITIVITY": True,
        "USE_MOVERS": 0,
        # Both exits compete in the tournament; use --exits atr_trailing to limit to one.
        "EXIT_TOURNAMENT": ["atr_trailing", "fixed_sl_tp"],
        "SENSITIVITY_EXIT_STRATEGY": "atr_trailing",
        # Optional: set RECENT_VALIDATION_START_DATE (or CLI) to run Phase 3B after WFO.
        "RECENT_VALIDATION_START_DATE": None,
        "RECENT_VALIDATION_END_DATE": None,
        # CCXT tail enabled by default: appends Kraken bars after the last DB timestamp.
        "RECENT_VALIDATION_USE_CCXT_TAIL": True,
        # Coins whose OOS-weighted robustness falls below this threshold are excluded from
        # Phase 2/3 combined portfolio. Set to None to disable the gate.
        "MIN_ROBUSTNESS_SCORE": 0.0,
        # Set True or pass --wfo-debug-metrics on run_full_pipeline.py: per-fold train-metric
        # len/finite counts and combined robustness during WFO (all orchestrator paths).
        "WFO_DEBUG_METRICS": False,
        # OOS robustness blend: 0.0 = pure IS robustness (original behaviour),
        # 1.0 = pure OOS Sharpe gate, 0.5 = equal blend (default).
        "OOS_ROBUSTNESS_BLEND_ALPHA": 0.5,
    }
