import os
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
        "MAX_SYMBOLS": 25,
        "START_DATE": os.getenv("GGTRADER_START_DATE", "2023-01-01"),
        "END_DATE": os.getenv("GGTRADER_END_DATE", "2025-12-31"),
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
        "MIN_CLOSED_TRADES_TRAIN": 3,
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
        "EXIT_TOURNAMENT": ["atr_trailing", "fixed_sl_tp", "trailing_stop"],
        "SENSITIVITY_EXIT_STRATEGY": "atr_trailing",
        # Optional: set RECENT_VALIDATION_START_DATE (or CLI) to run Phase 3B after WFO.
        "RECENT_VALIDATION_START_DATE": None,
        "RECENT_VALIDATION_END_DATE": None,
        # CCXT tail enabled by default: appends Kraken bars after the last DB timestamp.
        "RECENT_VALIDATION_USE_CCXT_TAIL": True,
        # Max number of coins that can use the same entry strategy in the combined portfolio.
        # Prevents a single strategy from dominating and creating correlated drawdowns.
        # Set to None to disable. 10 = no single strategy gets more than 10 coins.
        "MAX_COINS_PER_STRATEGY": 10,
        # Number of warmup bars fetched before START_DATE when computing the BTC EMA.
        # Ensures the EMA is fully warm from bar 1 of the actual backtest window.
        "EMA_WARMUP_BARS": 200,
        # Block new long entries on all coins when BTC close is below its 200-bar EMA.
        # Prevents catching falling knives in sustained crypto bear markets.
        # Applied in Phase 2/3 combined backtest only — WFO fold optimization is unaffected.
        "BTC_REGIME_FILTER": True,
        # Max fraction of portfolio capital any single coin can receive under OOS-weighted
        # allocation. Prevents over-concentration on a single high-robustness coin.
        "MAX_COIN_ALLOCATION": 0.25,
        # Coins whose OOS-weighted robustness falls below this threshold are excluded from
        # Phase 2/3 combined portfolio. Set to None to disable the gate.
        "MIN_ROBUSTNESS_SCORE": 0.1,
        # Minimum fraction of OOS folds that must be profitable (positive Sharpe) for a coin
        # to be included in the combined portfolio. 0.33 = at least 1 in 3 folds profitable.
        "MIN_FOLD_CONSISTENCY": 0.33,
        # Set True or pass --wfo-debug-metrics on run_full_pipeline.py: per-fold train-metric
        # len/finite counts and combined robustness during WFO (all orchestrator paths).
        "WFO_DEBUG_METRICS": False,
        # OOS robustness blend: 0.0 = pure IS robustness (original behaviour),
        # 1.0 = pure OOS Sharpe gate, 0.65 = weight OOS more than IS.
        "OOS_ROBUSTNESS_BLEND_ALPHA": 0.65,
        # --- Anti-overfitting scoring improvements (zero extra compute) ---
        # Z-score normalize composite metric components before blending so that
        # Calmar, ProfitFactor, Sharpe and Sortino are all on the same scale.
        # Set False to restore the original clipped-value blend.
        "TRAIN_METRIC_NORMALIZE_ZSCORE": True,
        # CV-based fold stability penalty: penalizes param combos whose IS metric
        # varies heavily across folds (sign of curve-fitting). 0.0 = disabled,
        # 0.3 = default (moderate), 1.0 = aggressive penalty.
        "PARAM_STABILITY_WEIGHT": 0.3,
        # Apply fold_consistency (fraction of folds with positive OOS Sharpe) as a
        # soft multiplier on gate_score. Set False to disable.
        "FOLD_CONSISTENCY_IN_GATE": True,
        # Floor for the fold_consistency multiplier. 0.25 means a strategy that is
        # never profitable OOS still keeps 25% of its gate score (harder gate).
        "FOLD_CONSISTENCY_GATE_FLOOR": 0.25,
        # Blend weight for the OOS Sharpe-of-Sharpes stability term. Tempers a
        # single outlier fold from inflating the OOS robustness score.
        # 0.0 = disabled (pure weighted mean), 0.3 = default.
        "OOS_STABILITY_WEIGHT": 0.3,
    }
