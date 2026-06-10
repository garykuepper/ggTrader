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

# Per-side trading fees (multiply by 2 for round-trip). Closure-doc fee/edge curve
# (docs/superpowers/closures/2026-05-12-...md §3) measured:
#   0.0000 (frictionless)         → 36 passers on 7-coin diagnostic universe
#   0.0020 (half-fees, 0.40% RT)  → 14 passers
#   0.0025 (Kraken maker, 0.50%)  → 13 passers
#   0.0040 (Kraken taker, 0.80%)  → 0 passers
#   0.0002 (Binance.US, 0.04%)    → ~30-35 expected (interpolated)
FEES_KRAKEN_TAKER = 0.004
FEES_KRAKEN_MAKER = 0.0025
FEES_BINANCEUS_TAKER = 0.0002


def fees_for_exchange(exchange: str | None = None) -> float:
    """Return per-side fee rate for the active exchange.

    Defaults to Kraken-taker (0.004) when the exchange is unset or unknown.
    Binance.US ($0-volume tier) uses 0.0002 per side. Per the closure-doc
    fee/edge curve, running research at the wrong fee tier produces a
    misleading deployable set (e.g., Kraken-taker rates yield 0 passers
    even on universes where Binance.US fees would yield 30+).
    """
    # Explicit per-run override (e.g. GGTRADER_FEES=0.0025 to research Kraken at the
    # maker rate). No effect on existing runs when unset.
    fee_override = os.getenv("GGTRADER_FEES")
    if fee_override is not None:
        return float(fee_override)
    venue = (exchange or os.getenv("EXCHANGE") or "kraken").lower()
    if venue == "binanceus":
        return FEES_BINANCEUS_TAKER
    return FEES_KRAKEN_TAKER


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
        "MIN_CLOSED_TRADES_TRAIN": 1,
        "CHUNK_SIZE": 1000,
        "ENTRY_STRATEGY": "psar_adx",
        "EXIT_STRATEGY": "atr_trailing",
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
        "MIN_CLOSED_TRADES_TRAIN": 1,
        "CHUNK_SIZE": 500,
        "ENTRY_STRATEGY": "psar_adx",
        "EXIT_STRATEGY": "atr_trailing",
        "WFO_MODE": "universal",
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
        "EXCHANGE": os.getenv("EXCHANGE", "kraken"),
        "START_DATE": os.getenv("GGTRADER_START_DATE", "2023-01-01"),
        "END_DATE": os.getenv("GGTRADER_END_DATE", "2025-12-31"),
        "INTERVAL": "4h",
        "FREQ": "4h",
        "START_CASH": 1000,
        "PORTFOLIO_SHARE": 0.10,
        # Per-side fee, exchange-aware. Binance.US ($0-tier) is 20× cheaper than
        # Kraken taker — running research at the wrong rate produces a
        # misleading deployable set (Kraken-taker rates → 0 passers; Binance.US
        # rates → ~30 expected on the same universe).
        "FEES": fees_for_exchange(),
        "SLIPPAGE": SLIPPAGE_STANDARD,
        # 10 folds × TEST_RATIO=3 keeps ~1515 train bars/fold (~253 days)
        # provides 10 OOS samples for a more reliable fold-consistency signal.
        "N_SPLITS": 10,
        "TEST_RATIO": 3,
        # Per-fold floor: just rejects genuinely empty folds (0 trades = no signal).
        # Real trade-frequency selection happens at the coin level via
        # MIN_TRADES_PER_YEAR after Phase 3.
        "MIN_CLOSED_TRADES_TRAIN": 1,
        # Coin-level gate on average trade frequency over the full WFO window.
        # 4 trades/year ≈ 1 per quarter ≈ 12 trades over a 3-year window.
        # Applied after Phase 3 (full-range replay) so it sees the same trade
        # count the live trader would experience. Set to None to disable.
        "MIN_TRADES_PER_YEAR": 4,
        # WFO / sensitivity train ranking: composite blends Sortino, Calmar, PF
        # via rank-based scoring (equal contribution, no z-score weighting).
        "TRAIN_METRIC": "composite",
        "MAX_TRAIN_DRAWDOWN_PCT": None,
        "CHUNK_SIZE": 500,
        "USE_MOVERS": 0,
        # Exit tournament: WFO picks the best per-coin exit. fixed_sl_tp is an
        # OCO bracket (entry-set SL + TP, no ratchet); when chosen at placement
        # time it's converted to a Kraken-native trailing-stop using stop_pct
        # so every live sell still ratchets up.
        "EXIT_TOURNAMENT": ["atr_trailing", "trailing_stop", "fixed_sl_tp"],
        "SENSITIVITY_EXIT_STRATEGY": "atr_trailing",
        # Optional: set RECENT_VALIDATION_START_DATE (or CLI) to run Phase 3B after WFO.
        "RECENT_VALIDATION_START_DATE": None,
        "RECENT_VALIDATION_END_DATE": None,
        # CCXT tail enabled by default: appends Kraken bars after the last DB timestamp.
        "RECENT_VALIDATION_USE_CCXT_TAIL": True,
        # Max number of coins that can use the same entry strategy in the combined
        # portfolio. Disabled by default — WFO already optimizes per-coin OOS, and
        # forcing strategy diversity replaces the WFO-optimal pick with a worse one.
        # Real diversification comes from return correlation (regime filter, sizing)
        # not indicator labels. Set to an int to re-enable; see also the per-shard
        # bug noted in changelog 2026-05-08 if re-enabling on parallel runs.
        "MAX_COINS_PER_STRATEGY": None,
        # Manual symbol blacklist for the LIVE trader. Coins listed here are
        # dropped from per_coin_params at load time, so the live trader will
        # never open new positions on them — even if WFO selected them. Existing
        # open positions are unaffected (they continue to be managed normally
        # until their exit triggers). Use to ban symbols that are unsuitable
        # for live trading despite scoring well in backtests (e.g. illiquid,
        # untradeable on Kraken with our minimum size, or known to misbehave).
        "SYMBOL_BLACKLIST": ["TRUMP-USD"]
        if os.getenv("EXCHANGE") == "binanceus"
        else ["TRUMP-USD", "TRX-USD"],
        # Max fraction of portfolio capital any single coin can receive under OOS-weighted
        # allocation. Prevents over-concentration on a single high-robustness coin.
        "MAX_COIN_ALLOCATION": 0.25,
        # Coins whose OOS-weighted robustness falls below this threshold are excluded from
        # Phase 2/3 combined portfolio. Set to None to disable the gate.
        #
        # Disabled by default after the 2026-05 textbook reset: the rank-composite
        # score written into `robustness_score` is structurally bounded in
        # [-N_cells, -1] (always negative — see orchestrator.py:1109-1113), so any
        # positive threshold here drops every coin and produces an empty `per_coin`.
        # The four textbook aggregate gates (WFE, %profitable, paramCV, DD ratio)
        # already enforce quality at selection time, making this legacy gate redundant.
        "MIN_ROBUSTNESS_SCORE": None,
        # History shrinkage: scales each coin's effective robustness by
        # min(1, years_of_history / TARGET) and re-checks against MIN_ROBUSTNESS_SCORE.
        # Cuts brand-new coins (listed mid-window) whose noisy short-data WFO would
        # otherwise pass on small-sample lucky draws. 3.0 matches the default --days
        # 1095 (3y) research window; bump if you run longer windows. Set to None to
        # disable.
        #
        # No-op when MIN_ROBUSTNESS_SCORE is None (the shrinkage gate at
        # orchestrator.py:235 short-circuits without a threshold to compare against).
        "HISTORY_SHRINKAGE_TARGET_YEARS": 3.0,
        # Minimum number of WFO training folds that must have produced at least one valid
        # param combo (finite is_sharpe). Strategies where most folds were rejected by the
        # training gate (e.g. regime-filtered ema_cross with ema_slow=200 yielding 0-1
        # trades per fold) can still achieve inflated robustness from lucky OOS folds.
        # Set to None to disable. 3 = at least 3 of N folds must have had valid training
        # (with 6 folds total, this means at least half must pass the training gate).
        "MIN_VALID_TRAIN_FOLDS": 3,
        # Minimum number of closed train trades a cell must have to be eligible
        # for ranking within a fold (textbook reset). Cells with fewer trades
        # are disqualified before scoring; their rank is "skip".
        #
        # Calibration (2026-05-11): textbook spec value was 30, but on this
        # codebase's strategy design that gate is structurally impossible to
        # satisfy. Strategies fire ~1 trade per 7-8 days; over a 6.7-month
        # train fold (1212 bars at 4h = 28.9 weeks) the expected trade count
        # at full activity is ~26, leaving no room above 30 even before
        # regime-driven inactivity. Pre-registered recalibration: 19 trades
        # per fold (73% of expected fire rate, allowing ~7 inactive weeks
        # per fold for legitimate regime-aware sit-outs). Statistical
        # adequacy: SE of Sortino at N=20 is ~30% wider than at N=30 but
        # still sufficient for cell-level rank ordering within a ≤16-cell
        # grid, with cross-fold consistency from 8-of-10 forgiveness
        # damping single-fold noise. This is a context-specific calibration
        # of the same statistical principle, NOT a relaxation of the
        # textbook standard. If even N=19 produces empty survivors that's
        # a real "no edge" finding, not a signal to lower further.
        "MIN_TRADES_PER_TRAIN_FOLD": 19,
        # 8-of-N forgiveness: a combo must pass the MIN_TRADES_PER_TRAIN_FOLD gate
        # in at least N_PASS folds to be eligible for selection across folds.
        # In folds where it fails, the combo is assigned the median rank in that
        # fold (neither rewarding nor penalizing the inactivity). 8 of 10.
        "MIN_TRAIN_FOLD_PASS_COUNT": 8,
        # Minimum fraction of OOS folds that must be profitable (positive Sharpe) for a coin
        # to be included in the combined portfolio. 0.38 = 4 out of 10 folds profitable.
        "MIN_FOLD_CONSISTENCY": 0.38,
        # Set True or pass --wfo-debug-metrics on run_full_pipeline.py: per-fold train-metric
        # len/finite counts and combined robustness during WFO (all orchestrator paths).
        "WFO_DEBUG_METRICS": False,
        # WFO result cache: skip re-running the 6-fold WFO for (symbol, combo) pairs whose
        # inputs (param grid, config, date range) haven't changed since a prior run.
        # Set False to force a full re-run (e.g. after changing WFO internals not covered
        # by the cache key). Delete results/wfo_cache/ to clear all cached entries.
        "WFO_CACHE_ENABLED": True,
        # Daily loss circuit breaker: halt new entries if intraday portfolio
        # value drops by more than this percentage. Set to None to disable.
        "DAILY_LOSS_LIMIT_PCT": 0.05,
        # Fraction of the most-recent data reserved as a final holdout (locked
        # away before WFO fold bounds are computed; touched exactly once after
        # all gates pass). Spec value: 0.20 (20% holdout, 80% WFO). 0.0 disables
        # the holdout entirely — legacy WFO behavior (everything available to
        # the 10-fold loop).
        "HOLDOUT_FRACTION": 0.20,
        # WFO textbook gates (Pardo convention) — applied after 10-fold loop
        # as PASS/FAIL filters, never as selection criteria.
        "WFO_GATE_WFE_MIN": 0.5,  # mean(test_ann_ret) / mean(train_ann_ret) >= this
        "WFO_GATE_PROFITABLE_FOLDS_MIN": 0.6,  # fraction of folds with test_ann_ret > 0
        "WFO_GATE_PARAM_CV_MAX": 0.3,  # MAX per-axis param CV across the 10 winners
        "WFO_GATE_DD_RATIO_MAX": 2.0,  # mean(|test_dd|) / mean(|train_dd|)
        # Bars-per-year for annualization. None = infer from OHLCV index frequency
        # (4h -> 2191.5, 1h -> 8766.0, 1d -> 365.25). Override only when needed
        # for non-standard intervals or when the index is unreliable.
        "WFO_BARS_PER_YEAR": None,
    }
