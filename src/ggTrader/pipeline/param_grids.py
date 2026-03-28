"""Walk-forward and sensitivity parameter grids (entry axes + exit axes)."""

from __future__ import annotations

from typing import Any

# --- Discovery (Iteration 6) Expanded Grids ---

DETAILED_ENTRY_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    "psar_adx": {
        # Pruned based on 261 WFO selections across 13 runs (empirical, not domain-only):
        # sar_acceleration: 0.02 selected 100% of the time — collapse to constant.
        # sar_maximum: 0.1 selected 100% of the time — collapse to constant.
        # adx_length: 10 selected only 2.7% — dropped; 30 dominates (71%).
        # use_dmp_cross: False selected 86%, True 14% — dropped True (treated as noise).
        "sar_acceleration": [0.02],
        "sar_maximum": [0.1],
        "adx_length": [14, 20, 30],
        "adx_threshold": [20, 25, 30, 35],
        "use_dmp_cross": [False],
        # Total: 1×1×3×4×1 = 12 combos (was 384, -97%)
    },
    "ema_cross": {
        # Restructured so ALL fast/slow pairs are valid (fast < slow guaranteed).
        # fast=50 and slow=[8,13] removed — they generated 9 invalid pairs that
        # wasted ~21% of ema_cross compute producing 0-trade results.
        "ema_fast": [3, 5, 9, 12, 20],
        "ema_slow": [21, 34, 50, 100, 200],
        # Total: 25 valid combos (unchanged)
    },
    "rsi_reversal": {
        "rsi_length": [5, 7, 10, 14, 21, 28],
        "rsi_oversold": [15, 20, 25, 30, 35, 40],
        "rsi_trend_filter": [False, True],
        # Total: 72 combos (unchanged — all values have meaningful selection rates)
    },
    "donchian_breakout": {
        # donchian_length: 100 selected 100% of the time across all runs — collapse to constant.
        "donchian_length": [100],
        # Total: 1 combo (was 7, -86%)
    },
    "macd_cross": {
        # macd_fast=5: 0 selections — dropped.
        # macd_slow=21: 0 selections — dropped.
        # macd_signal=9: 0 selections — dropped.
        "macd_fast": [8, 12, 16],
        "macd_slow": [26, 32],
        "macd_signal": [7, 11],
        # Total: 12 valid combos (was 36, -67%)
    },
    "supertrend_flip": {
        "st_length": [7, 10, 14, 20],
        "st_multiplier": [2.0, 3.0, 4.0, 5.0],
        # Total: 16 combos (unchanged — all values have meaningful selection rates)
    },
}

DETAILED_EXIT_AXIS_GRIDS: dict[str, dict[str, Any]] = {
    "atr_trailing": {
        "atr_length": [14, 21, 30],
        "atr_multiplier": [2.5, 3.5, 4.5, 6.0],
        # Total: 12 combos (unchanged — all values have meaningful selection rates)
    },
    "fixed_sl_tp": {
        # stop_pct: 1.5 selected 100% of the time — collapse to constant.
        # take_profit_pct: 3.0 selected 100% of the time — collapse to constant.
        "stop_pct": [1.5],
        "take_profit_pct": [3.0],
        # Total: 1 combo (was 12, -92%)
    },
    "trailing_stop": {
        # trailing_stop_pct: 3.0 selected 100% of the time — collapse to constant.
        "trailing_stop_pct": [3.0],
        # Total: 1 combo (was 4, -75%)
    },
}

# --- Standard grids (Historical/Broad Exploration) ---

COARSE_ENTRY_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    "psar_adx": {
        "sar_acceleration": [0.01, 0.02],
        "sar_maximum": [0.2],
        "adx_length": [14, 20],
        "adx_threshold": [20, 25, 30],
        "use_dmp_cross": [True, False],
    },
    "ema_cross": {
        "ema_fast": [9, 16, 50],
        "ema_slow": [21, 100, 200],
    },
    "rsi_reversal": {
        "rsi_length": [14, 21, 28],
        "rsi_oversold": [20, 25, 30],
        "rsi_trend_filter": [False],
    },
    "donchian_breakout": {
        "donchian_length": [15, 20, 30, 50],
    },
}

EXIT_AXIS_GRIDS: dict[str, dict[str, Any]] = {
    "atr_trailing": {
        "atr_length": [14, 21, 30],
        "atr_multiplier": [2.5, 3.5, 4.5, 6.0],
    },
    "fixed_sl_tp": {
        "stop_pct": [1.5, 2.0, 3.0],
    },
    "trailing_stop": {
        "trailing_stop_pct": [3.0, 5.0, 8.0, 12.0],
    },
}

# --- Sensitivity / Specific Combinations ---

COARSE_SENSITIVITY_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    name: {**COARSE_ENTRY_PARAM_GRIDS[name], **EXIT_AXIS_GRIDS["atr_trailing"]}
    for name in COARSE_ENTRY_PARAM_GRIDS
}

DETAILED_SENSITIVITY_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    name: {**DETAILED_ENTRY_PARAM_GRIDS[name], **DETAILED_EXIT_AXIS_GRIDS["atr_trailing"]}
    for name in DETAILED_ENTRY_PARAM_GRIDS
}


def build_param_grid(
    entry_strategy: str,
    exit_strategy: str,
    entry_book: dict[str, dict[str, Any]] | None = None,
    exit_book: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge entry-only params with exit-axis params for one (entry, exit) combination."""
    if entry_book is None:
        entry_book = DETAILED_ENTRY_PARAM_GRIDS
    if exit_book is None:
        exit_book = DETAILED_EXIT_AXIS_GRIDS
    entry_part = dict(entry_book.get(entry_strategy, {}))
    exit_part = dict(exit_book.get(exit_strategy, {}))
    return {**entry_part, **exit_part}


def merge_exit_axes_for_tournament(
    exit_book: dict[str, dict[str, Any]],
    exit_tournament: list[str],
) -> dict[str, Any]:
    """Union of all exit-axis dicts for exits in the tournament (WFO superset grid)."""
    merged: dict[str, Any] = {}
    for exit_name in exit_tournament:
        merged.update(exit_book.get(exit_name, {}))
    return merged


def build_wfo_superset_grids(
    entry_book: dict[str, dict[str, Any]],
    exit_book: dict[str, dict[str, Any]],
    exit_tournament: list[str],
    dry_run: bool,
) -> dict[str, dict[str, Any]]:
    """Per-entry-strategy grids: entry params + merged exit axes for EXIT_TOURNAMENT."""
    merged_exit_axes = merge_exit_axes_for_tournament(exit_book, exit_tournament)
    narrowed: dict[str, dict[str, Any]] = {}
    # CRITICAL FIX (Iteration 6): Only build grids for entries explicitly in the entry_book
    # This avoids running empty grids for unused strategies.
    for strategy_name in entry_book.keys():
        param_grid = dict(entry_book.get(strategy_name, {}))
        param_grid.update(merged_exit_axes)
        if dry_run:
            param_grid = {k: v[:2] if isinstance(v, list) else [v] for k, v in param_grid.items()}
        narrowed[strategy_name] = dict(param_grid)
    return narrowed
