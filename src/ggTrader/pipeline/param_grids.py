"""Walk-forward and sensitivity parameter grids (entry axes + exit axes)."""

from __future__ import annotations

from typing import Any

from ggTrader.indicators.strategies import ENTRY_REGISTRY

# --- Entry-only parameter grids (no exit-axis keys) ---

COARSE_ENTRY_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    "psar_adx": {
        "sar_acceleration": [0.01, 0.03],
        "sar_maximum": [0.1, 0.3],
        "adx_length": [14, 20],
        "adx_threshold": [25],
        "use_dmp_cross": [True],
    },
    "ema_cross": {
        "ema_fast": [5, 20],
        # Cap slow EMA vs 100 to cut extreme fast/slow pairs that often fold-dry in WFO.
        "ema_slow": [21, 50],
    },
    "rsi_reversal": {
        "rsi_length": [7, 21],
        "rsi_oversold": [30, 40],
    },
    "macd_cross": {
        "macd_fast": [10, 12],
        "macd_slow": [22, 26],
        "macd_signal": [9],
    },
    "bbands_mean_reversion": {
        "bb_length": [15, 20],
        "bb_std": [2.0],
    },
    "donchian_breakout": {
        "donchian_length": [10, 20],
    },
    "supertrend_flip": {
        "st_length": [7, 10],
        "st_multiplier": [2.0, 3.0],
    },
}

DETAILED_ENTRY_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    "psar_adx": {
        "sar_acceleration": [0.01, 0.02, 0.03],
        "sar_maximum": [0.1, 0.2, 0.3],
        "adx_length": [10, 14, 20],
        "adx_threshold": [15, 20, 25, 30],
        "use_dmp_cross": [True, False],
    },
    "ema_cross": {
        "ema_fast": [5, 9, 12, 20],
        "ema_slow": [21, 34, 50, 100],
    },
    "rsi_reversal": {
        "rsi_length": [7, 14, 21],
        "rsi_oversold": [20, 25, 30, 35, 40],
    },
    "macd_cross": {
        "macd_fast": [8, 12, 16],
        "macd_slow": [21, 26, 34],
        "macd_signal": [7, 9],
    },
    "bbands_mean_reversion": {
        "bb_length": [15, 20, 25],
        "bb_std": [2.0, 2.5],
    },
    "donchian_breakout": {
        "donchian_length": [10, 15, 20, 25],
    },
    "supertrend_flip": {
        "st_length": [7, 10, 14],
        "st_multiplier": [2.0, 2.5, 3.0],
    },
}

EXIT_AXIS_GRIDS: dict[str, dict[str, Any]] = {
    "atr_trailing": {
        "atr_length": [10, 20],
        "atr_multiplier": [2.0, 3.0],
    },
    "fixed_sl_tp": {
        "stop_pct": [2.0, 4.0],
        "take_profit_pct": [4.0, 8.0],
    },
}

DETAILED_EXIT_AXIS_GRIDS: dict[str, dict[str, Any]] = {
    "atr_trailing": {
        "atr_length": [10, 14, 20],
        "atr_multiplier": [2.0, 2.5, 3.0],
    },
    "fixed_sl_tp": {
        "stop_pct": [1.5, 2.0, 3.0, 5.0],
        "take_profit_pct": [3.0, 5.0, 8.0, 12.0],
    },
}

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
        entry_book = COARSE_ENTRY_PARAM_GRIDS
    if exit_book is None:
        exit_book = EXIT_AXIS_GRIDS
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
    for strategy_name in ENTRY_REGISTRY.keys():
        param_grid = dict(entry_book.get(strategy_name, {}))
        param_grid.update(merged_exit_axes)
        if dry_run:
            param_grid = {
                k: v[:2] if isinstance(v, list) else [v] for k, v in param_grid.items()
            }
        narrowed[strategy_name] = dict(param_grid)
    return narrowed
