"""Walk-forward and sensitivity parameter grids (entry axes + exit axes)."""

from __future__ import annotations

from typing import Any

from ggTrader.indicators.strategies import ENTRY_REGISTRY

# --- Discovery (Iteration 6) Expanded Grids ---

DETAILED_ENTRY_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    "psar_adx": {
        "sar_acceleration": [0.01, 0.02, 0.03],
        "sar_maximum": [0.1, 0.2, 0.3],
        "adx_length": [10, 14, 20],
        "adx_threshold": [15, 20, 25, 30],
        "use_dmp_cross": [True, False],  # RE-ACTIVATED
    },
    "ema_cross": {
        "ema_fast": [5, 9, 12, 20],
        "ema_slow": [21, 34, 50, 100],
    },
    "rsi_reversal": {
        "rsi_length": [7, 14, 21, 28],  # RE-INTRODUCED RSI-7
        "rsi_oversold": [20, 25, 30, 35, 40], # EXPANDED
    },
    "donchian_breakout": {
        "donchian_length": [10, 15, 20, 30, 50], # EXPANDED
    },
}

DETAILED_EXIT_AXIS_GRIDS: dict[str, dict[str, Any]] = {
    "atr_trailing": {
        "atr_length": [14, 21, 30],
        "atr_multiplier": [2.5, 3.5, 4.5, 6.0], # EXPANDED
    },
    "fixed_sl_tp": {
        "stop_pct": [1.5, 2.0, 3.0], # EXPANDED
        "take_profit_pct": [3.0, 6.0, 10.0, 15.0, 25.0, 40.0], # EXPANDED FOR MOON SHOTS
    },
    "trailing_stop": {
        "trailing_stop_pct": [3.0, 5.0, 8.0, 12.0], # EXPANDED
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
            param_grid = {
                k: v[:2] if isinstance(v, list) else [v] for k, v in param_grid.items()
            }
        narrowed[strategy_name] = dict(param_grid)
    return narrowed
