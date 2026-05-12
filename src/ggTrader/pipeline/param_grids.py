"""Walk-forward and sensitivity parameter grids (entry axes + exit axes)."""

from __future__ import annotations

from typing import Any

# --- Discovery (Iteration 6) Expanded Grids ---

DETAILED_ENTRY_PARAM_GRIDS: dict[str, dict[str, Any]] = {
    # WFO Textbook Reset (2026-05-10): coarsened per the articulation test.
    # If we can't explain why value A should behave differently from adjacent
    # value B, drop one. Selection bias from max-of-N scales with N; smaller
    # grids = fewer noise draws = smaller expected outlier. Target ≤16 cells
    # per (entry × exit) pair. Largest cross product post-coarsening: 4×4 = 16.
    "psar_adx": {
        # Residue cleanup (2026-05-11): three STRONG single-value pins inherited
        # from commit 33d863d (2026-03-27) were justified by "100% selection
        # across 261 pre-reset WFO runs" — a corpus generated under the
        # leaky-selection methodology that the textbook reset later invalidated.
        # Re-expanded to theory-justified pairs (articulation-disciplined: each
        # pair is behaviorally distinct, NOT survivor-tuned to frictionless tops).
        #
        # sar_acceleration: 0.02 (Wilder canonical) vs 0.03 (slightly faster
        #   acceleration suited to higher-vol 4h crypto). 0.01 already domain-
        #   rejected by c8a53c2 ("step so small SAR barely accelerates").
        # sar_maximum: 0.1 (half-canonical, conservative trailing buffer) vs
        #   0.2 (canonical Wilder maximum AF, responsive trailing).
        # use_dmp_cross: False (ADX-threshold entry) vs True (DI+/DI- cross
        #   entry) — fundamentally distinct ADX-based entry logics; pinning to
        #   one removes an entire textbook entry mode from search.
        # adx_length: 14 (Wilder canonical, responsive) vs 30 (long-period
        #   smoothing). Articulation-defensible pair; empirically-dropped 10
        #   has no canonical theory backing.
        # adx_threshold: 20 (loose trend filter) vs 30 (tight). Articulation pair.
        "sar_acceleration": [0.02, 0.03],
        "sar_maximum": [0.1, 0.2],
        "adx_length": [14, 30],
        "adx_threshold": [20, 30],
        "use_dmp_cross": [False, True],
        # Total: 2×2×2×2×2 = 32 combos (was 4; pre-residue-cleanup textbook size).
        # NB: this exceeds the 16-cells-per-combo target from the articulation
        # commit. Tradeoff is intentional — the prior cap was achieved partly by
        # pre-reset empirical pins now known to be circular.
    },
    "ema_cross": {
        # ema_fast: 9 (short) vs 12 (intermediate). Dropped 5 (too noisy at 4h).
        # ema_slow: 50 (medium trend) vs 200 (long trend). Dropped 100 (interp).
        "ema_fast": [9, 12],
        "ema_slow": [50, 200],
        # Total: 4 combos (was 9)
    },
    "mtf_momentum": {
        # 4h EMA cross gated by daily-equivalent slow EMA. fast/slow at single
        # canonical values; the discriminating axis is the daily filter strength.
        # mtf_daily_ema: 100 (intermediate) vs 200 (long). Dropped 50 (too short).
        "ema_fast": [9],
        "ema_slow": [21],
        "mtf_daily_ema": [100, 200],
        # Total: 2 combos (was 12)
    },
    "rsi_reversal": {
        # rsi_length: 10 (short, fast) vs 21 (long, smooth). Dropped 14 (interp).
        # rsi_oversold: 25 (moderate) vs 30 (loose). Dropped 20 (extreme), 35
        # (very loose — fires too often).
        # rsi_trend_filter: dropped True path; this strategy is mean-reversion,
        # adding a trend filter creates a different strategy (adx_filtered_rsi).
        "rsi_length": [10, 21],
        "rsi_oversold": [25, 30],
        "rsi_trend_filter": [False],
        # Total: 4 combos (was 24)
    },
    "adx_filtered_rsi": {
        # adx_max is the discriminating axis: 15 (very range-only) vs 25 (loose).
        # Dropped 20 (interp). rsi_oversold collapsed to single canonical 30.
        "rsi_length": [14],
        "rsi_oversold": [30],
        "adx_length": [14],
        "adx_max": [15, 25],
        # Total: 2 combos (was 9)
    },
    "donchian_breakout": {
        # donchian_length: 30 (short channel, frequent breakouts) vs 100 (long
        # channel, fewer high-conviction breakouts). Dropped 50 (interp).
        "donchian_length": [30, 100],
        # Total: 2 combos (was 3)
    },
    "macd_cross": {
        # Residue cleanup (2026-05-11): the chain that arrived at [8, 16]
        # passed through 33d863d's empirical drop of {5, 9, 21} ("0 selections"
        # under pre-reset leaky methodology). The articulation reset (ad74480)
        # then independently dropped 12 as "interpolation" between 8 and 16,
        # but 12 IS the canonical Appel MACD fast EMA. Re-adding 12 restores
        # the textbook canonical value to the search space (theory-justified,
        # not survivor-tuned).
        #
        # macd_fast: 8 (faster Appel variant) / 12 (canonical Appel) / 16 (slower).
        # macd_slow: 26 = canonical Appel MACD slow EMA. Single-pin is theory-
        #   justified at the canonical value (NOT empirically-justified at 26).
        # macd_signal: 9 = canonical Appel signal EMA. Same theory pin.
        "macd_fast": [8, 12, 16],
        "macd_slow": [26],
        "macd_signal": [9],
        # Total: 3 combos (was 2)
    },
    "supertrend_flip": {
        # st_length: 10 (short, responsive) vs 20 (long, smooth). Dropped 7, 14.
        # st_multiplier: 2.0 (tight) vs 4.0 (loose). Dropped 3.0 (interp), 5.0
        # (very loose).
        "st_length": [10, 20],
        "st_multiplier": [2.0, 4.0],
        # Total: 4 combos (was 16)
    },
    "bbands_mean_reversion": {
        # bb_length: canonical 20 only. Dropped 14, 30.
        # bb_std: 1.5 (loose — fires often) vs 2.5 (tight — fires at extremes).
        # Dropped 2.0 (interp).
        "bb_length": [20],
        "bb_std": [1.5, 2.5],
        # Total: 2 combos (was 9)
    },
    "stoch_rsi_reversal": {
        # Lengths collapsed to canonical 14. stochrsi_oversold: 15 (strict) vs
        # 20 (moderate). Discriminates between strict and loose oversold thresholds.
        "stochrsi_rsi_length": [14],
        "stochrsi_stoch_length": [14],
        "stochrsi_oversold": [15, 20],
        # Total: 2 combos (was 12)
    },
    "keltner_breakout": {
        # kc_length: canonical 20 only. kc_multiplier: 1.0 (tight channel,
        # frequent breakouts) vs 2.0 (wide channel, high-conviction breakouts).
        # Dropped 1.5 (interp).
        "kc_length": [20],
        "kc_multiplier": [1.0, 2.0],
        # Total: 2 combos (was 9)
    },
}

DETAILED_EXIT_AXIS_GRIDS: dict[str, dict[str, Any]] = {
    # Coarsened per articulation test. Each value represents a meaningfully
    # different exit behavior.
    "atr_trailing": {
        # atr_length: 14 (responsive) vs 30 (smooth). Dropped 21 (interp).
        # atr_multiplier: 2.5 (tight trailing) vs 4.5 (loose). Dropped 3.5
        # (interp), 6.0 (very loose).
        "atr_length": [14, 30],
        "atr_multiplier": [2.5, 4.5],
        # Total: 4 combos (was 12)
    },
    "fixed_sl_tp": {
        # stop_pct: 2% (tight) vs 5% (loose). Dropped 1.5, 3 (interp).
        # take_profit_pct: 4% (modest) vs 10% (asymmetric high-reward trend
        # capture). Dropped 3, 6 (interp).
        "stop_pct": [2.0, 5.0],
        "take_profit_pct": [4.0, 10.0],
        # Total: 4 combos (was 16)
    },
    "trailing_stop": {
        # 3% (tight, frequent stops) vs 8% (loose, lets winners run).
        # Dropped 5 (interp), 12 (very loose).
        "trailing_stop_pct": [3.0, 8.0],
        # Total: 2 combos (was 4)
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
    "mtf_momentum": {
        "ema_fast": [9],
        "ema_slow": [21],
        "mtf_daily_ema": [100, 200],
    },
    "rsi_reversal": {
        "rsi_length": [14, 21, 28],
        "rsi_oversold": [20, 25, 30],
        "rsi_trend_filter": [False],
    },
    "adx_filtered_rsi": {
        "rsi_length": [14],
        "rsi_oversold": [30],
        "adx_length": [14],
        "adx_max": [20],
    },
    "donchian_breakout": {
        "donchian_length": [15, 20, 30, 50],
    },
    "stoch_rsi_reversal": {
        "stochrsi_rsi_length": [14],
        "stochrsi_stoch_length": [14],
        "stochrsi_oversold": [20],
    },
    "keltner_breakout": {
        "kc_length": [20],
        "kc_multiplier": [1.5],
    },
}

EXIT_AXIS_GRIDS: dict[str, dict[str, Any]] = {
    "atr_trailing": {
        "atr_length": [14, 21, 30],
        "atr_multiplier": [2.5, 3.5, 4.5, 6.0],
    },
    "fixed_sl_tp": {
        "stop_pct": [1.5, 2.0, 3.0],
        "take_profit_pct": [2.0, 3.0, 4.0, 6.0],
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
