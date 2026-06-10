import itertools
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.indicators.indicator_precompute import IndicatorPrecomputer
from ggTrader.indicators.strategies import (
    EXIT_PARAM_AXIS_KEYS,
    filter_strat_params_for_exit,
    get_entry_strategy,
    get_exit_strategy,
)

# Defaults for portfolio-level config keys
_DEFAULT_CONFIG = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.001,
    "SLIPPAGE": 0.0005,
    "FREQ": "4h",
    "N_JOBS": -1,  # Default to all cores for vectorized runs
    "USE_CASH_SHARING": True,  # New config for grouping
    "ENTRY_STRATEGY": None,  # Entry strategy name (e.g., "psar_adx")
    "EXIT_STRATEGY": None,  # Exit strategy name (e.g., "atr_trailing")
}

# Performance settings
vbt.settings.caching["enabled"] = False


def _listify_param(x: Any) -> list:
    """Normalize scalar or list parameter values."""
    return x if isinstance(x, list) else [x]


def _get_exit_axis_keys(exit_strategy_name: str) -> tuple[str, ...]:
    """Return the param-grid keys that drive the exit dimension for the given exit strategy."""
    return EXIT_PARAM_AXIS_KEYS.get(exit_strategy_name, ())


def _expand_entries_for_exit_product(
    entries: np.ndarray, n_symbols: int, n_exit: int
) -> np.ndarray:
    """Repeat each entry-combo block once per exit-param variant.

    Works for any exit strategy: ATR trailing (n_atr variants) or fixed SL/TP
    (n_stop × n_tp variants).  Replaces the old ATR-only helper.
    """
    n_time, w = entries.shape
    n_e = w // n_symbols
    parts: list[np.ndarray] = []
    for e in range(n_e):
        sl = entries[:, e * n_symbols : (e + 1) * n_symbols]
        for _ in range(n_exit):
            parts.append(sl.copy())
    return np.hstack(parts) if parts else entries


def _merge_entry_exit_param_combos(
    entry_combos: List[dict],
    strat_params: dict,
    exit_axis_keys: tuple[str, ...],
) -> List[dict]:
    """Cartesian product of entry param rows × exit-axis tuples.

    Replaces the old ATR-only helper.  ``exit_axis_keys`` is obtained from
    ``EXIT_PARAM_AXIS_KEYS[exit_strategy_name]``.  Float-casts are applied to
    keys whose values are floats in the param grid.
    """
    relevant_keys = [k for k in exit_axis_keys if k in strat_params]
    if not relevant_keys:
        return [dict(ec) for ec in entry_combos]
    lists = [_listify_param(strat_params[k]) for k in relevant_keys]
    out: List[dict] = []
    for ec in entry_combos:
        for tup in itertools.product(*lists):
            row = dict(ec)
            for k, v in zip(relevant_keys, tup):
                row[k] = float(v) if isinstance(v, (float, np.floating)) else v
            out.append(row)
    return out


class FastBacktest:
    """Vectorized backtest engine wrapping VectorBT's Portfolio API.

    Features:
    - Optimization-ready (lightweight, minimal copying)
    - Full metric generation
    - Supports grouped portfolios (sensitivity analysis)
    """

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        params: dict,
        config: Optional[dict] = None,
        mover_mask: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            ohlcv: OHLCV DataFrame (MultiIndex columns: symbol, field)
            params: Strategy parameters
            config: Configuration dictionary (fees, slippage, etc)
            mover_mask: Optional boolean mask to filter entries/exits
        """
        # Merge caller config with defaults
        self.config = {**_DEFAULT_CONFIG, **(config or {})}

        self.ohlcv = ohlcv
        self.params = params
        self.mover_mask = mover_mask
        self.pf = None  # Portfolio cache
        self.entries = None
        self.exits = None
        self._last_param_combos: Optional[List[dict]] = None

    def run(self, show_progress: bool = False) -> vbt.Portfolio:
        """Execute backtest."""
        if self.ohlcv.empty:
            raise ValueError(
                "OHLCV data is empty. Check your symbols, date range, and database connection."
            )

        self._last_param_combos = None

        # 1. Generate Signals (strategy registry; handles scalar params as 1-combo grids)
        entries, exits, price_for_orders = self._generate_signals_vectorized(show_progress)

        # 2. Apply Mover Mask (if exists)
        if self.mover_mask is not None:
            entries = self._apply_mover_mask(entries)

        entries, exits, price_for_orders = self._sort_multindex_columns_for_groupby(
            entries, exits, price_for_orders
        )

        # 3. Determine Grouping
        group_by, use_cash_sharing = self._determine_grouping(entries)

        # 4. Create Portfolio
        self.entries = entries
        self.exits = exits
        self.pf = self._create_portfolio(
            price_for_orders, entries, exits, group_by, use_cash_sharing
        )
        return self.pf

    def _sort_multindex_columns_for_groupby(
        self,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        price_for_orders: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Lex-sort MultiIndex columns so VectorBT group_by is contiguous (all paths)."""
        if not isinstance(entries.columns, pd.MultiIndex):
            return entries, exits, price_for_orders

        # Optimization: for single-symbol backtests (common in WFO per-coin workers),
        # broadcast the only price column across all param-combo columns.
        if len(price_for_orders.columns) == 1:
            sym_name = price_for_orders.columns[0]
            price_arr = price_for_orders[sym_name].reindex(entries.index).to_numpy(dtype=np.float64)
            close_aligned = np.column_stack([price_arr] * entries.shape[1])
            price_aligned = pd.DataFrame(
                close_aligned, index=entries.index, columns=entries.columns
            )
            return entries, exits, price_aligned

        sorted_cols = entries.columns.sort_values()
        entries = entries[sorted_cols]
        exits = exits[sorted_cols]

        if entries.columns.equals(price_for_orders.columns):
            return entries, exits, price_for_orders.reindex(columns=entries.columns)

        # Multi-symbol path: Align price columns to entries by matching the
        # 'symbol' level (last level)
        sym_level = entries.columns.get_level_values(-1)
        pieces = []
        for sym in sym_level:
            if sym not in price_for_orders.columns:
                raise KeyError(
                    f"Symbol {sym!r} not in price columns: {price_for_orders.columns.tolist()}"
                )
            ser = price_for_orders[sym].reindex(entries.index)
            pieces.append(ser.to_numpy(dtype=np.float64, copy=True))
        close_aligned = np.column_stack(pieces)
        price_aligned = pd.DataFrame(close_aligned, index=entries.index, columns=entries.columns)
        return entries, exits, price_aligned

    def _generate_signals_vectorized(
        self, show_progress: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates signals using vectorized pre-computation and broadcasting.

        This path avoids redundant indicator computation by pre-computing each
        indicator once, then combining via numpy broadcasting. Dispatches to
        registered entry/exit strategies based on config.
        """
        ohlcv = self.ohlcv.astype(np.float64)

        # Unpack OHLCV
        close = ohlcv.xs("close", axis=1, level=1, drop_level=True)
        high = ohlcv.xs("high", axis=1, level=1, drop_level=True)
        low = ohlcv.xs("low", axis=1, level=1, drop_level=True)
        ohlcv.xs("open", axis=1, level=1, drop_level=True)

        # Initialize precomputer
        precomputer = IndicatorPrecomputer(close, high, low)

        # Extract strategy params
        strat_params = {
            k: v
            for k, v in self.params.items()
            if k
            not in [
                "START_CASH",
                "PORTFOLIO_SHARE",
                "FEES",
                "SLIPPAGE",
                "FREQ",
                "PARAM_PRODUCT",
            ]
        }

        # Get entry and exit strategies from config, or use defaults
        entry_strategy_name = self.config.get("ENTRY_STRATEGY") or "psar_adx"
        exit_strategy_name = self.config.get("EXIT_STRATEGY") or "atr_trailing"

        entry_strategy = get_entry_strategy(entry_strategy_name)
        exit_strategy = get_exit_strategy(exit_strategy_name)

        # Superset grids (all EXIT_TOURNAMENT axes merged): drop foreign exit-axis keys.
        strat_params = filter_strat_params_for_exit(strat_params, exit_strategy_name)

        entries, entry_param_combos = entry_strategy.compute_entries(precomputer, strat_params)

        n_symbols = close.shape[1]
        exit_axis_keys = _get_exit_axis_keys(exit_strategy_name)
        n_exit = 1
        for k in exit_axis_keys:
            if k in strat_params:
                n_exit *= len(_listify_param(strat_params[k]))

        if n_exit > 1:
            entries = _expand_entries_for_exit_product(entries, n_symbols, n_exit)

        self._last_param_combos = _merge_entry_exit_param_combos(
            entry_param_combos, strat_params, exit_axis_keys
        )

        exits, stops, price_for_orders_arr = exit_strategy.compute_exits(
            entries, precomputer, strat_params, n_symbols
        )

        # Convert to DataFrames with proper indexing
        # Build column index: (param_combo_idx, symbol_0), (param_combo_idx, symbol_1), etc.
        n_combos = entries.shape[1] // n_symbols
        param_levels = []
        symbol_levels = []

        for combo_idx in range(n_combos):
            for sym_idx, symbol in enumerate(close.columns):
                param_levels.append(combo_idx)
                symbol_levels.append(symbol)

        multi_index = pd.MultiIndex.from_arrays(
            [param_levels, symbol_levels], names=["param_combo", "symbol"]
        )

        entries_df = pd.DataFrame(entries, index=close.index, columns=multi_index)
        exits_df = pd.DataFrame(exits, index=close.index, columns=multi_index)
        price_df = pd.DataFrame(price_for_orders_arr, index=close.index, columns=multi_index)

        # VectorBT requires group_by labels to be contiguous; lex-sort MultiIndex columns.
        if isinstance(entries_df.columns, pd.MultiIndex):
            sorted_cols = entries_df.columns.sort_values()
            entries_df = entries_df[sorted_cols]
            exits_df = exits_df[sorted_cols]
            price_df = price_df[sorted_cols]

        precomputer.clear_cache()

        return entries_df, exits_df, price_df

    def _apply_mover_mask(self, entries: pd.DataFrame) -> pd.DataFrame:
        """Applies the mover mask to filter entries."""
        # Align mover_mask with entries (Broadcast if MultiIndex)
        m_mask = self.mover_mask.vbt.broadcast_to(entries)
        # We don't strictly NEED to mask exits as the stop logic handles it,
        # but masking entries ensures no new trades start for non-movers.
        return entries & m_mask

    def _determine_grouping(self, entries: pd.DataFrame) -> Tuple[Any, bool]:
        """Determines the 'group_by' argument for VectorBT."""
        use_grouping = self.config.get("USE_CASH_SHARING", True)
        if use_grouping:
            if isinstance(entries.columns, pd.MultiIndex):
                # Grid Search: Always group by params (drop symbol level)
                # Assuming standard VBT param_product output where symbol is last level
                # and params are previous levels
                group_by = entries.columns.droplevel(-1)
            else:
                # Single run: Group all symbols into one combined portfolio
                # Only if there's more than one column to group
                if entries.shape[1] > 1:
                    group_by = np.full(entries.shape[1], 0)
                else:
                    group_by = False
        else:
            # Independent assets, no cash sharing
            group_by = False

        return group_by, use_grouping

    def _create_portfolio(
        self,
        close: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        group_by: Any,
        cash_sharing: bool,
    ) -> vbt.Portfolio:
        """Creates the VectorBT Portfolio object."""
        idx = entries.index
        close = close.reindex(idx)
        exits = exits.reindex(idx)
        if not close.columns.equals(entries.columns):
            close = close.reindex(columns=entries.columns)
        if not exits.columns.equals(entries.columns):
            exits = exits.reindex(columns=entries.columns)

        # Force writable copies of everything to prevent Numba read-only errors
        # This is critical for metrics like profit_factor which modify arrays in-place
        close_writable = close.values.copy() if hasattr(close, "values") else close.copy()
        entries_writable = entries.values.copy() if hasattr(entries, "values") else entries.copy()
        exits_writable = exits.values.copy() if hasattr(exits, "values") else exits.copy()

        pf = vbt.Portfolio.from_signals(
            close=pd.DataFrame(close_writable, index=close.index, columns=close.columns),
            entries=pd.DataFrame(entries_writable, index=entries.index, columns=entries.columns),
            exits=pd.DataFrame(exits_writable, index=exits.index, columns=exits.columns),
            init_cash=float(self.config["START_CASH"]),
            fees=float(self.config["FEES"]),
            slippage=float(self.config["SLIPPAGE"]),
            freq=self.config["FREQ"],
            size=float(self.config["PORTFOLIO_SHARE"]),
            size_type="percent",
            cash_sharing=cash_sharing,
            group_by=group_by,
        ).copy()
        return pf

    def get_stats(self) -> dict:
        """Return aggregate metrics formatted for ResultsManager."""
        if self.pf is None:
            raise ValueError("Run the backtest first.")

        total_value = self.pf.final_value()
        if isinstance(total_value, pd.Series):
            total_value = total_value.sum()

        total_profit = self.pf.total_profit()
        if isinstance(total_profit, pd.Series):
            total_profit = total_profit.sum()

        # Derive profit percentage from actual initial cash
        init_cash = float(self.config["START_CASH"])
        profit_pct = (total_profit / init_cash) * 100

        import math

        def _safe(val: object, default: float = 0.0) -> float:
            """Replace None, NaN, or Inf with default for JSON safety."""
            if val is None:
                return default
            try:
                v = float(val)
            except (TypeError, ValueError):
                return default
            return default if (math.isnan(v) or math.isinf(v)) else v

        return {
            "total_value": _safe(total_value),
            "total_profit": _safe(total_profit),
            "profit_pct": _safe(profit_pct),
            "total_trades": int(self.pf.trades.count().sum()),
            "win_rate": _safe(self.pf.trades.win_rate().mean()) * 100,
            "sharpe": _safe(self.pf.sharpe_ratio().mean()),
            "sortino": _safe(self.pf.sortino_ratio().mean()),
            "max_drawdown": _safe(self.pf.max_drawdown().min()) * 100,
        }

    def save_detailed_plots(self, results_manager, filename: str = "backtest_detailed") -> None:
        """
        Saves a comprehensive multi-panel plot of the backtest.
        Safe for cash_sharing=True.
        """
        if self.pf is None:
            raise ValueError("Run the backtest first.")

        # Subplots that provide a good overview for shared-cash portfolios
        subplots = [
            "cumulative_returns",
            "drawdowns",
            "daily_returns",
            "cash_sharing",
        ]

        try:
            fig = self.pf.plot(subplots=subplots)
            results_manager.save_plot(fig, filename)
        except Exception as e:
            print(f"Warning: Could not generate detailed plots: {e}")
            # Fallback to basic plot if specific subplots fail
            try:
                fig = self.pf.plot()
                results_manager.save_plot(fig, f"{filename}_basic")
            except Exception as e:
                pass
