import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Any, Optional, Tuple, Union

from ggTrader.indicators.signals import SignalFactory
from ggTrader.utils.vbt_patches import apply_vbt_patches

# Apply patches immediately on import
apply_vbt_patches()


# Defaults for portfolio-level config keys
_DEFAULT_CONFIG = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.001,
    "SLIPPAGE": 0.0005,
    "FREQ": "4h",
    "N_JOBS": -1,  # Default to all cores for vectorized runs
    "MIN_TRADES": 0,  # Minimum trades to accept a result
    "USE_CASH_SHARING": True,  # New config for grouping
}

# Performance settings
vbt.settings.caching["enabled"] = False


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
        config: dict | None = None,
        signal_factory: Any = None,
        mover_mask: pd.DataFrame | None = None,
    ):
        """
        Args:
            ohlcv: OHLCV DataFrame (MultiIndex columns: symbol, field)
            params: Strategy parameters
            config: Configuration dictionary (fees, slippage, etc)
            signal_factory: Optional custom signal factory
            mover_mask: Optional boolean mask to filter entries/exits
        """
        # Merge caller config with defaults
        self.config = {**_DEFAULT_CONFIG, **(config or {})}

        self.ohlcv = ohlcv
        self.params = params
        self.mover_mask = mover_mask
        self.pf = None  # Portfolio cache
        self.signal_factory = signal_factory or SignalFactory

    def run(self, show_progress: bool = False) -> vbt.Portfolio:
        """Execute backtest."""
        if self.ohlcv.empty:
            raise ValueError(
                "OHLCV data is empty. Check your symbols, date range, " "and database connection."
            )

        # 1. Generate Signals
        entries, exits, price_for_orders = self._generate_signals(show_progress)

        # 2. Apply Mover Mask (if exists)
        if self.mover_mask is not None:
            entries = self._apply_mover_mask(entries)

        # 3. Determine Grouping
        group_by, use_cash_sharing = self._determine_grouping(entries)

        # 4. Create Portfolio
        self.pf = self._create_portfolio(
            price_for_orders, entries, exits, group_by, use_cash_sharing
        )
        return self.pf

    def _generate_signals(
        self, show_progress: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs the signal factory to generate entry/exit signals."""
        # Clean params for signal factory (remove non-strategy params)
        strat_params = {
            k: v
            for k, v in self.params.items()
            if k not in ["START_CASH", "PORTFOLIO_SHARE", "FEES", "SLIPPAGE", "FREQ"]
        }

        # Run signal generation
        # Pass full OHLCV as signal factory expects it
        # Use float64 to avoid Numba read-only assignment errors in metrics
        ohlcv = self.ohlcv.astype(np.float64)

        # Unpack OHLCV
        close = ohlcv.xs("close", axis=1, level=1, drop_level=True)
        high = ohlcv.xs("high", axis=1, level=1, drop_level=True)
        low = ohlcv.xs("low", axis=1, level=1, drop_level=True)
        open_ = ohlcv.xs("open", axis=1, level=1, drop_level=True)

        sf = self.signal_factory.run(
            close=close,
            high=high,
            low=low,
            open_=open_,
            **strat_params,
            param_product=self.config.get("PARAM_PRODUCT", True),
            n_jobs=self.config.get("N_JOBS", -1),
            show_progress=show_progress,
        )

        # Prepare execution arrays
        # Use close price for execution
        # Ensure we use specific 'close' column to avoid ambiguity
        price_for_orders = self.ohlcv.xs("close", level=1, axis=1)

        return sf.entries, sf.exits, price_for_orders

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
                group_by = np.full(entries.shape[1], 0)
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

        def _safe(val: float, default: float = 0.0) -> float:
            """Replace NaN/Inf with default for JSON safety."""
            return default if (math.isnan(val) or math.isinf(val)) else val

        return {
            "total_value": _safe(float(total_value)),
            "total_profit": _safe(float(total_profit)),
            "profit_pct": _safe(float(profit_pct)),
            "total_trades": int(self.pf.trades.count().sum()),
            "win_rate": _safe(float(self.pf.trades.win_rate().mean() * 100)),
            "sharpe": _safe(float(self.pf.sharpe_ratio().mean())),
            "sortino": _safe(float(self.pf.sortino_ratio().mean())),
            "max_drawdown": _safe(float(self.pf.max_drawdown().min() * 100)),
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
            except:
                pass
