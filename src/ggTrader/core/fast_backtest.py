"""Vectorized backtest engine using VectorBT Portfolio API."""

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.indicators.signals import SignalFactory


# Defaults for portfolio-level config keys
_DEFAULT_CONFIG = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.001,
    "SLIPPAGE": 0.0005,
    "FREQ": "4h",
    "N_JOBS": -1,  # Default to all cores for vectorized runs
    "MIN_TRADES": 0,  # Minimum trades to accept a result
}

# Performance settings
vbt.settings.caching["enabled"] = True


class FastBacktest:
    """Vectorized backtest engine wrapping VectorBT's Portfolio API.

    Args:
        ohlcv_df: MultiIndex DataFrame (symbol, ohlcv) from load_data_and_setup.
        params: Signal parameters for SignalFactory (may contain lists for broadcasting).
        config: Portfolio-level settings dict (CONSTANTS). Recognised keys:
            START_CASH, PORTFOLIO_SHARE, FEES, SLIPPAGE, FREQ.
        mover_mask: Optional boolean DataFrame (dates x symbols) to zero-out
            entries for symbols not in the daily top-N movers.
    """

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        params: dict,
        config: dict | None = None,
        mover_mask: pd.DataFrame | None = None,
    ) -> None:
        self.ohlcv_df = ohlcv_df
        self.params = params
        self.mover_mask = mover_mask

        # Merge caller config with defaults
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self.start_cash = float(cfg["START_CASH"])
        self.max_position = float(cfg["PORTFOLIO_SHARE"])
        self.fees = float(cfg["FEES"])
        self.slippage = float(cfg["SLIPPAGE"])
        self.freq = cfg["FREQ"]
        self.n_jobs = int(cfg["N_JOBS"])

        self.pf = None

    def run(self, show_progress: bool = False) -> vbt.Portfolio:
        """Execute the vectorized backtest and return the VBT Portfolio."""
        if self.ohlcv_df.empty:
            raise ValueError(
                "OHLCV data is empty. Check your symbols, date range, "
                "and database connection."
            )

        # 0. Performance optimization: downcast to float32
        ohlcv = self.ohlcv_df.astype(np.float32)

        # 1. Unpack OHLCV
        close = ohlcv.xs("close", axis=1, level=1, drop_level=True)
        high = ohlcv.xs("high", axis=1, level=1, drop_level=True)
        low = ohlcv.xs("low", axis=1, level=1, drop_level=True)
        open_ = ohlcv.xs("open", axis=1, level=1, drop_level=True)

        # 2. Generate signals via SignalFactory (supports broadcasting)
        sf = SignalFactory.run(
            close=close,
            high=high,
            low=low,
            open_=open_,
            **self.params,
            param_product=True,
            show_progress=show_progress,
            n_jobs=self.n_jobs,
        )

        entries = sf.entries
        exits = sf.exits
        price_for_orders = sf.price_for_orders

        # 3. Apply dynamic mover mask if provided
        if self.mover_mask is not None:
            if isinstance(entries.columns, pd.MultiIndex):
                # SignalFactory MultiIndex: last level is the symbol name
                symbols = entries.columns.get_level_values(-1)
                # Build mask aligned to entries by mapping symbol → column
                aligned_mask = self.mover_mask.reindex(columns=symbols.unique())[
                    symbols.tolist()
                ]
                aligned_mask.columns = entries.columns
                aligned_mask = (
                    aligned_mask.reindex(index=entries.index, method="ffill")
                    .fillna(False)
                    .astype(bool)
                )
            else:
                # Flat columns: direct reindex
                aligned_mask = (
                    self.mover_mask.reindex(
                        index=entries.index, columns=entries.columns
                    )
                    .fillna(False)
                    .astype(bool)
                )
            entries = entries & aligned_mask

        # 4. Run VBT Portfolio with proper position sizing
        # Determine grouping:
        # - If MultiIndex (Grid Search), group by Params (drop symbol level)
        # - If Index (Single Run), group all into one (portfolio of symbols)
        if isinstance(entries.columns, pd.MultiIndex):
            # Assumes Symbol is the last level (vbt standard)
            # Group by all param levels, aggregating across symbols
            group_by = entries.columns.droplevel(-1)
        else:
            # Single run with multiple symbols -> 1 Portfolio
            n_cols = entries.shape[1]
            group_by = np.full(n_cols, 0)

        self.pf = vbt.Portfolio.from_signals(
            close=price_for_orders,
            entries=entries,
            exits=exits,
            init_cash=self.start_cash,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq,
            size=self.max_position,
            size_type="percent",
            cash_sharing=True,
            group_by=group_by,
        )
        return self.pf

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
        init_cash = self.start_cash
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

    def save_detailed_plots(
        self, results_manager, filename: str = "backtest_detailed"
    ) -> None:
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
