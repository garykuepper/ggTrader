# src/ggTrader/core/FastBacktest.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from ggTrader.indicators.signals import Signals


class FastBacktest:
    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        params: dict,
        start_cash: float = 10000.0,
        fees: float = 0.001,
    ):
        self.ohlcv_df = ohlcv_df
        self.params = params
        self.start_cash = start_cash
        self.fees = fees
        self.pf = None

    def run(self):
        # 1. Unpack Data
        close = self.ohlcv_df.xs("close", axis=1, level=1, drop_level=True)
        high = self.ohlcv_df.xs("high", axis=1, level=1, drop_level=True)
        low = self.ohlcv_df.xs("low", axis=1, level=1, drop_level=True)
        open_ = self.ohlcv_df.xs("open", axis=1, level=1, drop_level=True)

        # 2. GET SIGNALS (The Golden Source)
        # Dynamically unpacks params so WFO can inject new values every loop
        entries, exits, _, price_for_orders = Signals.calc_signals(
            close=close, high=high, low=low, open_=open_, **self.params
        )

        # 3. RUN VECTORBT
        self.pf = vbt.Portfolio.from_signals(
            close=price_for_orders,
            entries=entries,
            exits=exits,
            init_cash=self.start_cash,
            fees=self.fees,
            slippage=0.0005,
            freq="4h",
        )
        return self.pf

    def get_stats(self) -> dict:
        """Returns metrics formatted EXACTLY like Portfolio.stats_dict()"""
        if self.pf is None:
            raise ValueError("Run the backtest first.")

        # Handle VectorBT's multi-column outputs by summing/averaging
        total_value = self.pf.final_value()
        if isinstance(total_value, pd.Series):
            total_value = total_value.sum()

        total_profit = self.pf.total_profit()
        if isinstance(total_profit, pd.Series):
            total_profit = total_profit.sum()

        # Calculate derived metrics
        init_cash = self.start_cash * (
            len(self.pf.wrapper.columns) if self.pf.wrapper.ndim > 1 else 1
        )
        profit_pct = (total_profit / init_cash) * 100

        return {
            "total_value": total_value,
            "total_profit": total_profit,
            "profit_pct": profit_pct,
            "total_trades": self.pf.trades.count().sum(),
            "win_rate": self.pf.trades.win_rate().mean() * 100,
            "sharpe": self.pf.sharpe_ratio().mean(),
            "sortino": self.pf.sortino_ratio().mean(),
            "max_drawdown": self.pf.max_drawdown().min() * 100,
        }
