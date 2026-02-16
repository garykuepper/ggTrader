from tabulate import tabulate
import pandas as pd

from ggTrader.core.portfolio import Portfolio
from ggTrader.core.position import Position
from ggTrader.core.screener import Screener
from ggTrader.indicators.signals import Signals
from ggTrader.data.kraken.data_manager import KrakenData
from ggTrader.data.kraken.historical_data import KrakenHistoricalData


class Trading:
    """
    Main trading engine for simulation and paper trading.
    """

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        start_cash: float = 10000,
        top_n_movers: int = 25,
        max_position: float = 0.2,
        strategy_params: dict = None,
    ):
        self.portfolio = Portfolio(start_cash)
        self.ohlcv_df = ohlcv_df
        self.time_range = date_range
        if len(date_range) == 0:
            raise ValueError(
                "Provided date_range is empty. Ensure data is loaded correctly "
                "for the requested symbols and date range."
            )
        self.current_date = pd.Timestamp(date_range[0]).tz_convert("UTC")
        self.screener = Screener()
        self.top_n_movers = top_n_movers
        self.signals_dict = {}
        self.daily_movers = pd.DataFrame()
        self.max_position = max_position
        self.strategy_params = strategy_params or {
            "adx_threshold": 25,
            "adx_length": 14,
            "sar_acceleration": 0.02,
            "sar_maximum": 0.2,
            "atr_multiplier": 3.0,
            "atr_length": 14,
            "use_dmp_cross": False,
        }
        self.all_movers_per_day = {}
        self.precalculated_signals = {}
        self.bulk_entries = pd.DataFrame()
        self.bulk_exits = pd.DataFrame()
        self.bulk_prices = pd.DataFrame()

    def check_buy(self):
        """
        Check for buy signals for the current date's top movers.
        """
        # Only check movers for the current date
        if not self.daily_movers.empty:
            for symbol in self.daily_movers["symbol"].tolist():
                if symbol in self.signals_dict:
                    self.check_buy_by_symbol_and_date(symbol, self.current_date)

    def check_buy_by_symbol_and_date(self, symbol: str, date: pd.Timestamp):
        """
        Execute buy logic for a specific symbol on a specific date.
        """
        if symbol not in self.signals_dict:
            return
        row = self.signals_dict.get(symbol).loc[date]
        price = row["close"]

        signal = row["signal"]
        if signal == 1:
            qty = self.portfolio.qty_to_buy(price, percent=self.max_position)
            pos = Position(symbol, qty, price, date)
            if qty != 0.0:
                self.portfolio.add_position(pos)

    def check_sell(self):
        """
        Check existing positions for exit signals or stop loss triggers.
        """
        # check positions
        if self.portfolio.positions:
            for position in self.portfolio.positions:
                # update position price first
                # update position price first (ensure we only get the numeric price, usually 'close')
                symbol_data = self.ohlcv_df[position.symbol].loc[self.current_date]
                if isinstance(symbol_data, pd.Series):
                    price = symbol_data["close"]
                else:
                    # If it's a DataFrame (shouldn't be, but safe handle), take first close
                    price = (
                        symbol_data["close"].iloc[0]
                        if "close" in symbol_data
                        else symbol_data.iloc[0]
                    )

                position.update_price(price)

                # check stop loss from signal processing (trailing stop)
                signal_row = self.signals_dict.get(position.symbol).loc[
                    self.current_date
                ]
                stop_loss = signal_row.get("stop_loss", position.stop_loss)
                position.stop_loss = max(position.stop_loss, stop_loss)

                stop_loss_triggered = position.current_price <= position.stop_loss

                # check sell signal from strategy
                sell_signal = signal_row.get("signal") == -1

                # Proceed to sell if conditions are met
                if stop_loss_triggered or sell_signal:
                    self.portfolio.close_position(position, self.current_date)

    def update_stats(self):
        """Record equity for the current step."""
        self.portfolio.record_equity(self.current_date)

    def calc_signals(self, symbols: list[str]):
        """
        Calculate signals for the given symbols using the strategy parameters.
        """
        for symbol in symbols:
            if symbol not in self.signals_dict:
                if symbol in self.ohlcv_df.columns.levels[0]:
                    # Keep multi-index levels for compatibility with vbt and signals
                    ohlcv = self.ohlcv_df.xs(symbol, axis=1, level=0, drop_level=False)
                    # We might need some buffer data for indicators, but for now we calc on full range
                    self.signals_dict[symbol] = Signals.calculate_ohlcv_signals(
                        ohlcv, **self.strategy_params
                    )

    def prepare_simulation_data(self):
        """
        Pre-calculates all movers and all signals for the entire period to speed up the run loop.
        """
        # print(f"DEBUG: Pre-calculating simulation data for {len(self.time_range)} days...")
        all_unique_movers = set()

        # 1. Pre-calculate all movers for the period
        for date in self.time_range:
            daily = self.screener.get_historical_daily_kraken_by_volume(
                date, top_n=self.top_n_movers
            )
            if not daily.empty:
                syms = daily["symbol"].tolist()
                self.all_movers_per_day[date] = syms
                all_unique_movers.update(syms)

        # 2. Pre-calculate all signals in bulk
        # print(f"DEBUG: Calculating bulk signals for {len(all_unique_movers)} unique symbols...")
        # Get all relevant OHLCV data once
        symbols_list = sorted(list(all_unique_movers))
        # Filter for symbols actually in ohlcv_df
        available_symbols = [
            s for s in symbols_list if s in self.ohlcv_df.columns.levels[0]
        ]

        if available_symbols:
            relevant_ohlcv = self.ohlcv_df[available_symbols]

            # Use the internal calc_signals to get the wide DataFrames for plotting
            close = relevant_ohlcv.xs("close", axis=1, level=1, drop_level=True)
            high = relevant_ohlcv.xs("high", axis=1, level=1, drop_level=True)
            low = relevant_ohlcv.xs("low", axis=1, level=1, drop_level=True)
            open_ = relevant_ohlcv.xs("open", axis=1, level=1, drop_level=True)

            (
                entries,
                exits,
                stop_df,
                price_for_orders,
            ) = Signals.calc_signals(
                close=close,
                high=high,
                low=low,
                open_=open_,
                **self.strategy_params,
            )

            self.bulk_entries = entries
            self.bulk_exits = exits
            self.bulk_prices = price_for_orders

            # Still populate the dict for the current simulation loop logic
            self.precalculated_signals = {}
            for symbol in available_symbols:
                sig_df = pd.DataFrame(index=self.ohlcv_df.index)
                sig_df["close"] = close[symbol]
                sig_df["signal"] = 0
                sig_df.loc[entries[symbol], "signal"] = 1
                sig_df.loc[exits[symbol], "signal"] = -1
                sig_df["stop_loss"] = stop_df[symbol]
                self.precalculated_signals[symbol] = sig_df

            # print("DEBUG: Bulk signal calculation complete.")

    def run(self):
        """Execute the simulation."""
        # print("DEBUG: Trading.run started (optimized)")
        # Pre-calculate data if not already done
        if not self.all_movers_per_day or not self.precalculated_signals:
            self.prepare_simulation_data()

        for current_date in self.time_range:
            self.current_date = current_date

            # Use pre-calculated movers
            movers = self.all_movers_per_day.get(current_date, [])
            self.daily_movers = pd.DataFrame({"symbol": movers})

            # Use pre-calculated signals
            self.signals_dict = self.precalculated_signals

            self.check_sell()
            self.check_buy()
            self.update_stats()


if __name__ == "__main__":
    date_range = pd.date_range(
        start="2023-01-01", end="2023-01-05", freq="1d"
    ).tz_localize("UTC")
    k = KrakenData()
    k_h = KrakenHistoricalData()

    for date in date_range:
        print(f"\nHistorical Movers for {date}")
        historical_movers_by_day = k_h.get_historical_movers_by_day(date)

        print(
            tabulate(
                historical_movers_by_day.head(10), headers="keys", tablefmt="github"
            )
        )
