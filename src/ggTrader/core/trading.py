"""Legacy bar-by-bar trading simulator (paper/backtest-style loop)."""

from __future__ import annotations

import pandas as pd

from ggTrader.core.portfolio import Portfolio
from ggTrader.core.position import Position
from ggTrader.core.screener import Screener
from ggTrader.indicators.signals import Signals


class Trading:
    """Main trading engine for the legacy iterative simulation."""

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        date_range: pd.DatetimeIndex,
        start_cash: float = 10000,
        top_n_movers: int = 25,
        max_position: float = 0.2,
        strategy_params: dict | None = None,
        use_movers: bool = False,
    ) -> None:
        self.portfolio = Portfolio(int(start_cash))
        self.ohlcv_df = ohlcv_df
        self.time_range = date_range
        self.use_movers = use_movers

        if len(date_range) == 0:
            raise ValueError(
                "Provided date_range is empty. Ensure data is loaded correctly for the requested "
                "symbols and date range."
            )

        self.current_date = pd.Timestamp(date_range[0]).tz_convert("UTC")
        self.screener = Screener()
        self.top_n_movers = top_n_movers
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

        self.signals_dict: dict[str, pd.DataFrame] = {}
        self.daily_movers = pd.DataFrame()

        self.all_movers_per_day: dict[pd.Timestamp, list[str]] = {}
        self.precalculated_signals: dict[str, pd.DataFrame] = {}

    def check_buy(self) -> None:
        if not self.daily_movers.empty:
            for symbol in self.daily_movers["symbol"].tolist():
                if symbol in self.signals_dict:
                    self.check_buy_by_symbol_and_date(symbol, self.current_date)

    def check_buy_by_symbol_and_date(self, symbol: str, date: pd.Timestamp) -> None:
        if symbol not in self.signals_dict:
            return
        row = self.signals_dict.get(symbol).loc[date]
        price = row["close"]

        if row["signal"] == 1:
            qty = self.portfolio.qty_to_buy(price, percent=self.max_position)
            if qty != 0.0:
                self.portfolio.add_position(Position(symbol, qty, price, date))

    def check_sell(self) -> None:
        if not self.portfolio.positions:
            return

        for position in list(self.portfolio.positions):
            symbol_data = self.ohlcv_df[position.symbol].loc[self.current_date]
            price = symbol_data["close"] if isinstance(symbol_data, pd.Series) else symbol_data.iloc[0]
            position.update_price(price)

            signal_row = self.signals_dict.get(position.symbol).loc[self.current_date]
            stop_loss = signal_row.get("stop_loss", position.stop_loss)
            position.stop_loss = max(position.stop_loss, stop_loss)

            stop_loss_triggered = position.current_price <= position.stop_loss
            sell_signal = signal_row.get("signal") == -1

            if stop_loss_triggered or sell_signal:
                self.portfolio.close_position(position, self.current_date)

    def update_stats(self) -> None:
        self.portfolio.record_equity(self.current_date)

    def prepare_simulation_data(self) -> None:
        all_unique_movers: set[str] = set()

        if self.use_movers:
            for date in self.time_range:
                daily = self.screener.get_historical_daily_kraken_by_volume(
                    date, top_n=self.top_n_movers
                )
                if not daily.empty:
                    syms = daily["symbol"].tolist()
                    self.all_movers_per_day[date] = syms
                    all_unique_movers.update(syms)
        else:
            all_symbols = self.ohlcv_df.columns.levels[0].tolist()
            all_unique_movers.update(all_symbols)
            for date in self.time_range:
                self.all_movers_per_day[date] = all_symbols

        symbols_list = sorted(all_unique_movers)
        available_symbols = [s for s in symbols_list if s in self.ohlcv_df.columns.levels[0]]
        if not available_symbols:
            return

        relevant_ohlcv = self.ohlcv_df[available_symbols]
        close = relevant_ohlcv.xs("close", axis=1, level=1, drop_level=True)
        high = relevant_ohlcv.xs("high", axis=1, level=1, drop_level=True)
        low = relevant_ohlcv.xs("low", axis=1, level=1, drop_level=True)
        open_ = relevant_ohlcv.xs("open", axis=1, level=1, drop_level=True)

        entries, exits, stop_df, _price_for_orders = Signals.calc_signals(
            close=close,
            high=high,
            low=low,
            open_=open_,
            **self.strategy_params,
        )

        self.precalculated_signals = {}
        for symbol in available_symbols:
            sig_df = pd.DataFrame(index=self.ohlcv_df.index)
            sig_df["close"] = close[symbol]
            sig_df["signal"] = 0
            sig_df.loc[entries[symbol], "signal"] = 1
            sig_df.loc[exits[symbol], "signal"] = -1
            sig_df["stop_loss"] = stop_df[symbol]
            self.precalculated_signals[symbol] = sig_df

    def run(self) -> None:
        if not self.all_movers_per_day or not self.precalculated_signals:
            self.prepare_simulation_data()

        for current_date in self.time_range:
            self.current_date = current_date
            movers = self.all_movers_per_day.get(current_date, [])
            self.daily_movers = pd.DataFrame({"symbol": movers})
            self.signals_dict = self.precalculated_signals

            self.check_sell()
            self.check_buy()
            self.update_stats()


if __name__ == "__main__":
    date_range = pd.date_range(start="2023-01-01", end="2023-01-05", freq="1d").tz_localize("UTC")
    screener = Screener()
    for date in date_range:
        print(f"\nHistorical Movers for {date}")
        screener.print_historical_daily_kraken_by_volume(date)

