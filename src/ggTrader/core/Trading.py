from tabulate import tabulate

from ggTrader.core.Portfolio import Portfolio
from ggTrader.core.Position import Position
from ggTrader.core.Screener import Screener
from ggTrader.indicators import Signals
import pandas as pd
from ggTrader.data.KrakenData import KrakenData
from ggTrader.data.KrakenHistoricalData import KrakenHistoricalData


class Trading:

    def __init__(self, ohlcv_df: pd.DataFrame, date_range: pd.DatetimeIndex, start_cash=10000, 
                 top_n_movers=25, max_position=0.2, strategy_params=None):
        self.portfolio = Portfolio(start_cash)
        self.ohlcv_df = ohlcv_df
        self.time_range = date_range
        self.current_date = pd.Timestamp(date_range[0]).tz_convert('UTC')
        self.screener = Screener()
        self.top_n_movers = top_n_movers
        self.signals_dict = {}
        self.daily_movers = pd.DataFrame()
        self.max_position = max_position
        self.strategy_params = strategy_params or {
            'adx_threshold': 25,
            'adx_length': 14,
            'sar_acceleration': 0.02,
            'sar_maximum': 0.2,
            'atr_multiplier': 3.0,
            'atr_length': 14,
            'use_dmp_cross': False
        }

    def check_buy(self):
        # Only check movers for the current date
        if not self.daily_movers.empty:
            for symbol in self.daily_movers['symbol'].tolist():
                if symbol in self.signals_dict:
                    self.check_buy_by_symbol_and_date(symbol, self.current_date)

    def check_buy_by_symbol_and_date(self, symbol: str, date: pd.Timestamp):
        if symbol not in self.signals_dict:
            return
        row = self.signals_dict.get(symbol).loc[date]
        price = row['close']

        signal = row['signal']
        if signal == 1:
            qty = self.portfolio.qty_to_buy(price, percent=self.max_position)
            pos = Position(symbol, qty, price, date)
            if qty != 0.0:
                self.portfolio.add_position(pos)



    def check_sell(self):
        # check positions
        if self.portfolio.positions:
            for position in self.portfolio.positions:
                # update position price first
                # update position price first (ensure we only get the numeric price, usually 'close')
                symbol_data = self.ohlcv_df[position.symbol].loc[self.current_date]
                if isinstance(symbol_data, pd.Series):
                    price = symbol_data['close']
                else:
                    # If it's a DataFrame (shouldn't be, but safe handle), take first close
                    price = symbol_data['close'].iloc[0] if 'close' in symbol_data else symbol_data.iloc[0]
                
                position.update_price(price)

                # check stop loss from signal processing (trailing stop)
                signal_row = self.signals_dict.get(position.symbol).loc[self.current_date]
                stop_loss = signal_row.get('stop_loss', position.stop_loss)
                position.stop_loss = max(position.stop_loss, stop_loss)

                stop_loss_triggered = position.current_price <= position.stop_loss

                # check sell signal from strategy
                sell_signal = signal_row.get('signal') == -1

                # Proceed to sell if conditions are met
                if stop_loss_triggered or sell_signal:
                    self.portfolio.close_position(position, self.current_date)

    def update_stats(self):
        pass

    def calc_signals(self, symbols: list[str]):
        signals = Signals()
        for symbol in symbols:
            if symbol not in self.signals_dict:
                if symbol in self.ohlcv_df.columns.levels[0]:
                    # Keep multi-index levels for compatibility with vbt and signals
                    ohlcv = self.ohlcv_df.xs(symbol, axis=1, level=0, drop_level=False)
                    # We might need some buffer data for indicators, but for now we calc on full range
                    self.signals_dict[symbol] = signals._atr_trailing_stop_long_ohlc_touch_2d(
                        ohlcv, **self.strategy_params)



    def run(self):
        print("DEBUG: Trading.run started")
        #
        for current_date in self.time_range:
            print(f"Running for {current_date}")
            self.current_date = current_date
            print(f"DEBUG: Checking movers for {self.current_date}")
            self.daily_movers = self.screener.get_historical_daily_kraken_by_volume(self.current_date,
                                                                               top_n=self.top_n_movers)
            print(f"DEBUG: Found {len(self.daily_movers)} movers")
            
            print(f"DEBUG: Calculating signals for movers")
            self.calc_signals(self.daily_movers['symbol'].tolist())
            print(f"DEBUG: signals calculated")

            # rank movers by volume/ ADX?

            print(f"DEBUG: Checking sell conditions")
            self.check_sell()
            print(f"DEBUG: check_sell done")

            print(f"DEBUG: Checking buy conditions")
            self.check_buy()
            print(f"DEBUG: check_buy done")

            self.update_stats()


if __name__ == "__main__":
    date_range = pd.date_range(start='2023-01-01', end='2023-01-05', freq='1d').tz_localize('UTC')
    k = KrakenData()
    k_h = KrakenHistoricalData()

    for date in date_range:
        print(f"\nHistorical Movers for {date}")
        historical_movers_by_day = k_h.get_historical_movers_by_day(date)

        print(tabulate(historical_movers_by_day.head(10), headers="keys", tablefmt="github"))