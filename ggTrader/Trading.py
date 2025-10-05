from tabulate import tabulate

from ggTrader.Portfolio import Portfolio
from ggTrader.Position import Position
from ggTrader.Screener import Screener
from ggTrader.Signals import Signals
import pandas as pd
from utils.KrakenData import KrakenData


class Trading:

    def __init__(self, ohlcv_dict: dict[str, pd.DataFrame], date_range: pd.DatetimeIndex, start_cash=10000):
        self.portfolio = Portfolio(start_cash)
        self.ohlcv_data = ohlcv_dict
        self.time_range = date_range
        self.current_date = pd.Timestamp(date_range[0]).tz_convert('UTC')
        self.screener = Screener()
        self.top_n_movers = 25
        self.signals_dict = {}
        self.daily_movers = pd.DataFrame()
        self.max_position = 0.2

    def check_buy(self):
        # check signals
        # check signal by symbol and date
        #  self.check_buy_by_symbol_and_date(symbol, date)
        pass

    def check_buy_by_symbol_and_date(self, symbol: str, date: pd.Timestamp):
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
                position.update_price(self.ohlcv_data[position.symbol].loc[self.current_date])

                # check stop loss
                stop_loss_triggered = position.current_price <= position.stop_loss

                # check sell signal from strategy

                # Proceed to sell if conditions are met
                if stop_loss_triggered:
                    date = pd.Timestamp.today().tz_localize('UTC')
                    self.portfolio.close_position(position, date)
                else:

                    # update stop loss if new stop loss is higher
                    # stop_loss = max(stop_loss, new_stop_loss)
                    continue

    def check_stop_loss(self):
        pass

    def update_stats(self):
        pass

    def calc_signals(self,symbols: list[str]):
        signals = Signals()
        for symbol in symbols:
            if symbol not in self.signals_dict.keys():
                ohlcv = self.ohlcv_data[symbol]
                start = self.time_range[0]
                end = self.time_range[-1]
                # only calc signals for the specified range.
                self.signals_dict[symbol] = signals.compute(ohlcv.loc[start:end])



    def run(self):

        #
        for current_date in self.time_range:
            print(f"Running for {current_date}")
            self.current_date = current_date
            print(f"Checking for historical movers on {self.current_date}")
            self.daily_movers = self.screener.get_historical_daily_kraken_by_volume(self.current_date,
                                                                               top_n=self.top_n_movers)
            self.calc_signals(self.daily_movers['ticker'].tolist())
            # check signals --> save to signal
            # rank movers by volume/ ADX?

            self.check_sell()

            self.check_buy()

            self.update_stats()


if __name__ == "__main__":
    date_range = pd.date_range(start='2024-01-01', end='2024-01-31', freq='4h').tz_localize('UTC')
    kData = KrakenData()

    print(f"Loading OHLCV data for {date_range[0]} to {date_range[-1]}..")
    ohlcv_dict = kData.get_all_ohlcv_dict(interval="4h")
    print(f"Loaded {len(ohlcv_dict)} symbols.")
    trader = Trading(ohlcv_dict, date_range, start_cash=10000)
    trader.current_date = date_range[0]
    trader.check_buy()

    screen = Screener()
    date = pd.Timestamp(date_range[0]).tz_convert('UTC')

    top_movers = screen.get_historical_daily_kraken_by_volume(date, top_n=25)
    top_mover = top_movers['ticker'].iloc[0]
    print(top_mover)
    print(top_movers.shape)
    symbols = ['LTC']
    trader.calc_signals(symbols)
    print(tabulate(trader.signals_dict[symbols[0]].tail(10), headers='keys', tablefmt='github'))

    date = pd.Timestamp("2024-01-30 00:00:00+00:00")
    trader.check_buy_by_symbol_and_date(symbols[0], date)
    trader.portfolio.print_positions()
    trader.portfolio.print_trades()
    trader.portfolio.print_stats_df()
