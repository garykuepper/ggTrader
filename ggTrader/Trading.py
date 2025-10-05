from ggTrader.Portfolio import Portfolio
import pandas as pd
from utils.KrakenData import KrakenData

class Trading:

    def __init__(self, ohlcv_dict: dict, date_range, start_cash=10000):
        self.portfolio = Portfolio(start_cash)
        self.ohlcv_data = ohlcv_dict
        self.time_range = date_range


    def check_buy(self):
        # check signals
        # check signal by symbol and date
        #  self.check_buy_by_symbol_and_date(symbol, date)
        pass

    def check_buy_by_symbol_and_date(self, symbol: str, date: pd.Timestamp):
        pass

    def check_sell(self):
        # check positions
        if self.portfolio.positions:
            for position in self.portfolio.positions:
                # update position price first

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

    def run(self):

        self.check_sell()

        self.check_buy()

if __name__ == "__main__":

    date_range = pd.date_range(start='2023-01-01', end='2023-01-31', freq='4h')
    print(type(date_range))
    kData = KrakenData()
    ohlcv_dict = kData.get_all_kraken_historical_csv(interval="4h")
    trader = Trading(ohlcv_dict, date_range, start_cash=10000)

