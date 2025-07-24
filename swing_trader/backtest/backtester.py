from swing_trader.data.mr_data import MrData
from tabulate import tabulate
from swing_trader.trading.portfolio import Portfolio


class Backtester:
    def __init__(self, db, tickers, strategy, portfolio,  start_date=None, end_date=None):
        self.filtered_signals = None
        self.bh_value = 0.0
        self.final_value = 0.0
        self.transactions = []
        self.db = db
        self.tickers = tickers
        self.strategy = strategy
        self.portfolio = portfolio
        self.initial_cash = self.portfolio.get_cash()
        self.start_date = start_date
        self.end_date = end_date
        self.mr_data = MrData()

    def prepare_data(self):
        for ticker in self.tickers:
            print(f"Downloading data for {ticker} from {self.start_date} to {self.end_date}...")
            self.mr_data.get_stock_data(ticker, self.start_date, self.end_date)
            print(f"Enriching {ticker} with indicators...")
            self.mr_data.enrich_ticker_with_indicator(ticker, self.mr_data.add_rsi, ['momentum_rsi'])
            self.mr_data.enrich_ticker_with_indicator(ticker, self.mr_data.add_macd, ['trend_macd', 'trend_macd_signal'])

    def run_strategy(self):
        print("Generating swing signals...")
        self.strategy.generate_signals()
        sigs = self.strategy.get_signals()
        self.filtered_signals = sigs[(sigs['Date'] >= self.start_date) & (sigs['Date'] <= self.end_date)]

    @staticmethod
    def add_transaction(action, ticker, date, qty, price):
        value = qty * price
        return [action, ticker, date, qty, f"${price:,.2f}", f"${value:,.2f}"]

    def update_portfolio_price(self, portfolio, ticker, date):
        price = self.mr_data.get_stock_price(ticker, date)
        portfolio.update_prices(ticker, price)

    def signal_transaction(self, portfolio, signal, ticker, date):
        price = self.mr_data.get_stock_price(ticker, date)
        if signal == "BUY":
            qty = portfolio.cash / price
            portfolio.add_position(ticker, qty, price)
            return self.add_transaction("BUY", ticker, date, qty, price)
        elif signal == "SELL" and ticker in portfolio.positions:
            qty = portfolio.positions[ticker]["quantity"]
            portfolio.remove_position(ticker, qty, price)
            return self.add_transaction("SELL", ticker, date, qty, price)
        return None

    def simulate_portfolios(self):
        print("Simulating portfolios...")
        bh_portfolio = Portfolio(cash=self.initial_cash, name="Buy and Hold Portfolio")
        spy_price = self.mr_data.get_stock_price('SPY', self.start_date)
        qty = bh_portfolio.cash / spy_price
        bh_portfolio.add_position('SPY', qty, spy_price)
        prev_signal = None
        transactions = []

        for _, row in self.filtered_signals.iterrows():
            cur_signal = row['strat_swing_signal']
            if prev_signal == cur_signal or cur_signal == 'HOLD':
                continue
            txn = self.signal_transaction(self.portfolio, cur_signal, 'SSO', row['Date'])
            if txn:
                transactions.append(txn)
            prev_signal = cur_signal

        self.update_portfolio_price(self.portfolio, 'SSO', self.end_date)
        self.update_portfolio_price(bh_portfolio, 'SPY', self.end_date)

        self.transactions = transactions
        self.final_value = self.portfolio.total_portfolio_value()
        self.bh_value = bh_portfolio.total_portfolio_value()

    def report(self):
        print(tabulate(
            self.transactions[-10:],
            headers=['Action', 'Ticker', 'Date', 'Quantity', 'Price', 'Value'],
            floatfmt=('.0f', '', '.2f', '.2f', '.2f'),
            tablefmt='github'
        ))
        performance = (self.final_value / self.bh_value - 1) * 100
        summary = [
            ["Final Portfolio Value", f"${self.final_value:,.2f}"],
            ["Buy and Hold Portfolio Value", f"${self.bh_value:,.2f}"],
            ["Relative Performance", f"{performance:.2f}%"]
        ]
        print("\nPerformance Summary:")
        print(tabulate(summary, tablefmt="github"))
