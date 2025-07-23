import pandas as pd

class Backtester:
    def __init__(self, db, signal_ticker, signal_field="strat_swing_signal", target_field="strat_swing_target", initial_cash=10000):
        self.db = db
        self.signal_ticker = signal_ticker
        self.signal_field = signal_field
        self.target_field = target_field
        self.initial_cash = initial_cash
        self.collection = db["stock_data"]
        self.prices = {}


    # TODO: THis whole logic needs to be cleaned up like wtf

    def run(self, start_date, end_date, benchmark_ticker="SPY"):
        signal_df = self._load_signals(start_date, end_date)

        # Shows unique tickers in the signal DataFrame
        # TODO: Load tickets into database if not present

        tickers = signal_df[self.target_field].dropna().unique().tolist()

        tickers += [self.signal_ticker, benchmark_ticker]  # Ensure benchmark & signal ticker are included

        self._load_price_data(tickers, start_date, end_date)
        result_df = self._simulate_trades(signal_df)
        benchmark_df = self._get_benchmark(benchmark_ticker, start_date, end_date)
        if benchmark_df is not None:
            result_df = result_df.merge(benchmark_df, on="Date", how="left")
            self._print_comparison(result_df)
        return result_df

    def _load_signals(self, start_date, end_date):
        cursor = self.collection.find({
            "Ticker": self.signal_ticker,
            "Date": {"$gte": start_date.strftime('%Y-%m-%d'), "$lte": end_date.strftime('%Y-%m-%d')},
            self.signal_field: {"$exists": True}
        })

        docs = list(cursor)
        if not docs:
            print(f"⚠️ No signal documents found for {self.signal_ticker} between {start_date} and {end_date}")
            return pd.DataFrame()  # Empty DataFrame to prevent crash

        df = pd.DataFrame(docs)

        if "Date" not in df.columns:
            print("❌ 'Date' column missing from signal DataFrame")
            print("Sample document:", docs[0] if docs else "None")
            return pd.DataFrame()

        return df.sort_values("Date").reset_index(drop=True)


    def _load_price_data(self, tickers, start_date, end_date):

        # TODO: Probably a better way to do this
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        for ticker in set(tickers):
            cursor = self.collection.find({
                "Ticker": ticker,
                "Date": {
                    "$gte": start_str,
                    "$lte": end_str
                }
            })
            data = {doc["Date"]: doc["Close"] for doc in cursor if "Date" in doc and "Close" in doc}
            if not data:
                print(f"⚠️ No price data found for {ticker}")
            self.prices[ticker] = data


    def _get_price(self, ticker, date):
        return self.prices.get(ticker, {}).get(date)

    def _simulate_trades(self, signal_df):
        cash = self.initial_cash
        position = None
        history = []

        for _, row in signal_df.iterrows():
            date = row["Date"]
            signal = row[self.signal_field]
            target = row.get(self.target_field)

            # TODO: Logic needs cleaning up

            if signal in ["BUY", "SELL"] and target:
                cash, position = self._open_position(cash, position, target, date, signal)

            elif signal == "HOLD":
                cash, position = self._close_position(cash, position, date)

            portfolio_value = self._get_position_value(position, date)
            total_value = cash + portfolio_value

            history.append({
                "Date": date,
                "Cash": cash,
                "Position": position["ticker"] if position else None,
                "Shares": position["shares"] if position else 0,
                "Portfolio": portfolio_value,
                "Total": total_value
            })

        df = pd.DataFrame(history)
        self._print_summary(df)
        return df

    def _open_position(self, cash, position, ticker, date, direction):
        price = self._get_price(ticker, date)
        if price is None:
            return cash, position

        # Close any current position
        if position:
            cash, _ = self._close_position(cash, position, date)

        shares = int(cash // price)
        if shares == 0:
            return cash, None

        cash -= shares * price
        position = {"ticker": ticker, "shares": shares, "direction": direction}
        return cash, position

    def _close_position(self, cash, position, date):
        if not position:
            return cash, None

        exit_price = self._get_price(position["ticker"], date)
        if exit_price:
            value = position["shares"] * exit_price
            # Assume inverse ETF is short (profit when rising); just treat as value for now
            cash += value
        return cash, None

    def _get_position_value(self, position, date):
        if not position:
            return 0
        price = self._get_price(position["ticker"], date)
        return price * position["shares"] if price else 0

    def _get_benchmark(self, benchmark_ticker, start_date, end_date):
        cursor = self.collection.find({
            "Ticker": benchmark_ticker,
            "Date": {
                "$gte": start_date.strftime('%Y-%m-%d'),
                "$lte": end_date.strftime('%Y-%m-%d')
            }

        })
        price_map = {doc["Date"]: doc["Close"] for doc in cursor}
        dates = sorted(price_map.keys())
        if len(dates) < 2:
            print("Not enough benchmark data.")
            return None

        start_price = price_map[dates[0]]
        benchmark_curve = [
            {"Date": date, "Benchmark": self.initial_cash * (price_map[date] / start_price)}
            for date in dates
        ]
        return pd.DataFrame(benchmark_curve)

    def _print_summary(self, df):
        total_return = (df["Total"].iloc[-1] / self.initial_cash - 1) * 100
        print(f"\n✅ Strategy Backtest Summary")
        print(f"Final Value: ${df['Total'].iloc[-1]:,.2f}")
        print(f"Total Return: {total_return:.2f}%")

    def _print_comparison(self, df):
        strat_end = df["Total"].iloc[-1]
        bench_end = df["Benchmark"].iloc[-1]

        strat_ret = (strat_end / self.initial_cash - 1) * 100
        bench_ret = (bench_end / self.initial_cash - 1) * 100
        delta = strat_ret - bench_ret

        print(f"\n📊 Strategy vs SP500 (Buy & Hold):")
        print(f"Strategy Final:  ${strat_end:,.2f} ({strat_ret:.2f}%)")
        print(f"SP500 Final:     ${bench_end:,.2f} ({bench_ret:.2f}%)")
        print(f"Difference:      {delta:.2f}%")

    def plot_equity_curve(self, df):
        import matplotlib.pyplot as plt

        # TODO: Plot buy and sell signals

        if "Benchmark" not in df.columns:
            print("⚠️ No Benchmark data found — plotting strategy only.")
            df = df.dropna(subset=["Total"])
            plt.plot(df["Date"], df["Total"], label="Strategy")
        else:
            df = df.dropna(subset=["Total", "Benchmark"])
            plt.plot(df["Date"], df["Total"], label="Strategy")
            plt.plot(df["Date"], df["Benchmark"], label="Benchmark (Buy & Hold)")

        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.title("Equity Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
