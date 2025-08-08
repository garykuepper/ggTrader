from tabulate import tabulate

import pandas as pd

from portfolio import Portfolio, Position



class BacktestSimulator:
    def __init__(self, symbol_strategy_map, price_data_map, position_pct_map, initial_cash=10000):
        self.symbol_strategy_map = symbol_strategy_map
        self.price_data_map = price_data_map
        self.position_pct_map = position_pct_map
        self.portfolio = Portfolio(cash=initial_cash)
        self.trades = []

    def run(self):
        all_dates = set()
        for df in self.price_data_map.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)

        # Precompute signals once per symbol
        for symbol, strategy in self.symbol_strategy_map.items():
            df = self.price_data_map[symbol]
            if not hasattr(strategy, 'signal_df') or strategy.signal_df.empty:
                strategy.find_crossovers(df)

        for date in all_dates:
            for symbol, strategy in self.symbol_strategy_map.items():
                df = self.price_data_map[symbol]
                if date not in df.index:
                    continue
                row = df.loc[date]

                # Get signal safely
                try:
                    srow = strategy.signal_df.loc[date]
                    signal = srow.get('signal', 0)
                    if pd.isna(signal):
                        signal = 0
                except KeyError:
                    signal = 0

                # Update current position first (price + trailing stop tracking)
                if self.portfolio.position_exists(symbol):
                    self.portfolio.update_position(symbol, current_price=row['close'])
                    position = self.portfolio.get_position_by_symbol(symbol)
                    if position.trailing_stop and position.trailing_stop.is_triggered(row['close']):
                        qty_to_report = position.qty
                        print(f"{date} {symbol} TRAILING STOP SELL @ {row['close']:.2f} "
                              f"(stop: {position.trailing_stop.stop_price:.2f}) total: {position.current_value:.2f}")
                        self.portfolio.exit_position_by_symbol(symbol)
                        self.trades.append({
                            'date': date, 'symbol': symbol, 'action': 'trailing_stop_sell',
                            'price': row['close'], 'qty': qty_to_report
                        })
                        continue

                # Position sizing
                position_pct = self.position_pct_map.get(symbol, 0.10)

                # Entry only if flat
                if signal == 1 and not self.portfolio.position_exists(symbol):
                    price = float(row['close'])
                    if price <= 0:
                        continue
                    cash_to_use = self.portfolio.cash * position_pct
                    qty = cash_to_use / price
                    if qty <= 0:
                        continue
                    pos = Position(symbol, qty, price, date, trailing_pct=strategy.trailing_pct)
                    if self.portfolio.enter_position(pos):
                        total = qty * price
                        print(f"{date} {symbol} BUY  @ {price:.2f} total: {total:.2f}")
                        self.trades.append({'date': date, 'symbol': symbol, 'action': 'buy', 'price': price, 'qty': qty})
                    else:
                        print(f"{date} {symbol} BUY signal but insufficient cash.")

                # Exit if signaled
                elif signal == -1:
                    position = self.portfolio.get_position_by_symbol(symbol)
                    if position:
                        price = float(row['close'])
                        qty_to_report = position.qty
                        total = position.current_value
                        print(f"{date} {symbol} SELL @ {price:.2f} total: {total:.2f}")
                        self.portfolio.exit_position_by_symbol(symbol)
                        self.trades.append({'date': date, 'symbol': symbol, 'action': 'sell', 'price': price, 'qty': qty_to_report})
                    else:
                        # no position to sell; skip
                        pass

    def print_trade_history(self):
        print(tabulate(self.trades, headers="keys", tablefmt="github"))
        print(f"Ending cash: {self.portfolio.cash:.2f} | "
              f"Positions value: {self.portfolio.total_positions_value():.2f} | "
              f"Total equity: {self.portfolio.total_equity():.2f}")