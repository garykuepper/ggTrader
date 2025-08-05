import pandas as pd
from ta.trend import EMAIndicator

class EmaTrailingStrategy:
    def __init__(self, df, fast_window=5, slow_window=14, trailing_pct=None, min_hold_bars=0, starting_cash=1000):
        self.df = df.copy()
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.trailing_pct = trailing_pct
        self.min_hold_bars = min_hold_bars
        self.starting_cash = starting_cash

    def calculate_indicators(self):
        self.df['ema_fast'] = EMAIndicator(self.df['close'], window=self.fast_window).ema_indicator()
        self.df['ema_slow'] = EMAIndicator(self.df['close'], window=self.slow_window).ema_indicator()

    def generate_signals(self):
        fast_above = self.df['ema_fast'] > self.df['ema_slow']
        shifted = fast_above.shift(1).fillna(False).astype(bool)
        cross_up = fast_above & (~shifted)
        cross_down = (~fast_above) & shifted
        return cross_up, cross_down

    def simulate_trades(self, cross_up, cross_down):
        buy_signals = self.df.loc[cross_up, ['close']].reset_index().rename(columns={'date': 'buy_date', 'close': 'buy_price'})
        sell_signals = self.df.loc[cross_down, ['close']].reset_index().rename(columns={'date': 'sell_date', 'close': 'sell_price'})

        trades = pd.merge_asof(
            buy_signals.sort_values('buy_date'),
            sell_signals.sort_values('sell_date'),
            left_on='buy_date', right_on='sell_date',
            direction='forward'
        ).dropna()

        if trades.empty:
            return pd.DataFrame()

        actual_trades = []
        cash = self.starting_cash

        for _, trade in trades.iterrows():
            buy_date, buy_price = trade['buy_date'], trade['buy_price']
            planned_sell_date, planned_sell_price = trade['sell_date'], trade['sell_price']

            df_slice = self.df.loc[buy_date:planned_sell_date]
            highest_price = buy_price
            trailing_stop_price = highest_price * (1 - self.trailing_pct) if self.trailing_pct else None
            triggered = False
            actual_sell_price = planned_sell_price
            actual_sell_date = planned_sell_date

            if self.trailing_pct is not None:
                for i, (dt, row) in enumerate(df_slice.iterrows()):
                    if i < self.min_hold_bars:
                        continue
                    price_high, price_low = row['high'], row['low']

                    if price_high > highest_price:
                        highest_price = price_high
                        trailing_stop_price = highest_price * (1 - self.trailing_pct)

                    if price_low <= trailing_stop_price:
                        actual_sell_price = row['open']
                        actual_sell_date = dt
                        triggered = True
                        break

            shares_bought = cash / buy_price
            sell_value = shares_bought * actual_sell_price
            profit = sell_value - cash
            cash = sell_value

            actual_trades.append({
                'buy_date': buy_date,
                'buy_price': buy_price,
                'sell_date': actual_sell_date,
                'sell_price': actual_sell_price,
                'trailing_stop_triggered': triggered,
                'buy_value': cash - profit,
                'sell_value': sell_value,
                'profit': profit,
            })

        trades_df = pd.DataFrame(actual_trades).sort_values('buy_date').reset_index(drop=True)
        trades_df['cum_profit'] = trades_df['profit'].cumsum()
        return trades_df

    def run(self):
        self.calculate_indicators()
        cross_up, cross_down = self.generate_signals()
        trades = self.simulate_trades(cross_up, cross_down)
        return trades, self.df  # return updated df with indicators

