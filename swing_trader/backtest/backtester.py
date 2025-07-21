def backtest_signals(df):
    trades = []
    position = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0
    for i, row in df.iterrows():
        if row['signal'] == 1 and position == 0:
            entry_price = row['close']
            position = 1
            trades.append({'action': 'buy', 'price': entry_price, 'date': row['date']})
        elif row['signal'] == -1 and position == 1:
            exit_price = row['close']
            position = 0
            trades.append({'action': 'sell', 'price': exit_price, 'date': row['date']})
    # Add PnL calculation etc.
    return trades
