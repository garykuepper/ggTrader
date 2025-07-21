import pandas as pd
from db.mongodb import get_collection

def sma_crossover(symbol, short_window=20, long_window=50):
    col = get_collection("stocks")
    data = list(col.find({"symbol": symbol}))
    df = pd.DataFrame(data).sort_values('date')
    df['sma_short'] = df['close'].rolling(window=short_window).mean()
    df['sma_long'] = df['close'].rolling(window=long_window).mean()
    # Generate signals
    df['signal'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
    df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1
    return df
