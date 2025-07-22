import pandas as pd
from swing_trader.db.mongodb import MongoDBClient
from swing_trader.strategy.strategy import Strategy

class SwingStrategy(Strategy):
    def __init__(self, symbol, short_window=20, long_window=50):
        super().__init__(symbol)
        self.short_window = short_window
        self.long_window = long_window
        self.db_client = MongoDBClient()

    def generate_signals(self):
        # 1. Use correct collection name
        col = self.db_client.get_collection("stock_data")
        data = list(col.find({"symbol": self.symbol}))
        if not data:
            print(f"No stock data found in DB for symbol: {self.symbol}")
            raise KeyError("No data found for symbol in MongoDB.")

        df = pd.DataFrame(data)
        # 2. Lowercase all column names
        df.columns = [str(col).lower() for col in df.columns]

        # 3. Ensure 'date' and 'close' columns exist
        if 'date' not in df.columns:
            print("Available columns:", df.columns)
            raise KeyError("No 'date' column found in data.")
        if 'close' not in df.columns:
            print("Available columns:", df.columns)
            raise KeyError("No 'close' column found in data.")

        # 4. Sort and process
        df = df.sort_values('date')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
        df['signal'] = 0
        df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
        df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1
        return df

    def insert_signals(self, symbol, signals_df, short_window, long_window):
        collection = self.db_client.get_collection('signals')
        records = signals_df.to_dict('records')
        for record in records:
            record['symbol'] = symbol
            record['short_window'] = short_window
            record['long_window'] = long_window
        if records:
            collection.insert_many(records)
