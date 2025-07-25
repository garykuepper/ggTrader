import pandas as pd


class SwingStrategy:
    def __init__(self, db, signal_ticker, long_ticker, short_ticker, rsi_high=70, rsi_low=30):
        self.db = db
        self.signal_ticker = signal_ticker
        self.long_ticker = long_ticker
        self.short_ticker = short_ticker
        self.collection = db['stock_data']
        self.required_indicators = [
            "trend_ema_fast",
            "trend_ema_slow",
            "trend_macd",
            "trend_macd_signal",
            "momentum_rsi",
        ]
        self.rsi_high = rsi_high
        self.rsi_low = rsi_low

    def generate_signals(self):
        df = self._load_signal_data()
        if df is None:
            return
        df['signal_score'] = 0
        df['signal_score'] += (df['trend_ema_fast'] > df['trend_ema_slow']).astype(int)
        df['signal_score'] += (df['trend_macd'] > df['trend_macd_signal']).astype(int)
        df['signal_score'] += df['momentum_rsi'].between(50, self.rsi_high).astype(int)

        # Compute signal conditions vectorized
        buy_mask = (df['trend_ema_fast'] > df['trend_ema_slow']) & (df['trend_macd'] > df['trend_macd_signal']) & (
            df['momentum_rsi'].between(50, self.rsi_high))

        # === Exit Signal (new) ===
        df['exit_signal'] = (
                (df['trend_macd'] < df['trend_macd_signal']) |
                (df['momentum_rsi'] < 50))

        sell_mask = ((df['trend_ema_fast'] < df['trend_ema_slow']) & (df['trend_macd'] < df['trend_macd_signal']) & (
                df['momentum_rsi'] < self.rsi_low))
        df['strat_swing_signal'] = 'HOLD'
        df.loc[buy_mask, 'strat_swing_signal'] = 'BUY'
        df.loc[sell_mask, 'strat_swing_signal'] = 'SELL'
        df['strat_swing_target'] = None
        df.loc[buy_mask, 'strat_swing_target'] = self.long_ticker
        df.loc[sell_mask, 'strat_swing_target'] = self.short_ticker

        updated = 0
        for _, row in df.iterrows():
            self.collection.update_one(
                {"_id": row["_id"]},
                {"$set": {
                    "strat_swing_signal": row['strat_swing_signal'],
                    "strat_swing_target": row['strat_swing_target']
                }}
            )
            updated += 1

        print(f"✅ Swing signals generated: {updated} records updated for {self.signal_ticker}.")

    def _load_signal_data(self):
        docs = list(self.collection.find({"Ticker": self.signal_ticker}))
        if not docs:
            print(f"⚠️ No data found for {self.signal_ticker}")
            return None

        df = pd.DataFrame(docs).sort_values("Date").reset_index(drop=True)
        # print(df.columns)  # Should include 'momentum_rsi', 'trend_macd', etc.
        if not all(col in df.columns for col in self.required_indicators):
            print(f"⚠️ Missing required indicators. Run enrichment first.")
            return None

        return df

    def get_signals(self):
        df = self._load_signal_data()
        if df is None:
            return pd.DataFrame(columns=["Ticker", "Date", "strat_swing_signal", "strat_swing_target"])

        return df[["Ticker", "Date", "strat_swing_signal", "strat_swing_target"]].reset_index(drop=True)

    def get_long_ticker(self):
        return self.long_ticker

    def get_short_ticker(self):
        return self.short_ticker
