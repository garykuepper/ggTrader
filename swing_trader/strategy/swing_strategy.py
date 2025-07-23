import pandas as pd

class SwingStrategy:
    def __init__(self, db, signal_ticker, long_ticker, short_ticker):
        self.db = db
        self.signal_ticker = signal_ticker
        self.long_ticker = long_ticker
        self.short_ticker = short_ticker
        self.collection = db['stock_data']
        self.required_indicators = [
            "trend_ema_fast",
            "trend_ema_slow",
            "trend_macd",
            "trend_macd_signal"
        ]

    def generate_signals(self):
        df = self._load_signal_data()
        if df is None:
            return

        updated = 0

        for _, row in df.iterrows():
            signal, target = self._determine_signal(row)

            update_fields = {
                "strat_swing_signal": signal,
                "strat_swing_target": target
            }

            self.collection.update_one(
                {"_id": row["_id"]},
                {"$set": update_fields}
            )
            updated += 1

        print(f"✅ Swing signals generated: {updated} records updated for {self.signal_ticker}.")

    def _load_signal_data(self):
        docs = list(self.collection.find({"Ticker": self.signal_ticker}))
        if not docs:
            print(f"⚠️ No data found for {self.signal_ticker}")
            return None

        df = pd.DataFrame(docs).sort_values("Date").reset_index(drop=True)

        if not all(col in df.columns for col in self.required_indicators):
            print(f"⚠️ Missing required indicators. Run enrichment first.")
            return None

        return df

    def _determine_signal(self, row):
        ema_fast = row["trend_ema_fast"]
        ema_slow = row["trend_ema_slow"]
        macd = row["trend_macd"]
        macd_sig = row["trend_macd_signal"]

        if ema_fast > ema_slow and macd > macd_sig:
            return "BUY", self.long_ticker
        elif ema_fast < ema_slow and macd < macd_sig:
            return "SELL", self.short_ticker
        else:
            return "HOLD", None
