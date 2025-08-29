from datetime import datetime, timedelta, timezone

from utils.DataProvider import YFinanceProvider, KrakenProvider, HybridProvider

end = datetime.now(timezone.utc)
start_long = end - timedelta(days=365)   # backtest -> yfinance
start_short = end - timedelta(days=120)    # live -> kraken

# Direct providers
yf_dp = YFinanceProvider()
kr_dp = KrakenProvider()

df_yf = yf_dp.get_data("BTC-USD", "4h", start_long, end)
print("\nYF:", df_yf.shape, df_yf.index.min(), "→", df_yf.index.max())

df_kr = kr_dp.get_data("BTC-USD", "4h", start_short, end)
print("\nKR:", df_kr.shape, df_kr.index.min(), "→", df_kr.index.max())

# Hybrid/autoselect
auto_dp = HybridProvider(mode="auto")
df_auto = auto_dp.get_data("BTC-USD", "4h", start_short, end)
print("\nAUTO:", df_auto.shape, df_auto.index.min(), "-->", df_auto.index.max())