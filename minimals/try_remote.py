from utils.KrakenHistoricalData import KrakenHistoricalData

k = KrakenHistoricalData()

k.use_remote("https://garygigabytes.com/kraken/parquet")  # no trailing slash
df = k.read_parquet_remote(pair="BTC-USD", interval="1d")
print(df.head())

df_multi = k.get_ohlcv_df_remote(symbols=["BTC", "ETH", "AAVE"], interval="1d", quote="USD")

print(df_multi.head(10))