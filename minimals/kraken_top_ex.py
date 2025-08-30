from utils.kraken_yfinance_cmc import get_top_kraken_usd_pairs

# Get only YF-compatible pairs, sorted by Kraken volume
df_yf = get_top_kraken_usd_pairs(top_n=30, require_yf=True)

# Select the top 10 tickers
symbols = df_yf["YF Ticker"].head(10).tolist()

print(symbols)
