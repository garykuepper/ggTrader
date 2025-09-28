from utils.kraken_yfinance_cmc import get_top_kraken_usd_pairs_by_volume


df = get_top_kraken_usd_pairs_by_volume(top_n=30, exclude_stables=True)
print(df)