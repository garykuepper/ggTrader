from utils.kraken_yfinance_cmc import get_top_kraken_by_volume
from tabulate import tabulate
df = get_top_kraken_by_volume(top_n=20)

print(tabulate(df, headers="keys", tablefmt="github"))