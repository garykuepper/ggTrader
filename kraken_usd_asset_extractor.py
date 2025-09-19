# Python 3.13+

from utils.kraken_yfinance_cmc import get_kraken_asset_pairs_usd
import pandas as pd

kraken_usd_pairs = get_kraken_asset_pairs_usd()
df = pd.DataFrame(kraken_usd_pairs)

print("BTC" in df["base_common"].values)
print(df["base_common"].eq("LTC").any())