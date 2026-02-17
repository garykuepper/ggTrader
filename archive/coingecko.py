import os

from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate
load_dotenv()
cg = CoinGeckoAPI(demo_api_key=os.getenv('COIN_GECKO'))

price = cg.get_price(ids='bitcoin', vs_currencies='usd')
exchanges = cg.get_exchanges_list()
exchanges_df = pd.DataFrame(exchanges)
print(exchanges_df.info())
col = ['id', 'name', 'year_established', 'trade_volume_24h_btc','trust_score_rank','country']
print(tabulate(exchanges_df[col].head(10), headers='keys', tablefmt='github'))

print(kraken_volume)