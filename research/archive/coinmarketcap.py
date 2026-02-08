from coinmarketcapapi import CoinMarketCapAPI
import os
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate

load_dotenv()
cmc = CoinMarketCapAPI(api_key=os.getenv('CMC_API_KEY'))

rep = cmc.cryptocurrency_info(symbol='BTC')  # See methods below
thangs = cmc.cryptocurrency_listings_latest().data
thangs_df = pd.DataFrame.from_dict(thangs)
print(thangs_df.info())
cols = ['cmc_rank','id', 'name','symbol', 'slug']
print(tabulate(thangs_df[cols].head(10), headers='keys', tablefmt='github', showindex=False))

print(rep.credit_count)

