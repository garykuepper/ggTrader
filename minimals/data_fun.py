import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from tabulate import tabulate


def get_sample_data(symbol: str):
    df = yf.download(symbol, period='7d', interval='4h', auto_adjust=True, progress=False, multi_level_index=False)
    return df


# end_date = datetime.now(timezone.utc)
# start_date = end_date - timedelta(days=30)

symbol = 'BTC-USD'
df = get_sample_data(symbol)

print(tabulate(df, headers='keys', tablefmt='github'))
print(df.dtypes)
print(df.index.dtype)


