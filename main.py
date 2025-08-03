# woot

import yfinance as yf
from datetime import datetime, timedelta
from tabulate import tabulate

symbol = "SPY"
interval = "1h"
end_date = datetime(2025, 8, 2)
start_date = end_date - timedelta(days=3)

df = yf.download(symbol,
                 start=start_date,
                 end=end_date,
                 interval=interval,
                 progress=False,
                 multi_level_index=False,
                 auto_adjust=True)

print(tabulate(df, headers='keys', tablefmt='github'))