import yfinance as yf
from tabulate import tabulate


ticker = ['BTC-USD']
period = '1y'
interval = '1d'
df = yf.download(tickers=ticker, period=period, interval=interval)

print(tabulate(df, headers='keys', tablefmt='github'))


