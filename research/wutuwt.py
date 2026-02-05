import yfinance as yf

symbol = ['BTC-USD','ETH-USD']
interval = '1d'
df = yf.download(symbol, interval=interval)

print(df.head())
print(df.info())
df_btc = df.xs('BTC-USD',axis=1,level='Ticker')
print(df_btc.head())

df_btc.columns = df_btc.columns.str.lower()

print(df_btc)