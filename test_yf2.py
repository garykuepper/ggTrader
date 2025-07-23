import yfinance as yf

print("String input:")
df1 = yf.download('SPY', start='2023-01-01', end='2023-01-10')
print(df1.columns)

print("\nList input:")
df2 = yf.download(['SPY'], start='2023-01-01', end='2023-01-10')
print(df2.columns)
