import yfinance as yf

data = yf.download("LINK-USD", period='3d', end="2025-03-01")

print(data)