import yfinance as yf
import pandas as pd
import importlib

# Helper to check if a module is available without importing it
def is_pyarrow_available():
    spec = importlib.util.find_spec("pyarrow")
    return spec is not None

symbols = ["BTC-USD", "ETH-USD"]
data = yf.download(symbols, period="1y", interval="1d", group_by='ticker', auto_adjust=True)
# Convert to a MultiIndex DataFrame
df = pd.concat([data[symbol] for symbol in symbols], axis=1, keys=symbols)
print(df.head())

if is_pyarrow_available():
    df.to_parquet("crypto_data.parquet")
    df = pd.read_parquet("crypto_data.parquet")
else:
    print("pyarrow not installed. Saving as CSV instead.")
