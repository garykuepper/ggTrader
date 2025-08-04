import yfinance as yf
import pandas as pd
from tabulate import tabulate

# Get the most active stocks from Yahoo Finance
response = yf.screen('most_actives', sortAsc=True)
most_active = []

for quote in response["quotes"]:
    row = [
        quote.get('shortName', ''),                        # Company name
        quote.get("symbol", ''),                           # Ticker symbol
        quote.get("regularMarketPrice", 0),                # Current price
        quote.get("regularMarketChangePercent", 0),        # % change today
        quote.get("regularMarketVolume", 0),               # Today's volume
        quote.get("averageDailyVolume10Day", 0),           # 10-day avg volume
        quote.get("fiftyTwoWeekHigh", 0),                  # 52-week high
        quote.get("fiftyTwoWeekLow", 0),                   # 52-week low
        quote.get("fiftyTwoWeekHighChangePercent", 0),     # % from 52w high
        quote.get("fiftyTwoWeekLowChangePercent", 0),      # % from 52w low
        quote.get("fiftyDayAverage", 0),                   # 50-day avg price
        quote.get("fiftyDayAverageChangePercent", 0),      # % from 50d avg
        quote.get("marketCap", 0)                          # Market cap
    ]
    most_active.append(row)

# Define headers for the DataFrame
headers = [
    "name", "symbol", "close", "%chg", "volume", "vol_10d",
    "52w_high", "52w_low", "%from_high", "%from_low",
    "50d_avg", "%from_50d", "mkt_cap"
]

# Create a pandas DataFrame
df = pd.DataFrame(most_active, columns=headers)

# Optional: sort by 10-day average volume, descending
df = df.sort_values(by="vol_10d", ascending=False)

# Print the DataFrame as a table
print(tabulate(df, headers='keys', tablefmt='github'))

# Now you can use all pandas features, e.g.:
# df.head(), df.describe(), df[df['%chg'] > 2], etc.