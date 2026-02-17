import pandas as pd

import pyarrow as pa

print(pd.__version__, pa.__version__)

date_index = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
# Example with custom dataframes
df1 = pd.DataFrame({
    ('Symbol1', 'Open'): [10, 11, 12],
    ('Symbol1', 'High'): [12, 13, 14],
    ('Symbol1', 'Low'): [9, 10, 11],
    ('Symbol1', 'Close'): [11, 12, 13],
    ('Symbol1', 'Volume'): [100, 110, 120]
}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

df2 = pd.DataFrame({
    ('Symbol2', 'Open'): [20, 21, 22],
    ('Symbol2', 'High'): [22, 23, 24],
    ('Symbol2', 'Low'): [19, 20, 21],
    ('Symbol2', 'Close'): [21, 22, 23],
    ('Symbol2', 'Volume'): [200, 210, 220]
}, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

df3 = pd.DataFrame({
    ('Symbol2', 'Open'): [20, 22],
    ('Symbol2', 'High'): [22, 24],
    ('Symbol2', 'Low'): [19, 21],
    ('Symbol2', 'Close'): [21, 23],
    ('Symbol2', 'Volume'): [200, 220]
}, index=pd.to_datetime(['2023-01-01', '2023-01-03']))

# Combine them using pd.concat
combined_ohlcv = pd.concat([df1, df3], axis=1)
print(combined_ohlcv.head())

print(combined_ohlcv['Symbol1'].head())


import pandas as pd

df = pd.DataFrame({
    "timestamp": ["2025-10-20", "2025-10-21"],
    "open": [100, 105],
    "high": [110, 115],
    "low": [95, 100],
    "close": [108, 112],
    "volume": [1000, 1500]
})
new_data = pd.DataFrame({
    "timestamp": ["2025-10-22", "2025-10-23"],
    "open": [113, 117],
    "high": [118, 120],
    "low": [110, 114],
    "close": [116, 119],
    "volume": [1600, 1800]
})
df = pd.concat([df, new_data], ignore_index=True)
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
df = df.set_index("timestamp").sort_index().drop_duplicates()

print(df.head())