import pandas as pd
import os


path = "data/kraken_hist_1d_latest"

def get_ohlcv_dict(path: str):
    with os.scandir(path) as it:
        files = [entry.name for entry in it if entry.is_file()]
    num_files = len(files)

    ohlcv_dict = {}

    for f in files:
        print(f"({files.index(f) + 1}/{num_files}) Processing {f} ")
        file_path = os.path.join(path, f)
        ticker = f.split("_")[0]
        ohlcv_dict[ticker] = pd.read_csv(file_path)

    return ohlcv_dict

ohlcv_dict = get_ohlcv_dict(path)

dates = pd.date_range(start="2024-01-01", end="2025-06-30", freq="4h")

# Build a list of DataFrames with ticker info
dfs = []
for ticker, df in ohlcv_dict.items():
    if df is None or df.empty:
        continue
    df = df.copy()
    df["ticker"] = ticker  # add the ticker column
    # If you want the ticker first:
    df = df[["ticker"] + [c for c in df.columns if c != "ticker"]]
    dfs.append(df)

N = 100
# Concatenate into a single DataFrame
combined = pd.concat(dfs, axis=0).sort_index()
top_by_volume = combined.sort_values(['date','volume'], ascending=[True, False])
top_per_day = top_by_volume.groupby('date').head(N).reset_index(drop=True)

top_per_day.to_csv("data/kraken_historical_volume_movers.csv")





