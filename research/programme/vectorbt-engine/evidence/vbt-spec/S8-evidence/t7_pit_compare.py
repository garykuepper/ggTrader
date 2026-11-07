"""Compare ggTrader's committed S&P500 PIT file against upstream fja05680 'Updated' CSV (same format)."""
import pandas as pd, gzip
a = pd.read_csv("/home/flynn/ggTrader/data/universe/sp500_constituents_history.csv.gz", index_col=0, parse_dates=True)["tickers"].str.split(",").map(set)
b = pd.read_csv("dl/fja_updated.csv", index_col=0, parse_dates=True)["tickers"].str.split(",").map(set)
print("ggTrader file:", a.index.min().date(), a.index.max().date(), len(a), "| upstream:", b.index.min().date(), b.index.max().date(), len(b))
common = a.index.intersection(b.index); print("common dates", len(common))
diff = [(d, len(a[d]-b[d]), len(b[d]-a[d])) for d in common if a[d]!=b[d]]
print("dates with differing sets:", len(diff)); print(diff[:5], diff[-3:])
print("upstream-only dates (after ggTrader end):", [d.date() for d in b.index.difference(a.index)][:5], "n=", len(b.index.difference(a.index)))
for d in ["2021-01-29","2023-06-30","2025-12-31"]:
    d=pd.Timestamp(d); x=a[a.index<=d].iloc[-1]; print(d.date(), "n_ggt", len(x), "n_up", len(b[b.index<=d].iloc[-1]))
