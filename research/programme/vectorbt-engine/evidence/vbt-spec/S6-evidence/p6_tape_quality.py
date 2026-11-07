"""Tape quality: weekday stamps, split-like discontinuities in the cached adjusted closes, volume, dup rows."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
c = pd.read_parquet("tape_close.parquet"); v = pd.read_parquet("tape_volume.parquet")
print("dow counts (0=Mon):", c.index.dayofweek.value_counts().sort_index().to_dict(), "| hours:", sorted(set(c.index.hour)))
print("index dups:", int(c.index.duplicated().sum()), "| monotonic:", c.index.is_monotonic_increasing)
r = c.pct_change(fill_method=None)
ratios = np.array([2, 3, 4, 5, 10, 1.5, 20, 8, 6, 7, 15])
hits = []
for sym in r.columns:
    s = r[sym].dropna()
    big = s[(s.abs() > 0.35)]
    for d, x in big.items():
        ratio = 1 / (1 + x)           # prior/current price
        for k in list(ratios) + list(1 / ratios):
            if abs(ratio / k - 1) < 0.04:
                hits.append((sym, d.date().isoformat(), round(x, 3), round(float(ratio), 3))); break
print("split-like one-day jumps (|ret|>35% and price ratio within 4% of a standard split ratio):", len(hits))
for h in hits[:25]: print("  ", h)
# recent ones are the dangerous ones: a split inside the tape that the cache may have mis-adjusted
print("count by year:", pd.Series([h[1][:4] for h in hits]).value_counts().sort_index().to_dict())
# 'adjustment basis patchwork' probe: nothing detectable without raw prices; report Yahoo adj factor for AAPL 2024-01-02: tape vs fresh
print("AAPL tape close 2024-01-02:", float(c['AAPL'].loc['2024-01-02']))
print("zero-volume closes (cells):", int(((v == 0) & c.notna()).sum().sum()), "| NaN volume where close present:", int((v.isna() & c.notna()).sum().sum()))
print("dollar-volume available for top-N ranking: yes (close*volume)")
