"""PIT member-day coverage: how much of the point-in-time universe has a price on that day?"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from ggTrader.lab.data import equity_universe_between
from pit import membership_mask
c = pd.read_parquet("tape_close.parquet")
uni = equity_universe_between(pd.Timestamp("2019-03-01", tz="UTC"), pd.Timestamp("2026-11-05", tz="UTC"))
full = sorted(set(uni) | set(c.columns))
m = membership_mask(c.index, full)
have = c.reindex(columns=full).notna()
rows = []
for y, g in m.groupby(m.index.year):
    mm = g.to_numpy(); hh = have.loc[g.index].to_numpy()
    rows.append((y, int(mm.sum()), round(100 * (mm & hh).sum() / mm.sum(), 2)))
print("year  member-cells  % with a close on that day")
for r in rows: print(*r)
nodata = [s for s in full if s not in c.columns]
print("PIT members with no cached bars at all:", len(nodata), nodata)
ever = m[nodata].any()
print("their member-days:", int(m[nodata].sum().sum()), "of", int(m.to_numpy().sum()), f"({100*m[nodata].sum().sum()/m.to_numpy().sum():.2f}%)")
