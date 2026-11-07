import sys, time, json, pathlib
sys.path.insert(0, "/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S6-evidence")
import numpy as np, pandas as pd
from ggTrader.lab.data import eligible_at
from ggTrader.lab.strategy import LabConfig
from pit import pit_mask
close = pd.read_parquet("/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S6-evidence/tape_close.parquet")
ohlcv = pd.concat({s: close[[s]].rename(columns={s: "close"}) for s in close.columns}, axis=1)
sm = json.load(open("/home/flynn/ggTrader/data/universe/sp500_sectors.json"))
rng = np.random.default_rng(20261106)
dates = sorted(rng.choice(close.index[1:], 100, replace=False))
print("pandas", pd.__version__)
for mh in (400, 60):
    m = pit_mask(close, sector_map=sm, min_history_bars=mh)
    mism = 0; tot = 0
    for d in dates:
        e, _ = eligible_at(d, ohlcv.loc[:d], LabConfig(min_history_bars=mh), universe="sp500")
        v = set(m.columns[m.loc[d].to_numpy()]); mism += len(set(e) ^ v); tot += len(e)
    print("min_hist", mh, "dates", len(dates), "mismatches", mism, "mean elig", tot/len(dates))
