"""Verify vectorised PIT mask == eligible_at on >=200 sampled dates, 4 configs. Time + memory."""
import sys, time, json, resource
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from ggTrader.lab.data import eligible_at
from ggTrader.lab.strategy import LabConfig
import json as _json, pathlib
def load_sector_map():  # copy of registry.load_sector_map (importing the strategies package needs extra deps)
    return _json.load(open(pathlib.Path("/home/flynn/ggTrader/data/universe/sp500_sectors.json")))
def apply_sector_constraints(symbols, max_sec, _sm=None):  # verbatim logic of registry.apply_sector_constraints
    sector_map = load_sector_map(); selected = []; counts = {}
    for sym in symbols:
        sec = sector_map.get(sym, "Unknown"); c = counts.get(sec, 0)
        if c < max_sec: selected.append(sym); counts[sec] = c + 1
    return selected
from ggTrader.data.core.index_constituents import load_sp500_history
from pit import pit_mask

close = pd.read_parquet("tape_close.parquet")
print("tape", close.shape, close.index.tz, flush=True)
# eligible_at only needs (symbol, 'close')
ohlcv = pd.concat({s: close[[s]].rename(columns={s: "close"}) for s in close.columns}, axis=1)
hist = load_sp500_history()
chg = [d for d in hist.index if close.index[0] <= d <= close.index[-1]]
edge = sorted({close.index[min(max(close.index.searchsorted(d) + k, 0), len(close) - 1)] for d in chg for k in (-1, 0, 1)})
rng = np.random.default_rng(7)
rand = sorted(rng.choice(close.index, 160, replace=False))
dates = sorted(set(pd.DatetimeIndex(edge)) | set(pd.DatetimeIndex(rand)))
print("snapshot changes in range:", len(chg), "| edge dates:", len(edge), "| random:", len(rand), "| total sampled:", len(dates), flush=True)
sector_map = load_sector_map()
cfgs = {
  "min_hist=400 (lab default)": dict(min_history_bars=400),
  "min_hist=60 (live)": dict(min_history_bars=60),
  "min_hist=60 + max_stocks=50": dict(min_history_bars=60, max_stocks=50),
  "min_hist=60 + max_sector_count=3": dict(min_history_bars=60, max_sector_count=3),
}
res = {}
for name, kw in cfgs.items():
    t = time.time(); m = pit_mask(close, sector_map=sector_map, **kw); tv = time.time() - t
    mism = 0; ndiff_dates = 0; tot = 0
    t = time.time()
    for d in dates:
        cfg = LabConfig(min_history_bars=kw["min_history_bars"], max_stocks=kw.get("max_stocks"))
        e, _ = eligible_at(d, ohlcv.loc[:d], cfg, universe="sp500")
        if kw.get("max_sector_count"):
            e = apply_sector_constraints(e, kw["max_sector_count"])   # as EnsembleSignal.select does
        v = set(m.columns[m.loc[d].to_numpy()])
        x = set(e) ^ v; mism += len(x); ndiff_dates += bool(x); tot += len(e)
        if x and ndiff_dates <= 3: print("   diff", d.date(), sorted(x)[:8], flush=True)
    res[name] = dict(vector_build_s=round(tv, 2), eligible_at_loop_s=round(time.time() - t, 1), dates=len(dates),
                     symbol_mismatches=mism, dates_with_mismatch=ndiff_dates, mean_eligible=round(tot/len(dates), 1),
                     mask_shape=list(m.shape), mask_mb=round(m.to_numpy().nbytes/1e6, 1))
    print(name, res[name], flush=True)
# full-grid timing of the original per-bar approach, extrapolated
t = time.time(); [eligible_at(d, ohlcv.loc[:d], LabConfig(min_history_bars=60)) for d in dates[:20]]
per = (time.time() - t) / 20
res["original_per_bar_s"] = round(per, 3); res["original_full_tape_est_s"] = round(per * len(close), 0)
print("original per-bar", round(per, 3), "s -> est full tape", round(per * len(close)), "s (ESTIMATE; cost grows with past length)", flush=True)
res["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
json.dump(res, open("p4_pit_verify.json", "w"), indent=1)
