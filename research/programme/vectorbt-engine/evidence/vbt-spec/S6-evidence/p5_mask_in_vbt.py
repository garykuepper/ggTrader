"""PIT mask x vbt signals: entries&mask, exit-on-removal, delisted tails, memory/time."""
import sys, time, resource
sys.path.insert(0, '.')
import numpy as np, pandas as pd, vectorbt as vbt
from pit import pit_mask, membership_mask

close = pd.read_parquet("tape_close.parquet")
T, N = close.shape
print("tape", close.shape, flush=True)

rsi = vbt.RSI.run(close, window=14).rsi.droplevel(0, axis=1)            # vbt-native indicator (defaults: SMA-style; plumbing test only)
entries = rsi.vbt.crossed_below(30)
exits = rsi.vbt.crossed_above(50)
print("raw entries", int(entries.values.sum()), "exits", int(exits.values.sum()), flush=True)

t = time.time(); mask = pit_mask(close, min_history_bars=60); print("mask build", round(time.time()-t, 3), "s; eligible cells", int(mask.values.sum()), flush=True)
member = membership_mask(close.index, close.columns)

def run(ent, ext, label):
    t = time.time()
    pf = vbt.Portfolio.from_signals(close, ent, ext, init_cash=10000.0, size=0.03, size_type="percent",
                                    fees=0.0, slippage=0.0005, freq="1D")
    dt = time.time() - t
    rec = pf.trades.records_readable if False else pf.trades.records
    held = pf.assets() > 0
    held = held
    held_nonmember_bars = int((held & ~member).values.sum())
    ent_nonmember = int(((ent) & ~member).values.sum())
    tot = float(pf.value().sum(axis=1).iloc[-1])
    print(f"{label:34s} sim {dt:5.1f}s trades={len(rec):5d}  entry-signals-on-nonmember={ent_nonmember:5d}  held-bars-nonmember={held_nonmember_bars:6d}  sum(final value)={tot:,.0f}", flush=True)
    return pf

pf0 = run(entries, exits, "A unmasked (legacy lookahead)")
pf1 = run(entries & mask, exits, "B entries&mask (current semantics)")
removed = member.shift(1, fill_value=False) & ~member                      # first bar after leaving the index
exits_rm = exits | removed
pf2 = run(entries & mask, exits_rm, "C entries&mask + exit-on-removal")
print("removal events in tape:", int(removed.values.sum()), "| symbols that ever left:", int(removed.any().sum()), flush=True)

# delisted tails: symbols whose price history ends well before the tape end
last_valid = close.apply(lambda s: s.last_valid_index())
dead = last_valid[last_valid < close.index[-1] - pd.Timedelta(days=10)]
print("symbols with data ending >10d before tape end:", len(dead), "| e.g.", dict(list(dead.dt.date.astype(str).items())[:6]), flush=True)
# what does vbt do with an open position when price turns NaN forever?
inpos_at_end = (pf1.assets() > 0).iloc[:, :]
stranded = [s for s in dead.index if bool(inpos_at_end[s].iloc[-1])]
print("B: positions still 'open' in symbols whose data already ended:", len(stranded), stranded[:8], flush=True)
last_bar = close.notna() & ~close.notna().shift(-1, fill_value=False)       # last valid bar of each series
terminal = last_bar & pd.DataFrame(np.broadcast_to((close.index < close.index[-1]).reshape(-1, 1), close.shape), index=close.index, columns=close.columns)
pf3 = run(entries & mask, exits_rm | terminal, "D + terminal exit at last valid bar")
stranded3 = [s for s in dead.index if bool((pf3.assets() > 0)[s].iloc[-1])]
print("D: stranded after terminal exit:", len(stranded3), flush=True)
# valuation of a stranded position under NaN prices (pf1) -- is it carried at last price?
if stranded:
    s = stranded[0]; v = pf1.value()[s]
    print(f"{s}: last valid {last_valid[s].date()}  value on that bar {v.loc[last_valid[s]]:.2f}  value at tape end {v.iloc[-1]:.2f}  (NaN? {np.isnan(v.iloc[-1])})", flush=True)
print("maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)

# ---- diagnostics of residuals ----
held3 = (pf3.assets() > 0)
resid = (held3 & ~member).sum()
resid = resid[resid > 0].sort_values(ascending=False)
print("D residual held-nonmember bars by symbol (top 8):", resid.head(8).to_dict(), flush=True)
print("D stranded:", stranded3, {s: str(last_valid[s].date()) for s in stranded3}, flush=True)
# raggedness of tape end (live-relevant): last bar coverage
n_last = close.iloc[-1].notna().sum(); n_prev = close.iloc[-2].notna().sum()
print(f"tape end coverage: last bar {close.index[-1].date()} has {n_last}/{N} closes; previous {close.index[-2].date()} has {n_prev}/{N}", flush=True)
# gaps inside live histories (not at ends): count interior NaN runs among symbols with >=60 bars and still alive
interior = 0
for c in close.columns:
    s = close[c]; fv, lv = s.first_valid_index(), s.last_valid_index()
    if fv is None: continue
    interior += int(s.loc[fv:lv].isna().sum())
print("interior NaN cells (between first and last valid bar):", interior, flush=True)

# ---- E: terminal exit must win over a same-bar entry ----
pf4 = run(entries & mask & ~terminal, exits_rm | terminal, "E D + drop same-bar entry at terminal")
stranded4 = [s for s in close.columns if bool((pf4.assets() > 0)[s].iloc[-1]) and s in dead.index]
print("E: stranded after terminal exit (and entry suppressed):", stranded4, flush=True)
print("E: held-bars-nonmember residual:", int(((pf4.assets() > 0) & ~member).values.sum()), flush=True)
