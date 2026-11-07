"""S1 prototype: ensemble in the proposed contract (built-in vbt indicators, run once per
unique param, votes combined per combo, one grouped from_signals over all combos)."""
import itertools
import resource
import time

import numpy as np
import pandas as pd
import vectorbt as vbt

vbt.settings["engine"] = "rust"
rng = np.random.default_rng(0)
T, S = 1700, 500
idx = pd.bdate_range("2019-06-01", periods=T)
close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.015, (T, S)), 0)), idx,
                     [f"S{i:03d}" for i in range(S)])
volume = pd.DataFrame(rng.lognormal(13, 0.4, (T, S)), idx, close.columns)
GRID = dict(min_agree=[2, 3, 4], min_agree_exit=[1, 2], bb_alpha=[2.0, 2.5],
            rsi_oversold=[25, 30], ema_fast=[10, 20])
t0 = time.time()
# --- voters: one vbt call per indicator, param lists vectorised by vbt itself -------------
bb = vbt.BBANDS.run(close, window=20, alpha=GRID["bb_alpha"], ddof=1, short_name="bb")
rsi = vbt.RSI.run(close, window=27, ewm=True, short_name="rsi")       # span 2n-1 ~ Wilder 14
ema_f = vbt.MA.run(close, window=GRID["ema_fast"], ewm=True, short_name="emaf")
ema_s = vbt.MA.run(close, window=50, ewm=True, short_name="emas")
macd = vbt.MACD.run(close, 12, 26, 9, macd_ewm=True, signal_ewm=True)
vma = vbt.MA.run(volume, window=20)
t_ind = time.time() - t0
# per-voter boolean arrays keyed by the voter's own param
bb_ent = {a: bb.close_crossed_below(bb.lower).xs(a, level="bb_alpha", axis=1).to_numpy()
          for a in GRID["bb_alpha"]}
bb_ext = {a: bb.close_crossed_above(bb.middle).xs(a, level="bb_alpha", axis=1).to_numpy()
          for a in GRID["bb_alpha"]}
rsi_ent = {o: rsi.rsi_crossed_below(float(o)).to_numpy() for o in GRID["rsi_oversold"]}
rsi_ext = rsi.rsi_crossed_above(50.0).to_numpy()
emaf = {f: ema_f.ma.xs(f, level="emaf_window", axis=1) for f in GRID["ema_fast"]}
slow = ema_s.ma.droplevel(0, axis=1)
ema_ent = {f: emaf[f].vbt.crossed_above(slow).to_numpy() for f in GRID["ema_fast"]}
ema_ext = {f: emaf[f].vbt.crossed_below(slow).to_numpy() for f in GRID["ema_fast"]}
h = macd.hist
macd_ent = ((close <= close.rolling(10).min()) & (h > h.rolling(10).min())).to_numpy()
macd_ext = h.vbt.crossed_below(0.0).to_numpy()
vspike = (volume.to_numpy() > 1.5 * vma.ma.to_numpy())
combos = list(itertools.product(*GRID.values()))
E, X, keys = [], [], []
for ma, mx, a, o, f in combos:
    ev = (bb_ent[a].astype(np.int8) + rsi_ent[o] + ema_ent[f] + macd_ent
          + (bb_ent[a] & vspike))
    xv = bb_ext[a].astype(np.int8) + rsi_ext + ema_ext[f] + macd_ext + bb_ext[a]
    E.append(ev >= ma)
    X.append(rsi_ext | (xv >= mx))
    keys.append((ma, mx, a, o, f))
cols = pd.MultiIndex.from_tuples([k + (s,) for k in keys for s in close.columns],
                                 names=list(GRID) + ["symbol"])
entries = pd.DataFrame(np.hstack(E), idx, cols)
exits = pd.DataFrame(np.hstack(X), idx, cols)
t_sig = time.time() - t0 - t_ind
pf = vbt.Portfolio.from_signals(np.tile(close.to_numpy(), len(combos)), entries, exits,
                                size=0.03, size_type="percent", sl_stop=0.10, sl_trail=True,
                                cash_sharing=True, group_by=list(GRID), freq="1D")
t_pf = time.time() - t0 - t_ind - t_sig
sh = pf.returns().vbt.returns(freq="1D", year_freq="252 days").sharpe_ratio()
print(f"cols={entries.shape[1]} groups={len(sh)} ind={t_ind:.1f}s sig={t_sig:.1f}s "
      f"pf={t_pf:.1f}s total={time.time()-t0:.1f}s peakRSS="
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB")
print("best", sh.replace([np.inf, -np.inf], np.nan).idxmax())
