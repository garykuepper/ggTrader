"""D2: how parameters become columns in vbt 0.28.5 and how to pick winners.

Covers: IndicatorFactory.run(param_product=True) vs zipped params, run_combs, Portfolio-level
parameter broadcasting (sl_stop / size as Index), selection via pf.sharpe_ratio()/stats().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

rng = np.random.default_rng(1)
idx = pd.bdate_range("2022-01-03", periods=500)
close = pd.DataFrame(
    100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, (500, 4)), axis=0)),
    index=idx,
    columns=list("ABCD"),
)

print("=== A. param_product=True vs zipped ===")
prod = vbt.MA.run(close, window=[10, 20, 50], ewm=[False, True], param_product=True)
print("product: shape", prod.ma.shape, "levels", list(prod.ma.columns.names))
zipped = vbt.MA.run(close, window=[10, 20, 50], ewm=[False, True, False])
print("zipped : shape", zipped.ma.shape, "levels", list(zipped.ma.columns.names))
# TRAP: unequal-length param lists without param_product raise
try:
    vbt.MA.run(close, window=[10, 20, 50], ewm=[False, True])
except Exception as e:  # noqa: BLE001
    print("unequal lengths w/o param_product ->", type(e).__name__, str(e)[:90])

print("\n=== B. run_combs: all unordered pairs of one param list (fast < slow) ===")
fast, slow = vbt.MA.run_combs(close["A"], window=[5, 10, 20, 50], r=2, short_names=["fast", "slow"])
print("fast cols:", [c for c in fast.ma.columns][:6])
print("entries:", fast.ma_crossed_above(slow).shape, "(6 pairs = C(4,2)); fast<slow built in)")

print("\n=== C. Portfolio-level param broadcasting ===")
entries = vbt.RSI.run(close, window=14).rsi_crossed_below(30)
exits = vbt.RSI.run(close, window=14).rsi_crossed_above(50)
sl_vals = [0.02, 0.05, 0.10]
sl = pd.Index(sl_vals, name="sl_stop")

# C1. A bare Index of params does NOT expand into columns: it is just a (3,) array and fails to broadcast
try:
    vbt.Portfolio.from_signals(close, entries, exits, sl_stop=sl, sl_trail=True, freq="D")
except Exception as e:  # noqa: BLE001
    print("C1 plain Index ->", type(e).__name__, str(e)[:80])

# C2. Manual product: tile the signals with keys=, give sl_stop a (1, n_cols) row
ent_t = entries.vbt.tile(len(sl), keys=sl)
ext_t = exits.vbt.tile(len(sl), keys=sl)
sl_row = np.repeat(sl.values, close.shape[1])[None, :]
pf2 = vbt.Portfolio.from_signals(close, ent_t, ext_t, sl_stop=sl_row, sl_trail=True, freq="D")
print("C2 tile+row ->", list(pf2.wrapper.columns.names), pf2.wrapper.shape_2d)

# C3. Idiomatic: make the stop a parameter/output of an IndicatorFactory so the level is created for you
def sig_apply(close, rsi_lo, sl):
    r = vbt.RSI.run(close, window=14).rsi.values
    e = (r < rsi_lo) & (np.roll(r, 1, 0) >= rsi_lo)
    x = (r >= 50) & (np.roll(r, 1, 0) < 50)
    return e, x, np.full(close.shape, sl)

Sig = vbt.IndicatorFactory(input_names=["close"], param_names=["rsi_lo", "sl"],
                           output_names=["entries", "exits", "sl_stop"]).from_apply_func(sig_apply)
sig = Sig.run(close, rsi_lo=[25, 30], sl=sl_vals, param_product=True)
pf3 = vbt.Portfolio.from_signals(close, sig.entries, sig.exits, sl_stop=sig.sl_stop, sl_trail=True,
                                 size=0.1, size_type="percent", init_cash=100_000.0, cash_sharing=True,
                                 group_by=[0, 1], freq="D")
print("C3 factory ->", list(pf3.wrapper.grouper.get_columns().names), "groups:", pf3.wrapper.grouper.get_group_count())
pf = pf3
print("\n=== D. selection ===")
sh = pf.sharpe_ratio()  # Series indexed by (rsi_lo, sl) group; ann. factor from year_freq (default 365 days!)
print(sh.round(3).to_string())
best = sh.replace([np.inf, -np.inf], np.nan).idxmax()
print("idxmax (rsi_lo, sl):", best)
print("\npf.stats() for the winning group:\n", pf.stats(column=best, group_by=[0, 1]).to_string())
print("\nDefault year_freq =", vbt.settings.returns["year_freq"], "| pass year_freq='252 days' to match ggTrader")
print("Sharpe with year_freq=252:\n", pf.sharpe_ratio(year_freq="252 days").round(3).to_string())
