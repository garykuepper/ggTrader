"""D3: numerically compare ggTrader hand-written primitives against vectorbt 0.28.5 equivalents.

Run:  PYTHONPATH=/home/flynn/ggTrader/src .venv/bin/python d3_primitive_equivalence.py
Output is the raw material for the report table (max abs diff, correlation, warm-up lengths).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.lab import metrics as gm
from ggTrader.lab.gates import dsr_check
from ggTrader.lab.simulate import compute_atr_stop, compute_vol_scalar
from ggTrader.lab.strategies import indicators as gi

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
N, S = 1200, 6
idx = pd.bdate_range("2021-01-04", periods=N, tz="UTC")
ret = rng.normal(0.0004, 0.016, (N, S))
close = pd.DataFrame(100 * np.exp(np.cumsum(ret, 0)), index=idx, columns=list("ABCDEF"))
high = close * (1 + np.abs(rng.normal(0, 0.006, (N, S))))
low = close * (1 - np.abs(rng.normal(0, 0.006, (N, S))))
volume = pd.DataFrame(rng.integers(1_000, 100_000, (N, S)).astype(float), index=idx, columns=close.columns)


def cmp(name: str, a: pd.DataFrame, b: pd.DataFrame, skip: int = 0) -> dict:
    """Compare two frames: leading-NaN warm-up of each, and diff on the jointly-valid region."""
    a = pd.DataFrame(a).reset_index(drop=True)
    b = pd.DataFrame(b).reset_index(drop=True)
    b.columns = a.columns
    wa = int(a.isna().all(axis=1).cumprod().sum())
    wb = int(b.isna().all(axis=1).cumprod().sum())
    m = a.notna() & b.notna()
    m.iloc[:skip] = False
    d = (a - b).where(m).abs()
    diff = float(np.nanmax(d.values)) if m.values.any() else float("nan")
    corr = float(np.corrcoef(a.values[m.values], b.values[m.values])[0, 1]) if m.values.sum() > 2 else float("nan")
    nan_mismatch = int((a.isna() ^ b.isna()).iloc[skip:].values.sum())
    r = dict(case=name, warm_ggt=wa, warm_vbt=wb, max_abs_diff=diff, corr=round(corr, 8),
             nan_mismatch_cells=nan_mismatch, cells=int(m.values.sum()))
    print(f"{name:58s} warm(ggt/vbt)={wa}/{wb:<3d} max|d|={diff:.3e} corr={corr:.8f} nanmismatch={nan_mismatch}")
    return r


rows = []
# ---- SMA / rolling std / Bollinger --------------------------------------------------------------
sma_g = close.rolling(20, min_periods=20).mean()
rows.append(cmp("SMA20: pandas rolling vs vbt.MA(ewm=False)", sma_g, vbt.MA.run(close, 20).ma))
std_g = close.rolling(20, min_periods=20).std()
rows.append(cmp("rolling std20 ddof=1: pandas vs vbt.MSTD default (ddof=0)", std_g, vbt.MSTD.run(close, 20).mstd))
rows.append(cmp("rolling std20 ddof=1: pandas vs vbt.MSTD(ddof=1)", std_g, vbt.MSTD.run(close, 20, ddof=1).mstd))
bb = vbt.BBANDS.run(close, window=20, alpha=2.0, ddof=1)
low_g = sma_g - 2.0 * std_g
rows.append(cmp("Bollinger lower(20,2): ggt formula vs vbt.BBANDS(ddof=1)", low_g, bb.lower))
rows.append(cmp("Bollinger lower(20,2): ggt formula vs vbt.BBANDS default", low_g, vbt.BBANDS.run(close, 20, alpha=2.0).lower))
# ggTrader bb_raw = -%b ; vbt percent_b property
rows.append(cmp("Bollinger -%b: gi.bb_raw vs -vbt.percent_b(ddof=1)", gi.bb_raw(close, 20, 2.0), -bb.percent_b))

# ---- EMA -------------------------------------------------------------------------------------
ema_g = close.ewm(span=20, adjust=False).mean()
rows.append(cmp("EMA20 span adjust=False: pandas vs vbt.MA(ewm=True)", ema_g, vbt.MA.run(close, 20, ewm=True).ma))
rows.append(cmp("EMA20 ...same, after 100-bar burn-in", ema_g, vbt.MA.run(close, 20, ewm=True).ma, skip=100))

# ---- RSI -------------------------------------------------------------------------------------
rsi_g = -gi.rsi_raw(close, 14)  # rsi_raw is negated
rows.append(cmp("RSI14 Wilder(ggt) vs vbt.RSI(14, ewm=False) [SMA/Cutler]", rsi_g, vbt.RSI.run(close, 14).rsi))
rows.append(cmp("RSI14 Wilder(ggt) vs vbt.RSI(14, ewm=True) [span=14]", rsi_g, vbt.RSI.run(close, 14, ewm=True).rsi))
rsi_v27 = vbt.RSI.run(close, 27, ewm=True).rsi  # span 27 == alpha 1/14
rows.append(cmp("RSI14 Wilder(ggt) vs vbt.RSI(27, ewm=True) [alpha=1/14]", rsi_g, rsi_v27))
rows.append(cmp("RSI14 Wilder(ggt) vs vbt.RSI(27, ewm=True) after 150-bar burn-in", rsi_g, rsi_v27, skip=150))

# RSI zero-loss edge: monotonic rising series. ggTrader -> NaN (avg_loss.replace(0,nan)); vbt -> 100
mono = pd.DataFrame({"M": np.linspace(100, 130, 80)}, index=idx[:80])
print("RSI on monotonic rise (last bar): ggt =", float(-gi.rsi_raw(mono, 14).iloc[-1, 0]),
      "| vbt(27,ewm) =", float(vbt.RSI.run(mono, 27, ewm=True).rsi.iloc[-1, 0]))

# ---- MACD ------------------------------------------------------------------------------------
macd_g = gi.macd_raw(close, 12, 26, 9)  # histogram
m = vbt.MACD.run(close, 12, 26, 9, macd_ewm=True, signal_ewm=True)
rows.append(cmp("MACD hist(12,26,9): gi.macd_raw vs vbt.MACD(ewm both)", macd_g, m.hist))
rows.append(cmp("MACD hist ... after 60-bar burn-in", macd_g, m.hist, skip=60))
rows.append(cmp("MACD hist ... after 200-bar burn-in", macd_g, m.hist, skip=200))
md = vbt.MACD.run(close, 12, 26, 9).hist
rows.append(cmp("MACD hist(12,26,9): vs vbt.MACD DEFAULT (macd_ewm=False => SMA)", macd_g, md, skip=200))

# ---- ATR (ggTrader: SMA of true range via compute_atr_stop) ----------------------------------
atr_g = compute_atr_stop(high, low, close, 14, 1.0) * close  # undo clip(lower=0.001) not hit here
rows.append(cmp("ATR14: ggt SMA-of-TR vs vbt.ATR(14, ewm=False)", atr_g, vbt.ATR.run(high, low, close, 14, ewm=False).atr))
rows.append(cmp("ATR14: ggt SMA-of-TR vs vbt.ATR default (ewm=True)", atr_g, vbt.ATR.run(high, low, close, 14).atr))
# Wilder ATR reference: alpha=1/14
tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()]).groupby(level=0).max()
tr = np.maximum(np.maximum(high - low, (high - close.shift()).abs()), (low - close.shift()).abs())
atr_w = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
rows.append(cmp("ATR14: Wilder (alpha=1/14) vs vbt.ATR(27, ewm=True)", atr_w, vbt.ATR.run(high, low, close, 27).atr, skip=150))

# ---- Stochastic (not in current ggTrader; reference = textbook %K with rolling min/max) -----------
ll = low.rolling(14, min_periods=14).min()
hh = high.rolling(14, min_periods=14).max()
k_ref = 100 * (close - ll) / (hh - ll)
st = vbt.STOCH.run(high, low, close, k_window=14, d_window=3)
rows.append(cmp("Stoch %K14: textbook(high/low) vs vbt.STOCH", k_ref, st.percent_k))
rows.append(cmp("Stoch %D3 (SMA of %K): textbook vs vbt.STOCH", k_ref.rolling(3, min_periods=3).mean(), st.percent_d))

# ---- OBV (not in current ggTrader) ---------------------------------------------------------------
cr = close.round(0)  # force ties (unchanged closes)
sign = np.sign(cr.diff()).fillna(0.0)
obv_ref = (sign * volume).cumsum()
obv_v = vbt.OBV.run(cr, volume).obv
rows.append(cmp("OBV w/ tied closes: textbook sign(0)=0 vs vbt.OBV", obv_ref, obv_v))
n_ties = int((cr.diff() == 0).values.sum())
print("tied-close bars in OBV test:", n_ties)
# vbt rule: tie -> +volume (because `close < prev` is False). Rebuild it to confirm the diagnosis.
obv_vbtrule = pd.DataFrame(np.where(cr < cr.shift(1), -volume, volume), index=cr.index, columns=cr.columns).cumsum()
rows.append(cmp("OBV ties: 'tie=+volume' rule vs vbt.OBV (diagnosis check)", obv_vbtrule, obv_v))
ref_nt = (np.sign(close.diff()).fillna(0) * volume).cumsum()
rows.append(cmp("OBV no ties: textbook vs vbt.OBV (raw)", ref_nt, vbt.OBV.run(close, volume).obv))
rows.append(cmp("OBV no ties: textbook vs vbt.OBV minus first-bar volume", ref_nt, vbt.OBV.run(close, volume).obv - volume.iloc[0]))

# ---- Keltner: build from EMA + ATR ---------------------------------------------------------------
kel_mid = vbt.MA.run(close, 20, ewm=True).ma
print("Keltner channel: no built-in in vbt 0.28.5 (no class named KELTNER/KC):",
      [n for n in dir(vbt) if "kelt" in n.lower() or n.upper() == "KC"])

# ---- Entry/exit signals: ggTrader rsi_signals vs vbt crossed_below/above -------------------------
e_g, x_g = gi.rsi_signals(close, 14, 30, 50)
r_v = vbt.RSI.run(close, 27, ewm=True)
e_v = r_v.rsi_crossed_below(30)
x_v = r_v.rsi_crossed_above(50)
n_e_g, n_e_v = int(e_g.values.sum()), int(e_v.values.sum())
print(f"RSI entry signals: ggt={n_e_g} vbt(27,ewm)={n_e_v} symmetric diff={(e_g.values ^ e_v.values).sum()}"
      f" | after 150 burn-in diff={(e_g.values[150:] ^ e_v.values[150:]).sum()}")
print("   (vbt crossed_below is strict '<' now and '>=' before; ggt uses prev>=30 & now<30: same convention?)",
      "ggt exits=", int(x_g.values.sum()), "vbt exits=", int(x_v.values.sum()))

# ---- Metrics: curve_stats vs vbt returns accessor ------------------------------------------------
eq = pd.Series(100_000 * np.cumprod(1 + rng.normal(0.0005, 0.011, 800)), index=pd.bdate_range("2022-01-03", periods=800, tz="UTC"))
g = gm.curve_stats(eq)
r = eq.pct_change().dropna()
racc = r.vbt.returns(freq="D", year_freq="252 days")
print("\n--- metrics (same curve) ---")
print(f"Sharpe   ggt={g['sharpe']:.10f} vbt(year_freq=252d)={racc.sharpe_ratio():.10f} vbt(default 365d)={r.vbt.returns(freq='D').sharpe_ratio():.10f}")
print(f"Sortino  ggt={g['sortino']:.10f} vbt={racc.sortino_ratio():.10f}  (definitions differ: ggt=mean/std(neg rets), vbt=mean/sqrt(mean(min(r,0)^2)))")
dd_v = racc.max_drawdown() * 100
print(f"MaxDD%   ggt={g['max_drawdown_pct']:.10f} vbt={dd_v:.10f}")
years_cal = (eq.index[-1] - eq.index[0]).days / 365.25
print(f"CAGR%    ggt={g['cagr_pct']:.6f} vbt annualized()(252-bar year)={racc.annualized()*100:.6f} (ggt uses calendar days/365.25; vbt uses bar-count/252)")
print(f"AnnVol%  ggt={g['ann_vol_pct']:.8f} vbt={racc.annualized_volatility()*100:.8f}")
print(f"Calmar   vbt={racc.calmar_ratio():.6f} (ggt curve_stats has no Calmar)")
# zero-variance behaviour
flat = pd.Series(1.0, index=eq.index[:50])
print("flat curve Sharpe: ggt =", gm.curve_stats(flat * 100)["sharpe"], "| vbt =", flat.pct_change().dropna().vbt.returns(freq="D", year_freq="252 days").sharpe_ratio())

# ---- Deflated Sharpe -----------------------------------------------------------------------------
print("\n--- DSR ---")
R = pd.DataFrame(rng.normal(0.0003, 0.01, (252, 48)), index=pd.bdate_range("2023-01-02", periods=252))
sh = R.vbt.returns(freq="D", year_freq="252 days").sharpe_ratio()
j = int(sh.values.argmax())
rj = R.iloc[:, j]
from scipy.stats import kurtosis, skew  # noqa: E402

d_g = dsr_check(float(sh.iloc[j]), n_obs=len(rj), n_trials=48, skew=float(skew(rj)), kurtosis_excess=float(kurtosis(rj)))
d_v = R.vbt.returns(freq="D", year_freq="252 days").deflated_sharpe_ratio()  # nb_trials=48, var across cols
print(f"best col SR={sh.iloc[j]:.3f}  ggt DSR={d_g.dsr_value:.6f} (E[maxSR]ann={d_g.expected_max_sr:.3f})  vbt0.28.5 DSR[best col]={d_v.iloc[j]:.6f}")
# Does vbt's formula reproduce ggt's when fed the same inputs? (excess vs non-excess kurtosis bug)
from vectorbt.returns import metrics as vm  # noqa: E402

est = float(sh.iloc[j]) / np.sqrt(252)
sk, ku = float(skew(rj)), float(kurtosis(rj))
ve = float(np.var(sh.values, ddof=1)) / 252
vv_excess = vm.deflated_sharpe_ratio(est_sharpe=np.array([est]), var_sharpe=ve, nb_trials=48, backtest_horizon=len(rj), skew=np.array([sk]), kurtosis=np.array([ku]))
vv_nonex = vm.deflated_sharpe_ratio(est_sharpe=np.array([est]), var_sharpe=ve, nb_trials=48, backtest_horizon=len(rj), skew=np.array([sk]), kurtosis=np.array([ku + 3]))
print(f"vbt fn with excess kurt (0.28.5 behaviour)={vv_excess[0]:.8f}  with non-excess (1.1.1 fix)={vv_nonex[0]:.8f}")

vv_match = vm.deflated_sharpe_ratio(est_sharpe=np.array([est]), var_sharpe=1.0 / (len(rj) - 1), nb_trials=48,
                                    backtest_horizon=len(rj), skew=np.array([sk]), kurtosis=np.array([ku + 3]))
print(f"vbt fn with var_sharpe=1/(T-1) + non-excess kurt = {vv_match[0]:.10f}  vs ggt dsr_check = {d_g.dsr_value:.10f}"
      f"  |diff|={abs(vv_match[0]-d_g.dsr_value):.2e}")

# ---- Vol targeting -------------------------------------------------------------------------------
sc_g = compute_vol_scalar(close, 0.10, 20, 2.0)
ew = close.pct_change().mean(axis=1)
rv = ew.vbt.returns(freq="D", year_freq="252 days").rolling_annualized_volatility(window=20, minp=20)
sc_v = (0.10 / rv).shift(1).clip(0.1, 2.0).fillna(1.0)
print(f"\nvol scalar: ggt vs vbt rolling_annualized_volatility max|d|={float((sc_g - sc_v).abs().max()):.3e}")

# ---- stops --------------------------------------------------------------------------------------
import vectorbt.portfolio.base as pb  # noqa: E402

print("td_stop/time-stop kwarg in Portfolio.from_signals:", "td_stop" in pb.Portfolio.from_signals.__code__.co_varnames)
print("sl_stop/sl_trail/tp_stop present:", all(k in pb.Portfolio.from_signals.__code__.co_varnames for k in ("sl_stop", "sl_trail", "tp_stop")))

pd.DataFrame(rows).to_csv("/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-research/D-examples/d3_results.csv", index=False)
