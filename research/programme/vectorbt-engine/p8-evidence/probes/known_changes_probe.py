"""P8 probe: one number (or array digest) per known upstream release-note item, raw vectorbt API.

Run with the candidate venv's interpreter, `-I`, no ggTrader import. Prints one JSON object
{quantity: {"v": value(s) | "ERR ..."}}; compare across venvs with compare.py.
"""

import json
import warnings

import numpy as np
import pandas as pd
import vectorbt as vbt

warnings.simplefilter("ignore")
import os
if os.environ.get("P8_ENGINE"):
    vbt.settings["engine"] = os.environ["P8_ENGINE"]
rng = np.random.default_rng(7)
idx = pd.bdate_range("2020-01-01", periods=400)
close = pd.Series(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, 400))), idx, name="x")
high, low = close * 1.004, close * 0.996
volume = pd.Series(rng.integers(1000, 5000, 400).astype(float), idx)
ret = close.pct_change(fill_method=None).dropna()
fat = pd.Series(rng.standard_t(3, 500) * 0.01, pd.bdate_range("2020-01-01", periods=500))
fat_nan = fat.copy()
fat_nan.iloc[[10, 11, 200, 400]] = np.nan
out = {}


def rec(name, fn):
    try:
        v = fn()
        if isinstance(v, (pd.Series, pd.DataFrame)):
            a = v.to_numpy(dtype=float).ravel()
            ok = a[~np.isnan(a)]
            v = {
                "n_valid": int(ok.size),
                "n_nan": int(a.size - ok.size),
                "sum": float(ok.sum()) if ok.size else None,
                "last": float(ok[-1]) if ok.size else None,
                "first_valid": int(np.argmax(~np.isnan(a))) if ok.size else None,
            }
        elif isinstance(v, (np.floating, float, int, np.integer)):
            v = float(v)
        out[name] = {"v": v}
    except Exception as e:  # noqa: BLE001
        out[name] = {"v": f"ERR {type(e).__name__}: {str(e)[:80]}"}


# ---- 1.1.1 ----
r252 = lambda s: s.vbt.returns(freq="1D", year_freq="252 days")
rec("dsr_fat_tail_bias_false", lambda: r252(fat).deflated_sharpe_ratio(nb_trials=48, var_sharpe=252 / 499, bias=False))
rec("dsr_fat_tail_bias_true", lambda: r252(fat).deflated_sharpe_ratio(nb_trials=48, var_sharpe=252 / 499, bias=True))
rec("dsr_with_nan_returns", lambda: r252(fat_nan).deflated_sharpe_ratio(nb_trials=48, var_sharpe=252 / 499, bias=False))
big = pd.Series(1e6 + rng.normal(0, 1e-2, 400), idx)
rec("rolling_std_large_offset", lambda: big.vbt.rolling_std(20, ddof=1) if hasattr(big.vbt, "rolling_std") else vbt.MSTD.run(big, 20).mstd)
rec("mstd_window20_ddof1", lambda: vbt.MSTD.run(close, 20, ddof=1).mstd)
rec("bbands_ddof1_lower", lambda: vbt.BBANDS.run(close, 20, alpha=2.0, ddof=1).lower)
rec("expanding_mean_minp_gt_len", lambda: close.vbt.expanding_mean(minp=500))
rec("expanding_std_minp_gt_len", lambda: close.vbt.expanding_std(minp=500))
rec("drawdowns_empty_input", lambda: pd.Series([], dtype=float).vbt.drawdowns.max_drawdown())
# ---- 1.1.2 ----
rec("cond_var_with_nan", lambda: r252(fat_nan).cond_value_at_risk(cutoff=0.05))
rec("cond_var_clean", lambda: r252(fat).cond_value_at_risk(cutoff=0.05))
rec("value_at_risk_nan", lambda: r252(fat_nan).value_at_risk(cutoff=0.05))
rec("is_sorted_nb_nan", lambda: float(vbt.generic.nb.is_sorted_nb(np.array([1.0, np.nan, 2.0]))))
# ---- 1.2.0 ----
rec("mstd_ewm_true", lambda: vbt.MSTD.run(close, 20, ewm=True).mstd)
rec("ewm_std_span", lambda: close.vbt.ewm_std(span=20) if hasattr(close.vbt, "ewm_std") else close.ewm(span=20).std())
rec("obv_ties", lambda: vbt.OBV.run(pd.Series([1, 2, 2, 2, 3, 1, 1.0] * 5, name="c"), pd.Series([10.0] * 35)).obv)
tied_close = pd.Series(np.round(close.to_numpy() / 2) * 2, idx)
rec("obv_ties_panel", lambda: vbt.OBV.run(tied_close, volume).obv)
ent = pd.Series(False, idx)
ext = pd.Series(False, idx)
ent.iloc[[5, 60, 120, 180, 260]] = True
ext.iloc[[30, 90, 150, 200, 300]] = True
flat_close = close.copy()
flat_close.iloc[60:91] = flat_close.iloc[60]  # 60->90 trade exactly breakeven before fees
pf = vbt.Portfolio.from_signals(flat_close, ent, ext, fees=0.0, freq="1D")
rec("trades_expectancy_breakeven", lambda: pf.trades.expectancy())
rec("trades_profit_factor", lambda: pf.trades.profit_factor())
rec("trades_win_rate", lambda: pf.trades.win_rate())
trail = ret.copy()
trail.iloc[-30:] = np.nan
rec("maxdd_returns_trailing_nan", lambda: trail.vbt.returns(freq="1D").max_drawdown())
rec("maxdd_value_trailing_nan", lambda: close.where(np.arange(400) < 370).vbt.drawdowns.max_drawdown())
rec("drawdown_records_trailing_nan", lambda: float(close.where(np.arange(400) < 370).vbt.drawdowns.records_readable.shape[0]))
rec("ulcer_index_exists", lambda: float(r252(ret).ulcer_index()) if hasattr(r252(ret), "ulcer_index") else "absent")
# ---- master (post 1.2.0) ----
flat = pd.Series(0.0, idx[:100])
rec("sharpe_flat_zero_returns", lambda: r252(flat).sharpe_ratio())
rec("sortino_flat_zero_returns", lambda: r252(flat).sortino_ratio())
pos_const = pd.Series(0.001, idx[:100])
rec("sharpe_const_positive", lambda: r252(pos_const).sharpe_ratio())
rec("sortino_const_positive", lambda: r252(pos_const).sortino_ratio())
rec("rolling_min", lambda: close.vbt.rolling_min(20))
rec("rolling_max", lambda: close.vbt.rolling_max(20))
ui = (close * 100).astype("uint32")
rec("atr_unsigned_ints", lambda: vbt.ATR.run((ui + 5).astype("uint32"), (ui - 5).astype("uint32"), ui, 14).tr)
rec("diff_int8_overflow", lambda: pd.Series(np.array([100, -100, 100, -100], dtype="int8")).vbt.diff())
# ---- sanity anchors (should never move) ----
rec("sharpe_252_normal", lambda: r252(ret).sharpe_ratio())
rec("sortino_252_normal", lambda: r252(ret).sortino_ratio())
rec("ann_return_252", lambda: r252(ret).annualized())
rec("ann_vol_252", lambda: r252(ret).annualized_volatility())
rec("maxdd_normal", lambda: r252(ret).max_drawdown())
rec("rsi_wilder27", lambda: vbt.RSI.run(close, 27, ewm=True).rsi)
rec("atr14_sma", lambda: vbt.ATR.run(high, low, close, 14, ewm=False).atr)
rec("macd_ewm_hist", lambda: vbt.MACD.run(close, 12, 26, 9, macd_ewm=True, signal_ewm=True).hist)
rec("ma20", lambda: vbt.MA.run(close, 20).ma)
rec("bb_default_lower", lambda: vbt.BBANDS.run(close, 20, alpha=2).lower)
out["_versions"] = {"vbt": vbt.__version__}
print(json.dumps(out, sort_keys=True))
