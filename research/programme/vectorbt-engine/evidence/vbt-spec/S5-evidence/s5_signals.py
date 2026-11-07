"""S5 prototype step 2: per-voter ensemble signals under one implementation arm.

Arms (one change at a time, so differences are attributable):
  A  ggt     : ggTrader indicators.py, run in the project venv (pandas 2.3.3)  [legacy]
  B  ggt     : the same file, run in venv120 (pandas 3.1.1)                     [stack]
  C  vbt_m   : vbt 1.2.0 indicator kernels configured to match the old convention
               (BBANDS ddof=1, RSI window=2n-1 ewm, MACD macd_ewm=signal_ewm=True,
               MA ewm) + ggTrader's own prev/now crossover comparisons          [kernels]
  D  vbt_mx  : C + vbt's own crossed_above / crossed_below semantics           [crossovers]
  E  vbt_def : vbt defaults (BBANDS ddof=0, RSI SMA window=14, MACD SMA) + vbt
               crossovers                                                      [defaults]
Deployed parameters (EnsembleSignal defaults, live signal_runner):
  bb 20/2.0, rsi 14/30/50, ema 20/50, macd 12/26/9 div 20, vbb vol 20 x2.0.
Writes one parquet per (voter, side) of booleans, plus a live-window panel:
for each of the last K sessions, signals on the last bar of a 365-calendar-day
slice (what signal_runner.generate_signals sees).
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd

P = dict(bb_p=20, bb_k=2.0, rsi_p=14, rsi_os=30, rsi_ex=50, ema_f=20, ema_s=50,
         m_f=12, m_s=26, m_sig=9, div=20, vol_p=20, vol_m=2.0)
VOTERS = ("bb", "rsi", "ema", "macd", "vbb")


def load_ggt(path: str):
    spec = importlib.util.spec_from_file_location("ggt_indicators", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ggt_voters(ind, close, volume):
    out = {}
    out["bb"] = ind.bb_signals(close, P["bb_p"], P["bb_k"])
    out["rsi"] = ind.rsi_signals(close, P["rsi_p"], P["rsi_os"], P["rsi_ex"])
    out["ema"] = ind.ema_signals(close, P["ema_f"], P["ema_s"])
    out["macd"] = ind.macd_signals(close, P["m_f"], P["m_s"], P["m_sig"], P["div"])
    out["vbb"] = ind.volume_bb_signals(close, volume, P["bb_p"], P["bb_k"], P["vol_p"], P["vol_m"])
    return out


def _x_below(a, b):  # ggTrader convention: prev >= b and now < b
    return ((a.shift(1) >= b.shift(1)) & (a < b)).fillna(False).astype(bool)


def _x_above(a, b):  # ggTrader convention: prev < b and now >= b
    return ((a.shift(1) < b.shift(1)) & (a >= b)).fillna(False).astype(bool)


def vbt_voters(close, volume, matched: bool, vbt_cross: bool):
    import vectorbt as vbt

    def frame(x):
        return pd.DataFrame(np.asarray(x), index=close.index, columns=close.columns)

    def xb(a, b):
        if vbt_cross:
            b = b if isinstance(b, pd.DataFrame) else pd.DataFrame(b, index=a.index, columns=a.columns)
            return frame(a.vbt.crossed_below(b))
        return _x_below(a, b if isinstance(b, pd.DataFrame) else pd.DataFrame(b, index=a.index, columns=a.columns))

    def xa(a, b):
        if vbt_cross:
            b = b if isinstance(b, pd.DataFrame) else pd.DataFrame(b, index=a.index, columns=a.columns)
            return frame(a.vbt.crossed_above(b))
        return _x_above(a, b if isinstance(b, pd.DataFrame) else pd.DataFrame(b, index=a.index, columns=a.columns))

    ddof = 1 if matched else 0
    bb = vbt.BBANDS.run(close, window=P["bb_p"], alpha=P["bb_k"], ddof=ddof)
    lower, mid = frame(bb.lower), frame(bb.middle)
    if matched:
        rsi = frame(vbt.RSI.run(close, window=2 * P["rsi_p"] - 1, ewm=True).rsi)
    else:
        rsi = frame(vbt.RSI.run(close, window=P["rsi_p"]).rsi)
    ma = vbt.MA.run(close, window=[P["ema_f"], P["ema_s"]], ewm=True)
    ema_f = frame(ma.ma.iloc[:, : close.shape[1]])
    ema_s = frame(ma.ma.iloc[:, close.shape[1]:])
    if matched:
        mc = vbt.MACD.run(close, P["m_f"], P["m_s"], P["m_sig"], macd_ewm=True, signal_ewm=True)
    else:
        mc = vbt.MACD.run(close, P["m_f"], P["m_s"], P["m_sig"])
    hist = frame(mc.hist)
    vol_avg = frame(vbt.MA.run(volume, window=P["vol_p"]).ma)

    out = {}
    out["bb"] = (xb(close, lower), xa(close, mid))
    out["rsi"] = (xb(rsi, float(P["rsi_os"])), xa(rsi, float(P["rsi_ex"])))
    out["ema"] = (xa(ema_f, ema_s), xb(ema_f, ema_s))
    price_low = frame(close.vbt.rolling_min(P["div"], minp=P["div"]))
    hist_low = frame(hist.vbt.rolling_min(P["div"], minp=P["div"]))
    ent = ((close <= price_low) & (hist > hist_low)).fillna(False).astype(bool)
    ent.iloc[: P["m_s"] + P["m_sig"] + P["div"]] = False  # policy mask kept from ggTrader
    out["macd"] = (ent, xb(hist, 0.0))
    spike = (volume > P["vol_m"] * vol_avg).fillna(False)
    out["vbb"] = ((xb(close, lower) & spike).astype(bool), xa(close, mid))
    return out


def ensemble(v, min_agree=2, min_agree_exit=2):
    ev = sum(v[k][0].astype(int) for k in VOTERS)
    xv = sum(v[k][1].astype(int) for k in VOTERS)
    return (ev >= min_agree), (v["rsi"][1] | (xv >= min_agree_exit))


def run(arm, close, volume, ind):
    if arm == "ggt":
        return ggt_voters(ind, close, volume)
    if arm == "vbt_m":
        return vbt_voters(close, volume, matched=True, vbt_cross=False)
    if arm == "vbt_mx":
        return vbt_voters(close, volume, matched=True, vbt_cross=True)
    if arm == "vbt_def":
        return vbt_voters(close, volume, matched=False, vbt_cross=True)
    raise ValueError(arm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--ggt-indicators", default="/home/flynn/ggTrader/src/ggTrader/lab/strategies/indicators.py")
    ap.add_argument("--live-sessions", type=int, default=120)
    a = ap.parse_args()
    close = pd.read_parquet(os.path.join(a.evidence, "tape_close.parquet"))
    volume = pd.read_parquet(os.path.join(a.evidence, "tape_volume.parquet"))
    ind = load_ggt(a.ggt_indicators) if a.arm == "ggt" else None
    out_dir = os.path.join(a.evidence, "arms", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    t = time.perf_counter()
    v = run(a.arm, close, volume, ind)
    t_full = time.perf_counter() - t
    v["ENS"] = ensemble(v)
    for k, (e, x) in v.items():
        e.astype(bool).to_parquet(os.path.join(out_dir, f"{k}_entries.parquet"))
        x.astype(bool).to_parquet(os.path.join(out_dir, f"{k}_exits.parquet"))
    # Live-window panel: 365 calendar days ending at each of the last K sessions.
    rows_e, rows_x = {}, {}
    for asof in close.index[-a.live_sessions:]:
        sl = close.loc[asof - pd.Timedelta(days=365): asof]
        vs = volume.loc[sl.index]
        sl = sl.loc[:, sl.notna().sum() >= 60]  # signal_runner min_history_bars=60
        vv = run(a.arm, sl, vs[sl.columns], ind)
        e, x = ensemble(vv)
        rows_e[asof] = e.iloc[-1].reindex(close.columns, fill_value=False)
        rows_x[asof] = x.iloc[-1].reindex(close.columns, fill_value=False)
    pd.DataFrame(rows_e).T.astype(bool).to_parquet(os.path.join(out_dir, "LIVE_entries.parquet"))
    pd.DataFrame(rows_x).T.astype(bool).to_parquet(os.path.join(out_dir, "LIVE_exits.parquet"))
    meta = {"arm": a.arm, "tag": a.tag, "pandas": pd.__version__, "full_tape_s": round(t_full, 2)}
    try:
        import vectorbt as vbt

        meta["vectorbt"] = vbt.__version__
        if hasattr(vbt, "settings") and "engine" in vbt.settings:
            meta["engine_setting"] = vbt.settings["engine"]
    except Exception:
        meta["vectorbt"] = None
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
