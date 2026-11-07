"""D1: ~17 rolling folds (12m train / 3m test) with vbt's own splitter + IndicatorFactory grid.

Runs on synthetic data. Reports: fold count, what the splitter gives, whether the 48-combo grid x
folds x symbols is ONE from_signals call, wall time and peak RSS.

Usage: python d1_wfo_splitter.py [--symbols 30] [--no-stack]
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.insert(0, "/home/flynn/ggTrader/src")
from ggTrader.lab.wfo import generate_folds  # read-only use, for fold dates  # noqa: E402


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def synth(n_sym: int, start="2020-01-01", end="2026-06-30", seed=0) -> pd.DataFrame:
    idx = pd.bdate_range(start, end, tz="UTC")
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.015, size=(len(idx), n_sym))
    # mild mean reversion so signals are not pure noise
    for t in range(1, len(idx)):
        r[t] -= 0.05 * r[t - 1]
    px = 100 * np.exp(np.cumsum(r, axis=0))
    return pd.DataFrame(px, index=idx, columns=[f"S{i:03d}" for i in range(n_sym)])


# --- custom indicator: 4-voter ensemble, 3*2*2*2*2 = 48 combos --------------------------------
def ens_apply(close, min_agree, min_agree_exit, bb_alpha, rsi_oversold, ema_fast):
    """Per-combo apply func. close is (T, cols) ndarray. Uses vbt's own numba kernels."""
    from vectorbt.generic import nb as gnb
    from vectorbt.indicators import nb as inb

    # RSI (vbt kernel, Wilder-equivalent needs span=2n-1, see d3 script)
    up = np.where(np.diff(close, axis=0, prepend=np.nan) > 0, np.diff(close, axis=0, prepend=np.nan), 0.0)
    dn = np.where(np.diff(close, axis=0, prepend=np.nan) < 0, -np.diff(close, axis=0, prepend=np.nan), 0.0)
    rs = inb.ma_nb(up, 27, True) / inb.ma_nb(dn, 27, True)
    rsi = 100 - 100 / (1 + rs)
    ma = gnb.rolling_mean_nb(close, 20, minp=20)
    sd = gnb.rolling_std_nb(close, 20, minp=20, ddof=1)
    lower = ma - bb_alpha * sd
    ef = gnb.ewm_mean_nb(close, int(ema_fast), minp=0, adjust=False)
    es = gnb.ewm_mean_nb(close, 50, minp=0, adjust=False)

    def cross_below(a, b):
        return (a < b) & (np.roll(a, 1, 0) >= np.roll(b, 1, 0))

    def cross_above(a, b):
        return (a >= b) & (np.roll(a, 1, 0) < np.roll(b, 1, 0))

    thr = np.full_like(close, rsi_oversold)
    mid = np.full_like(close, 50.0)
    e_votes = (
        cross_below(rsi, thr).astype(np.int8)
        + cross_below(close, lower).astype(np.int8)
        + cross_above(ef, es).astype(np.int8)
        + (close <= gnb.rolling_min_nb(close, 10, minp=10)).astype(np.int8)
    )
    x_votes = cross_above(rsi, mid).astype(np.int8) + cross_above(close, ma).astype(np.int8)
    entries = e_votes >= min_agree
    exits = x_votes >= min_agree_exit
    entries[:60] = False  # warm-up guard
    return entries, exits


Ens = vbt.IndicatorFactory(
    class_name="Ens",
    input_names=["close"],
    param_names=["min_agree", "min_agree_exit", "bb_alpha", "rsi_oversold", "ema_fast"],
    output_names=["entries", "exits"],
).from_apply_func(ens_apply)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=30)
    ap.add_argument("--warm", type=int, default=60, help="warm-up bars before each train window")
    ap.add_argument("--mem-only", action="store_true", help="stop after Portfolio call")
    args = ap.parse_args()

    close = synth(args.symbols)
    print(f"data: {close.shape}, {close.index[0].date()} .. {close.index[-1].date()}")

    # 1. ggTrader's own calendar folds (reference)
    folds = generate_folds(pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC"))
    print(f"ggTrader generate_folds -> {len(folds)} folds; first {folds[0]}, last {folds[-1]}")

    # 2. vbt rolling_split, FIXED-LENGTH variant (what the API is natively good at)
    train_bars, test_bars = 252, 63
    tr, te = close.vbt.rolling_split(window_len=train_bars + test_bars, set_lens=(train_bars,), n=17)
    print("vbt rolling_split(n=17): train", tr[0].shape, "test", te[0].shape)
    starts = [ix[0].date() for ix in tr[1]]
    gaps = np.diff([close.index.get_loc(ix[0]) for ix in tr[1]])
    print("  start dates:", starts[:4], "... stride in bars (linspace, NOT fixed):", gaps.tolist())
    print("  returned index is reset:", tr[0].index[:3].tolist(), "(DatetimeIndex lost; dates in tr[1])")

    # 3. vbt range_split with explicit calendar boundaries -> true 12m/3m calendar folds
    idx = close.index
    wstart = np.array([max(idx.searchsorted(f.train_start) - args.warm, 0) for f in folds])
    wend = np.array([idx.searchsorted(f.test_end, side="right") - 1 for f in folds])
    win, win_idx = close.vbt.range_split(start_idxs=wstart, end_idxs=wend)
    print(f"vbt range_split(calendar): {win.shape} (ragged windows are NaN-padded at the tail)")
    print("  window lengths:", [len(w) for w in win_idx][:6], "...")
    n_split = len(folds)
    n_sym = close.shape[1]

    # 4. ONE indicator run over all folds x symbols x 48 combos (param_product -> columns)
    t0 = time.time()
    ens = Ens.run(
        win,
        min_agree=[2, 3, 4],
        min_agree_exit=[1, 2],
        bb_alpha=[2.0, 2.5],
        rsi_oversold=[25, 30],
        ema_fast=[10, 20],
        param_product=True,
    )
    print(f"Ens.run: entries {ens.entries.shape}; column levels {list(ens.entries.columns.names)}; "
          f"{time.time()-t0:.1f}s, peak RSS {rss_mb():.0f} MB")

    # 5. ONE Portfolio call: every (combo, fold, symbol) is a column; cash shared per (combo, fold)
    group_levels = [0, 1, 2, 3, 4, 5]  # 5 param levels + split_idx; level 6 = symbol (None name)
    cols = ens.entries.columns
    # win has columns (split_idx, symbol); Ens adds param levels in front
    print("  levels:", list(cols.names))
    # simplest correct broadcast: let vbt broadcast close (T, folds*syms) against entries
    t0 = time.time()
    pf = vbt.Portfolio.from_signals(
        win, ens.entries, ens.exits, size=0.03, size_type="percent", fees=0.0005, slippage=0.0005,
        init_cash=100_000.0, cash_sharing=True, group_by=group_levels, freq="D",
    )
    print(f"Portfolio.from_signals: {pf.wrapper.shape_2d} cols -> {pf.wrapper.grouper.get_group_count()} groups; "
          f"{time.time()-t0:.1f}s, peak RSS {rss_mb():.0f} MB")

    if args.mem_only:
        print(f"MEM_ONLY symbols={args.symbols} cols={pf.wrapper.shape_2d[1]} peak_rss_mb={rss_mb():.0f}")
        return
    # 6. per-fold TRAIN-window Sharpe and best-combo selection
    rets = pf.returns()  # (T, groups)
    print("returns:", rets.shape)
    dates = pd.DataFrame({k: pd.Series(w) for k, w in enumerate(win_idx)})  # position -> date per split
    best = []
    for k, f in enumerate(folds):
        d = win_idx[k]
        pos_tr = np.flatnonzero((d >= f.train_start) & (d < f.train_end))
        pos_te = np.flatnonzero((d >= f.test_start) & (d <= f.test_end))
        r_k = rets.xs(k, axis=1, level="split_idx")  # rets columns: 6 levels (params, split_idx)
        sh_tr = r_k.iloc[pos_tr].vbt.returns(freq="D", year_freq="252 days").sharpe_ratio()
        sh_te = r_k.iloc[pos_te].vbt.returns(freq="D", year_freq="252 days").sharpe_ratio()
        # TRAP: vbt returns +inf Sharpe when std == 0 (a combo that never traded). idxmax would pick it.
        n_inf = int(np.isinf(sh_tr).sum())
        sh_tr = sh_tr.replace([np.inf, -np.inf], np.nan)
        sh_te = sh_te.replace([np.inf, -np.inf], np.nan)
        win_combo = sh_tr.idxmax()
        best.append((k + 1, win_combo, float(sh_tr.max()), float(sh_te[win_combo]), n_inf))
    out = pd.DataFrame(best, columns=["fold", "winner(min_agree,exit,bb,rsi,ema)", "train_sharpe", "test_sharpe", "n_inf_train"])
    print(out.to_string(index=False))
    print(f"peak RSS {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
