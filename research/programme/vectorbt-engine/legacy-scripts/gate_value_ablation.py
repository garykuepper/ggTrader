#!/usr/bin/env python
"""ML gate value ablation: does the LightGBM feature gate select better entries?

The production gate (models/ensemble_gate.joblib) filters live ensemble buys by
predicted P(profitable) >= 0.55. Its value has only ever been measured by
classification precision, never by its effect on realized returns. This script
asks the decisive, return-based question on out-of-sample entries:

    Do the entries the gate KEEPS earn a higher forward return than the entries
    it BLOCKS (and than taking ALL entries ungated)?

It reuses the exact deployed entry generation (EnsembleSignal) and features
(feature_gate.extract_features), trains the gate with the same TimeSeriesSplit
the production model uses, and on each held-out fold compares the realized
5-day forward return of kept vs blocked entries. If kept ~= blocked ~= all, the
gate is a near-random filter that only cuts trade count — net-negative.
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ggTrader.lab.data import fetch_stock_ohlcv
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig, SignalTargets
from ggTrader.lab.train_gate import FORWARD_DAYS, LGB_PARAMS, N_SPLITS, TRAIN_START
from ggTrader.paper.feature_gate import DEFAULT_THRESHOLD, FEATURE_NAMES, extract_features


def build_entry_returns(ohlcv: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    """Ensemble entries -> features + CONTINUOUS forward return (not just sign)."""
    records: list[dict] = []
    sym_cols = list(entries.columns)
    available = set(ohlcv.columns.get_level_values(0))
    close_frames = {s: ohlcv[s]["close"].dropna() for s in sym_cols if s in available}
    vol_frames: dict = {}
    high_frames: dict = {}
    low_frames: dict = {}
    for s in sym_cols:
        if s not in available:
            continue
        v = ohlcv[s].get("volume")
        vol_frames[s] = (
            v.dropna()
            if v is not None and not v.empty
            else pd.Series(1.0, index=close_frames[s].index)
        )
        h = ohlcv[s].get("high")
        high_frames[s] = h.dropna() if h is not None and not h.empty else None
        lo = ohlcv[s].get("low")
        low_frames[s] = lo.dropna() if lo is not None and not lo.empty else None

    for bar_date in entries.index:
        for symbol in sym_cols:
            if not entries.loc[bar_date, symbol] or symbol not in close_frames:
                continue
            close = close_frames[symbol]
            if bar_date not in close.index:
                continue
            bar_idx = close.index.get_loc(bar_date)
            if bar_idx < 20 or bar_idx + FORWARD_DAYS >= len(close):
                continue
            fwd_ret = float(close.iloc[bar_idx + FORWARD_DAYS] / close.iloc[bar_idx] - 1.0)
            high = high_frames.get(symbol)
            low = low_frames.get(symbol)
            feats = extract_features(
                close.iloc[: bar_idx + 1],
                vol_frames[symbol].iloc[: bar_idx + 1],
                bar_date,
                high=high.iloc[: bar_idx + 1] if high is not None else None,
                low=low.iloc[: bar_idx + 1] if low is not None else None,
            )
            feats["label"] = int(fwd_ret > 0)
            feats["fwd_ret"] = fwd_ret
            feats["date"] = bar_date
            records.append(feats)

    return pd.DataFrame(records)


def _stats(rets: np.ndarray) -> dict:
    """Per-entry return stats (the 5-day forward return distribution)."""
    if len(rets) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "win": float("nan"),
            "sharpe": float("nan"),
        }
    mean = float(np.mean(rets))
    std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    return {
        "n": len(rets),
        "mean": mean,
        "median": float(np.median(rets)),
        "win": float(np.mean(rets > 0)),
        # per-entry Sharpe of the 5-day return (mean/std), not annualized
        "sharpe": mean / std if std > 0 else float("nan"),
    }


def _row(label: str, s: dict) -> str:
    return (
        f"| {label:<28} | {s['n']:>6} | {s['mean'] * 100:>7.3f}% | {s['median'] * 100:>7.3f}% "
        f"| {s['win'] * 100:>5.1f}% | {s['sharpe']:>6.3f} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="ML gate value ablation (return-based, OOS)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--start", default=TRAIN_START)
    args = ap.parse_args()

    from ggTrader.data.core.index_constituents import normalize_yf_ticker, sp500_members_asof

    today = pd.Timestamp.now(tz="UTC").normalize()
    symbols = sorted({normalize_yf_ticker(t) for t in sp500_members_asof(today)})
    print(f"[load] SP500 {len(symbols)} symbols from {args.start}", flush=True)
    ohlcv = fetch_stock_ohlcv(symbols, start=args.start)

    sym_cols = list(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in sym_cols}, axis=1)
    volume = pd.concat(
        {s: ohlcv[s].get("volume", pd.Series(1.0, index=ohlcv.index)) for s in sym_cols}, axis=1
    )
    cfg = LabConfig(min_history_bars=60)
    targets: SignalTargets = EnsembleSignal(cfg)._generate_signals(close, volume)
    print(f"[entries] {int(targets.entries.sum().sum())} ensemble entry signals", flush=True)

    df = build_entry_returns(ohlcv, targets.entries).sort_values("date").reset_index(drop=True)
    print(f"[dataset] {len(df)} entries with forward returns", flush=True)

    X = df[FEATURE_NAMES].values
    y = df["label"].values
    fwd = df["fwd_ret"].values

    # Out-of-sample gate scores via the SAME TimeSeriesSplit the production model uses.
    proba = np.full(len(df), np.nan)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for tr, te in tscv.split(X):
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X[tr], y[tr])
        proba[te] = model.predict_proba(X[te])[:, 1]

    evald = ~np.isnan(proba)  # first fold is train-only under TimeSeriesSplit
    kept = evald & (proba >= args.threshold)
    blocked = evald & (proba < args.threshold)

    all_s = _stats(fwd[evald])
    kept_s = _stats(fwd[kept])
    blocked_s = _stats(fwd[blocked])
    coverage = kept.sum() / evald.sum() if evald.sum() else float("nan")

    print("\n" + "=" * 78)
    print("ML GATE VALUE ABLATION — out-of-sample 5-day forward return by gate decision")
    print("=" * 78)
    print(
        f"Threshold {args.threshold} | OOS entries evaluated: {int(evald.sum())} "
        f"| gate coverage: {coverage:.1%}"
    )
    print("\n| Bucket                       |      n |    mean |  median |   win | Sharpe |")
    print("| :--------------------------- | -----: | ------: | ------: | ----: | -----: |")
    print(_row("ALL entries (no gate)", all_s))
    print(_row("KEPT (gate buys)", kept_s))
    print(_row("BLOCKED (gate skips)", blocked_s))

    lift = kept_s["mean"] - all_s["mean"]
    sep = kept_s["mean"] - blocked_s["mean"]
    print(
        f"\nKEPT vs ALL mean-return lift: {lift * 100:+.3f}%   "
        f"KEPT vs BLOCKED separation: {sep * 100:+.3f}%"
    )
    verdict = (
        "ADDS VALUE — kept entries clearly out-earn blocked"
        if sep > 0.002 and lift > 0.0
        else "NEAR-RANDOM / NET-NEGATIVE — gate cuts ~{:.0%} of trades without earning it".format(
            1 - coverage
        )
    )
    print(f"VERDICT: {verdict}")
    print(
        "\nNote: 5-day forward return is the gate's own training objective; this measures "
        "entry selection, not full-hold P&L. Per-entry Sharpe is mean/std of the 5d return."
    )


if __name__ == "__main__":
    main()
