#!/usr/bin/env python3
"""ML pre-screen: evaluate a signal strategy's entry quality via LightGBM.

Usage:
    python scripts/ml_signal_screen.py --signal macd_divergence
    python scripts/ml_signal_screen.py --signal volume_bb_reversion --start 2022-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ggTrader.lab.data import equity_universe_between, fetch_stock_ohlcv
from ggTrader.lab.strategies.indicators import extract_close, extract_volume
from ggTrader.lab.strategies.signals import build_signal_strategy
from ggTrader.lab.strategy import LabConfig
from ggTrader.paper.feature_gate import FEATURE_NAMES, extract_features


def collect_entries(
    signal_name: str,
    ohlcv: pd.DataFrame,
    symbols: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Generate entries for a signal and build a feature+label DataFrame."""
    cfg = LabConfig(min_history_bars=50)
    strat = build_signal_strategy(signal_name, cfg)

    plans = {pd.Timestamp(start, tz="UTC"): [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)

    close_df = extract_close(ohlcv, symbols)
    vol_df = extract_volume(ohlcv, symbols)

    rows = []
    for sym in symbols:
        if sym not in targets.entries.columns:
            continue
        entry_bars = targets.entries.index[targets.entries[sym]]
        close_s = close_df[sym].dropna()
        vol_s = (
            vol_df[sym].dropna() if sym in vol_df.columns else pd.Series(1.0, index=close_s.index)
        )

        for bar in entry_bars:
            if bar not in close_s.index:
                continue
            feats = extract_features(close_s, vol_s, bar)
            bar_idx = close_s.index.get_loc(bar)
            if bar_idx + 5 >= len(close_s):
                continue
            fwd_ret = close_s.iloc[bar_idx + 5] / close_s.iloc[bar_idx] - 1.0
            feats["label"] = int(fwd_ret > 0)
            feats["symbol"] = sym
            feats["bar_date"] = str(bar.date())
            rows.append(feats)

    return pd.DataFrame(rows)


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train LightGBM with 5-fold time-series CV, return metrics."""
    import lightgbm as lgb
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import TimeSeriesSplit

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    tscv = TimeSeriesSplit(n_splits=5)
    all_preds = np.zeros(len(y))
    all_true = np.zeros(len(y))
    mask = np.zeros(len(y), dtype=bool)

    model = None
    for train_idx, test_idx in tscv.split(X):
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=20,
            verbose=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        all_preds[test_idx] = preds
        all_true[test_idx] = y[test_idx]
        mask[test_idx] = True

    y_true = all_true[mask]
    y_pred = all_preds[mask]

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist())) if model else {}

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_samples": int(mask.sum()),
        "n_positive": int(y_true.sum()),
        "feature_importances": importances,
    }


def main():
    parser = argparse.ArgumentParser(description="ML pre-screen for signal quality")
    parser.add_argument("--signal", required=True, help="Signal strategy name")
    parser.add_argument("--start", default="2021-01-01", help="Data start date")
    parser.add_argument("--end", default=None, help="Data end date (default: now)")
    parser.add_argument("--universe", default="sp500", help="Stock universe")
    args = parser.parse_args()

    end = args.end or str(pd.Timestamp.now(tz="UTC").normalize().date())
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    print(f"ML Pre-Screen: {args.signal}")
    print(f"Universe: {args.universe} | {args.start} -> {end}")
    print()

    symbols = equity_universe_between(start_ts, end_ts, universe=args.universe)
    print(f"Loading OHLCV for {len(symbols)} symbols...")
    ohlcv = fetch_stock_ohlcv(symbols, start=args.start, end=end)

    sym_cols = sorted(ohlcv.columns.get_level_values(0).unique())
    print(f"Generating entries for {args.signal}...")
    df = collect_entries(args.signal, ohlcv, sym_cols, args.start, end)

    if len(df) < 50:
        print(f"Only {len(df)} entries — too few for meaningful ML evaluation.")
        sys.exit(1)

    print(f"Training LightGBM on {len(df)} entries...")
    results = train_and_evaluate(df)

    prec = results["precision"]
    if prec < 0.50:
        verdict = "DROP"
    elif prec < 0.55:
        verdict = "BORDERLINE"
    else:
        verdict = "STRONG"

    print()
    print(f"{'Signal':<25} {args.signal}")
    print(f"{'Precision':<25} {prec:.4f}")
    print(f"{'Recall':<25} {results['recall']:.4f}")
    print(f"{'F1':<25} {results['f1']:.4f}")
    print(f"{'Samples':<25} {results['n_samples']}")
    print(f"{'Positive rate':<25} {results['n_positive'] / results['n_samples']:.2%}")
    print(f"{'Verdict':<25} {verdict}")
    print()
    print("Top features:")
    sorted_feats = sorted(results["feature_importances"].items(), key=lambda x: -x[1])
    for feat, imp in sorted_feats[:5]:
        print(f"  {feat:<20} {imp}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = results_dir / f"ml_screen_{args.signal}_{ts}.json"
    results["signal"] = args.signal
    results["verdict"] = verdict
    results["start"] = args.start
    results["end"] = end
    results["universe"] = args.universe
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {outpath}")


if __name__ == "__main__":
    main()
