"""Offline training for the ML feature gate.

Generates entry labels from historical ensemble signals, extracts features,
trains a LightGBM classifier with TimeSeriesSplit, and saves the model.

Usage:
    docker compose run --rm ggtrader_live python -m ggTrader.lab.train_gate
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ggTrader.lab.data import fetch_stock_ohlcv
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig
from ggTrader.paper.feature_gate import DEFAULT_THRESHOLD, FEATURE_NAMES, extract_features

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_PATH = MODEL_DIR / "ensemble_gate.joblib"
META_PATH = MODEL_DIR / "ensemble_gate_meta.json"

TRAIN_START = "2019-01-01"
FORWARD_DAYS = 5
N_SPLITS = 5

LGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "random_state": 42,
}


def _build_dataset(ohlcv: pd.DataFrame, entries: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix + labels from ensemble entry signals."""
    records: list[dict] = []
    sym_cols = list(entries.columns)
    available_syms = set(ohlcv.columns.get_level_values(0))
    close_frames = {s: ohlcv[s]["close"].dropna() for s in sym_cols if s in available_syms}
    volume_frames: dict = {}
    high_frames: dict = {}
    low_frames: dict = {}
    for s in sym_cols:
        if s not in available_syms:
            continue
        v = ohlcv[s].get("volume")
        volume_frames[s] = (
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
            if not entries.loc[bar_date, symbol]:
                continue
            if symbol not in close_frames:
                continue

            close = close_frames[symbol]
            volume = volume_frames[symbol]

            if bar_date not in close.index:
                continue
            bar_idx = close.index.get_loc(bar_date)
            if bar_idx < 20:
                continue

            # 5-day forward return label
            future_idx = bar_idx + FORWARD_DAYS
            if future_idx >= len(close):
                continue
            fwd_ret = close.iloc[future_idx] / close.iloc[bar_idx] - 1.0
            label = 1 if fwd_ret > 0 else 0

            high = high_frames.get(symbol)
            low = low_frames.get(symbol)
            feats = extract_features(
                close.iloc[: bar_idx + 1],
                volume.iloc[: bar_idx + 1],
                bar_date,
                high=high.iloc[: bar_idx + 1] if high is not None else None,
                low=low.iloc[: bar_idx + 1] if low is not None else None,
            )
            feats["label"] = label
            feats["date"] = bar_date
            feats["symbol"] = symbol
            records.append(feats)

    return pd.DataFrame(records)


def train() -> None:
    """Run the full training pipeline."""
    print("=" * 60)
    print("ML Feature Gate — Training Pipeline")
    print("=" * 60)

    # 1. Fetch data
    print(f"\n[1/6] Fetching SP500 OHLCV from {TRAIN_START}...")
    from ggTrader.data.core.index_constituents import normalize_yf_ticker, sp500_members_asof

    today = pd.Timestamp.now(tz="UTC").normalize()
    members = sp500_members_asof(today)
    symbols = sorted({normalize_yf_ticker(t) for t in members})
    ohlcv = fetch_stock_ohlcv(symbols, start=TRAIN_START)
    print(
        f"    Loaded {len(ohlcv.columns.get_level_values(0).unique())} symbols, {len(ohlcv)} bars"
    )

    # 2. Generate ensemble entry signals
    print("\n[2/6] Generating ensemble entry signals...")
    sym_cols = list(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in sym_cols}, axis=1)
    volume = pd.concat(
        {s: ohlcv[s].get("volume", pd.Series(1.0, index=ohlcv.index)) for s in sym_cols}, axis=1
    )

    cfg = LabConfig(min_history_bars=60)
    ensemble = EnsembleSignal(cfg)
    from ggTrader.lab.strategy import SignalTargets

    targets: SignalTargets = ensemble._generate_signals(close, volume)
    n_entries = int(targets.entries.sum().sum())
    print(f"    Found {n_entries} entry signals")

    # 3. Build feature matrix
    print("\n[3/6] Building feature matrix + labels...")
    df = _build_dataset(ohlcv, targets.entries)
    print(f"    Dataset: {len(df)} samples, {df['label'].mean():.1%} positive")

    if len(df) < 100:
        print("    ERROR: Too few samples to train. Aborting.")
        return

    # 4. Train with TimeSeriesSplit
    print(f"\n[4/6] Training LightGBM ({N_SPLITS}-fold TimeSeriesSplit)...")
    df = df.sort_values("date").reset_index(drop=True)
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.55).astype(int)

        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / len(y_test)
        coverage = (tp + fp) / len(y_test)

        fold_metrics.append(
            {
                "fold": fold_i + 1,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "coverage": round(coverage, 4),
            }
        )
        print(
            f"    Fold {fold_i + 1}: acc={accuracy:.3f} prec={precision:.3f} "
            f"rec={recall:.3f} cov={coverage:.3f}"
        )

    # 5. Train final model on all data
    print("\n[5/6] Training final model on full dataset...")
    final_model = lgb.LGBMClassifier(**LGB_PARAMS)
    final_model.fit(X, y)

    importance = dict(zip(FEATURE_NAMES, final_model.feature_importances_.tolist()))
    print("    Feature importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"      {feat:15s}: {imp}")

    # 6. Save
    print(f"\n[6/6] Saving model to {MODEL_PATH}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Safe: we only load this model ourselves from a trusted local path
    joblib.dump(final_model, MODEL_PATH)

    meta = {
        "trained_at": str(pd.Timestamp.now()),
        "train_start": TRAIN_START,
        "n_samples": len(df),
        "positive_rate": round(float(df["label"].mean()), 4),
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "forward_days": FORWARD_DAYS,
        "threshold": DEFAULT_THRESHOLD,
        "lgb_params": LGB_PARAMS,
        "fold_metrics": fold_metrics,
        "feature_importance": importance,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, default=str))
    print(f"    Metadata saved to {META_PATH}")

    # Summary
    avg_prec = np.mean([m["precision"] for m in fold_metrics])
    avg_cov = np.mean([m["coverage"] for m in fold_metrics])
    print("\n" + "=" * 60)
    print(f"DONE. Avg precision={avg_prec:.3f}, avg coverage={avg_cov:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    train()
