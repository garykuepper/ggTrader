# ML Feature Gate for Paper Trader

**Date:** 2026-06-22
**Status:** Approved
**Scope:** Paper trader signal filter — train offline, deploy as buy-signal gate

---

## Goal

Filter the ensemble strategy's buy signals through a LightGBM binary classifier before the paper trader executes orders. Only act on signals where the model predicts `P(profitable) > threshold`. This reduces false entries without requiring a new signal type.

## Architecture

```
signal_runner.py: generate_signals()
        │
        ▼ raw buys: ["AAPL", "MSFT", ...]
        │
  ┌─────┴─────┐
  │  ML Gate   │  feature_gate.py
  └─────┬─────┘
        │
        ▼ filtered buys: ["AAPL"]  (only P > threshold)
        │
  trader.py executes
```

## Components

### 1. Feature extraction — `src/ggTrader/paper/feature_gate.py`

Public interface:

```python
class FeatureGate:
    def __init__(self, model_path: Path | None = None, threshold: float = 0.55):
        """Load pre-trained model. If model_path is None or missing, gate is disabled."""

    def extract_features(self, close: pd.Series, volume: pd.Series, bar_date: pd.Timestamp) -> dict:
        """Compute features for one symbol at one bar."""

    def score(self, features: dict) -> float:
        """Return P(profitable) for a single entry signal."""

    def filter_buys(self, buys: list[str], ohlcv: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
        """Filter buy list, return (kept_buys, scores_dict)."""
```

**Features (10, OHLCV-only):**

| # | Feature | Description |
|---|---------|-------------|
| 1 | `vol_5d` | 5-day realized volatility (annualized std of returns) |
| 2 | `vol_20d` | 20-day realized volatility |
| 3 | `vol_ratio` | vol_5d / vol_20d — expansion/contraction regime |
| 4 | `rsi_14` | RSI(14) value at entry bar |
| 5 | `bb_pctb` | Bollinger Band %B — position within bands |
| 6 | `ret_5d` | 5-day trailing return into entry |
| 7 | `ret_20d` | 20-day trailing return into entry |
| 8 | `volume_ratio` | bar volume / 20-day avg volume |
| 9 | `atr_ratio` | ATR(5) / ATR(20) — short-term vs long-term range |
| 10 | `day_of_week` | 0-4 (Monday–Friday) |

All features are computed from data available at the entry bar (no lookahead).

### 2. Training script — `src/ggTrader/lab/train_gate.py`

CLI entry point (run manually, not in cron):

```bash
docker compose run --rm ggtrader_live python -m ggTrader.lab.train_gate
```

Training pipeline:
1. Fetch SP500 OHLCV from DB (2019-01-01 to present)
2. Run EnsembleSignal over the full history to identify all entry bars
3. For each entry: compute 5-day forward return → label = 1 if return > 0, else 0
4. Extract 10 features at each entry bar
5. Train LightGBM with TimeSeriesSplit (5 folds, no shuffle)
6. Report per-fold: accuracy, precision, recall, F1, and simulated Sharpe improvement
7. Save final model (trained on all-but-last-fold) to `models/ensemble_gate.joblib`
8. Save training metadata to `models/ensemble_gate_meta.json` (date, n_samples, feature importance, fold metrics)

**LightGBM hyperparameters (v1 defaults):**
- `n_estimators`: 200
- `max_depth`: 5
- `learning_rate`: 0.05
- `min_child_samples`: 50
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `objective`: binary
- `metric`: binary_logloss

No hyperparameter tuning in v1 — these are conservative defaults that resist overfitting on small datasets.

### 3. Integration in paper trader — `src/ggTrader/paper/signal_runner.py`

After `generate_signals()` returns buys:

```python
from ggTrader.paper.feature_gate import FeatureGate

gate = FeatureGate()  # loads from default path, or disabled if no model
if gate.enabled:
    buys, scores = gate.filter_buys(buys, ohlcv)
    # Log: "ML gate: kept 3/7 — AAPL(0.72), MSFT(0.61), NVDA(0.58)"
```

**Graceful degradation:** If `models/ensemble_gate.joblib` doesn't exist, the gate is disabled and all signals pass through unchanged. This means the paper trader works identically to today until we train and deploy a model.

### 4. Telegram notifications

Gate decisions are included in the daily notification:
- "ML gate: passed 3/7 signals (AAPL 0.72, MSFT 0.61, NVDA 0.58)"
- "ML gate: blocked 4 signals (TSLA 0.42, META 0.38, ...)"
- "ML gate: disabled (no model file)"

### 5. Model artifact — `models/ensemble_gate.joblib`

Stored in the project `models/` directory (gitignored). Contains the trained LightGBM Booster serialized via joblib. Metadata sidecar at `models/ensemble_gate_meta.json`.

## Labeling strategy

**Target:** 5-day forward close > entry close (binary).

**Rationale:** The ensemble's typical hold period is ~20 trading days (monthly rebalance). A 5-day horizon captures the initial directional move without requiring the full hold period. This gives more training samples per entry and tests whether the market confirms the signal's direction quickly.

**Class balance:** Expected ~52-55% positive (slight long bias in equities). No resampling needed — LightGBM handles mild imbalance natively.

## Evaluation criteria

The gate is worth deploying if, on the held-out test fold:
1. **Precision > 0.60** — at least 60% of passed signals are profitable
2. **Sharpe improvement** — portfolio of gated entries has higher Sharpe than unfiltered
3. **Coverage > 30%** — gate doesn't filter so aggressively that it never trades

If these aren't met, we ship the infrastructure but leave the model file absent (gate disabled) and iterate on features or horizon.

## What this does NOT do

- No WFO integration (future work: per-fold retraining in the lab)
- No automated retraining (manual `train_gate` when desired)
- No sell-signal gating (only filters buys)
- No online learning or model drift detection (v2)
- No external data dependencies (OHLCV-only features)

## Dependencies

- `lightgbm` — add to requirements
- `joblib` — already available (scikit-learn dependency)
- Existing: `pandas`, `numpy`, ensemble signal infrastructure

## File layout

```
src/ggTrader/
├── lab/
│   └── train_gate.py          # offline training script
├── paper/
│   ├── feature_gate.py        # gate class + feature extraction
│   └── signal_runner.py       # (modified) integrate gate after signals
models/
├── ensemble_gate.joblib       # trained model (gitignored)
└── ensemble_gate_meta.json    # training metadata (gitignored)
```
