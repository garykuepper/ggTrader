# backtest

Backtest engine wrappers and statistically honest validation infrastructure.
Populated in **Phase 6**:

- `vectorized.py` — VectorBT wrapper (replaces ad-hoc usage in `core/fast_backtest.py`)
- `event_driven.py` — optional `nautilus_trader` integration for fill realism
- `cv.py` — `PurgedKFold`, `CombinatorialPurgedCV` (via `skfolio`)
- `metrics.py` — Deflated Sharpe Ratio, Probability of Backtest Overfitting
  (Bailey et al.), Sharpe, Sortino, max DD, Calmar
