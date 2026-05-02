# ggTrader Roadmap

Last updated: 2026-05-02

Live trading has been running for ~4 weeks. All major infrastructure gaps (WFO structure, sizing, multi-asset support, and observability) are now shipped. The remaining work focuses on risk refinement, ML experiment tracking, and further research optimizations.

---

## Priority Summary

Top items ranked by impact × feasibility, reflecting current state:

| Rank | Item | Theme | Status | Effort | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | Position health scoring | Hardening | Ready | ~3h | Flag stale positions or those deviating from backtest stats |
| 2 | MLflow experiment tracking | Infra | Not started | ~half day | Low cost, makes research pipeline presentable |
| 3 | LLM "Post-Mortem" reports | Ops | Concept | ~4h | Use LLM to analyze `position_closes.csv` and explain performance |
| 4 | Automated Risk Scaling | Risk | Concept | ~4h | Reduce exposure automatically during equity curve drawdowns |
| 5 | Multi-Timeframe Confirmation | Perf | Not started | Multi-day | Reduce false entries in choppy regimes |
| 6 | Coarse Screening for WFO | Perf | Not started | ~4h | Cut compute by 30-50% for larger strategy grids |

---

## Completed / Shipped Recently ✅

### Stock Trading Foundation — `Shipped 2026-05-02` ✅
Implemented `BaseExecutionEngine` with multi-asset support. Added `StockExecutionEngine` for Alpaca (Paper/Live), `YFinanceDataLoader` with DB caching, and stock macro regime filters (SPY + VIX).

### Daily Loss Circuit Breaker — `Shipped 2026-05-02` ✅
Halt new entries if intraday portfolio drops beyond a configurable threshold (default 5%). Integrated with `BaseExecutionEngine` for both Crypto and Stocks.

### Scoring & Selection Tweaks — `Shipped 2026-05-02` ✅
Refined WFO parameter selection logic: favored Sortino/Calmar over Sharpe, increased fold consistency gate (4/10 folds), and biased selection further toward Out-of-Sample (OOS) performance.

### Grafana Dashboard & DB Mirroring — `Shipped 2026-05-02` ✅
Visual equity curves, trade history, and PnL per trade via a Grafana container. Includes real-time mirroring of live trade logs to TimescaleDB and a backfill tool (`ggt db sync-live`).

### WFO Fold Increase (N_SPLITS 8 → 10) — `Shipped 2026-05-02` ✅
Increased OOS data points for more reliable robustness gating. Maintaining ~253 days of training data per fold.

### Fresh WFO Research Run — `Completed 2026-05-01` ✅
Incorporated April bear market data. Promoted fresh parameters for 29 symbols to live trading. YTD CAGR in validation: **48.46%**.

---

## Extension & Extension Research

### Stock Trading (`--asset-class stocks`) — `Shipped` ✅
Extend ggTrader to trade US equities alongside crypto using yfinance for data and Alpaca for execution. Implementation complete: [stock_trading_plan.md](stock_trading_plan.md).

---

## Research & Strategy

### Scoring & Selection Tweaks — `Shipped` ✅
- **Fold consistency gate**: 0.38 (requires 4 profitable folds with 10 total).
- **OOS robustness blend alpha**: 0.70 (further favor out-of-sample signal).
- **Composite weights**: Bias toward Sortino (0.30) and Calmar (0.30).

### LLM "Post-Mortem" reports — `Concept`
Integrate an LLM into the `ggt pnl-daily` flow to analyze trade logs and provide natural-language commentary.

### Automated Risk Scaling — `Concept`
"Equity Curve Trading": Automatically reduce risk when the system-wide equity curve is below its own moving average.

### Multi-Timeframe Confirmation — `Not Started`
Enter on 4h signal only when the daily trend agrees. Requires architectural support for multiple bar frequencies in the precomputer.

---

## Infrastructure & Observability

### MLflow Experiment Tracking — `Not Started`
Log WFO research runs (params, metrics, fold results) to MLflow for visual comparison across runs.

### WFO Result Diffing (`ggt compare`) — `Not Started`
`ggt compare research_20260402 research_20260419` — side-by-side coin selection changes, strategy drift, CAGR delta.

---

## Live Trading Hardening

### Daily Loss Circuit Breaker — `Shipped` ✅
Halt new entries if intraday portfolio drops beyond a configurable threshold. Persistent state across restarts.

### BTC Regime Filter — `Shipped` ✅
Live trader now applies the tiered regime filter. Blocks new entries during bear markets.

### Strategy Exit Signal Execution — `Shipped` ✅
`fixed_sl_tp` coins now execute strategy exit signals correctly.

### Exchange Reconciliation on Startup — `Shipped` ✅
`_reconcile_positions()` queries Kraken/Alpaca on startup and handles stale/untracked positions.

### Dust Handling — `Shipped` ✅
Three-layer protection against dust positions polluting state and reports.

### Position Health Scoring — `Not Started`
For each open position: days held vs expected hold time from backtest. Flag stale positions open 3x longer than average.
