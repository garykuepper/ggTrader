# ggTrader Roadmap

Last updated: 2026-05-02

Live trading has been running for ~4 weeks. Most of the critical hardening gaps and performance levers (WFO structure, sizing, observability) are now shipped. The remaining work focuses on risk hardening (circuit breakers), further research refinements, and expanding into new asset classes.

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

### Grafana Dashboard & DB Mirroring — `Shipped 2026-05-02` ✅
Visual equity curves, trade history, and PnL per trade via a Grafana container. Includes real-time mirroring of live trade logs to TimescaleDB and a backfill tool (`ggt db sync-live`).

### WFO Fold Increase (N_SPLITS 8 → 10) — `Shipped 2026-05-02` ✅
Increased OOS data points for more reliable robustness gating. Maintaining ~253 days of training data per fold.

### Fresh WFO Research Run — `Completed 2026-05-01` ✅
Incorporated April bear market data. Promoted fresh parameters for 29 symbols to live trading. YTD CAGR in validation: **48.46%**.

### Real-Time Trade Fill Alerts (Telegram) — `Shipped 2026-04-23` ✅
Rich Telegram HTML messages for every entry and exit. Includes strategy, PnL, and exit reason.

### Adaptive Position Sizing — `Shipped 2026-04-23` ✅
Volatility-normalized sizing (`--adaptive-sizing`) now available to bound risk per trade.

---

## Immediate / High-Value Extensions

### Stock Trading (`--asset-class stocks`) — `Specced`
Extend ggTrader to trade US equities alongside crypto using yfinance for data and Alpaca for execution. Architecture is asset-agnostic. Spec: [stock_trading_plan.md](stock_trading_plan.md).

---

## Research & Strategy

### Scoring & Selection Tweaks — `Ready`
- **Fold consistency gate**: 0.33 → 0.38 (requires 4 profitable folds with 10 total).
- **OOS robustness blend alpha**: 0.65 → 0.70 (further favor out-of-sample signal).
- **Composite weights**: Bias toward Sortino (0.30) and Calmar (0.30).

### LLM "Post-Mortem" reports — `Concept`
Integrate an LLM into the `ggt pnl-daily` flow to analyze trade logs and provide natural-language commentary: *"Strategy X is struggling with fakeouts in this ranging market; consider tightening the RSI threshold."*

### Automated Risk Scaling — `Concept`
"Equity Curve Trading": Automatically reduce `TARGET_RISK_PCT` or `CAPITAL_PER_TRADE` when the system-wide equity curve is below its own moving average, and scale back up when performance recovers.

### Multi-Timeframe Confirmation — `Not Started`
Enter on 4h signal only when the daily trend agrees. Requires architectural support for multiple bar frequencies in the precomputer.

---

## Infrastructure & Observability

### MLflow Experiment Tracking — `Not Started`
Log WFO research runs (params, metrics, fold results) to MLflow for visual comparison across runs. The data is already structured for this — currently requires reading markdown reports manually. Low integration cost.

### WFO Result Diffing (`ggt compare`) — `Not Started`
`ggt compare research_20260402 research_20260419` — side-by-side coin selection changes, strategy drift, CAGR delta. Replaces manually reading two markdown reports.

---

## Live Trading Hardening

### BTC Regime Filter — `Shipped` ✅
Live trader now applies the tiered regime filter. Visible in logs as `[Regime] EMA(100) — BTC bull=False, alt bull=False, blocked=[...]`. Blocks new entries during bear markets while leaving open positions and TSLs untouched. Low-correlation coins (BTC-corr < 0.3) pass through unfiltered.

### Strategy Exit Signal Execution — `Shipped` ✅
`fixed_sl_tp` coins now execute strategy exit signals correctly — cancel open OCO, place market sell, record exit. `atr_trailing` / `trailing_stop` coins continue to rely on the exchange TSL (correct behavior).

### Exchange Reconciliation on Startup — `Shipped` ✅
`_reconcile_positions()` queries Kraken on startup, reconciles against `active_positions.json`, and handles stale/untracked/dust positions.

### Risk Control Gates at Load Time — `Shipped` ✅
`MAX_COINS_PER_STRATEGY`, `MIN_ROBUSTNESS_SCORE`, and `SYMBOL_BLACKLIST` are enforced in the live loader. Visible in logs as `[Gates] Dropped N coin(s)...`.

### Dust Handling — `Shipped` ✅
Three-layer protection against dust positions polluting state and reports (Skip sub-$1, filter reports, proactive cleanup).

### Correct ATR Trailing Stop Calculation — `Shipped` ✅
Stop is now computed as `fill_price - atr_multiplier * current_atr` using the live ATR value.

### Daily Loss Circuit Breaker — `Ready`
Halt new entries if intraday portfolio drops beyond a configurable threshold. Last remaining gap from the original live trading analysis.

### Position Health Scoring — `Not Started`
For each open position: days held vs expected hold time from backtest, unrealized PnL vs ATR-based expectation. Flag stale positions open 3x longer than the strategy's average hold time.

### Backtesting Against Real Fills — `Not Started`
Compare what ggTrader's signals would have predicted vs what actually executed. Slippage measurement, fill quality analysis. Would expose real vs assumed execution cost.
