# ggTrader Roadmap

Last updated: 2026-04-10

---

## Immediate / High-Value Extensions

### Stock Trading (`--asset-class stocks`) — `Specced`

Extend ggTrader to trade US equities alongside crypto using yfinance for data and Alpaca for execution. The architecture is already designed for it — 14 core modules are asset-agnostic. SPY regime filter replaces BTC regime, daily bars replace 4h. Full implementation spec: [stock_trading_plan.md](stock_trading_plan.md).

### Real-Time Trade Fill Alerts (Telegram) — `Specced`

Push rich Telegram/Discord messages on every entry and exit fill — strategy name, entry price, PnL, hold time, exit reason. Closes the biggest operational blind spot: right now a fired stop isn't visible until the next `pnl-daily` run (up to 24h lag). The notifier infrastructure already exists (`src/ggTrader/utils/notifier.py`). Detailed spec with hook points and message format: [archive/future_tweaks_plan.md](archive/future_tweaks_plan.md#notifications--alerting).

### Cashflow Ledger (`ggt cashflow`) — `Specced`

Track manual deposits/withdrawals in a small CSV ledger and net them out of all return/risk metrics. Currently the $100 capital injection on 2026-04-03 contaminates the equity curve, Sharpe, Sortino, and max drawdown calculations. Implementation adds a `ggt cashflow add/list/remove` CLI and adjusts the equity curve builder. Detailed spec: [archive/future_tweaks_plan.md](archive/future_tweaks_plan.md#cashflow-ledger-deposits--withdrawals).

---

## Research & Strategy

### Multi-Timeframe Confirmation — `Not Started`

Enter on 4h signal only when the daily trend agrees. Would reduce false entries in choppy/ranging regimes. Architecturally non-trivial — requires the precomputer to hold multiple bar frequencies simultaneously.

### Adaptive Position Sizing — `Not Started`

Scale position size by inverse ATR rather than fixed `PORTFOLIO_SHARE=0.10`. Higher-vol coins get smaller allocations, lower-vol get larger. Pure execution logic change — no WFO or strategy modifications needed.

### Volume-Profile / VWAP Entries — `Not Started`

Rolling VWAP-anchored signals for crypto (24/7 markets make session VWAP tricky). Could pair with existing Keltner/Donchian breakout strategies as a confirmation filter. Blocked on volume data quality assessment.

### WFO Fold Increase (N_SPLITS 8 → 10) — `Not Started`

Two more OOS data points for the robustness gate without starving the training window (~253 days vs ~299 days currently). The 6→8 fold increase was a key driver of the March improvement. ~25% compute increase. Test after the observation period.

### Coarse Screening for WFO Runtime — `Not Started`

Pre-filter step: run each strategy with 1 default param combo per coin before expanding the full grid. Skip strategy/coin pairs that produce zero trades or -inf Sharpe. Could cut compute by 30-50% as more strategies are added.

### Scoring & Selection Tweaks — `Not Started`

Three independent experiments to run **one at a time** with a research run between each:
- Fold consistency gate: 0.33 → 0.38 (with 10 folds, requires 4 profitable folds)
- OOS robustness blend alpha: 0.65 → 0.70 (further favor out-of-sample signal)
- Composite weights: bias toward Sortino (0.30) and Calmar (0.30), reduce Sharpe and ProfitFactor to 0.20 each

---

## Infrastructure & Observability

### Grafana Dashboard — `Not Started`

TimescaleDB already has all the trade and balance data. A Grafana container in docker-compose pulling from Postgres would provide equity curves, trade history visualizations, and strategy performance breakdowns. Biggest gap from a portfolio/demo standpoint.

### MLflow Experiment Tracking — `Not Started`

Log WFO research runs (params, metrics, fold results) to MLflow for visual comparison across runs. The data is already structured for this — currently requires reading markdown reports manually. Low integration cost.

### WFO Result Diffing (`ggt compare`) — `Not Started`

`ggt compare research_20260402 research_20260409` — side-by-side coin selection changes, strategy drift, CAGR delta. Replaces manually reading two markdown reports.

---

## Live Trading Hardening

### BTC Regime Filter — `Specced` **CRITICAL**

The live trader enters during bear markets that the research pipeline filters out. Port `_compute_btc_regime_mask()` and `_apply_tiered_regime_mask()` from `orchestrator.py` into the live execution loop. Always fetch BTC OHLCV even if BTC is not in the coin universe. Full gap spec: [live_trading_gap_analysis.md](live_trading_gap_analysis.md#11-btc-regime-filter--critical).

### Strategy Exit Signal Execution — `Specced` **HIGH**

Live code computes exit signals but ignores them — relies solely on exchange TSL. For `fixed_sl_tp` coins, the strategy exit signal is the primary close mechanism but is currently dead code. Full gap spec: [live_trading_gap_analysis.md](live_trading_gap_analysis.md#12-strategy-exit-signal-execution--high).

### Exchange Reconciliation on Startup — `Specced` **HIGH**

No reconciliation between `active_positions.json` and actual Kraken state on restart. Risk of doubled positions or ghost positions. Full gap spec: [live_trading_gap_analysis.md](live_trading_gap_analysis.md#21-no-exchange-reconciliation-on-startup--high).

### Risk Control Gates at Load Time — `Specced` **MEDIUM**

`MAX_COINS_PER_STRATEGY`, `MIN_ROBUSTNESS_SCORE`, and `MAX_COIN_ALLOCATION` are enforced in research but not in the live loader. Full gap spec: [live_trading_gap_analysis.md](live_trading_gap_analysis.md#3-risk-control-gaps).

### Daily Loss Circuit Breaker — `Not Started`

Halt new entries if intraday portfolio drops beyond a configurable threshold. One of the remaining gaps from the [live trading gap analysis](live_trading_gap_analysis.md#6-recommended-implementation-order).

### Position Health Scoring — `Not Started`

For each open position: days held vs expected hold time from backtest, unrealized PnL vs ATR-based expectation. Flag stale positions open 3x longer than the strategy's average hold time.

### Backtesting Against Real Fills — `Not Started`

Compare what ggTrader's signals would have predicted vs what actually executed. Slippage measurement, fill quality analysis. Would expose real vs assumed execution cost.

---

## Priority Summary

Top items ranked by impact x feasibility:

| Rank | Item | Theme | Status | Why |
|------|------|-------|--------|-----|
| 1 | Stock trading (`--asset-class stocks`) | Extension | Specced | Full plan written, architecture ready, diversifies away from crypto-only |
| 2 | Trade fill alerts (Telegram) | Extension | Specced | Notifier infra exists, closes biggest operational blind spot |
| 3 | BTC regime filter (live) | Hardening | Specced | CRITICAL gap — live is systematically more aggressive than research |
| 4 | Grafana dashboard | Infra | Not Started | Biggest portfolio/demo gap, data already in TimescaleDB |
| 5 | Cashflow ledger | Extension | Specced | Small scope, permanently fixes contaminated metrics |
| 6 | Strategy exit execution (live) | Hardening | Specced | HIGH gap — `fixed_sl_tp` exits are dead code in live |
| 7 | Exchange reconciliation | Hardening | Specced | HIGH gap — risk of doubled/ghost positions on restart |
| 8 | Adaptive position sizing | Strategy | Not Started | Pure execution change, reduces portfolio drawdown |
| 9 | MLflow tracking | Infra | Not Started | Low cost, makes research pipeline presentable |
| 10 | WFO fold increase (8→10) | Strategy | Not Started | Proven approach, wait for observation period |
