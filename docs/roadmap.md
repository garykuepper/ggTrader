# ggTrader Roadmap

Last updated: 2026-04-19

Live trading has been running for ~2 weeks. Most of the critical hardening gaps from the original `live_trading_gap_analysis.md` are now shipped. The remaining work is split between performance levers (research-side tuning, sizing) and operational improvements (alerts, dashboards, stock trading).

---

## Priority Summary

Top items ranked by impact × feasibility, reflecting current state:

| Rank | Item | Theme | Status | Effort | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | Fresh WFO research run | Perf | Ready | ~1-2h compute | Current params are 17 days old (2026-04-02), pre-bear regime. Highest-probability perf lever with zero code work |
| 2 | Trade fill alerts (Telegram) | Ops | Shipped 2026-04-23 | — | Notifier infra exists; only operational blind spot left after this week's fixes |
| 3 | Adaptive position sizing | Perf | Shipped 2026-04-23 (opt-in) | — | Real DD reduction — scale by inverse ATR so high-vol coins get smaller slices |
| 4 | Grafana dashboard | Infra | Not started | ~half day | TimescaleDB already has the data; biggest demo/portfolio gap |
| 5 | WFO fold increase (8→10) | Perf | Not started | 10 min + research | Bundle with #1; marginal but proven robustness gain |
| 6 | Stock trading (`--asset-class stocks`) | Extension | Specced | Multi-day | Full plan written, diversifies away from crypto-only correlation |
| 7 | Scoring tweaks (fold consistency, OOS alpha) | Perf | Not started | 10 min + research | Run one at a time with research run between each |
| 8 | Daily loss circuit breaker | Hardening | Not started | ~2h | Last remaining gap from original live trading analysis |
| 9 | Cashflow ledger (`ggt cashflow`) | Ops | Specced | ~4h | Lower priority now — Kraken ledger fetch already provides deposit-adjusted metrics. Useful for out-of-band tracking |
| 10 | MLflow experiment tracking | Infra | Not started | ~half day | Low cost, makes research pipeline presentable |

---

## Immediate / High-Value Extensions

### Stock Trading (`--asset-class stocks`) — `Specced`

Extend ggTrader to trade US equities alongside crypto using yfinance for data and Alpaca for execution. The architecture is already designed for it — 14 core modules are asset-agnostic. SPY regime filter replaces BTC regime, daily bars replace 4h. Full implementation spec: [stock_trading_plan.md](stock_trading_plan.md).

### Real-Time Trade Fill Alerts (Telegram) — `Shipped 2026-04-23`

Rich Telegram HTML messages fire on every entry and every exit. Single exit hook inside `_record_exit` covers all four sell paths (`strategy_signal`, `trailing_stop`, `oco_exit`, `emergency_rollback`). See [changelog.md](changelog.md) 2026-04-23 entry for implementation detail.

### Cashflow Ledger (`ggt cashflow`) — `Specced`

Track manual deposits/withdrawals in a small CSV ledger and net them out of return/risk metrics. **Lower priority than originally scoped** — the Kraken ledger fetch added on 2026-04-07 already provides deposit-adjusted metrics for anything routed through the exchange. Still useful for out-of-band capital flows (transfers from hardware wallets, etc.) and as a manual override when Kraken's ledger misses something. Detailed spec: [archive/future_tweaks_plan.md](archive/future_tweaks_plan.md#cashflow-ledger-deposits--withdrawals).

---

## Research & Strategy

### Fresh WFO Research Run — `Ready`

Current live params are from `research_20260402_123248` — 17 days old and pre-bear-regime. A fresh run incorporates the most recent market data (including the current bear phase) and may shift coin selection toward names that held up better. Zero code changes. Command: `docker compose run --rm ggtrader_live python -u ggt.py research`. Bundle with WFO fold increase for maximum efficiency.

### Adaptive Position Sizing — `Shipped 2026-04-23 (opt-in)`

Volatility-normalized sizing shipped as an opt-in flag (`--adaptive-sizing`). See [changelog.md](changelog.md) 2026-04-23 entry. Off by default to keep attribution clean during the 2026-04-02 research-run observation window; flip on when ready to A/B against weight-based sizing.

### Multi-Timeframe Confirmation — `Not Started`

Enter on 4h signal only when the daily trend agrees. Would reduce false entries in choppy/ranging regimes. Architecturally non-trivial — requires the precomputer to hold multiple bar frequencies simultaneously.

### Volume-Profile / VWAP Entries — `Not Started`

Rolling VWAP-anchored signals for crypto (24/7 markets make session VWAP tricky). Could pair with existing Keltner/Donchian breakout strategies as a confirmation filter. Blocked on volume data quality assessment.

### WFO Fold Increase (N_SPLITS 8 → 10) — `Not Started`

Two more OOS data points for the robustness gate without starving the training window (~253 days vs ~299 days currently). The 6→8 fold increase was a key driver of the March improvement. ~25% compute increase. Bundle with the next fresh research run.

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

`ggt compare research_20260402 research_20260419` — side-by-side coin selection changes, strategy drift, CAGR delta. Replaces manually reading two markdown reports.

---

## Live Trading Hardening

### BTC Regime Filter — `Shipped 2026-04-xx`

Live trader now applies the tiered regime filter. Visible in logs as `[Regime] EMA(100) — BTC bull=False, alt bull=False, blocked=[...]`. Blocks new entries during bear markets while leaving open positions and TSLs untouched. Low-correlation coins (BTC-corr < 0.3) pass through unfiltered.

### Strategy Exit Signal Execution — `Shipped 2026-04-xx`

`fixed_sl_tp` coins now execute strategy exit signals correctly — cancel open OCO, place market sell, record exit. `atr_trailing` / `trailing_stop` coins continue to rely on the exchange TSL (correct behavior). Verified 2026-04-18 with TRX-USD exit via `strategy_signal`.

### Exchange Reconciliation on Startup — `Shipped 2026-04-09`

`_reconcile_positions()` queries Kraken on startup, reconciles against `active_positions.json`, and handles stale/untracked/dust positions. Positions that can't be recorded get `pending_repair=True` for the next auto-sync cycle.

### Risk Control Gates at Load Time — `Shipped`

`MAX_COINS_PER_STRATEGY`, `MIN_ROBUSTNESS_SCORE`, and `SYMBOL_BLACKLIST` are enforced in the live loader. Visible in logs as `[Gates] Dropped N coin(s)...`.

### Dust Handling — `Shipped 2026-04-10 → 2026-04-19`

Three-layer protection against dust positions polluting state and reports:

1. Reconciliation skips sub-$1 untracked exchange balances (2026-04-10)
2. Report filters positions with cost basis below $1 (2026-04-10)
3. Proactive cleanup of local-state dust on every reconcile (2026-04-19)

### Correct ATR Trailing Stop Calculation — `Shipped 2026-04-17`

Stop is now computed as `fill_price - atr_multiplier * current_atr` using the live ATR value, instead of using the backtest's historical peak-based `stop_price`. Positions entered before this fix may still have oversized `stop_pct` values (e.g., AKT-USD at 20.39%); those will clear on their next exit.

### Daily Loss Circuit Breaker — `Not Started`

Halt new entries if intraday portfolio drops beyond a configurable threshold. Last remaining gap from the original live trading analysis.

### Position Health Scoring — `Not Started`

For each open position: days held vs expected hold time from backtest, unrealized PnL vs ATR-based expectation. Flag stale positions open 3x longer than the strategy's average hold time.

### Backtesting Against Real Fills — `Not Started`

Compare what ggTrader's signals would have predicted vs what actually executed. Slippage measurement, fill quality analysis. Would expose real vs assumed execution cost.
