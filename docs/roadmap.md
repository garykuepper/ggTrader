# 🚀 ggTrader Roadmap

**Last updated:** 2026-05-02  
**Status:** Live trading active for ~4 weeks. Major multi-asset and risk infra shipped.

---

## 🔝 High-Impact Priorities

| Rank | Item | Theme | Status | Effort | Strategic Why |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1️⃣ | **Position health scoring** | Hardening | Ready | ~3h | Identify "zombie" trades open 3x longer than backtest avg |
| 2️⃣ | **MLflow experiment tracking** | Infra | Not started | ~4h | Visual dashboard for WFO results and parameter drift |
| 3️⃣ | **LLM "Post-Mortem" reports** | Ops | Concept | ~4h | Auto-generate natural language performance commentary |
| 4️⃣ | **Automated Risk Scaling** | Risk | Concept | ~4h | Scale exposure based on equity curve slope (equity-curve trading) |
| 5️⃣ | **Multi-Timeframe Confirmation** | Perf | Not started | Multi-day | Reduce false entries by aligning 4h signals with Daily trend |

---

## 🛠 Planned Enhancements

### Research & Strategy
| Feature | Description | Status |
| :--- | :--- | :--- |
| **Multi-TF Gates** | Only enter on a 4h signal if the Daily timeframe trend is bullish. | `Not Started` |
| **Coarse Screening** | Use a fast, loose pass to prune 70% of strategy combos before full WFO. | `Not Started` |
| **Alpha Blending** | Dynamically adjust `OOS_ROBUSTNESS_BLEND_ALPHA` based on market volatility. | `Concept` |

### Infrastructure & Risk
| Feature | Description | Status |
| :--- | :--- | :--- |
| **MLflow Sync** | Automated logging of every `ggt research` run to a central MLflow UI. | `Ready` |
| **Equity Scaling** | Reduce `TARGET_RISK_PCT` by 50% if Equity < 20-day Moving Average. | `Concept` |
| **WFO Compare** | CLI tool (`ggt compare`) to diff two research folders for selection drift. | `Not Started` |
| **Backtest vs Real** | Measure "Slippage Gap" by replaying backtest logic against actual fill logs. | `Not Started` |

---

## ✅ Completed Milestones

### 📈 Phase 4: Multi-Asset & Risk (May 2026)
*   **Stock Trading Foundation**
    *   `BaseExecutionEngine` for shared logic between Crypto/Stocks.
    *   `StockExecutionEngine` for Alpaca Paper/Live execution.
    *   `YFinanceDataLoader` with 1980+ data and TimescaleDB caching.
    *   Automated S&P 500 volume-ranked universe builder.
*   **Intraday Protection**
    *   **Daily Loss Circuit Breaker**: Auto-halts entries on 5% drawdown.
    *   Persistent circuit breaker state across bot restarts.
*   **WFO Robustness Refinement**
    *   Increased splits to **10 folds** for more granular OOS data.
    *   Biased scoring toward **Sortino & Calmar** ratios (risk-adjusted).
    *   Increased OOS alpha blend to **0.70**.

### 📊 Phase 3: Observability (Apr 2026)
*   **Grafana Dashboard Integration**
    *   Real-time Equity Curve, PnL Dots, and Trades Table.
    *   "Asset Run" dropdown to toggle between Crypto (`LIVE`) and Stocks (`LIVE_STOCKS`).
*   **Real-time DB Mirroring**
    *   Direct syncing of all entry/exit events to TimescaleDB.
    *   `ggt db sync-live` utility for historical CSV backfills.

### 🛡 Phase 2: Live Hardening (Mar 2026)
*   **Tiered Regime Filtering**: EMA-based macro gate (BTC EMA 50/200).
*   **Telegram/Discord Alerts**: Rich HTML notifications for every trade fill.
*   **Adaptive Position Sizing**: Volatility-normalized risk per trade.
*   **Exchange Reconciliation**: Auto-detect server-side OCO/TSL triggers on heartbeat.

### 🏗 Phase 1: Core Engine (Feb 2026)
*   **Vectorized WFO Pipeline**: High-speed backtesting with Numba.
*   **FastBacktest Engine**: 1000x faster than iterative simulations.
*   **Unified CLI**: The `ggt` command-line interface.

---
*Back to [README.md](../README.md)*
