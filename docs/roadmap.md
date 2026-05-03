# 🚀 ggTrader Roadmap

**Last updated:** 2026-05-03 · **Status:** Live ~4 weeks · Multi-asset shipped

**Status legend:** ✅ Shipped · 🟡 In progress · 🔵 Ready · ⚪ Concept

---

## 🔝 Top Priorities (Next)

| # | Item | Theme | Status | Effort | Why |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | Position health scoring | Hardening | 🔵 | ~3h | Flag "zombie" trades open 3× longer than backtest avg |
| 2 | MLflow experiment tracking | Infra | ⚪ | ~4h | Visual dashboard for WFO results and parameter drift |
| 3 | LLM post-mortem reports | Ops | ⚪ | ~4h | Auto-generate natural-language performance commentary |
| 4 | Automated risk scaling | Risk | ⚪ | ~4h | Scale exposure off equity-curve slope |
| 5 | Multi-timeframe confirmation | Perf | ⚪ | Multi-day | Align 4h signals with daily trend |

---

## 🛠 Backlog

### Research & Strategy
| Feature | Description | Status |
| :-- | :-- | :-- |
| Multi-TF gates | Only enter on 4h signal if daily trend bullish | ⚪ |
| Coarse screening | Fast/loose pass to prune ~70% of strategy combos before WFO | ⚪ |
| Alpha blending | Adjust `OOS_ROBUSTNESS_BLEND_ALPHA` based on volatility regime | ⚪ |

### Infra & Risk
| Feature | Description | Status |
| :-- | :-- | :-- |
| Equity scaling | Cut `TARGET_RISK_PCT` 50% if equity < 20-day MA | ⚪ |
| WFO compare | `ggt compare` to diff two research folders for selection drift | ⚪ |
| Backtest-vs-real | "Slippage gap" by replaying backtest logic against real fills | ⚪ |

---

## ✅ Shipped (most recent first)

### 2026-05 — Multi-Asset & Risk
- **Stocks**: `BaseExecutionEngine`, `StockExecutionEngine` (Alpaca), `YFinanceDataLoader` (1980+ bars, TimescaleDB cache)
- **Stock universe**: `scripts/update_universe_stocks.py` — manual S&P 500 volume rank (auto-rank pending)
- **Daily-loss circuit breaker**: 5% intraday cap, persistent across restarts
- **Stock macro filters**: SPY EMA gate + VIX volatility gate
- **WFO refinement**: 10 folds, OOS alpha 0.70, Sortino/Calmar bias

### 2026-04 — Observability + Adaptive Sizing
- **Grafana dashboard**: equity, PnL dots, trades; LIVE / LIVE_STOCKS toggle
- **TimescaleDB live mirror**; `ggt db sync-live` for CSV backfill
- **Monthly auto-recalibration**: WFO runs internally on day 1 ~01:00 UTC, hot-reloads params
- **Market-regime line** in 08:00 PnL report (BTC + altcoin status, live ccxt)
- **Adaptive volatility-normalized sizing** (opt-in `--adaptive-sizing`)
- **Trailing-stop floor**: `MIN_TRAILING_STOP_PCT` / `MIN_ATR_TRAILING_PCT` (default 4%)
- **CLI**: `ggt trade-report`, `ggt repair`, `ggt pnl-daily`, `--dry-run-sizing`

### 2026-03 — Live Hardening
- **Tiered regime filter**: BTC EMA 20/200 (high-corr) → altcoin index (mid) → exempt (low)
- **Telegram + Discord** trade-fill alerts
- **Exchange reconciliation**: server-side TSL/OCO detection on heartbeat

### 2026-02 — Core Engine
- **Vectorized WFO** pipeline with Numba
- **FastBacktest** engine (vbt-based)
- **Unified `ggt` CLI**

---
*Back to [README.md](../README.md)*
