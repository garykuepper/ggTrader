# Roadmap

**Last updated:** 2026-05-08 · **Status:** Live ~5 weeks · Crypto-only (stocks pipeline removed 2026-05-08) · BTC regime filter currently disabled

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

### 2026-05 — Crypto-only refocus
- **Stocks pipeline removed** (2026-05-08): full purge of `StockExecutionEngine`, `stock_regime_filtering`, `cached_yfinance_loader`, SPY/VIX gates, Alpaca sync, S&P 500 benchmark, and the `--asset-class` flag. Project is crypto-only ahead of a Kraken-CLI transition.
- **Daily-loss circuit breaker**: 5% intraday cap, persistent across restarts
- **WFO refinement**: 10 folds, OOS alpha 0.70, Sortino/Calmar bias

### 2026-04 — Observability + Adaptive Sizing
- **Grafana dashboard**: equity, PnL dots, trades
- **TimescaleDB live mirror**; `ggt db sync-live` for CSV backfill
- **Monthly auto-recalibration**: WFO runs internally on day 1 ~01:00 UTC, hot-reloads params
- **Market-regime line** in 08:00 PnL report (BTC + altcoin status, live ccxt)
- **Adaptive volatility-normalized sizing** (opt-in `--adaptive-sizing`)
- **Trailing-stop floor**: `MIN_TRAILING_STOP_PCT` / `MIN_ATR_TRAILING_PCT` (default 4%)
- **CLI**: `ggt trade-report`, `ggt repair`, `ggt pnl-daily`, `--dry-run-sizing`

### 2026-05 (cont.) — Sizing & Filter Rework
- **Weighted sizing default**: live trader allocates per coin from OOS robustness, capped at `MAX_COIN_ALLOCATION`. `--adaptive-sizing` still available as opt-in.
- **Single-leader regime filter**: collapsed back to BTC-only after research showed BTC+ETH dual-leader and 3-tier (BTC/altcoin/exempt) variants underperformed. Filter is currently `BTC_REGIME_FILTER=False` while we collect more comparison data.
- **Universal trailing stops**: `fixed_sl_tp` legacy positions now convert to native trailing-stop at placement time, so every live sell ratchets up.
- **Fear & Greed Index** in PnL reports + signals header (alternative.me, daily snapshot persisted to `fear_greed_index` table).
- **`ggt signals` command**: per-symbol diagnostic table — entry/exit, tier, correlations, %vs_EMA, in-position, blocked-reason.
- **Correlation matrix script**: `scripts/coin_correlation_matrix.py` for visualising which coins move with BTC.
- **WFO selection-funnel reporting**: research reports now show how many coins each gate dropped.

### 2026-03 — Live Hardening
- **Telegram + Discord** trade-fill alerts
- **Exchange reconciliation**: server-side TSL/OCO detection on heartbeat

### 2026-02 — Core Engine
- **Vectorized WFO** pipeline with Numba
- **FastBacktest** engine (vbt-based)
- **Unified `ggt` CLI**

---
*Back to [README.md](../README.md)*
