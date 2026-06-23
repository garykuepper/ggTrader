# Roadmap

> **Organization.** §1 is the thesis. §2 is time-bounded work (~4 weeks). §3 is open research. §4 is infrastructure. For dated history see [`changelog.md`](changelog.md). For the pre-lab roadmap (crypto venue migration, WFO textbook reset, etc.) see [`archive/roadmap-pre-lab.md`](archive/roadmap-pre-lab.md).

---

## At a glance

| State | Detail |
|---|---|
| ✅ **Lab shipped** | vectorbt-centric research bench (2026-06-15). 12 strategies, parameter sweep, WFO framework, monthly folds, DB persistence. |
| ✅ **WFO validated** | Ensemble reversion (bb+rsi+ema 3-voter) clears WFO: OOS Sharpe 0.84 vs SPY 0.59, CAGR 18.8%, WFE 1.25. First strategy to beat SPY risk-adjusted in honest OOS. |
| ✅ **6-voter ensemble** | Expanded to 6 sub-signals (+ MACD divergence, volume-confirmed BB, multi-timeframe reversion). Conviction-weighted sizing variant available. |
| ✅ **ML feature gate** | LightGBM classifier (precision 0.585, 3.2K samples). Filters low-confidence buys in production. ML pre-screen script for evaluating new signals. |
| 🟢 **Paper live** | Ensemble on Alpaca paper ($102K account). Cron fires 1:30 PM PT Mon–Fri. ML gate + risk guardrails active. |
| 🔵 **Next up** | Paper monitoring → live go-live. Target: 5–10 clean trading days, then fund Alpaca live ($1K), swap keys. |
| 🧪 **Research** | Conviction ensemble WFO validation, expanded signal library, regime-aware signal selection. |

**Status legend:** ✅ Done · 🟢 Live · 🔵 Next · 🧪 Research · ⏸ Deferred

---

## 1. North star

Build a **flexible multi-strategy research lab** with honest walk-forward evaluation, and deploy capital only against edges that survive true out-of-sample testing. The lab tests strategies on US equities (SP500/Nasdaq-100/Russell 2000) via vectorbt, with monthly rolling folds and point-in-time universe membership.

**Current thesis (2026-06-20).** Signal diversification is the primary driver of edge. The 3-voter ensemble (BB reversion + RSI reversion + EMA cross) beat SPY in 17-fold WFO because reversion and trend signals are negatively correlated — when trend gets whipsawed, reversion thrives. Expanding the signal library (6 voters, conviction weighting) and validating in WFO is the active research direction.

---

## 2. In flight (~4 weeks)

### 2a. Paper trading → live go-live

| Status | Step |
|:---:|---|
| ✅ | Alpaca paper adapter, signal runner, trader, Telegram alerts, cron |
| ✅ | ML feature gate (LightGBM, precision 0.585) |
| ✅ | Risk guardrails (30 max positions, 3.3%/trade, 5% concentration, 3% daily loss halt, 15% drawdown halt) |
| ✅ | DAY time-in-force fix for fractional/notional orders |
| 🟢 | Monitor paper fills for 5–10 clean trading days |
| 🔵 | Fund Alpaca live account ($1K) |
| 🔵 | Swap to live API keys |
| 🔵 | Confirm position sizing / order fills match expectations |

### 2b. Expanded signal validation

| Status | Item | Notes |
|:---:|---|---|
| ✅ | MACD divergence signal | `MACDDivergenceSignal` — bearish/bullish divergence detection |
| ✅ | Volume-confirmed BB reversion | `VolumeBBReversionSignal` — BB touch + volume spike confirmation |
| ✅ | Multi-timeframe reversion | `MultiTimeframeReversionSignal` — weekly RSI oversold + daily BB touch |
| ✅ | 6-voter ensemble | `EnsembleSignal` expanded from 3 to 6 sub-signals, `min_agree` configurable |
| ✅ | Conviction-weighted sizing | `EnsembleConvictionSignal` — size by average strength of agreeing signals |
| ✅ | ML pre-screen script | `scripts/ml_signal_screen.py` — LightGBM precision gate for any signal |
| 🧪 | WFO validation of 6-voter ensemble | Running — compare against 3-voter baseline |
| 🧪 | WFO validation of conviction sizing | Compare risk-adjusted metrics vs fixed sizing |

### 2c. Signal library expansion (research)

| Status | Item | Notes |
|:---:|---|---|
| ⚪ | Keltner Channel reversion | ATR-based bands, orthogonal to BB |
| ⚪ | Stochastic momentum | %K/%D crossover in oversold zone |
| ⚪ | Volume profile signals | VWAP deviation + volume anomaly detection |
| ⚪ | Breadth-based signals | Advance/decline ratio, new highs/lows |

---

## 3. Research directions

*Open exploration — not time-bounded. Ordered roughly by expected payoff.*

| | Direction | Status | Notes |
|:---:|---|:---:|---|
| A | **Conviction ensemble tuning** | 🧪 | Optimize strength-weighted sizing thresholds in WFO |
| B | **Regime-aware signal selection** | ⚪ | Select which sub-signals vote based on vol regime |
| C | **Bayesian parameter selection** | ⚪ | Posterior over IS performance → highest lower-credible-bound |
| D | **Statistical arbitrage / pairs** | ⚪ | Cointegrated pair z-score reversion (orthogonal to momentum) |
| E | **Meta-allocation across strategies** | ⚪ | Equal-risk or rolling meta-optimizer across signal families |
| F | **Deep RL for position sizing** | ⚪ | PPO/SAC agent for sizing, entries stay rule-based |

---

## 4. Infrastructure

| Status | Item | Notes |
|:---:|---|---|
| ✅ | TimescaleDB persistence (`lab_runs`, `lab_periods`) | Immutable run history |
| ✅ | Equity OHLCV backfill (SP500, 4M rows) | DB-only — no live downloads during backtests |
| ✅ | Tiingo data loader | Fallback for delisted tickers yfinance misses |
| ✅ | Multi-universe support | `--universe sp500\|nasdaq100\|russell2000` |
| ✅ | Parameter sweep framework | `--sweep` + `--sweep-param` for grid search |
| ✅ | WFO framework | `--wfo` for rolling train/test with OOS scoring |
| ✅ | WFE monitoring + circuit breaker | Walk-forward efficiency tracking, auto-halt on degradation |
| ✅ | NDH + DSR robustness gates | Neighbor distance heuristic, deflated Sharpe ratio |
| ✅ | 2.07× WFO speed | Collapsed per-fold metric accessors to single `returns()` extraction |
| ⚪ | WFO DB persistence | Persist per-fold WFO results to TimescaleDB (deferred to go-live) |
| ⚪ | `ggt compare` — diff two research runs | Side-by-side metric comparison |

---

## 5. Evolution since lab creation

The lab was created 2026-06-15, replacing ~26K lines of legacy live-trading infrastructure with a focused vectorbt research bench. Key milestones:

| Date | Milestone |
|---|---|
| 2026-06-15 | **Lab core** (Plan 1) — momentum bench, `xs_momentum` + `dual_momentum`, bit-identical validation |
| 2026-06-16 | **Signal strategies** (Plan 2) — `EmaCrossSignal`, `WfoTournamentSignal`, `simulate_signals()` |
| 2026-06-16 | **Equity backfill** (Plan 3) — 4M rows into TimescaleDB, old research code deleted |
| 2026-06-17 | Parameter sweep framework (`--sweep`, `--sweep-param`) |
| 2026-06-18 | Reversion strategies — `bb_reversion`, `rsi_reversion`. First to beat SPY in WFO (Sharpe 0.80) |
| 2026-06-18 | WFO framework (`--wfo`) — rolling train/test, OOS scoring, WFE, circuit breaker |
| 2026-06-18 | Robustness gates — NDH plateau filter, DSR deflated Sharpe |
| 2026-06-19 | Vol targeting overlay, trailing stop (confirmed destructive on reversion) |
| 2026-06-20 | **Ensemble validated in WFO** — Sharpe 0.84 vs SPY 0.59, CAGR 18.8%, WFE 1.25 |
| 2026-06-20 | Multi-universe support (SP500, Nasdaq-100, Russell 2000) |
| 2026-06-20 | Paper trading deployed on Alpaca ($102K) — signal runner, trader, Telegram, cron |
| 2026-06-22 | ML feature gate (LightGBM, precision 0.585) + risk guardrails |
| 2026-06-22 | Conviction-weighted ensemble sizing (`EnsembleConvictionSignal`) |
| 2026-06-22 | Tiingo data loader — fallback for delisted tickers |
| 2026-06-23 | Expanded to 6-voter ensemble (MACD divergence, volume-confirmed BB, multi-timeframe) |
| 2026-06-23 | ML pre-screen script for evaluating signal quality |

---

## Reference

- **Architecture**: [`architecture.md`](architecture.md)
- **CLI usage**: [`cli_reference.md`](cli_reference.md)
- **Changelog**: [`changelog.md`](changelog.md)
- **Pre-lab roadmap**: [`archive/roadmap-pre-lab.md`](archive/roadmap-pre-lab.md)
- **WFO equity results**: [`equity_monthly_walkforward.md`](equity_monthly_walkforward.md)

*Back to [README.md](../README.md)*
