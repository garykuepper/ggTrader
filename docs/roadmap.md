# Roadmap

> **Organization.** §1 is the thesis. §2 is time-bounded work (~4 weeks). §3 is open research. §4 is infrastructure. For dated history see [`changelog.md`](changelog.md). For the pre-lab roadmap (crypto venue migration, WFO textbook reset, etc.) see [`archive/roadmap-pre-lab.md`](archive/roadmap-pre-lab.md). For the full engineering snapshot, see [`project_snapshot_2026-06-23.md`](project_snapshot_2026-06-23.md).

---

## At a glance

| State | Detail |
|---|---|
| ✅ **Lab shipped** | vectorbt-centric research bench (2026-06-15). 10 strategies, parameter sweep, WFO framework, 17-fold rolling windows, DB persistence. |
| ✅ **Voter ablation redone under fixed gates (2026-06-24)** | The 2026-06-23 ablation ran on the broken NDH gate (anchor-driven) + pre-ATR exits. Clean 11-config rerun on full 603-stock SP500 **overturns the dilution thesis**: adding voters didn't dilute — the bug did. Best real edge = **`core+macd+vbb` (drop MTF, 5 voters): Sharpe 0.89 / CAGR 10.5% / DD -10.5% / gate 14/17 / anchor only 3/17** (the 3 anchors are the 2022 bear folds; folds 5–17 all PASS). 6-voter is healthy too (0.92 / -10.3% / 11/17), not the old "0.36 dilution". 3-voter core: 0.68 / -20.5% / 10/17. SPY: 0.58 / 13.0% / -22.1%. Tooling: `scripts/ablation_voters.py` (3-way parallel, ~30min). |
| ⚠️ **MTF is the one harmful voter (2026-06-24)** | `core+mtf` is the worst non-degenerate config (Sharpe 0.49 / CAGR 6.0%). The 2026-06-23 "MTF harmful" call survives the rerun. Dropping bb or rsi → zero trades (2-voter unanimity artifact). ema is the droppable core voter (`core-ema` = bb+rsi still 0.79). **Open: live-config decision** — current paper trader runs the 6-voter; `core+macd+vbb` is the better candidate but switch not yet made. |
| ✅ **ML feature gate** | LightGBM classifier (precision 0.585, 10 features, 3.2K samples). Known defect: ATR feature uses close-only proxy, not true range. |
| 🟢 **Paper live** | Ensemble on Alpaca paper ($102,459) — actually the **6-voter** (`DEFAULT_VOTERS = ALL_VOTERS` in `ensemble.py`), not the 3-voter as previously documented. Cron fires 1:30 PM PT Mon–Fri. ML gate + risk guardrails active. PnL baseline reset 2026-06-23. **Candidate switch: `core+macd+vbb` (5v) — best ablation config.** |
| 🔵 **Next up** | Asymmetric exit architecture + ATR fix → WFO validation → paper monitoring → live go-live. |
| 🧪 **Research** | IC-weighted voting, regime-adaptive gating, exit optimization. |

**Status legend:** ✅ Done · 🟢 Live · 🔵 Next · 🧪 Research · ❌ Rejected · ⏸ Deferred

---

## 1. North star

Build a **flexible multi-strategy research lab** with honest walk-forward evaluation, and deploy capital only against edges that survive true out-of-sample testing. The lab tests strategies on US equities (SP500/Nasdaq-100/Russell 2000) via vectorbt, with rolling 12mo/3mo folds and point-in-time universe membership.

**Current thesis (2026-06-24, revised).** The ensemble works because BB reversion, RSI exhaustion, and EMA trend signals are negatively correlated in their failure modes — when one gets whipsawed, the others don't fire. The earlier claim that "adding MACD/VolBB voters dilutes" was a **gate/exit-bug artifact** — under the fixed NDH gate + ATR exits, MACD and VolBB *add* edge: **`core+macd+vbb` (5 voters, MTF dropped) is the best config** (Sharpe 0.89, DD -10.5%, 14/17 gate-validated). Only MTF is genuinely harmful. So the lever set is: **drop MTF, prefer the 5-voter config, then smarter exits, weighted voting, and ML gate fixes.**

---

## 2. In flight (~4 weeks)

### Phase 1: Alpha Architecture Fixes (Immediate)

These are high-payoff, low-risk changes to the existing codebase. Each must be WFO-validated before deployment.

| Status | Item | File(s) | Notes |
|:---:|---|---|---|
| ✅ | **Asymmetric exit logic** | `ensemble.py` | RSI exit fires independently (`exits = rsi_ext \| (exit_votes >= min_agree_exit)`). New `min_agree_exit` sweep param (default=`min_agree` for backward compat). Both `EnsembleSignal` and `EnsembleConvictionSignal` updated. |
| ✅ | **ATR True Range fix** | `feature_gate.py`, `train_gate.py` | `extract_features()` now accepts optional `high`/`low` series for true ATR. `filter_buys()` and `_build_dataset()` thread OHLCV high/low through. Falls back to close-diff when unavailable. Retrain `ensemble_gate.joblib` after fix. |
| ✅ | **WFO validation of Phase 1** | `wfo.py` | Done on full 603-stock universe (2026-06-24). Finding: gates reject ~every fold for both voter sets → results are anchor-driven, not edge-validated. Asymmetric exit + ATR fix are in; the blocker is now the over-strict gates, not the exits. |
| ✅ | **Gate over-rejection fixed (2026-06-24)** | `gates.py`, `wfo.py` | Root cause: NDH used a diagonal (Moore) ±1 neighborhood that, on a coarse grid, counted 47/48 cells as "neighbors" → grid-wide quality check, never a plateau. Fixed: axis-aligned (von-Neumann) neighborhood + exclude regime params (min_agree). Gate PASS folds: 3-voter 0→11/17, 6-voter 2→11/17. WFO now deploys real winners. Remaining marginal lever: variance cap (0.20). |
| ✅ | **Stability investigated — deployable config found (2026-06-24)** | — | The "0/17 stability" was a metric artifact (demands exact 5-tuple repeat). Per-axis winner modes are stable: ema_fast=10 (13/17), bb_std=2.5 (11/17), rsi_oversold=25 (10/17); `min_agree_exit` is noise; `min_agree` is the one regime-dependent knob (=1 won in 2024, =2 elsewhere). A **single fixed modal 3-voter** (min_agree=2, min_agree_exit=2, bb_std=2.5, rsi_oversold=25, ema_fast=10), deployed unchanged across all 17 folds with no reopt: **OOS Sharpe 0.70 vs SPY 0.58, MaxDD -13.0% vs -22.1%, WFE 1.28** — but **CAGR 7.7% vs SPY 13.0%**. Verdict: a stable, deployable **defensive** edge (wins risk-adjusted + drawdown, trails raw growth). Natural next lever to lift CAGR = regime-adaptive `min_agree` (research direction A). Live deployment decision deferred. |

### Phase 2: ML Gate & Voting Upgrades

| Status | Item | File(s) | Notes |
|:---:|---|---|---|
| ⚪ | **IC-weighted signal voting** | `ensemble.py`, `wfo.py` | Replace unweighted `entry_votes` sum with Sharpe-weighted linear combination. Per-signal IS Sharpe computed during each WFO fold's training sweep. Zero-out signals with negative train Sharpe. Requires running individual signal backtests per fold. |
| ⚪ | **Adaptive ML threshold** | `feature_gate.py` | Replace static 0.55 threshold with rolling percentile of recent classifier scores. Loosens gate in low-vol (collect steady alpha), tightens in high-vol (avoid cluster risk). Start with score-based percentile; VIX integration deferred until data pipeline exists. |
| ⚪ | **Exit signal optimization** | `ensemble.py`, `sweep.py` | Add `exit_mode` sweep param: symmetric (current), time-based (N days), profit-target, RSI-priority. WFO grid sweeps over exit variants alongside entry params. |

### Phase 3: Paper → Live Go-Live

| Status | Step |
|:---:|---|
| ✅ | Alpaca paper adapter, signal runner, trader, Telegram alerts, cron |
| ✅ | ML feature gate (LightGBM, precision 0.585) |
| ✅ | Risk guardrails (30 max positions, 3.3%/trade, 5% concentration, 3% daily loss halt, 15% drawdown halt) |
| ✅ | DAY time-in-force fix for fractional/notional orders |
| ✅ | PnL baseline reset ($102,459 — 2026-06-23) |
| 🟢 | Monitor paper fills for 5–10 clean trading days |
| 🔵 | Deploy Phase 1 alpha fixes to paper (after WFO validation) |
| 🔵 | Fund Alpaca live account ($1K) |
| 🔵 | Swap to live API keys |
| 🔵 | Confirm position sizing / order fills match expectations |

### Completed & Rejected (2b)

| Status | Item | Notes |
|:---:|---|---|
| ✅ | MACD divergence signal | `MACDDivergenceSignal` — built and tested. **Helps under fixed gates (2026-06-24):** `core+macd+vbb` Sharpe 0.89 vs core 0.68. (Old "near-zero" call was a gate-bug artifact.) |
| ✅ | Volume-confirmed BB reversion | `VolumeBBReversionSignal` — built and tested. **Helps:** part of the best config `core+macd+vbb` (14/17 gate-validated, anchor only 3/17). |
| ✅ | Multi-timeframe reversion | `MultiTimeframeReversionSignal` — built and tested. **Harmful** (confirmed 2026-06-24 rerun): `core+mtf` Sharpe 0.49 vs core 0.68 — the only voter that consistently degrades. |
| ↩️ | 6-voter ensemble | Original "rejected, Sharpe 0.36, dilution" verdict **reversed (2026-06-24)**: under fixed gates the 6-voter is 0.92 / DD -10.3% / 11-17 gate-pass. The standout is the **5-voter (6-voter minus MTF)**, not the 3-voter. |
| ✅ | Conviction-weighted sizing | `EnsembleConvictionSignal` — Sharpe 0.83 vs 0.84 baseline. Risk reducer, not alpha generator. Available but not deployed. |
| ✅ | ML pre-screen script | `scripts/ml_signal_screen.py` — LightGBM precision gate for evaluating new signals. |
| ❌ | Trailing stops on reversion | Both fixed and ATR-adaptive stops destroy reversion returns by exiting during expected drawdown. |
| ❌ | Momentum strategies | `xs_momentum`, `dual_momentum` — deeply negative OOS Sharpe. Well-arbitraged in US large caps. |

---

## 3. Research directions

*Open exploration — not time-bounded. Ordered by expected payoff given ablation findings.*

| | Direction | Status | Notes |
|:---:|---|:---:|---|
| A | **Regime-adaptive signal selection** | 🧪 | Classify market regime (trending vs mean-reverting via ADX/vol) and conditionally activate only signals suited to current regime. Must be trained only on WFO train window to prevent lookahead. **Now the top lever:** 2026-06-24 stability data shows `min_agree` is the one regime-dependent knob (looser entry won in 2024) — adapting it is the most direct path to lift the fixed 3-voter's CAGR (7.7%) toward SPY while keeping its Sharpe/drawdown edge. |
| B | **Dynamic position sizing (Kelly)** | ⚪ | Use ML gate probability + recent win/loss ratio to compute fractional Kelly sizing per trade. Quarter-Kelly standard for estimation error. `EnsembleConvictionSignal.sizes` already supports per-bar sizing. |
| C | **Cross-sectional entry ranking** | ⚪ | When multiple entries fire simultaneously, rank by conviction score + sector diversification. Prevents correlated cluster entries. |
| D | **Macro-enriched ML features** | ⚪ | Add VIX level/change, sector relative strength, earnings proximity, short interest to `extract_features()`. Orthogonal to price-based technicals. Requires external data pipeline. |
| E | **Regime-aware WFO folds** | ⚪ | Exponential time-weighting within training windows to discount data from misaligned regimes. **High risk**: adds new decay-rate parameter that can itself be overfit. Reduces effective sample size. Deprioritized behind simpler regime approaches (A). |
| F | **Statistical arbitrage / pairs** | ⚪ | Cointegrated pair z-score reversion (orthogonal to momentum). |
| G | **S&P MidCap 400 universe** | ⚪ | Institutional blind spot — more inefficient than SP500, more liquid than Russell 2000. Requires PIT membership data sourcing. |

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
| ✅ | NDH + DSR robustness gates | Neighbor density hurdle, deflated Sharpe ratio |
| ✅ | 2.07x WFO speed | Collapsed per-fold metric accessors to single `returns()` extraction |
| ✅ | WFO per-fold progress output | `flush=True` + `PYTHONUNBUFFERED=1` for Docker real-time output |
| ✅ | `--max-stocks` preload trim | Universe trimmed before data load, not just strategy selection |
| ⚪ | WFO DB persistence | Persist per-fold WFO results to TimescaleDB (deferred to go-live) |
| ⚪ | `ggt compare` — diff two research runs | Side-by-side metric comparison |
| ⚪ | Per-signal backtest within WFO folds | Required for IC-weighted voting (Phase 2). Run individual signal backtests during training sweep. |

---

## 5. Key Research Findings (2026-06-15 → 2026-06-23)

### What Worked

| Finding | Evidence |
|---------|---------|
| 5-voter ensemble (BB+RSI+EMA+MACD+VolBB) | **Best config (2026-06-24):** OOS Sharpe 0.89 / DD -10.5% vs SPY 0.58 / -22.1%. 14/17 folds gate-validated (anchor only in 2022 bear). |
| 3-voter ensemble (BB+RSI+EMA) | OOS Sharpe 0.68 / DD -20.5% under fixed gates. Solid but beaten by the 5-voter. (Earlier 0.61–0.84 figures were a 50-stock subset.) |
| Negatively correlated signal families | Reversion + trend signals cancel each other's failures. |
| vectorbt grouped simulation | 50-stock, 17-fold WFO in ~47s. 2.07x speedup from profiling. |
| Point-in-time universe membership | Prevents 1-3% annual survivorship bias inflation. |
| NDH + DSR robustness gates | Caught overfit parameter spikes across 17 folds. |

### What Failed

| Finding | Evidence |
|---------|---------|
| ~~6-voter ensemble dilution~~ | **Retracted (2026-06-24).** The 0.61→0.36 "dilution" was a broken-gate + pre-ATR-exit artifact. Under fixed gates: 6-voter 0.92, 5-voter 0.89. Adding voters helps; only MTF hurts. |
| Trailing stops on reversion | Exits during expected drawdown, destroying the reversion thesis. |
| Momentum strategies | OOS Sharpe -5.11. Well-arbitraged in US large caps. |
| MTF signal | Ablation: removing it *improved* Sharpe 0.55 → 0.64. Actively harmful. |
| Conviction sizing | Sharpe 0.83 vs 0.84. No alpha improvement. |

### Known Defects

| Defect | Location | Impact |
|--------|----------|--------|
| ~~ATR uses close-only proxy~~ | ~~`feature_gate.py`~~ | **Fixed.** Now uses true range when high/low available. |
| ~~Symmetric exit consensus~~ | ~~`ensemble.py`~~ | **Fixed.** RSI exit fires independently; `min_agree_exit` decouples entry/exit thresholds. |
| Static ML threshold | `feature_gate.py:164` | Blocks good trades in low-vol, passes bad trades in high-vol |
| Model needs retraining | `models/ensemble_gate.joblib` | Stale: trained with close-only ATR proxy. Retrain with `python -m ggTrader.lab.train_gate`. |

---

## 6. Evolution timeline

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
| 2026-06-23 | **6-voter expansion → rejected** (later retracted). OOS Sharpe 0.36. Ablation: RSI critical, MTF harmful, rest noise. |
| 2026-06-24 | **Voter ablation redone under fixed gates → dilution thesis overturned.** 11-config rerun (`scripts/ablation_voters.py`): best = `core+macd+vbb` (5v, Sharpe 0.89, 14/17 gate). MACD/VolBB help; only MTF harmful. 6-voter healthy (0.92). |
| 2026-06-23 | WFO infrastructure fixes (sweep grid 12K→24, max-stocks preload, progress output) |
| 2026-06-23 | PnL baseline reset ($102,459). "Reverted to 3-voter" was documented but **never coded** — live still runs the 6-voter default. |
| 2026-06-23 | Project snapshot + alpha expansion engineering report |

---

## Reference

- **Project snapshot**: [`project_snapshot_2026-06-23.md`](project_snapshot_2026-06-23.md)
- **Architecture**: [`architecture.md`](architecture.md)
- **CLI usage**: [`cli_reference.md`](cli_reference.md)
- **Changelog**: [`changelog.md`](changelog.md)
- **Pre-lab roadmap**: [`archive/roadmap-pre-lab.md`](archive/roadmap-pre-lab.md)
- **WFO equity results**: [`equity_monthly_walkforward.md`](equity_monthly_walkforward.md)

*Back to [README.md](../README.md)*
