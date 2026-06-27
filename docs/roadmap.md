# Roadmap

> **Organization.** §1 is the thesis. §2 is time-bounded work (~4 weeks). §3 is open research. §4 is infrastructure. For dated history see [`changelog.md`](changelog.md). For the pre-lab roadmap (crypto venue migration, WFO textbook reset, etc.) see [`archive/roadmap-pre-lab.md`](archive/roadmap-pre-lab.md). For the full engineering snapshot, see [`project_snapshot_2026-06-23.md`](project_snapshot_2026-06-23.md).

---

## At a glance

| State | Detail |
|---|---|
| ✅ **Lab shipped** | vectorbt-centric research bench (2026-06-15). 10 strategies, parameter sweep, WFO framework, 17-fold rolling windows, DB persistence. |
| ✅ **5-voter shipped as default (2026-06-25)** | The 2026-06-24 ablation overturned the dilution thesis (adding voters didn't dilute — the broken NDH gate did). The live-config decision is now **made and coded**: `DEFAULT_VOTERS = FIVE_VOTERS` (`bb+rsi+ema+macd+vbb`, MTF dropped) in `ensemble.py`. Best edge = **`core+macd+vbb`: Sharpe 0.89 / DD -10.5% / gate 14/17 / anchor only 3/17**. MTF stays out (the one genuinely harmful voter, `core+mtf` 0.49). Tooling: `scripts/ablation_voters.py`. |
| ✅ **3% exposure shipped — beats SPY outright (2026-06-25)** | Root cause of the CAGR gap was under-deployment (5-voter sat 61.5% in cash). `SIGNAL_POSITION_SIZE = 0.03` (up from 0.02) now the default in `data.py`. OOS **CAGR 16.2% / Sharpe 1.09 / DD -11%** beats SPY (13.0% / 0.58 / -22.1%) at a *higher* Sharpe. Lever diagnostic: `scripts/lever_diagnostic.py`. |
| ✅ **NDH variance-cap exemption (2026-06-26)** | `ndh_check` no longer over-rejects perfectly-robust plateaus: when every neighbor is profitable (density 1.0), Sharpe dispersion isn't an overfit signal, so the variance cap is exempted. SP500 gate tally **13/17 → 16/17**; OOS **Sharpe 1.12 / CAGR 16.3% / DD -11%**, WFE 1.13 (no overfit). |
| ✅ **Stability-aware live params (2026-06-26)** | `select_live_params` now prefers fold-proven combos (`_pick_live_winner`, `MIN_LIVE_STABILITY`) instead of the best score on the most-recent train window — closes an overfit-to-recent-regime trap (SP500 moved from a 0/17-stability combo to a 2/17 fold-proven one). |
| ✅ **ML feature gate** | LightGBM classifier (precision 0.585, 10 features, 3.2K samples). ATR true-range fix landed; gate retrained (1.9.0). |
| 🟢 **Paper live** | 5-voter ensemble on Alpaca paper ($102,459), `SIGNAL_POSITION_SIZE = 0.03`. Cron fires 1:30 PM PT Mon–Fri. ML gate + risk guardrails active. **Honest fill logging (2026-06-26):** only real fills booked to `paper_trades`; after-close/partial fills reconciled next run via `paper_pending_orders`. PnL baseline reset 2026-06-23. |
| 🔵 **Next up** | Paper monitoring of the shipped 5-voter/3% config → fund Alpaca live ($1K) → live go-live. Research: portfolio blend (SP500 + MidCap 400), reversion-aware regime map. |
| 🧪 **Research** | Portfolio blend / diversification, regime-aware exposure, IC-weighted voting, Kelly sizing. |

**Status legend:** ✅ Done · 🟢 Live · 🔵 Next · 🧪 Research · ❌ Rejected · ⏸ Deferred

---

## 1. North star

Build a **flexible multi-strategy research lab** with honest walk-forward evaluation, and deploy capital only against edges that survive true out-of-sample testing. The lab tests strategies on US equities (SP500/Nasdaq-100/Russell 2000) via vectorbt, with rolling 12mo/3mo folds and point-in-time universe membership.

**Current thesis (2026-06-26).** The ensemble works because BB reversion, RSI exhaustion, and EMA trend signals are negatively correlated in their failure modes — when one gets whipsawed, the others don't fire. MACD and VolBB *add* edge (the earlier "dilution" was a gate/exit-bug artifact); only MTF is genuinely harmful. The **5-voter `core+macd+vbb` config is now the shipped default**, and the CAGR gap to SPY was closed not by a smarter signal but by **fixing under-deployment** — flat 3% position sizing deploys the idle 61% cash and beats SPY outright at a higher Sharpe (16.2% CAGR / 1.09). With voters, exposure, and the gate fixes (NDH neighborhood + variance exemption) all landed, the remaining levers are **diversification (portfolio blend across SP500 + MidCap 400), reversion-aware regime exposure, weighted voting, and ML gate fixes.**

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
| ✅ | **Stability investigated — deployable config found (2026-06-24)** | — | The "0/17 stability" was a metric artifact (demands exact 5-tuple repeat). Per-axis winner modes are stable: ema_fast=10 (13/17), bb_std=2.5 (11/17), rsi_oversold=25 (10/17); `min_agree_exit` is noise; `min_agree` is the one regime-dependent knob (=1 won in 2024, =2 elsewhere). A **single fixed modal 3-voter** (min_agree=2, min_agree_exit=2, bb_std=2.5, rsi_oversold=25, ema_fast=10), deployed unchanged across all 17 folds with no reopt: **OOS Sharpe 0.70 vs SPY 0.58, MaxDD -13.0% vs -22.1%, WFE 1.28** — but **CAGR 7.7% vs SPY 13.0%**. CAGR gap later closed by 3% exposure scaling (research direction A). |
| ✅ | **3% exposure scaling shipped (2026-06-25)** | `data.py` | `SIGNAL_POSITION_SIZE` 0.02 → 0.03. Deploys the idle 61% cash; **OOS CAGR 16.2% / Sharpe 1.09 / DD -11% beats SPY outright.** See research direction A. |
| ✅ | **NDH variance-cap exemption (2026-06-26)** | `gates.py` | Variance cap exempted when neighborhood density = 1.0 (all neighbors profitable). SP500 gates 13/17 → 16/17; OOS Sharpe 1.12 / CAGR 16.3% / DD -11%, WFE 1.13. |
| ✅ | **Stability-aware live-param selection (2026-06-26)** | `wfo.py` | `select_live_params` prefers fold-proven combos (`_pick_live_winner`, `MIN_LIVE_STABILITY`) over best-recent-window score — closes an overfit-to-recent-regime trap. |

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
| ✅ | Deploy 5-voter + 3% exposure to paper (Docker rebuild, gate retrained 1.9.0) |
| ✅ | Honest paper fill logging + next-run reconciliation (`paper_pending_orders`) |
| 🟢 | Monitor paper fills of the shipped config for 5–10 clean trading days |
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
| A | **Exposure scaling (was: regime-adaptive `min_agree`)** | ✅ **shipped** 🧪 | **Resolved and shipped (2026-06-25).** Root cause of the CAGR gap = **under-deployment**: the static 5-voter sat 61.5% in cash. **Exposure scaling won and is now the default** — `SIGNAL_POSITION_SIZE=0.03` (up from 0.02) in `data.py` gives **OOS CAGR 16.2% / Sharpe 1.09 / DD -11%**, beating SPY outright (13.0% / 0.58 / -22.1%) at a higher Sharpe. `min_agree` can't lift CAGR past ~12%; vol-targeting is dominated (0.25 → 15.0% / 0.96 / -14.5%). **Surprise:** reversion earns most in *down/turbulent* regimes and is flat in calm uptrends — inverse of trend-following intuition, so naive "scale up in calm uptrend" gating would *hurt*. **Still open:** does a reversion-aware regime map beat flat 0.03? (`scripts/lever_diagnostic.py`.) |
| B | **Dynamic position sizing (Kelly)** | ⚪ | Use ML gate probability + recent win/loss ratio to compute fractional Kelly sizing per trade. Quarter-Kelly standard for estimation error. `EnsembleConvictionSignal.sizes` already supports per-bar sizing. |
| C | **Cross-sectional entry ranking** | ⚪ | When multiple entries fire simultaneously, rank by conviction score + sector diversification. Prevents correlated cluster entries. |
| D | **Macro-enriched ML features** | ⚪ | Add VIX level/change, sector relative strength, earnings proximity, short interest to `extract_features()`. Orthogonal to price-based technicals. Requires external data pipeline. |
| E | **Regime-aware WFO folds** | ⚪ | Exponential time-weighting within training windows to discount data from misaligned regimes. **High risk**: adds new decay-rate parameter that can itself be overfit. Reduces effective sample size. Deprioritized behind simpler regime approaches (A). |
| F | **Statistical arbitrage / pairs** | ⚪ | Cointegrated pair z-score reversion (orthogonal to momentum). |
| H | **Portfolio blend (Large + Mid Cap diversification)** | 🧪 | **Tool shipped 2026-06-25** (`scripts/portfolio_blend.py`). Computes OOS WFO returns for SP500, MidCap 400, a unified blended universe, and 50/50, 70/30, and risk-parity blends; measures asset-class correlation and diversification benefit. Motivated by the favorable midcap Sharpe edge (direction G) — open question is whether a blend lifts risk-adjusted return above either sleeve alone. Note: risk-parity weighting carries the usual in-sample-vol caveat. |
| G | **S&P MidCap 400 universe** | 🟡 | **Researched 2026-06-25** (`scripts/midcap_research.py`, bias-quantified — clean PIT data doesn't exist publicly, so current snapshot + calibrated haircut). **Promising but not deploy-clean.** Midcap reversion beats the midcap index decisively: raw CAGR 10.9% / Sharpe 0.91, bias-adjusted 14.5% / 1.07, vs **MDY 9.1% / 0.40** (huge Sharpe edge); ≈ SPY's 14.7% CAGR but far better Sharpe (1.07 vs 0.70). Survivorship Δ is **favorable** (SP500 calibration: snapshot *under*states PIT by 3.6pp CAGR — the dropped names are good reversion fodder — so the snapshot result is a conservative floor). **BUT the WFO ran anchor-driven: gates passed only 6/17 folds, anchor used 15/17, and the circuit breaker halted at fold 5.** So this validates the *defensive config* on midcaps, not adaptive selection. Same over-rejection pattern as the pre-fix SP500 NDH gate — the gates likely over-reject on noisier midcap names. **Next: investigate gate/circuit-breaker calibration for midcaps before any deployment.** Data: midcap400 snapshot universe + MDY backfilled; coverage 400/400. |

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
| ✅ | MCP server for research DB | `scripts/mcp_server.py` (FastMCP) — query backtest runs/metrics, list strategies/tickers, ops status, run backtests over TimescaleDB. Code-review hardened (contextlib.closing conn handling, env creds). |
| ✅ | Portfolio-blend analysis tool | `scripts/portfolio_blend.py` — SP500 / MidCap 400 / blended-universe + 50-50/70-30/risk-parity OOS comparison with correlation. |
| ✅ | Stability-aware live-param selection | `select_live_params` / `_pick_live_winner` in `wfo.py` (`MIN_LIVE_STABILITY`) |
| ✅ | Honest paper fill logging + reconciliation | Real fills only to `paper_trades`; pending fills reconciled next run via `paper_pending_orders` |
| ✅ | Any-universe backfill + benchmark | `data.py` — backfill any universe with its benchmark (wired midcap400 + MDY) |
| ⚪ | WFO DB persistence | Persist per-fold WFO results to TimescaleDB (deferred to go-live) |
| ⚪ | `ggt compare` — diff two research runs | Side-by-side metric comparison |
| ⚪ | Per-signal backtest within WFO folds | Required for IC-weighted voting (Phase 2). Run individual signal backtests during training sweep. |

---

## 5. Key Research Findings (2026-06-15 → 2026-06-23)

### What Worked

| Finding | Evidence |
|---------|---------|
| 5-voter ensemble (BB+RSI+EMA+MACD+VolBB) | **Shipped default.** Ablation (2026-06-24): Sharpe 0.89 / DD -10.5% at 0.02 sizing, 14/17 gate-validated. With 3% exposure + NDH variance-exemption (2026-06-26): **OOS Sharpe 1.12 / CAGR 16.3% / DD -11%**, 16/17 gates — beats SPY (0.58 / 13.0% / -22.1%). |
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
| 2026-06-24 | Lever diagnostic: CAGR gap is under-deployment, not signal — exposure scaling beats SPY |
| 2026-06-25 | **5-voter + 3% exposure shipped to main & paper** (`DEFAULT_VOTERS=FIVE_VOTERS`, `SIGNAL_POSITION_SIZE=0.03`). Gate retrained 1.9.0. First config to beat SPY outright (16.2% CAGR / 1.09). |
| 2026-06-25 | MidCap 400 research (snapshot + MDY backfill, bias-quantified) — promising but anchor-driven. Senior code-review pass (voter bypass, concentration guardrail, O(1) sweep lookups, conn leaks). |
| 2026-06-25 | MCP research server + portfolio-blend tool (code-review hardened) |
| 2026-06-26 | **NDH variance-cap exemption** — SP500 gates 13→16/17, OOS Sharpe 1.12 / CAGR 16.3% / DD -11% |
| 2026-06-26 | Stability-aware `select_live_params` (prefer fold-proven combos) |
| 2026-06-26 | Honest paper fill logging + next-run reconciliation (`paper_pending_orders`) |

---

## Reference

- **Project snapshot**: [`project_snapshot_2026-06-23.md`](project_snapshot_2026-06-23.md)
- **Architecture**: [`architecture.md`](architecture.md)
- **CLI usage**: [`cli_reference.md`](cli_reference.md)
- **Changelog**: [`changelog.md`](changelog.md)
- **Pre-lab roadmap**: [`archive/roadmap-pre-lab.md`](archive/roadmap-pre-lab.md)
- **WFO equity results**: [`equity_monthly_walkforward.md`](equity_monthly_walkforward.md)

*Back to [README.md](../README.md)*
