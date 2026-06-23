# Roadmap

**ARCHIVED — Legacy code removed 2026-06-16.** The codebase is now **lab-only** — all live execution engines, strategy orchestration, and per-coin backtesting infrastructure have been removed. This roadmap references deleted modules and phases that no longer exist. See [Architecture](architecture.md) for the current lab structure.

**Previous status (pre-cleanup, now archived):** Phase 1 deployed on Binance.US (legacy pre-reset params) but zero fills since go-live · Edge-search (2026-06-08) found no deployable crypto edge with the current signal library — cheap levers exhausted.

---

This document is kept as historical reference only. For forward direction on new strategy research, see the lab-based architecture in [Architecture](architecture.md) and available strategies in [CLI Reference](cli_reference.md).

---

> **Organization.** This doc is grouped by **function**, not chronology. §2 is time-bounded (next ~6 weeks); §§3–5 are open exploration. For "what happened when," see [`changelog.md`](changelog.md).
>
> **Vocabulary.** Acronyms used here — WFO, OOS (out-of-sample), CV (coefficient of variation), OHLCV (open/high/low/close/volume), PnL (profit and loss), ML (machine learning), RL (reinforcement learning) — are defined up front in the [Architecture Guide](architecture.md#vocabulary).

---

## Contents

1. [North star](#1-north-star) — revised thesis after the edge-search
2. [In flight](#2-in-flight-next-6-weeks) — venue migration, hardening, methodology debt, strategy-library redesign (time-bounded)
3. [Strategy methodology — beyond WFO](#3-strategy-methodology--beyond-wfo)
   - [3.1 Currently in the codebase](#31-currently-in-the-codebase)
   - [3.2 Pipeline integration debt](#32-methodology-debt--pipeline-integration)
   - [3.3 Near-term techniques (A–F)](#33-new-techniques-to-explore)
   - [3.4 ML / modern (G–M)](#34-ml--modern-techniques-longer-horizon)
4. [Infrastructure & ops](#4-infrastructure--ops)
5. [Shelved / deferred](#5-shelved--deferred)
6. [Reference](#6-reference)

---

## At a glance

| State | Detail |
|---|---|
| 🟢 **Live now** | Binance.US Phase 1 — legacy pre-reset params (ADA / DOGE / ETH / TRX), adaptive sizing capped at 10% per position, ~$137 portfolio. **Deployed 2026-05-23 but has never opened a trade** — the gates correctly find nothing tradeable in this regime. |
| 🟢 **Paper live** | Ensemble strategy on Alpaca paper ($103K account). Cron fires 1:30 PM PT Mon-Fri. First signal-driven run 2026-06-22 (DAY order fix applied same day). ML feature gate active, risk guardrails enforced. |
| ✅ **WFO validated** | **Ensemble + vol targeting clears WFO** (2026-06-20): OOS Sharpe 0.84 vs SPY 0.59, CAGR 18.8% vs 13.0%, MaxDD -23.4% vs -22.1%, WFE 1.25. First strategy to beat SPY risk-adjusted in honest out-of-sample testing across 17 rolling folds. |
| ✅ **ML feature gate** | LightGBM classifier trained on 3,203 historical entries (10 OHLCV features, 5-day fwd label). Avg precision 0.585, coverage 42%. Filters low-confidence buy signals in production. |
| ✅ **Risk guardrails** | 30 max positions, 3.3%/trade, 5% max concentration, 3% daily loss halt, 15% max drawdown halt. Telegram notifications on halt events. |
| 🔵 **Next up** | **Paper monitoring + live go-live.** Target: 5-10 clean trading days (by ~2026-07-07). Remaining for live: (1) fund Alpaca live account ($1K), (2) swap to live API keys, (3) confirm DAY orders fill cleanly on paper first. |
| 🧪 **Open exploration** | 13 methodology directions in §3. Near-term candidates: Bayesian param selection (C), funding-rate carry portfolio (L), stat-arb pairs (E). |
| ⏸ **Deferred** | Phase 2 crypto cutover, TRX deployment (90d, re-eval ≥ 2026-08-10), post-only limit entry on Kraken |

**Status legend:** ✅ Done · 🟡 Active · 🔵 Next · ⚪ Idea · ⏸ Deferred

---

## 1. North star

Build a **flexible multi-strategy trader** with a clean data → signal → execution → risk envelope, and deploy capital only against an edge that survives true out-of-sample (OOS) testing. The walk-forward-optimization-on-momentum-signals pipeline is **one technique among several** — funding-rate carry, cash-and-carry, and (planned) machine-learning / regime-conditional methods all coexist behind the same data layer, execution layer, and risk envelope.

**Revised thesis (2026-06-08).** The WFO textbook-reset arc suggested fees were the *sole* binding constraint: the deployable set was empty at Kraken taker (0.80% round-trip) and looked like it would open up at Binance.US taker (0.04% round-trip). The 2026-06-08 edge-search **disproved the optimistic half**: at Binance.US fees the realistic universe still yields **zero** full-cascade passers, and the widest universe's best portfolio (+13.8%) **loses to BTC buy-and-hold (+38.8%)** with no clean positive holdout. So the binding constraint is now understood as **fees *and* the current momentum signal library** — low fees were necessary but not sufficient. The forward direction is therefore: (a) redesign the signal library toward reversion / regime-aware methods (§3), and/or (b) take the same engine to a market where the existing methodology may have more room — **US equities via Alpaca (§5)**.

**Update (2026-06-12).** Direction (b) has now been tested honestly and the answer is in: the per-stock WFO tournament, run through the strategy-agnostic monthly walk-forward harness over 64 out-of-sample months of point-in-time S&P 500 (2021-02 → 2026-05), returned **+0.9% total vs SPY +103.8%** (Sharpe 0.12 vs 0.89) — a clear NO-GO. Momentum baselines through the same harness were market-like (xs_momentum +126.0%, Sharpe 0.82 — beats SPY on raw return only by taking ~30% more vol), which validates the harness while confirming there is no free risk-adjusted edge in plain cross-sectional momentum either. Full results: [`equity_monthly_walkforward.md`](equity_monthly_walkforward.md) §4. Conclusion: **the current WFO signal library does not transfer to equities** — the binding constraint is the signal library itself, in both markets. Direction (a), a genuinely different signal design, is the only live path; the equities harness is now the cheap, honest test bench for any candidate.

---

## 2. In flight (next ~6 weeks)

*Time-bounded work. Each subsection ordered by sequencing, not priority.*

### 2a. Venue migration — Kraken Pro → Binance.US

| Status | Step | Notes |
|:---:|---|---|
| ✅ | Binance.US smoke tests (auth, spread, depth) | Snapshots at `results/binanceus_smoke/snapshots.jsonl` |
| ✅ | `BinanceUSSpotBroker` integration | `src/ggTrader/execution/binanceus_spot.py` |
| ✅ | Historical OHLCV backfill — Binance.US top-50 universe (28 coins listed, 22 unlisted skipped cleanly) | Done 2026-05-23 via `scripts/backfill_binanceus_universe.py`. 19 coins have full 3-year history, 9 have partial. |
| 🟡 | **Phase 1** — live trader on Binance.US with **legacy pre-reset params**, small size | Deployed 2026-05-23. **Has never opened a trade** — the gates find nothing tradeable in this regime (consistent with the edge-search). Venue mechanics only partially validated: auth / data / heartbeat work, and a residential-IP rotation that broke the Binance.US key allowlist (`-2015` crash loop) was fixed 2026-06-04 by re-allowlisting — but fill quality and balance reconciliation remain **unexercised** because no order has fired. |
| ✅ | **Phase 2 prerequisite** — fix legacy `MIN_ROBUSTNESS_SCORE` gate dropping textbook output ([#11](https://github.com/garykuepper/ggTrader/issues/11)) | Fixed 2026-05-22. Root cause: legacy gate threshold `0.1` incompatible with the textbook rank-composite score (structurally always negative). Set default to `None`; added guard in `_apply_wfo_selection_gates`; made `base_execution_engine.load_optimized_params` `None`-safe. Verified: research run produces populated `per_coin`, engine loads symbols, heartbeat completes. |
| ⏸ | **Phase 2** — switch to textbook-validated set | **On hold.** The 2026-05-23 "textbook-validated" set didn't survive re-test: 2026-06-08 holdouts were SUI **−8.06%** and DOGE **−1.29% / flat**. There is nothing to cut over to until the signal library is redesigned (§2d / §3) — swapping params is pointless when none have a positive OOS holdout. |
| ⏸ | **TRX 90-day re-evaluation** (≥ 2026-08-10) | Criteria pre-registered in closure doc §5. Volume ≥ $500K/24h + spread ≤ 15bp → migrate; otherwise continue defer or unshelve Path B (Kraken maker-only for TRX). |

### 2b. Hardening / cleanup

| Status | Effort | Item | Notes |
|:---:|:---:|---|---|
| 🔵 | ~2h | GitHub issue #10 — `fee_entry` recording bug from live-execution audit | Fix + backfill scope. |
| ✅ | — | Rebuild live Docker image to pick up `f70d721` (venue filter on the database loader) | Done 2026-05-23. Pre-fix the loader's `(symbol, interval, date)`-only query was crashing `pivot()` with a duplicate-index error once Binance.US 4-hour bars started overlapping Kraken's in the `ohlcv` table. |
| ✅ | — | `auto_trader.py` `regime_allowance` alignment + visible tracebacks | Fixed 2026-05-22. `auto_trader.py` now surfaces tracebacks instead of swallowing them in the sleep loop. (`regime_allowance` itself was removed when the regime filter was deleted 2026-05-23.) |
| ✅ | — | Daily-loss circuit breaker (`DAILY_LOSS_LIMIT_PCT`, 5% intraday cap) | `base_execution_engine.py`, persistent across restarts, tests in `test_circuit_breaker.py`. |
| 🔵 | ~3h | Position health scoring ("zombie trade" detector) | Flag positions held 3× longer than backtest average holding time. |
| ✅ | — | Migrate file-based state to TimescaleDB | Done 2026-05-22 — `data/active_positions.json`, `data/spy_cache/*.parquet`, and `results/correlation_matrix/*/correlation_matrix.csv` all migrated to DB. Single source of truth for live state. |
| 🔵 | ~10min | Don't create `results/auto_trader_<ts>/` dir on every script invocation | `scripts/auto_trader.py` calls `ResultsManager("auto_trader")` unconditionally at startup, creating an empty timestamped dir that nothing writes to. Each verification/dry-run pollutes `results/`. Fix: lazy-init the manager only when there's actual output to persist, or skip entirely if `DRY_RUN` is set. |
| 🔵 | ~30min | Eliminate `run_results.json` Phase B — stop writing the file and remove the disk fallback in `state_manager.py` | Phase A (DB-first auto-detect with disk fallback) is already shipped. Phase B touches 10+ files (`cmd_backtest`, `cmd_production`, `cmd_signals`, `cmd_report`, `cmd_status`, `portfolio_optimizer`, `base_execution_engine`, etc.). Do as a focused session after Phase 1 completes. |

### 2c. Methodology debt (banked from WFO closure)

| Status | Item | Notes |
|:---:|---|---|
| 🔵 | Grid-size-aware parameter coefficient-of-variation (CV) gate | Current 0.30 threshold is mechanically inflated for large parameter grids — `psar_adx` jumped from CV 0.39 (4 cells) to 1.22 (32 cells) without any actual instability change. Fix: either axis-aware CV (`unique_picks / grid_size_per_axis`) or `log(N_cells)`-relative threshold. Theory-justify before re-running. |
| 🔵 | Pre-register *shape* metrics alongside distributional stats | Closure-doc §6 calibration lesson — gate-pass counts must be extracted on every research run, not retroactively reconstructed from worker logs. Cheap pipeline change. |

### 2d. Strategy-library redesign (active focus — banked from the 2026-06-08 edge-search)

The edge-search ([`archive/edge_search_report_2026-06-08.md`](archive/edge_search_report_2026-06-08.md)) closed out the "tune the cheap levers" hypothesis: gates, universe size, and venue/fees have all been swept and none produce a deployable crypto edge with the **current** entry library. The only remaining crypto lever is a different *signal*.

**Update (2026-06-18).** Mean-reversion strategies (`bb_reversion`, `rsi_reversion`) are the first signals to beat SPY risk-adjusted in WFO (Sharpe 0.80 vs 0.59, CAGR 17%+ vs 13%). Trailing stops confirmed destructive on reversion. The research direction is now: (1) vol targeting overlay, (2) signal ensemble (combine reversion + trend), (3) conviction-weighted sizing, (4) expand the reversion signal library. See plan: [`superpowers/plans/2026-06-19-edge-research-phase2.md`](superpowers/plans/2026-06-19-edge-research-phase2.md).

**Update (2026-06-20).** **Ensemble + vol targeting validated in WFO.** 17-fold rolling 12mo/3mo walk-forward on SP500 (2021-01 → 2026-04): OOS Sharpe 0.84 vs SPY 0.59, CAGR 18.8% vs 13.0%, WFE 1.25. Baseline ensemble without vol targeting nearly identical (Sharpe 0.83, CAGR 18.0%) — signal diversification is the primary driver. Circuit breaker fired once (fold 7) and self-recovered. Recommended live params: `min_agree=2, bb_period=20, bb_std=2.0, rsi_period=14, rsi_oversold=30, ema_fast=20, ema_slow=50, vol_target=0.25`. Stable in 11/17 folds.

| Status | Item | Notes |
|:---:|---|---|
| ✅ | Reversion-focused entry set (initial) | `bb_reversion` + `rsi_reversion` landed 2026-06-18. Both beat SPY in WFO. Trailing stops destructive on reversion — confirmed and documented. |
| ✅ | **Vol targeting overlay** | `compute_vol_scalar` + `simulate_signals` integration shipped + tested. WFO sweep: marginal benefit over baseline ensemble (+0.01 Sharpe). Best: `vol_target=0.20-0.25`. |
| ✅ | **Signal ensemble** | `EnsembleSignal` (bb+rsi+ema majority vote) **clears WFO** — OOS Sharpe 0.84, CAGR 18.8%, WFE 1.25. First strategy to beat SPY risk-adjusted in honest OOS. Paper trading on Alpaca. |
| 🔵 | Conviction-weighted sizing | Size positions by signal strength (BB distance below band, RSI depth below threshold) instead of fixed 2%. |
| 🔵 | Expanded reversion signals | MACD divergence, volume-confirmed reversion, multi-timeframe reversion. |
| ⏸ | Regime-aware *signal selection* (not just sizing) | Vol targeting gives 80% of the benefit at 10% of the complexity. Revisit after ensemble results. |
| ⏸ | Wire up / de-mock the cross-sectional + HMM paradigm | Deferred — focus is on reversion edge discovery, not infrastructure. |

---

## 3. Strategy methodology — beyond WFO

*Open exploration. Items here are not time-bounded — they're the menu of directions worth investing in.*

The codebase now hosts three distinct methodologies. The roadmap goal is to keep that pluralism — different markets reward different techniques, and a single-methodology bot is fragile.

> **Edge-search lens (2026-06-08).** Cheap levers (gates, universe, fees) are exhausted — see §2d. Re-read this menu accordingly: directions that change the **signal or its conditioning** (A regime sizing, C robust selection, D ML gate, E stat-arb, plus the reversion-focused entry set) now rank above anything that only re-tunes thresholds or re-weights the existing momentum entries — those will not clear the OOS bar the sweep just established.

### 3.1 Currently in the codebase

| Status | Methodology | Where |
|:---:|---|---|
| ✅ | **WFO on directional momentum signals** — live. 11 entry × 3 exit, monthly recalibration. Currently on legacy pre-reset params; Phase 2 param cutover is on hold (no positive-holdout set exists — §2d). | `core/wfo.py`, `core/orchestrator.py`, `indicators/strategies.py` |
| 🧪 | **Cross-sectional momentum + HMM regime gate** — landed 2026-06-06, **offline-only**. Whole-universe ranking paradigm with a Hidden Markov Model regime filter; evaluated via `run_cross_sectional_research.py` (writes `run_type='cross_sectional_research'`, ignored by the live trader). HMM emission features still mocked. Not wired to the per-coin live engine — see §2d. | `strategies/momentum/cross_sectional.py`, `strategies/regime/hmm_filter.py`, `backtesting/wfo.py` |
| ✅ | **Cash-and-carry (CashAndCarryBTC)** — Phase 3.5. Runs against synthetic basis; needs Kraken Futures historical data (or a venue with linear dated quarterlies) for real-data backtest. See [`archive/kraken_futures_backfill_design.md`](archive/kraken_futures_backfill_design.md). | `strategies/carry/cash_and_carry.py` |
| ✅ | **Funding-rate carry (FundingCarryBTC)** — Phase 4. Long spot + short PF_XBTUSD perp; harvest funding. Hysteretic thresholds. Backtested on real Kraken funding-rate history (2025-05-15 → 2026-05-18). | `strategies/carry/funding_carry.py` |

### 3.2 Methodology debt — pipeline integration

The carry strategies live in `strategies/carry/` under the **new** Phase 3 architecture (Signal → Sizer → Order, Strategy + FeatureStore protocols). The WFO strategies live in `indicators/strategies.py` under the **old** architecture. They are not yet unified.

| Status | Item | Notes |
|:---:|---|---|
| ⚪ | Unify old-arch WFO strategies with new-arch `Strategy` protocol | Touchpoints: `strategies/base.py`, `strategies/loader.py`, `core/orchestrator.py`. Major lift — done incrementally as each WFO strategy is touched. |
| ⚪ | Carry strategies run through the live `ExecutionEngine` (not just backtest) | Requires (a) `BinanceUSPerpetualBroker` or Kraken Futures wiring for the short leg, and (b) portfolio sizer that treats hedged pairs as one position. |
| ⚪ | Strategy ensemble layer — allocate capital across WFO + carry by risk-adjusted return | Replaces today's "WFO chooses everything" portfolio construction. Could be as simple as equal-risk weighting per methodology, or as fancy as a meta-optimizer. |

### 3.3 New techniques to explore

*Near-term — buildable on top of the existing stack. Roughly ordered by effort × expected payoff.*

| | Technique | Effort | Status |
|:---:|---|:---:|:---:|
| A | [Regime-conditional allocation](#a-regime-conditional-allocation-1-week) | ~1 wk | ⏸ Deferred — vol targeting covers 80% |
| B | [Volatility-targeting at the portfolio level](#b-volatility-targeting-at-the-portfolio-level-3-5-days) | ~3-5d | ✅ WFO validated (marginal over baseline) |
| B+ | [Signal ensemble](#b-signal-ensemble-2-3-days) | ~2-3d | ✅ **WFO validated — Sharpe 0.84 vs SPY 0.59** |
| C | [Bayesian / robust parameter selection](#c-bayesian--robust-parameter-selection-1-2-weeks) | ~1-2 wk | ⚪ |
| D | [ML feature gates on WFO entries](#d-ml-feature-gates-on-top-of-wfo-entries-2-3-weeks) | ~2-3 wk | ⚪ |
| E | [Statistical arbitrage / pairs trading](#e-statistical-arbitrage--pairs-trading-3-4-weeks) | ~3-4 wk | ⚪ |
| F | [Ensemble + rolling meta-allocation](#f-ensemble-of-strategies-with-rolling-window-meta-allocation-1-2-weeks-after-cd) | ~1-2 wk | ⚪ |

#### A. Regime-conditional allocation (~1 week)
The previous binary BTC bull/bear filter was removed in 2026-05-23 — it underperformed unfiltered trading and added complexity without measurable edge. The right replacement is to condition **position sizing** on regime, not entry permission: scale exposure by BTC trend strength (e.g., distance from EMA200, or realized-vol percentile). Also: add an **altcoin regime** based on BTC dominance, not just BTC price.

- Why: most of the 2026-05 fee/edge cliff is concentrated in chop regimes. If we already know the regime, we shouldn't size as if we don't.
- Risk: regime detection is itself fragile — easy to overfit. Use simple, theory-justified thresholds. The deletion of the binary filter is a deliberate reset; the replacement must justify its complexity against the now-default "no regime modulation."

#### B. Volatility-targeting at the portfolio level (~3-5 days)
Today's `--adaptive-sizing` is **per-coin** vol normalization. The portfolio-level version targets a fixed realized portfolio vol (e.g. 15% annualized) and rescales the whole book. This is a standard institutional technique that ggTrader doesn't yet use.

- Why: caps drawdown in regime breaks without needing to predict them. Works well with regime-conditional allocation above.

#### B+. Signal ensemble (~2-3 days)
Combine orthogonal signal types — reversion (bb_reversion, rsi_reversion) and trend (ema_cross) — into a single strategy that enters when N-of-M signals agree. This is fundamentally different from the WFO tournament (which picks one signal per fold): the ensemble holds multiple views simultaneously and enters on consensus.

- Why: reversion and trend signals are negatively correlated — when trend gets whipsawed in chop, reversion thrives. A diversified signal set smooths the equity curve, which is exactly what Sharpe rewards. The bb_reversion Sharpe of 0.80 is promising, but it will have drawdown periods where trend signals would have helped.
- Risk: majority-vote thresholds are another parameter to overfit. Keep it simple (2-of-3), validate with WFO, and compare against the best single signal.

#### C. Bayesian / robust parameter selection (~1-2 weeks)
WFO's point-estimate-per-fold is wasteful — it throws away the full in-sample (IS) distribution. A Bayesian alternative: build a posterior distribution over parameter performance, then pick parameters by **highest lower-credible-bound on OOS Sharpe ratio** rather than highest point estimate. Sparse-fire cells naturally get wider credible intervals → their lower bound is worse → they rank below dense-fire cells, no hard cliff needed.

- Why: the grid-size-dependent parameter-CV gate is a symptom of treating WFO as a hypothesis test instead of a parameter inference problem. The Bayesian framing dissolves the issue.
- Risk: significantly more compute (bootstrap or Markov Chain Monte Carlo). Justify with a side-by-side run before adopting.

#### D. Machine-learning (ML) feature gates on top of WFO entries (~2-3 weeks)
Train a binary classifier (gradient-boosted decision trees or a small neural network) on **WFO entry signals** with features = (regime indicators, microstructure features, time-of-day, recent realized volatility). Target: 1 if the entry's forward 24-hour return is positive, 0 otherwise. Use the classifier's probability as a gate — e.g., only take entries where `P > 0.55`.

- Why: signal-conditional filtering is the natural place ML adds value — far easier than predicting raw returns. Keeps the WFO methodology intact and just adds a guardrail.
- Risk: dataset is small (a few thousand entries across all coins/strategies). Heavy regularization required. Watch for look-ahead bias.

#### E. Statistical arbitrage / pairs trading (~3-4 weeks)
With Binance.US's fee structure, BTC/ETH or similar coin-pair "stat-arb" (statistical arbitrage) becomes viable in a way it isn't on Kraken. Cointegration test → z-score entry → reversion exit. Pure mean-reversion methodology, completely orthogonal to the trend-following WFO set.

- Why: diversification of methodology. Stat-arb performs best when trend strategies are getting chopped up.
- Risk: cointegration is unstable in crypto — relationships break. Needs continuous monitoring + automatic decommission of broken pairs.

#### F. Ensemble of strategies with rolling-window meta-allocation (~1-2 weeks, after C+D)
Once we have WFO + carry + (possibly) stat-arb + ML-filtered WFO, the meta-question is **how to allocate capital across methodologies**. Simple version: equal-risk weighting (each methodology gets capital proportional to `1/recent_volatility`). Stronger version: rolling-window meta-optimizer that allocates capital proportional to each methodology's recent risk-adjusted return, with a floor (never zero) so diversification is preserved.

### 3.4 ML / modern techniques (longer-horizon)

*Need more infrastructure (feature pipelines, model serving, monitoring) than §3.3. Listed roughly by ramp-up cost.*

| | Technique | Effort |
|:---:|---|:---:|
| G | [Gradient-boosted regime classifier](#g-gradient-boosted-regime-classifier-2-3-weeks) | ~2-3 wk |
| H | [Microstructure features from L2 order book](#h-microstructure-features-from-l2-order-book-3-4-weeks-needs-new-data-layer) | ~3-4 wk |
| I | [Deep RL for position sizing](#i-deep-rl-for-position-sizing-6-8-weeks-experimental) | ~6-8 wk |
| J | [Transformer signal models](#j-transformer-signal-models-6-10-weeks-experimental) | ~6-10 wk |
| K | [On-chain + sentiment features](#k-on-chain--sentiment-features-3-4-weeks-needs-new-data-layer) | ~3-4 wk |
| L | [Funding-rate / basis surface modeling](#l-funding-rate--basis-surface-modeling-2-3-weeks-builds-on-phase-4) | ~2-3 wk |
| M | [Cross-exchange arbitrage scanner](#m-cross-exchange-arbitrage-scanner-2-weeks-scoped-narrow) | ~2 wk |

#### G. Gradient-boosted regime classifier (~2-3 weeks)
Build a regime model from scratch (the prior binary BTC EMA filter was deleted on 2026-05-23). An XGBoost or LightGBM classifier outputs `P(market is trending)` from features like realized volatility (multi-window), the term structure of BTC futures basis, BTC dominance (BTC's share of total crypto market cap), BTC funding rate, USDT premium against USD, and volume-weighted Average Directional Index (ADX). Use the probability as a continuous regime score that scales position sizing (§3.3.A).

- Why: regimes aren't binary. A soft probability is more informative than an EMA crossover and degrades more gracefully when wrong.
- Risk: feature drift in crypto is real. Retrain monthly alongside the WFO recalibration; monitor calibration with reliability diagrams.

#### H. Microstructure features from L2 order book (~3-4 weeks; needs new data layer)
Subscribe to Level-2 (L2) order book snapshots from Binance.US (or an aggregator) and persist 1-minute features: bid-ask imbalance, depth-weighted spread, top-of-book pressure, large-trade footprint. Use as **additional features** for the ML gate (§3.3.D) or as a standalone "execution quality" signal that delays an entry by one bar if microstructure looks toxic.

- Why: most academic crypto-alpha research post-2022 lives in microstructure, not in OHLCV. WFO on bar data leaves money on the table that L2 features would catch.
- Risk: order book history is expensive to store. Need TimescaleDB compression + tiered retention. The signal may also decay under Binance.US's already-low 0.04% round-trip fee regime — but worth checking.

#### I. Deep reinforcement learning (RL) for position sizing (~6-8 weeks; experimental)
A PPO (Proximal Policy Optimization) or SAC (Soft Actor-Critic) agent that, given current state = (portfolio, open positions, recent price + regime features, time-since-entry), outputs a per-coin sizing action in `[0, MAX_COIN_ALLOCATION]`. **Entry signals stay rule-based** (WFO/carry); only the sizing decision is learned. Reward = risk-adjusted realized PnL with a drawdown penalty.

- Why: position sizing is where path-dependence really matters and where a learned policy plausibly beats heuristics like the Kelly criterion or volatility-targeting. Constraining RL to sizing (not entry selection) keeps the action space small and interpretable.
- Risk: RL on financial data is notoriously sample-poor. Walk-forward training with strict no-lookahead is essential; first iterations will likely produce noisy results. Treat as research, not a deployable line item.

#### J. Transformer signal models (~6-10 weeks; experimental)
Pretrain a small transformer (Temporal Fusion Transformer (TFT) style, or patch-based) on multi-coin OHLCV + features, then fine-tune to predict next-K-bar return distributions. Use the predicted **distribution shape** (not just the point estimate) — e.g., enter when the median is positive AND the 10th percentile is above a volatility-scaled threshold.

- Why: transformers handle multi-series multi-horizon prediction well and naturally output distributions. Crypto data volumes (~100,000 bars/coin × 50+ coins × multi-feature) are right at the edge of what works without overfitting.
- Risk: highest overfit risk on the roadmap. Pure research item; would gate adoption on rigorous OOS evaluation against the current WFO baseline.

#### K. On-chain + sentiment features (~3-4 weeks; needs new data layer)
Integrate on-chain metrics (exchange netflow, stablecoin issuance, miner reserves) from a provider like Glassnode or CryptoQuant, plus sentiment (Fear & Greed already wired; add Twitter/Reddit volume + tone via cheap LLM-based scoring). Use as features for the ML gate (§3.3.D) or as standalone regime overlays.

- Why: orthogonal information sources to price and volume. Crypto's structural reflexivity means on-chain often leads price more reliably than equity equivalents do.
- Risk: data quality and provider lock-in. Free tiers usually give daily granularity, which limits intraday use. Start with the cheapest viable feed and validate signal-to-noise before scaling up.

#### L. Funding-rate / basis surface modeling (~2-3 weeks; builds on Phase 4)
Generalize `FundingCarryBTC` (single-coin BTC perpetual-futures funding) to a **portfolio of funding carries** across BTC / ETH / SOL perpetuals, with dynamic allocation by realized funding vs. realized hedge cost. Add a basis-arbitrage leg: when the perpetual-vs-spot basis exceeds a funding-equivalent threshold, lean into the dislocation.

- Why: funding-carry has the highest empirical Sharpe ratio of any strategy in the codebase. Scaling it across coins is the lowest-friction way to add carry exposure.
- Risk: capacity-limited — funding rates compress as capital flows in. Position size must respect funding-rate response to deployment (your own trade moves the basis).

#### M. Cross-exchange arbitrage scanner (~2 weeks; scoped narrow)
A read-only monitor that compares Binance.US, Kraken, Coinbase, OKX prices on the same pairs and logs persistent spreads beyond 50 basis points (0.5%). Not a trading strategy yet — start by **measuring** whether the opportunity exists at our account size and API access level. If yes, scope an executor.

- Why: cheap to build, immediate diagnostic value. If structural spreads exist they're free alpha for the cost of multi-venue exchange-API integration.
- Risk: low. Worst case it confirms there's no opportunity at our size.

---

## 4. Infrastructure & ops

| Status | Item | Notes |
|:---:|---|---|
| ✅ | Binance.US data architecture | Multi-venue OHLCV schema verified end-to-end (2026-05-23). 28 of top-50 Binance.US coins backfilled; 22 unlisted skipped cleanly. |
| ✅ | Per-venue availability (listing) registry | 2026-06-07. Layer-1/2 per-venue listing snapshots under `data/universe/*_listings.json`; the ranker now **hard-fails without a snapshot** instead of silently ranking unlistable coins. |
| ✅ | Research backtests on the execution venue | 2026-06-04. `EXCHANGE`-aware data resolution so WFO optimizes on Binance.US order books (not Kraken) and auto-tails fresh bars past the 05-23 go-live freeze. |
| ✅ | Volume-floor research universe | 2026-06-04. Universe selected by a min avg-daily-USD floor ($50K/day → ~15 liquid Binance.US pairs) instead of a fixed top-N; fixed a sum-masquerading-as-average bug (~30× inflation). |
| ✅ | WFO metric extraction — 2.07× faster | 2026-06-05. Collapsed per-fold vectorbt metric accessors (58% of runtime) to a single `returns()` extraction + vbt numba kernels. Bit-identical; all gate verdicts unchanged. |
| ✅ | Offline gate-replay + fee-override tooling | 2026-06-08. `scripts/gate_replay.py` replays the 4 aggregate gates on a snapshot; `GGTRADER_FEES` env override in `run_config.py` for fee-tier experiments. |
| ✅ | Container runs as non-root `appuser` (uid 1000) | 2026-06-07. Dockerfile hardening; results tree chowned off root. |
| ⚪ | MLflow experiment tracking for WFO runs | ~4h. Runs are currently directories under `results/research/`; a visual diff + search interface would be a material UX improvement. |
| ✅ | Per-run `wfo_stats_snapshot.json` | Landed during WFO arc. Discipline: never purge the cache before reading what the run produced. |
| ⚪ | Large language model (LLM) post-mortem reports | ~4h. Auto-generate natural-language performance commentary for the daily PnL report. |
| ⚪ | `ggt compare` — diff two research runs for selection drift | Helpful when comparing fee/edge regime runs (the WFO arc did this by hand). |
| ⚪ | Backtest-vs-real "slippage gap" replay | Replay backtest logic against real fills to quantify slippage attribution per coin. |

---

## 5. Shelved / deferred

| Status | Item | Why |
|:---:|---|---|
| ⏸ | Post-only limit entry on Kraken | Binance.US 0.02% taker is 12× cheaper than Kraken 0.25% maker. No code complexity needed. Spec at [`superpowers/specs/2026-05-12-post-only-limit-entry.md`](superpowers/specs/2026-05-12-post-only-limit-entry.md) — kept as reference for the 90-day TRX re-eval scenario. |
| ⏸ | TRX deployment (90-day defer) | Binance.US TRX volume too thin ($2K/24h, 1246× less than Kraken). Re-eval criteria pre-registered (closure doc §5). |
| ⚪ | Multi-timeframe gates (4h on daily trend) | Carried over from prior roadmap. Worth revisiting after methodology pluralism (§3.3) is in place. |
| ⏸ | **US equities via Alpaca — research answered NO-GO for current methodology (2026-06-12)** | Reconsidered 2026-06-08; the research question is now answered. The honest 64-month out-of-sample S&P 500 walk-forward ([`equity_monthly_walkforward.md`](equity_monthly_walkforward.md) §4) shows the WFO tournament earns +0.9% vs SPY +103.8%, and momentum baselines are market-like (Sharpe 0.82 vs SPY 0.89). **Do not build Alpaca execution for the current signal library** — there is nothing to deploy. The monthly harness (`--strategy` plug-ins) stays as the test bench; revisit execution work only after some strategy beats SPY risk-adjusted per §5 of that doc. Reference material still archived: [`archive/stock_trading_plan.md`](archive/stock_trading_plan.md), [`archive/alpaca_cli.md`](archive/alpaca_cli.md). |

---

## 6. Reference

- **Daily ops**: [`live_trading_guide.md`](live_trading_guide.md), [`cli_reference.md`](cli_reference.md)
- **System layout**: [`architecture.md`](architecture.md)
- **Dated history**: [`changelog.md`](changelog.md)
- **WFO reset + venue migration arc**: [`superpowers/closures/2026-05-12-wfo-textbook-reset-and-venue-migration.md`](superpowers/closures/2026-05-12-wfo-textbook-reset-and-venue-migration.md)
- **Completed Phase notes (historical)**: [`archive/phase3_architecture_feedback.md`](archive/phase3_architecture_feedback.md), [`archive/phase4_funding_carry.md`](archive/phase4_funding_carry.md), [`archive/kraken_futures_backfill_design.md`](archive/kraken_futures_backfill_design.md)

*Back to [README.md](../README.md)*
