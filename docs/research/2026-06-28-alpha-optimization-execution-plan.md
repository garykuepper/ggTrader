# Engineering Strategy & Research Report: Alpha Optimization and Diversification Execution Plan

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** June 28, 2026
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

The ggTrader system is an automated, vectorized research lab and execution pipeline designed to evaluate whether systematic trading strategies can reliably outperform a buy-and-hold benchmark of the S&P 500 index after transaction costs. The core infrastructure is built on Python, utilizing the vectorbt library for high-speed grouped portfolio simulations, TimescaleDB for storing point-in-time constituent histories, and a daily execution script running on paper trading accounts.

The current production system deploys a **5-voter mean-reversion ensemble** (Bollinger Bands, RSI, EMA cross, MACD divergence, and Volume-BB). Trades execute when a minimum consensus of two voters is met (oversold); exits trigger independently when RSI crosses back above 50 or a consensus exit occurs. Positions are sized at a flat 3% of equity per trade, deploying ~61% of otherwise idle capital.

In honest out-of-sample (OOS) testing under a rolling 12-month-train / 3-month-test walk-forward optimization (WFO) engine, the system achieves a Sharpe ratio of 1.12 versus SPY's 0.58, a CAGR of 16.3%, and a maximum drawdown of -11%. Integrity is validated by a dual-gate harness: a Neighborhood Density Hurdle (NDH) to eliminate unstable parameter choices and a Deflated Sharpe Ratio (DSR) filter to adjust for selection bias and data-snooping.

```
                                 +--------------------------------+
                                 |   Point-in-Time S&P 500 DB     |
                                 +---------------+----------------+
                                                 v
                                 +---------------+----------------+
                                 |  Rolling 12M / 3M WFO Engine   |
                                 +---------------+----------------+
                                                 v
                                 +---------------+----------------+
                                 |    Robustness Gates: NDH/DSR   |
                                 +---------------+----------------+
                       +-------------------------+-------------------------+
                       | (Pass)                                            | (Fail)
                       v                                                   v
         +-------------+-------------+                       +-------------+-------------+
         |   Recommended Parameters  |                       |   Conservative Anchor     |
         +-------------+-------------+                       +-------------+-------------+
                       +-------------------------+-------------------------+
                                                 v
                                 +---------------+----------------+
                                 |   5-Voter Ensemble Execution   |
                                 |    (BB, RSI, EMA, MACD, VOL)   |
                                 +---------------+----------------+
                                                 v
                                 +---------------+----------------+
                                 |  Capital Allocation / Sizing   |
                                 +--------------------------------+
```

Extensive post-mortem analyses have established that **per-trade selection levers are exhausted**. ML entry gating via LightGBM on daily technical features proved anti-predictive; exit-rule tuning (fixed profit targets, time stops) failed OOS, with replacement-style exits generating -39.7% drawdowns. Trailing stops, momentum overlays, weekly multi-timeframe indicators, and a 6th voter were all rejected for performance degradation.

The edge is therefore **structural and scale-based, not selection-based**. This report confines further equity research to structurally bounded voting and capital-allocation levers that are executable on a $1,000 account, and parks all market-neutral/alternative-asset work behind an explicit capital gate.

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Sizing |
|---|---|---|---|---|---|
| **Validated 5-Voter Ensemble** | 1.12 | 16.3% | -11.0% | 16/17 folds | Flat 3% / signal |
| **S&P 500 (SPY)** | 0.58 | ~13.0% | -22.0% | N/A | Buy-and-hold |

## 3. Actionable Research Directions (S&P 500 Equity Book)

Ranked by `Expected Payoff × OOS Survival Probability / Implementation Effort`, restricted to levers executable on the current $1,000 account.

### Rank 1: Spearman Rank IC-Weighted Voting

**Mechanism.** Replace the unweighted majority rule with votes scaled by each indicator's dynamic Spearman rank Information Coefficient (IC) over the 12-month training window.

Because the voters are per-name boolean triggers feeding a consensus count, the IC is **not** computed on the boolean outputs. Instead, on each day *t* in the training window we rank all *N* universe names by the indicator's raw value (e.g., raw RSI level, or % distance of price below the lower Bollinger Band) and compute the Spearman correlation between that cross-sectional ranking and the cross-sectional ranking of the subsequent 3-day forward returns. That scalar is `IC_j`.

Voting weights (updated only at the quarterly rebalance, not daily):

```
w_j = max(0, IC_j) / Σ_k max(0, IC_k)
```

An indicator with negative or statistically insignificant training IC gets weight 0 (pruned). Entry fires when `Σ w_j · Signal_{j,t} ≥ Threshold_t`.

**Degrees-of-freedom guard.** Define a boolean voter's "error" as `vote − sign(3-day forward return)` (vote = 1 buy / 0 hold; sign = 1 if return > 0 else 0). If the pairwise correlation of two voters' error series in-sample exceeds 0.70, cluster them and assign a single shared weight.

**Why it differs from rejected work.** The 2026-06-24 voter-reconciliation under corrected gates established the 5-voter pool (dropping the weekly MTF voter) as the optimal config at OOS Sharpe 0.89 — proving the old "6-voter dilution" collapse (0.61→0.36) was a *broken-gating artifact*, not signal dilution. This lever adds no indicators and touches no entry/exit code; it only re-weights the existing validated pool. It is the single signal-side refinement with zero strategy-code change.

**WFO framework.** Sweep consensus threshold ∈ [0.30, 0.70] and IC lookback ∈ {3, 6, 12} mo inside each training fold; select on OOS Sharpe. The DSR gate **must count the IC re-weighting, threshold, and lookback as additional trials**. Restrict weight updates to the quarterly step.

**Payoff / Effort / Failure.** Moderate (+0.05 to +0.10 Sharpe, unvalidated). Medium effort (`ensemble.py` aggregation). Primary risk: weight lag — IC computed over a trending training fold over-allocates authority to the wrong voter when the test fold flips regime, producing whipsaws.

### Rank 2: Bounded Expectancy-Scaled Sizing (Global Exposure Scalar)

**Mechanism.** Replace flat 3% with a sizing that scales by the book-level rolling win probability *p* and payoff ratio *b* (avg win return / avg loss return) from the active 12-month fold. Because *p* and *b* are **book-level scalars, not per-asset metrics**, this is not per-position Kelly — every new entry receives the *same* multiplier. It is a single dynamic global exposure scalar on the base 3% size:

```
S_t = clip( φ · (p·b − (1−p)) / b ,  0.0,  Cap )      # Cap = 1.5
size_t = 0.03 · S_t                                    # base 3% scaled
```

with φ ∈ [0.10, 0.25] (conservative fractional-Kelly multiplier). Enforce a hard 100%-deployed portfolio-sum constraint (never leveraged). Require ≥ 50 historical trades before *p, b* are trusted.

**Why it differs from rejected work.** Flat 3% beat SPY chiefly by deploying idle cash unused under 2% sizing — the edge is in scale. This scales that lever by realized book-level expectancy, leaving signal generation untouched. Unlike the rejected conviction-weighted model (which sized on single-signal depth and overfit), it uses only portfolio-level win/loss statistics.

**WFO framework.** Sweep φ ∈ [0.10, 0.25] in-fold; select on composite Sharpe + Sortino.

**Payoff / Effort / Failure.** Low (+0.02 to +0.05 Sharpe, unvalidated; the +0.15 Sharpe / -15% DD claims from earlier drafts are unsupported). Medium effort (`signal_runner.py` size conversion). Primary risk: **adverse equity-curve-momentum chasing** — `S_t` is a lagged book-level scalar, so scaling up on recent high expectancy is a bet on PnL persistence; a rapid regime flip pushes size to the 1.5× cap at the local equity-curve peak, magnifying the ensuing drawdown.

### Rank 3: Cross-Sectional Concurrency Ranking

**Mechanism.** When concurrent entry signals exceed available cash, the current system fills in arbitrary (alphabetical / DB-ingest) order. Replace that arbitrary tiebreak with a deterministic rank at the execution boundary: rank firing names by % distance below their 20-day moving average and fill only the top *N* deepest deviations until cash is exhausted.

**Two prerequisite checks before any code (both cheap, both gating).**
1. **Measurability.** Confirm the vectorbt sim actually rations cash *and* is fill-order-sensitive. If grouped cash-sharing fills all concurrent signals in the backtest, the "arbitrary fill" baseline does not exist in-sim and this lever is a live-only effect that WFO cannot measure. Establish this first.
2. **Is depth even predictive?** Ranking by deviation depth asserts "deeper oversold ⇒ larger reversion." That is a cross-sectional selection claim — the same family the LightGBM entry-gate falsified. Pre-screen with the Rank 1 IC machinery: does 20-DMA deviation depth have positive cross-sectional IC on forward reversion return? If not, principled ranking is no better than random fill — stop. (Build the IC harness once; reuse it here.)

**WFO framework.** Simulate the full universe with and without the ranking filter; go/no-go on aggregate OOS Sharpe + Sortino vs the arbitrary-fill baseline. DSR counts the ranking rule as a selection trial.

**Payoff / Effort / Failure.** Low-Medium (+0.03 to +0.05 Sharpe, unvalidated, conditional on both prerequisites passing). Low effort (`rank_entries()` before order submission). Primary risk: reversion-profile truncation — if deep-deviation names are high-beta and mutually correlated, prioritizing them in systemic panics raises realized cluster risk and magnifies drawdowns.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. VIX-Based Regime Throttling — REJECTED.** Throttling book exposure on VIX is the dropped regime-gating lever. Mean-reversion profits from down-market turbulence and short-term vol wicks, so a VIX throttle fires exactly when the edge is strongest, muting profitable turbulence and destroying OOS alpha.

**B. Multi-Universe Asset Sleeves — REJECTED.** The gate-honest 3-way blend (SP500, MidCap 400, Nasdaq-100) with dynamic allocation produced blended Sharpe 1.05 (< 1.12 SP500 core) and target-vol 3-sleeve CAGR 14.08% (< 16.35% core). Diversification *hurt*, driven by the measured 0.70 SP500↔MidCap correlation — these are not independent return streams, making "Sharpe > 2.00" claims mathematically impossible.

**C. MidCap 400 Survivorship Bias Assumption — CORRECTED.** The assumed 1–3%/yr survivorship inflation was disproven: the PIT snapshot delta is *favorable* — the snapshot *understates* historical returns rather than overstating them.

## 5. Operational Roadmap: Recommended First Action

The S&P 500 reversion research phase is complete; the 5-voter config passed all robustness gates (OOS Sharpe 1.12) and is paper-validated. **Recommended first action: go live — fund the $1,000 retail account with real capital and run the validated system as-is (flat 3% sizing).**

This establishes an operational baseline, reconciles real transaction costs and execution latencies against the vectorbt model, and proves the daily cron jobs run robustly on the server.

All further research runs as parallel branches. No code reaches live until it survives the identical WFO framework, passes NDH and DSR, and shows statistically significant outperformance over the live baseline.

## 6. Quantitative Contrarian Evaluation & Parked Research

**Is further S&P 500 reversion research a productive use of limited development resources, or is the strategy at its structural limit?**

The contrarian case: US large-caps are the most efficient, most heavily arbitraged asset class; OOS Sharpe 1.12 is a logical ceiling for daily price-action reversion on a retail latency profile. Every attempt to second-guess individual trades (LightGBM filter, take-profit optimization) degraded OOS performance — selection-based edge on this feature set is exhausted.

**Resolution.** Run live as a low-maintenance capital-preservation sleeve, and execute **Rank 1 (IC-weighted voting) as the singular final experiment** on the equity book. If IC-weighting fails to beat the 1.12 baseline post-DSR, the equity book is officially closed to further research, and development transitions to alternative asset classes with structural, leverage-driven premiums.

### Parked Direction: Delta-Neutral Carry Arbitrage Sleeve on Kraken Futures

Opened only once operating capital exceeds **$10,000**. Buy spot on Kraken Spot, short equivalent notional on Kraken Futures, capture perpetual funding / calendar basis (timed via the CF Benchmarks KFRI). Open when 30-day average funding or basis exceeds an 8–10% APR cost-adjusted hurdle.

**Why parked, not active:**
- **Capital.** On $1,000, splitting across spot/futures sub-accounts with the necessary 2–3× collateral buffer on the short leg deploys only a fraction of capital — dollar returns are negligible.
- **Data (verified 2026-06-28).** The funding data is *thinner than assumed*. `funding_rates` currently holds **1 symbol, ~12 months (2025-05-15 → 2026-05-18), stale ~6 weeks**; `perp_ohlcv` reaches back to 2022 but is also single-symbol and stale. A 12M/3M WFO yields essentially zero clean OOS folds on one name. **The first carry task is a multi-symbol funding backfill + live ingestion job — not a hurdle sweep.**
- **Risk profile.** Carry is negative-skew (collect micro-payments, expose to severe liquidation tails on basis spikes / execution lag); margin liquidation on the short leg is the dominant failure mode.
