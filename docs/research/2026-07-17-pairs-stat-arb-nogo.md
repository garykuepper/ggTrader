# Market-Neutral Pairs / Stat-Arb Mean Reversion: NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-17
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/pairs_stat_arb.py`
(`PairsStatArb`): the lab's **first market-neutral construction** — every
prior strategy tried (ensemble reversion, momentum, leveraged-ETF timing) is
long-only directional. Mechanism: enumerate same-GICS-sector symbol pairs
from the SP500 universe, filter to pairs with trailing 126-day return
correlation ≥ 0.7, compute a naive 1:1 (unhedged) log-price spread z-score
per qualifying pair, and go long the relatively-cheap leg / short the
relatively-expensive leg whenever `|z| ≥ entry_z`, holding until `|z| ≤
exit_z` (subject to a `min_hold_months` floor), re-scored at each monthly
rebalance. Swept `entry_z` (1.5/2.0/2.5), `exit_z` (0.25/0.5/0.75,
constrained `< entry_z`), `max_pairs` (5/10/20), `min_hold_months` (1/2) —
54 raw combos, full honest walk-forward, SP500 universe, `eval_start
2015-01-01` through present (42 folds, 12mo train / 3mo test).

This was candidate Rank 1 from `RESEARCH_SNAPSHOT.md` §6's internally-derived
list, picked specifically because it's the cheapest market-neutral idea to
execute (needs only OHLCV already in TimescaleDB, no new data source) and
its diversification value was expected to hold even at a modest standalone
Sharpe. **Result: NO-GO on both counts.** The strategy is not merely
unprofitable — its return stream costs money outright (negative CAGR), and
the walk-forward stability/gate metrics show the mechanism isn't finding a
real edge, not just a low one. It did, however, cleanly achieve the
*mechanical* market-neutrality goal (near-zero realized correlation/beta to
SPY) — see §2 — which at least validates that the long/short construction
itself works as designed; the failure is in the entry/exit signal, not the
market-neutral plumbing.

Closed/exhausted context this report does not re-litigate: the long-only
technical-indicator ensemble (deployed, Sharpe 1.12 SP500 core), the
leveraged-ETF rotation and trend-following NO-GOs (`2026-07-16-*`), and the
entry-gate/exit-rule/sizing rejections cataloged in `RESEARCH_SNAPSHOT.md`
§4. None of those inform this result — this is a genuinely different
mechanism (securities-pair mean reversion, not single-symbol timing) and its
failure mode is also different (see §6).

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Sizing |
|---|---|---|---|---|---|
| **Pairs/stat-arb (this report)** | -0.42 | -2.6% | -28.2% | 13/42 folds | 1/`max_pairs` per leg pair, weight-based |
| **SPY benchmark** | 0.88 | 15.2% | -33.7% | N/A | Buy-and-hold |

Aggregate WFE **-0.16** (target ≥ 0.50 — not just missed, negative, meaning
OOS performance is *worse* than train performance predicts, the signature of
overfitting rather than a real but modest edge). The regime-circuit-breaker
halt was active for the large majority of the 42-fold history (`[H]` flag on
28 of 42 folds) — like the rejected breadth-driven leveraged rotation, this
strategy spent most of its backtest trading defensive anchor params, not its
own selected signal. The WFO's final "recommended live" combo
(`entry_z2.5_exit_z0.5_max_pairs20_min_hold_months2`) was selected as train
winner in only **1 of 42 folds** — the same fold-instability signature
(noise, not edge) that has sunk IC-weighted voting, Kelly sizing, and
overnight-gap reversion in prior arcs.

**Market-neutrality check (the strategy's actual design goal, not just
Sharpe):** OOS return correlation to SPY = **0.092**, beta = **0.030**. This
is genuinely low — the long/short construction is doing its job
mechanically, unlike a strategy that claims market-neutrality but still
carries hidden directional beta. But near-zero correlation to a benchmark
that returned +15.2%/yr is not useful if the strategy's own return is
negative; market-neutrality is a necessary, not sufficient, condition for
this to be a good diversifier.

## 3. Actionable Research Directions

None ranked — this is a closure report. See §6 for the one redesign
question worth a decisive look before abandoning same-sector pairs trading
entirely, and the parked alternative construction.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Same-sector, unhedged (1:1), monthly-rebalanced pairs/stat-arb mean
reversion — REJECTED.** OOS Sharpe -0.42 vs SPY 0.88, CAGR -2.6% vs 15.2%,
MaxDD -28.2% vs SPY's own -33.7% (worse risk-adjusted despite a nominally
shallower drawdown — the strategy loses money slowly rather than tracking
the market). Aggregate WFE -0.16 (below the 0.50 floor and negative, not
just low). Regime halt active 28/42 folds; live-recommended params selected
in only 1/42 folds (unstable, noise-driven selection). Market-neutrality
design goal was achieved (OOS correlation to SPY 0.092, beta 0.030) — the
long/short mechanics work correctly; the entry/exit signal itself (same-
sector correlation-filtered spread z-score) does not carry a tradeable edge
at this cadence and hedge ratio. See §6 for the specific redesign
hypothesis (daily/weekly rebalance vs. this run's monthly cadence) that
would need to be tested separately before concluding the *mechanism class*
(pairs/stat-arb generally) is dead, not just this particular
implementation.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy or further sweep this same-sector, monthly-rebalanced,
unhedged pairs implementation.** The evidence (negative WFE, majority-halt,
1/42 param stability) points to overfitting/noise, not a modest-but-real
edge worth tuning. Before writing off market-neutral pairs trading as a
mechanism class entirely, run the one decisive follow-up named in §6 (faster
rebalance cadence) — if that also fails, close the pairs/stat-arb line for
good rather than trying further variants (beta-hedging, cross-sector pairs,
tighter correlation filters) on top of a cadence that may be structurally
too slow for what mean-reversion pair spreads actually need.

No live-config changes — this strategy is not deployed and this report does
not propose deploying it.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** classic pairs-trading literature (Gatev, Goetzmann
& Rouwenhorst and most practitioner implementations) monitors spreads daily
and can enter/exit within days of a divergence — this lab's `weights`-
strategy harness only supports **monthly** rebalancing
(`ggTrader.lab.data.rebalance_dates` returns one date per calendar month,
shared by every `target_kind="weights"` strategy in the lab, not a choice
`PairsStatArb` made). Is the NO-GO here actually a verdict on same-sector
spread mean-reversion as a mechanism, or is it a verdict on trying to run a
signal whose natural holding period is days-to-weeks through an
infrastructure built for monthly-cadence portfolio construction? A z-score
that crosses `entry_z` mid-month and reverts before the next month-end
rebalance is invisible to this harness entirely — the strategy can only see
and act on whatever the spread looks like on the last trading day of each
month.

**Resolution:** Plausible enough to be worth one decisive test, not
resolved by this report. The regime-halt rate (28/42) and negative WFE both
look like a signal that's mistimed rather than one that's fundamentally
absent — those are consistent with "arrives at the wrong observation
frequency," not just "no edge exists." But this is not free to test: it
would require either (a) a genuinely daily/weekly rebalance path added to
the `weights` harness (a real infrastructure change, not a parameter sweep —
touches `rebalance_dates`, `_sweep_fold_weights`, and every existing
weights-strategy consumer's assumptions about rebalance cadence), or (b) a
scoped variant that only changes `pairs_stat_arb`'s own re-scoring frequency
without changing the shared harness. Do not build either speculatively —
this is a real engineering investment and should only be undertaken with a
specific decision to keep investigating pairs/stat-arb, not as a reflexive
next step.

### Parked Direction: Beta-hedged, faster-cadence pairs construction
If the monthly-cadence hypothesis above is confirmed worth pursuing, the
redesign should also drop two other `v1` simplifications flagged in the
implementation's own docstring at the same time (retest once, not
incrementally): the naive 1:1 unhedged spread (a proper hedge ratio from a
rolling OLS/Kalman beta would reduce spread noise unrelated to the actual
mean-reverting component), and the complete absence of short-borrow-fee/
margin-interest cost modeling (irrelevant to this NO-GO — costs would only
have made the negative result worse — but would matter before any future
GO). Gate: only pick this up after the cadence question above is answered;
building a beta-hedge on top of a cadence mismatch would confound which fix
actually helped.
