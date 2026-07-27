# Market-Neutral Pairs / Stat-Arb Mean Reversion: NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-17 (re-run and confirmed 2026-07-26)
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

> **STATUS 2026-07-26 — re-run complete, NO-GO CONFIRMED, candidate CLOSED.**
> The 2026-07-25 audit (§2.6) invalidated this verdict for 2023 onward: SPY
> index contamination made the correlation-coverage guard reject every pair,
> so the strategy was believed to have selected nothing for roughly the last
> third of its window. The contamination is now fixed (`1b9b7f5`) and the full
> 42-fold WFO has been re-run on corrected data. **Every headline number came
> back unchanged and the verdict stands.** See §7 for the re-run, including one
> anomaly in the comparison that is deliberately left open rather than
> explained away.

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
32 of 42 folds — this report originally said 28; recounted from both run logs
2026-07-26) — like the rejected breadth-driven leveraged rotation, this
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
just low). Regime halt active 32/42 folds; live-recommended params selected
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

### Parked Direction: Beta-hedged, faster-cadence pairs construction (§7 note)
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

## 7. Re-run on Corrected Data (2026-07-26) — Verdict Confirmed

The 2026-07-25 implementation audit invalidated this report's verdict for the
2023+ portion of its window (audit §2.6). This section records the re-run that
resolves it.

### 7.1 The contamination mechanism is real, and measured

`pairwise_correlations` requires `len(sub) >= lookback * 0.8` jointly-valid
rows after `dropna()` (`pairs_stat_arb.py:55-56`). Once SPY's stray
`04:00`/`05:00` rows inject a NaN row for every real row, no pair can reach 80%
coverage of a 126-row window. Measured directly on the frame the WFO actually
builds (`cli.py` path, `eval_start` 2015-01, 764-symbol SP500 + SPY, contaminated
vs `collapse_daily_duplicates`-corrected, memo cache cleared between arms):

| Rebalance date | Qualifying pairs (contaminated) | Qualifying pairs (corrected) |
|---|---|---|
| 2022-11-30 | 2,753 | 2,753 |
| 2023-04-28 | **0** | 1,104 |
| 2023-09-29 | **0** | 552 |
| 2024-06-28 | **0** | 371 |
| 2025-06-30 | **0** | 1,980 |

The cliff to exactly zero lands precisely at the 2022-12-27 boundary where the
stray rows begin, and the corrected counts decline smoothly — the genuine
post-2022 correlation breakdown the audit predicted. Confirmed separately that
the real `cli.py` load did carry the damage: 4,199 raw rows across only 3,557
unique dates (642 duplicate-date rows, four times-of-day: 04:00/05:00/16:00/
17:00), collapsing to 3,373 clean rows.

**Correction to the audit's supporting evidence.** The audit's A/B probe
reported that a no-SPY frame *also* returned 0 qualifying pairs, which would
have contradicted the SPY explanation. That line was an artifact:
`_cached_qualifying_pairs` keys on `(asof, corr_lookback, corr_min, eligible)`
and never on the DataFrame (`pairs_stat_arb.py:106`), so the second arm
returned the first arm's cached value instead of a second measurement. The
table above clears the cache between arms. The audit's conclusion was right;
one of its measurements was not.

### 7.2 Re-run result: unchanged

Full 42-fold WFO, 54 combos, SP500, rolling 12mo train / 3mo test, on corrected
data:

| Metric | Original (2026-07-17) | Re-run (2026-07-26) |
|---|---|---|
| OOS Sharpe | -0.42 | **-0.42** |
| OOS CAGR | -2.6% | **-2.6%** |
| OOS MaxDD | -28.2% | **-28.2%** |
| SPY baseline Sharpe | 0.88 | 0.88 |
| Aggregate WFE | -0.16 | **-0.16** |
| Gate pass rate | 13/42 | 13/42 |
| Regime halt active | 32/42 | 32/42 |
| Winner param stability | 1/42 folds | 1/42 folds |

Train-side metrics did move as the Sharpe-deflation fix predicts (train Sharpe
0.56 → 0.62, train CAGR 3.7% → 4.1%), and the fold-16 train winner changed —
so the re-run demonstrably executed corrected code (the WFO table also now
renders the fixed `IS_SR`/`OOS_SR` columns).

**The NO-GO stands on every count**: negative OOS Sharpe and CAGR against SPY's
0.88/15.2%, negative WFE, halt active on 32 of 42 folds, and live-recommended
params selected in 1 of 42 folds. Nothing in the corrected data rescues the
strategy.

### 7.3 Open anomaly — not explained, deliberately not explained away

The OOS aggregate is **bit-identical** across the two runs on three separate
statistics, and the per-fold `n/a` WFE pattern in the 2023+ folds matches
exactly (folds 29/31/36/37/39/40/41/42 `n/a` in both; folds 30/32/33/34/35/38
carrying real values of -1.42, 2.24→2.25, -2.79→-2.80, 1.29, 0.04, -2.33).

That is difficult to reconcile with §7.1. If the pre-fix run had genuinely
traded a flat book for the last third of its window, those folds should have
produced *no* OOS return stream and every one of them should read `n/a` — and
the stitched OOS curve, and therefore the aggregate, should have moved when
they started trading. Two hypotheses were checked and rejected:

- **Halt path bypassing the strategy.** It does not; on a halt or gate failure
  the harness sets `deploy_params = anchor.params` and still runs the strategy
  (`wfo.py:695-700`), so a pairless book stays flat regardless.
- **SPY joined the frame after 2026-07-17.** It did not — `universe + ["SPY"]`
  has been in `cli.py` since `81300e6` (2026-06-15), before the original run.

A third possibility not yet tested: the stray SPY rows may have been *backfilled*
into the DB between 2026-07-17 and the 2026-07-25 audit, in which case the
original run was executed against a then-clean frame and was never contaminated
at all — which would explain the identical aggregate exactly. This is checkable
from row-insertion timestamps and is the leading candidate.

**Why this does not gate the verdict:** both readings lead to the same place.
Either the original run was clean (and -0.42 was always a valid measurement),
or it was contaminated (and the corrected re-run independently reproduces
-0.42). The strategy is rejected under both. The anomaly matters for confidence
in the *audit's* blast-radius claim — how many other reports were actually
affected by §2.0 — not for this candidate, and it is carried forward as an open
item in the audit document rather than resolved here.

