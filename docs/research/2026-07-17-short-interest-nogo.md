# Hard-to-Borrow / Short-Interest Signal (Free-Data-Only Cut): NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-17
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/short_interest.py`
(`ShortInterestStrategy`): the free-data-only cut of candidate #3 from
`WEB_RESEARCH_CANDIDATES.md`'s first web-research batch. Mechanism:
equal-weight the quintile of the eligible SP500 universe with the
*smallest* recent increase in days-to-cover (a proxy for "transitioning
toward hard-to-borrow"), avoiding names whose short-interest is rising
fastest, monthly rebalance. Deliberately excludes the paid real-time
cost-to-borrow/utilization feed (Ortex, IHS Markit) the original candidate
also considered — only FINRA's free consolidated short-interest data.

**New data infrastructure built for this candidate** (this was scoped as a
"one new, simple free data feed" candidate, but the actual engineering
included a real, non-obvious correctness bug worth recording): FINRA
publishes consolidated short interest via a public Query API
(`api.finra.org`, no key required) with settlement-date granularity, but:

- Its own metadata claims "available online for one rolling year" — empirically
  false; real coverage goes back to 2020-04-15.
- **Settlement dates cannot be assumed from a calendar rule.** The naive
  "15th and last calendar day of each month" assumption (matching FINRA's
  nominal bi-monthly cycle) silently missed **71 of 151 dates (47%)** in the
  first backfill attempt, because FINRA shifts the actual settlement date to
  the nearest preceding business day whenever the nominal date falls on a
  weekend/holiday. Fixed by adding `discover_settlement_dates()`, which
  finds real settlement dates from data (one anchor-symbol query) rather
  than guessing them — this is now the correct, general-purpose way any
  future FINRA short-interest work in this repo should enumerate cycles.
- Point-in-time correctness: a settlement's days-to-cover figure is gated
  behind a conservative 15-day publish-lag assumption (`available_as_of`)
  so the strategy can never see a cycle's data before FINRA would actually
  have published it.

Backfilled 150 settlement dates × 631 (ever-member) SP500 symbols,
86,793 rows cached in a new `short_interest` table. Full honest walk-forward,
SP500 universe, `eval_start 2020-07-01` (limited by real data availability,
not the 2015 convention other strategies use) through present, 20 folds,
6 combos (`lookback_cycles` × `quintile`).

**Result: NO-GO**, and unlike `max_effect` (the prior lowest-effort
candidate, which showed a genuine-but-weak effect with healthy WFO
diagnostics), this rejection has the noise/overfitting signature —
consistent with the original candidate report's own caveat that the
underlying academic evidence (Asquith-Pathak-Ritter 2005) is *insignificant*
on a value-weighted (i.e. large-cap-dominated) basis, with the effect
concentrated in small/illiquid names outside this project's SP500 universe.

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Sizing |
|---|---|---|---|---|---|
| **short_interest (this report)** | 0.27 | 3.8% | -27.1% | 15/20 folds | Equal-weight, bottom quintile (safest DTC trend) |
| **SPY benchmark** | 0.61 | 13.0% | -24.5% | N/A | Buy-and-hold |

Aggregate WFE **0.11** (well below the 0.50 target — a weak/overfitting
signature, not a modest-but-real edge). Regime halt active on **16/20 folds
(80%)** — the strategy spent most of its backtest trading defensive anchor
params rather than its own selected signal, the same pattern seen in the
rejected `pairs_stat_arb` and breadth-driven leveraged-rotation arcs. The
WFO's recommended live combo (`lookback_cycles2_quintile5`) was selected as
train winner in only **1 of 20 folds** — noise, not convergence.

No diversification follow-up was run (unlike `max_effect`'s July 17
correlation check) — that check is only informative when the standalone
result shows a genuine-but-weak effect with healthy WFO diagnostics;
here, the low WFE and 80% regime-halt rate already indicate the signal
itself isn't being reliably found, so measuring its correlation to the
deployed core wouldn't add information.

## 3. Actionable Research Directions

None ranked — this is a closure report.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Short-interest / days-to-cover trend filter, free-data-only,
SP500 — REJECTED.** OOS Sharpe 0.27 vs SPY 0.61, CAGR 3.8% vs 13.0%, MaxDD
-27.1% (worse than SPY's own -24.5%). Aggregate WFE 0.11 (below the 0.50
floor), regime halt active 16/20 folds (80%), live-recommended params
stable in only 1/20 folds — a noise/overfitting signature, not a modest
edge. Consistent with the original candidate's own caveat that the
short-interest effect's academic support (Asquith-Pathak-Ritter 2005) is
statistically insignificant on a value-weighted (large-cap) basis; this
project's SP500 universe is exactly that basis.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy or pursue `short_interest` further.** The rejection
mirrors the literature's own large-cap caveat closely enough that a
same-mechanism retry (tighter correlation filters, different quintile
cuts, cost-to-borrow instead of days-to-cover) is unlikely to change the
outcome without also changing the universe to small/illiquid names — a
different project scope, not queued here. Move to the next lowest-effort
candidate per `next_steps.md`'s effort-ordered backlog: **#2 analyst
estimate-revision momentum** (already found infeasible for an honest
historical WFO — free sources are current-snapshot-only, see the July 17
session note) is now also effectively deprioritized; next up is **#12
PEAD**, **#13 index-deletion overshoot fade**, or **#7 Anomaly-Driven
Demand**.

**Infrastructure kept regardless of this NO-GO:** `short_interest_data.py`
(FINRA fetch/cache/point-in-time-gate) and the 86,793-row backfill are
real, reusable assets — `discover_settlement_dates()` in particular is the
correct general-purpose way to enumerate any future FINRA short-interest
work's real settlement cycles, not just this one candidate's.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** the original candidate's academic support was
explicitly split between "short interest as demand proxy" (Asquith-Pathak-
Ritter, weak on a value-weighted basis) and a separate, unconfirmed claim
about *cost-to-borrow* mattering independently of raw short-interest counts
— is the free-data proxy tested here (days-to-cover trend) simply too noisy
a stand-in for the real signal (actual borrow fees), rather than evidence
the mechanism itself is dead?

**Resolution:** Plausible, but not worth pursuing without the paid data.
The literature caveat this report opened with (value-weighted
insignificance) applies to short interest itself, not just its trend --
meaning even a perfect-fidelity short-interest measure would likely still
show a weak large-cap effect. Testing the cost-to-borrow refinement would
require the paid Ortex/IHS Markit feed the original candidate flagged as a
feasibility risk — not worth committing budget to on the strength of one
free-data NO-GO. Closing this line; the free-data-only version has been
fairly tested and rejected.
