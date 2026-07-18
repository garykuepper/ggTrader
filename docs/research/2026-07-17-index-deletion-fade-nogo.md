# S&P 500 Index-Deletion Overshoot Fade: NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-17
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/index_deletion.py`
(`IndexDeletionFadeStrategy`): candidate #13 from
`WEB_RESEARCH_CANDIDATES.md`. Mechanism: on an S&P 500 index-deletion
event, go long the deleted stock and hold for `hold_days`, betting on
mean-reversion of the mechanical, price-insensitive selling pressure
created by index/benchmark-tracking funds forced to sell simultaneously.
Event-driven, not the cross-sectional quintile-bucket pattern used
elsewhere in this session — a genuinely different construction. Needed
**zero new data infrastructure**: deletion events are derived directly
from `ggTrader.data.core.index_constituents.load_sp500_history()` (already
maintained in this repo, 2712 snapshots back to 1996), and the standard
SP500 OHLCV pull already includes every historically-deleted symbol (a
union of all-ever-members). This made it the fastest candidate to build in
the whole session — no backfill step at all.

**Found and fixed a real infrastructure bug along the way.** The first WFO
run crashed (`IndexError: index 0 is out of bounds for axis 0 with size
0`) deep inside vectorbt. Root cause: `simulate_weights` (shared by every
weight-based strategy in this lab) had no handling for the case where
*every* combo in a fold's grid produces zero selected symbols across the
entire fold window — a scenario only an event-driven, sparse-signal
strategy like this one can realistically trigger (the shortest
`hold_days=21` sweep value could plausibly miss all ~20-25 yearly
deletion events within a given fold's monthly rebalance grid).
`vbt.Portfolio.from_orders` on a fully empty size/close/group_by
degenerates to zero groups and crashes rather than returning a sane
"never held anything" (flat cash) result. Fixed in `simulate.py` with a
short-circuit for the all-empty case (falls back to a flat equity curve
without invoking vbt at all) — this protects any future sparse-event
strategy in this lab, not just this one. Two regression tests added
(`test_simulate.py`).

**Result: clean NO-GO**, and not a close call. Full honest walk-forward,
SP500 universe, `eval_start 2015-01-01` (42 folds), swept `hold_days`
(21/42/63).

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Sizing |
|---|---|---|---|---|---|
| **index_deletion_fade (this report)** | 0.30 | 6.0% | **-68.7%** | 17/42 folds | Equal-weight, active deletion events |
| **SPY benchmark** | 0.76 | 15.2% | -33.7% | N/A | Buy-and-hold |

Aggregate WFE could not be computed (`nan`) — a symptom of the same
sparse-selection instability that caused the crash before the
`simulate_weights` fix, not a formatting issue. Gate pass rate **17/42
(40%)** and regime halt active on **32/42 folds (76%)** — the strategy
spent the large majority of its backtest on defensive anchor params, not
its own selected signal, the worst regime-halt rate of any candidate
closed this session. The **-68.7% max drawdown** is the standout finding:
nearly double SPY's own worst drawdown over the same window, and worse
than every other candidate tested this session including the explicitly
leveraged-ETF strategies.

## 3. Actionable Research Directions

None ranked — this is a closure report.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. S&P 500 index-deletion overshoot fade, SP500 — REJECTED.** OOS Sharpe
0.30 vs SPY 0.76, CAGR 6.0% vs 15.2%, MaxDD **-68.7%** vs SPY's own
-33.7%. Gate pass 17/42 (40%), regime halt active 32/42 folds (76%). The
mechanism's own literature caveat (Greenwood & Sammon's "Disappearing
Index Effect" — the price impact has been shrinking over decades) doesn't
fully explain a result this poor; the more likely explanation is that many
real S&P 500 deletions are driven by genuine fundamental deterioration
(the company is failing, not just mechanically unlucky in index-committee
timing), so "buy the deleted stock and wait for reversion" is
systematically buying into falling knives at least as often as it's
buying mechanically-oversold-but-fundamentally-fine names — the -68.7%
drawdown is consistent with a handful of concentrated, severe
single-name blowups (e.g., a deletion driven by bankruptcy risk that
kept falling) dominating the equal-weighted book.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy or pursue `index_deletion_fade` further.** No live-config
changes. The `simulate_weights` fix (handling all-empty-combo folds
gracefully) is a real, standing infrastructure improvement independent of
this rejection — kept regardless.

Move to the next lowest-effort candidate per `next_steps.md`'s
effort-ordered backlog: **#7 Anomaly-Driven Demand** (new feed + real
engineering tier), then the scraping-tier candidates (#1, #9, #8, #5),
then #4/#14 last.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** would restricting to deletions with a clean,
identifiable "index-mechanical" cause (e.g., replaced by a larger/more
liquid peer during a sector reshuffle, not accompanied by a simultaneous
earnings warning, credit downgrade, or bankruptcy filing) filter out the
falling-knife cases and isolate a real, smaller effect?

**Resolution:** Plausible in principle, but a real engineering lift — it
would require cross-referencing each deletion event against a
corporate-actions/news feed to classify the *reason* for deletion, which
is a new data source this project doesn't have (the original candidate's
own data-requirements section only assumed "public reconstitution
announcements + standard OHLCV," not a reason-classification layer). Given
the magnitude of this rejection (-68.7% MaxDD, worst of the session) and
that the underlying academic literature already documents the effect
decaying toward zero even without this distress-filtering, this isn't
worth the added engineering. Closing this line rather than parking it.
