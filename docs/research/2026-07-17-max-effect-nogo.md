# MAX Effect / Lottery-Demand Anomaly: NO-GO (Standalone and as a Diversifier)

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-17
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/max_effect.py`
(`MaxEffectStrategy`): candidate #11 from `WEB_RESEARCH_CANDIDATES.md`'s
first web-research batch. Mechanism: compute MAX (Bali, Cakici & Whitelaw
2011's definition — the single highest daily return over a trailing window)
per stock, equal-weight the lowest-MAX quintile of the eligible SP500
universe (avoiding the "lottery" stocks the literature finds underperform),
monthly rebalance. A long-only defensive/behavioral-tilt sleeve, structured
identically to the existing `idio_vol` strategy (same quintile-bucket,
weight-based pattern) since both are portfolio-construction filters rather
than directional-timing signals. Swept `window` (21/42/63 trading days) and
`quintile` (4/5) — 6 combos, full honest walk-forward, SP500 universe,
`eval_start 2015-01-01` through present (42 folds).

This was the lowest-effort candidate in the current backlog — no new data
source, pure computation on OHLCV already in TimescaleDB — picked
specifically to clear cheap candidates before the higher-effort ones
(Form 4 scraping, analyst-revision feeds, etc.). It closes clean and fast:
**NO-GO both as a standalone strategy and as a diversification sleeve.**
Unlike the same-day `pairs_stat_arb` rejection, this is not an overfitting
verdict — the walk-forward health metrics here are good (see §2) — it is a
genuine "real effect, but not economically useful in this context" result:
the signal behaves as the literature describes, it just isn't strong enough
to beat SPY outright, nor uncorrelated enough with the deployed core to add
diversification value.

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass Rate | Sizing |
|---|---|---|---|---|---|
| **max_effect (this report)** | 0.39–0.45 | 5.0–5.1% | -36.5% | 31/42 folds | Equal-weight, bottom MAX quintile |
| **SPY benchmark** | 0.76–0.88 | 15.2% | -33.7% | N/A | Buy-and-hold |

(Two runs, same code/window, produced slightly different Sharpe/WFE —
0.39/WFE 0.84 vs. 0.45/WFE 0.97 — both computed minutes apart via
`eval_end=str(pd.Timestamp.now().date())`; this is the same eval-window-
drift sensitivity already documented for other levers in `roadmap.md`'s
July 13 entry, not a bug. Both runs agree qualitatively: Sharpe well below
SPY, WFE comfortably clears the 0.50 gate floor.)

Aggregate WFE **0.84–0.97** (well above the 0.50 target — genuinely *not*
an overfitting signature, unlike `pairs_stat_arb`'s -0.16 the same day).
Gate pass rate **31/42 folds (74%)** — healthy. Live-recommended winner
(`quintile4_window21` or `quintile4_window42` depending on run) stable in
**6/42 folds** — better than `pairs_stat_arb`'s 1/42, though still not
strongly convergent.

**Diversification check** (ad hoc, `scripts/max_effect_correlation_check.py`,
same honest-WFO methodology used for `idio_vol`'s 2026-07-07 correlation
study): OOS return correlation to the deployed 5-voter `EnsembleSignal`
core = **0.692**, beta = **0.638**, 2637 overlapping trading days. This is
meaningfully *higher* than `idio_vol`'s 0.447 core-correlation — and that
lower-correlation candidate still failed to add value when tested as a 4th
blend sleeve (July 8, Sharpe 1.03→1.01 with `idio_vol` added). A candidate
with *higher* core-correlation than an already-rejected diversifier has no
basis for a different outcome.

*Caveat on this check's own ensemble number:* the correlation script's
ensemble WFO ran on the same 2015–2026 window as `max_effect` (to keep the
two streams comparable) and reported Sharpe 0.35 for the core — this is
**not** a new finding about the deployed core's quality, it is the same
eval-window-drift artifact documented in `roadmap.md`'s July 13 entry (the
validated "1.12" headline uses a different, narrower window). Only the
*correlation* figure is the actual result of this check.

## 3. Actionable Research Directions

None ranked — this is a closure report.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. MAX effect / lottery-demand quintile filter (standalone, long-only,
SP500) — REJECTED.** OOS Sharpe 0.39–0.45 vs SPY 0.76–0.88, CAGR ~5% vs
15.2%, MaxDD -36.5% (worse than SPY's own -33.7%). WFE 0.84–0.97 (healthy,
not overfit) and gate pass rate 31/42 (74%, healthy) — the mechanism works
as described in the literature, it simply isn't a strong enough standalone
edge to compete with buy-and-hold SPY.

**B. MAX effect as a diversification sleeve — REJECTED.** OOS return
correlation to the deployed 5-voter ensemble core = 0.692 (beta 0.638) —
higher than `idio_vol`'s 0.447, which itself wasn't enough to improve the
3-sleeve blend when tested (July 8: Sharpe 1.03→1.01, worse MaxDD, adding
`idio_vol`). No basis to expect a higher-correlation candidate would do
better; not queued for a blend test.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy or pursue `max_effect` further, standalone or blended.**
Move to the next lowest-effort candidate in `WEB_RESEARCH_CANDIDATES.md`'s
first wave (#2 analyst estimate-revision momentum, #7 Anomaly-Driven
Demand, or the free-data cut of #3 short-interest/cost-to-borrow) or the
remaining internal §6 candidates (PEAD, options-derived signal, revisit
crypto-carry) — this closes the cheapest item in the backlog, as intended,
before ramping into the higher-effort ones (new data-source integration,
scraping).

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** the literature (`MAXβ`, Gorman et al.) suggests
part of the original MAX effect is really an equity-issuance or
overreaction/reversal artifact rather than pure lottery demand — would a
beta-purged or shorter-horizon (weekly rather than monthly) version behave
differently, given this test used the raw monthly-rebalance construction?

**Resolution:** Not worth pursuing. Unlike `pairs_stat_arb`'s cadence
question (where the rejection showed overfitting symptoms consistent with
a mistimed signal), this result showed *healthy* WFO diagnostics (WFE
0.84–0.97, 74% gate pass) — the signal is being measured accurately at
this cadence, it's just economically weak and too correlated with the
existing core to matter. A cadence or beta-purge refinement is optimizing
a mechanism that already isn't providing value, not fixing a measurement
artifact. Closing this line of inquiry; not parking a refinement.
