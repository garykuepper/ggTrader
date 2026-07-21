# Treasury Term-Structure Factors, ETF-Approximation (Candidate A5): NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-20
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/treasury_curve.py`
(`TreasuryCurveStrategy`): candidate A5 from `WEB_RESEARCH_CANDIDATES.md`'s
2026-07-19 cross-asset register — explicitly labeled an **ETF
approximation, not a replication** of Filipović, Pelger & Ye's *"Shrinking
the Term Structure"* (NBER WP 32472, 2024, now accepted at the *Review of
Finance*). The paper proposes 4 investable term-structure factors from a
rich Treasury cash-flow cross-section; this implementation uses only 3
duration-bucket ETFs (SHY 1-3yr, IEF 7-10yr, TLT 20yr+) and cannot be
presumed to reproduce the paper's 4th "complexity" factor.

**Mechanism.** A curve steepener/flattener regime signal using real FRED
constant-maturity Treasury yields (`DGS2`, `DGS10` — free, no API key,
reusing `fred_data.py` from candidate A1). When the 10y-2y slope is
running unusually steep vs. its own trailing history (z-score above a
threshold), go all-in long duration (TLT) for rolldown; when unusually
flat/inverted, go all-in short duration (SHY); in between, hold the belly
(IEF). A one-hot regime allocation, kept simple given the explicit
approximation framing.

**Decisive test, per the A1 lesson.** SPY isn't the natural benchmark for
a Treasury-only strategy, and neither is a naive standalone read of the
WFO's own Sharpe — the real question, matching this candidate's actual
claim (dynamic curve positioning beats a static duration policy), is
whether the regime-switching signal beats simply holding one duration
bucket statically. Full honest walk-forward, `treasury_curve` universe,
`eval_start 2013-01-01` (50 folds), sweeping `steep_threshold` ∈
{0.5, 1.0, 1.5} × `flat_threshold` ∈ {-1.5, -1.0, -0.5), **followed by a
decisive comparison against static SHY/IEF/TLT/equal-weight baselines on
the identical OOS window.**

**Result: clean NO-GO.** The dynamic curve-regime strategy underperforms
the best static baseline (100% SHY) on every metric.

## 2. Quantitative Performance Context

**Standalone WFO:**

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | WFE |
|---|---|---|---|---|
| treasury_curve (dynamic) | 0.19 | 0.8% | -8.2% | 0.44 (below 0.50 floor) |
| SPY (reference only — not the real benchmark) | 0.74 | 13.8% | -33.7% | N/A |

Regime halt active — the recommended live combo was selected in only
17/50 folds (34%).

**Decisive test: dynamic vs static duration policy**, same OOS window
(2014-01-01 to 2026-06-30):

| Configuration | Sharpe | CAGR | Max Drawdown |
|---|---|---|---|
| **dynamic (curve regime)** | **0.19** | **0.8%** | **-8.2%** |
| static 100% SHY | **0.90** | 1.5% | **-5.7%** |
| static equal 1/3 each | 0.21 | 1.5% | -29.5% |
| static 100% IEF | 0.27 | 1.8% | -23.9% |
| static 100% TLT | 0.15 | 1.4% | -48.4% |
| SPY (reference only) | 0.74 | 13.8% | -33.7% |

**Static 100% SHY dominates every configuration tested, including the
dynamic strategy, on both Sharpe and drawdown.** The dynamic strategy's
CAGR (0.8%) is lower than SHY's own CAGR (1.5%) despite the regime signal
likely spending significant time in the SHY state itself (2014-2023 was a
broadly rising-rate environment favoring short duration) — the periods
where the signal actively extended duration into IEF/TLT evidently hurt
more than they helped, net.

## 3. Actionable Research Directions

None ranked — this is a closure report.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Treasury curve steepener/flattener regime timing (SHY/IEF/TLT,
FRED 10y-2y slope z-score) — REJECTED.** Dynamic timing underperforms the
best static duration policy (100% SHY) on Sharpe (0.19 vs 0.90), CAGR
(0.8% vs 1.5%), and MaxDD (-8.2% vs -5.7%). Same failure pattern as
candidate A1 (dynamic FX hedge overlay): a plausible-sounding regime-
timing signal subtracts value relative to simply holding the historically
best-performing static allocation, once actually tested against that
baseline rather than against an unrelated equity benchmark.

**Known implementation limitations, for the record:**
- 3-ETF approximation cannot reproduce the paper's 4th "complexity"
  factor by construction — this report closes the *approximation*, not
  the paper's own actual model, which remains untested and would require
  richer instruments (cash Treasuries, STRIPS, or Treasury futures) this
  project doesn't have.
- The one-hot regime allocation (100% into a single bucket at a time) is
  a deliberately simple construction chosen for testability; a continuous
  duration-tilt version was not tried and could behave differently,
  though the static-baseline result suggests the underlying timing
  signal itself, not just the discretization, is the problem.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy `treasury_curve` in any configuration.** No live-config
changes. Move to the next candidate in the register's home-lab priority
order: **A8 (headline/LLM sentiment on small/mid-cap equities)** — needs
credible point-in-time headline data as the first feasibility check
before any build.

**Infrastructure kept regardless of this NO-GO:** the `treasury_curve`
universe (SHY/IEF/TLT) and `fred_data.py`'s DGS2/DGS10 caching remain
reusable for any future rates/duration candidate.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** would a continuous duration tilt (e.g. a smooth
allocation between SHY and TLT proportional to the z-score, rather than
a one-hot switch) recover value the discrete regime-switching
construction loses at each hard transition?

**Resolution:** Plausible but unlikely to reverse the verdict — the
decisive test already shows that simply holding 100% SHY *statically*,
with no timing signal or transitions at all, beats the dynamic strategy
outright. A smoother version of the same underlying signal would still
need to identify *when* extending duration pays off, and the one-hot
version's failure to do so profitably suggests the signal itself (not
its discretization) lacks the necessary predictive content over this
sample. Not ranked as an active next step.
