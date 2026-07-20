# Commodity Medium-Term Trend (Candidate A3): NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-20
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/commodity_trend.py`
(`CommodityTrendStrategy`): candidate A3 from `WEB_RESEARCH_CANDIDATES.md`'s
2026-07-19 cross-asset register — the trend leg split out of the original
combined "commodity carry+trend+basis reversal" idea (see A2/A4 for the
untested carry/basis-reversal legs). Source: Bloomberg Professional
Services, *"Capturing curve, carry and trend premia in commodity
markets"* (Feb 2026) — practitioner-grade commentary for Bloomberg's BERY
index, not independent academic research.

**Mechanism.** Cross-sectional 12-1 month momentum (this lab's standard
`lookback=252, skip=21` convention) across 14 liquid single-commodity
ETFs — metals (GLD, SLV, CPER, PALL, PPLT, DBB), energy (USO, UNG, UGA),
agriculture (DBA, CORN, WEAT, SOYB, CANE). Broad multi-commodity baskets
(DBC, GSG) deliberately excluded from the ranking universe — including
them would dilute the cross-sectional signal, not add a distinct
commodity. Equal-weight the top-N by trailing momentum, monthly rebalance
(reuses this lab's established `xs_momentum` pattern). Layered with a
volatility-regime filter: when the commodity universe's average trailing
realized volatility is itself running unusually hot relative to its own
recent history (z-score above a threshold), skip the rebalance entirely
(go to cash) rather than chase momentum into a volatility spike — a
distinct, testable mechanism from the equity VIX-regime-throttling idea
already rejected elsewhere in this lab, applied to a different asset
class on its own merits.

Two currently-active alternative single-commodity ETFs (BAL — cotton,
JJC — copper futures) were checked and found delisted 2023; excluded from
the universe.

**Result: clean NO-GO.** Full honest walk-forward, `commodity_trend`
universe, `eval_start 2013-01-01` (50 folds), sweeping `top_n` ∈ {3, 5, 7}
× `vol_z_threshold` ∈ {1.5, 2.0, 3.0}.

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass |
|---|---|---|---|---|
| **commodity_trend (this report)** | 0.13 | 1.0% | -37.0% | 27/50 (54%) |
| SPY benchmark | 0.74 | 13.8% | -33.7% | N/A |

Near-zero risk-adjusted return, and **worse drawdown than the equity
benchmark** despite this being a commodity strategy specifically framed
around a volatility-regime filter meant to avoid crash exposure — the
filter did not prevent a deeper drawdown than simply holding SPY.
Aggregate WFE could not be computed (`nan`) — several individual folds
show `n/a` WFE (train-period reference score of zero or near-zero,
producing an undefined ratio), consistent with a strategy that spends
many folds either in cash (regime filter tripped) or holding a thin,
frequently-changing top-3 basket with unstable train-period performance.
**Regime halt active** — the recommended live combo (`top_n3_vol_z_
threshold1.5`) was selected in 41/50 folds (82%, nominally high
stability) but the regime-halt mechanism still overrode it in favor of
the anchor combo, consistent with the underlying instability visible in
the fold-by-fold OOS Sharpe swinging from -3.57 to +4.69 fold to fold.

## 3. Actionable Research Directions

None ranked — this is a closure report.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Commodity medium-term trend (12-1 cross-sectional momentum + vol-
regime filter, 14-ETF single-commodity universe) — REJECTED.** OOS Sharpe
0.13 vs SPY 0.74, MaxDD -37.0% (worse than SPY's -33.7%), gate pass 54%,
aggregate WFE undefined, regime halt active despite a nominally high
82%-fold stability rate for the recommended combo — the underlying
per-fold Sharpe is too volatile fold-to-fold for the regime-halt
mechanism to trust it. The volatility-regime filter (this candidate's
one genuinely novel mechanism relative to plain cross-sectional momentum)
did not prevent — and did not measurably improve on — a drawdown deeper
than the equity benchmark's.

**Known implementation limitations, for the record:**
- 14-ETF universe is a retail approximation of the practitioner
  commentary's broader BERY-index construction (which likely spans a
  wider futures-based commodity set); ETF-only sourcing was a deliberate
  scope choice per this project's home-lab/ETF-workflow default, not an
  oversight.
- The volatility-regime filter is a market-wide (average-across-universe)
  realized-vol z-score, not a per-commodity filter — a genuinely
  different construction than the equity VIX-regime-throttling idea
  already rejected, but conceptually adjacent; worth noting if commodity
  trend is ever revisited, since a blunt regime gate on entries has now
  failed in two different asset classes.
- The carry (A2) and short-term basis-reversal (A4) legs from the same
  original combined idea remain untested — this closure applies only to
  the trend leg in isolation, per the register's own explicit warning not
  to assume the three sub-signals combine as originally proposed without
  testing each separately.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy `commodity_trend` in any configuration.** No live-config
changes. Move to the next candidate in the register's home-lab priority
order: **A5 (Treasury term-structure factors, static/ETF-approximation
version only)** — free data (TLT/IEF/SHY), must be explicitly labeled an
approximation per the register's own caveat (the underlying paper
specifies 4 factors, a 3-ETF version only captures 3).

**Infrastructure kept regardless of this NO-GO:** the `commodity_trend`
universe (14 single-commodity ETFs) and the strategy's cross-sectional-
momentum + vol-regime-filter code remain reusable for any future
commodity-asset-class candidate.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** would testing the carry (A2) or short-term
basis-reversal (A4) legs separately — or the three combined per the
original pre-split idea — recover a real edge this isolated trend-only
cut misses, given the register's own framing treats trend as one of three
economically distinct, low-correlated drivers rather than the whole
story?

**Resolution:** Plausible and explicitly flagged as an open question by
the register itself (not resolved by this report). A2 (commodity carry)
is ETF-approximable with the same universe already built here — a
natural, low-incremental-cost next test if commodity exposure is
revisited. A4 (short-term basis reversal) is **not** ETF-approximable per
the register's own data-requirements note (needs actual adjacent-contract
futures data), so it remains out of scope for this project's current
retail/ETF-only data access regardless of this trend result. Not ranked
as an immediate next step — A5 is next per the effort-ordered queue — but
worth returning to if the queue is revisited.
