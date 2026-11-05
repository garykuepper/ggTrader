# Short-Volume-Ratio ("Stealthy Shorts" Free-Data Cut): NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-19
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/short_volume_ratio.py`
(`ShortVolumeRatioStrategy`): the free-data-only cut of candidate #5
("stealthy shorts") from `WEB_RESEARCH_CANDIDATES.md`. Mechanism:
market-neutral — long the lowest-quintile / short the highest-quintile of
the eligible SP500 universe by trailing average short-volume ratio
(FINRA daily short volume / total volume). The second market-neutral
construction in this lab after `pairs_stat_arb`.

**Data infrastructure built:** `src/ggTrader/lab/short_volume_data.py` —
FINRA's public daily short-sale-volume CDN
(`cdn.finra.org/equity/regsho/daily/`), a different dataset than
`short_interest_data.py`'s bi-monthly consolidated short *interest*.
Verified via bisection that the CDN only hosts files from **2018-08-01**
onward (a clean retention boundary, confirmed on multiple weekday
spot-checks). Backfilled 2018-08-01 through present: **1,189,652 rows**
across 676 SP500 symbols and 2,078 business days, 78 errors (consistent
with expected market holidays, ~70-80 over the window).

**Explicit, known fidelity gap:** the source paper (Goyal, Reed,
Smajlbegovic & Soebhag, JFE 2025) decomposes short volume into
liquidity-*demanding* vs. liquidity-*supplying* components using
transaction-level exchange data (Cboe/Nasdaq/NYSE) this project doesn't
have. This implementation uses only the coarser proxy the original
candidate write-up itself flagged as the retail-feasible fallback:
trailing average short-volume ratio alone, without any demand/supply
classification. This is not a claim of testing the paper's actual finding
— it's the acknowledged-coarser version.

**Result: clean NO-GO.** Full honest walk-forward, SP500 universe,
`eval_start 2018-09-01` (27 folds), swept `lookback_days`
(10/20/40) × `quintile` (4/5).

## 2. Quantitative Performance Context

| System Configuration | OOS Sharpe | CAGR | Max Drawdown | Gate Pass | Regime Halt |
|---|---|---|---|---|---|
| **short_volume_ratio (this report)** | 0.21 | 1.1% | -9.2% | 16/27 (59%) | 20/27 (74%) |
| SPY benchmark | 0.72 | 17.0% | -33.7% | N/A | N/A |

Aggregate WFE sits exactly at the 0.50 floor — not a strongly overfit
result, but not a healthy margin above it either. Stability was actually
the highest seen among this session's market-neutral strategies (winner
selected in 15/27 folds = 56%), but that stability is around a combo that
simply isn't generating a real edge, not evidence of one.

The characteristic signature here is a market-neutral book with a
**shallow drawdown (-9.2%) and a near-zero return (CAGR 1.1%, Sharpe
0.21)** — consistent with realized volatility low enough that even small
noise dominates the Sharpe ratio. This reads as "the long/short
construction is mechanically correct but isn't capturing a differentiating
signal," not "the signal exists but got overfit away" (the `pairs_stat_arb`
pattern) or "the signal only worked in an earlier regime" (the PEAD/
insider-cluster/congress-trades pattern). No diversification/blend
follow-up was run — that step is reserved for candidates showing genuine
standalone promise, and this didn't clear that bar on the long window to
begin with (unlike the three candidates that required the matched-window
check).

## 3. Actionable Research Directions

None ranked — this is a closure report.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Short-volume-ratio market-neutral sleeve (free-data-only proxy for
"stealthy shorts"), SP500 — REJECTED.** OOS Sharpe 0.21 vs SPY 0.72, CAGR
1.1% vs 17.0%, gate pass 16/27 (59%), regime halt active 20/27 folds
(74%, persistently active at the end of the backtest). The plain
short-volume-ratio proxy, without the paper's liquidity-demand/supply
decomposition, does not carry a usable market-neutral edge in this
implementation.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy `short_volume_ratio` in any configuration.** No
live-config changes.

**Infrastructure kept regardless of this NO-GO:** `short_volume_data.py`
and the 1,189,652-row backfill (2018-2026, daily granularity) remain
reusable — this is a genuinely different, higher-frequency FINRA dataset
than `short_interest_data.py`'s bi-monthly one, and could support future
research needing daily short-sale-volume context (e.g., a volume-spike
overlay on another signal) independent of this specific rejection.

Move to the next remaining candidate per `next_steps.md`'s effort-ordered
backlog: **#4 crypto funding-rate carry** and **#14 options IV skew**
(different asset class/execution model, or paid-data risk) — the last two
in the active queue. **#8 retail-attention factors** remains paused
(Google Trends rate-limit lockout) and can be revisited once that clears.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** would the paper's actual liquidity-demand/supply
decomposition (not just raw short-volume ratio) recover a real edge that
this coarser proxy dilutes — i.e., is the near-zero result a genuine
absence of signal, or a fidelity artifact of the free-data simplification?

**Resolution:** Plausible, but the fidelity gap is fundamental, not an
engineering shortcut that could be closed with more effort on free data.
The paper's classification requires order-level or at minimum
quote-level tick data (to determine whether a given short sale executed at
the bid, mid, or ask, and relative to prevailing liquidity) — genuinely
unavailable without a paid, institutional-grade data feed (the same class
of blocker as candidate #7's WRDS requirement, though here at least the
underlying short-volume aggregate is free). Not worth pursuing further
without a specific decision to acquire that data; closing this line
rather than parking a refinement that depends on data this project
doesn't have access to.
