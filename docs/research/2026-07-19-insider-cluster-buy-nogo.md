# Insider Cluster-Buying (SEC Form 4): NO-GO — Standalone and as a Blend Sleeve

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-07-19
**Audience:** Principal Engineering Team & Quantitative Research Collaborators

## 1. Executive Summary & Core Engine Audit

This report tests `src/ggTrader/lab/strategies/insider_cluster.py`
(`InsiderClusterBuyStrategy`): candidate #1 from `WEB_RESEARCH_CANDIDATES.md`
— the highest-effort item in the backlog. Mechanism: flag stocks where 3+
distinct insiders make open-market purchases (SEC Form 4 transaction code
"P") within a compressed ~2-week window, excluding scheduled 10b5-1 plan
trades; go long the resulting basket, holding on a sweep of `hold_days`
(126/189/252 ≈ 6/9/12 months).

**Data infrastructure built for this candidate:** a full SEC EDGAR Form 4
pipeline (`src/ggTrader/lab/form4_data.py`) — ticker→CIK resolution (SEC's
free bulk `company_tickers.json`), per-issuer filing enumeration
(`data.sec.gov` submissions API, following pagination into older
filing-history pages), and raw ownership-document XML parsing
(`defusedxml`, per security guidance for externally-sourced XML). Feasibility
was verified against real SEC endpoints before committing to the build.
Backfilled the full SP500 (ever-member) universe since 2015: **833,158
transaction rows across 764 symbols**, essentially zero fetch errors across
a realistically-scoped, resumable, rate-limited (~6 req/sec, respecting
SEC's guidance) ~24-hour run.

**This report follows the same discipline PEAD's closure established**: a
promising-looking long-window standalone result was re-verified on the
deployed blend's exact matched eval window before being trusted, and a
real blend test was run rather than stopping at a correlation number.
Both steps overturned the initial impression.

Initial long-window (2015-2026, 40 folds) SP500 test: OOS Sharpe **0.76**,
essentially tied with SPY's 0.77, but with a notably shallower drawdown
(-21.0% vs SPY's -33.7%) and the **lowest OOS-return correlation to the
deployed core of any candidate tested this session (0.382)** — lower than
`idio_vol`'s 0.447 and `pead`'s 0.422, both of which still failed their
own blend tests. That combination (near-tie Sharpe, much lower drawdown,
lowest correlation seen) was distinctive enough to warrant the same
decisive follow-up PEAD got, rather than a correlation-only report.

**Result: NO-GO, both standalone and as a blend sleeve.** On the matched
window the deployed blend was actually validated on (2021-2026), the
standalone edge weakens substantially (Sharpe 0.39 vs SPY 0.58 — no longer
a near-tie, a real underperformance), and adding it as a 4th blend sleeve
produces an essentially flat-to-marginally-worse result (Sharpe
1.14→1.12, MaxDD -5.39%→-5.45%) — not the "doesn't hurt" bar, the "must
actually improve" bar this project has consistently applied to
diversification-sleeve candidates (see `idio_vol`'s July 8 closure).

## 2. Quantitative Performance Context

| System Configuration | Window | OOS Sharpe | CAGR | MaxDD | Gate Pass | WFE |
|---|---|---|---|---|---|---|
| **insider_cluster_buy@sp500 (initial, long window)** | 2015-06–2026-07 (40 folds) | 0.76 | 13.7% | -21.0% | 29/40 (73%) | n/a |
| **insider_cluster_buy@sp500 (matched window)** | 2021-01–2026-04 (17 folds) | **0.39** | 7.6% | -24.9% | n/a | 0.82 |
| SPY (matched window) | 2021-01–2026-04 | 0.58 | 13.0% | -22.1% | N/A | N/A |

| Blend Configuration (matched window, `--max-leverage 1.0`) | Sharpe | CAGR | MaxDD |
|---|---|---|---|
| **3-sleeve baseline** (ensemble @ sp500+midcap400+nasdaq100) — deployed | **1.14** | 9.92% | **-5.39%** |
| **4-sleeve** (+ insider_cluster_buy@sp500) | 1.12 | 9.83% | -5.45% |

**Diversification check** (initial long-window run,
`scripts/insider_cluster_correlation_check.py`, same methodology as
`idio_vol`/`max_effect`/`pead`'s checks): OOS return correlation to the
deployed 5-voter `EnsembleSignal` core = **0.382**, beta = 0.478 — the
lowest correlation of any candidate tested this session. As with PEAD,
this number is not wrong, but characterizes a return stream whose apparent
standalone quality doesn't fully survive to the matched window — low
correlation to the core matters less when the candidate's own edge is
weaker than it first appeared.

Gate pass rate on the long-window run was 29/40 (73%), reasonably
healthy — this is not an overfitting-signature rejection like
`pairs_stat_arb`/`short_interest`/`index_deletion_fade`; it's a genuine
"the edge is real but too small/inconsistent to add value" result, closer
in character to `idio_vol` and `pead`.

## 3. Actionable Research Directions

None ranked — this is a closure report.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**A. Insider cluster-buying, standalone, SP500 — REJECTED (matched
window).** OOS Sharpe 0.39 vs SPY 0.58 on the deployed blend's own
validation window — a real underperformance, not the near-tie the
long-window test suggested. Confirms (a third time this session, after
PEAD) that a long, favorable eval window can hide edge concentration in a
period that doesn't overlap with the window that actually matters for a
deployment decision.

**B. Insider cluster-buying as a 4th blend sleeve
(`insider_cluster_buy@sp500` alongside the deployed
`ensemble@sp500,midcap400,nasdaq100`) — REJECTED.** Sharpe 1.14→1.12,
MaxDD -5.39%→-5.45% — flat to marginally worse, not an improvement. Third
diversification-sleeve candidate rejected this way this session
(`idio_vol` July 8, `pead` July 17, now this) — the pattern holds across
all three: moderate-to-low correlation to the deployed core (0.382-0.692
across the three) has not been sufficient on its own to produce a blend
improvement in any case tested so far.

## 5. Operational Roadmap: Recommended First Action

**Do not deploy `insider_cluster_buy` in any configuration.** No
live-config changes.

**Infrastructure kept regardless of this NO-GO:** `form4_data.py` (SEC
EDGAR Form 4 fetch/parse/cache) and the 833,158-row backfill are real,
reusable assets independent of this specific strategy's outcome — any
future insider-transaction research (e.g. the parked candidate #6, Form
144 non-execution, or a differently-constructed cluster-buy variant) can
build on this pipeline without repeating the ~24-hour backfill.

Move to the next lowest-effort remaining candidate per `next_steps.md`'s
effort-ordered backlog: **#9 Congressional STOCK Act trades**, **#8
retail-attention factors**, **#5 stealthy shorts** (same scraping-tier
group as this one), then **#4 crypto funding-rate carry** and **#14
options IV skew** last.

## 6. Contrarian Evaluation & Parked Research

**Contrarian question:** the mechanism's own academic support (Cohen,
Malloy & Pomorski 2012) specifically found the alpha concentrated in
*opportunistic* (irregularly-timed) trades, not routine/scheduled ones —
this implementation excludes 10b5-1 plans but doesn't otherwise
distinguish opportunistic from routine-but-not-plan-scheduled purchases
(e.g. a CFO who happens to buy every January regardless of a formal
10b5-1 plan). Would a stricter opportunistic-trade filter recover the
edge that a broader "any non-plan purchase cluster" definition dilutes?

**Resolution:** Plausible, but not queued speculatively. The gate pass
rate (73%) and moderate WFE (0.82 on the matched window) don't show the
overfitting signature that would suggest "wrong definition, not absent
edge" the way, say, `pairs_stat_arb`'s cadence question did — this looks
more like a real-but-small effect correctly measured, similar to
`idio_vol`'s and `pead`'s shape. Building an opportunistic-vs-routine
classifier (which itself requires nontrivial engineering — inferring
routineness from an individual insider's multi-year trading pattern, not
available from a single filing) is a real project, not a quick refinement.
Worth a look only if a future decision specifically re-opens the
insider-signal line with that scoped ask; not worth building on
speculation now.
