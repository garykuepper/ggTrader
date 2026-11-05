# Net-issuance screen: Part A NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-10-23 (Pacific time)
**Audience:** Owner and quantitative research collaborators

## 1. Executive Summary & Core Engine Audit

**NO-GO. Close the net-issuance screen.** The sole registered hypothesis, French
`BIG NegNI` (large-stock net repurchasers), fails both Part A gates. Its full-window
base-cost Newey-West statistic is **0.26** with one-sided **p = 0.398**
([results](_net_issuance_screen_results.json), `H1.A1.t`, `H1.A1.p`), against
registered bars of **t ≥ 3.0** and **p ≤ 0.00135** (`inputs.A1_t_bar`,
`inputs.A1_p_bar`). Its since-1994 net CAGR advantage is **+0.48 pt/yr**
(`H1.A2.cagr_diff_pp`), below **+1.0 pt/yr** (`inputs.A2_bar_pp`). Both gate flags
and the overall flag are false (`H1.A1.pass`, `H1.A2.pass`, `H1.overall_pass`).
There is one confirmatory hypothesis, so no within-screen Holm adjustment
([registered brief §4](briefs/2026-10-23-net-issuance-screen.md)).

The useful diagnostic runs in the opposite direction: heavy issuers, `BIG HiNI`,
lag the French market at the base haircut, with full-window **t = −3.20** and
since-1994 **−5.26 pt/yr** (`issuance_shape["BIG HiNI"].base.full_t`,
`issuance_shape["BIG HiNI"].base.since_1994_cagr_diff_pp`). This is evidence
about the weak **short side** of an issuance spread. A long-only taxable stock
account cannot earn that short-side return by buying `BIG NegNI`. The favorable
diagnostic `ME5 NI3` also cannot replace H1 after returns were seen.

This is an idea-level screen on survivorship-free French value-weighted
portfolios against French market total return, not an out-of-sample test of
ggTrader's exact stock rules or SPY. No candidate advances to B0/B1. Momentum
and the four classic factor hypotheses remain closed under their own reports;
the separate production paper observation and freeze remain governed by their
existing decisions.

## 2. Quantitative Performance Context

All measured figures below come from the [results JSON](_net_issuance_screen_results.json).
The registered [brief §§3–7](briefs/2026-10-23-net-issuance-screen.md) defines the
series, gates, and diagnostics. The full window is **1963-07–2026-04, 754 months**
(`inputs.full_window.start`, `.end`, `.n_months`); A2 is
**1994-01–2026-04, 388 months** (`inputs.A2_window.start`, `.end`, `.n_months`);
the post-publication diagnostic is **2009-01–2026-04, 208 months**
(`inputs.post_2008_window.start`, `.end`, `.n_months`). Bartlett Newey-West uses
**12 monthly lags** (`inputs.nw_lags`). The base and stress annual haircuts are
**1.0%** and **2.0%** (`inputs.base_cost`, `inputs.stress_cost`), subtracted monthly.

### Registered gate table

| H1: `BIG NegNI` | Measured | Registered bar | Pass |
|---|---:|---:|---|
| A1 full-window NW t | 0.26 (`H1.A1.t`) | ≥ 3.0 (`inputs.A1_t_bar`) | false (`H1.A1.pass`) |
| A1 one-sided p | 0.398 (`H1.A1.p`) | ≤ 0.00135 (`inputs.A1_p_bar`) | false (`H1.A1.pass`) |
| A2 since-1994 net CAGR difference | +0.48 pt/yr (`H1.A2.cagr_diff_pp`) | ≥ +1.0 pt/yr (`inputs.A2_bar_pp`) | false (`H1.A2.pass`) |
| Part A | — | A1 and A2 | **NO-GO** (`H1.overall_pass`: false) |

### H1 stability and risk diagnostics

The following subperiod values use the base haircut. Each row's cited JSON
prefix is `H1.diagnostics.subperiods[i]`; columns map to its `start`, `end`,
`n_months`, `t`, `p`, and `cagr_diff_pp` keys respectively.

| JSON row | Window | Months | NW t | One-sided p | Net CAGR difference |
|---|---|---:|---:|---:|---:|
| `H1.diagnostics.subperiods[0]` | 1963-07–1993-12 | 366 | 0.12 | 0.452 | +0.23 pt/yr |
| `H1.diagnostics.subperiods[1]` | 1994-01–2008-12 | 180 | 0.90 | 0.185 | +2.02 pt/yr |
| `H1.diagnostics.subperiods[2]` | 2009-01–2026-04 | 208 | −1.24 | 0.892 | −0.93 pt/yr |

The separate post-2008 net CAGR diagnostic is **−0.93 pt/yr**
(`H1.diagnostics.post_2008_cagr_diff_pp`). At the stress haircut, full-window
**t = −1.10**, **p = 0.865**, and since-1994 net advantage **−0.62 pt/yr**
(`H1.diagnostics.stress.full_t`, `.full_p`, `.since_1994_cagr_diff_pp`).

Gross CAPM arithmetic alpha is **+2.02% annualized** with market beta **0.88**
(`H1.diagnostics.capm.alpha_ann_pct`, `.beta`); no alpha significance test is
claimed. The worst gross rolling relative 12 months are **−15.52%** from
**1999-03–2000-02** (`H1.diagnostics.worst_12m.relative_pct`, `.start`, `.end`).
Base-cost relative wealth falls **−30.47%** from its **2002-07** peak to its
**2021-01** trough and has no recorded recovery (`H1.diagnostics.relative_drawdown.max_drawdown_pct`,
`.peak`, `.trough`, `.recovery`: null, `.recovery_months`: null).

### Issuance-shape and univariate diagnostics

These cells are **diagnostics only**. For each row, the JSON prefix shown in the
first column combines with these column keys: `base.full_t`, `base.full_p`,
`base.since_1994_cagr_diff_pp`, `base.post_2008_cagr_diff_pp`, and the same four
under `stress`. The one-sided p values test **positive** excess, so a very high
p beside a negative t does not erase evidence of underperformance. CAGR values
are percentage points per year relative to the French market.

| JSON prefix / portfolio | Base t / p | Base 1994 / 2009 Δpp | Stress t / p | Stress 1994 / 2009 Δpp |
|---|---:|---:|---:|---:|
| `issuance_shape["BIG NegNI"]` | 0.26 / 0.398 | +0.48 / −0.93 | −1.10 / 0.865 | −0.62 / −2.05 |
| `issuance_shape["BIG ZeroNI"]` | −0.26 / 0.603 | +0.51 / −0.91 | −0.99 / 0.839 | −0.59 / −2.03 |
| `issuance_shape["BIG LoNI"]` | −2.03 / 0.979 | −1.65 / −3.02 | −3.14 / 0.999 | −2.73 / −4.13 |
| `issuance_shape["ME5 NI2"]` | −1.45 / 0.927 | −1.14 / −1.79 | −2.68 / 0.996 | −2.22 / −2.91 |
| `issuance_shape["ME5 NI3"]` | +1.01 / 0.157 | +2.77 / +3.90 | +0.13 / 0.450 | +1.65 / +2.74 |
| `issuance_shape["ME5 NI4"]` | −0.87 / 0.808 | −1.02 / +1.36 | −1.72 / 0.957 | −2.11 / +0.22 |
| `issuance_shape["BIG HiNI"]` | −3.20 / 0.999 | −5.26 / −1.38 | −3.94 / 0.99996 | −6.31 / −2.50 |
| `univariate_repurchasers` (`< 0 `) | +1.52 / 0.0645 | +0.67 / −0.80 | −0.04 / 0.514 | −0.44 / −1.92 |

The univariate `< 0 ` portfolio includes small caps
(`univariate_repurchasers.includes_small_caps`: true). French separates
`NegNI` and `ZeroNI` from the five issuer quintiles; `BIG LoNI` is still an
issuer bucket, not the registered repurchaser leg
([brief §11](briefs/2026-10-23-net-issuance-screen.md)). The shape is not
monotone: `ME5 NI3` has positive diagnostic returns, while `BIG HiNI` has a
large deficit. Neither observation licenses a new confirmatory test here.

### Limits on inference

French `BIG` is the largest NYSE market-equity quintile of the source sort,
**not** an S&P 500 constituent portfolio or ggTrader's point-in-time stock
implementation. The benchmark is French market total return, **not SPY**.
The source's annual June rebalance and the fixed policy haircuts do not measure
actual turnover, fills, investor taxes, or after-tax performance. The market
benchmark is gross of those frictions. The anomaly definition was developed
using historical returns: the full-window gate and much of the since-1994
window are partly in-sample. The post-publication weakness is diagnostic, not
an independent rescue or a substitute gate. Serial selection of this candidate
after earlier screens is not covered by the single-test error rate.

## 3. Actionable Research Directions

**No net-issuance candidate advances.** H1 failed both gates, so the approved
[forward-paper B0/B1 standard](briefs/2026-10-23-forward-paper-standard.md)
has no candidate to evaluate from this screen. The `BIG HiNI` result suggests
the academic issuance spread is driven by heavy issuers lagging; it does not
establish a usable long-only market-beating leg. Keep the existing paper
observation and production freeze under their separate rules.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**Net repurchasers (`BIG NegNI`) — CLOSED NO-GO.** Registered A1 **t = 0.26**
(`H1.A1.t`) and A2 **+0.48 pt/yr** (`H1.A2.cagr_diff_pp`) both miss their
registered thresholds. Do not promote a favorable NI diagnostic or change the
window after seeing these results.

**Momentum and four classic factors — CLOSED NO-GO.** Their separate registered
screens found no qualifying long-only candidate: [momentum](2026-10-20-momentum-french-screen-nogo.md)
and [factors](2026-10-21-factor-screen-nogo.md). Their prior decisions are not
reopened by this NI diagnostic.

## 5. Operational Roadmap: Recommended First Action

**Record the NO-GO in the research snapshot, worklist, and changelog.** The
registered [brief §8](briefs/2026-10-23-net-issuance-screen.md) requires closure
when either gate fails; both failed here. This documentation does not authorize
a return rerun, a production change, a broker order, or real capital. Paper
observation and the current production freeze continue under their own rules.

## 6. Contrarian Evaluation & Parked Research

The strongest counterargument is that the heavy-issuer short side has a large
negative return statistic (`issuance_shape["BIG HiNI"].base.full_t`) and that
another long-only NI bucket, `ME5 NI3`, has a positive since-1994 diagnostic
(`issuance_shape["ME5 NI3"].base.since_1994_cagr_diff_pp`). Neither validates
the registered `BIG NegNI` long leg. Choosing a new bucket after reading the
results would be a new, unregistered hypothesis; short-side returns cannot be
captured by the owner's long-only taxable stock account.

With momentum, four classic factors, and net issuance all closed NO-GO under
the two-part standard, **no well-supported long-only large-cap anomaly remains
testable on free data from the reviewed set**. The next step is an **owner
discussion of research direction** using these closures and the standing
paper-trading reliability work as evidence. This screen supplies no basis to
change the production freeze or paper-trading plan.
