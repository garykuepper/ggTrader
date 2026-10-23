# Century Synthetic Leverage: registered rule keeps 30%, with a ruin flag

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-10-23
**Audience:** Owner and quantitative research collaborators
**Brief:** [approved study](briefs/2026-10-23-century-leverage.md), [A23 reproduction ruling r1](briefs/2026-10-23-century-leverage-ruling-r1.md), [endpoint ruling r2](briefs/2026-10-23-century-leverage-ruling-r2.md)
**Raw evidence:** [_century_leverage_results.json](_century_leverage_results.json), [_century_leverage_calibration.json](_century_leverage_calibration.json), [_century_leverage_a23_gate.json](_century_leverage_a23_gate.json)

## 1. Executive Summary & Core Engine Audit

**Mechanical outcome: keep 30%.** No tested alternative passes every condition of the pre-registered §6 rule (`recommendation_rule.proposal` and `recommendation_rule.passing_alternatives`). This is a result about a **synthetic historical portfolio**, not an out-of-sample validation of an investable 1929 fund and not an instruction to change the owner's current A23 allocation.

The primary 24%/15% federal-tax model gives 30% SSO a **10.71% after-tax CAGR** against **9.47%** for 0%, a **+1.24 percentage point/year** premium, but the 30% path reaches **−91.4%** max drawdown against **−84.6%** unlevered. The 1926–1945 block containing the 1929–32 collapse has those same drawdowns, so **30% itself trips the −90% ruin flag** (`scenarios["0.84%|24/15|wash=False|shiller"]["30"]["full"]` and `["1926-1945"]` versus the same paths for `["0"]`). The smaller 15% mix is the largest tested mix that remains above that line: −87.9%, at **10.18% CAGR**, or **0.54 point/year less** than 30% (same scenario, `["15"]["full"]`).

The fit is reproducible: [calibration driver](../../scripts/century_leverage_calibrate.py) reads the checksummed frozen real SPY/SSO inputs and French RF, aligns **4,995** sessions from **2006-06-22 through 2026-04-30**, and reproduces annual fitted spreads **0.00918382333377512** at a 0.84% fee and **0.008683823333775214** at 0.89% exactly (`calibration.aligned_window`, `calibration.aligned_sessions` and `calibration.fees`). The [X harness](../../scripts/century_leverage_a23_x.py) executes commit `53b739c`'s A23 source with only frozen fetch substitution and reproduces the gate's SHA-256 `645d4ee2dad880e61bffd147cee0985842f01fab1dae3caf4508fd567ddb6be4` (`_century_leverage_a23_gate.json:runs.X.output_sha256`). The [study driver](../../scripts/century_leverage.py) uses the A23 lot book and fixed month-end ±8-point band, with fresh lots and terminal liquidation for each window. No strategy signal or drawdown trade is part of this study.

## 2. Quantitative Performance Context

**Reading the tables.** All monetary figures are terminal wealth from each separately initialized window. Recovery is **longest completed / still-open elapsed**, in trading sessions; an em dash means no open recovery at the window end. Drawdown and recovery use NAV after annual tax payments but before terminal-liquidation tax. For every full and era row below, the exact JSON path is `scenarios["0.84%|{bracket}|wash=False|shiller"][mix][window]`; displayed columns are `after_tax_cagr`, `after_tax_terminal`, `max_drawdown`, `longest_completed_recovery_sessions`, `unrecovered_peak_elapsed_sessions` and `ruin`. The observed start/end dates are in each row's `window` field. The 0.84% fee is the primary case; the 0.89% fee is a sensitivity, with a separately refitted spread. Neither rate is applied as a historical tax schedule.

### Full 1926-07-01 → 2026-06-30 window

| Bracket | SSO mix | After-tax CAGR | Terminal wealth | Max DD | Recovery completed / open | Ruin |
|---|---:|---:|---:|---:|---:|---|
| 24/15 | 0% | 9.47% | $852.32m | -84.6% | 4,776 / 19 | no |
| 24/15 | 15% | 10.18% | $1.62bn | -87.9% | 4,789 / 19 | no |
| 24/15 | 30% | 10.71% | $2.63bn | -91.4% | 4,833 / 19 | yes |
| 24/15 | 45% | 11.17% | $3.99bn | -93.9% | 4,891 / 19 | yes |
| 24/15 | 60% | 11.60% | $5.85bn | -95.7% | 6,080 / 19 | yes |
| 40.8/23.8 | 0% | 9.06% | $584.18m | -84.8% | 4,803 / 19 | no |
| 40.8/23.8 | 15% | 9.83% | $1.18bn | -88.0% | 4,804 / 19 | no |
| 40.8/23.8 | 30% | 10.30% | $1.80bn | -91.5% | 4,845 / 19 | yes |
| 40.8/23.8 | 45% | 10.76% | $2.73bn | -94.0% | 4,945 / 19 | yes |
| 40.8/23.8 | 60% | 11.26% | $4.29bn | -95.7% | 6,095 / 19 | yes |

### Rolling after-tax CAGR versus 0% (primary 24%/15% bracket only)

These are **fresh tax-lot simulations** at matched month-end dates, not differences on one continuing tax book. The 10-year sample has **1,080** windows and the 20-year sample **960** (`rolling_summary["10"]["0"].windows` and `rolling_summary["20"]["0"].windows`). Each cell gives the worst difference, its start/end dates, and share trailing 0%, from `rolling_summary[tenor][mix].{worst_difference_vs_0,worst_window,share_trailing_0}`. Higher-bracket rolling results were not computed in the frozen artifact.

| SSO mix | Worst 10-year difference (dates); share trailing | Worst 20-year difference (dates); share trailing |
|---|---|---|
| 0% | +0.00 pp/year (1926-07-31 → 1936-07-31); 0.0% | +0.00 pp/year (1926-07-31 → 1946-07-31); 0.0% |
| 15% | -1.47 pp/year (1929-08-30 → 1939-08-31); 25.6% | -0.30 pp/year (1928-02-29 → 1948-02-28); 14.6% |
| 30% | -3.09 pp/year (1929-08-30 → 1939-08-31); 21.3% | -0.57 pp/year (1964-07-31 → 1984-07-31); 9.0% |
| 45% | -5.50 pp/year (1929-08-30 → 1939-08-31); 22.9% | -1.24 pp/year (1929-08-30 → 1949-08-31); 13.0% |
| 60% | -7.49 pp/year (1929-08-30 → 1939-08-31); 23.7% | -2.14 pp/year (1929-08-30 → 1949-08-31); 16.1% |

### Every registered subperiod, with fresh lots

Rows use the same scenario path and fields as the full table, with `window` set to the named era. All wealth figures start from the study's equal-capital window initialization. The final era follows r2 and ends in June 2026.

| Era | Bracket | SSO mix | After-tax CAGR | Terminal wealth | Max DD | Recovery completed / open | Ruin |
|---|---|---:|---:|---:|---:|---:|---|
| 1926-1945 | 24/15 | 0% | 5.31% | $274,390 | -84.6% | 4,776 / 16 | no |
| 1926-1945 | 24/15 | 15% | 5.81% | $300,827 | -87.9% | 4,789 / 16 | no |
| 1926-1945 | 24/15 | 30% | 5.78% | $299,185 | -91.4% | 4,833 / 16 | yes |
| 1926-1945 | 24/15 | 45% | 5.71% | $295,103 | -93.9% | 90 / 4,868 | yes |
| 1926-1945 | 24/15 | 60% | 5.43% | $280,595 | -95.7% | 90 / 4,868 | yes |
| 1926-1945 | 40.8/23.8 | 0% | 4.76% | $247,887 | -84.8% | 4,803 / 16 | no |
| 1926-1945 | 40.8/23.8 | 15% | 5.30% | $273,820 | -88.0% | 4,804 / 16 | no |
| 1926-1945 | 40.8/23.8 | 30% | 5.33% | $275,178 | -91.5% | 4,845 / 16 | yes |
| 1926-1945 | 40.8/23.8 | 45% | 5.28% | $272,666 | -94.0% | 90 / 4,868 | yes |
| 1926-1945 | 40.8/23.8 | 60% | 5.04% | $260,984 | -95.7% | 90 / 4,868 | yes |
| 1946-1972 | 24/15 | 0% | 10.29% | $1.41m | -37.5% | 1,034 / 12 | no |
| 1946-1972 | 24/15 | 15% | 11.42% | $1.85m | -42.2% | 1,051 / 12 | no |
| 1946-1972 | 24/15 | 30% | 12.33% | $2.31m | -47.3% | 1,067 / 12 | no |
| 1946-1972 | 24/15 | 45% | 13.27% | $2.89m | -52.2% | 1,068 / 12 | no |
| 1946-1972 | 24/15 | 60% | 14.38% | $3.75m | -55.6% | 1,073 / 12 | no |
| 1946-1972 | 40.8/23.8 | 0% | 9.64% | $1.20m | -37.9% | 1,051 / 12 | no |
| 1946-1972 | 40.8/23.8 | 15% | 10.76% | $1.58m | -42.7% | 1,067 / 12 | no |
| 1946-1972 | 40.8/23.8 | 30% | 11.67% | $1.97m | -47.8% | 1,067 / 12 | no |
| 1946-1972 | 40.8/23.8 | 45% | 12.73% | $2.54m | -52.6% | 1,073 / 12 | no |
| 1946-1972 | 40.8/23.8 | 60% | 13.75% | $3.23m | -55.9% | 1,076 / 12 | no |
| 1973-1981 | 24/15 | 0% | 5.01% | $155,271 | -48.5% | 1,346 / 275 | no |
| 1973-1981 | 24/15 | 15% | 4.39% | $147,108 | -53.0% | 1,401 / 275 | no |
| 1973-1981 | 24/15 | 30% | 4.08% | $143,246 | -58.8% | 1,403 / 275 | no |
| 1973-1981 | 24/15 | 45% | 3.26% | $133,416 | -63.9% | 1,427 / 275 | no |
| 1973-1981 | 24/15 | 60% | 2.52% | $125,033 | -68.1% | 1,663 / 275 | no |
| 1973-1981 | 40.8/23.8 | 0% | 4.49% | $148,400 | -48.7% | 1,348 / 275 | no |
| 1973-1981 | 40.8/23.8 | 15% | 3.92% | $141,278 | -53.1% | 1,403 / 275 | no |
| 1973-1981 | 40.8/23.8 | 30% | 3.69% | $138,549 | -58.9% | 1,403 / 275 | no |
| 1973-1981 | 40.8/23.8 | 45% | 2.94% | $129,775 | -63.9% | 1,427 / 275 | no |
| 1973-1981 | 40.8/23.8 | 60% | 2.27% | $122,383 | -68.1% | 1,663 / 275 | no |
| 1982-1999 | 24/15 | 0% | 16.30% | $1.51m | -33.1% | 437 / — | no |
| 1982-1999 | 24/15 | 15% | 17.71% | $1.88m | -37.3% | 475 / — | no |
| 1982-1999 | 24/15 | 30% | 18.93% | $2.26m | -42.4% | 486 / — | no |
| 1982-1999 | 24/15 | 45% | 20.16% | $2.72m | -46.6% | 493 / — | no |
| 1982-1999 | 24/15 | 60% | 21.40% | $3.27m | -50.4% | 506 / — | no |
| 1982-1999 | 40.8/23.8 | 0% | 15.43% | $1.32m | -33.1% | 439 / — | no |
| 1982-1999 | 40.8/23.8 | 15% | 16.82% | $1.64m | -37.3% | 480 / — | no |
| 1982-1999 | 40.8/23.8 | 30% | 18.02% | $1.97m | -42.5% | 486 / — | no |
| 1982-1999 | 40.8/23.8 | 45% | 19.26% | $2.38m | -46.7% | 493 / — | no |
| 1982-1999 | 40.8/23.8 | 60% | 20.54% | $2.88m | -50.5% | 508 / — | no |
| 2000-2026-06 | 24/15 | 0% | 7.68% | $709,671 | -54.8% | 1,671 / 19 | no |
| 2000-2026-06 | 24/15 | 15% | 8.15% | $796,575 | -61.0% | 1,691 / 19 | no |
| 2000-2026-06 | 24/15 | 30% | 8.50% | $868,055 | -65.3% | 1,779 / 19 | no |
| 2000-2026-06 | 24/15 | 45% | 8.79% | $930,925 | -70.4% | 3,233 / 19 | no |
| 2000-2026-06 | 24/15 | 60% | 8.97% | $972,099 | -76.1% | 3,297 / 19 | no |
| 2000-2026-06 | 40.8/23.8 | 0% | 7.19% | $629,482 | -54.8% | 1,676 / 19 | no |
| 2000-2026-06 | 40.8/23.8 | 15% | 7.66% | $706,944 | -61.1% | 1,691 / 19 | no |
| 2000-2026-06 | 40.8/23.8 | 30% | 8.05% | $776,858 | -65.3% | 1,785 / 19 | no |
| 2000-2026-06 | 40.8/23.8 | 45% | 8.36% | $838,308 | -70.5% | 3,233 / 19 | no |
| 2000-2026-06 | 40.8/23.8 | 60% | 8.55% | $877,862 | -76.2% | 3,297 / 19 | no |

### Fee, wash and distribution sensitivities

Each entry is the full-window after-tax CAGR at the listed scenario key and mix; column keys are `scenarios["0.89%|{bracket}|wash=False|shiller"][mix]["full"].after_tax_cagr`, `scenarios["0.84%|{bracket}|wash=True|shiller"][mix]["full"].after_tax_cagr`, and `scenarios["0.84%|{bracket}|wash=False|zero"][mix]["full"].after_tax_cagr`. The 0.89% fee re-fit leaves displayed CAGR unchanged; the zero-distribution run is a modelling sensitivity, not a realized tax outcome.

| Bracket | SSO mix | Fee 0.89%, refitted | Wash handling on | Zero distributions |
|---|---:|---:|---:|---:|
| 24/15 | 0% | 9.47% | 9.45% | 9.99% |
| 24/15 | 15% | 10.18% | 10.12% | 10.57% |
| 24/15 | 30% | 10.71% | 10.64% | 10.97% |
| 24/15 | 45% | 11.17% | 11.11% | 11.40% |
| 24/15 | 60% | 11.60% | 11.51% | 11.69% |
| 40.8/23.8 | 0% | 9.06% | 9.01% | 9.87% |
| 40.8/23.8 | 15% | 9.83% | 9.67% | 10.39% |
| 40.8/23.8 | 30% | 10.30% | 10.18% | 10.74% |
| 40.8/23.8 | 45% | 10.76% | 10.67% | 11.13% |
| 40.8/23.8 | 60% | 11.26% | 11.19% | 11.46% |

## 3. Actionable Research Directions

**Present the 30% versus 15% risk and return trade-off at the 2027-09-30 A23 review.** The 30% path's primary after-tax premium over 0% is **+1.24 points/year**, but its worst drawdown is **6.8 points deeper** and crosses the registered ruin line. The high-rate 1973–1981 block is also worse for 30%: **4.08%** after-tax CAGR against **5.01%** for 0%, or **−0.94 point/year** (`scenarios["0.84%|24/15|wash=False|shiller"]["30"]["1973-1981"].after_tax_cagr` and `["0"]["1973-1981"].after_tax_cagr`). At 15%, the full-window drawdown improves to **−87.9%**, while full-window CAGR falls from **10.71% to 10.18%** (the same scenario's `["15"]["full"]` and `["30"]["full"]`). This is a proposal for a dated discussion only. A different target would require the owner's amendment to A23's permitted action set; there is no current allocation change.

The 30% full-window maximum drawdown is also the 1926–1945 block maximum; that block's recovery takes **4,833 completed sessions**, with **16 sessions** still open at its end (`scenarios["0.84%|24/15|wash=False|shiller"]["30"]["1926-1945"]`). The primary 30% arm trails 0% in **21.3%** of rolling 10-year windows, with a worst **−3.09 points/year** gap, and in **9.0%** of rolling 20-year windows, worst **−0.57 points/year** (`rolling_summary["10"]["30"]` and `rolling_summary["20"]["30"]`). These are risks to show alongside the century CAGR, not signals for a timed exit.

## 4. Completed & Closed Research Arcs

**A change under the registered century rule is NO-GO.** The gate's below-30 clause is met because the 30% arm itself trips ruin, but it is **only a necessary condition**. Both 0% and 15% fail the independent requirement to exceed 30%'s full-window CAGR by at least +0.5 point/year; 45% fails that CAGR hurdle, the high-rate comparison and the no-ruin test; 60% clears the CAGR hurdle but fails the high-rate comparison and no-ruin test (`recommendation_rule.alternatives`). The resulting `keep 30` is a **design limitation of the pre-registered rule flagged by the senior quant**: the rule can retain a ruin-flagged incumbent even when a lower-risk arm exists. It is not an after-the-fact reinterpretation of the gate, nor a recommendation to ignore the ruin observation.

The original A23 modern real-SSO study and A24 timing rider remain separate findings in the [September report](2026-09-30-static-fractional-leverage.md). The century study does not turn synthetic backward history into an independent confirmation of their modern returns.

## 5. Operational Roadmap: Recommended First Action

**Queue paper-trading reliability work for the approximate 2026-12-29 production-freeze lift.** Keep the existing observation and freeze terms from [next steps](../next_steps.md). This report changes no trading code, allocation, rebalance band, or execution schedule. Its allocation trade-off belongs at the owner's **2027-09-30** review; it does not authorize an intra-year sale or any drawdown-triggered exit.

## 6. Contrarian Evaluation & Parked Research

The registered rule in the [approved brief §6](briefs/2026-10-23-century-leverage.md#6-pre-registered-recommendation-rule-for-the-2027-review) is quoted exactly:

> The **only** way this study changes the senior quant's advice for the
> 2027-09-30 A23 annual review is to recommend moving from 30% to a fixed
> alternative that satisfies **all four** conditions on unrounded primary
> 24%/15% results: (a) full-window after-tax CAGR exceeds 30%'s by at least
> **+0.5 percentage point per year**; (b) its 1973–1981 after-tax CAGR is
> **not below** 30%'s; (c) its full-window max drawdown is no more than
> **10 percentage points deeper** than 30%'s; and (d) it never trips the
> **−90% ruin flag**. A recommendation **below 30%** additionally requires
> that the 30% arm itself trips the ruin flag **or** trails the 0% arm in
> full-window after-tax CAGR. Otherwise advise **keep 30%**. If multiple
> alternatives pass, report all and identify the one with highest full-window
> after-tax terminal wealth under this fixed rule; do not use diagnostics to
> rescue a failed gate. This is advice for a dated review only, not automatic
> authorization to trade or an intra-year exit.

**Model limits.** The French value-weighted all-US market is **not the S&P 500**; both legs are proxy paths. The daily-reset 2× fund is synthetic throughout, and **no such fund existed before 2006**. A fund closure or termination during a −90% event, loss of daily exposure, changing financing terms, margin stress and counterparty failure are not modelled (`provenance.synthetic`, `provenance.market_proxy`). Shiller's S&P dividend yield is applied to the all-US French index; its monthly-average price and annual dividend field are converted to a yield and treated as a cash distribution at that month's last trading close, an approximation rather than an observed month-end payment (`provenance.dividend_model` and `provenance.distribution_basis`). The 2× synthetic leg pays zero distributions; the frozen real SSO series has **69** distribution days through the study endpoint (`sso_real_distribution_check.distribution_days`), so its historical tax character is not replicated.

The tax book applies current-law federal rates hypothetically throughout the century, omits the ordinary-loss offset, approximates wash sales, and pays annual taxes from the portfolio; historical tax schedules and state taxes are absent (`provenance.tax_model`). The spread is fitted on the modern **2006-06-21 → 2026-04-30** overlap, with aligned returns starting the next session, then carried backward; it is a composite tracking residual, not an identified historical financing premium (`calibration.registered_window`, `calibration.aligned_window`, `calibration.fit`). The §6 implementation tests the ruin flag on **the full window only**, not every subperiod (`recommendation_rule.alternatives[*].conditions.no_ruin` and [rule code](../../src/ggTrader/lab/century_leverage.py)). The full window ends on **2026-06-30**; **July and August 2026 are excluded** because Shiller's dividend field is blank, with no imputation under [r2](briefs/2026-10-23-century-leverage-ruling-r2.md) (`provenance.full_window` and `provenance.excluded_months`).

**Next step:** present the quantified trade-off at the 2027-09-30 A23 review and queue paper-trading reliability work for the approximate 2026-12-29 freeze lift. Neither step changes the current book or creates a drawdown exit.
