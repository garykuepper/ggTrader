# Classic stock factor screen: Phase 1 NO-GO

**Classification:** Internal Quantitative Research & Engineering Strategy  
**Date:** 2026-10-21 (Pacific time)  
**Audience:** Owner and quantitative research collaborators  

## 1. Executive Summary & Core Engine Audit

**NO-GO. The classic stock factor screen is closed and reported as NO-GO. No paid data.** None of the four primary large-cap hypotheses passes: `overall_pass = false` for H1 (BIG HiOP), H2 (BIG HiBM), H3 (BIG LoINV), and H4 (50/50 Value + Momentum Combo) in the [final results JSON](_factor_screen_results.json) (`hypotheses.H1.overall_pass`, `hypotheses.H2.overall_pass`, `hypotheses.H3.overall_pass`, `hypotheses.H4.overall_pass`). The advancing factor is null (`advancing_factor = null`), and `next` records: *"Factor screen closed NO-GO, no paid data. Propose the next research step."* The verdict follows the "None pass" row of the [registered brief §8](briefs/2026-10-20-factor-screen.md), with costs and decision rules in §§5–6 unchanged. Phase 2 (paid survivorship-free data, exact S&P 500 rules, after-tax evaluation) is cancelled. A pass would have authorized research only, never deployment.

None of the four candidates is close on statistical significance test A1 under the pre-registered Holm step-down procedure (family-wise error rate $\alpha = 0.00135$ across the four hypotheses, equivalent to a one-sided $t \ge 3.00$ for a single test). The best full-window Newey-West $t$-statistic is **1.34** ($p = 0.0903$) for H4 (`hypotheses.H4.A1.t`, `hypotheses.H4.A1.p`), which falls far short of the first Holm threshold of $\alpha/4 = 0.0003375$ (approximate normal equivalent $t \approx 3.40$; `hypotheses.H4.A1.holm_threshold`). Holm testing therefore stops at Rank 1; all four hypotheses fail A1 (`pass_holm = false`). Standalone statistics are similarly weak: H2 (Value) achieves $t = 1.01$ ($p = 0.155$; `hypotheses.H2.A1.t`, `hypotheses.H2.A1.p`), H3 (Investment) achieves $t = 0.66$ ($p = 0.256$; `hypotheses.H3.A1.t`, `hypotheses.H3.A1.p`), and H1 (Profitability) achieves $t = 0.13$ ($p = 0.449$; `hypotheses.H1.A1.t`, `hypotheses.H1.A1.p`). Even without multiple-testing adjustment, none approaches the single-hypothesis $t \ge 3.00$ hurdle.

Economic persistence test A2 (net compound annual growth rate difference versus the French market since 1994 $\ge +1.0$ pt/yr after base haircut) fails across the board. The best A2 outcome is H1 (Robust Profitability) at **+0.83 pt/yr** (`hypotheses.H1.A2.cagr_diff_pp`), which fails the +1.0 pt/yr threshold and also fails A1 with a near-zero $t$ of 0.13. The other three candidates fail A2 completely by trailing the market net of costs since 1994: H3 (Conservative Investment) at **0.00 pt/yr** ($-0.0007$ pt/yr; `hypotheses.H3.A2.cagr_diff_pp`), H4 (Value + Momentum Combo) at **-0.56 pt/yr** (`hypotheses.H4.A2.cagr_diff_pp`), and H2 (Value) at **-0.62 pt/yr** (`hypotheses.H2.A2.cagr_diff_pp`).

Both classic long-history routes on survivorship-free academic data — momentum (closed on 2026-10-20 in [`docs/research/2026-10-20-momentum-french-screen-nogo.md`](2026-10-20-momentum-french-screen-nogo.md)) and these four classic factor portfolios — are now closed under the two-part standard. This is an independent academic-data historical screen on Ken French value-weighted portfolios, not an out-of-sample test of our exact trading rules or S&P 500 universe. The benchmark is French market total return (`Mkt-RF + RF`), not SPY. The [production-replica audit](2026-10-19-production-replica-audit-nogo.md) remains the separate closure of the deployed daily-bar stock-picking construction. No validated active stock candidate remains.

## 2. Quantitative Performance Context

All measured figures below come directly from the [final results JSON](_factor_screen_results.json); each cell cites its exact JSON key path. Registered windows, cost haircuts, and decision thresholds are taken from the [registered brief §§5–6](briefs/2026-10-20-factor-screen.md).

- **Common analysis window:** 1963-07 through 2026-04 (754 monthly observations; `inputs.common_window.start`, `inputs.common_window.end`, `inputs.common_window.n_months`), matching the inception of the operating profitability and investment series.
- **A2 economic persistence window:** 1994-01 through 2026-04 (388 monthly observations; `inputs.A2_window.start`, `inputs.A2_window.end`, `inputs.A2_window.n_months`).
- **Standard errors:** Bartlett Newey-West with 12 monthly lags (`inputs.nw_lags = 12`).
- **Multiple testing adjustment:** Holm step-down across the family of four hypotheses at overall family-wise error rate $\alpha = 0.00135$ (`inputs.holm_alpha = 0.00135`).
- **Cost haircuts:** Base annual cost is 1.0%/yr for H1–H3 and 1.5%/yr for H4; stress annual cost is 2.0%/yr for H1–H3 and 3.0%/yr for H4 (`hypotheses.*.base_cost`, `hypotheses.*.stress_cost`). Haircuts are subtracted monthly ($c/12$) before compounding.

### Critical brief caveats

1. **French BIG is not the S&P 500:** French's "BIG" portfolios split stocks above the NYSE median market capitalization, encompassing roughly 1,000+ US equity names including large- and mid-caps, not the ~500 large-cap constituents of the S&P 500. Furthermore, Value (HiBM) is defined strictly as book-to-market equity, which excludes intangible assets, brand equity, and capitalized R&D. A pass would show that a factor exists in broad large-caps; it would not prove that an investable S&P 500 implementation clears Part B.
2. **Partly in-sample for OP and INV:** Robust profitability and conservative investment were formally introduced in academic literature circa 2013–2015 (Novy-Marx 2013; Fama and French 2015). Consequently, the 1963–2012 history is retrospective backtesting, and only the 2013-01 through 2026-04 diagnostic interval (160 months) represents an out-of-sample or near-publication evaluation window.
3. **Haircuts are policy assumptions:** The 1.0%/yr base and 2.0%/yr stress allowances for H1–H3 (and 1.5%/yr base and 3.0%/yr stress for H4) are registered policy assumptions reflecting rebalancing and turnover friction. They are not calibrated turnover, execution-cost, or investor-tax measurements. Passing establishes robustness only relative to the stated policy haircuts.

### Primary hypotheses and registered gates

The confirmatory family comprises four large-cap (Big) hypotheses evaluated at base cost. Hypotheses are sorted by unrounded $p$-value for the Holm step-down procedure:

| Hypothesis | Factor Portfolio | Base Cost | Full-Window NW t | One-Sided p | Holm Rank | Holm Threshold | A1 Pass | Post-1994 Net Δpp | A2 Pass (≥ +1.0 pt/yr) | Overall Pass |
|---|---|---|---|---|---|---|---|---|---|---|
| **H4** | 50% BIG HiBM + 50% BIG HiPRIOR | 1.5%/yr (`hypotheses.H4.base_cost`) | 1.34 (`hypotheses.H4.A1.t`: 1.338846) | 0.0903 (`hypotheses.H4.A1.p`: 0.090310) | 1 (`hypotheses.H4.A1.holm_rank`) | 0.0003375 (`hypotheses.H4.A1.holm_threshold`) | false (`hypotheses.H4.A1.pass_holm`) | -0.56 pt/yr (`hypotheses.H4.A2.cagr_diff_pp`: -0.559214) | false (`hypotheses.H4.A2.pass`) | **false** (`hypotheses.H4.overall_pass`) |
| **H2** | BIG HiBM (Value) | 1.0%/yr (`hypotheses.H2.base_cost`) | 1.01 (`hypotheses.H2.A1.t`: 1.013603) | 0.155 (`hypotheses.H2.A1.p`: 0.155386) | 2 (`hypotheses.H2.A1.holm_rank`) | 0.000450 (`hypotheses.H2.A1.holm_threshold`) | false (`hypotheses.H2.A1.pass_holm`) | -0.62 pt/yr (`hypotheses.H2.A2.cagr_diff_pp`: -0.620700) | false (`hypotheses.H2.A2.pass`) | **false** (`hypotheses.H2.overall_pass`) |
| **H3** | BIG LoINV (Investment) | 1.0%/yr (`hypotheses.H3.base_cost`) | 0.66 (`hypotheses.H3.A1.t`: 0.655138) | 0.256 (`hypotheses.H3.A1.p`: 0.256189) | 3 (`hypotheses.H3.A1.holm_rank`) | 0.000675 (`hypotheses.H3.A1.holm_threshold`) | false (`hypotheses.H3.A1.pass_holm`) | 0.00 pt/yr (`hypotheses.H3.A2.cagr_diff_pp`: -0.000702) | false (`hypotheses.H3.A2.pass`) | **false** (`hypotheses.H3.overall_pass`) |
| **H1** | BIG HiOP (Profitability) | 1.0%/yr (`hypotheses.H1.base_cost`) | 0.13 (`hypotheses.H1.A1.t`: 0.129357) | 0.449 (`hypotheses.H1.A1.p`: 0.448537) | 4 (`hypotheses.H1.A1.holm_rank`) | 0.001350 (`hypotheses.H1.A1.holm_threshold`) | false (`hypotheses.H1.A1.pass_holm`) | +0.83 pt/yr (`hypotheses.H1.A2.cagr_diff_pp`: 0.826675) | false (`hypotheses.H1.A2.pass`) | **false** (`hypotheses.H1.overall_pass`) |

Holm evaluation stops at Rank 1 (H4): $p = 0.0903 > 0.0003375$. All four hypotheses fail A1. Separately, A2 fails across all four hypotheses: H1 (+0.83 pt/yr) falls short of the +1.0 pt/yr bar, while H2, H3, and H4 all fail to generate positive net excess returns versus the market since 1994.

### Subperiod diagnostics at base cost

Diagnostics are reported to examine regime stability and decay; they cannot override failed gates. Periods are inclusive:

| Hypothesis & Factor | Subperiod | Inclusive Dates | Months | NW t | One-Sided p | Net CAGR Difference vs Market |
|---|---|---|---|---|---|---|
| **H1: BIG HiOP** | Early | 1963-07 to 1993-12 (`hypotheses.H1.diagnostics.subperiods[0].start`, `.end`) | 366 (`hypotheses.H1.diagnostics.subperiods[0].n_months`) | -0.70 (`hypotheses.H1.diagnostics.subperiods[0].t`: -0.701616) | 0.759 (`hypotheses.H1.diagnostics.subperiods[0].p`: 0.758541) | -0.62 pt/yr (`hypotheses.H1.diagnostics.subperiods[0].cagr_diff_pp`: -0.618340) |
| H1: BIG HiOP | Middle | 1994-01 to 2012-12 (`hypotheses.H1.diagnostics.subperiods[1].start`, `.end`) | 228 (`hypotheses.H1.diagnostics.subperiods[1].n_months`) | 0.51 (`hypotheses.H1.diagnostics.subperiods[1].t`: 0.513083) | 0.304 (`hypotheses.H1.diagnostics.subperiods[1].p`: 0.303947) | +0.86 pt/yr (`hypotheses.H1.diagnostics.subperiods[1].cagr_diff_pp`: 0.863175) |
| H1: BIG HiOP | Recent (post-2013) | 2013-01 to 2026-04 (`hypotheses.H1.diagnostics.subperiods[2].start`, `.end`) | 160 (`hypotheses.H1.diagnostics.subperiods[2].n_months`) | 0.74 (`hypotheses.H1.diagnostics.subperiods[2].t`: 0.743096) | 0.229 (`hypotheses.H1.diagnostics.subperiods[2].p`: 0.228712) | +0.77 pt/yr (`hypotheses.H1.diagnostics.subperiods[2].cagr_diff_pp`: 0.770673) |
| **H2: BIG HiBM** | Early | 1963-07 to 1993-12 (`hypotheses.H2.diagnostics.subperiods[0].start`, `.end`) | 366 (`hypotheses.H2.diagnostics.subperiods[0].n_months`) | 2.00 (`hypotheses.H2.diagnostics.subperiods[0].t`: 1.996108) | 0.0230 (`hypotheses.H2.diagnostics.subperiods[0].p`: 0.022961) | +2.76 pt/yr (`hypotheses.H2.diagnostics.subperiods[0].cagr_diff_pp`: 2.760982) |
| H2: BIG HiBM | Middle | 1994-01 to 2012-12 (`hypotheses.H2.diagnostics.subperiods[1].start`, `.end`) | 228 (`hypotheses.H2.diagnostics.subperiods[1].n_months`) | 0.03 (`hypotheses.H2.diagnostics.subperiods[1].t`: 0.034861) | 0.486 (`hypotheses.H2.diagnostics.subperiods[1].p`: 0.486095) | -0.47 pt/yr (`hypotheses.H2.diagnostics.subperiods[1].cagr_diff_pp`: -0.472102) |
| H2: BIG HiBM | Recent (post-2013) | 2013-01 to 2026-04 (`hypotheses.H2.diagnostics.subperiods[2].start`, `.end`) | 160 (`hypotheses.H2.diagnostics.subperiods[2].n_months`) | 0.04 (`hypotheses.H2.diagnostics.subperiods[2].t`: 0.036116) | 0.486 (`hypotheses.H2.diagnostics.subperiods[2].p`: 0.485595) | -0.84 pt/yr (`hypotheses.H2.diagnostics.subperiods[2].cagr_diff_pp`: -0.844382) |
| **H3: BIG LoINV** | Early | 1963-07 to 1993-12 (`hypotheses.H3.diagnostics.subperiods[0].start`, `.end`) | 366 (`hypotheses.H3.diagnostics.subperiods[0].n_months`) | 1.16 (`hypotheses.H3.diagnostics.subperiods[0].t`: 1.162596) | 0.122 (`hypotheses.H3.diagnostics.subperiods[0].p`: 0.122497) | +1.29 pt/yr (`hypotheses.H3.diagnostics.subperiods[0].cagr_diff_pp`: 1.291730) |
| H3: BIG LoINV | Middle | 1994-01 to 2012-12 (`hypotheses.H3.diagnostics.subperiods[1].start`, `.end`) | 228 (`hypotheses.H3.diagnostics.subperiods[1].n_months`) | 0.24 (`hypotheses.H3.diagnostics.subperiods[1].t`: 0.239050) | 0.406 (`hypotheses.H3.diagnostics.subperiods[1].p`: 0.405533) | +0.59 pt/yr (`hypotheses.H3.diagnostics.subperiods[1].cagr_diff_pp`: 0.593976) |
| H3: BIG LoINV | Recent (post-2013) | 2013-01 to 2026-04 (`hypotheses.H3.diagnostics.subperiods[2].start`, `.end`) | 160 (`hypotheses.H3.diagnostics.subperiods[2].n_months`) | -0.55 (`hypotheses.H3.diagnostics.subperiods[2].t`: -0.547831) | 0.708 (`hypotheses.H3.diagnostics.subperiods[2].p`: 0.708096) | -0.89 pt/yr (`hypotheses.H3.diagnostics.subperiods[2].cagr_diff_pp`: -0.893754) |
| **H4: Combo** | Early | 1963-07 to 1993-12 (`hypotheses.H4.diagnostics.subperiods[0].start`, `.end`) | 366 (`hypotheses.H4.diagnostics.subperiods[0].n_months`) | 3.04 (`hypotheses.H4.diagnostics.subperiods[0].t`: 3.040699) | 0.00118 (`hypotheses.H4.diagnostics.subperiods[0].p`: 0.001180) | +2.39 pt/yr (`hypotheses.H4.diagnostics.subperiods[0].cagr_diff_pp`: 2.394143) |
| H4: Combo | Middle | 1994-01 to 2012-12 (`hypotheses.H4.diagnostics.subperiods[1].start`, `.end`) | 228 (`hypotheses.H4.diagnostics.subperiods[1].n_months`) | -0.03 (`hypotheses.H4.diagnostics.subperiods[1].t`: -0.034649) | 0.514 (`hypotheses.H4.diagnostics.subperiods[1].p`: 0.513820) | -0.14 pt/yr (`hypotheses.H4.diagnostics.subperiods[1].cagr_diff_pp`: -0.138370) |
| H4: Combo | Recent (post-2013) | 2013-01 to 2026-04 (`hypotheses.H4.diagnostics.subperiods[2].start`, `.end`) | 160 (`hypotheses.H4.diagnostics.subperiods[2].n_months`) | -0.56 (`hypotheses.H4.diagnostics.subperiods[2].t`: -0.555033) | 0.711 (`hypotheses.H4.diagnostics.subperiods[2].p`: 0.710564) | -1.19 pt/yr (`hypotheses.H4.diagnostics.subperiods[2].cagr_diff_pp`: -1.192020) |

### Modern decay: post-2013 performance

The post-2013 interval (2013-01 through 2026-04; 160 months) isolates modern market performance:
- **H1 (BIG HiOP):** Net CAGR difference is **+0.77 pt/yr** (`hypotheses.H1.diagnostics.post_2013_cagr_diff_pp`: 0.770673), with $t = 0.74$. While positive, it remains below the +1.0 pt/yr hurdle and follows a negative early period ($-0.62$ pt/yr).
- **H2 (BIG HiBM):** Net CAGR difference is **-0.84 pt/yr** (`hypotheses.H2.diagnostics.post_2013_cagr_diff_pp`: -0.844382), trailing the market outright with $t = 0.04$.
- **H3 (BIG LoINV):** Net CAGR difference is **-0.89 pt/yr** (`hypotheses.H3.diagnostics.post_2013_cagr_diff_pp`: -0.893754), with negative $t = -0.55$.
- **H4 (Combo):** Net CAGR difference is **-1.19 pt/yr** (`hypotheses.H4.diagnostics.post_2013_cagr_diff_pp`: -1.192020), with negative $t = -0.56$.

Three of the four factors have generated negative net excess returns over the past 13+ years.

### Century-long history (1927–2026) for Value and Combo

Value (BIG HiBM) and the Value + Momentum Combo (H4) have returns extending back to 1927-01 (1,192 monthly observations; 99.3 years):

| Hypothesis & Factor | Inclusive Dates | Months | NW t | One-Sided p | Net CAGR Difference vs Market |
|---|---|---|---|---|---|
| **H2: BIG HiBM** | 1927-01 to 2026-04 (`hypotheses.H2.diagnostics.history_1927.start`, `.end`) | 1,192 (`hypotheses.H2.diagnostics.history_1927.n_months`) | 1.69 (`hypotheses.H2.diagnostics.history_1927.t`: 1.693980) | 0.0451 (`hypotheses.H2.diagnostics.history_1927.p`: 0.045135) | +0.96 pt/yr (`hypotheses.H2.diagnostics.history_1927.cagr_diff_pp`: 0.958825) |
| **H4: Combo** | 1927-01 to 2026-04 (`hypotheses.H4.diagnostics.history_1927.start`, `.end`) | 1,192 (`hypotheses.H4.diagnostics.history_1927.n_months`) | 2.51 (`hypotheses.H4.diagnostics.history_1927.t`: 2.505821) | 0.00611 (`hypotheses.H4.diagnostics.history_1927.p`: 0.006108) | +1.25 pt/yr (`hypotheses.H4.diagnostics.history_1927.cagr_diff_pp`: 1.249475) |

Even over an entire century of survivorship-free data:
- Value achieves a Newey-West $t$ of only 1.69 ($p = 0.0451$), failing the single-hypothesis threshold $t \ge 3.00$ by a wide margin, and generating +0.96 pt/yr net excess CAGR.
- The 50/50 Combo achieves $t = 2.51$ ($p = 0.00611$), also failing $t \ge 3.00$. Its century-long return (+1.25 pt/yr) was driven entirely by the pre-1994 era (+2.39 pt/yr in 1963–1993); since 1994, it has lost -0.56 pt/yr relative to the market.

### Stress-cost diagnostics

Under the registered stress cost haircuts (2.0%/yr for H1–H3; 3.0%/yr for H4), all four factors produce negative net excess returns since 1994 and non-significant full-window statistics:

| Hypothesis & Factor | Stress Annual Cost | Full-Window NW t | Full-Window p | Post-1994 Net Δpp vs Market |
|---|---|---|---|---|
| **H1: BIG HiOP** | 2.0%/yr (`hypotheses.H1.diagnostics.stress.annual_cost`) | -1.79 (`hypotheses.H1.diagnostics.stress.full_t`: -1.787401) | 0.963 (`hypotheses.H1.diagnostics.stress.full_p`: 0.963064) | -0.28 pt/yr (`hypotheses.H1.diagnostics.stress.post_cagr_diff_pp`: -0.276008) |
| **H2: BIG HiBM** | 2.0%/yr (`hypotheses.H2.diagnostics.stress.annual_cost`) | 0.21 (`hypotheses.H2.diagnostics.stress.full_t`: 0.212760) | 0.416 (`hypotheses.H2.diagnostics.stress.full_p`: 0.415757) | -1.71 pt/yr (`hypotheses.H2.diagnostics.stress.post_cagr_diff_pp`: -1.711063) |
| **H3: BIG LoINV** | 2.0%/yr (`hypotheses.H3.diagnostics.stress.annual_cost`) | -0.62 (`hypotheses.H3.diagnostics.stress.full_t`: -0.620049) | 0.732 (`hypotheses.H3.diagnostics.stress.full_p`: 0.732387) | -1.10 pt/yr (`hypotheses.H3.diagnostics.stress.post_cagr_diff_pp`: -1.095929) |
| **H4: Combo** | 3.0%/yr (`hypotheses.H4.diagnostics.stress.annual_cost`) | -0.93 (`hypotheses.H4.diagnostics.stress.full_t`: -0.934583) | 0.825 (`hypotheses.H4.diagnostics.stress.full_p`: 0.824998) | -2.19 pt/yr (`hypotheses.H4.diagnostics.stress.post_cagr_diff_pp`: -2.191121) |

### Capital Asset Pricing Model (CAPM) diagnostics

CAPM regressions fit gross portfolio excess returns over the risk-free rate ($R_{i,t} - R_{f,t}$) on market excess returns ($R_{m,t} - R_{f,t}$) over the 1963-07 to 2026-04 window. Following the momentum precedent, no alpha $t$-statistic is reported (the prior calculation treated estimated beta as fixed and was mis-specified; a valid test is out of scope for a diagnostic):

| Hypothesis & Factor | Annualized Arithmetic Alpha | Market Beta |
|---|---|---|
| **H1: BIG HiOP** | 1.39% (`hypotheses.H1.diagnostics.capm.alpha_ann_pct`: 1.391123) | 0.95 (`hypotheses.H1.diagnostics.capm.beta`: 0.954847) |
| **H2: BIG HiBM** | 2.40% (`hypotheses.H2.diagnostics.capm.alpha_ann_pct`: 2.404024) | 0.98 (`hypotheses.H2.diagnostics.capm.beta`: 0.980697) |
| **H3: BIG LoINV** | 2.13% (`hypotheses.H3.diagnostics.capm.alpha_ann_pct`: 2.125824) | 0.91 (`hypotheses.H3.diagnostics.capm.beta`: 0.914606) |
| **H4: Combo** | 2.46% (`hypotheses.H4.diagnostics.capm.alpha_ann_pct`: 2.463611) | 0.99 (`hypotheses.H4.diagnostics.capm.beta`: 0.988804) |

All four portfolios exhibit market betas close to 1.0 (0.91 to 0.99). Gross arithmetic alphas range from 1.39% to 2.46%, but these are pre-cost, pre-tax, arithmetic averages over 63 years that are dominated by pre-1994 returns.

### Worst rolling 12 months & Relative drawdown

Gross worst 12-month relative return measures the worst compounded wealth ratio minus one: $\prod (1 + R_p) / \prod (1 + R_m) - 1$. Relative drawdown measures the maximum peak-to-trough drop in net relative wealth $\prod (1 + R_p - c/12) / \prod (1 + R_m)$ from an initial 1.0 over the 1963-07 to 2026-04 window:

| Hypothesis & Factor | Worst Rolling 12m Relative Return | Worst 12m Window | Maximum Relative Drawdown | Peak Date | Trough Date | Recovery Date | Recovery Elapsed |
|---|---|---|---|---|---|---|---|
| **H1: BIG HiOP** | -11.94% (`hypotheses.H1.diagnostics.worst_12m.relative_pct`: -11.937052) | 1999-03 to 2000-02 (`hypotheses.H1.diagnostics.worst_12m.start`, `.end`) | -31.55% (`hypotheses.H1.diagnostics.relative_drawdown.max_drawdown_pct`: -31.545766) | 1973-06 (`hypotheses.H1.diagnostics.relative_drawdown.peak`) | 1984-01 (`hypotheses.H1.diagnostics.relative_drawdown.trough`) | unrecovered (`hypotheses.H1.diagnostics.relative_drawdown.recovery`: null) | null (`hypotheses.H1.diagnostics.relative_drawdown.recovery_months`: null) |
| **H2: BIG HiBM** | -28.08% (`hypotheses.H2.diagnostics.worst_12m.relative_pct`: -28.078687) | 2019-10 to 2020-09 (`hypotheses.H2.diagnostics.worst_12m.start`, `.end`) | -59.44% (`hypotheses.H2.diagnostics.relative_drawdown.max_drawdown_pct`: -59.436102) | 2008-09 (`hypotheses.H2.diagnostics.relative_drawdown.peak`) | 2020-09 (`hypotheses.H2.diagnostics.relative_drawdown.trough`) | unrecovered (`hypotheses.H2.diagnostics.relative_drawdown.recovery`: null) | null (`hypotheses.H2.diagnostics.relative_drawdown.recovery_months`: null) |
| **H3: BIG LoINV** | -18.50% (`hypotheses.H3.diagnostics.worst_12m.relative_pct`: -18.496155) | 1999-03 to 2000-02 (`hypotheses.H3.diagnostics.worst_12m.start`, `.end`) | -28.39% (`hypotheses.H3.diagnostics.relative_drawdown.max_drawdown_pct`: -28.393540) | 1990-09 (`hypotheses.H3.diagnostics.relative_drawdown.peak`) | 2000-02 (`hypotheses.H3.diagnostics.relative_drawdown.trough`) | 2001-08 (`hypotheses.H3.diagnostics.relative_drawdown.recovery`) | 18 months (`hypotheses.H3.diagnostics.relative_drawdown.recovery_months`: 18) |
| **H4: Combo** | -13.14% (`hypotheses.H4.diagnostics.worst_12m.relative_pct`: -13.136053) | 2008-08 to 2009-07 (`hypotheses.H4.diagnostics.worst_12m.start`, `.end`) | -41.47% (`hypotheses.H4.diagnostics.relative_drawdown.max_drawdown_pct`: -41.468001) | 2008-09 (`hypotheses.H4.diagnostics.relative_drawdown.peak`) | 2020-09 (`hypotheses.H4.diagnostics.relative_drawdown.trough`) | unrecovered (`hypotheses.H4.diagnostics.relative_drawdown.recovery`: null) | null (`hypotheses.H4.diagnostics.relative_drawdown.recovery_months`: null) |

H1, H2, and H4 are in unresolved relative drawdowns versus the market. Value (H2) suffered a -59.44% relative drawdown from September 2008 through September 2020 and has not recovered. The 50/50 Combo (H4) suffered a -41.47% relative drawdown over the same period and also remains unrecovered as of April 2026. Only Conservative Investment (H3) recovered from its maximum relative drawdown (-28.39% from 1990-09 to 2000-02, recovering in August 2001 after 18 months).

### Small-cap diagnostics (strictly outside the confirmatory Holm family)

Small-cap portfolios are reported solely as diagnostic context and are **strictly outside the confirmatory Holm family**. As noted in the brief (§§7 and 11), ggTrader has no point-in-time small-cap universe, no liquidity screening, and no small-cap execution infrastructure. Haircuts are 3.0%/yr base and 6.0%/yr stress:

| Diagnostic Portfolio | Base Cost | Full NW t | One-Sided p | Post-1994 Net Δpp | Post-2013 Net Δpp | Stress Cost | Stress NW t | Stress Post Δpp | CAPM Alpha | CAPM Beta | Worst 12m Relative Return (Dates) | Max Relative Drawdown (Peak to Trough) | Recovery |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SMALL HiOP** | 3.0%/yr | 0.47 (`small_cap_diagnostics['SMALL HiOP'].t`: 0.470761) | 0.319 (`small_cap_diagnostics['SMALL HiOP'].p`: 0.318906) | -0.34 pt/yr (`small_cap_diagnostics['SMALL HiOP'].cagr_diff_pp`: -0.342529) | -6.67 pt/yr (`small_cap_diagnostics['SMALL HiOP'].post_2013_cagr_diff_pp`: -6.673010) | 6.0%/yr | -1.38 | -5.10 pt/yr | 2.29% | 1.21 | -32.11% (1972-07 to 1973-06) | -63.20% (2010-05 to 2025-11) | unrecovered (null) |
| **SMALL HiBM** | 3.0%/yr | 0.90 (`small_cap_diagnostics['SMALL HiBM'].t`: 0.897719) | 0.185 (`small_cap_diagnostics['SMALL HiBM'].p`: 0.184668) | +0.83 pt/yr (`small_cap_diagnostics['SMALL HiBM'].cagr_diff_pp`: 0.825044) | -6.31 pt/yr (`small_cap_diagnostics['SMALL HiBM'].post_2013_cagr_diff_pp`: -6.305500) | 6.0%/yr | -0.83 | -5.15 pt/yr | 4.03% | 1.07 | -32.00% (2019-04 to 2020-03) | -70.03% (2006-03 to 2025-10) | unrecovered (null) |
| **SMALL LoINV** | 3.0%/yr | 0.64 (`small_cap_diagnostics['SMALL LoINV'].t`: 0.641368) | 0.261 (`small_cap_diagnostics['SMALL LoINV'].p`: 0.260642) | -0.05 pt/yr (`small_cap_diagnostics['SMALL LoINV'].cagr_diff_pp`: -0.048669) | -6.75 pt/yr (`small_cap_diagnostics['SMALL LoINV'].post_2013_cagr_diff_pp`: -6.754938) | 6.0%/yr | -1.19 | -5.32 pt/yr | 2.81% | 1.17 | -24.48% (1998-04 to 1999-03) | -71.54% (1983-07 to 2025-07) | unrecovered (null) |

Small-cap portfolios experienced catastrophic relative underperformance post-2013 ($-6.31$ to $-6.75$ pt/yr annualized deficit versus the market) and remain in severe, unrecovered relative drawdowns exceeding $-63\%$ to $-71\%$. Even if investable, they offer no positive evidence.

## 3. Actionable Research Directions

**No factor candidates advance.** With `advancing_factor = null`, Phase 2 is cancelled and no paid survivorship-free data will be purchased. No trading rules or tax replays will be constructed for these factors.

The quantitative screen establishes that classic long-only equity factors cannot meet the pre-registered bar for the owner's active stock-trading path. Both classic long-history avenues (momentum and the four classic factors) are now exhausted on clean survivorship-free data.

## 4. Completed & Closed Research Arcs (Do NOT Re-Propose)

**Classic stock factors (HiOP, HiBM, LoINV, Value+Momentum Combo) — CLOSED NO-GO.** Evaluated across 63 to 99 years of French library data; all four primary hypotheses fail both statistical significance A1 (best $t = 1.34$ vs Holm hurdle $t \approx 3.40$) and economic persistence A2 (three trail the market since 1994; profitability $+0.83$ pt/yr falls below $+1.0$ pt/yr). Post-2013 excess return is negative for Value, Investment, and Combo. Do not re-propose or re-screen these factors on adjusted windows or alternate factor definitions.

**Momentum — CLOSED NO-GO.** Century-long French screen closed NO-GO on 2026-10-20 ([`docs/research/2026-10-20-momentum-french-screen-nogo.md`](2026-10-20-momentum-french-screen-nogo.md)). Neither size passed Holm or the post-1994 $+1.0$ pt/yr bar. `xs_momentum` and `dual_momentum` were the same portfolio in the corrected replay.

**Deployed stock picks — CLOSED NO-GO.** The [production-replica audit](2026-10-19-production-replica-audit-nogo.md) remains the authority for that separate closure. Production remains frozen.

## 5. Operational Roadmap: Recommended First Action

**Close the factor screen and record NO-GO across all tracking artifacts.**

The final gate outcome is:
> NO-GO. Factor screen closed NO-GO, no paid data. None of the four primary hypotheses passes (`hypotheses.H1.overall_pass = false`, `hypotheses.H2.overall_pass = false`, `hypotheses.H3.overall_pass = false`, `hypotheses.H4.overall_pass = false`). Phase 2 is cancelled (`advancing_factor = null`).

Per standing owner instructions (2026-10-20):
- Recommendations must focus on research evidence and paper-trading reliability.
- The paper-trading path remains in its frozen observation window (through approximately 2026-12-29).
- Follow standing owner guidelines regarding taxable account boundaries and project continuity; do not recommend closing the stock-picking sleeve or stopping paper trading.
- The owner's separate A23 allocation (70/30 SPY/SSO) remains unchanged.

## 6. Contrarian Evaluation & Parked Research

### Contrarian evaluation

The primary contrarian counter-argument is that academic factor portfolios use simple book-to-market and operating profitability definitions that ignore modern intangible assets (such as capitalized research & development and software investments), and that a more sophisticated multi-factor composite or lower execution fee might preserve alpha.

However, the empirical evidence does not support pursuing this variation:
1. Post-1994 net compound growth differences are negative or near-zero for Value ($-0.62$ pt/yr), Investment ($0.00$ pt/yr), and Combo ($-0.56$ pt/yr).
2. Modern post-2013 performance is negative for three of the four factors, indicating that factor returns have compressed further in the recent decade.
3. Profitability's 63-year full-window $t$-statistic of 0.13 demonstrates an absence of statistical edge over the long sample.
4. The two-part standard was pre-registered to prevent retrospective searching and curve-fitting after observing negative outcomes.

### Parked ideas and proposed next research step

Both classic long-history routes — cross-sectional momentum and the four classic factors — are now closed under the two-part standard.

As the next research step, needing separate owner authorization before any work begins, we propose repairing public-availability timing for two event-driven strategies parked since the 2026-10-20 triage:
1. **Post-earnings announcement drift (PEAD):** Requires acquiring verified historical public earnings announcement timestamps to replace trading-day close assumptions and eliminate lookahead bias ([`docs/research/2026-07-17-pead-nogo.md`](2026-07-17-pead-nogo.md)).
2. **Insider cluster buying:** Requires acquiring SEC Form 4 public filing timestamps rather than transaction dates to model real-world signal availability ([`docs/research/2026-07-19-insider-cluster-buy-nogo.md`](2026-07-19-insider-cluster-buy-nogo.md)).

Both strategies were parked because their event timing was not certifiably point-in-time. Repairing these timestamps would establish whether clean event-driven signals exist for the taxable stock sleeve, without violating point-in-time rigor.
