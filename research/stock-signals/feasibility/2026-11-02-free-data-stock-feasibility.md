# Free-Data Stock Feasibility Inventory

**Classification:** Internal Quantitative Research & Engineering Strategy
**Date:** 2026-11-02
**Status:** Feasibility inventory. **No candidate is recommended here, no return was computed, and no test is authorized.** It feeds the owner discussion at gate G3. Machine-readable matrix: [`_free_data_feasibility_matrix.json`](_free_data_feasibility_matrix.json).

Terms: **Part A** = long free-history test (Holm-corrected Newey–West t ≥ 3 and ≥ +1.0 pt/yr net since 1994). **B0** = kill-only replay on the clean 2021-01-31 → 2026-04-30 tape. **B1** = forward shadow book of ≥ 18 months. Rules: [forward-paper standard](../../programme/evidence-standard/2026-10-23-forward-paper-standard.md). **PIT** = point-in-time (only information available on each decision date). **pt/yr** = percentage points per year. **TE** = tracking error (annualized volatility of excess return). **MDE** = minimum detectable effect.

## 1. Selection rule and scope

Families are listed for **coverage of mechanisms and free-data availability**, not by realized or published return. The author states this rule was set before enumerating; that chronology is asserted, not independently witnessed. The inventory is **non-exhaustive**. Recorded NO-GOs and parked event studies appear only for orientation. Where a listed family overlaps closed work, the overlap is labeled an inference and the closure is **not** silently extended to it (see §3). Owner constraints apply: free sources only, taxable account, long-only feasibility considered, crypto out of scope.

## 2. What was source-checked (access date 2026-11-02)

| Source | Verified from the page | Not verified |
|---|---|---|
| [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Lists monthly (many also daily) files for short-term and long-term reversal, beta, variance, residual variance, accruals, earnings/price, cash-flow/price, dividend yield, momentum, net share issues, and 5–49 industry portfolios | Per-file start dates (not on the index page); release lag; any terms of use (the page carries only a copyright notice). The page states: "We reconstruct the full history of returns each month when we update the portfolios" and that historical returns can change, so no unrevised historical vintage is available |
| [AQR datasets](https://www.aqr.com/Insights/Datasets) | Quality-sorted portfolios (US long sample from 1956), BAB and HML-Devil series, listed Aug 31 2026 | Terms of use (not stated on the page); construction details |
| [Global-q factors](https://global-q.org/factors.html) | CSV q-factor files, 1967–2025, July 2026 release; also two sorted benchmark-portfolio sets (18 size/investment-to-assets/ROE and 6 size/expected-growth portfolios) | Construction, implementability and terms of use of the benchmark-portfolio files |
| [SEC 13F data sets](https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets) | July 2013 onward, updated quarterly | Field list and filing-date field |
| [SEC Financial Statement Data Sets](https://www.sec.gov/dera/data/financial-statement-data-sets.html) | 2009 Q1 onward, updated quarterly; late-quarter filings appear in the next file | Whether an acceptance date is included |

Not source-checked in this pass (no classification given): 52-week-high momentum, the earnings-announcement premium, stock-level seasonality, buyback/net-payout yield, and all crypto families.

## 3. Matrix summary (20 rows; non-exhaustive)

Each row keeps three things apart: the **recorded verdict** (a documented NO-GO or status for that exact work), an **inferred overlap** with closed work (an inference, labeled as such), and **eligibility** (what is verified about data and history).

| Classification | Count | Rows |
|---|---:|---|
| `potential_part_a_source` | 4 | S01 short-term reversal, S02 long-term reversal, S03 low beta/variance, S04 accruals |
| `family_scope_review_undetermined` | 3 | S05 valuation variants, S06 industry momentum, S07 quality (QMJ) |
| `testable_b0_b1_only` | 2 | S10 13F holdings signals, S11 PIT-fundamentals enabler |
| `infeasible_free` | 2 | S09 analyst revisions, C06 options skew / anomaly-driven demand (register findings) |
| `closed_no_reopen` | 4 | C01 momentum, C02 four classic factors, C03 net issuance, C05 other recorded NO-GOs |
| `parked` | 5 | S08 q-factors, S12 retail attention, S13 factor momentum, C04 PEAD/insider, C07 turn-of-month |

What the labels mean, and do not mean:

- `potential_part_a_source` means a free source is **listed** that might supply long-history data. History start dates, construction and point-in-time status are **unverified** (the French library index page states no start dates, and the library says it reconstructs the full history of returns each update, so an unrevised historical vintage is not available; lagged portfolio formation is a separate question and was not rechecked). It is not a recommendation and not a statement that the test is feasible.
- `family_scope_review_undetermined` rows have **no recorded verdict for their own variant**; the [family-scope review](2026-11-02-family-scope-review.md) could not decide whether the related closure applies. A related family has a recorded NO-GO, but the [research index](../../RESEARCH_INDEX.md) says a NO-GO does not automatically apply to every related idea. Whether the closure applies needs a documented family-scope review. The same overlap uncertainty applies to S02, S03, S04 and S13 and is stated in their rows.
- `testable_b0_b1_only` rows cannot satisfy Part A on free data (history begins 2009 or 2013). The approved standard requires Part A **first**, so using them would need an owner amendment; this inventory does not propose one.
- Terms-of-use caveat: several free sources do not state terms of use on the pages read here. Verify terms before any use.
- S08: the checked page lists two sorted benchmark-portfolio sets (18 size/investment-to-assets/ROE portfolios and 6 size/expected-growth portfolios). Their construction, implementability and terms of use were not checked.

## 4. Observations for the direction discussion (not recommendations)

1. **Several listed families overlap recorded work.** S02, S03 and S04 each have an inferred overlap with a closed or NO-GO result (value, `idio_vol`/MAX, profitability/quality); the overlap has not been measured, and each row carries its own turnover, tax or data uncertainties. No ranking among them is implied.
2. **Free history is the binding constraint for fundamentals and holdings.** Stock-level PIT fundamentals start in 2009 and 13F in 2013 on the sources checked, so those signals reach only B0/B1 unless Part A is run on a long-history portfolio proxy.
3. **The Part A significance bar is demanding relative to modest, high-tracking-error effects** (§5, a calculation under assumptions).

## 5. Statistical-power note for the A1 significance test (calculation under stated assumptions)

*Calculation, not a finding about any strategy.* This note approximates the power of the **A1** test only: a one-sided t ≥ 3.0 on monthly arithmetic excess returns (p ≈ 0.00135 for a single test; Holm hurdles for a family can be higher). It estimates **neither A2** (the since-1994 CAGR difference after costs, before tax, a different statistic and window) **nor the probability of passing both A1 and A2**.

Approximate the t-statistic by `t ≈ μ / (TE × k / √Y)`, where μ is the annualized arithmetic mean excess return (pt/yr), `TE` the annual tracking error, `Y` years of data, and `k` an inflation factor for Newey–West autocorrelation (1.0 = ignored, 1.25 = illustrative). Power uses a normal approximation: 80% power requires `μ ≥ (3.0 + 0.8416) × TE × k / √Y`.

Minimum detectable true annualized arithmetic excess (pt/yr) for 80% power at t ≥ 3 (values are `k = 1.0` / `k = 1.25`):

| TE (pt/yr) | Y = 32.3 (since 1994) | Y = 63 | Y = 100 |
|---:|---:|---:|---:|
| 2 | 1.35 / 1.69 | 0.97 / 1.21 | 0.77 / 0.96 |
| 3 | 2.03 / 2.53 | 1.45 / 1.81 | 1.15 / 1.44 |
| 4 | 2.70 / 3.38 | 1.94 / 2.42 | 1.54 / 1.92 |
| 6 | 4.06 / 5.07 | 2.90 / 3.63 | 2.30 / 2.88 |

Expected A1 t-statistic when the true annualized arithmetic excess is +1.0 pt/yr (a value chosen only because the A2 bar is +1.0 pt/yr; it is not the A2 statistic): at Y = 63, TE 4 → t ≈ 1.98 (1.59 with k = 1.25); at Y = 100, TE 4 → 2.50 (2.00). The tracking error at which a true +1.0 pt/yr gives an expected A1 t of exactly 3.0 is `TE = √Y / 3`: 1.89 (Y = 32.3), 2.65 (Y = 63), 3.33 (Y = 100), each divided by `k`.

Hand check: Y = 63, TE = 3, k = 1.0: standard error = 3/√63 = 0.378; 1.0/0.378 = 2.65 and 3.8416 × 0.378 = 1.45. An independent review recomputed (63, 3, 1) as 1.452, 2.646, 2.646 and (32.3, 4, 1.25) as 3.380, 1.137, 1.516, matching the table.

Interpretation (assumption-dependent):

- 80% A1 power needs an information ratio (annual excess ÷ TE) near `3.84/√Y`: about 0.48 at 63 years and 0.68 since 1994.
- A true +1 pt/yr effect at TE ≥ ~2.7 pt/yr and 63 years is more likely than not to **fail A1** under these assumptions, so failing A1 does not show the effect is below +1 pt/yr. Relaxing t after seeing results would raise false positives. This is a design property of the approved standard, stated for the direction discussion; it changes no standard or past verdict.
- TE values are illustrative, not measured for any family. No return series were read.

## 6. What would change this inventory

- Confirmation of per-file start dates and terms of use for the sources above.
- A free source with documented PIT acceptance dates for fundamentals before 2009.
- An owner amendment about candidates that can only reach B0/B1.
- The source-check of the families listed as not checked.

**No candidate is recommended. G3 (direction) remains an owner decision.**
