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

Not source-checked in the original pass (no classification given): 52-week-high momentum, the earnings-announcement premium, stock-level seasonality, buyback/net-payout yield, and all crypto families. The later 52-week-high work is linked below.

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

## Follow-up: finding winners near their 52-week highs — 2026-11-10

The owner's current objective emphasizes finding winners, allowing gains to grow and managing losses. A question is whether a stock's current price relative to its previous 52-week high helps rank future winners. This is a related momentum-family question and does not reverse the existing momentum NO-GO. The owner subsequently authorized a separate study: its [fixed specification](2026-11-10-high52-screen-registration.md) and independently reviewed [result](2026-11-10-high52-screen-results.md) record that both hurdles were not met.

[George and Hwang (2004)](https://www.bauer.uh.edu/tgeorge/papers/gh4-paper.pdf) study rankings by price relative to a prior 52-week high. Their main comparison uses broad CRSP stocks, equally weighted winner and loser portfolios, and overlapping six-month holdings; the long–short paper result is not an S&P 500 long-only, after-cost result. A copy was admitted to DeepThought's paper library under SHA-256 `b6e80dd8a7355d29358240fe2df60a0fa2f47eac45b5de4b51a1a8e5380becaf`; library audit classifies it `needs_verification`, so search eligibility and publication status must not be assumed from admission.

[Open Source Asset Pricing](https://www.openassetpricing.com/) offers replicated signals and portfolio data. Its maintainers' [historical issue](https://github.com/OpenSourceAP/CrossSection/issues/95) flags a potential split-adjustment lookahead issue for High52, records a 2024 code fix, and notes subsequent work. Their October 2026 update also reports High52 split-adjustment changes. The later study checked the release's published code and provenance; it did not independently rebuild point-in-time underlying data. Present-day adjusted data alone does not prove decision-time availability.

At the initial follow-up no 52-week-high returns were downloaded or tested; the subsequent metadata check exposed a website preview, as noted below. The later screen uses a broad-market comparator, not SPY, and does not establish an implementable next-session strategy. The low-accruals screen's preliminary result is recorded separately in its [feasibility file](2026-11-02-accruals-feasibility.md); neither investigation authorizes a production change.

### Source availability and split handling

The [Open Source Asset Pricing 2026.10 data release](https://huggingface.co/datasets/OpenSourceAP/CrossSection) documents monthly firm-level signals by CRSP identifier and month, as well as separate long-only quantile and long–short portfolio returns. Its documented end date is December 2025, so it cannot by itself supply current stock picks. The data card states CC BY 4.0 and requests citation. The source describes signals as available at month-end and portfolio returns as close-to-close; a practical stock strategy still needs a later executable fill, security-identifier mapping, membership, costs and tax treatment. Availability and correctness of the required historical universe and underlying prices have not been established.

The provider's [2026.10 announcement](https://www.openassetpricing.com/) says it added split adjustments for High52. The [signal code, pinned to commit 9f053398](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/Predictors/High52.py), divides the month-end split-adjusted price by the highest adjusted daily price in the 12 calendar months including that month. The maintainers explain that common future split adjustments cancel in this ratio; this reasoning depends on consistent adjustments and does not establish the validity of every dataset vintage. Their [issue 205](https://github.com/OpenSourceAP/CrossSection/issues/205) records a reverse-split problem as addressed in October 2026. The [release notes](https://huggingface.co/datasets/OpenSourceAP/CrossSection/blob/Data-Release-2026.10/Release%20Notes%202026.10.md) say the data were produced with code version 2.1.0, whose tag resolves to that same commit, and describe both the split fix and inclusion of the current month. The 2026.10 data branch resolved to `dd1054254c4501776432caa6d7ce5ecd9586f6a9`; the published original-construction portfolio file's LFS SHA-256 is `76599d94beab94cc8973d8a126e8efbed98ee79f677738c55b2389f17387ba96`. This is provider-reported provenance, not an independent rebuild. The file's recorded commit predates the code tag, so their correspondence has not been independently proved. Stock-level coverage remains unverified.

The release's signal documentation identifies the original High52 construction as equal-weighted, using 30% tails and six-month portfolios. These settings do not establish an S&P 500 top-N book or a verified after-tax result. The later frozen-file screen confirmed nonmissing observations in its registered windows.

**Relationship to the closed work:** the earlier [French momentum screen](../momentum/2026-10-20-momentum-retest.md#5-phase-1-is-momentum-real-and-in-which-company-size-free-data) selected stocks on their prior 12–2-month returns, divided by company size, and used value-weighted portfolios. High52 instead ranks a current price against its preceding high, with the original paper's equal-weighted broad-market portfolios. These are distinct calculations, but they are economically related momentum constructions and share historical markets. The [momentum closure](../momentum/2026-10-20-momentum-french-screen-nogo.md#3-actionable-research-directions) says no further momentum work; the owner explicitly authorized this sibling as a separate test without reversing that closure.

The data-card website automatically displayed a preview containing historical portfolio returns while its documentation was being opened. The High52 documentation's notes also exposed a qualitative description of the original paper's results. These exposures were not used to choose a rule, but the source was not wholly unseen; the later brief discloses them. The owner authorized the separate fixed test after this initial metadata check. Its long-only portfolio failed the fixed hurdles; it is not a verified stock-selection or exit system, and the earlier momentum closure remains intact.

### Local stock-tape availability check

The engine evidence's frozen tape was located at `/home/flynn/ggTrader-vbt12/data/snapshots/ohlcv_20261106.parquet`. Its SHA-256, `6573e84065d54796a6240051c3d83a68b32a079241665fb44001ec0cbc789b2e`, matches the [recorded engine evidence](../../programme/vectorbt-engine/p5-evidence/_p5_ref_xs_momentum.json). It has 1,110,874 rows across 605 symbols, dated 2019-03-01 through 2026-10-30, with no duplicate symbol/date pairs. Only symbols and dates were inspected; no returns or signals were calculated. The tape is local to a different checkout and is not tracked here.

As a bounded coverage check, compare its dates with the most recent snapshot on or before each date from the tracked vintage-1 membership fixture (SHA-256 `b30739a3baba60de0f472404c5b653514e140ace8d4b89c3875d5bf1e184a472`) and the preceding 252 XNYS sessions, including the decision day (`exchange_calendars` 4.13.2):

| Date | Fixture members | Members with a same-day bar | Same-day members with all 252 session dates |
| --- | ---: | ---: | ---: |
| 2021-01-29 | 505 | 486 | 483 |
| 2026-04-30 | 503 | 503 | 501 |

The snapshot dates used were 2021-01-21 and 2026-04-09 respectively. **Correction:** an earlier version of this table reported 484/481 and 501/499 because it compared raw fixture symbols directly to tape symbols. The corrected table uses the project's `normalize_yf_ticker` (dots to hyphens); BRK.B and BF.B both have tape under their normalized names. Of the 19 still missing on 2021-01-29, 12 have no rows anywhere in this tape and seven have rows only outside that date. Across all 64 month-end trading dates from January 2021 through April 2026, the same-day missing count ranged from 0 to 19 members (median 4, mean 7.12); 22 distinct normalized symbols were absent on at least one month-end. These counts cover ticker/date presence only, not price validity or 252-session histories throughout the window.

The monthly counts were reproduced using the [coverage audit](code/stock_tape_coverage.py), which records its input and code hashes and calendar version in the [machine-readable result](_stock_tape_coverage_20261110.json). It uses only membership, dates and symbols, and refuses mismatched hashes or an existing output file.

Rows under a reused ticker must not be attributed to the historical company without an identity check; the [root-cause audit](../../audits/root-cause-2026-11-04/lenses/2026-11-04-lens1-backtest-validity.md) documents examples and additional tape identity faults. Missing symbols and short histories can reflect ticker changes, listings, data gaps or calendar conventions; this check does not establish each cause. The 252-session checks are two dates only and are illustrative, not a registered lookback. Timestamps alone do not validate OHLC values, splits, dividends, data identity, missing delistings, or whether membership was known on each decision date. The historical S&P fixture has documented limitations and is not proof of a survivorship-safe universe. The earlier engine evidence marks the tape `deep_checked: false`. This availability finding does not authorize a replay or a new momentum study.

**Potentially recoverable rename gaps:** the frozen tape contains earlier-dated rows under later tickers for five companies whose own announcements identify the changes: [MMC → MRSH, 2026-01-14](https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html); [ABC → COR, 2023-08-30](https://investor.cencora.com/news/news-details/2023/AmerisourceBergen-becomes-Cencora-in-alignment-with-the-companys-growing-global-footprint-and-central-role-in-pharmaceutical-access-and-care/default.aspx); [ANTM → ELV, 2022-06-28](https://ir.elevancehealth.com/events-and-presentations/event-details/2022/Corporate-Rebranding-to-New-Name-Elevance-Health-Inc/default.aspx); [BLL → BALL, 2022-05-10](https://investors.ball.com/news-presentations/press-releases/detail/82/ball-board-declares-quarterly-dividend-stock-ticker-symbol-changing-to-ball); and [RE → EG, 2023-07-10](https://www.everestglobal.com/us-en/news-media/press-releases/2023/everest-to-rebrand-company-name-and-nyse-ticker). Against the 64-month coverage record, matching the old fixture ticker to the later ticker strictly before each announced effective date would find a same-day row for **154 of 456 missing member-months**: MMC 60, ABC 31, ANTM 17, BLL 16 and RE 30. The January 2021 sample would fall from 19 missing to 14; 302 missing member-months would remain across the full window.

This is an availability diagnostic, not a repair. A ticker-change announcement establishes the issuer relationship, not that the historical bars under the successor symbol are accurate, uniformly adjusted or safe to splice. Source identity, split/dividend adjustment and point-in-time use still need checking; an alias applied retrospectively must not imply the future ticker was known at the historical decision date. No snapshot, membership file or database was changed.

**Fresh-source consistency check, 2026-11-11 UTC:** using `yfinance` 1.8.0 with `auto_adjust=True`, successor-symbol closes were downloaded from 2021-01-01 up to (excluding) each change date and compared with the same successor-symbol closes in the hash-verified frozen tape. Positive overlapping observations were 1,263 for MRSH, 668 for COR, 373 for ELV, 340 for BALL and 631 for EG; no date appeared on only one side for these intervals. For each symbol, the tape-to-download close ratio was nearly constant: after normalizing by its median, the largest deviation was below 0.001% in each case. Median ratios were 1.005832, 1.000000, 1.004352, 1.003186 and 1.005437 respectively. A constant scaling is consistent with different adjustment vintages, but its cause was not independently established.

This compares two versions of the same upstream source, not an independent price or security-identity verification. It checks close only, not OHLC, volume, delistings, distribution completeness or all decision dates, and the fresh downloads were not frozen, so they may later change. It does not prove these series are safe to splice or authorize changing the tape, membership, database, trading or production configuration.

### Another possible stock-selection mechanism: R&D spending relative to price

The [Chan–Lakonishok–Sougiannis paper](https://doi.org/10.1111/0022-1082.00411) examines whether companies' research-and-development spending is reflected in their share prices. The [Open Source Asset Pricing code](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/Predictors/RD.py) defines its `RD` characteristic as R&D expense divided by market equity. Its 2026.10 metadata describes annual, equal-weighted portfolios using 20% tails; this is not an individual S&P 500 strategy, and the long-only side and its exact history have not been evaluated here. A high ratio can result from a low share price, not just high R&D spending, so it overlaps economically with value and distressed-stock ideas even though it is not the same formula as the closed book-to-market factor. Its relation to the prior value closure has not been reviewed or resolved.

This is a different question from 52-week-high ranking, but not automatically an independent discovery. Published paper results and the provider's multi-signal preview have been exposed; the subsequent [exploratory comparison](2026-11-10-rd-exploratory-screen.md) discloses that selection and the many previously explored candidates. The [SEC financial-statement datasets](https://www.sec.gov/dera/data/financial-statement-data-sets.html) begin in 2009 and contain filing-level limitations, so they do not by themselves provide the 1994–2008 stock-level history or an identity-safe, point-in-time pipeline. Expenses, missing disclosures, filing availability, accounting revisions, company identity, turnover, costs and taxable gains all need treatment before a real stock implementation. The owner has not approved this candidate as a separate exception to any closed family; the exploratory comparison changes no prior verdict.

The publisher page and [NBER working paper](https://www.nber.org/system/files/working_papers/w7223/w7223.pdf) were located. An initial DeepThought admission returned `refused: incoherent` because the PDF's first page does not contain the supplied DOI and no title was supplied for the identity check. Supplying the source-verified title allowed admission under DOI `10.3386/w7223`, SHA-256 `ef9f458e37b5f8a5bc950a27ff1d97aa4c96aeac8ee0ee3c2b2c3d7922c9e55e`. A subsequent served-source lookup excludes that record, so admission must not be mistaken for search eligibility or peer-review verification. This is the NBER working paper, not an assertion that these bytes are the published journal version.
