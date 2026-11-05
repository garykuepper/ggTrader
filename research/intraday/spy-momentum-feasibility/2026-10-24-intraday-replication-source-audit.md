# Intraday replication source audit

Date: 2026-10-24; access follow-up 2026-10-25. Scope: source identity, the quoted results, and applicability to the owner's long-only, unleveraged SPY proposal. No market data, trading, raw-data backtests, or broker configuration changes were performed.

## Findings that change the feasibility report

The cited negative studies were located; an initial blocked SSRN request was not evidence that a paper did not exist. However, their implications were overstated. The noise-band replication with a pooled 0.35 Sharpe explicitly describes a strong first post-publication period followed by a weak later regime. Another replication's claim of a statistically significant Sharpe collapse has a unit error in its published analysis code. Neither is a test of the owner's long-only, unleveraged half-hour rule. The QQQ ORB paper also had the wrong SSRN identifier in the report: the correct identifier is **4416622**, not 4898296.

Evidence labels here distinguish **full primary text read**, **author summary read**, **indexed primary abstract only**, and **secondary abstract only**. Finding a published number is not independently reproducing it.

## 1. Paz Serpa: noise-band SPY replication, SSRN 7290621

**Full primary text read.** Sheimy Paz Serpa’s [paper source](https://raw.githubusercontent.com/PazSheimy/spy-intraday-momentum-oos/main/paper/Out-of-Sample_Intraday_Momentum_SPY_PazSerpa_2026.tex), §§2–4, confirms the quoted OOS results and a non-significant mean-difference test (p = 0.255). It uses IQFeed, long/short positions, up to 4x leverage and assumed costs, omitting dividends. Its May 1 start precedes the original May 10 posting. The [author repository](https://github.com/PazSheimy/spy-intraday-momentum-oos) exposes paper/code but not licensed observations. Direct SSRN access returned 403. Published returns remain unreproduced; the significance claim has the code defect below.

### Statistical code defect: Sharpe annualization and standard error use different units

**Analytical code finding, with an arithmetic self-check.** In the author's [analysis script](https://raw.githubusercontent.com/PazSheimy/spy-intraday-momentum-oos/main/src/zarattini_paper_analysis.py), lines 94–102, the Sharpe ratios are annualized with `sqrt(252)`, but the standard errors are calculated as `sqrt((1 + 0.5 * annual_SR**2) / number_of_daily_observations)`. The leading variance term has not been annualized. This is not a valid use of the per-observation IID approximation.

If `SR_a = sqrt(252) * SR_d`, consistent propagation gives:

```
SE(SR_a) = sqrt(252) * sqrt((1 + 0.5 * SR_d**2) / N)
         = sqrt((252 + 0.5 * SR_a**2) / N)
```

Using only rounded published inputs (`SR_a = 1.34, 0.39`; `N = 2333, 480`), a standard-library arithmetic check produced:

```
Published-code approximation: z = 17.1871
Consistently scaled IID approximation: z = 1.1935
Two-sided normal p for the latter: 0.2327
```

The paper's z of 17.053 uses unrounded data; these calculations do not reproduce that exact statistic. They identify the scale error and show why its claimed extreme significance cannot be accepted. The corrected approximation is itself conditional on IID/normal assumptions and independent samples; serial dependence and non-normality would require further inference work on the actual return series. It does not establish that the two regimes have identical performance. For the underlying methodological reference, see [Lo (2002), publisher record](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios).

## 2. Delgado: noise-band SPY audit, SSRN 7323419

**Primary registry abstract/identity checked on 2026-10-25; full methods inaccessible.** José Enrique Delgado, August 2026. The [Crossref registered abstract](https://api.crossref.org/works/10.2139/ssrn.7323419), independently retrieved after the owner’s access audit, reports pooled Sharpe 0.35, but early-period return 19.3% / Sharpe 1.057 through August 29, 2025 (23.3% / 1.284 from the exact post-publication start), followed by −8.1% / −0.461 for September 2, 2025–August 14, 2026. It expressly rejects immediate decay and discusses implementation reconciliation and subsequent searches. Direct SSRN and readable-fetch requests returned 403; bounded searches found no accessible author full text. This is abstract-level support for regime instability, not a full-methods audit.

## 3. Fetna: futures ORB study, SSRN 7428398

**Full author technical report and registration text read; SSRN PDF inaccessible.** Mulham Fetna’s [report](https://raw.githubusercontent.com/mulhamfetna/trading-strategy-finder/main/subprojects/Parametric-Indicators/shareable/orb_study_bundle/docs/WS-ORB-REPORT.md) and [registration](https://raw.githubusercontent.com/mulhamfetna/trading-strategy-finder/main/subprojects/Parametric-Indicators/shareable/orb_study_bundle/docs/WS-ORB-PREREGISTRATION.md) are dated August 23, 2026. Confirmation is 2018–2024; exploration is 2010–2017; fresh observations end August 7, 2026. The grid’s +$1.57M gross / −$6.49M after $25 round-trip costs is descriptive, not a pooled portfolio test. Besides zero passing cells, there are 39 positive after-cost cells, 138 underpowered cells and 28 powered negatives. The report admits roll-day contamination and silver-session/DST uncertainty. One contract per trade is modeled. Licensed observations and claimed pre-registration chronology were not independently audited; no harness was run. This futures protocol is not a direct ETF replication.

## 4. Demirev: QQQ ORB replication, SSRN 5325376

**Primary registry abstract/identity checked on 2026-10-25; full methods remain inaccessible.** [Crossref](https://api.crossref.org/works/10.2139/ssrn.5325376) confirms Steve Demirev, the requested title and DOI. Its abstract contains the 672.71% historical replication claim, optimized extensions and a statement that code/data are provided; no usable repository or full manuscript was obtained. Posted year is 2025; DOI creation is June 27, 2025 and metadata redeposit is August 14, 2026. These are not manuscript-version dates. The earlier January 2025 date came from a secondary record and is not adopted as verified. Registered claims are not reproduced returns or new OOS evidence.

## 5. García Arano: Gao-rule Substack replication

**Full primary article read.** Carlos García Arano’s [September 30, 2026 post](https://deadsignalslab.substack.com/p/replication-intraday-momentum-eight) reports net losses on four ETFs at 2 bp round-trip cost. Its pooled common-panel 2014–2026 mean is −2.30 bp/day, day-clustered t −5.17. However, it describes a 09:30–10:00 signal rather than explicitly including the prior close, and incorrectly labels the original sample approximately 2004–2013. It links no implementation or raw artifact; EODHD observations are licensed. This is a non-peer-reviewed author claim, not established replication of either the original anchor or our long-only variant.

## 6. Original QQQ ORB paper: correct identifier and operational mismatch

**Full primary paper read.** Zarattini–Aziz, [original QQQ paper](https://concretumgroup.com/wp-content/uploads/2026/02/Can-Day-Trading-Really-Be-Profitable.pdf), first April 24, 2023; version September 22, 2025. The [author index](https://concretumgroup.com/papers/) identifies SSRN **4416622**. Its approximately 675% QQQ return (table: 676%) versus 169% passive return covers 2016–February 17, 2023. Both directions, up to 4x leverage, $0.0005/share commission and zero slippage are modeled. Entry follows the first candle’s direction at the second five-minute open; stop is the opposite first-candle extreme, target 10R or session close. Its TQQQ extension adds product leverage. These are in-sample author results, not evidence for our unleveraged long-only rule.

## Remaining gaps and decision boundary

No source is labelled fabricated merely because it was blocked. Demirev’s identity and registered result claims are now confirmed, while full methods and actual returns remain unreproduced; full methods remain unresolved for Delgado; the Fetna SSRN paper was blocked, but its underlying technical report and registration were read. The raw returns behind all replications remain unreproduced. Public-text fingerprints and the arithmetic self-check are saved in [_intraday_source_verification.json](_intraday_source_verification.json). Source consistency, cost assumptions, dividend conventions, leverage and direction differ enough that their Sharpe ratios cannot be pooled or transferred to the owner's strategy.

The defensible conclusion is that the literature raises cost and regime-risk concerns and warrants a carefully specified independent screen. It does not establish a profitable edge, and it also does not justify claiming that every proposed intraday rule has already been decisively refuted.

## 7. Final Gao access follow-up (2026-10-25)

The owner’s Perplexity access audit supplied the author-linked final PDF. We independently verified the [academic-site link](https://sites.google.com/site/szlwebpage), downloaded a valid 21-page typeset PDF, matched its DOI and checked the methods and tables cited in the feasibility report. This closes access, not exact implementation equivalence or raw-return reproduction.

The separate [Zhou appendix](https://guofuzhou.github.io/Appendix_IntraDay.pdf) is accessible and has **14 PDF pages including its cover**, not 13 total pages. Its revision date is not established. Crossref independently confirms Delgado’s title/author/DOI and abstract, but neither outstanding replication full text was obtained. See [dated fingerprints and metadata](_intraday_full_text_verification_2026-10-25.json). No protocol choice or probe was approved by these access findings.
