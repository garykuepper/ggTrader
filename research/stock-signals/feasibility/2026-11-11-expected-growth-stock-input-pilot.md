# Expected-growth stock-input feasibility pilot

**Status: data-feasibility design.** The purpose is to find out whether the published
expected-growth idea can be reconstructed for individual stocks using information
available at the time. This pilot does not calculate stock returns, select stocks to
buy, authorize paper trading or change the production freeze. A later return study
requires its own dated specification and owner approval.

The [published-portfolio result](2026-11-11-expected-growth-results.md) is a reason to
investigate, not a list of holdings. Its recent comparison was with a provider market
portfolio, not SPY, and depends on an assumed cost. The 2023–2025 listing snapshots
were identified after that result; they are not an unseen performance sample.

## Decision checkpoint

**Additional free timing lead (metadata only, 2026-11-12):** [FinzData's
catalog](https://finzdata.com/datasets) describes a free earnings-dates endpoint
for SEC Form 8-K Item 2.02 filings since 2004, with accession, acceptance time and
before/during/after-market labels. This could help locate original filing evidence
for a bounded timing cross-check. An 8-K acceptance time is not necessarily the
first public earnings announcement, and coverage, time-zone interpretation,
accuracy and usage rights have not been independently verified. The provider's
[valuation documentation](https://finzdata.com/valuation-data) limits free daily
market-cap history to the last 12 months, so it does not supply the required
multi-year training history for free. No API key was requested, no provider rows
were read, and no signal or return was calculated.

The earnings-date access period itself needs confirmation: the catalog calls
earnings dates free and describes coverage from 2004, but another [provider
page](https://finzdata.com/compare/nasdaq-data-link) describes the free tier as
covering only the last 12 months of every dataset. These statements do not
establish that old earnings records are available without payment. Verify the
endpoint's actual free history and applicable terms before planning around it;
the source does not resolve the historical-input gap on the evidence available.
The provider's [public sample page](https://finzdata.com/samples) describes
ten-row intraday price samples, not earnings-filing timing samples, so those
previews cannot resolve the earnings endpoint's historical coverage or timing.
Its [API quickstart](https://finzdata.com/docs) instructs users to register for a
key and authenticate requests; no no-account historical earnings sample was
identified in that documentation. This is not proof that none exists.

The portfolio screen cannot identify today's stocks. Before proposing an individual-stock
return test, establish these inputs together at the same decision times:

1. **A usable universe:** dated security identities, primary exchange and eligible
   share classes, including delisted securities; verify source rights and timing.
2. **Training history:** at least 30 prior monthly fits from point-in-time inputs.
   The investigated archive's optimistic calendar bound allows at most 26 fits
   before April 2024 after the 12-month predictor lag; no fits have been verified.
3. **Firm inputs:** accounting facts and market equity with verified availability,
   fiscal alignment and share-class treatment. Document missing and ambiguous data.
4. **Execution and loss handling:** a separately specified tradable universe,
   benchmark including SPY, costs, corporate actions and exit assumptions.

The April 2024 example below remains a data diagnostic. It is not a ready forecast
or an approved return study. The evidence for these constraints is detailed below.
The experimental fourth-quarter derivation also rejects an all-zero issuer ID;
this check does not verify that a nonzero ID belongs to the intended security.
Its output preserves the two supplied source periods, values, accessions and
availability times for audit; it does not independently verify those sources or
whether their accounting versions are comparable.

**Sort-method cross-check:** The provider's [2023 factor construction document](https://global-q.org/uploads/1/2/2/6/122679606/factorstd_2023feb.pdf)
describes independent monthly NYSE size and expected-growth sorts, followed by
their intersection, with value-weighted monthly portfolios. This supports the
prototype's use of a growth breakpoint across the supplied eligible NYSE reference
set rather than recalculating it only among large stocks. It does not verify the
prototype's interpolation or tie handling, the eligible panel, security-level
weights for multiple share classes, or the exact build of the 2026 portfolio
release. The document also describes a beginning-of-month formation convention;
it does not establish that a replica can execute at those prices using only
information available beforehand. No stock selection or returns were calculated.

**Independent size diagnostic:** [French's NYSE market-equity breakpoints](https://mba.tuck.dartmouth.edu/pages/Faculty/ken.french/Data_Library/det_me_breakpoints.html)
include a monthly median based on eligible NYSE common shares. Their population
is not the expected-growth population: French excludes REITs and closed-end funds,
whereas the q-factor document specifies excluding financial firms and negative
book equity. Consequently this is a plausibility check, not a substitute for the
model's own size breakpoint or missing security-level market values. In the
previously recorded [vintage comparison](../../programme/RESEARCH_GUIDE_DETAILS.md),
29 of 36 monthly medians in 2021–2023 differed between the archived 2024 and 2026
files, with a maximum difference of 2.20% relative to the earlier value. The
[raw comparison](_size_breakpoint_vintages_20261111.json) records the inputs and
scope. Neither vintage proves what was available at a historical decision time;
differences in the median do not establish changes in selected stocks or returns.

**Forecast month boundary:** The experimental `forecast` helper explicitly labels
fit and decision months in UTC. In a [synthetic check](_forecast_utc_month_boundary_20261111.json),
29 earlier fits plus one April fit failed its 30-fit requirement at 19:59 New York
time on April 30, 2024, but passed at 20:01 that evening, when UTC had entered May.
That is consistent with the helper's documented UTC convention; it is not a claim
that the two-minute difference affected a real decision. If a future study defines
formation months by New York calendar dates, this convention can admit its current
local month's fit prematurely. Specify and test the intended calendar and fit
availability before using the helper. The pilot's specified April 29, 2024,
16:00 New York cutoff is April 29 in both time zones, so this particular month
boundary does not affect that cutoff. No stock forecast was calculated there.

**Related research lead:** Yuan Tian's [2023 conference poster](https://www.aeaweb.org/conference/2023/program/paper/hGDetzfF)
describes forecasting one-year-ahead investment using 43 predictors and a rolling
60-month training window, and reports lower out-of-sample forecast error for
gradient boosting than a conventional linear model. The [later published paper's
available description](https://www.sciencedirect.com/science/article/pii/S0927539826000952)
uses CRSP and Compustat and reports a high-minus-low return spread. Neither a
forecast-error improvement nor a long–short spread establishes an investable
long-only edge over SPY after costs and taxes. This is a related alternative, not
independent confirmation of the selected three-predictor model, and the 43 inputs
do not resolve the current identity, dated-universe and market-value gaps. The
publisher's available text says model selection used
forecast accuracy rather than returns, a useful separation if implemented as stated.
Neither source excerpt establishes point-in-time input vintages, investable costs,
taxes, or reproducible current holdings. The poster's forecast sample ends in 2020;
its validation window should not be assumed to describe the later paper unchanged.
The publisher's description of its four-model comparison reports that changing
only the model class or only the predictor set produced a small increase in the
high-minus-low return spread; the larger reported increase required both gradient
boosting and the larger predictor set. This is a published long–short comparison,
not evidence that replacing the existing linear model alone will improve a
long-only portfolio, and the underlying return series has not been located.

The author's [August 2022 working paper](https://mcusercontent.com/dbb77fe8dfbbb09d4ad3e861b/files/a9cfdc8f-71fd-c437-5091-0ba4d846e365/Yuan_Tian.pdf)
specifies that, at month-end `t`, its 60-month training predictors run from `t−71`
through `t−12`, with one-year-ahead targets through `t`; forecasts use predictors
known at `t`. It also describes a four-month accounting lag, an earnings-announcement
assumption for quarterly revenue and earnings, and zero imputation for 16 annual
change variables. The paper describes five training-data subsets for tuning, but
the inspected passage does not say how observations are assigned to them or whether
overlapping outcomes are separated. The calendar lag and announced-date assumptions
do not themselves establish historical dissemination or accounting vintages.
These observations are about the 2022 working paper; the full 2026 published
methodology and release-to-code binding have not been checked. No returns were
calculated.

For the long-only question, Table 5 of that working paper reports an average
**0.89% monthly return above the risk-free rate** for its high-score decile from
July 1969 through December 2020. The sort uses NYSE breakpoints, value weighting
and monthly rebalancing. This is an arithmetic average from an already studied
period, not a compounded return, a SPY comparison or an after-tax result. The
reported high-minus-low spread also depends on the low decile and is not a
long-only benchmark. Trading costs, turnover and executable prices remain
unverified; the table alone cannot establish whether this would beat SPY or the
project's existing expected-growth screen. No new portfolio test was run.

Table 7 also reports high-score decile arithmetic excess returns of **0.81% per
month** for six-month cohorts and **0.75% per month** for twelve-month cohorts,
over the same stated period. Each result averages overlapping value-weighted
cohorts and rebalances the aggregate monthly; it does not describe a single
buy-and-hold position or a rule that lets individual winners run indefinitely.
These are risk-free-rate comparisons from the historical paper, not SPY-relative,
after-cost or after-tax performance. The work predates the later publication and
does not establish that the longer-holding results survive implementation costs.

**Alternative research lead:** The [original paper](https://theinvestmentcapm.com/uploads/1/2/2/6/122679606/houmoxuezhang2020rf.pdf)
also describes a composite of percentile-ranked valuation, operating cash flow and
ROE change, oriented according to their predicted return relationship. Unlike the
regression-based expected-growth forecast, that construction does not require the
prior 30 fitted regressions. The authors report a historical long–short factor,
not a long-only stock list or a comparison with SPY. This is a distinct, previously
published candidate, not a replacement for the selected model or evidence of an
implementable edge. Dated universe, eligible securities, accounting definitions,
public availability, rank conventions, costs and survivorship still need verification.
Any stock-level return test needs its own specification and owner approval.

The university-hosted [supplementary tables](https://qa.jhucareyredesign.com/sites/default/files/2023-01/finance-conference-june-2019-lu-zhang-paper-min.pdf)
provide a more specific starting point. Table A.4 describes monthly value-weighted
deciles formed with NYSE breakpoints; its highest-scoring decile reports a 1.16%
average monthly excess return in 1967–2018. That is an arithmetic average over an
already studied period, not annualized compounded growth or a SPY comparison;
trading costs and taxes are not established by that table. Table A.5 uses independent size and composite-score
sorts, then averages high-minus-low returns across the size groups. These are
different constructions; the high decile is not the large-stock high-score portfolio.
The paper does not here resolve rank direction by variable, ties, missing values,
trade timing or the rights and point-in-time coverage of stock-level inputs. No
portfolio calculation or recommendation follows from transcribing these tables.

A possible component cross-check is Open Source Asset Pricing's `CBOperProf`,
which appears in the inspected predictor schema described later in this note.
Its [pinned construction code](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/Predictors/CBOperProf.py)
uses the same broad cash-flow numerator and assets denominator as the paper, but
it fills missing revenue, cost of goods and SG&A levels with zero before calculating
changes. The ggTrader prototype instead requires those three levels; its handling of
missing annual changes also needs to be compared to the provider's lagged-level
procedure. Thus `CBOperProf` is a candidate for a matched, fully observed-input
cross-check, not a drop-in substitute. The exact code-to-data build binding,
formation dates, entity mapping and actual values remain unverified. Its presence
does not supply the missing valuation and ROE-change components or an NYSE
reference universe. No predictor rows or returns were examined in this comparison.

The same dataset's [pinned `RoE` construction](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/Predictors/RoE.py)
uses annual net income divided by annual common equity. The expected-growth
paper instead specifies quarterly income before extraordinary items divided by
the previous quarter's book equity, then subtracts the ROE from four quarters
earlier. Taking the year-over-year difference of this dataset's `RoE` column would
therefore not reproduce the paper's predictor. This is a comparison of published
code and definitions, not a check of the corresponding data release or values.
The dataset's [quarterly `roaq` code](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/Predictors/roaq.py)
uses quarterly income over lagged **assets**, so it also is not the required ROE.
Neither observation rules out constructing the paper's predictor from another
source; the exact signal build and corresponding data values were not checked.

On 2026-11-11 a bounded query for PERMNO 12490 and April 2024 through the provider's
Hugging Face dataset viewer failed twice with HTTP 500 and an authentication-check
timeout; it returned no rows. This is an API failure, not evidence of missing IBM
coverage. The provider documents column-selective Parquet reads as another route,
but the local environment lacks the required `fsspec` package. Any later attempt
should pin the data revision, verify the security mapping and preserve provenance;
the viewer's current default release was not pinned. No package was installed.

A follow-up column-selective read from the pinned revision succeeded using a
temporary `uv` environment. It returned `CBOperProf` approximately **0.153274**
for PERMNO 12490 in both March and April 2024; the [probe record](_osap_ibm_cboperprof_probe_20261111.json)
records the values, source, runtime and limits. This is the provider's retrospectively
published signal, not a verified calculation from contemporaneous SEC inputs. The
provider describes observations as usable at month-end: the April observation is
therefore later than the April 29 pilot cutoff. The previously cited IBM mapping does
not independently prove this identifier's identity in either 2024 month. No return
or ranking was calculated, and the required NYSE reference universe is still missing.

The pinned provider [release notes](https://huggingface.co/datasets/OpenSourceAP/CrossSection/resolve/dd1054254c4501776432caa6d7ce5ecd9586f6a9/Release%20Notes%202026.10.md)
say the October 2026 data was created with code version 2.1.0. The provider's
currently resolved `v2.1.0` tag points to the same `9f053398...` commit cited for
the construction code above. This is a publisher-attested version association, not
an independently reproduced build or proof of the per-signal inputs. The current
tag target's commit date is after both the data-release commit and release-notes
commit; tag identity at build time is not independently established. However, a
repository snapshot dated before the initial data-release commit has identical
SHA-256 hashes for the two cited predictor files and the annual-data preparation
file. That verifies these specific files' observed contents at the two repository
revisions, not the exact build's inputs or a trusted timestamp. The [metadata
record](_osap_code_release_binding_20261111.json) preserves the dates and hashes.
Individual values, accounting mappings and 2024 availability remain unverified.

**Current-use limitation:** The provider's [October 2026 release page](https://www.openassetpricing.com/data/)
says most of its current data run through December 2025, with some signals ending
earlier. It does not establish the last available observation for every predictor,
but the release description is insufficient to supply a current November 2026
stock screen. A current screen needs separately sourced, timely inputs and the
same identity, rights and availability checks; historical availability cannot be
inferred from the release date. This is a coverage-metadata check, not a signal or
return calculation.

A bounded read of the pinned file for PERMNO 12490 found non-null `CBOperProf`
for October–December 2025, but null values for each labeled month from January
2026 through June 2027. The [record](_osap_ibm_cashflow_coverage_20261111.json)
captures this one-security check. Future-dated row labels do not make a predictor
available, and this example does not establish coverage for other securities or
fields. Its identity and source-vintage limitations are the same as above.

One additional component agrees numerically. The provider's March 2024 `DelDRC`
for PERMNO 12490 is approximately **−0.00374936**. Applying its [published formula](https://github.com/OpenSourceAP/CrossSection/blob/c626bf886f53894e4b7f7dedf2c167d1c84b7801/Signals/pyCode/Predictors/DelDRC.py)
to IBM's [2022 balance sheet](https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/ibm-20221231xex13.htm)
gives the same displayed value: current deferred income changed from $12,518 million
to $12,032 million, divided by average assets of $132,001 million and $127,243
million. The [record](_osap_ibm_deldrc_20261111.json) pins the inputs and code.
This is evidence consistent with those candidate values and periods, not proof of
the provider's underlying inputs, CIK linkage or their availability in March 2024.
`DelDRC` covers current deferred income and uses average assets; `CBOperProf` also
uses noncurrent deferred income and other components. The broader mismatch remains.

The provider's March 2024 `AssetGrowth` is **−0.03604518**, matching the change
from $132,001 million to $127,243 million in IBM's reported assets using the
[published formula](https://github.com/OpenSourceAP/CrossSection/blob/c626bf886f53894e4b7f7dedf2c167d1c84b7801/Signals/pyCode/Predictors/AssetGrowth.py).
The [comparison record](_osap_ibm_asset_growth_20261111.json) supports a consistent
relative change, but the ratio cannot establish the absolute asset amounts, the
denominator used for `GP`, the issuer linkage or the inputs' availability in 2024.

A later [bond-linker name check](_bond_linker_ibm_name_check_20261111.json) found
PERMNO 12490 in a retrospectively published dated-name file, with a name consistent
with International Business Machines and a reported name spell covering March and
April 2024. The [publisher's documentation](https://github.com/Alexander-M-Dickerson/trace-data-pipeline/blob/main/stage1/DATA_DICTIONARY.md)
describes the file as a dated PERMNO-to-firm-name reference; the bundle attributes
its name information to CRSP. The bundle says using it requires no entitlement,
but its inputs are licensed and the applicable use and redistribution rights were
not independently assessed. A name match corroborates the IBM lead but does not
prove the SEC CIK or security class, establish 2024 availability, or guarantee an
independent source from the other CRSP-based mapping. The small name file was read
through HTTP ranges and not retained; no stock ranking or return was calculated.

For a diagnostic comparison, selected candidate tags from IBM's 2022 annual filing
give a partial cash-flow numerator of $21,343 million, or 0.167734 of reported assets,
when the unresolved accrued-expense change is set to zero solely for arithmetic.
The provider's March value implies a numerator $1,840 million lower if that same
denominator and all other assumptions are used. This residual does **not** establish
the accrued-expense change or explain the difference: the provider's actual fiscal
period and inputs, tag mappings, share identity and data vintages are unverified.
IBM's filing separately reports other accrued expenses of $4,111 million and $3,892
million, whose $219 million change alone does not match the residual; it is not
verified as the Compustat accrued-expense definition. The [input and arithmetic record](_ibm_cash_flow_crosscheck_20261111.json)
preserves this diagnostic. No stock ranking or return was calculated.

The provider's March 2024 `OperProfRD` value for the same PERMNO is approximately
**0.151238**. The [pinned construction code](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/Predictors/OperProfRD.py)
describes the operating numerator before the cash-based working-capital adjustments.
The selected SEC-tag operating numerator is $20,646 million. **If** the provider's
denominator is the same $127,243 million of assets, its ratio implies an operating
numerator of $19,244 million, $1,402 million lower. Under the additional assumption
that both provider measures use compatible inputs and the pinned definitions, their
difference implies a $259 million working-capital adjustment; the incomplete SEC
candidate implies $697 million when the unknown accrued-expense change is set to
zero. Thus the original $1,840 million gap cannot be attributed solely to accrued
expenses under these assumptions. The [diagnostic record](_osap_ibm_operating_decomposition_20261111.json)
preserves the values. The denominator, fiscal period, input mappings, exact build
binding and accounting vintages remain unverified; different missing-data rules and
filters also matter. This is not an identified cause or an investable signal.

**Important mapping caveat:** IBM's [income statement](https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/ibm-20221231xex13.htm)
presents SG&A and research, development and engineering as separate expense lines.
An older [Compustat data guide](https://tevgeniou.github.io/EquityRiskFactors/CompustatManualChpt5.pdf)
describes its SG&A item as including research and development, and separately notes
that its R&D item generally excludes engineering. Therefore the diagnostic above,
which directly substitutes IBM's separate SG&A line and then adds its entire RD&E
line, may add expenses that the provider's combined SG&A already contains. It is
an arithmetic illustration, not a verified Compustat-equivalent numerator. For
scale only, if the whole $6,567 million RD&E line were included in the provider's
SG&A and the same amount were the R&D add-back, the candidate operating numerator
would be $14,079 million, not $20,646 million. That assumption is not verified for
IBM or the provider's 2024 release, and it does not explain the observed gap. Obtain
the actual field definitions, components and vintages before attributing any residual.
IBM's report separately describes $6,267 million of scientific research and
development and $299 million of product engineering within the $6,567 million
reported RD&E total; the displayed rounded components differ from the total by
$1 million. Those categories are a useful mapping lead, not proof that either is
the provider's R&D field or that the older guide applies unchanged to its release.

The provider's March 2024 `GP` for the same PERMNO is approximately **0.289006**;
its [published code](https://github.com/OpenSourceAP/CrossSection/blob/c626bf886f53894e4b7f7dedf2c167d1c84b7801/Signals/pyCode/Predictors/GP.py)
defines gross profit as revenue less cost of goods, divided by assets. The selected
SEC values give `(60,530 − 27,842) / 127,243`, or approximately **0.256894**.
If the provider uses that same asset denominator, its ratio implies a gross numerator
of $36,774 million, versus $32,688 million from the selected SEC-tag values: a
$4,086 million difference before SG&A or R&D. IBM's statement displays gross
profit of $32,687 million; the $1 million difference from subtracting the selected
rounded inputs illustrates a separate rounding limitation. The [record](_osap_ibm_gross_profit_20261111.json)
preserves the comparison. The provider's denominator, fiscal period, revenue/cost
definitions and data vintage remain unverified, so this difference identifies a
mapping question, not a cause or a reliable stock signal.

One possible definition difference has documentary support but is not quantified
for this comparison. The older [Compustat guide](https://tevgeniou.github.io/EquityRiskFactors/CompustatManualChpt5.pdf)
describes COGS as excluding depreciation allocated to cost of goods and certain
intangible amortization, with specific exceptions. IBM's filing says some
capitalized software amortization is included in reported software cost. The saved
SEC facts also report $2,407 million of depreciation and $2,395 million of intangible
amortization for 2022, but those totals are not an allocation to the reported cost
line or verified Compustat adjustments. Neither total can be subtracted from the
$4,086 million difference as a demonstrated reconciliation. Actual provider COGS,
its depreciation treatment and the original field vintage remain needed.

The provider's [pinned annual-data preparation code](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/DataDownloads/CompustatAnnual.py)
delays annual observations six months after their report month and carries each
across twelve monthly signal dates. If the inspected data release used this code
and IBM has an ordinary December year-end without an intervening calendar change,
its March and April 2024 signals would use December 2022 annual inputs, consistent
with the diagnostic's chosen period. This is a conditional code inference, not
verification of the actual source row or its values. The delay also differs from
the expected-growth paper's four-month rule, so the series cannot be treated as
an exact implementation at every formation date. Six months is a convention,
not proof that every filing was public or that later revisions were absent.

For the seven selected 2021 balance-sheet tags in that diagnostic, the saved SEC
company-facts response reports identical values in IBM's original 2021 annual
filing and its 2022 annual filing's comparative column. The two accessions and
values are recorded in the [diagnostic artifact](_ibm_cash_flow_crosscheck_20261111.json).
This rules out a difference between those two presentations for those tags, not
intervening revisions, differences in Compustat mapping, or the other components.
In particular, the two reported `Other accrued expenses and liabilities` balances
are not established as the vendor's `XACC` inputs, so the residual remains unexplained.

The [Research Insight North America data guide](https://tevgeniou.github.io/EquityRiskFactors/CompustatDataGuide.pdf)
describes `XACC` as including items such as accrued salaries and pensions, while
excluding other liabilities and deferred compensation; a combined-figure code can
indicate that accrued expenses were included in another item. The guide is an older
hosted copy, not proof of the provider's 2024 mapping or missing-value treatment.
IBM's [2022 filing](https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/ibm-20221231xex13.htm)
reports compensation and benefits separately, rising from $3,204 million to $3,481
million, and says its combined other-accrued line contains derivative liabilities
of $60 million and $546 million in the two respective years. Those disclosures show
why neither the combined line alone nor a simple addition of whole balance-sheet
lines establishes `XACC`: their components need classification and the vendor's
actual observations remain unknown. They do not explain the $1,840 million residual.

**Next actionable check:** Resolve modeling rights for a candidate source before
using it to construct panels. An [unsent permission draft](2026-11-11-listing-data-permission-draft.md)
sets out the specific intended uses and asks the rights holder to distinguish
raw data, derived results and AI-assisted processing. If permitted, build and independently verify two
consecutive dated input panels, including primary exchange, issuer and share-class
identity, market-equity basis and contemporaneous filing availability. Then assess
whether the same process can provide the 30 prior fits; a count of archive months
is not a count of valid fits. The January 2021 archive snapshot was committed too
late for a beginning-of-month panel, so its conditional earliest 30-fit decision
is August 2024. Three older 2020 files could improve that bound only if their rights,
timing, completeness and rounded market values prove adequate. Until then, the
April 2024 panel is not sufficient for a stock-level expected-growth forecast.

**Older accounting lead and lookahead hazard:** The SEC's separate [Financial
Statement Data Sets](https://www.sec.gov/dera/data/financial-statement-data-sets.html)
cover XBRL submissions from April 15, 2009 onward; its [documentation](https://www.sec.gov/files/financial-statement-data-sets.pdf)
says the nominal 2009 Q1 file has headers only. These filing extracts may help
extend accounting history, but do not supply the dated NYSE universe or market
equity needed for the regressions. The SEC reprocessed the datasets in December
2024, so a current extract is not itself proof of what the extracted dataset
contained on an earlier date. Its `prevrpt` field indicates a submission was
*subsequently* amended. Filtering historical observations using that present-day
flag would use later information; select individual original and amended filings
by independently verified availability at each decision time instead. Acceptance
timestamps in the extract do not by themselves prove public dissemination. This
is a documentation review, not a coverage or vintage validation of any file.

The separate locally saved **Financial Statement and Notes** 2021 Q2 extract
illustrates the risk. Its `sub.tsv` marks Iteris's June 1, 2021 annual filing
`prevrpt=1` and includes a same-issuer, same-period 10-K/A filed June 7. Its
[documentation](https://www.sec.gov/files/aqfsn_1.pdf) defines the flag using
amendments before the dataset cutoff. The [spot-check record](_sec_prevrpt_example_20261111.json)
identifies the archive hash and accession numbers. The pair is consistent with
the warning; matching issuer and period alone does not independently prove the
amendment's relationship. Both submissions report the same unsegmented 2021-03-31
`Assets` value, $126,869,000, in this extract; a flag alone does not mean this
particular input changed. The check does not compare the other facts. The extract's
acceptance time is not proof of public availability at an intraday cutoff.

Another possible source, [JANalytics' US stock universe](https://github.com/JANalytics/us-stock-universe),
describes a current-company dataset and a file of changes since its prior refresh.
It draws listing information from Nasdaq Trader directories and describes its SEC
public-float measure as potentially up to a year old and excluding insider-held
shares. Those fields are not a verified historical listing universe or dated total
market equity, and the project does not remove the upstream rights question.
Its historical coverage, timing, accuracy and terms were not verified; no source
rows or returns were examined.

**Market-equity cross-check lead:** The [Apex project](https://github.com/cosmin-novac/apex)
describes a monthly S&P 500 market-cap panel built from SEC share disclosures,
FINSABER prices and separately reconstructed index membership. Its stated method
adjusts share counts for splits and flags price stitching repairs. This may offer
a narrow comparison for individual share-count calculations, but an S&P-only
panel cannot establish the full NYSE breakpoints. The repository license does not
by itself settle rights to underlying data; effective versus publication dates,
share classes, restatements, split handling, coverage and source provenance need
independent checks. Existing FINSABER concerns are documented in the [price-source
review](2026-11-02-free-data-stock-feasibility.md). No panel rows or returns were
read, and the project's performance claims were not assessed.

**Timing limitation found in construction code:** The [inspected revision](https://github.com/cosmin-novac/apex/blob/61626b30efca51b6a2cd83dace688ad89ac19af2/tools/build_megacap_panel.py)
indexes XBRL share observations by the reported end date rather than the filing
date, and fills months before the first observation backward from that observation.
It also applies a centered outlier filter using neighboring observations. These
operations can depend on information unavailable at a past decision date. The
output flags some pre-observation rows as `backfill`, but that flag alone does not
address report-date versus filing-date timing or the centered filter. This code
review does not quantify affected panel rows, prove the inspected revision built
the distributed file, or validate an individual company. Do not treat the panel
as point-in-time evidence without rebuilding or independently checking these steps.
The inspected writer exports `src` labels such as `xbrl` and `backfill`, but not
the selected filing accession or filing date in each panel row. Its separate
[metadata file](https://raw.githubusercontent.com/cosmin-novac/apex/61626b30efca51b6a2cd83dace688ad89ac19af2/data/megacap_meta.json)
describes sources and repairs; it does not establish per-row filing availability.
Filtering out `backfill` alone is therefore insufficient to verify the remaining
rows' timing. This is a schema and code inspection; the distributed panel was not
read and no affected-row count was estimated.

A separate [SEC-extract date check](_sec_share_count_date_gap_20261111.json) compared
the `ddate` field with filing dates in the saved 2021 Q2 Financial Statement and
Notes archive. **Correction:** The archive's own readme says `ddate` is rounded to
the nearest month-end. The previously reported ten-day median is a gap from that
rounded date, not the median gap from the actual share-count date. Of the matching
submissions, 953 had only rounded dates after their filing date; this is not a
count of invalid or future-dated reported facts. The archive was later processed
and the sample is neither a complete historical universe nor a share-class check.

For example, the extract labels AMD's share count in its April 28 10-Q April 30,
while the [filing](https://ir.amd.com/financial-information/sec-filings/content/0000002488-21-000056/amd-20210327.htm)
and its original inline-XBRL context date the same 1,215,020,976 shares to April 23.
The extract's `datp` is 7.0; this archive's readme describes it as a date proximity
in days, without specifying direction. Verify exact original contexts and public
availability instead of treating a rounded date or filing date as sufficient proof.

As a narrow alternative, a saved [SEC company-facts record](_sec_ibm_companyfacts_share_date_20261111.json)
for IBM preserves an `end` date of February 10, 2021 and 893,594,090 shares; its
referenced [10-K](https://www.sec.gov/Archives/edgar/data/51143/000155837021001489/ibm-20201231x10k.htm)
reports the same as-of date and count. The record's `frame` is `CY2020Q4I`, so the
frame label must not replace the exact `end` date. This one match supports a source
for further checks, not a verified historical universe, security-class mapping or
public-availability time. Company-facts revisions and missing values still require
independent treatment; no historical market cap or return was calculated.
The existing [instant-fact helper](../_lib/sec_facts.py) reads only the `us-gaap`
namespace, so it does not ingest this `dei:EntityCommonStockSharesOutstanding`
fact. The separate [market-equity helper](../_lib/listed_market_equity.py) expects
caller-supplied, already verified share-class rows; it does not bridge that gap.
Any future importer must preserve the original end date and accession, require
independently established availability, and resolve issuer and class identity.
Adding the `dei` field alone would still miss a concrete case: the saved Berkshire
Hathaway company-facts file contains only seven observations for this concept,
the latest filed in 2011, while its [2023 annual report](https://www.berkshirehathaway.com/2023ar/2023ar.pdf)
separately reports Class A and Class B shares as of February 12, 2024. The
[coverage record](_berkshire_companyfacts_shares_gap_20261111.json) pins the file
and observed counts. The SEC says its [company-facts API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
aggregates facts applying to the whole entity; whether dimensional treatment or
another cause explains this specific gap was not established. Do not substitute
zero or assume the issuer has no listed stock when a value is absent.
A [third-party rendering of the same filing](https://last10k.com/sec-filings/1067983/0000950170-24-019719.htm)
groups the two reported counts under separate Class A and Class B members of the
shares-outstanding concept. Direct inspection of the [original 10-K's inline
XBRL](https://www.sec.gov/Archives/edgar/data/1067983/000095017024019719/brka-20231231.htm)
confirms `dei:EntityCommonStockSharesOutstanding` facts of 566,618 and
1,310,805,008 with an instant of February 12, 2024 and separate
`us-gaap:StatementClassOfStockAxis` members `CommonClassAMember` and
`CommonClassBMember`, respectively. This establishes the original fact contexts
for this filing; the [context record](_berkshire_original_share_contexts_20261111.json)
also lists the 2021 facts. It does not prove why company-facts omitted them. The 10-K also
reports Class A-to-B conversions and repurchases during 2023. Consequently,
an old class count cannot be assumed current merely because its split basis is
correct; a future importer needs a documented freshness and intervening-change
policy as well as class-specific prices. The filing alone does not establish the
dates or amounts of all changes between a reported count and a later decision.

The SEC's [filing index](https://www.sec.gov/Archives/edgar/data/1067983/000095017024019719/0000950170-24-019719-index.htm)
records this 10-K as accepted on February 26, 2024 at 06:08:31. Berkshire's
[February 20 announcement](https://berkshirehathaway.com/news/feb2024.pdf)
scheduled release of the 2023 annual report for approximately 8:00 a.m. Eastern
on February 24, and its [February 24 release](https://www.berkshirehathaway.com/news/feb2424.pdf)
said the report had been posted. The issuer's currently available
[annual report PDF](https://www.berkshirehathaway.com/2023ar/2023ar.pdf) contains
the two February 12 counts. An [Internet Archive capture at 14:53:37 UTC on
February 24](https://web.archive.org/web/20240224145337id_/https://berkshirehathaway.com/2023ar/2023ar.pdf)
also contains those same counts. The retrieved capture is pinned in the
[context record](_berkshire_original_share_contexts_20261111.json); a later
15:19:14 UTC capture had identical bytes. This supports availability through the
issuer site by the earlier capture time, before SEC acceptance, subject to trust in
the archive's timestamp. It does not establish the first publication instant or
justify using the planned 8 a.m. release time as an exact timestamp. SEC acceptance
is not necessarily the first moment information became public.

The saved 2021 Q2 SEC notes archive (`2021q2_notes.zip`, SHA-256
`fbbc99dff34cdd1b898a7e5b34b66b89fcf313bc4a620b2248a9ae663cb97d22`)
provides a concrete alternative to investigate:
for Berkshire's 2021 first-quarter 10-Q, accession `0001564590-21-022703`, its
`num.tsv` contains separate `dei:EntityCommonStockSharesOutstanding` rows and its
`dim.tsv` identifies them as `ClassOfStock=CommonClassA` and `CommonClassB`. The
respective counts, 637,990 and 1,326,572,128, match the cover of the
[original filing](https://www.sec.gov/Archives/edgar/data/1067983/000156459021022703/brka-10q_20210331.htm).
The original inline-XBRL contexts also identify those two class members and an
instant of April 22, 2021, while both archive rows carry the rounded `ddate`
April 30 and `datp` 8.0. The archive lists the filing date as
May 3; this is not evidence the counts were publicly available on April 22 or 30.
This single match supports investigating the dimensional archive, but does not
establish complete historical coverage, availability times, current shares,
security mapping or usable market equity. In particular,
the archive's rounded date and proximity field must not be silently converted into
an exact date without independently verified semantics.
The 2021 filing's `BRK.A` and `BRK.B` ticker facts have the corresponding class
members, but their XBRL contexts cover January 1 through March 31; the share facts
are instants on April 22. The same dimension hash therefore matches the class
segments while hiding a difference in period context. Preserve each fact's own
period and independently check the mapping and information timing; do not treat
the hash match as proof that ticker and share data are simultaneous.

In a broader [metadata count](_sec_dimensional_share_coverage_20261111.json) of
the same archive, 6,481 of 6,487 10-K and 10-Q submissions contained at least one
selected share-count fact. Of those submissions, 940 had dimensional facts but no
unsegmented fact; 16 had both. This is an accession count, not a count of stocks
or proof of usable class coverage: the check did not validate fact dates, values,
freshness, dimensions or availability. It does show that an importer restricted
to unsegmented facts would miss every selected fact in those 940 submissions.
For those 940, joining share facts to nonempty `TradingSymbol` facts on both
accession and exact dimension hash matched at least one share dimension in 288
submissions and every selected share dimension in 35. The other 652 had no such
exact-hash match. Berkshire's two dimensions matched `BRK.A` and `BRK.B`. These
are structural matches across the filing, not verified identities or simultaneous
facts; a mismatch does not prove a mapping is impossible. A future importer needs
an independently checked mapping for nonmatching dimensions and must preserve
dates and availability rather than rely on this join alone.

A narrower [fixture check](_sec_sp500_share_join_20261111.json) starts from the
505 labels in the frozen S&P snapshot dated June 4, 2021. Among 10-K and 10-Q
filings in the Q2 archive dated through June 28, 495 labels matched a ticker fact
and each had one observed CIK within this limited set. Of those labels, 456 also
had an unsegmented share-count fact in a matching filing, while 19 had a share
fact matching the ticker's exact dimension hash; 475 had either. The unsegmented
count may represent the issuer rather than the matched class, and neither match
establishes a usable value or public-availability time. Fixture labels can be
retrospective, other periods and forms were not checked, and the S&P fixture is
not the NYSE reference universe. These counts are feasibility leads, not coverage
estimates for the forecasting model. A second calculation reproduced these counts
from the same files; it was performed by the same analyst and is not an independent
source or identity review.
Across six already-saved company-facts files, a [small accession-and-value match](_sec_companyfacts_notes_date_matches_20261111.json)
found only four unique 2021 Q2 share-count pairs in the notes archive; two pairs
had different company-facts and month-end-rounded notes dates. This is a selected
micro-sample, not a coverage or error-rate estimate, and matching values alone does
not prove identical XBRL contexts. Check original filings for decisions.
For one pair, Coca-Cola's [original 10-Q](https://www.sec.gov/Archives/edgar/data/21344/000002134421000014/ko-20210402.htm)
and inline-XBRL context both date the count to April 23, 2021, matching
company-facts; the notes extract labels it April 30. This spot-check supports that
one match, not the correctness or availability of the other observations.

**SimFin feasibility lead:** Its [pricing page](https://www.simfin.com/en/prices/)
advertises seven years of fundamentals API history on the free tier but five years
of delayed bulk-download history; access and exact coverage were not tested. The
page lists detailed and as-reported bulk variants under paid plans, while its
[license text](https://www.simfin.com/en/commercial-license/) describes personal
research use for FREE/BASIC and restricts sharing data with other parties. Its
pricing FAQ says downloaded data must be deleted after subscription cancellation;
whether and how that applies to a free account, derived artifacts and AI-assisted
processing needs clarification. The provider describes a typical 24–48-hour
processing delay on its [data page](https://www.simfin.com/en/fundamental-data-download/),
not a historical timestamp guarantee. Historical revisions, delisted securities,
primary-exchange history, security classes and market-equity coverage are unverified.
This could be an accounting cross-check, not yet a reproducible dated universe; no
account was created, data downloaded, or paid subscription requested.

SimFin's [backtesting tutorial](https://www.simfin.com/en/simfin-screener-backtesting-tutorial/)
says its tool retains companies that traded in the past and were delisted. This
is a provider statement about the backtesting tool, not independent confirmation
that the free API or bulk files contain every delisted security, the relevant
price and accounting history, or point-in-time exchange membership. Those coverage
questions and the source-rights constraints remain open.

The provider's [SDK field documentation](https://raw.githubusercontent.com/SimFin/simfin/master/simfin/names.py)
describes `Publish Date` as the initial 10-K or 10-Q publication date and says
`Restated Date` is the most recent restatement, with values from that latest
restatement. Filtering such values solely by the original publication date can
therefore introduce later revisions. The same documentation describes `Shares
(Basic)` as a split-adjusted weighted average over the period, not a verified
point-in-time class-specific share count. These comments do not establish the
contents of every current free-tier response or the full history of revisions.
Verify original values, dissemination and the exact endpoint's semantics before
building a panel or using shares to estimate market equity.

The provider's separate [Python `val_signals` implementation](https://raw.githubusercontent.com/SimFin/simfin/master/simfin/signals.py)
defaults to `REPORT_DATE`, no date offset and diluted shares; it forward-fills
the selected share counts to the supplied price index and multiplies by closing
price for `MARKET_CAP`. If caller-supplied rows are indexed by reporting-period
end, those defaults can expose a share figure before its filing was public. Diluted
period-average shares also do not automatically represent the paper's point-in-time
market-equity definition. Callers can provide different inputs and arguments; this
inspection does not establish how the hosted screener or current API computes its
market cap. No SimFin source rows or market-cap values were read.

The experimental [training helper](../_lib/expected_growth_training.py) can list
period ends from annual reports with independently supplied availability times,
excluding reports unavailable at the fit cutoff. This is **not a complete fiscal
calendar**: a year-end can be known before a late or missing report is filed. The
stale-year guard also needs independently verified year-ends known at the cutoff;
using only filed reports can silently admit a stale target. Synthetic tests show
this failure and the guard's behavior when the missing year-end is supplied. The
helper returns an immutable set that the training guard refuses directly, but a
caller can still bypass that check by converting it; completeness is not enforced. The
target evidence now omits supplied fiscal ends after the fit date; this prevents
those future period ends appearing as observed history, but does not verify when
the remaining calendar information became available or whether it is complete. The
helper does not establish report completeness, authenticity or public timing; SEC
acceptance alone is not proof of dissemination. No real training panel uses it.
Its accepted forms include annual and transition reports and their amendments
(`10-K`, `10-K/A`, `10-KT`, `10-KT/A`); other forms require separate handling.
The forecast helper checks each coefficient's stated availability and excludes the
decision month's fit, but cannot verify that a fit was actually estimated in its
named month or that its training inputs were available then. A corrected synthetic
integration example labels a fit using a June revision as June, rather than April;
it is not evidence that any real monthly fits have been reconstructed.

## First checkpoint: one dated input panel

Use **April 29, 2024 at 16:00 New York time** as a fixed diagnostic cutoff. The April
27 UTC screener commit and April 29 UTC directory archive precede that cutoff; use
their recorded capture times, not a claim that every underlying field was available
earlier. Freeze the exact inputs, commands, code and hashes before reporting counts.
This date is a coverage example, not a proposed backtest window or performance test.

The existing [tape session-coverage audit](_full_lookback_coverage_20261110.json)
records 501 exact-label same-day matches among 503 fixture members on April 30,
2024, with 497 having 252 recorded sessions. This is a **later date** than the
April 29 pilot cutoff, so it cannot establish what was available to that pilot.
It checks session presence, not clean prices or identity, and the S&P 500 fixture
cannot supply the full NYSE breakpoint universe. No returns were calculated here.
Using the same hash-pinned tape, fixture and coverage function for April 29 gives
501 exact-label same-day matches among 503 fixture members, and 497 with 252
recorded sessions. The result is `tape_session_coverage_20240429.json` in the
reference folder (SHA-256
`13b10a3408baf496a70375b5c90146632b3eb3725066ec1b79a33b3238a28f45`). Same-day
date presence does not prove that a bar was available at the pilot's decision time.

The two missing exact labels are `MMC` and `WRK`; the four incomplete 252-session
histories are `GEV`, `KVUE`, `SOLV` and `VLTO`. The tape has an `MRSH` row on
April 29, and [Marsh later announced](https://www.marsh.com/content/mmc-web/mmc-v2/en/news-events/2025/december/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html)
the `MMC`→`MRSH` ticker change effective in January 2026. This supports an alias
lead, not an automatic price-series identity or adjustment validation. `WRK` was
involved in a later [July 2024 combination](https://www.sec.gov/Archives/edgar/data/1732845/000095014224001867/eh240504483_8k.htm);
the tape's later `SW` label appears on the pilot date, but replacing `WRK` with it
requires security-identity and corporate-action checks and would use knowledge
from after the pilot date. No alias was applied in the count. For two short histories,
S&P's [March 2024 announcement](https://press.spglobal.com/2024-03-27-GE-Vernova-and-Solventum-Set-to-Join-S-P-500-Dentsply-Sirona-to-Join-S-P-MidCap-400-Others-to-Join-S-P-SmallCap-600)
identifies the new GEV and SOLV additions; short history must not automatically
be treated as a data fault or filled with a parent company's returns.

The [WestRock merger filing](https://www.sec.gov/Archives/edgar/data/1732845/000095014224001867/eh240504483_8k.htm)
states that ordinary WestRock shares were converted to one Smurfit WestRock share
plus $5 cash each, subject to the stated exceptions. Thus this is not merely a
ticker rename: a replay spanning the merger would need to account for both the
share conversion and cash, including its timing and tax treatment. This later
filing supports a retrospective corporate-action check, not the use of future
identity information in April 2024. The `SW` tape row has not been validated as a
historical WestRock price and no merger adjustment has been calculated.

The frozen Yahoo-sourced external reference (`prices.parquet`, SHA-256
`7b58eeafc855a502469fc5090eb703f76516172deebe13d51609fd6eb2c5f3f5`)
contains both `WRK` and `SW` rows on April 29, 2024. Its `close` fields are
$48.54 and $42.81, respectively; the fields labelled `adj_close` also differ.
This is a single-date, same-source comparison, not independent verification of
either price or its adjustment basis. It does show that blindly using the reference
`SW` row as the `WRK` row would not preserve the observed `WRK` close. No returns
or merger adjustment were computed. The selection and full field values are in
`wrk_sw_reference_20240429.json` in the reference folder (SHA-256
`0d4fd84fd6947654895dab5615a3beafd8e334ce026e5175c88ed89092b0c6f1`).

The reference also has a `WRK` row for each of the 252 NYSE sessions from
April 28, 2023 through April 29, 2024. This checks dates only; it does not verify
prices, security identity or availability at the decision time. The record is
`wrk_reference_sessions_20240429.json` in the reference folder (SHA-256
`d54b372addde40cd51044c6792725a1ba74e4e214e5d263b3614553a7e6ca573`).
Across all 503 fixture members on the same date, the reference has 498 exact-label
rows and 494 complete 252-session histories. Its missing labels are `BK`, `EQR`,
`FI`, `MMC` and `PARA`; the four short histories are `GEV`, `KVUE`, `SOLV` and
`VLTO`. The original tape has more exact-label coverage overall, while this
reference includes `WRK`. Combining sources requires identity and price-basis
checks; these counts alone do not authorize filling gaps. The date-only record is
`reference_member_coverage_20240429.json` in the reference folder (SHA-256
`7954065f109b9f814a0840503c47157bd110177473a08e5163b689a632803ec1`).

As an independent retrospective cross-check, a manager's [March 2024 13F](https://www.sec.gov/Archives/edgar/data/1544676/000154467624000006/xslForm13F_X02/13fv3.xml)
reports WestRock common shares with $60,576,250 value for 1,225,000 shares, or
$49.45 per share. The reference tape's March 28 `WRK` close is $49.45; its `SW`
close is $44.85. The SEC's [form instructions](https://www.sec.gov/about/forms/form13f.pdf)
call for quarter-end values at the final trading close. This is one manager's
reported valuation, not proof that the filer, identifier or data vendor is error-free.
The [filing index](https://www.sec.gov/Archives/edgar/data/1544676/000154467624000006/0001544676-24-000006-index.htm)
dates it May 14, so it was unavailable for the April 29 pilot. Inputs and selected
values are in `wrk_13f_crosscheck_20240328.json` in the reference folder (SHA-256
`9533d74f731a2b16e268394f24ebf907cc9285fe044290571c78b25d28a6c5ba`). This does
not validate April's prices, continuous history or a merger adjustment. No returns
were calculated.

A second manager's [third-quarter 2023 information table](https://www.sec.gov/Archives/edgar/data/928052/000092805223000004/xslForm13F_X02/13fq32023.xml)
reports 500 WestRock common shares at $17,900, implying $35.80 per share. The
same frozen reference has a `WRK` close of $35.80 on September 29, 2023. This is
another retrospective valuation check, not proof of price continuity, independent
validation of the entire series, or evidence that the filing was available at a
particular decision time. Its exact publication timestamp was not checked.

An additional [SEC information table](https://www.sec.gov/Archives/edgar/data/1584639/000142050624000253/Form13F_InfoTable.xml)
reports 30,912 WestRock shares at $1,283,466, implying about $41.52 per share.
A [third-party index](https://13f.info/manager/0001584639-yakira-capital-management-inc)
identifies that accession as a fourth-quarter 2023 filing; the frozen reference's
December 29 `WRK` close is $41.52. The SEC filing index was not accessible in this
check, so the period attribution remains supported by that secondary index rather
than independently verified from the filing cover. This is another retrospective
spot check, with the same limitations as above; it is not a return calculation.

The June 2024 check demonstrates why 13F values cannot simply be divided by
shares without validation. The [FNY June information table](https://www.sec.gov/Archives/edgar/data/1742998/000174299824000004/xslForm13F_X02/FNYIA.xml)
labels values as dollars and reports 600 WestRock shares with value `30`, implying
$0.05 if read literally. The frozen reference's June 28 `WRK` close is $50.26.
The [SEC index](https://www.sec.gov/Archives/edgar/data/1742998/000174299824000004/0001742998-24-000004-index.htm)
confirms the report period as June 30 and filing date as August 5. A
[third-party rendering](https://disclosery.com/filings/0001742998-24-000004)
interprets this manager's figures as thousands, which is consistent with the scale
but has not been independently established; the SEC's 2023 reporting change calls
for dollar values. Do not automatically rescale or use this row as independent price
confirmation. The filing is also later than the pilot and cannot establish
contemporaneous availability. No returns were calculated.

An independent manager's [information table](https://www.sec.gov/Archives/edgar/data/1621915/000108514624003126/infotable.xml)
reports 5,096 WestRock shares valued at $256,125, implying approximately $50.26
per share, matching the frozen reference's June 28 close. Its [cover](https://www.sec.gov/Archives/edgar/data/1621915/000108514624003126/primary_doc.xml)
identifies the June 30 reporting period. This supports that one reference price,
but does not resolve the FNY filing's units, establish continuous series accuracy,
or validate the merger adjustment. The filing is retrospective to the pilot.

On March 28 the pilot tape's `SW` close is 40.648911; the reference's `SW`
adjusted close is 40.161665 and its `WRK` adjusted close is 49.158990. The fields
are not necessarily on the same adjustment basis, so numerical proximity does not
establish identity. Selected fields and input hashes are in
`wrk_sw_tape_fields_20240328.json` in the reference folder (SHA-256
`0b6bfece02472c309b16fdf4e544bdaf594351ac42dba3d8d1480dff5e1558f8`). No return,
corporate-action adjustment or automatic mapping was calculated.

For the 39 dates where both reference labels and the pilot `SW` label overlap
between February 1 and June 28, 2024, the same-date pilot/reference `SW` price
ratio ranges from 1.012132 to 1.012937. The pilot `SW`/reference `WRK` ratio
ranges from 0.790548 to 0.886935. This pattern is consistent with the pilot label
tracking the reference `SW` series more closely, but shared source lineage,
sparse `SW` reference coverage (39 rows versus 103 for `WRK`) and unknown
adjustment bases prevent an identity conclusion. Counts, ratios and input hashes
are in `wrk_sw_source_ratio_2024.json` (SHA-256
`c357c8d3bfa97cd00095933e66c467d6394634bfd26fb80b4d71f5622868e48c`). These are
same-date cross-source ratios, not investment returns; no mapping was applied.

The initial hand-picked diagnostic sample is IBM, KO, WMT, JPM, BRK/A, BRK/B and
AACT. It includes a financial-sector example, a known two-class issuer and a row with
zero reported market cap; it is not representative. Five have an exact symbol in
both frozen files; Berkshire's two screener labels do not. This is a string comparison,
not identity verification. The sample and source hashes are recorded in
`ggTrader-refs/nyse-screener-snapshot-20240131/input_pilot_sample_2024-04-29.json`
(SHA-256 `1f4823fbfa3d20381a191297e36083b2f01495c369a820bb2a67843b61dc91b6`).

**Exchange coverage check:** The same April 27 screener commit has 2,812 NYSE,
4,017 Nasdaq and 306 AMEX rows, with no exact ticker overlap among those three files.
That is not proof that the labels identify primary listings correctly: `LXFR` appears
in the screener's AMEX file, while the April directory labels it NYSE and Luxfer's
[February 2024 10-K](https://www.sec.gov/Archives/edgar/data/1096056/000109605624000011/lxfr-20231231.htm)
identifies its ordinary shares as NYSE-listed. The screener query's grouping and any
intervening listing change remain unverified. Source hashes and the comparison are in
`three_exchange_check_20240427.json` in the reference folder (SHA-256
`2533514c8e717aaf7da00fce20b1fb3895870f2c78bd4773415df6a6b5ba665e`). In particular,
do not use the screener file name alone to set the NYSE breakpoint flag.

**Descriptive size comparison:** In the April screener's NYSE-labelled file, an
unverified `common stock`/`ordinary shares` name filter leaves 1,685 rows with
positive reported market caps, with a $3.405 billion median. Requiring an exact
NYSE/non-ETF/non-test match in the archived directory leaves 1,683 rows and a
$3.405 billion median. The later June-2024-vintage French file reports 1,195
qualifying NYSE stocks and a $4.058 billion median for April 2024. These are not
equivalent universes or dates: the name filter includes funds and misses some share
classes; REIT and other eligibility, multi-class market caps, source timing and
French's historical availability remain unresolved. The French archive was captured
after the pilot cutoff and cannot be used as proof of the cutoff's known breakpoint.
The descriptive calculation and source hashes are in
`ggTrader-refs/nyse-screener-snapshot-20240131/size_reference_check_202404.json`
(SHA-256 `6dce07644a51157d34cc58900aa5ce620f473c87b62d37e8851dc7c6dbb49293`).
It does not test stock returns or establish the expected-growth cutoff.

1. **Identity and eligibility:** Compare the archived exchange directory and screener
   with dated issuer and security identifiers. Record separately exact matches,
   aliases needing independent verification, missing identities, exchange conflicts,
   funds, preferred shares and other exclusions. Names and ticker substitution are
   leads, not proof of identity or of common-stock eligibility.
2. **Market value:** For a small, explicitly listed sample, reconcile as-traded prices,
   share classes, share counts and their availability times. Record stale counts,
   splits and missing or unlisted classes. Do not add the screener's class rows as
   firm values: the Berkshire example shows why that can double-count.
3. **Accounting inputs:** For the same sample, record each necessary annual and
   quarterly fact, fiscal period, accession and independently verified public time.
   Show unavailable, stale, ambiguous and definition-mismatched facts as missing;
   do not substitute later revisions or older fiscal years silently.
4. **Coverage report:** Show the input funnel and reason counts, including firms that
   cannot be evaluated. Retain a dated source and an explicit rule for each exclusion.
   Do not infer population coverage from a small, convenient sample.

### Sample input coverage checkpoint

This is a manual inventory of the evidence already recorded below, not an automated
coverage calculation. The seven security labels represent six issuers. “Checked”
means only the specific check described; it does not certify an input for modeling.

| Issuer | Identity evidence | Market-value evidence | Annual accounting evidence | Quarterly ROE inputs |
| --- | --- | --- | --- | --- |
| IBM | 2024 filing identifies IBM | Count × screener price agrees with reported cap; quote timing and class coverage unverified | Eligible 2022 year-end and filing identified; required facts and definitions not audited | Not audited |
| KO | 2024 filing identifies KO | Count × screener price agrees with reported cap; quote timing and class coverage unverified | 2022 assets checked; candidate tags inspected; liability definitions and revisions unresolved | Q4 income and prior-quarter equity candidates assembled for two years; accounting consistency and complete revision history unverified |
| WMT | 2024 filing identifies WMT | Count × screener price agrees with reported cap; quote timing and class coverage unverified | 2023 assets and revenue checked; candidate tags inspected; model definitions unresolved | Not audited |
| JPM | 2024 filing identifies JPM; financial-sector eligibility unresolved | Count × screener price agrees with reported cap; quote timing and class coverage unverified | Eligible 2022 year-end and filing identified; required facts and definitions not audited | Not audited |
| Berkshire | 2024 filing identifies both share classes; directory aliases need handling | Class rows can double-count firm value; historical counts and class-level prices need reconciliation | Eligible 2022 year-end and filing identified; required facts and definitions not audited | Not audited |
| AACT | 2024 filing identifies Class A; blank-check eligibility unresolved | 2024 filing reports both class counts; April counts and class valuation unverified | 2022 statements identified; filing-type treatment and required facts unresolved | Not audited |

**Current funnel:** six of six issuers have the cited identity and eligible-period
document evidence; zero of six has a fully audited set of the model's inputs in this
pilot. This is a statement about completed checks, not proof that the data do not
exist. The accounting prerequisite includes prior-year changes and quarterly income
and equity, consistent filing versions, and the model's definitions. No forecast or
stock ranking follows from this inventory. The underlying sample, share-count and
candidate-tag artifact names and hashes are given in the sections below.

### First market-value cross-check

For two selected examples, the screener's recorded last sale multiplied by shares
reported in a filing accepted before the cutoff agrees with its market-cap field to
within $0.50: Walmart's $60.16 times 8,058,048,674 shares and Coca-Cola's $61.74
times 4,312,456,168 shares. Walmart's [count](https://www.sec.gov/Archives/edgar/data/104169/000010416924000056/wmt-20240131.htm)
is as of March 13, 2024; Coca-Cola's [count](https://www.sec.gov/Archives/edgar/data/21344/000002134424000009/ko-20231231.htm)
is as of February 16. The SEC and screener source hashes and calculations are saved in
`ggTrader-refs/nyse-screener-snapshot-20240131/single_class_size_crosscheck_20240429.json`
(SHA-256 `7daad9df8718d9baefdfbc5d7de8d5b579c0b33fe48e1cf081b88a9906ad8b94`).

This numerical agreement does not establish the exact time or adjustment basis of
the screener price, intervening changes in shares, complete share-class coverage or
reliability for other issuers. It does not resolve the separate Berkshire multi-class
double-counting example, and the figures are not a stock recommendation.

A follow-up checked IBM and JPMorgan with the same method: price multiplied by the
share counts in their 2023 10-Ks also matched their respective screener market-cap
fields within $0.50. JPMorgan is a financial-sector example, not an established
eligible model constituent. In the current SEC company-facts responses, no cover-page
share-count observations for Berkshire or AACT were found with filing dates in the
checked December 2023–April 2024 window; that does not mean their filings lack such
counts. The SEC response for the historical AACT issuer currently names Kodiak AI and
uses a zero-padded string for its CIK, illustrating why present-day names cannot be
used as historical identity. Source hashes, observations and calculations are in
`share_count_sample_expansion_20240429.json` in the reference folder (SHA-256
`17a275b4c92c07cdc042bc815af604c3f5245c85c38b7343b746fd8b22699962`).

The absence from that company-facts field is not absence from the filing: AACT's
[2023 10-K](https://www.sec.gov/Archives/edgar/data/1853138/000162828024007635/aac-20231231.htm),
accepted before the cutoff, reports 50 million Class A and 12.5 million Class B
shares outstanding on February 23, 2024. It says the sponsor owned all Class B
shares. These are dated counts, not proof of April share counts or an instruction to
value both classes at the listed Class A price. Their availability in the filing does
not establish that an automated, class-aware extraction is reliable.

**Quarterly income candidate:** Coca-Cola's [first-quarter 2024 release](https://www.sec.gov/Archives/edgar/data/21344/000002134424000013/a2024q1earningsreleaseex-9.htm)
is dated April 30, after the pilot cutoff. For the preceding quarter, the experimental
`derive_q4` helper subtracts the nine-month `NetIncomeLoss` reported in the
[October 2023 10-Q](https://www.sec.gov/Archives/edgar/data/21344/000002134423000060/0000021344-23-000060-index.htm)
from the annual value in the [February 2024 10-K](https://www.sec.gov/Archives/edgar/data/21344/000002134424000009/0000021344-24-000009-index.htm):
$10,714 million − $8,741 million = **$1,973 million** for September 30–December 31,
2023. Both filings were accepted before the cutoff; their acceptance times are used
here as provisional timing proxies, not proof of public dissemination or conservative
availability bounds. The selected inputs and source hash are
in `ko_q4_income_candidate_20240429.json` in the reference folder (SHA-256
`f96fe54e976f4237f049be37b6d293c9d111fe6ce8e50137805c987da49aabec`).
This uses two selected versions, not a complete revision search; their accounting
consistency and equivalence to the published model's income measure remain unchecked.
It is not an ROE change or a stock signal.

**ROE-change input candidate:** The selected 2022 annual and nine-month income
figures give $2,031 million for that year's fourth quarter. The 2023 and 2022
third-quarter `StockholdersEquity` values are $26,325 million and $22,805 million,
respectively. Applying the prototype's ratio formula gives
`1,973 / 26,325 − 2,031 / 22,805 = −0.01411`. The selected inputs, four filing
indexes, acceptance-time proxies and helper hash are in `ko_q4_roe_candidate_20240429.json`
in the reference folder (SHA-256
`acf4664e0448a60c1793a9a4f912e1c2f85e19d386a30c2239209e445a31ac88`). This is an
arithmetic and data-availability example, not a validated published predictor:
the income and equity definitions, cross-filing comparability and complete revision
history still need verification. No ranking or return was computed.

**Cross-check and timing:** Coca-Cola's [February 13, 2024 earnings release](https://www.sec.gov/Archives/edgar/data/21344/000002134424000004/a2023q4earningsreleaseex-9.htm)
directly reports the same $1,973 million and $2,031 million as fourth-quarter net
income attributable to Coca-Cola shareowners. The selected [2023](https://www.sec.gov/Archives/edgar/data/21344/000002134423000060/ko-20230929.htm)
and [2022](https://www.sec.gov/Archives/edgar/data/21344/000002134422000042/ko-20220930.htm)
quarterly statements label both equity values as attributable to those shareowners,
separate from noncontrolling interests. This supports consistency of the selected
GAAP numerator and denominator, but does not establish their Compustat equivalence
or rule out other revisions. The release's [8-K was accepted February 13](https://www.sec.gov/Archives/edgar/data/21344/000002134424000004/0000021344-24-000004-index.htm),
seven days before the 10-K used above. A filing-only pipeline can therefore delay
an available quarterly observation; the current helper does not extract earnings
releases or establish when every feature was first public.

**Book-equity gap:** The [Global-q methodology](https://global-q.org/uploads/1/2/2/6/122679606/factorstd_2026jul.pdf)
defines quarterly book equity using shareholders' equity, available deferred taxes
and investment tax credits, and a deduction for preferred stock, with specified
fallbacks. The example above uses reported shareholders' equity without those
adjustments. Its −0.01411 is therefore a placeholder calculation, not a verified
reconstruction of the method; the adjustment inputs and their availability must be
checked for both quarters before using it as a predictor.

The source filings do contain relevant tax balances: deferred-tax assets and
liabilities were $1,575 million and $2,733 million at September 29, 2023, and
$1,708 million and $3,124 million at September 30, 2022, respectively. The saved
company-facts response contains matching `DeferredIncomeTaxAssetsNet` and
`DeferredIncomeTaxLiabilitiesNet` observations from those two 10-Qs. Their dates and
amounts do not establish how Compustat's combined deferred-tax-and-investment-credit
item is constructed, or whether either balance should be substituted for it. A text
search of the two filings did not find “preferred stock”; that is not proof that the
required preferred-stock adjustment is zero. The model input remains unresolved.

An identity trap matters when following up that gap: Coca-Cola Consolidated is a
different SEC issuer (CIK 317540) from The Coca-Cola Company (CIK 21344). The
former's [2022 annual filing](https://www.sec.gov/Archives/edgar/data/317540/000031754023000009/coke-20221231.htm)
discusses its own preferred shares; that is not evidence about Coca-Cola Company's
preferred-stock adjustment. No preferred-stock observations for the two selected
10-Q accessions and period ends were found in the saved Coca-Cola Company
company-facts response. Absence of tagged observations is not proof of zero.

**Machine-readable coverage check:** The saved current Coca-Cola company-facts
response (SHA-256 `87a4173fcd3d662d18d19e0e885095aa5e3812b74c54d1ee55808ff1540a8081`)
contains no observations from the February 13 earnings-release 8-K accession
`0000021344-24-000004`, across any namespace, concept or unit. In particular,
neither `NetIncomeLoss` nor `ProfitLoss` has an observation from that filing. The
[SEC index](https://www.sec.gov/Archives/edgar/data/21344/000002134424000004/0000021344-24-000004-index.htm)
identifies the release as exhibit 99.1. For this issuer and release, a company-facts
feed alone cannot supply the earlier announcement values; a separate, reliably
timestamped source or exhibit extraction would need validation. This is a check of
one current response, not a claim about all companies or all 8-K filings.

The same metadata check on two other hand-picked fourth-quarter releases found no
observations from IBM's [January 24, 2024 8-K](https://www.sec.gov/Archives/edgar/data/51143/000005114324000004/0000051143-24-000004-index.htm)
or Walmart's [February 20, 2024 8-K](https://www.sec.gov/Archives/edgar/data/104169/000010416924000019/0000104169-24-000019-index.htm)
in their saved current company-facts responses. Their subsequent annual reports
were filed February 26 and March 15, respectively. The three checks, accessions and
source hashes are in `earnings_release_companyfacts_check_20240429.json` in the
reference folder (SHA-256
`7ef176421cedf1addce74dadf147a26a33950f5cc08e6018ae86ce4ed50fef29`). These are
three selected examples, not an estimate of overall coverage; filing dates do not
establish the first public time of each underlying item.

**Exhibit-layout feasibility:** Two issuer-hosted earnings exhibits were saved and
hashed in the reference folder. With the installed pandas version, the Coca-Cola
exhibit has separate quarterly and annual income tables, each with the same
shareowner-income row label. Walmart's exhibit has quarterly and annual values in
one table. A row-label match without checking period headers can therefore select an
annual value as quarterly income. `earnings_exhibit_layout_probe.json` records the
source URLs, hashes, parser version, table shapes and matching rows (SHA-256
`72f08af274cf43930e1464bac042ed973a6131a2b8215d02b5bdb5d0d8bbc343`). The documents
were downloaded retrospectively. This is a two-document layout probe, not a validated
parser or evidence of broad extractability; ambiguous periods, GAAP versus adjusted
figures, revisions and source availability still require explicit checks.

The experimental SEC fact normalizer now accepts either a numeric CIK or a strictly
ten-digit zero-padded CIK string, while requiring an exact match to the expected
issuer. A filtered real observation from that response was normalized successfully;
the current company name and CIK do not by themselves establish the historical
security mapping. An Opus review found no blocking issue; cross-family review remains
unconfirmed, and both required reviews are still needed before a merge.

The April screener's two prices also match, to the nearest cent, the April 26 close
in the frozen external Yahoo-sourced reference: $60.16 for Walmart and $61.74 for
Coca-Cola. That reference's dividend-adjusted closes are lower ($58.71 and $57.53),
illustrating why the price basis matters when multiplying by an unadjusted share
count. This two-name match does not establish independent source lineage, exact quote
times or corporate-action completeness. Source hashes and values are recorded in
`price_basis_crosscheck_20240426.json` in the same reference folder (SHA-256
`d8523dd8d4db7400b72a243dd564fdf6dfd2a80e9fffedf62f29494bde9a92c5`).

Berkshire illustrates why the share-count date and publication time are distinct.
Its [2024 first-quarter report](https://www.berkshirehathaway.com/qtrly/1stqtr24.pdf)
gives April 19 counts of 562,539 A and 1,311,384,883 B shares, but the [SEC accepted
that report on May 6](https://www.sec.gov/Archives/edgar/data/1067983/000095017024053185/0000950170-24-053185-index.htm),
after the pilot cutoff. No earlier public source for those counts has been verified.
At the frozen screener prices, using them instead of the February counts would lower
the illustrative two-class value by about $2.24 billion (0.26%); that calculation is
descriptive and must not be used to construct the April 29 input panel.

**Earlier split-basis check:** The January 2021 screener snapshot reports Walmart at
$140.49. The later Yahoo-sourced reference records $46.83 for January 29, 2021;
three times that price matches the screener. This is consistent with Walmart's later
[three-for-one split](https://corporate.walmart.com/news/2024/01/30/walmart-announces-3-for-1-stock-split).
The 2021 screener price multiplied by the 2,829,285,926 shares in Walmart's
[December 2020 filing](https://stock.walmart.com/sec-filings/all-sec-filings/content/0000104169-20-000076/wmt-20201031.htm)
matches the snapshot's reported $397.49 billion market cap to within $0.50. Pairing
the later split-adjusted price with that earlier share count without adjustment would
understate this example's value by about two thirds. Source hashes and arithmetic
are saved in `walmart_split_basis_202101.json` in the reference folder (SHA-256
`ad5e443c9a8752f8189b61849ff3543021ac7c32eac5f328ecaf61167442038e`). This is a
single example; the screener's exact quote time and source lineage remain unverified.

Across the 24 archived 2021–2022 month-end snapshots, the screener prices for both
Walmart and Coca-Cola matched the latest prior reference close within half a cent
after multiplying Walmart's later reference prices by three. The comparison used
the commit's New York date and, before 16:00 ET, a prior date. All 48 comparisons
matched; source hashes and observations are in `price_basis_sample_2021_2022.json`
(SHA-256 `03b1fb6d511c6ff42a847f38a799c199103348b6898cbf343541a2343e7690b7`).
The adjustment uses knowledge of Walmart's later 2024 split and is only a retrospective
diagnostic; it is not a point-in-time input. Two hand-picked stocks do not establish
full-universe price accuracy, exact quote times, independent source lineage or fills.

### Initial identity and timing check

The following SEC filings were accepted before the cutoff and identify the listed
securities. They support the mapping; they do not rule out an intervening change or
prove the listing files complete.

| Sample symbol | Issuer and SEC CIK | Evidence accepted in 2024 |
| --- | --- | --- |
| IBM | International Business Machines, 0000051143 | [February 26 10-K](https://www.sec.gov/Archives/edgar/data/51143/000005114324000012/ibm-20231231.htm) identifies IBM on NYSE. |
| KO | Coca-Cola, 0000021344 | [February 20 10-K](https://www.sec.gov/Archives/edgar/data/21344/000002134424000009/ko-20231231.htm) identifies KO common stock on NYSE. |
| WMT | Walmart, 0000104169 | [March 15 10-K](https://www.sec.gov/Archives/edgar/data/104169/000010416924000056/wmt-20240131.htm) identifies WMT common stock on NYSE. |
| JPM | JPMorgan Chase, 0000019617 | [February 16 10-K](https://www.sec.gov/Archives/edgar/data/19617/000001961724000225/jpm-20231231.htm) identifies JPM common stock; its SEC filing describes a financial-services firm. |
| BRK/A, BRK/B | Berkshire Hathaway, 0001067983 | [February 26 10-K](https://www.sec.gov/Archives/edgar/data/1067983/000095017024019719/brka-20231231.htm) identifies classes BRK.A and BRK.B. The directory uses dots; the screener uses slashes. |
| AACT | Ares Acquisition Corporation II, 0001853138 | [February 28 10-K](https://www.sec.gov/Archives/edgar/data/1853138/000162828024007635/aac-20231231.htm) identifies AACT as Class A ordinary shares and describes a search for an initial business combination. |

**AACT identity after the cutoff:** Ares changed its name to Kodiak AI on September
23, 2025, completed its business combination on September 24, and began trading as
KDK on September 25, according to its later [8-K](https://www.sec.gov/Archives/edgar/data/1853138/000119312525225299/d933659d8k.htm).
The [filing index](https://www.sec.gov/Archives/edgar/data/1853138/000119312525225299/0001193125-25-225299-index.html)
dates that report September 30, 2025. These later facts explain the current name;
they must not be used to label or select the security at the April 2024 cutoff. A
matching CIK alone does not establish continuity of the business or share class.

The prototype's exact four-calendar-month rule has a material effect on this date:
December 31, 2023 year-ends become eligible on April 30, **after** the April 29
cutoff; Walmart's January 31, 2024 year-end becomes eligible May 31. Filing before
the cutoff alone is insufficient. The actual newest eligible year and each fact's
availability still need checking; do not silently substitute an older eligible year
when its fact is missing. These
linked filing pages were inspected now and are not frozen filing artifacts.

### Eligible annual-period source check

Under the prototype's calendar rule, the newest eligible annual end is December 31,
2022 for IBM, KO, JPM and Berkshire, and January 31, 2023 for Walmart. SEC filing
indexes show that each corresponding annual report was accepted in 2023:

| Issuer | Eligible end | Earlier filing evidence |
| --- | --- | --- |
| IBM | 2022-12-31 | [10-K accepted February 28, 2023](https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/0001558370-23-002376-index.htm) |
| KO | 2022-12-31 | [10-K accepted February 21, 2023](https://www.sec.gov/Archives/edgar/data/21344/000002134423000011/0000021344-23-000011-index.htm) |
| WMT | 2023-01-31 | [10-K accepted March 17, 2023](https://www.sec.gov/Archives/edgar/data/104169/000010416923000020/0000104169-23-000020-index.htm) |
| JPM | 2022-12-31 | [10-K accepted February 21, 2023](https://www.sec.gov/Archives/edgar/data/19617/000001961723000231/0000019617-23-000231-index.htm) |
| Berkshire | 2022-12-31 | [SEC filing directory dated February 27, 2023](https://www.sec.gov/Archives/edgar/data/1067983/000095017023004451/) |
| AACT | 2022-12-31 | [S-1 accepted March 29, 2023](https://www.sec.gov/Archives/edgar/data/1853138/000119312523083839/0001193125-23-083839-index.htm) contains 2022 financial statements; its use requires a filing-type decision. |

These links establish the presence and timing of candidate documents, not coverage
or equivalence of individual accounting fields. Later filings available by the cutoff
may contain revised comparatives, so these original reports are not automatically the
selected versions. The sample also contains financial-sector and blank-check examples;
their eligibility under the proposed method is not decided by this table.

### First fact and revision spot check

Two original annual reports and their next-year comparatives were manually checked;
both sets of filings were accepted before the pilot cutoff. Values below are in USD
millions. Matching two reports does not establish that no other available filing
revised a fact, or that the numerical helpers select the right filing.

| Issuer and eligible period | Original report | Next-year comparative | Checked observation |
| --- | --- | --- | --- |
| Coca-Cola, 2022-12-31 | [2022 10-K](https://www.sec.gov/Archives/edgar/data/21344/000002134423000011/ko-20221231.htm) | [2023 10-K](https://www.sec.gov/Archives/edgar/data/21344/000002134424000009/ko-20231231.htm) | Total assets: 92,763 in both. |
| Walmart, year ended 2023-01-31 | [2023 10-K](https://www.sec.gov/Archives/edgar/data/104169/000010416923000020/wmt-20230131.htm) | [2024 10-K](https://www.sec.gov/Archives/edgar/data/104169/000010416924000056/wmt-20240131.htm) | Total assets: 243,197 in both; annual total revenue: 611,289 in both. |

Walmart's reports also list annual **net sales of 605,881**, distinct from total
revenue. Which measure corresponds to the published model's accounting input is not
established by this check. Period bounds, XBRL tags, intervening amendments, other
components and the source files still need verification and pinning before any
programmatic signal calculation.

### Machine-readable sample

The SEC company-facts responses for Walmart and Coca-Cola were inspected for the two
annual-report accessions above. A reduced extract of only those accessions and eligible
periods is saved as `ggTrader-refs/nyse-screener-snapshot-20240131/sec_selected_facts_20240429.json`
(SHA-256 `2f16c9b9dc8bd78c3860a5189dc57cd68f3aa1ab331e90a01b753ac1080c7149`). The
complete SEC responses were not saved or hashed. This extract is not a complete
revision history and cannot prove which filing was latest among all available filings.

Using the two [filing-index](https://www.sec.gov/Archives/edgar/data/104169/000010416923000020/0000104169-23-000020-index.htm)
[timestamps](https://www.sec.gov/Archives/edgar/data/104169/000010416924000056/0000104169-24-000056-index.htm)
for Walmart and the corresponding [2023](https://www.sec.gov/Archives/edgar/data/21344/000002134423000011/0000021344-23-000011-index.htm)
and [2024](https://www.sec.gov/Archives/edgar/data/21344/000002134424000009/0000021344-24-000009-index.htm)
indexes for Coca-Cola, the experimental instant and annual-flow selectors chose the
2023 accession at a December 1, 2023 decision and the 2024 accession at the pilot
cutoff, within this reduced extract. The values in the two selected versions matched
the manual check above. Coca-Cola `Revenues` was 43,004 million in both versions.

For Walmart, `Revenues` was 611,289 million while
`RevenueFromContractWithCustomerExcludingAssessedTax` was 605,881 million; treating
the tags as interchangeable changes the input. The appropriate model mapping remains
unverified. This is a two-issuer input check, not a completeness or signal test. The
tested local versions of `sec_facts.py`, `sec_annual_flows.py` and `filing_vintages.py`
had SHA-256 prefixes `f0200d51`, `eeb39c6f` and `d5994f09` respectively; they remain
experimental and the extraction was not independently reviewed.

**Follow-up on assets:** Full current company-facts JSON responses were subsequently
saved in the same reference folder (`sec_companyfacts_0000104169.json` SHA-256
`49330f98609f70fcef4054edc90bc811398a58c4bf26dc0b218d607967a8dbe4`;
`sec_companyfacts_0000021344.json` SHA-256
`87a4173fcd3d662d18d19e0e885095aa5e3812b74c54d1ee55808ff1540a8081`). These are
retrospective responses, not archived 2024 snapshots. For the eligible period and
`Assets` tag, each contains five observations filed by the cutoff: the two 10-Ks
and three intervening 10-Qs, all with the same value. SEC filing-index or submissions
acceptance times were checked separately. With these five observations, the selector
chooses the latest 10-Q on December 1, 2023 and the 2024 10-K at the pilot cutoff.
Thus the earlier reduced-extract December result does **not** identify the latest
available version. The observations, UTC times and selections are recorded in
`assets_filing_versions_20240429.json` (SHA-256
`2c8215ce77e3ccd81d48186f80d16fa815e909688944167f9c5840beff87e28c`). This checks
one tag and period per company, not source completeness or the full signal.

**Candidate-tag coverage:** A two-year metadata check searched the two annual-report
accessions per issuer for candidate tags already used in an earlier feasibility
inventory. Assets, debt candidates, revenue candidates, SG&A and inventory appear
for both eligible years for both issuers. The initial candidate list did not find
Walmart's cost-of-goods or receivables tags, or Coca-Cola's separate accounts-payable
and accrued-liability tags, in either year. Neither issuer had the chosen R&D or
deferred-revenue candidates in those reports. This does not mean those economic
amounts are zero or absent from the filings.

A follow-up tag-name inspection found Walmart `CostOfRevenue` and
`ReceivablesNetCurrent`, and Coca-Cola `AccountsPayableAndAccruedLiabilitiesCurrent`
and `AccountsPayableTradeCurrent` in the 2024 reports. Whether and how those tags
match the published model's Compustat inputs remains unresolved; a combined liability
must not be counted twice. The candidate list, year-end matches, accessions and full
source hashes are in `expected_growth_candidate_tag_pilot_20240429.json` in the same
reference folder (SHA-256
`fee3e0fe6f7a896b018672edc79afefb25fb6a33d345e5a693fdeea9e6dfcb56`). This checks
tag presence in two reports per company, not all filings, availability of every fact,
accounting equivalence, population coverage or an investable signal.

**Additional issuer metadata check:** The same 14 fixed candidate-tag groups were
searched in the saved IBM, JPMorgan, Berkshire and AACT company-facts responses for
both 2021 and 2022 fiscal year-ends. Restricting to filings dated before April 29,
2024, and to full-year durations for flow items, the number of groups with at least
one candidate in each year was IBM 10, JPMorgan 3, Berkshire 3 and AACT 0. The
records, exact tags, accessions and input hashes are in
`expected_growth_candidate_tags_additional_issuers_20240429.json` in the reference
folder (SHA-256 `c02e2b572cd1bf2b1ced65ca3d8fce8e5e732a9aadd83d8f041914ce64de8104`).
This retrospective metadata screen uses filing dates as a coarse filter, not
verified public timestamps. Counts are not usable-feature coverage: alternate or
issuer-specific tags may exist, some model components can be missing under the
published rules, and a present tag may not match the required definition. JPMorgan
and AACT are diagnostic examples whose model eligibility is unresolved. The work
does not select stocks or calculate returns.

For IBM, a follow-up inspection of the saved response found `CostOfRevenue`,
`DeferredRevenueCurrent` and `DeferredRevenueNoncurrent` in both years in accession
`0001558370-23-002376`, the [2022 annual filing](https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/0001558370-23-002376-index.htm).
None was in the fixed candidate list above. The two deferred-revenue tags total
$16,095 million at the 2021 year-end and $15,531 million at the 2022 year-end in
that filing. These are candidate mappings discovered after the count; neither the
cost tag nor the deferred-revenue pair has been proven equivalent to the model's
Compustat fields. In particular, absence from the initial list cannot justify
substituting zero for the change.

IBM's [annual-report exhibit](https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/ibm-20221231xex13.htm)
labels the 2022 figure as $27,842 million of total cost, matching `CostOfRevenue`;
it lists selling, general and administrative expense ($18,609 million) and
research, development and engineering expense ($6,567 million) separately. The
accounting policies also say amortization of capitalized software costs is included
in software cost within cost of sales. This confirms a concrete definition issue:
the available tag is not automatically equivalent to the model's Compustat COGS,
and the separate R&D line cannot be inserted into its formula without reconciling
the Compustat SG&A definition. No numerical substitution has been validated.

**Concrete presentation mismatch:** Coca-Cola's [2022 10-K](https://www.sec.gov/Archives/edgar/data/21344/000002134423000011/ko-20221231.htm)
reports 2022 other accrued expenses of $5,643 million and separately reports $934
million of variable consideration payable. Its [2023 10-K](https://www.sec.gov/Archives/edgar/data/21344/000002134424000009/ko-20231231.htm)
reports the same 2022 total accounts payable and accrued expenses ($15,749 million),
but lists other accrued expenses of $6,577 million and no separate variable-
consideration line. The difference is exactly $934 million, consistent with a
presentation regrouping; the filing does not explicitly establish its cause.

For `OtherAccruedLiabilitiesCurrent`, comparing the 2022 10-K's 2021 and 2022
figures gives a $625 million increase. Mixing the 2023 report's 2022 value with the
2022 report's 2021 value gives a $1,559 million increase. Both reports were available
at the pilot cutoff. A latest-version rule alone cannot establish that accounting
components have comparable definitions across years. This is not yet an approved
mapping of the model's accrued-expense input. The extracted facts, sources and
arithmetic are in `ko_accrual_presentation_20240429.json` in the same reference
folder (SHA-256 `d43fab92d056968d1a9dc22946359d6dc4a2fa13cf4e436934c8632f3f70481c`).

The existing normalized-fact selector was then run on those three observations:
selecting the latest available version separately for each year reproduced the
$1,559 million mixed-filing change; restricting both observations to the 2022 report
before selection reproduced $625 million. The [current selector](../_lib/filing_vintages.py)
returns only the latest row per period, so a caller that discards older rows first
cannot recover the same-filing pair. This is a diagnostic, not a rule that the older
report is always preferable: even a shared filing does not prove comparable
accounting definitions. The selected facts and input hash are recorded in
`ko_same_filing_selector_check_20240429.json` (SHA-256
`d1333f0d3eb4db499ed094300419bd58099894273d5f11f7ec0a905f70af95ce`).
The experimental `available_versions` helper now retains all validated versions
available at the cutoff; the existing latest-only behavior is unchanged. On this
sample it preserves all three observations, including the two from the same filing.
It does not choose a pair or validate their economic comparability.
An independent synthetic review found no cutoff or compatibility defect; it did not
validate the SEC data or establish a stock signal.

The broader combined payable-and-accrued line is unchanged between the two reports
for 2022. It increased by $1,130 million from 2021 to 2022, including a $705 million
increase in separately disclosed trade payables and a $425 million increase in the
remainder. Adding the combined change and the payable change would count that $705
million twice. The [2022 report](https://www.sec.gov/Archives/edgar/data/21344/000002134423000011/ko-20221231.htm)
also says the line includes a business-sale advance, so this arithmetic does not prove
the remainder or combined line matches the model's Compustat inputs. The checked
facts and arithmetic are in `ko_combined_liability_change_20240429.json` (SHA-256
`fa7a0bf45688e6cf1f23ee8198295c5db20810359a6d868713f1ea490e634796`).

The [research details](../../programme/RESEARCH_GUIDE_DETAILS.md#sec-sample-notes)
record the source checks and outstanding limits. The existing filing and market-value
helpers are synthetic-tested primitives, not a verified production data pipeline.

## Before a stock-level return study

- Establish a dated, sufficiently complete reference universe beyond one example,
  including delisted names, ticker changes and permitted use. Quantify coverage and
  exclusions across dates; do not reconstruct history from today's surviving names.
- Establish point-in-time prices, class-level and firm-level values, accounting
  mappings, fiscal-period alignment and reporting delays. Resolve the source's
  treatment of unlisted classes, security weights and NYSE breakpoints.
- Assemble and audit the historical cross-sectional regression inputs. The prototype
  needs at least 30 eligible prior monthly fits and uses up to 120; a single-date
  panel or the 2023–2025 listing snapshots alone cannot supply that history.
  The synthetic-tested `investment_change` helper computes the difference between
  successive annual asset-growth rates, but does not select fiscal years or check
  when their figures became public. A training fit must use only targets available
  by its fit time and the method's appropriately lagged predictors. Independent
  review found its formula and input checks sound; it now requires named asset
  arguments following that review. The caller must also apply the method's monthly
  winsorization and report exclusions. A synthetic integration example combines
  the filing selector and target helper: the example refuses to calculate until
  its last annual fact is available, and a later revision does not alter the earlier
  result. The experimental `investment_target` helper applies that cutoff to
  explicitly named, consecutively spaced annual periods and a single issuer. It
  now shares its selection with `investment_target_evidence`, which returns the
  issuer, cutoff, selected values, filing accessions, supplied availability
  times, fiscal-year-end set and continuity breaks known by the cutoff alongside
  the target. Later-available breaks are omitted from the evidence. Synthetic
  tests cover a later revision; the evidence
  helper has not been independently reviewed or used in a real stock pipeline.
  These records do not verify their inputs or establish accounting comparability.
  The selection also applies the prototype's exact four-calendar-month rule on the New York date,
  which can differ from the published method's monthly convention. It requires a
  caller-supplied set of fiscal-year ends known at the fit time and refuses a stale
  target if a newer eligible year is in that set. It also refuses a window that
  spans another known fiscal year-end. It relies on that issuer-specific set being
  complete and independently verified, as well as on verified publication times; it does not
  enforce predictor lag, revision comparability or winsorization. An independent synthetic
  review found no blocking defect in the earlier version; a later independent Opus
  review also checked the future-row and four-month changes and found no blocking
  defect. A separate review found no blocking defect in the known-year-end check;
  the subsequent intervening-period safeguard has not been independently reviewed.
  The current prototype also requires a caller-supplied mapping of continuity-break
  dates to independently verified availability times. A break available by the fit
  cutoff and effective between the first year-end and fit date, inclusive, blocks
  the target; later-available breaks cannot change an earlier result. This is
  conservative, and an empty or incomplete mapping does not establish continuity;
  the prototype cannot verify event identity, effective date or availability. The
  continuity safeguard has synthetic tests but no independent review. No pipeline
  uses it.

  **IBM target input example:** IBM's [2022 annual filing](https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/0001558370-23-002376-index.htm),
  accepted February 28, 2023, contains assets for year-ends 2020, 2021 and 2022 of
  $155.971 billion, $132.001 billion and $127.243 billion. The successive asset
  changes are −15.37% and −3.60%; their difference is +11.76 percentage points.
  IBM [spun off Kyndryl in 2021](https://www.sec.gov/Archives/edgar/data/51143/000155837021014643/ibm-20211103xex99d3.htm)
  and reported a [pension obligation and asset transfer in 2022](https://www.sec.gov/Archives/edgar/data/51143/000110465923029024/tm231774d5_ars.pdf).
  These events complicate interpreting the mechanical change as organic investment;
  their individual effects have not been isolated. The source hash and arithmetic
  are in `ibm_asset_change_corporate_actions_20240429.json` in the reference folder
  (SHA-256 `6be4fdb158efe3a9ea7174ea268bdc152664a35cf835a7af51e561727343e412`).
  This is a data example, not a fitted forecast or return test. A future study must
  explicitly define its corporate-action treatment and point-in-time eligibility.

  An earlier version of the prototype target guard was exercised on the three
  selected facts from that one filing. With the SEC acceptance time as a proxy, it
  returned no target one second before that filing and the calculated value at a
  May 1, 2023 cutoff. The earlier cutoff checks only the selected-file availability
  boundary; the February 28, 2023 cutoff is not a valid fit date under the four-month
  rule. The filtered facts,
  code hashes and outputs are in `ibm_guard_single_filing_20230501.json` in the
  reference folder (SHA-256
  `b3162634806a1b44823116cd165c43faca6e0f7a8f36b6f6726d252e6da5fdb1`). Other IBM
  filings were deliberately excluded, so this does not establish the latest available
  values, complete history, accounting comparability or a valid training observation.

  **Broader version check:** The current saved IBM company-facts response contains
  13 asset observations for these three year-ends filed before May 1, 2023, across
  ten accessions. Their acceptance times were matched to the current SEC submissions
  record. At the May 1 cutoff, the latest-version selector picks the 2022 10-K for
  the 2020 and 2021 values and the April 2023 10-Q for the 2022 value. The values
  agree with the single-filing example, so its arithmetic is unchanged; its selected
  filing for the 2022 value was not the latest. The source and code hashes, selected
  versions and timestamps are in `ibm_assets_versions_guard_20230501.json` in the
  reference folder (SHA-256
  `2487781fe48ba4e5f939b3797354d2222bab466fad394133e0eecb1f3365a5f9`). Both SEC
  responses are retrospective, and acceptance is a timing proxy, not proof of
  first publication or complete revision coverage. Corporate-action comparability
  and the remaining conditions for a model fit are still unverified.
- Separately specify the portfolio construction, decision and execution times,
  missing-data and delisting treatment, baseline including SPY, turnover, costs,
  taxes and loss measures. Pin dates, inputs and methods before requesting approval
  for any returns. Report estimates and uncertainty rather than new GO/NO-GO labels.

If a dependency cannot be verified, record its effect on coverage and the next source
to check. A partial input panel is evidence about feasibility, not a winning strategy.

**Rounded-date limitation for the SEC notes-archive checks below:** The saved
archives' readme says numeric `ddate` and submission `period` are rounded to the
nearest month-end. Counts, matching, 350–380-day spacing and four-month eligibility
checks based on those fields therefore concern rounded labels, not necessarily
exact fiscal dates. They can admit or reject a case differently from a rule using
the original dates, especially near a cutoff. The [metadata checker](code/sec_asset_period_coverage.py)
uses these rounded fields; it is not a point-in-time eligibility validator. Do not
interpret its results as exact year-end matches or valid training observations.
Recover original filing contexts and separately verify availability for that use.

For a concrete boundary example, AMD's [2021 annual report](https://www.sec.gov/Archives/edgar/data/2488/000000248822000016/amd-20211225.htm)
states its fiscal year ended December 25, 2021. Under this prototype's exact
four-calendar-month rule, that date reaches April 25, 2022; a December 31 rounded
label reaches April 30. At a hypothetical April 29 cutoff, the former passes the
date rule while the latter does not. This illustrates a possible false exclusion,
not a full input-panel result: the 2022 extract was not inspected, and passing
the date rule alone does not establish public availability or eligibility.

**Annual-assets presence check:** In the two saved 2021 Q1/Q2 SEC notes archives,
selected 10-K, 10-Q and 8-K filings dated through June 28 contain dimensionless
US-GAAP `Assets` facts in USD with rounded December 31 labels for 2018, 2019 and 2020 for 480,
4,269 and 5,433 distinct filing CIKs respectively; 459 CIKs have all three dates.
This is a metadata-presence check, not eligible-firm coverage or a training fit:
the archives are only two quarters, the dates need not be the issuer's fiscal
year-ends, and values, public availability, revisions and issuer mappings were not
verified. Other fiscal calendars are excluded by construction. The explicit
US-GAAP-filtered result is `ggTrader-refs/sec-fsnds-2021q2/assets_period_presence_usgaap_20210629.json`
(SHA-256 `b07b80171b68f69a3d466d007606b550275f6ff73edf128dc5f88f1c18ab4ec7`).
An earlier preliminary artifact did not enforce the taxonomy filter; the corrected
scan produced the same counts. No targets, forecasts or returns were calculated.

Adding the current consolidated 2020 Q1 archive raises the number of filing CIKs
with all three rounded year-end labels from 459 to 3,555. This suggests the short
filing window explains much of the observed gap, but is still not an eligible-firm
coverage estimate or proof of usable values. The three-archive metadata counts and
hashes are in `ggTrader-refs/sec-fsnds-2021q2/assets_presence_2020q1_2021q1_q2_20210629.json`
(SHA-256 `371793b7c1d07451bf892b67080e45c70569fecbf038f6ae2cb45accdff40379`).
The additional archive is saved in `ggTrader-refs/sec-fsnds-2020q1/`. Fiscal-year
identity, revisions, precise availability, accounting comparability and historical
archive vintage remain unverified; no targets or returns were computed.

The [SEC's notes-data documentation](https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets)
says these datasets were published monthly beginning in November 2020 and,
starting in March 2024, monthly files are consolidated into quarterly files after
a year. It also identifies later processing corrections to some older datasets.
The downloaded 2021 quarterly archives therefore cannot themselves establish what
was available on June 29, 2021; their filing-date filters do not establish when
either the consolidated files or a particular fact was disseminated. The SEC's
[EDGAR guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
also describes post-acceptance corrections and deletions, and notes that some
late filings are disseminated the following business day. A point-in-time panel
needs independently supported filing availability and revision handling; today's
files and their hashes alone do not supply those. These notes do not prove that
the selected 2021 facts were changed.

A separate fiscal-period metadata check uses the latest eligible 10-K period
observed in the three archives, another observed 10-K period 350–380 days earlier,
and dimensionless USD `Assets` dates in those filings. It finds 3,646 filing CIKs
with three appropriately spaced asset dates; 167 of their latest periods do not end
on December 31. This is not a validated training panel: the archives may omit a
newer annual filing, an earlier comparative date is not independently established
as a fiscal year-end, and multiple filings or asset observations may conflict. The
four-month and filing-date checks are only eligibility bounds, not proof of public
availability or comparable accounting. Counts, method and input hashes are in
`ggTrader-refs/sec-fsnds-2021q2/assets_annual_period_metadata_20210629.json`
(SHA-256 `a20553b7165e0e225ea9ba161649b8cc58f879af9f0a986fcf009d49ac99ef7e`).
The later [metadata checker](code/sec_asset_period_coverage.py) reproduces the
counts and input hashes; synthetic tests cover date eligibility, ambiguous periods
and taxonomy filtering. Hashes recorded after reading do not prove historical
availability. No asset-growth targets or return tests were calculated.

A stricter within-filing value check finds 3,549 CIKs with three selected
periods having positive, finite, non-conflicting values within their respective
filings, and a unique eligible earlier date in the older filing. This uses the
same observed annual-period pairing, not an eligible stock universe. It does not
select the latest revision or compare overlapping values between different
filings; a newer revision can change an input. It also does not validate fiscal
year identity, publication timing or accounting comparability. The metadata-only
result is `ggTrader-refs/sec-fsnds-2021q2/assets_positive_within_filing_20210629.json`
(SHA-256 `272b9d7a08ec75021cb2db1542b39f7467ea15903b2cd31ab98f2d887f3445b6`).
No individual values, targets or returns were reported by this check.

To check cross-filing revisions, a separate scan compared the same prior-period
`Assets` value in the selected older and newer 10-K for each CIK, choosing the
latest filing date for each reported period and refusing same-date ties. Among
3,642 pairs with one positive finite value in each filing, 3,446 values agreed
and 196 differed; 89 otherwise eligible pairs had a missing or ambiguous value.
This is not a restatement rate: causes and economic comparability were not checked,
rounded dates may combine distinct original dates, and amendments and later
quarterly filings are outside this comparison. Metadata,
input hashes and five accession-level examples are in
`ggTrader-refs/sec-fsnds-2021q2/assets_overlapping_annual_values_20210629.json`
(SHA-256 `06a095b2be475f692cd14ce4bf876be93c8f51fefd1e887e0e304cd1b773c925`).
Filing date is not proof of public availability; no targets or returns were calculated.

A separate reconstruction reproduced all three counts. Of the 196 different
same-period values, 125 differ by at least 0.1%, 88 by at least 1%, and 57 by at
least 10%, each relative to the earlier filing's value. The median absolute
difference among the 196 is about 0.51%. The [aggregate record](_asset_revision_magnitudes_20261111.json)
pins the archives and the separately saved checking script. Large differences
could reflect accounting revisions, corporate actions, unit or issuer errors, or
source processing; their causes were not established. The counts are not a
representative stock-universe estimate and are not asset-growth targets.

The largest relative difference is an instructive entity-continuity example, not
evidence that assets actually grew by that amount. The [2019 Malo Holdings 10-K](https://www.sec.gov/Archives/edgar/data/1769804/000121390020008058/f10k2019_maloholdings.htm)
reports $224 in assets at December 31, 2019. The next annual filing, under the
same CIK, is [Augmedix's 2020 10-K](https://www.sec.gov/Archives/edgar/data/1769804/000121390021018844/f10k2020_augmedixinc.htm):
its comparative balance sheet reports $15,739,017 for that date. Its notes describe
an October 2020 reverse acquisition and say the private Augmedix accounting
acquirer's statements became the historical statements. The [example record](_asset_revision_extreme_example_20261111.json)
identifies the selected filings and checking script. A stable CIK and period date
therefore do not by themselves establish a continuous economic entity; callers
need an explicit corporate-action and comparability policy. This single extreme
case was selected after seeing the differences and says nothing about the causes
of the other 195.

The selected General Dynamics example now also exercises `available_facts` with
amounts transcribed from its [2019](https://www.sec.gov/Archives/edgar/data/40533/000004053320000015/gd-2019123110k.htm)
and [2020](https://www.sec.gov/Archives/edgar/data/40533/000004053321000010/gd-20201231.htm)
annual reports. With assumed next-day availability timestamps, the selector returns
$48,841 million before the second assumed timestamp and $49,349 million afterward.
The [test](../_lib/tests/test_filing_vintages.py) checks selector behavior, not the
true publication times or completeness of intervening filings. The later report
attributes the restatement to a change in pension accounting; it does not establish
that these values are comparable for the model or that GD was eligible for a trade.

An earlier version of the experimental continuity guard was exercised on four manually transcribed
asset facts from these two filings, with the October 5, 2020 merger supplied as
a known break and a June 29, 2021 fit cutoff. It returned no target; the calculation
function was replaced with a raising stub to verify it was never called. The
[record](_augmedix_continuity_guard_20261111.json) includes the selected facts,
source links and code hashes. Filing acceptance times are proxies, the year ends
are assumed, and the facts do not constitute a complete filing history. The
caller supplied the event; this check does not show that the guard can discover
events or verify their historical availability.

A [follow-up record](_augmedix_timed_continuity_guard_20261111.json) exercises
the current timestamp-aware guard on the same selected facts and fit cutoff. It
uses acceptance of the 2020 10-K that describes the merger as the event's
availability proxy and again verifies that no target calculation is called. This
proxy is later than the effective date but is not proof of actual dissemination;
the event was still supplied manually, and the same data and coverage limits apply.

One selected difference is confirmed in the source documents: DTE Energy's
[2019 10-K](https://www.sec.gov/Archives/edgar/data/936340/000093634020000127/dteenergy2019123110k.htm)
reports 2019 total assets of $41.882 billion, while its [2020 10-K](https://www.sec.gov/Archives/edgar/data/936340/000093634021000112/dte-20201231.htm)
reports $42.268 billion for the same year-end. Both match the archive's selected
facts. The reason for the $386 million difference has not been established, so it
must not be labelled a restatement or attributed to a particular transaction.
The finding confirms one discrepancy, not the accuracy or causes of the other 195.

The published balance sheets locate $385 million of the difference in prepaid
postretirement costs: $69 million in the earlier filing versus $454 million in
the later comparative. The other $1 million is in other assets ($182 million
versus $183 million). These line differences do not establish why the figures
changed. The 2020 filing's notes say some prior-year balances were reclassified
for presentation, but do not tie that general statement to this particular
difference. Choosing different available filing versions can therefore change an
asset-growth input, even when both values were extracted correctly.

For a separate timing spot-check, the saved 2021 Q2 archive contains DTE's
[April 27 10-Q](https://www.sec.gov/Archives/edgar/data/936340/000093634021000154/0000936340-21-000154-index.htm)
with a 2020 year-end `Assets` fact of $45.496 billion, matching the 2020 10-K.
The archive records acceptance at 10:16:00; the filing index records 10:15:34.
This is another example of a discrepancy between the archive time and the index,
not proof of a general rounding rule or of public dissemination at acceptance.
The check covers the selected 10-Q and two asset dates in this one archive; it
does not establish complete revision history or a usable point-in-time panel.

Combining those presence checks with the June 2021 S&P fixture gives a narrower
diagnostic: 501 fixture labels have one observed filing CIK in the two 2021
archives, and 382 of those labels have asset facts for all three exact December
dates across the three archives. The remaining 119 lack at least one date; four
other fixture labels have no exact ticker match or more than one CIK. Labels are
not independent companies (the 501 labels map to 496 distinct observed CIKs).
No association, fiscal year-end, value, eligibility or availability was verified;
the fixture may use retrospective symbols and the full NYSE reference universe
is still missing. This is not a percentage of usable model inputs. Counts and
hashes: `ggTrader-refs/sec-fsnds-2021q2/sp500_unique_label_asset_period_overlap_20210629.json`
(SHA-256 `ffc94fc24bbc62d8b66746d0dbcd82abd339e5e9937b4376f608b8010d3e9989`).

A separate [overlap check](_sp500_asset_revision_overlap_20261111.json) joins those
single-observed-CIK labels to the selected 10-K pairs used in the revision scan.
Of 501 labels, 391 have one positive, unambiguous value in each selected filing;
19 of those pairs differ, including two by at least 1% of the earlier value.
Another 100 labels have no selected annual pair and ten have missing or ambiguous
values; four fixture labels lack a unique observed CIK. Some labels share a CIK.
This is a diagnostic of selected filings and observed labels, not a verified
issuer mapping, a representative eligible universe or complete revision history.
The fiscal periods, data availability and accounting comparability remain
unverified; no targets or returns were calculated.

The [two differences of at least 1%](_sp500_asset_revision_examples_20261111.json)
illustrate why comparability needs its own policy. For General Dynamics, 2019
assets increased from $48.841 billion to $49.349 billion in the later filing;
its [2020 filing](https://www.sec.gov/Archives/edgar/data/40533/000004053321000010/gd-20201231.htm)
attributes a matching $508 million change in 2019 current assets to an accounting
principle change. This is not an identified issuer-continuity break, so the
continuity guard alone would not address it. Occidental's 2019 assets changed
from $109.330 billion to $107.190 billion; the reason remains unverified. These
are selected filing examples, not verified point-in-time security mappings.

An additional comparison narrows the Occidental question without resolving its
cause. Its [September 2020 10-Q](https://www.sec.gov/Archives/edgar/data/797468/000079746820000033/oxy-20200930.htm)
and [2020 10-K](https://www.sec.gov/Archives/edgar/data/797468/000079746821000009/oxy-20201231.htm)
both show 2019 figures. Between those presentations, trade receivables and
accounts payable each fell by exactly $2.140 billion, matching the decline in
total assets and liabilities. This is consistent with an offsetting presentation
change, but the filings reviewed do not establish that explanation; other line
items also changed between the original annual report and the 10-Q. The later
10-K was accepted on February 26, 2021, so a retrospective value must not be
used at an earlier decision time merely because it describes 2019.

The hosted [Canli fundamentals interface](https://github.com/arhancanli/canli-fundamentals-mcp)
returned the same original and later 2019 asset values for both issuers when
queried by CIK. Its date-only `history` query selected the earlier figure before
each 2021 filing date and the revised figure on the filing date itself. This is
useful as a secondary interface over SEC companyfacts, not independent source data
or a historical security universe. Its own guidance recommends waiting until the
next trading day; that convention does not independently prove public availability.
The two underlying companyfacts snapshots were fetched in September 2026, so they
cannot establish what a historical source contained at the time. Request parameters,
accessions and snapshot hashes are in the [spot check](_canli_filing_vintage_spotcheck_20261111.json);
raw responses were not retained and source-use rights remain unverified.

For a future individual-stock pipeline, the candidate policy is to preserve each
filing version and its verified public time, select only information available
at the decision, and flag known accounting-basis or identity changes for an
explicit comparability decision. Preserve the source accessions and report the
affected and missing coverage; a numerical difference alone neither proves a
fault nor establishes comparability. This is a proposed research rule, not an
implemented or validated screen. How to handle unresolved basis changes and
whether comparable multi-year figures are available remain open.

**Additional archive lead:** [US Markets TimeMachine](https://github.com/supermodo/us-markets-timemachine)
describes dated snapshots of Nasdaq symbol directories and SEC ticker-to-CIK files.
Its [specification](https://github.com/supermodo/us-markets-timemachine/blob/main/docs/SPEC.md)
says its own captures begin when each source is integrated, without third-party
historical backfill. It therefore does not establish the long historical universe
needed here. Its [data notice](https://github.com/supermodo/us-markets-timemachine/blob/main/NOTICE-DATA.md)
distinguishes code licensing from source-data terms; Nasdaq-derived data use remains
unresolved. Snapshot dates, completeness, decision-time availability and any
independent benefit over the sources already examined have not been verified.
On 2026-11-11, a directory-name check through GitHub's contents API found 125
dated files each for the two SEC ticker mappings and 177 each for `otherlisted`
and `nasdaqlisted`, dated May 15 through November 10, 2026. Counts exclude names
after November 11 and do not verify file contents, missing sessions or when the
files were committed. Those directory names cover neither the April 2024 pilot
nor the training history needed for the expected-growth model.

A separate Internet Archive CDX metadata query for the exact Nasdaq Trader
`/dynamic/SymDir/otherlisted.txt` URL found HTTP-200 captures in five calendar
months from 2020 through 2024: January 2020, January and December 2022, and April
and June 2024. The query collapsed captures by month; it did not count every
capture, search all mirrors or inspect any file contents. The April capture is
dated April 29 at 02:54:25 UTC, before the pilot's April 29 close, but that is an
archive capture time rather than proof of the directory's source refresh time or
historical accuracy. Even if these captures are usable, their sparse coverage for
this exact URL cannot supply continuous monthly panels. Nasdaq source terms and
permission to use archived copies remain unresolved. The [CDX query](https://web.archive.org/cdx/search/cdx?url=www.nasdaqtrader.com%2Fdynamic%2FSymDir%2Fotherlisted.txt&output=json&filter=statuscode%3A200&from=2020&to=2024&collapse=timestamp%3A6&fl=timestamp%2Coriginal%2Cdigest%2Cstatuscode)
records the capture timestamps and archive digests; an HTTP 200 does not prove that
the captured content is a valid symbol directory.

**Historical listing lead:** [Alpha Vantage documents](https://www.alphavantage.co/documentation/#listing-status)
a dated `LISTING_STATUS` query for active or delisted US stocks and ETFs after
January 1, 2010. That range could cover the April 2024 pilot and some training
dates, but not the full long-history standard. A public demo request for active
listings on April 29, 2024 returned an empty JSON object rather than a list; at
that initial check no account key was available. Historical exchange and ticker
accuracy, revisions, delistings, security identifiers, API limits and permitted
research use remain unverified. This is a vendor claim and a source lead, not a
validated point-in-time universe or an accounting, market-value or return dataset.
The provider's [terms](https://www.alphavantage.co/terms_of_service/) describe a
personal, non-commercial license and list circumstances it considers commercial;
they do not establish whether this project's intended use, agent access or any
redistribution qualifies. Its [pricing page](https://www.alphavantage.co/premium/)
states a standard limit of 25 API requests per day and says some functions require
premium access. At the initial check no key was created and no subscription was purchased.

**Authenticated metadata check:** Using the owner's subsequently supplied key,
the April 29, 2024 active-list request returned 10,550 rows. After ticker
normalization, 497 of 503 labels in the frozen April 3 membership snapshot appeared;
`BK`, `BXP`, `DOC`, `EQR`, `FI` and `MMC` did not. `WRK` appeared as WestRock.
The response contained 53 additional rows with repeated raw symbols; the count
of matches did not verify company identity, security type or exchange.
For November 1, 2022, `META` appeared and `FB` and `TWTR` did not; the response
contained 107 repeated-symbol rows. Requests for June 1 and June 10, 2022,
October 27, 2022 and June 3, 2019 returned non-CSV responses; their cause is
unresolved and they are not evidence of empty universes. No price or return data
was requested. The summary and response hashes are in
`alpha_listing_check_20261111.json` in the reference folder (SHA-256
`25997d10ff9ef71ba205b8af70ff0fc2371c4bf1b6506a961c7391f355221b58`); raw
responses were not retained. A successful response does not establish completeness,
historical accuracy, unrevised data, permissible use or a valid backtest universe.
In a follow-up using the supplied key, the July 10, 2014 **active** query returned
7,096 rows, including Twitter (`TWTR`). The July 10, 2014 and January 3, 2023
**delisted** queries each returned an empty JSON object; the former therefore did
not reproduce the public demo's CSV. This leaves the Twitter delisting-date test
unresolved; the empty responses are not proof of missing companies or that the
free tier cannot serve the endpoint. The metadata-only summary is
`alpha_listing_followup_20261111.json` in the reference folder (SHA-256
`acda7f9cca9b96dfd078bb19e567b481899f309fc925a94aedb12e318992f5b3`);
raw responses were not retained.
On an immediate later retry, the July 10, 2014 delisted query returned a CSV
twice, with 431 rows on the parsed response, including `HNZ` and `AGP` with the
same dates as the demo. The June 1, 2022 active retry returned HTTP 200 and an
empty JSON object; the January 3, 2023 delisted retry again returned non-CSV.
The earlier July 2014 empty response was therefore not a stable indication that
the query was inaccessible. The reason for empty responses remains unknown; a
retry must not silently turn them into empty universes.

**Ticker-change test:** A later authenticated retry for June 1, 2022 returned
11,347 active rows, with `META` present and `FB` absent. Meta's [SEC-filed
announcement](https://www.sec.gov/Archives/edgar/data/1326801/000132680122000070/may312022-exhibit991.htm)
says the switch from `FB` to `META` took effect June 9. Thus this response uses
a later ticker label for that pre-change date; it cannot be treated as a verified
historical ticker universe. The October 27, 2022 list contained Twitter (`TWTR`),
consistent with its [filing](https://www.sec.gov/Archives/edgar/data/1418091/000119312522272772/d411753d8k.htm)
that trading was suspended before the following open. June 10, 2022 active and
January 3, 2023 delisted still returned HTTP 200 with empty JSON objects, so the
post-change and delisting-state checks remain unresolved. The metadata-only
summary is `alpha_listing_event_check_20261111.json` in the reference folder
(SHA-256 `35d08832183564ff0e32814d6e96a5c0da46aa2719f9ca7e1690824dd37be23d`).
Raw responses were not retained; their hashes and local request times do not
establish independent timestamping or enable reproduction on their own.
The currently consolidated SEC 2022 Q2 notes archive provides a contrasting
filing-level check: five selected Meta filings in the archive, filed April 27
through June 1, carry the ticker fact `FB`, including its May 31 announcement
filing. No later Meta ticker fact appeared among the selected forms in that
quarter's archive. The frozen archive is in `ggTrader-refs/sec-fsnds-2022q2/`
(SHA-256 `736eb8c4a7eb7dbe8c14b980c4f955b0ae03acf1603ce1629a33db5ad428f83f`);
`meta_ticker_transition_spotcheck.json` hashes to
`b26ff7a96ca59394ec76d2bb40ffdeaa2c151f80e88434a5df49e9cdd00d0f19`. These
are selected filing facts, not proof of continuous ticker validity, actual public
availability at a decision instant, or a verified historical archive vintage.
For a June 15 decision, carrying the latest prior selected filing fact forward
would still give `FB`: the latest selected fact was filed June 1, before the
June 9 change. Thus SEC facts prevent the premature `META` label in the first
check but, without an event-aware update, can also retain the old label after a
change. This example uses only 10-K, 10-Q and 8-K rows in the one archive and
does not rule out other filings or sources.
The archive's `SUB.accepted` is not second-accurate in these two examples:
Meta's [April 10-Q index](https://www.sec.gov/Archives/edgar/data/1326801/000132680122000057/0001326801-22-000057-index.htm)
says 20:42:59 while the archive says 20:43:00; its [May 31 index](https://www.sec.gov/Archives/edgar/data/1326801/000132680122000070/0001326801-22-000070-index.htm)
says 16:18:52 while the archive says 16:19:00. These two examples are consistent
with minute rounding but do not establish a general rounding rule or timezone.
An archive timestamp must not be treated as an independently verified precise
public-availability time, and even an exact acceptance time is not proof that a
filing was publicly available at that instant.
For exchange event data, [NYSE](https://www.nyse.com/regulation/corporate-actions-market-watch-proxy-compliance)
says it publishes upcoming corporate-action notices and describes an API for
[historical events](https://www.nyse.com/market-data/corporate-actions/market-event-feed)
with a purchase link; free historical access has not been established. FINRA's
[public Daily List](https://otce.finra.org/otce/dailyList?viewType=Symbol%2FName+Change)
covers OTC equity events, so it cannot be assumed to cover the full NYSE/Nasdaq
listed universe. Neither source was queried for event rows or used for a mapping.
The authenticated June 1, 2015 active list also returned `META` and omitted `FB`,
seven years before the change. Its 7,610-row response hashed to
`b6fca6cd655db7e5434b4cccb20ef2e32206b9ecbcf077be0d41e629d5380492`;
the raw bytes were not retained. This is not merely a one-day timing discrepancy.
A later June 3, 2019 request returned an empty object, despite the earlier
successful response for that date.
For an independent historical anchor, Facebook's [2013 annual report](https://www.sec.gov/Archives/edgar/data/1326801/000132680114000007/fb-12312013x10k.htm)
states that its Class A shares were listed as `FB` on Nasdaq, and the
[SEC filing index](https://www.sec.gov/Archives/edgar/data/1326801/000132680114000007/0001326801-14-000007-index.htm)
dates that filing January 31, 2014, before the 2015 decision date. This supports
the historical label for this one issuer, but manually finding it after the fact
does not make a point-in-time identity mapping for the full universe. A filing
can also become stale after a later symbol change, requiring timely event records.

**Ticker-reuse spot check:** A June 3, 2019 active-list retry returned `S` as
Sprint; the April 29, 2024 active response above returned `S` as SentinelOne.
Those labels are consistent with [Sprint's 2019 filing](https://www.sec.gov/Archives/edgar/data/101830/000010183019000022/sprintcorp201810-k.htm)
and [SentinelOne's filing](https://www.sec.gov/Archives/edgar/data/1583708/000158370823000014/s-20230131.htm),
which says it began trading under `S` in June 2021. The June 10, 2022 active
retry again returned an empty JSON object. The 2019 CSV had 8,570 rows and
SHA-256 `618a8a229cf3c5607650907f25cce842287f36023ae62a6dd0f3c73ac32e5d8a`;
raw bytes were not retained. This pair distinguishes these two users of `S`,
but does not repair the Meta rename error or validate other reused tickers.

**Reused-label collision:** The June 29, 2021 active response has no `ABC` row
and has two active `COR` rows: Cencora and CoreSite Realty. At that time,
AmerisourceBergen, now Cencora, used `ABC` according to its [May 2021 filing](https://www.sec.gov/Archives/edgar/data/1140859/000114085921000025/abc-20210331.htm);
its [announcement](https://www.sec.gov/Archives/edgar/data/1140859/000110465923096698/tm2324358d1_ex99-1.htm)
dates the switch to `COR` to August 30, 2023. [CoreSite's filing](https://www.sec.gov/Archives/edgar/data/1490892/000155837021000757/cor-20201231x10k.htm)
identifies its own `COR` listing. A ticker-only join would therefore be ambiguous
and could attach the wrong issuer's facts or prices. Rejecting duplicate ticker rows
would avoid that silent error but would not restore the missing `ABC` label. The
metadata-only record is `alpha_cor_collision_20210629.json` in the reference
folder (SHA-256 `de54ad63a232107d82065eb145ff06ce31bf4404cfdbe10afb2a634ec7e7db45`);
raw response bytes were not retained. No returns were calculated.

The saved June 2021 Nasdaq-derived NYSE screener snapshot contains `ABC` named
AmerisourceBergen and `COR` named CoreSite Realty, consistent with those SEC
filings. Its file `nyse_2021-06.json` in the reference folder hashes to
`81602eac3d5fba68f1cb16c59089502edfa3f381cf97f227f7686b9fa1c205e9`.
The snapshot supplies company names alongside symbols, but no CIK field; it is
therefore a useful cross-check, not by itself a verified security-to-issuer map.
Its availability at a particular decision time and data-use rights remain unverified.

**Dated NYSE-label comparison:** For June 29, 2021, an Alpha Vantage active
response and the previously saved Nasdaq-derived NYSE screener snapshot were
compared by normalized symbol only. Of 3,126 unique screener labels, 2,364
appeared somewhere in the Alpha response and 2,316 appeared with its `NYSE`
exchange label. The remainder cannot be interpreted as missing or faulty stocks:
the lists can differ in ticker notation, preferred shares, funds, venue definitions,
capture timing and later name changes. Neither list is a verified eligible-stock
universe. The screener's permitted modeling use is unresolved. No prices or returns
were compared. Counts and source hashes are in `alpha_nyse_snapshot_20210629.json`
in the reference folder (SHA-256
`d823a098cc922b3361f91305031d87826dae3f87edca401acf9d673460b20b68`);
the Alpha response was not retained, so its hash alone cannot reproduce the check.
Restricting the screener side to its 1,752 unique labels whose names contain
“common stock” gives 1,569 matches on any Alpha exchange and 1,534 labeled
`NYSE`; 183 do not match an Alpha label on any exchange. This text filter is a
rough diagnostic, not a verified common-stock or eligibility classification, and
unmatched labels can be renames or notation differences. The restricted summary
is `alpha_nyse_common_name_20210629.json` in the reference folder (SHA-256
`eb987ad89d7a1b889f940520b09c0d52fa48d0e026a39647ad427fbcec406dca`).
The provider's documented demo for **delisted** symbols on July 10, 2014 returned
431 CSV rows; none had a reported delisting date after that cutoff. The response
bytes hashed to `b7a4cfadffc7cada05b6e27169a17ebd44554c6143c73f860172f7d885125b38`
at this check but were not retained. Demo requests for **active** symbols on both
July 10, 2014 and April 29, 2024 returned `{}`; the current-listing demo returned
CSV. This confirms that one documented example responds, not that the 2014 list is
complete, historically unrevised or suitable for the pilot.
Two checks against SEC filings expose limitations of that demo response. It lists
Heinz (`HNZ`) as delisted June 7, 2013, consistent with its [merger announcement](https://www.sec.gov/Archives/edgar/data/46640/000119312513258009/d555504dex991.htm),
but gives an `ipoDate` of January 2, 2001; a [2006 SEC filing](https://www.sec.gov/Archives/edgar/data/46640/000095010306001998/filename1.htm)
describes Heinz shares in 1998 and 1999. That field must not be treated as a verified
first-listing date. The demo has no `DELL` row, although Dell's [filing](https://www.sec.gov/Archives/edgar/data/826083/000119312513415302/d610376dsc13e3a.htm)
reports its shares ceased trading in October 2013. The demo may be limited or
treat a later relisting differently; this does not establish what an authenticated
full response would contain. No returns were read or computed.
The same demo assigns December 28, 2006 as `ipoDate` to 90 of its 431 rows.
For one of them, Amerigroup (`AGP`), the company's [SEC filing](https://www.sec.gov/Archives/edgar/data/1064863/000095012303010420/w89759sv3.htm)
says the shares began public trading in 2001 and moved to the NYSE in 2003.
That is a second reason not to use this field as a first-listing date; the
cause of the repeated dates has not been established. Separately, the [March 1,
2004 list](https://www.sec.gov/Archives/edgar/vprr/0402/04026181.pdf) is from
specialist firm LaBranche's annual report, not a complete NYSE directory. Its
[next annual filing](https://www.sec.gov/Archives/edgar/data/1089044/000110465905011534/a05-4938_110k.htm)
says it handled about 22% of NYSE common-stock listings at year-end 2003. The list
can support individual spot checks, but absence does not imply a stock was not
listed on the NYSE. An analogous [March 2003 list](https://media.corporate-ir.net/media_files/NYS/LAB/reports/AR02.pdf)
is in the prior annual report; these snapshots do not establish continuous membership.

**Other free metadata lead:** [Market History's documentation](https://getmarkethistory.com/docs/guides/historical-stock-market-data)
describes a free symbol-discovery endpoint with inactive symbols and consolidated
US bars from 2016. On November 11, 2026, its exact `XLNX` metadata endpoint
reported an inactive Nasdaq security with 2016–2022 coverage, while the exact `WRK`
endpoint returned HTTP 404. Its `MMC` response described IEX-feed coverage, not a
verified NYSE listing history. These are three metadata checks, not a coverage
audit; a 404 on one endpoint does not prove the security is absent everywhere.
The provider says daily and minute bars require purchased access, and its daily
bars include extended hours, so its daily close is not necessarily the exchange
closing print. No bars or returns were requested. Identity history, source rights,
2016 coverage and the relation between observed trades and listing membership
remain unverified.
The no-key consolidated one-minute symbol catalog reported 15,771 entries and
`truncated=false` when checked on November 11. Of the 503 fixture members at the
April 29, 2024 pilot cutoff, 480 normalized exact symbols appeared, all with the
cutoff between their reported first and last timestamps. The other 23 were absent,
including `MMC`, `WRK`, `BRK-B` and `BF-B`; alternate ticker spellings and feeds
were not resolved. First/last timestamps do not prove a bar on the date, continuous
coverage, correct identity or membership. The response SHA-256 was
`a6307fc6d084191cc1c7fb1fa929bc14affe016546950c03ac1defa029ca117c`;
the response was not retained, and the changing catalog cannot be reproduced from
the hash alone. No market prices or returns were read.
The catalog spells two of those missing labels `BRK.B` and `BF.B`, and each
reported first/last timestamp spans the pilot date. Treating dots and hyphens as
potential class separators yields 482 label candidates and 21 unmatched labels
after that substitution, but this normalization is only a lead: neither security identity nor a bar
on the pilot date has been verified.

**Next check if access is available:** Request the April 29, 2024 active list and
compare its exact symbols with the 503-member frozen fixture using the April 3
membership snapshot. Report missing and extra labels, exchange and asset-type
coverage, duplicates, response hash, retrieval time and the API parameters without
recording the key. Check `WRK` and `MMC` separately against dated corporate records;
do not substitute today's ticker. Compare at least one earlier date and an acquired
or delisted name against independent dated records. A matching S&P 500 subset does
not establish a complete NYSE reference universe, point-in-time publication or
unrevised history. Do not calculate returns or use the response for modeling until
the applicable data-use terms are understood.

**External report check:** The submitted source survey provides useful leads but
its proposed validation dates need correction. Meta's [announcement](https://www.sec.gov/Archives/edgar/data/1326801/000132680122000070/may312022-exhibit991.htm)
sets the `FB` to `META` change at June 9, 2022, so June 1 is a pre-change check.
Twitter's [filing](https://www.sec.gov/Archives/edgar/data/1418091/000119312522272772/d411753d8k.htm)
says trading was suspended before the open on October 28, 2022; the report's
November 1 active-trading expectation is incorrect, even though formal removal
was [scheduled for November 8](https://www.sec.gov/Archives/edgar/data/876661/000087666122000890/ruleprovisionnotice.htm).
Trading suspension, legal delisting and the vendor's status definition are distinct.
The survey's claim that the SEC submissions API gives dated ticker and exchange
history is also unsupported: the [SEC describes](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
current ticker/exchange metadata and former **names**, not dated ticker histories.
Its suggestion that cover-page shares automatically provide market value by class
needs validation: [Alphabet's filing](https://www.sec.gov/Archives/edgar/data/1652044/000165204424000022/goog-20231231.htm)
reports three classes, including an unlisted class, while the SEC says its company
facts APIs aggregate facts applicable to the entire filing entity. No API key or
stock returns were used in these checks.

**Listing-history bound, revised:** The initial 2023–2025-only index showed 15 prior
months for the April 2024 pilot. A broader search found the NYSE file at a repository
snapshot in each month of 2021–2022: 24 files with 3,040–3,310 rows, distinct hashes,
and no blank or duplicate symbols. Combining these sampled files with the later
commit-month index gives 39 prior calendar months for April 2024 and 30 for July
2023. This supersedes the earlier July 2025 optimistic bound. It is still only a
calendar bound: a commit is not a regression fit, and source refresh time, complete
universe, accounting inputs, identity and availability at each decision are
unverified. The source files and `monthly_snapshot_audit_2021_2022.json` are in the
reference folder (audit SHA-256
`2c3a3169a3829376b65b36d47322c3a01752870e90cdb2c499f68c6f945f81a8`); the revised
calendar is `screener_forecast_history_calendar_v2.json` (SHA-256
`75f5a8bc4b57c1266056e155917898482985a44cb31daa5fefed5f9927f8c8d9`).

The corresponding Nasdaq files also exist at all 24 sampled root commits. They
contain 3,689–5,054 rows per file, with no blank or duplicate exact symbols. Between
0 and 67 rows per file have a missing, `n/a`, empty or zero `marketCap` field under
the recorded check; this is not a check that the remaining fields are valid or
timely. The hashes, URLs, commit times and counts are in
`nasdaq_snapshot_coverage_2021_2022.json` in the reference folder (SHA-256
`2699da97330da0e74a7dccfb479286cd7b6ad043fd3c27adaa28fa833ebc927f`). Files were
read for these counts, not saved locally. Their presence does not establish complete
NYSE/Nasdaq coverage, correct exchange or security classification, same-time quotes,
or eligibility for a point-in-time model. No stock returns were calculated.

**Nasdaq quote-timing check:** For Apple, the seven sampled quarter-end commits
from March 2021 through September 2022 were created before the final session's
close, and each saved last sale differs from the quarter-end close in a later
[SEC-filed pricing supplement](https://www.sec.gov/Archives/edgar/data/1419828/000095017023039956/aaplco09_final.htm).
The December 2022 commit was after the final close and its $129.93 last sale
matches that supplement. The comparison and source hashes are in
`nasdaq_aapl_quarter_end_check_2021_2022.json` in the reference folder (SHA-256
`3ffbf0caee1867c73050becc6d97de5fea296c802a9e61e42e07e808bc4dbe2b`). The filing
is retrospective verification, not an input available at formation. The seven
pre-close files cannot themselves represent the final close; later commits might,
but their availability and quote refresh still require checking. One matching
symbol does not validate the rest of the panel or establish quote timestamps.

**Post-close follow-up:** For all eight quarters, the first root commit after the
regular close and before the next regular open contained an Apple last sale matching
the supplement's quarter-end close. The source URLs, hashes, commit times and
comparison are in `nasdaq_post_close_aapl_probe_2021_2022.json` in the reference
folder (SHA-256 `933b94fdc7e4540d4b8b4ae65e5ebcf7821ccd05c4ccfd3cf2d2d2f504e843f6`).
This checks one symbol in eight files; root commit times are not independently
witnessed publication or quote times, and a root commit need not modify the Nasdaq
file. Other symbols, fields, months, exchange coverage and usage rights remain
unverified. A file first observed after the close cannot support trading at that
same close; a possible next-session implementation would need its own execution
and price assumptions. No returns were calculated.

The same eight post-close root commits contain NYSE, Nasdaq and AMEX files: respectively
3,046–3,179, 3,999–4,980 and 274–294 rows per file. No exact symbol appears twice
within a file or in two exchange files at the same commit. The simple missing-or-zero
market-cap check flags 431–469 NYSE, 1–17 Nasdaq and 18–23 AMEX rows per file; it
only recognizes null, empty, `n/a`, `N/A`, numeric zero and the string `0`. Counts,
file hashes and URLs are in `post_close_exchange_universe_probe_2021_2022.json` in
the reference folder (SHA-256
`ea4963bf34c6d610c1ba06dca5b63e979d001cc27db9e2a658b89a7ba1cb990e`). Exact
symbols are not issuer identities; other missing encodings, stale values, ineligible
securities, omitted listings and exchange changes remain unchecked. A root commit
does not prove each exchange file refreshed after the close.

For those eight commits, a [GitHub commit-metadata check](_post_close_commit_changes_20261111.json)
found each of the three exchange files listed as changed; the returned file lists
had no next page. This narrows the concern that a root commit changed only unrelated
files. It does not establish when the underlying data was fetched, when a quote was
formed, when the commit became public, or that every row changed. Commit times are
not independent witnesses; source rights, field accuracy and model suitability
remain unresolved. Only commit metadata was inspected, not additional stock rows.

A separate NYSE spot check compared `PNW` in two of those files with closing
prices in Pinnacle West's later [SEC-filed 2023 proxy](https://www.sec.gov/Archives/edgar/data/764622/000130817923000547/lpnw2023_def14a.pdf).
The snapshot values matched the filing's $64.51 for September 30 and $76.04 for
December 30, 2022. The [probe record](_pnw_post_close_spotcheck_20261111.json)
pins the two snapshot URLs and response hashes; it did not retain the raw files.
The later filing is a retrospective cross-check, not proof of information available
when the commits were made. One selected symbol and two dates do not validate other
rows, the market-cap field, capture timing, source rights or a research universe.
No return or ranking was calculated.

For the eight NYSE files, a text-only check matched 1,899–2,058 rows per file
whose name contains `common stock` or `ordinary shares`; 0–3 of those matched
rows per file had a missing-or-zero market-cap field under the same limited check.
The other flagged rows have names outside that text match. The input hashes and
counts are in `post_close_nyse_name_cap_probe_2021_2022.json` (SHA-256
`87d95cdd5b6a5d932b0ec12beab69d8ee3ef78977c129201b16a663eee51fd60`). This is
not an eligible-stock coverage estimate: names may omit those words, identify
funds or other ineligible securities, or change over time. Present market-cap
fields may also be stale or invalid.

**Usage rights:** The [archive README](https://github.com/rreichel3/US-Stock-Symbols)
describes the full files as raw Nasdaq-list data. GitHub's repository license endpoint
returned 404 when checked; that does not establish what rights the archive owner may
have. [Nasdaq's current website terms](https://www.nasdaq.com/legal) include
restrictions on copying content for data-analysis software and on sharing or creating
derivatives. Whether those terms apply to this third-party historical mirror, and
whether any separate permission exists, has not been determined. The diagnostic
checks above used the mirror, not Nasdaq's website. Treat use of its data for model
development or a return study as unresolved until the applicable rights are verified;
the source cannot currently be assumed usable for that purpose.

**Additional source checks (metadata only):**

- [Carnegie Mellon research archive](https://www.cs.cmu.edu/~pavlo/datasets/stocks/):
  The publisher describes exchange-specific symbol/name lists as of January 2008
  and cautions that the major-exchange lists may be incomplete. The page describes
  separate 1970–2006 price and volume history with missing observations, but does
  not establish a monthly dated listing universe, share counts or market-equity
  breakpoints. The snapshot could be a lead for a bounded historical cross-check,
  not a source of 30 verified training panels. Original sources, allowed model use,
  historical publication timing and accuracy remain unverified; no files or returns
  were downloaded for this check.

- [Valuein](https://github.com/valuein/valuein) describes a no-signup sample
  covering S&P 500 members, including former members, for five calendar years
  plus year-to-date, and a free registered tier with longer history. This could
  support a narrow cross-check of filing facts, not the full NYSE breakpoint
  universe. Its [methodology](https://github.com/valuein/valuein/blob/main/docs/METHODOLOGY.md)
  uses SEC acceptance timestamps; acceptance does not independently prove public
  dissemination. Its [price documentation](https://github.com/valuein/valuein/blob/main/docs/BRING_YOUR_OWN_PRICES.md)
  uses user-supplied prices and filed share counts, so it does not independently
  solve historical market-value, security-class or adjustment verification. The
  [service terms](https://valuein.biz/terms) need review for this project's intended
  model and artifact use; the repository's code license is not a data license.
  Provider claims, coverage and revisions have not been tested. No account was
  created, data downloaded or returns calculated.

  Follow-up: the terms say internal use and certain private predictive models are
  permitted, but downloading or using even the free sample accepts the data licence;
  raw redistribution is restricted and retention after termination is limited.
  Check the incorporated licence schedule and the intended use before obtaining it;
  do not commit raw sample data to the repository. The provider's price guide says
  it returns no multiple when a company has multiple listed share classes and only
  one class price is supplied, and warns that a missing split cannot be detected.
  These are documented safeguards and limitations, not independently tested results.
  A bounded future check could compare a known filing and revision against the SEC
  and this pipeline, including timestamps and raw concepts, without claiming either
  vendor's historical snapshot proves dissemination. A sample-manifest request via
  the web tool failed, so the actual sample contents and current vintage remain unknown.

- [Distill Markets](https://www.distillmarkets.com/) describes ten free as-of
  single-company lookups per day, but its full historical universe screen and
  panel export require a paid tier. The free endpoint could cross-check selected
  SEC accounting histories; it does not solve the free bulk-universe problem.
  Its timing semantics, data quality and terms have not been independently checked.
- [DeltaScreener](https://deltascreener.com/data) offers dated monthly snapshots
  of active stocks, with attribution requested, and identifies Yahoo Finance,
  SEC, Financial Modeling Prep and Alpha Vantage as underlying sources. A current
  snapshot is not evidence of delisted coverage or historical point-in-time
  vintages; whether the stated permission covers all underlying rights is unverified.
- The [JKP data pipeline](https://github.com/bkelly-lab/jkp-data/blob/main/README.md)
  states that generating its stock-level dataset requires a WRDS subscription and
  the underlying vendor licenses. Its freely available portfolio factors do not
  by themselves supply the dated security identities and market values needed here.

No account was created, data downloaded or return calculation made for these checks.

**Alpha Vantage share-count semantics:** The provider's [documentation](https://www.alphavantage.co/documentation/)
describes quarterly basic and diluted share counts. In one public demo response,
the Microsoft row dated March 31, 2024 reports 7.431 billion basic and 7.472
billion diluted shares. These match the **quarterly weighted-average EPS shares**
in Microsoft's [10-Q](https://www.sec.gov/Archives/edgar/data/789019/000095017024048288/msft-20240331.htm),
not a demonstrated point-in-time share count. That filing's cover reports
7,432,305,794 actual shares as of April 22; that is a different date and measure.
The [filing index](https://www.sec.gov/Archives/edgar/data/789019/000095017024048288/0000950170-24-048288-index.htm)
dates the filing April 25, after the vendor row's period end. A period-end label
does not establish that the figures were public at the period end. The [check record](_alpha_msft_shares_semantics_20261111.json)
preserves the selected facts and response hash; the demo response was not retained.
This single numerical match does not establish the provider's source, methodology
or other rows' semantics. Do not use these values as dated share-class counts or
derive market equity from them without separate verification. The provider's
[terms](https://www.alphavantage.co/terms_of_service/) distinguish personal and
commercial use; the permissions applicable to this project remain unverified.

**Alpha Vantage price-history constraint:** The provider's [documentation](https://www.alphavantage.co/documentation/)
says free daily requests return at most the latest 100 observations; full daily
history requires premium access. It describes a longer monthly series, but a
monthly close cannot establish a daily stop's intramonth trigger or fill. The
provider's [support page](https://www.alphavantage.co/support/api-key/) states
25 free requests per day for most datasets. Even if every call succeeded, 500
single-symbol requests would take at least 20 days at that stated allowance, before
retries or other calls. Those are documentation-based bounds, not tested coverage;
delisted histories, adjustment basis, permitted use and point-in-time availability
remain unverified. No price requests or return calculations were made for this check.

**Predictor-lag bound:** If January 2021 is the earliest usable predictor panel and
no earlier source is available, the method's 12-month predictor lag makes January
2022 the earliest possible fit month. The prototype excludes the decision month's
fit, so January 2022 through March 2024 supplies at most 27 prior monthly fits for
the April 2024 pilot, below its minimum of 30. Under these optimistic assumptions,
July 2024 is the earliest decision month with 30 prior fit months. This is only a
calendar upper bound: the first snapshot may not have been available at that month's
formation time, and no monthly predictor panels or fits have been verified. Earlier
valid source history could change the bound. The April date remains useful for input
diagnostics, not a ready forecast.

The actual saved January snapshot was committed January 30, 2021 at 21:36:25 UTC;
the saved root-commit history starts that day, and the repository's [current GitHub
metadata](https://api.github.com/repos/rreichel3/US-Stock-Symbols) reports creation
on January 30. It therefore cannot serve as an available beginning-of-January panel.
If February 2021 is the first possible beginning-of-month panel from this source,
then February 2022 is the earliest fit month and only **26** fit months precede April
2024; August 2024 becomes the earliest decision month with 30 prior fits. This is
still optimistic and conditional: the January snapshot's underlying refresh and
formation-time availability are not verified, and another historical source could
provide earlier panels. No fitted model is implied.

**Earlier candidate source:** The `abbadata/stock-tickers` repository has NYSE files
at commits dated October 31, November 28 and December 31, 2020. Its [historical
script](https://raw.githubusercontent.com/abbadata/stock-tickers/c4fcc1f28529faf74c1e56cc61905bdc9b82d56b/tickerlist.py)
identifies an old Nasdaq screener as the source. The three saved files contain 3,212,
3,254 and 3,282 rows, respectively; 841, 858 and 877 rows have an empty or `n/a`
market-cap field. Displayed values include rounded amounts such as `$1.02B`, rather
than established full-precision market values. The files, commit dates, hashes and
field counts are recorded in `abbadata_2020_listing_probe.json` in the reference
folder (SHA-256 `9cac871f581edc0cbe6756fea8396e21bb22a7b8dfb0271146ae27abb78c70fd`).
This is a lead for earlier listings, not a verified point-in-time universe: source
refresh, exchange classification, completeness, field precision and usage rights
are unresolved. These three files do not establish 30 valid model fits.

An exact-label comparison of the December 2020 file with the January 2021 screener
file found 2,753 shared labels. The files are from different dates, so differences
can reflect genuine listing changes as well as source conventions. For example,
the earlier file uses `BRK.A` and `BRK.B`, while the later one uses `BRK/A` and
`BRK/B`; exact matching misses those known share classes. Of the 877 December rows
with `n/a` market cap, 728 have a symbol containing a caret, dot, slash or hyphen.
Punctuation alone does not establish whether a row is a fund, preferred share,
warrant or eligible common stock. The input hashes and exact comparison are in
`abbadata_rreichel_label_comparison.json` in the reference folder (SHA-256
`83b620c56e3501a76e7d1f825f2ca2e5cafb61c8515d1b77676df04b60a62a63`). No ticker
mapping or universe exclusion follows automatically from this check.

If the 2020 files and the 2021 source could eventually be reconciled, an optimistic
calendar calculation would start predictor panels in November 2020 and fits in
November 2021. That gives at most **29** prior fit months by April 2024; May 2024
would be the earliest month with 30. This is conditional on continuous, timely and
compatible data across the source change, which has not been established. In
particular, the December file's commit time was before the final December trading
close, and its underlying quote time is unknown. The calculation does not certify
any fit or use the archive for a stock selection.

The same repository also has NYSE-file commits on January 1 and January 2, 2021,
after the December close and before the next regular session. Both files show IBM's
last sale as $125.88, matching the last trading-day close stated in IBM's later
[2021 proxy](https://www.ibm.com/investor/att/pdf/IBM_Proxy_2021.pdf). This is one
retrospective cross-check, not proof of every field's timing or accuracy. The files,
commit timestamps and hashes are recorded in `abbadata_dec2020_close_probe.json`
in the reference folder (SHA-256
`6a4303d31ff8b0b8d1d83600d0bd46f82a3d646bd0914c58cd9f555ec4b6dee1`). The proxy
is used only for later verification, not as information available at the 2021 cutoff.

One alternative checked, the [Tidy Finance factor library](https://huggingface.co/datasets/tidy-finance/factor-library),
describes historical portfolio-return series and trading-cost estimates constructed
from CRSP and other inputs. Its published files contain series identifiers, months
and aggregate results; the card does not offer the historical individual-security
market-value and exchange panel needed here. Its historical coverage therefore does
not resolve this input gap. Its many construction specifications are not independent
unseen tests; no series was downloaded or evaluated in this check.

For the separate [Open Source Asset Pricing firm-level predictors file](https://huggingface.co/datasets/OpenSourceAP/CrossSection/blob/dd1054254c4501776432caa6d7ce5ecd9586f6a9/Firm%20Level%20Characteristics/Predictors.parquet),
a footer-only schema read found `permno` and `yyyymm` among 211 columns, but no
plainly named market-equity, ticker or current-exchange field. `ExchSwitch` is a
signal name, not itself proof of an exchange label. The full column-name list,
source revision and footer hash are in `osap_predictors_schema_check.json` in the
reference folder (SHA-256
`8c218695924b5b801e6a7fbabd7d3045aaaacd79e3d95b9df3cfd77fba8e0f63`). This checks
names only: it does not rule out deriving a useful quantity from other fields, or
establish point-in-time identity, exchange, size coverage or data rights. No rows or
returns were read in this check.

One potential indirect route is the file's `SP` signal. The provider's [pinned
construction code](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/Predictors/SP.py)
defines it as Compustat sales divided by contemporaneous company-level CRSP market
equity. Recovering the denominator would require the exact same numerator, units,
security-to-company mapping and release transformation; whether these can be
reproduced point-in-time is unverified. `BM` instead uses a [fiscal-year-end market
value carried forward](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/Predictors/BM.py),
so it cannot simply serve as each month's size. These are code-level leads only;
release-to-code correspondence and actual values were not checked.

The provider's [pinned packaging code](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Shipping/Code/1_build.py)
describes `Predictors.parquet` as unsigned and writes the sign-adjusted signals to
a separate file. It also says direct `Size` and other WRDS-derived exclusions are
not written. That supports investigating the ratio's interpretation but does not
establish that this release was produced by exactly that code or that reconstructing
and using an underlying CRSP quantity is permitted. Verify source rights, matching
inputs and vintages before using the idea; no `SP` data was read here.

For one identity cross-check, [SAS's CRSP example](https://support.sas.com/documentation/cdl/en/etsug/68148/HTML/default/etsug_sasexccm_gettingstarted.htm)
identifies IBM as PERMNO 12490; a [WRDS sample](https://wrds-www.wharton.upenn.edu/demo/crsp/data/)
also pairs that identifier and ticker in November 2006. [CRSP](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-data-descriptions-guide-crspaccess/)
describes PERMNO as permanent for an issue's trading history, but these examples do not independently
establish IBM's listing status at the 2020 formation date. This single mapping is not
a historical universe mapping and does not resolve the ratio's input or rights gaps.

CRSP's [10-year workbook guide](https://www.crsp.org/wp-content/uploads/2023/08/CRSP10-User-Guide.pdf)
describes a potentially relevant panel: monthly capitalization, shares and name and
exchange histories for active and inactive securities. However, CRSP's [January 2025
release note](https://www.crsp.org/wp-content/uploads/mdaz_202501.pdf) describes
access for existing stock-database subscribers; it does not establish a free public
dataset. WRDS likewise says its [demo output](https://wrds-www.wharton.upenn.edu/demo/crsp/data/)
is a fixed sample and full queries require a subscription. Neither is a verified
free replacement for the missing historical universe. No workbook or returns were
downloaded for this check; a later data vintage would still need a point-in-time
availability assessment.

A separate [Market Cap Explorer repository](https://github.com/datadrivendecisions/marketcap)
publishes a global top-100 list by year and displays an MIT repository license. Its
[source notes](https://github.com/datadrivendecisions/marketcap/blob/main/DATA_SOURCES.md)
say most historical years were assembled retrospectively in 2026 from third-party
site histories, older lists and manual additions; the underlying capture emphasizes
current large companies. The project license alone does not establish rights to all
underlying sources. Annual global top-100 snapshots also cannot supply the full
monthly NYSE breakpoint or the eligible US stock universe. This is a possible
manual cross-check for selected companies, not a replacement panel; no data values
or stock returns were examined.

The SEC's [archived Section 13(f) security lists](https://www.sec.gov/rules-regulations/staff-guidance/official-list-section-13f-securities)
are a potential historical identity cross-check, not a market-cap or exchange panel.
The [first-quarter 2021 PDF](https://www.sec.gov/divisions/investment/13f/13flist2021q1.pdf)
includes issuer descriptions and security identifiers but also options and debt.
Its information sheet says the list is current March 15, while its printed run date
is April 2. The run date cannot establish availability for an April 1 decision;
an independent publication timestamp and any earlier version would be needed.
The PDF reserves rights in the identifiers and descriptions and prohibits
redistribution without permission. Whether the planned research use is permitted
requires separate verification. The list alone supplies neither a complete eligible
universe nor prices or exchange membership. No security rows were extracted.

[OpenFIGI's API documentation](https://www.openfigi.com/api/documentation) describes
a free identifier-mapping API and an `includeUnlistedEquities` option; listed-only
is the default. Its [allocation rules](https://www.openfigi.com/about/allocation-rules)
say updated ticker and other reference data become associated with existing FIGIs.
Neither statement establishes a dated historical ticker-to-issuer mapping or what
was known at an earlier decision. It is a possible identifier cross-check, not a
replacement for a point-in-time universe. No API mapping was requested here.

The SEC's [Financial Statement and Notes Data Sets documentation](https://www.sec.gov/files/aqfsn_1.pdf)
describes as-filed, nonnumeric XBRL text (`TXT`) joined to filing metadata (`SUB`)
by accession. This may provide a bulk way to investigate historical ticker facts
against issuer CIKs, including for companies no longer listed. It is a lead, not a
verified mapping: the documentation does not establish that `TradingSymbol` is
present, complete or correct in a particular release. The SEC's [dataset page](https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets)
says updates became monthly in November 2020 and that older files have been revised.
Release timing, filing acceptance and security-class identity need separate checks
before a historical decision could use a row.

In a two-issuer check of the current consolidated 2021 Q2 archive, all six
`TradingSymbol` facts for AmerisourceBergen's CIK 1140859 say `ABC`; both facts
for CoreSite's CIK 1490892 say `COR`. These come from eight filings, including
one 10-Q for each issuer, and match the independently checked filings above. The
archive and accession-level metadata are preserved under
`ggTrader-refs/sec-fsnds-2021q2/`; the archive SHA-256 is
`fbbc99dff34cdd1b898a7e5b34b66b89fcf313bc4a620b2248a9ae663cb97d22` and
`trading_symbol_spotcheck.json` hashes to
`6e78ff13daec02dbeb074370a569592cc10b30fad23b7bac63efab8ab686e5e7`.
This small, selected example is not a coverage estimate. A filing fact does not
establish continuous listing, intraday availability or a complete historical
universe, and this consolidated archive is not a verified 2021-vintage copy.
No stock returns were calculated.

For a broader label-only check, exact `TradingSymbol` facts from 10-K, 10-Q and
8-K filings dated April 1–June 28, 2021 in that archive were compared with the
3,126 normalized symbols in the June 2021 NYSE snapshot. There were 1,684 symbol
matches: 1,680 appeared with one CIK in the selected filings and four appeared
with multiple CIKs. Another 1,442 snapshot labels had no match in that interval.
This is not an accuracy or universe-coverage estimate: firms need not file one of
those forms in a quarter, and ticker matches do not prove issuer identity. Periods
were changed to hyphens on both sides; availability times, share-class identity,
eligibility and snapshot rights were not checked. The metadata-only counts and
input hashes are in `ggTrader-refs/sec-fsnds-2021q2/ticker_coverage_20210629.json`
(SHA-256 `9f10d526701eeea35ef0b0a2ba303ba640aec6cfa05bec360b63a4dae6c7e874`).
The later [metadata-only checker](code/sec_ticker_coverage.py) reproduces its counts,
ambiguous labels and input hashes. It uses only the supplied archive, so using one
quarter does not reconstruct the entire available filing history. Its hashes are
recorded after reading the inputs and do not independently prove a past vintage.

On a second date, the saved May 2022 NYSE snapshot (associated root commit
`57db3be79e536385dc479744061d1f10805aad53`, timestamp 2022-05-31 00:17:33 UTC)
contains 3,120 normalized labels. Against selected 2022 Q2 SEC filings dated before
June 1, 1,777 labels match a ticker fact; seven appear with multiple filing CIKs
and 1,343 have no exact match. The raw metadata result is
`ggTrader-refs/sec-fsnds-2022q2/ticker_coverage_20220601.json` (SHA-256
`2a36a9f72d15e81f55540b63be08e6552746f3fa79196bfd13358d97deca9e28`). A
commit timestamp alone does not prove source capture time or publication, and
the snapshot's permitted modeling use remains unresolved. Counts do not prove
issuer identity, eligibility, historical availability or absent filings.

Adding the current consolidated 2021 Q1 archive to Q2 raised the matched label
count from 1,684 to 1,694 of 3,126: ten labels were found only in Q1, while 1,564
were present in both quarters. Twelve matched labels appeared with multiple CIKs
across the two quarters, up from four in Q2 alone. These are label comparisons,
not valid issuer mappings: an older filing's symbol can be stale by June 29.
The result and both archive hashes are in
`ggTrader-refs/sec-fsnds-2021q2/ticker_coverage_q1_q2_20210629.json` (SHA-256
`af818d13a7ca1d584d51aa586ef4fa4798c8fd2e64b82508474c39224218ec92`). The Q1
archive is preserved in the adjacent `sec-fsnds-2021q1` reference folder. Filing
coverage, class identity, true publication time and source rights remain distinct
questions; no returns were calculated.

Among those 1,694 matched labels, the latest matching filing date was 1–30 days
before June 29 for 907 labels, 31–60 days for 734, 61–90 days for 47, and more
than 90 days for six (maximum 138). This is the age of the most recent observed
filing in the two archives, not the age or verified validity of the association.
The 1,432 unmatched labels are excluded from these bins. Metadata and input hashes
are in `ggTrader-refs/sec-fsnds-2021q2/ticker_match_filing_age_20210629.json`
(SHA-256 `b559c25a8e9cfc2c46c3bc543182219cce673d0ec6643acac5555bfd9245ba2a`).

Restricting the same snapshot to names containing `common stock` gives 1,752
labels, of which 1,379 match at least one ticker fact across the two SEC quarters;
one matched label has multiple filing CIKs and 373 have no match. This text filter
is not a validated common-stock classification: it can include funds and omit
common shares under other names. It is not evidence that unmatched issuers were
missing from SEC filings. The counts and input hashes are in
`ggTrader-refs/sec-fsnds-2021q2/ticker_common_name_q1_q2_20210629.json`
(SHA-256 `5023b57c13204cb4408183edfef798a35f398fbea5dfb9c4e392bb158fce7f7f`).

For a narrower cross-check, the frozen S&P membership fixture's latest snapshot
on or before June 29, 2021 is dated June 4 and contains 505 normalized labels.
Across the two SEC archives, 502 labels match at least one filing ticker; one of
those (`APA`) appears with multiple filing CIKs, while three labels have no match.
These are label counts, not verified company matches. The fixture can use
retrospective ticker labels, and S&P membership does not supply the full eligible
NYSE reference universe. Metadata and hashes are recorded in
`ggTrader-refs/sec-fsnds-2021q2/ticker_sp500_fixture_q1_q2_20210629.json`
(SHA-256 `62142d0d0ed768614e383c1ba09a50bec806a5c43789f9e16b897a651b9a5c40`).

The three unmatched fixture labels are `BAX`, `BF-B` and `FRC`. For `BAX`,
the SEC archive does contain Baxter International filings with `TradingSymbol`
values `BAX (NYSE)`, which the conservative exact-label comparison does not
normalize to `BAX`; the same filings also list values such as `BAX 24` for other
instruments. Baxter's [April 2021 filing exhibit](https://www.sec.gov/Archives/edgar/data/10456/000162828021008032/bax-20210331xex991.htm)
identifies its NYSE ticker as `BAX`. This example shows why a missing exact match
is not an absent filing; stripping arbitrary suffixes would also risk conflating
other instruments. For `BF-B`, Brown-Forman's CIK 14693 has `BFB` and `BFA`
ticker facts in both quarterly archives; its [June 2021 10-K](https://www.sec.gov/Archives/edgar/data/14693/000001469321000091/0000014693-21-000091-index.htm)
is among the filings. A separately filed [2021 fund record](https://www.sec.gov/Archives/edgar/data/837274/000083727421000026/ftvipn-px0621.htm)
uses `BF.B`. This supports a notation difference in this example but does not
justify stripping punctuation from arbitrary symbols. For `FRC`, neither archive
contains a selected 10-K, 10-Q or 8-K under CIK 1132979; this is not proof that
the company had no filings elsewhere or that the SEC data contains no facts for it.
An [October 2021 SEC-hosted disclosure](https://www.sec.gov/Archives/edgar/data/1053092/000095010321016206/dp160120_424b2-u6243.htm)
describes First Republic's reports as filed with the FDIC; its own 2022 proxy
also says its 10-K was filed with the FDIC rather than the SEC. This is a
plausible explanation for the gap, not verification of the 2021 Q1/Q2 FDIC
filing records. The published expected-growth sample excludes financial firms,
so this example does not establish an eligible-universe omission for that model;
it matters for any broader stock universe that includes banks.

The `APA` ambiguity has a documented corporate-action explanation: [Apache's
March 2021 filing](https://www.sec.gov/Archives/edgar/data/6769/000119312521063792/d129139d8k.htm)
says APA Corporation became its parent and replaced Apache as the publicly traded
company under `APA`. That is evidence about this transition, not a general rule
for choosing among multiple CIKs or a validated mapping of the historical tape.

The four ambiguous labels are `ELC`, `ENJ`, `ENO` and `QSR`. Their SEC rows
include related registrants, so a submission's CIK must not automatically be
assigned as the owner of every ticker fact in it. For example, the [April 30
Restaurant Brands 8-K](https://www.sec.gov/Archives/edgar/data/1618755/000161875521000017/qsr-20210430.htm)
names both the corporation and its limited partnership while showing `QSR` for
common shares and `QSP` for partnership units. The corporation's separate [10-Q](https://www.sec.gov/Archives/edgar/data/1618756/000161875621000022/qsr-20210331.htm)
identifies `QSR` as its common shares. This is a specific filing-level example,
not a resolution of all four labels; a robust mapping must account for the XBRL
context, registrant and security class rather than treating a shared cover-page
symbol as unambiguous evidence of ownership.

In these examples the SEC archive retains useful context: joining `TXT.dimh` to
`DIM.dimhash` gives `LegalEntity=RestaurantBrandsInternationalInc.` for the
partnership-submitted `QSR` fact. The Entergy parent 10-Q's `ELC`, `ENJ` and
`ENO` facts have dimensions naming subsidiary legal entities and mortgage-bond
classes, illustrating that a listed symbol need not be common stock. The four
selected contexts are in `ggTrader-refs/sec-fsnds-2021q2/ticker_context_spotcheck.json`
(SHA-256 `8fa8b2b9f04faf842d782c7502325a6e081b24958ad01087613bceb8bb259d7a`).
These filer-provided names are useful leads, not a verified general-purpose
security classification or an automatically resolved CIK mapping.

Across the 7,418 ticker-fact rows matching the 1,684 snapshot labels in the
same-quarter comparison, seven rows have a nonempty `coreg` field and 2,369
have nonzero dimensions. All 257 distinct nonzero dimension hashes resolved in
the archive's `DIM` table. Exact segment-name checks found `LegalEntity` on 46
rows, `ClassOfStock` on 2,173 and `EntityListingsExchange` on 342; these counts
overlap and repeated rows are not independent securities. They describe field
presence, not correct identity or classification. Missing these exact field names
does not prove the original filing lacks the information. Counts and input hashes
are in `ggTrader-refs/sec-fsnds-2021q2/ticker_context_coverage_20210629.json`
(SHA-256 `e7221603fc38e21772c2bcf5459745a0ca8e9a6d68f55ccea9524489096c7b57`).

Only 504 of the 1,684 matched ticker labels have any exact `ClassOfStock`
segment in these selected rows. The segments contain 204 distinct values,
including common shares and debt; `CommonStock` occurs on 265 distinct labels
and `CommonClassA` on 155, with possible overlap. Thus requiring this exact field
or one literal value would drop many matched labels before eligibility could be
determined. The absence of a segment does not establish that the original filing
lacks a class description. The full value counts and input hashes are in
`ggTrader-refs/sec-fsnds-2021q2/ticker_class_values_20210629.json` (SHA-256
`420f596978efde413cc03d13cc256ece25dc74911ca5bcc78f9ae889aa3d594d`). No class
was inferred from a name or admitted to a research universe by this check.

The related `Security12bTitle` text is a stronger metadata lead. Joining on exact
accession and XBRL context in the same sample gives one nonempty title string for
7,395 of 7,418 matched ticker-fact rows, none with multiple strings, and 23 with no
title. At least one single-title row exists for 1,681 of 1,684 matched labels;
490 labels have more than one distinct title string across their rows. Examples
include the 2021 `ABC`, `COR` and `QSR` filings above. Counts and input hashes
are recorded in `ggTrader-refs/sec-fsnds-2021q2/ticker_title_context_20210629.json`
(SHA-256 `8b29a0d60b16c79735ee32365729475efa639795c118ec293a5bfec2ca5403ba`).
An exact context match is not proof that a title identifies the same security;
different strings can be equivalent, and missing joins can have other causes.
No title was automatically classified, and no issuer mapping or universe was
approved by this check.

In a broader filing-context probe of the same Q2 archive, 23,888 distinct
accession/context pairs in selected 10-K, 10-Q and 8-K filings had a nonempty
`TradingSymbol`. Of those, 23,482 had one distinct nonempty
`SecurityExchangeName` in the same context, and 23,421 had exactly one ticker,
exchange and `Security12bTitle`. The remaining ticker contexts included 406
without a matching nonempty exchange. These are filing-context counts, not
unique securities or evidence of a complete universe. They show a possible way
to cross-check a reported exchange alongside a ticker, but do not establish
continuous listing, correct security class or public availability. The archive
is retrospective. The [probe record](_sec_exchange_context_probe_20261111.json)
includes examples, input and scanner hashes; no returns or targets were calculated.

Joining those fields from Q1 and Q2 to the rough `common stock` name subset of
the June 2021 NYSE snapshot gives 1,370 of 1,752 labels with one observed CIK
and at least one complete context reporting the exchange string `NYSE`. Of those
1,370, 21 also have another exchange string and 631 have multiple title strings
across matched contexts; spelling and timing differences can contribute. Another
373 labels have no ticker match, one has multiple observed CIKs, seven have one
CIK but no complete context, and one has complete contexts with no exact `NYSE`
string. These categories describe the selected files and are not identity or
eligible-universe classifications. Counts, method and hashes are in the
[overlap record](_sec_exchange_overlap_probe_20261111.json).

Expanding that metadata scan to every form in the same two archives increases
matched labels from 1,379 to 1,443; 309 still have no ticker match. Of the 64
additional labels, 52 occur in a `20-F`, 11 in a `40-F`, one in a `20-F/A`
and one in a `10-Q/A`; a label can occur in more than one form. The broader
scan also increases labels with multiple observed CIKs from one to two. Counts,
scanner hashes and input hashes are in the [all-form record](_sec_all_form_overlap_20261111.json).
The SEC describes [Form 20-F](https://www.sec.gov/about/forms/form20-f.pdf) and
[Form 40-F](https://www.sec.gov/about/divisions-offices/division-corporation-finance/financial-reporting-manual/frm-topic-16)
as foreign-issuer forms, so more ticker matches do not necessarily mean more
eligible domestic stocks for Global-q. Form type alone does not establish a
security's CRSP eligibility, issuer identity, primary listing or availability.
This is a retrospective metadata comparison, not a universe or a return test.

Among the 309 unmatched labels, the snapshot names contain `fund` for 123,
`acquisition` for 37, `bank` for 10 and `trust` for 45; these case-insensitive
substring counts overlap and do not classify securities. For one example,
the snapshot calls `AIF` common stock, while its [2021 SEC filing](https://www.sec.gov/Archives/edgar/data/1526697/000119312521165281/d179951d424b3.htm)
identifies Apollo Tactical Income Fund as a closed-end fund. The rough name
filter therefore includes at least this fund; the 309 unmatched labels cannot
be interpreted as 309 missing eligible stocks. The [unmatched record](_sec_unmatched_all_forms_20261111.json)
contains the labels, keyword counts and scanner hashes. No automatic exclusions
follow, and individual eligibility and source rights remain unverified.

**Fund-classification leads:** The SEC's [closed-end fund reports](https://www.sec.gov/data-research/sec-markets-data/closed-end-fund-information)
list registrant CIKs and names, with annual editions including July 2020 and
July 2021. The SEC says the list can include funds before public offering or
after operations cease, and its listed fields do not include ticker symbols.
The July 2021 edition postdates the June 29 pilot; the 2020 edition predates it
but cannot establish that the list stayed current or was available at the
decision time. Historical CIK-to-security mapping remains necessary.

The SEC's [Form N-CEN datasets](https://www.sec.gov/data-research/sec-markets-data/form-n-cen-data-sets)
include quarterly archives back to 2018. Their [schema](https://www.sec.gov/files/ncen_readme.pdf)
describes a submission table with accession, CIK and filing date, and a
closed-end security table with fund identifier, class title, exchange and ticker.
This is a useful cross-check for the unmatched labels, not a complete universe:
fields can be missing, annual reports and amendments need timing and identity
checks, and the SEC warns of extraction errors and omitted metadata. The SEC
[announced the flattened datasets in 2023](https://www.sec.gov/newsroom/whats-new/osd-announcement-230602-form-ncen-data-sets),
so today's 2021 files are retrospective even where the underlying filings
predate the pilot. The SEC also reports a current schema-version omission.
The subsequent metadata-only cross-check is described below; no returns were calculated.

For one concrete cross-check, an SEC-indexed [N-CEN filing](https://www.sec.gov/Archives/edgar/data/1526697/000114554921010989/primary_doc.xml)
for Apollo Tactical Income Fund reports CIK `0001526697`, investment-company type `N-2`,
ticker `AIF` and exchange `XNYS`, with a December 2020 reporting period and
February 25, 2021 signature date. The [example record](_ncen_aif_indexed_example_20261111.json)
transcribes those indexed fields. A later check downloaded the original XML and
filing index, confirming those fields and the February 25 filing date. The index's
acceptance time does not establish when the filing became public. This shows the relevant fields can occur together, not that
the ticker's June 2021 status or the whole archive's coverage is verified.

**Retrospective N-CEN cross-check:** The 2021 Q1 SEC archive, retrieved November 11,
2026, was matched by exact ticker label against the June 2021 screener's 1,752
unique names containing “common stock.” The [reproducible record](_sec_ncen_q1_common_name_20261111_v2.json)
contains 105 rows covering 98 labels from N-2 registrants, with filings dated
before June 29. The archive was updated in 2023 and was not available in this
form at the decision date. The `PMM` match has conflicting class labels in the
source filing, as checked below. These are leads for identity checks, not verified fund counts
or automatic universe exclusions. One quarterly archive cannot establish complete
coverage, contemporaneous availability, or the correctness of ticker mappings;
the screener's historical availability and redistribution rights remain unverified.
The inputs are the [SEC 2021 Q1 archive](https://www.sec.gov/files/dera/data/form-n-cen-data-sets/2021q1_ncen.zip)
and the frozen `nyse_2021-06.json` screener; their hashes and the driver hash are
in the record. The [driver](code/ncen_common_name_audit.py) checks the hashes
before reading either file and retains the matching rows for inspection.

**Four-quarter coverage check:** Extending the same exact-label comparison to
the SEC's 2020 Q3 through 2021 Q2 archives yields **172 distinct labels across
184 matching rows**, including 74 labels absent from the Q1-only matches. The
[combined record](_sec_ncen_four_quarter_common_name_20261111.json) retains the
source hashes, per-quarter audit hashes, filing dates and matched rows. Only
filings dated July 1, 2020 through June 28, 2021 are included; no same-day
filing-date availability is assumed. The four quarterly archives carry 2023
last-modified dates and do not prove what was accessible at the decision time.
Annual filings can report older periods, an accession need not establish a
stock's active status on June 29, and labels and share classes remain unverified.
The increased count measures the sensitivity of this diagnostic to archive
coverage, not the number of actual funds or a universe to exclude. No returns
were read or calculated.

For the AIF example, the original [SEC XML](https://www.sec.gov/Archives/edgar/data/1526697/000114554921010989/primary_doc.xml)
and [filing index](https://www.sec.gov/Archives/edgar/data/1526697/000114554921010989/0001145549-21-010989-index.html)
were saved locally with SHA-256 hashes `28c31166494c67f16a86c0962bfcc6e0cfc58003787d7b258c7cb7aa41ebc8bd`
and `ce0095fc3e2625dcbbef4c3935b613ffc5de8a86b51c4cbc210813d55cc93b6f`, respectively.

**PMM original-filing check:** The January 6, 2021 [filing index](https://www.sec.gov/Archives/edgar/data/844790/000086939221000016/0000869392-21-000016-index.html)
and [original XML](https://www.sec.gov/Archives/edgar/data/844790/000086939221000016/primary_doc.xml)
identify the N-2 registrant by CIK `0000844790`. In the same `securityRelatedItem`,
the description says `Preferred stock` and the class title says `PREFERRED`, while
the nested `commonStock` element gives `PMM` on `XNYS`. The SEC's [2020 Form N-CEN](https://www.sec.gov/files/formn-cen-info.pdf)
asks for common and preferred securities separately. The XML therefore preserves
an ambiguity in the original filing, rather than proving that the PMM ticker is a
preferred share or that the flattening process made the mismatch. Neither field
alone verifies the class. The locally saved XML and index hash to
`fe991d8e5c9c298d96c3ce3aaf4b90a6b4b1cd1a773c8f5f0af2f0b148cf45bc` and
`2ef2845965e14a7091b2bc484ee315541f6a9c5616a2e810544ee4267fb3334c`, respectively.

**PMM independent class corroboration:** The [MIAX Pearl exchange's Form 1 amendment](https://www.sec.gov/Archives/edgar/vprr/2100/21000461.pdf)
includes a securities list described as of June 24, 2021. On PDF page 289, it
labels `PMM`, Putnam Managed Municipal Income Trust shares of beneficial
interest, `CommonStock`. This supports the screener's common-share label despite
the conflicting N-CEN fields. A separate [2007 SEC-filed prospectus](https://www.sec.gov/Archives/edgar/data/844790/000092881607001346/a_mgdmuniprox.htm)
identified PMM as the fund's NYSE common shares; the 2007 statement alone does
not establish a continuous NYSE listing through 2021. The MIAX list concerns
MIAX securities, not NYSE membership, and its SEC-hosted PDF has a July 8, 2021
last-modified date, so this is retrospective corroboration, not proof the list
was available at the June 29 decision. The local PDF is
`/home/flynn/ggTrader-refs/sec-ncen-20261111/miax_pearl_symbols_20210624.pdf`,
SHA-256 `84ef04f8947a5499ef7f42670c8822a8c48402123dfabbf9e52d0195874a9abb`.
**MIAX label scan:** A conservative text scan of the equity-table pages recognizes
5,712 distinct `CommonStock` labels. Exact-label matches include 1,683 of the
screener's 1,752 common-name labels and all 172 four-quarter N-CEN fund leads.
The [record](_miax_common_labels_20261111.json) pins the inputs and lists the
matched fund-lead labels; the [driver](code/miax_common_labels.py) retains only
recognizable rows. This reinforces that common-stock wording does not establish
operating-company eligibility. The individual matches beyond PMM have not been
verified against issuers; identical labels do not prove identity, and missed rows
do not prove absence. The document's retrospective timing and MIAX-only scope
remain as above. Rights to model with the screener and MIAX list remain unverified.

**Issuer-name collision:** The June 2021 screener gives both `CMU` and `MFM` the
name “MFS Municipal Income Trust Common Stock.” The MIAX list instead names
`CMU` MFS High Yield Municipal Trust and `MFM` MFS Municipal Income Trust. A
[2020 SEC-filed report](https://www.sec.gov/Archives/edgar/data/809844/000119312520196959/d833449dncsrs.htm)
also identifies MFS High Yield Municipal Trust as `CMU`; a separate
[2019 SEC filing](https://www.sec.gov/Archives/edgar/data/1140392/000101359419000721/muniincome-ex991-112219.pdf)
lists `CMU` and `MFM` under those distinct fund names. This is a concrete name
conflict in the frozen screener, not proof of a ticker change or of the cause.
A ticker-only join can carry the wrong issuer name into an accounting-data
mapping, so the screener name alone must not establish issuer identity. These
documents and the retrospective MIAX list do not prove continuous listing or
point-in-time availability at the pilot decision.

**Name-review queue:** Across the 172 exact-label N-CEN leads, a conservative
comparison found normalized screener and registrant names agreeing for 128 labels;
44 have at least one disagreement. The [review record](_ncen_name_review_20261111.json)
preserves both sets of names, and the [driver](code/ncen_name_review.py) pins its
input and comparison rule. The rule removes the exact ` Common Stock` suffix,
normalizes case and punctuation, and does not equate words with symbols: for
example, “and” versus “&” can trigger review. Thus 44 is not a count of errors,
and agreement for 128 does not prove issuer or share-class identity. A ticker
with both agreeing and disagreeing filing names is included in the review queue.

The two multi-CIK labels are `NNA` and `REV`. The `NNA` ticker fact under
Navios Maritime Holdings has an equity-method-investee context and no matching
exchange or class in that context; Navios Maritime Acquisition's separate
[annual report](https://www.sec.gov/Archives/edgar/data/1437260/000119312521134842/d131813d20f.htm)
is another source in the sample. For `REV`, the [joint annual report](https://www.sec.gov/Archives/edgar/data/887921/000088792121000003/rev-20201231.htm)
names Revlon, Inc. and Revlon Consumer Products Corporation and states that the
parent owns all of the subsidiary's common shares. These examples show why a
ticker appearing under two filing CIKs is not by itself proof of ticker reuse
or two listed securities. The [context record](_sec_all_form_collision_contexts_20261111.json)
preserves the matched filing metadata; it does not establish a verified mapping
or the complete historical relationships.

The 21 multi-exchange labels split into 15 with `NYSE` and `CHX`, four with
`NYSE` and `NASDAQ`, and two with `NYSE` and `NYSEARCA`; see the
[string breakdown](_sec_exchange_variations_20261111.json). These are reported
strings, not 21 verified transfers. For example, IBM's [January 2021 filing](https://www.sec.gov/Archives/edgar/data/51143/000110465921009428/tm214501d1_8k.htm)
lists the same capital stock on both the New York and Chicago exchanges.
Consequently, a future identity join must allow concurrent exchange listings and
separately establish the primary-listing convention used for NYSE breakpoints.
The IBM example does not establish the history or listing status of the other
20 labels; the source and historical timing limitations above still apply.

The [July 2026 Global-q methodology](https://global-q.org/uploads/1/2/2/6/122679606/factorstd_2026jul.pdf)
specifies `primaryexch=N` for NYSE in its CRSP CIZ sample, along with
`conditionaltype=RW` and `tradingstatusflg=A`. Its stock-sample footnote also
specifies domestic common-stock flags; its expected-growth sort uses NYSE
breakpoints. The locally inspected methodology PDF hashes to
`9dbfd5f691809a45dbffb95f5684de7f69d157a70fcdbbe4907e4a517ac93adf`.
The [CRSP guide](https://www.crsp.org/wp-content/uploads/2023/08/CRSP10-User-Guide.pdf)
describes `PRIMEXCH` as the security's primary listing. A filing's list of all
registered exchanges is therefore insufficient by itself to reproduce the
primary-exchange flag; the prototype's `nyse` input requires that flag or a
validated equivalent. Neither source establishes a free, point-in-time history
of those flags for this pilot. The PDF also excludes a monthly CRSP price from
month-end market equity when its price date is not the final trading day;
prices and date availability still need verification in any alternative source.

Two post-hoc examples show that multiple exchange strings can reflect real
transfers. Chart Industries [announced](https://www.sec.gov/Archives/edgar/data/892553/000119312521010763/d16168dex991.htm)
its move from Nasdaq to NYSE effective February 1, 2021; the selected filing
contexts report Nasdaq on January 19 and NYSE on February 2. Insteel's later
[annual report](https://www.sec.gov/Archives/edgar/data/764401/000143774921024481/iiin20211002_10k.htm)
says its transfer took effect March 17; the selected contexts report Nasdaq on
March 3 and NYSE on April 22. Carrying forward only those selected filing labels
would therefore be stale around these transfers. Other filings or timely exchange
notices may narrow the gaps; the later annual report verifies the example
retrospectively, not what a historical decision could know. The
[example record](_sec_exchange_transfer_examples_20261111.json) contains the
selected contexts and hashes; these two examples do not measure an error rate.

For GTLS, the January announcement supplies the intended February 1 effective
date; a separate [MIAX circular](https://www.miaxglobal.com/sites/default/files/circular-files/MIAX_Options_RC_2021_08.pdf),
dated February 1, also describes the primary-listing transfer as effective that
day. This suggests an event-aware mapping could keep Nasdaq through January 29
and use NYSE from February 1, instead of waiting for the February 2 filing label.
That is a retrospective interpretation of the identified sources: neither the
archive filing date nor the circular's printed date proves when the information
was publicly available to a particular decision. A future mapping would need to
retain both a verified availability time and the effective date, and handle
cancelled or corrected announcements. Neither source establishes complete event
coverage; no automatic identity mapping or research-universe change follows.

The SEC's [exchange-delisting page](https://www.sec.gov/rules-regulations/exchange-delistings)
identifies Form 25-NSE as an EDGAR source for exchange-filed removals. Issuers
also file Form 25: Insteel's [Form 25](https://www.sec.gov/Archives/edgar/data/764401/000143774921006027/iiin20210310_25.htm),
signed March 15, 2021, identifies its Nasdaq common-stock withdrawal. This
filing does not itself establish the NYSE transfer date or its public-availability
time; the separate transfer evidence above is still needed. Its accession is not
in the saved Q1 financial-statement-and-notes archive, and the prior ticker scans
selected only 10-K, 10-Q and 8-K forms. A systematic event source would need to
check both issuer-filed Form 25 and exchange-filed 25-NSE, amendments, security
classes and actual event dates; these examples do not establish comprehensive
coverage, accurate availability or an executable universe.

**Quarterly-index check:** The SEC's frozen 2021 Q1 master index contains 450
`25-NSE` rows but only 225 distinct accessions; each occurs under two indexed CIKs.
For example, the same accession appears under Dime Community Bancshares and Nasdaq.
It also contains the previously identified Insteel 8-A12B and Form 25. The
[audit](_sec_master_index_q1_2021_20261111.json) pins the archive and counts; the
experimental [parser](../_lib/sec_master_index.py) retains all indexed parties per
accession. A CIK, form or index row does not identify the affected security class
or establish a listing event. The SEC says [quarterly indexes are rebuilt](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
to incorporate later corrections, so this 2026 capture is not proof of the index
contents or availability at a 2021 decision.

**Cross-vintage format check:** The same parser processed 2010 Q1 and 2024 Q1
captures. Their `25-NSE` rows likewise occur twice per accession: 270 rows / 135
accessions and 424 / 212, respectively. The [audit](_sec_master_index_cross_vintage_20261111.json)
pins both files and counts. These samples also contain `8-A12G` and `8-A12G/A`,
which this reader does not select. The SEC's [Form 8-A](https://www.sec.gov/files/form8.pdf)
distinguishes registrations under sections 12(b) and 12(g); their presence is
not by itself evidence of NYSE eligibility. Three quarters do not validate all
historical index formats, revisions, event types or class mappings.

**Index-to-notice spot check:** Before retrieving XML, five Nasdaq `25-NSE`
accessions were selected in sorted order from the 2021 Q1 index. All five SEC
XML responses parsed and their issuer CIKs matched the non-Nasdaq indexed party.
Their descriptions include two common-stock notices, one notes notice and two
unit notices. The [selection](_sec_form25_nasdaq_sample_selection_20261111.json)
and [results](_sec_form25_nasdaq_sample_20261111.json) pin the inputs and outputs;
the original XML is also covered by offline tests. This convenience sample is not
representative: it shows that form and issuer alone do not classify the security.
The [subsequent index check](_sec_form25_nasdaq_index_checks_20261111.json) found
four indexes link the previously retrieved XML path. The fifth uses the issuer CIK
in its path instead of the exchange CIK; both SEC paths returned identical XML
bytes. The selection's local timestamp is not independently witnessed. No event
dates, public availability or complete universe are established.

**Date evidence in the same sample:** Each of the five indexed `EX-99.25` text
files contains only its filename stem inside the document wrapper; none supplies
an event date. The SEC index displays an “Effectiveness Date” equal to the filing
date in all five, but that field has not been established as the last tradable
session. The SEC [distinguishes formal Form 25 delisting effectiveness](https://www.sec.gov/rules-regulations/staff-guidance/corporation-finance-interpretations/exchange-act-forms)
from the filing date. In a separate retrospective source, National General's
[annual filing](https://www.sec.gov/Archives/edgar/data/1578735/000157873521000032/nghc-20201231.htm)
describes its acquisition and common-share delisting on January 4, 2021; the
[filing index](https://www.sec.gov/Archives/edgar/data/1578735/000157873521000032/0001578735-21-000032-index.html)
dates that report February 26, so it cannot by itself be treated as information
available on January 4. The [exhibit check](_sec_form25_nasdaq_exhibits_20261111.json)
and [timing note](_sec_form25_event_timing_20261111.json) preserve these limited
observations. They do not establish the last tradable time, all applicable
corporate-action proceeds or a point-in-time universe.

Insteel also signed an [8-A registration statement](https://www.sec.gov/Archives/edgar/data/764401/000143774921005809/iiin20210305_8a12b.htm)
on March 12, 2021, identifying the same common-stock transfer from Nasdaq to
NYSE. Together, the 8-A and Form 25 describe the destination and departure,
respectively; neither document by itself establishes the March 17 trading
effective date or proves when its content became public. The earlier bulk scan
excluded both form types. Treat these forms as event-source leads, not as
complete, point-in-time listing records.

Security class is essential: IBM's saved current submissions root lists 12
`25-NSE` filings in its recent block, but a checked [November 2025 notice](https://www.sec.gov/Archives/edgar/data/51143/000087666125000849/xslF25X02/primary_doc.xml)
concerns maturing notes, not IBM common stock. Treating the issuer's Form 25 as a
company-wide delisting would be wrong. The same submissions row uses the relative
document path `xslF25X02/primary_doc.xml`; the current periodic selector accepts
only simple filenames and should not be extended to this form without separate,
safe path and class handling. The [example record](_sec_form25_security_class_example_20261111.json)
pins the local metadata and describes its limited scope; older submissions pages
were not checked.

The SEC's indexed structured 25-NSE examples include `issuer`, `exchange` and
`descriptionClassSecurity` fields, as shown by the [IBM notes notice](https://www.sec.gov/Archives/edgar/data/51143/000087666125000849/primary_doc.xml)
and a [common-stock notice](https://www.sec.gov/Archives/edgar/data/1222840/000087666121001526/primary_doc.xml).
The experimental `sec_form25` helper extracts those fields from supplied X0202
or X0203 XML and checks the issuer CIK. A [2006 SEC example](https://www.sec.gov/Archives/edgar/data/30371/000087666106000220/primary_doc.xml)
shows the same fields under X0202; this does not establish coverage of every
historical notice. The helper preserves the security description as text;
it does not classify the security, validate provenance, or infer public or event
timing. Tests include simplified structures transcribed from SEC-indexed
[NYSE](https://www.sec.gov/Archives/edgar/data/1030561/000087666106000234/primary_doc.xml)
and [Nasdaq](https://www.sec.gov/Archives/edgar/data/1354457/000135445726000831/primary_doc.xml)
notices as well as invented XML. They are not the original downloaded bytes; the
helper has not been independently reviewed. Issuer-filed HTML Form 25s are outside
its scope.

**Original-file follow-up:** With the owner's supplied SEC contact, three original
NYSE notice XML files were retrieved from the SEC and added as hash-checked offline
test fixtures. The helper extracts IBM's notes and Retail Properties' preferred and
common descriptions correctly for these samples. For the 2021 common-stock notice,
the SEC filing index links an exchange-filer CIK path; its XML matches the issuer-CIK
path byte for byte. The [evidence record](_sec_form25_original_files_20261111.json)
records the source URLs, hashes and index check. These samples do not establish
complete coverage, publication time, class identity beyond the reported description,
or effective trading dates; acceptance time is not a verified dissemination time.

An experimental offline binding check also reads both original Retail Properties
index pages. It checks the requested accession and unique raw XML path against a
frozen index hash, checks the XML against its separately supplied hash, then parses
the issuer and description. Tests reject swapped same-issuer notices, changed index
bytes and ambiguous links. The index does not itself supply the XML hash: callers
must independently establish where both files came from and which document hash
belongs to the accession. The check does not establish public availability, an
effective trading date or complete event history.

Matching the issuer alone is insufficient to identify a filing: Retail Properties
of America has a [2017 preferred-stock notice](https://www.sec.gov/Archives/edgar/data/1222840/000087666117000796/xslF25X02/primary_doc.xml)
and a separate [2021 common-stock notice](https://www.sec.gov/Archives/edgar/data/1222840/000087666121001526/xslF25X02/primary_doc.xml).
Before integration, the caller must independently bind the document bytes to the
selected accession and preserve their provenance; a CIK match does not do that.
The [example record](_sec_same_issuer_distinct_form25_20261111.json) transcribes
the rendered notices, not downloaded or hashed XML. Neither example establishes
publication time, the effective trading date, or a complete listing history.

The experimental `recent_listing_filings` helper inventories candidate 8-A12B,
25 and 25-NSE forms and amendments in a supplied SEC submissions recent block,
including safe nested document paths. It includes the filing-date cutoff and
records whether the supplied current snapshot advertises older pages; it does
not read those pages. A [local IBM probe](_sec_recent_listing_probe_20261111.json)
found the checked notes notice among 16 candidates through November 7, 2025;
older pages exist. That record's code hash predates the later page refactor.
Tests use synthetic metadata. This is an inventory, not a
security-class classifier, event-effective-date or publication-time verifier,
or a complete universe, and it has not been independently reviewed.

The related `listing_filings` helper requires every referenced submissions page
and inventories all candidate forms at or before the filing-date cutoff. It
rejects duplicate accessions and, by default, page-range discrepancies; an
explicit inspection mode reports discrepancies without resolving them. On the
saved BlackRock root and three referenced pages, strict mode refused two prior
range discrepancies; inspection mode found one candidate and reported both.
The [probe](_sec_listing_all_pages_probe_20261111_v2.json) pins the current
code and inputs. It reproduces the earlier result after a code-formatting and
docstring change; the older artifact records the earlier code hash. Neither
mode verifies event identity, publication timing, or source completeness.

For those two BlackRock page discrepancies, SEC filing indexes corroborate the
recorded [November 5, 2025](https://www.sec.gov/Archives/edgar/data/2012383/000119312525267043/0001193125-25-267043-index.htm)
and [March 3, 2025](https://www.sec.gov/Archives/edgar/data/2012383/000119312525044184/0001193125-25-044184-index.htm)
filing dates; each is one day after its supplied page's `filingTo`. This supports
the dates for these two rows, not the completeness or provenance of the pages.
The one candidate is an [8-A for 3.750% notes](https://www.sec.gov/Archives/edgar/data/2012383/000119312525072466/d899867d8a12b.htm),
with BlackRock, Inc. and BlackRock Finance, Inc. both named; it is not a common
stock event. The [example record](_sec_listing_page_discrepancy_examples_20261111.json)
keeps those links and page bounds. Filing acceptance is still not independent
proof of public availability at a decision time.

As a crude language probe, among the 1,681 labels with a single-title row,
1,588 have at least one title containing `common` or `ordinary`, and 165 have
at least one title containing `preferred`, `bond`, `note`, `debenture`,
`warrant`, `unit` or `depositary`; 86 labels occur in both groups and 14 in
neither. These are case-insensitive substring checks across each label's observed
titles, not security classifications. For example, a title can describe rights
or multiple classes, and `unit` can occur inside an unrelated word. The probe
and its input hashes are recorded in
`ggTrader-refs/sec-fsnds-2021q2/ticker_title_keyword_probe_20210629.json`
(SHA-256 `a25e1f98ec5daf18fd1c0ba5b3dc0449438994e343abaa691e4de09cc320c300`).
No label is accepted or excluded on this basis.

Ticker-fact extract dates need different treatment from accounting-period end
dates. In the 7,418 matched Q2 2021 rows, all rounded extract dates parsed as calendar
dates: 5,049 preceded the filing date, 136 equaled it and 2,233 followed it.
For example, an American Express filing dated April 23 has an extract date of
April 30; the original context was not checked in that example. A rounded extract
date is not itself a publication timestamp or a proven
ticker-effective date, so applying an accounting-period-end sanity rule to these
cover-page facts would reject many rows without establishing that the facts are
wrong. Counts and examples are in `ggTrader-refs/sec-fsnds-2021q2/ticker_context_dates_20210629.json`
(SHA-256 `7eab4e5f910769c229ced2ce9e36086cad31550f598e66ab0de311167298054a`).
This does not remove the need for independently verified publication time and
security identity.

Some differing title strings are cosmetic: Berkshire's `BRK.B` appears with
both `Class B Common Stock` and `ClassB Common Stock`. Others can represent
different classes: Kansas City Southern's [April 16, 2021 8-K](https://www.sec.gov/Archives/edgar/data/54480/000005448021000113/ksu-20210416.htm)
lists `KSU` for both its common and its preferred shares, and the archive contains
both title/context pairs. This is a filing claim, not independent proof of the
preferred security's trading symbol. A symbol-and-CIK pair alone cannot resolve
which security a price series represents; retain context and corroborate the
instrument before calculating any return. The examples were selected after the
aggregate check and do not estimate the prevalence of either issue.

The provider's [allocation manual](https://www.openfigi.com/docs/figi-allocation-rules.pdf)
adds an important limit: after an equity ticker change, the existing FIGI is
associated with the new ticker and the old ticker is no longer associated with
that FIGI. A stable FIGI therefore does not by itself provide the dated old-ticker
lookup needed to resolve the `ABC`/`COR` example. A separately dated mapping or
filing evidence is still needed; the manual does not establish historical API
coverage or prove what information was available on a past decision date.

The [Stock-Data project](https://github.com/TylerJForstrom/Stock-Data) also
publishes a derived listing-history archive from SEC and Nasdaq Trader directory
snapshots. Its [manifest](https://raw.githubusercontent.com/TylerJForstrom/Stock-Data/main/data/symbols/pit/manifest.json)
sets July 28, 2026 as the earliest reconstructable date for all four listed sources.
That excludes the 2021–2024 pilot and does not supply historical market values. It
may be useful for a future prospective identity check, after verifying the files,
their timestamp semantics, completeness and terms. The manifest claims a government
public-domain basis for all four sources, but two URLs are Nasdaq Trader directories;
that blanket licensing claim needs independent verification. No source rows or
returns were read for this metadata check.

[Sharadar's free sample](https://sharadar.com/sample) describes five years of
fundamentals and prices for 30 Dow companies. It may support a narrow accounting
cross-check, but cannot supply the full historical NYSE reference universe. The
provider's [terms](https://sharadar.com/terms) specify an individual personal-use
license and conditions on retaining and using data; applicable permissions would
need checking before a download or inclusion in research artifacts. The sample's
point-in-time revision history and historical constituent selection have not been
verified. No sample data or returns were requested.

Two additional repositories are possible sources of methods or narrow cross-checks,
not verified inputs. The [tenicho pipeline](https://github.com/tenicho/data-cleaning)
now describes a Sharadar-based build and says its exchange field is current rather
than point-in-time; it therefore cannot by itself reconstruct historical NYSE
breakpoints. Its build instructions require a Sharadar key and say bulk downloads
require an entitled plan. The [Johnbrick123 S&P 500 project](https://github.com/Johnbrick123/sp500-data)
describes a historical membership and price panel and lists halt and exit-gap
exceptions. Its claims, source rights, identity mapping and data have not been
verified, and an S&P-only panel is not the full NYSE reference set. No dataset,
price row or return calculation was used in this metadata review.

The [abbadata stock-tickers repository](https://github.com/abbadata/stock-tickers)
describes exchange-specific Nasdaq-derived lists. GitHub's commit metadata for
`data/nyse.csv` identifies [January 23, 2021](https://github.com/abbadata/stock-tickers/commit/bea2aa9d26b909c5eb46a535451b3fe58f9f7b04)
as its last update before both the June 2021 and April 2024 pilot dates; a query
found no later commits to that path through April 30, 2024. That file is too stale
to establish either pilot universe. Commit timestamps and a repository's update
claims do not independently verify source availability, completeness or rights.
Only repository and commit metadata were checked, not the stock rows or returns.

The [Stonks/tickers repository](https://github.com/Stonks/tickers) offers dated
exchange-labelled files, but its file tree is not a current-listing snapshot. The
[October 2023 tree](https://github.com/Stonks/tickers/tree/50c5e4a0d0c3943d663d40f3963ae2f345a67c69/tickers/nyse)
still contains `TWTR.csv`, although the NYSE reported that Twitter trading was
suspended before the open on October 28, 2022, following its acquisition
([SEC filing](https://www.sec.gov/Archives/edgar/data/876661/000087666122000890/ruleprovisionnotice.htm)).
File existence does not establish a listing or tradability on the tree date.
The [metadata record](_stonks_universe_metadata_20261111.json) describes the
check; no price rows or returns were examined. Git commit times and the source's
coverage, identity mapping and usage rights remain unverified.

A [third-party 2017 WIKI price-coverage index](https://gist.github.com/aaronhoffman/9ef359cbf39f2eefdf50fbd82f5d368a/2e4b5cfc9b74e6264d8344359f74d8ba283d9d5e)
contains 3,194 ticker labels and first/last observation dates, including some
that end before 2017. It has no `HNZ` label, although Heinz traded before its
[2013 acquisition](https://www.sec.gov/Archives/edgar/data/46640/000119312513258009/d555504dex991.htm).
This is a check of one unattributed metadata copy, not proof that every WIKI
version omitted Heinz; neither inclusion of delisted names nor observation-date
ranges establishes a complete dated universe. The [check record](_wiki_metadata_check_20261111.json)
preserves the hash and selected observations. The source vintage, coverage and
research-use rights are unverified; no price rows or returns were examined.

Another possible cross-check is [Quant-Lodge's ticker reference project](https://github.com/Quant-Lodge/ticker-reference-data),
which describes a ticker-rename cache and a historical trading-halt archive.
Its documentation says old halt files can change and that names or symbols may
be current rather than historical. The described rename-cache fields include no
effective date, so that description cannot establish a past universe. Its filing-date-only
claim for float data also does not prove intraday availability: the SEC explains
that [filing dates and dissemination timing differ](https://www.sec.gov/submit-filings/filer-support-resources/how-do-i-guides/determine-status-my-filing).
These are source leads only; data rows, completeness, timing, identity and use
rights were not checked, and no returns were calculated.

**Delisting-date caution for another source:** [Black Falcon Data's catalog](https://github.com/BlackFalconData-org/delisted-stocks-list)
describes SEC Form 25 and Form 15 records and says it computes a Form 25 effective
date as ten **business** days after filing. In contrast, an [NYSE delisting notice
filed with the SEC](https://www.sec.gov/Archives/edgar/data/1413159/000095017024019130/tgh-ex99_2.htm)
describes the usual interval as ten **calendar** days; the [SEC form instructions](https://www.sec.gov/about/forms/form25.pdf)
also state ten days and describe amendment handling. A computed date must not be
treated as a verified event date without checking the applicable notice, amendments
and exceptions. The legal effective date is also distinct from the last tradable
session or suspension date. This is a review of the catalog's stated method, not
a test of its records, coverage, permissions or implementation. No source data or
returns were downloaded for this check.

In a provisional name-string check on those 24 files, 93.0%–99.8% of rows containing
`common stock` or `ordinary shares` had a positive reported market cap. January 2021
was the weakest: 1,834 of 1,971 such rows. The name filter includes funds and misses
some common stocks; positive numbers do not prove freshness or valid eligibility.
Counts and input hashes are in `common_name_market_cap_coverage_2021_2022.json` in
the same folder (SHA-256
`9276fd52c0ac9141219455b035744bf87d7ef9070bc8e9d184ff492d1fe1bf01`).
