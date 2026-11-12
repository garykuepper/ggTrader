# Three ways to study protection

**2026-11-10 — exploration only.** The owner asked to explore all three measures. The numbers below are invented illustrations, not forecasts, targets or study results. A future confirmatory study must choose its primary measure in advance or account for multiple opportunities to pass.

**2026-11-11 comparison choice:** For the next protection proposal, the owner chose
80% SPY / 20% Treasury bills alongside SPY. This selects the comparison weights;
the bill instrument and rebalance rules still need specification. It does not approve
a study, start a shadow, or change trading.

**2026-11-11 primary measure:** The owner chose loss in the same down years as
the primary measure. Drawdown and recovery remain reported measures. The specific
rule for defining down years still needs to be fixed in the study brief.

**Observation-period choice:** The owner selected the draft's 2027–2029 example
for the study brief. The remaining mechanics and a study run still require their
own decisions; this choice does not start observation or change trading.

**Initial-purchase choice:** The owner chose to propose hypothetical purchases at
the final 2026 market close so that 2027 can qualify as a complete comparison
year. This is a study-design choice, not authorization to trade or run the study.

| Question | Simple illustration | What the eventual study must define | Limitation |
| --- | --- | --- | --- |
| How deep can a fall get? | A portfolio peaks at $100 and later reaches $70: its peak-to-trough decline is 30%. | Observation window, valuation frequency, whether withdrawals count, and whether to measure the worst decline or a specified market episode. | The worst observed fall can be dominated by one event; a short window may miss a worse event. |
| How much is lost in a bad year? | Starting at $100 and ending a specified year at $88 gives a 12% loss for that year, assuming no flows. | Calendar or rolling year, how a bad year is selected, flows, and the measure compared across the candidate and benchmark. | A calendar boundary can split a decline; adjacent rolling years overlap and are not independent crises. |
| How long to recover? | Starting from a $100 peak, a portfolio falls and returns to $100 two years after that peak. | Which peak and recovery date, treatment of cash flows and inflation, valuation frequency, and how to handle a period that has not recovered when the study ends. | Recovery can be unobserved within the sample; comparing only completed recoveries biases the picture. |

For each question, compare the proposed policy with a specified, investable fixed alternative, with consistent costs, cash yields, taxes and starting state. The owner has chosen the 80/20 SPY-and-Treasury-bill comparison alongside SPY; acceptable wealth sacrifice remains open. A less-exposed mix can reduce losses without establishing timing or stock-selection skill.

Also show wealth at the end of the period, including a separately labeled hypothetical after-tax scenario when inputs permit: a policy can reduce one loss measure while ending with less wealth. A short forward observation with no relevant decline cannot establish crisis protection. The historical cross-asset shadow's rules and evidence remain unresolved in the [shadow brief](2026-11-10-cross-asset-shadow-decision-brief.md). A defensive protocol requires the applicable prospective amendment before it can change existing gates.

An Opus review recommends comparing candidate and benchmark over the same dates; each portfolio's own worst year may occur in a different year and answers a different question. For recovery, distinguish time from the peak to regaining it from time from the low to regaining it. If the study ends before recovery, report the spell as at least the observed duration; do not drop it. Daily closing values can reveal lows missed by month-end observations, but neither necessarily captures an intraday low. Pre-inception index or cash proxies must be labeled and are not automatically investable.

The chosen primary measure compares returns in the same down years; the rule defining those years remains to be specified. Drawdown and recovery can remain descriptive. Requiring multiple registered outcomes to meet a criterion, or allowing any one to meet it, makes different statistical claims; an alternative-pass rule needs prespecified multiplicity control. A lower-risk portfolio may recover its own earlier peak sooner yet still have less wealth than its benchmark, so report recovery and the wealth trade-off separately.

For a concrete future design, define a reference portfolio's qualifying calendar years before analysis, then compare the candidate with the fixed alternative over those same years. Report the candidate's own worst decline and unfinished recoveries separately. This avoids counting heavily overlapping rolling years as independent, but may miss a crash that straddles year-end or a decline inside a year that finishes positive. Any wealth-sacrifice limit needs a prespecified horizon and margin; a statistically insignificant loss is not proof that the loss is acceptably small. The year-selection definition and margin remain open.

For a later data inventory, the repository has `data/external/a23_inputs/SPY.csv` and `BIL.csv`, along with French factor archives. Their existence alone does not verify date coverage, total-return construction, provenance, suitability or rights for this study. Inspect metadata and provenance before choosing a window, and do not imply pre-inception ETF history exists. No asset values or performance were computed for this note.

## Proposed first comparison: 1–5 year horizon

**Draft only; not registered or approved.** The owner selected 1–5 years as a planning horizon. The example candidate below is a concrete research proposal, not a decision to fund it or restart its old shadow.

- **Candidate:** the historical static cross-asset mix: 80% SPY and 20% divided equally among TLT, GLD and PDBC, with monthly rebalancing. This revisits the same allocation for a new question; prior results and examined periods remain seen, and the original shadow rule and status are not changed.
- **Alternatives:** the owner-selected 80% SPY / 20% Treasury-bill comparison, plus 100% SPY as an opportunity-cost reference. Monthly rebalancing remains a proposal, not part of the owner's choice. Define the bill instrument, rebalancing, costs and common valuation dates before any evaluation.
- **Same starting point:** use standardized hypothetical initial wealth and no external cash flows. Include reinvested distributions and the same assumptions about cash yield, trading costs and tax-payment funding. A real-account comparison requires separately established initial holdings, basis and displaced allocation.
- **Measure all three:** report peak-to-trough decline, loss in matched calendar years selected by a rule fixed beforehand, and time from a reference peak to regaining each portfolio's value at that date. Mark unrecovered episodes as at least the observed duration. Compare matching dates, and separately show each portfolio's own worst period.
- **Wealth trade-off:** report terminal wealth and annualized return for the candidate and both alternatives, before tax and under explicit ongoing-tax and terminal-liquidation scenarios when inputs permit. The owner does not want to set an acceptable wealth-sacrifice limit yet. Show the protection and wealth differences side by side without assigning a pass/fail result for that trade-off. Failure to prove a loss is different from evidence that it is within an acceptable limit.
- **Horizon and evidence:** specify one fixed evaluation start and end within 1–5 years before measuring anything. Historical windows and the candidate have already been examined, so retrospective comparisons are descriptive unless a prospective amendment explicitly establishes their permitted role. If no qualifying decline occurs or coverage is insufficient, defense evidence is inconclusive. A short window is a screen, not proof of sustained protection.
- **Before registration:** define the chosen down-year endpoint and calendar-year selection rule, and any wealth-sacrifice margin; specify the chosen benchmark's instrument and rebalancing; establish data coverage and source rights; and define costs, tax assumptions and operational limits. Any defense pathway needs the applicable approved framework and a dated prospective amendment. Do not start the historical shadow or change its rules as a consequence of this draft.

### Concrete prospective design to review

These are proposed mechanics, not owner-approved rules or a new decision about
the earlier shadow:

- **Portfolios:** Use SPY, TLT, GLD and PDBC for the candidate, with the weights
  above. Use BIL for the bill portion of the chosen 80/20 comparison, subject to
  confirming suitability and data rights. The third portfolio holds only SPY.
  [State Street](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-bloomberg-1-3-month-t-bill-etf-bil)
  describes BIL as an ETF tracking an index of Treasury bills with one to three
  months remaining, and listed a 0.1352% gross expense ratio on November 11,
  2026. It is a fund, not a bill held directly. If fund total returns are used,
  do not subtract its operating expense ratio again; separate trading costs still
  need modeling. This current expense figure is not a claim about past expenses.
- **Timing:** Start each hypothetical portfolio with $100,000 in cash. Set targets
  after the last trading close of each month and model trades at the next session's
  close. For the initial purchase, propose setting fixed targets after the
  penultimate 2026 session and modeling trades at the final 2026 close. Rebalance
  both mixed portfolios monthly. This avoids assuming that a close is known before
  trading at that same close. Exact target weights at the next close are a
  simplifying assumption: actual order quantities and fills need not achieve them.
- **Accounting:** Use total-return-consistent prices or separately reinvest
  distributions, without counting them twice. Charge each portfolio 0.01% of
  dollars bought and sold, including the initial purchase, as an illustrative
  assumption. Allow fractional shares for the hypothetical comparison and assume
  idle cash earns zero until invested. Report
  pretax wealth; estimate taxes separately only when the necessary assumptions and
  inputs are specified. Do not imply the cost represents actual execution.
- **Self-financing calculation:** Proposed target weights sum to one. At each
  modeled execution close, let `V` be wealth before trading, `h[i]` the current
  dollar value of asset `i`, `w[i]` its target weight, and `c = 0.0001` the cost
  per dollar bought or sold. Solve `W = V - c * sum(abs(w[i] * W - h[i]))` for
  nonnegative post-trade wealth `W`, then set each holding to `w[i] * W`. This
  pays the modeled trading cost from the same portfolio rather than borrowing to
  reach targets. For the initial $100,000 all-cash purchase, this gives about
  $99,990.001 invested and $9.999 paid in cost. Future after-tax scenarios need
  their own explicit tax-funding treatment; this equation is pretax and does not
  claim executable quantities or fills.
- **Matched downside observations:** Propose selecting complete calendar years
  in the future observation window when SPY's pretax total return, before modeled
  trading costs, is strictly below zero. Measure from the prior year's final
  session close to that year's final session close; exclude a year unless both
  endpoints and the intervening portfolio history are covered by the observation,
  including its 2026 initialization, and all
  three initial purchases have been modeled by the starting close. For
  each selected year report all three pretax returns after modeled costs and the
  candidate's difference from the 80/20 comparison in percentage points. Show
  every year and the number observed; any average is descriptive, not evidence
  of statistical significance. Also report each portfolio's worst
  daily-close drawdown, and recovery from a common SPY peak, including unfinished
  recoveries. If there is no qualifying year, report that explicitly rather than
  claiming protection. This is an observation rule, not a trading signal.
- **Common-episode proposal:** For the descriptive recovery comparison, select
  SPY's largest daily-close peak-to-trough percentage decline over the full
  observation window, including the final 2026 close after initial-purchase
  costs. If multiple troughs tie, use the earliest; for that trough, use the
  earliest preceding close at the same highest SPY value. For each portfolio,
  measure its change from its own wealth at that common peak to its wealth at
  SPY's trough. Starting at that trough, count sessions until each portfolio
  first regains its own wealth at the common peak; a portfolio already at or
  above that value at the trough has a zero-session recovery from the trough.
  Separately report sessions from the common peak to recovery. If a portfolio
  has not regained its peak-date wealth by the end, report at least the observed
  duration. If SPY has no strictly negative daily-close drawdown, report no
  qualifying episode. The episode is selected after observing the period, so
  this is descriptive and cannot establish a prospective trading signal. Inputs
  must have complete common sessions and positive portfolio wealth.
- **Open before any run:** Confirm the exact start and end boundaries for the selected
  2027–2029 period, confirm or revise the proposed down-year rule above and specify how multiple observations are interpreted,
  validate the inputs, and document the tax and execution limitations. The owner
  has not chosen a wealth-sacrifice margin, so show the trade-off without inventing
  one. A separate approval is needed for any return calculation.

**Selected three-year period, with mechanics still proposed:** Start with cash at the
penultimate 2026 session close (currently expected December 30), set initial targets
then, and model purchases at the final 2026 session close (currently expected
December 31). End observation at the final 2029 session close. Under the proposed
complete-year rule, 2027, 2028 and 2029 can each qualify if gross SPY is negative
in that year and all data and portfolio-history checks pass. This is a maximum of three
observations, not necessarily independent crises; it could be zero. Daily-close drawdown,
common-peak recovery and wealth differences would use the whole observation
window and remain descriptive. Each calendar-year return starts from the preceding
year-end value after any initial-purchase cost; total wealth from the initial cash
must also show that cost, without charging it twice. This period is not a registered or approved study;
exchange schedules can change, and data rights, cost, tax and
execution assumptions still need resolution before any run.

The [matched-down-year calculation](../../allocation/cross-asset-sleeve/code/matched_down_years.py)
has synthetic tests only. It can calculate returns from supplied curves and check
them against supplied session dates, but cannot verify that the calendar is complete
or the curves' initial purchases, provenance and costs are correct. It is not a
study driver and has not been run on historical returns. To check 2029, this helper
requires a supplied calendar extending into 2030 so it can identify the last 2029
session; that requires a calendar entry, not a 2030 portfolio return. The calendar
itself must be independently verified before use.

The earlier [descriptive driver](../../allocation/cross-asset-sleeve/code/descriptive_protection.py)
is not an implementation of this proposal: it fixes the 2021–2026 window and
2022–2025 annual rows, rebalances at the same month-end close, and overwrites its
output path. A future implementation needs the proposed timing, complete-year
selection and output protection without changing that historical driver or artifact.

**Input availability check:** On 2026-11-10, the local A23 manifest reported SPY from 1993-01-29 through 2026-10-23, BIL from 2007-05-30 through 2026-10-23, and the DTB3 rate from 1954-01-04 through 2026-10-22. All three files were present and matched their recorded SHA-256 hashes. This checks identity and the manifest's reported endpoints only, not observations, completeness, total-return treatment or source rights. Candidate TLT, GLD and PDBC inputs and their common coverage have not been checked. DTB3 is a rate series, not a directly investable fund or a ready-to-use total-return series.

**Candidate-leg provenance:** The [historical cross-asset report](../../allocation/cross-asset-sleeve/2026-09-28-cross-asset-sleeve.md) says the original study used adjusted yfinance closes from the database; its earlier holdout used DBC in place of PDBC and fetched prices separately. Those are reported historical methods, not evidence that the original input bytes are recoverable. The tracked [snapshot manifest](../../../data/snapshots/MANIFEST.json) names SPY, TLT, GLD, PDBC and BIL as requested extras for a later tape. The file was absent in this worktree and the original checkout, but a read-only copy was subsequently found at `/home/flynn/ggTrader-vbt12/data/snapshots/ohlcv_20261106.parquet`; its SHA-256 matches the manifest (`6573e84065d54796a6240051c3d83a68b32a079241665fb44001ec0cbc789b2e`).

A symbol-and-date-only check of that copy found all five legs on the same 1,919 dates from 2019-03-01 through 2026-10-16, with no duplicate or missing timestamps for any leg. SPY, TLT and GLD have another ten dates through 2026-10-30; PDBC and BIL end on October 16. Each leg has all SPY dates within the shared range. The [raw coverage record](../../allocation/cross-asset-sleeve/_protection_tape_date_coverage_20261111.json) records the counts and input hash. This uses SPY's recorded dates as the reference, not an independently verified exchange calendar; common omissions would be missed. Only symbol and timestamp columns were read, so the check establishes neither prices nor return validity. An eligible input set still needs calendar corroboration, corporate-action and distribution checks, source and rights documentation, and an agreed window before any analysis. Rebuilding the tape was not attempted.

An additional [calendar check](../../allocation/cross-asset-sleeve/_protection_tape_calendar_coverage_20261111.json) compares each leg's dates with the installed `pandas_market_calendars` 5.6.0 NYSE schedule over that range. All five have each of its 1,919 sessions, with no extra dates or non-midnight timestamps. This is a software-generated reference, not verification against exchange-issued records; it does not check fund-specific halts, price values, distributions or rights. No price columns or returns were read.

**Adjustment cross-check:** The current [yfinance loader](../../../src/ggTrader/data/live/yfinance_loader.py) requests `auto_adjust=True`, but the snapshot exports existing cached bars without recording the adjustment settings for each row. A [level comparison](../../allocation/cross-asset-sleeve/_protection_tape_adjustment_comparison_20261111.json) joins the tape closes to the later saved Yahoo adjusted closes used in the descriptive comparison, on 1,337 dates from 2021-01-04 through 2026-04-30 for each fund. The tape-to-reference ratio is approximately constant within each fund: about 1.00248 for SPY, 1.01194 for TLT, 1.00288 for BIL and 1.00000 for GLD and PDBC. No successive ratio change exceeds 0.5%. A constant difference in adjusted-price levels cancels in a within-source price ratio, but its cause was not established. Both files ultimately use Yahoo, so agreement is not independent validation of prices or distributions. The check is not a strategy or portfolio return calculation, a complete corporate-action audit, or proof of permitted use.

**Distribution spot-check:** Against the fund managers' published records, the saved Yahoo action fields match three TLT ex-dates in February–April 2026 and PDBC's December 23, 2024 distribution to within $0.0005 per share. The [event record](../../allocation/cross-asset-sleeve/_protection_distribution_spotcheck_20261111.json) lists the amounts and source hashes; the issuer records are [iShares](https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf) and [Invesco](https://www.invesco.com/us-rest/contentdetail?contentId=30764e28-1344-4e69-9e99-2bad61fa3a53). These four conveniently available examples are not a representative or complete distribution audit, and do not establish adjusted-price correctness, tax treatment or data-use rights. No portfolio returns were calculated.

**Rights check:** [yfinance's own documentation](https://ranaroussi.github.io/yfinance/) says it is unaffiliated with Yahoo and directs users to Yahoo's terms for rights to downloaded data. A search-indexed excerpt of [Yahoo's U.S. terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) says automated collection from its services requires express prior permission; the full page returned HTTP 999 when opened for this check. This excerpt does not establish which terms applied to these particular downloads, whether permission exists, or the rights to retain and use previously saved data. Verify applicable terms and authorization before relying on this source for a new study; the existing data-quality checks do not resolve rights.

**Free-source alternatives checked 2026-11-11:** [Tiingo's end-of-day API](https://www.tiingo.com/documentation/end-of-day) documents adjusted prices and cash distributions, but its [current terms](https://www.tiingo.com/tos/) prohibit persistent storage of Starter-plan data and derived products and say Starter does not include the predictive-model rights described for paid plans. That conflicts with this project's pinned-input and retained-result workflow; no Tiingo data was requested. [Twelve Data](https://twelvedata.com/etf) advertises historical ETF prices on its free plan, but its [adjustment guidance](https://support.twelvedata.com/en/articles/5179064-are-the-prices-adjusted) describes split adjustment and client-side dividend adjustment, while its [fundamentals page](https://twelvedata.com/fundamentals) lists the dividend endpoint under Grow/Venture rather than Basic. Its suitability for all five funds, total-return construction, storage and research use remain unverified. Do not assume either free tier is a reproducible replacement based on request limits alone.

**Alpha Vantage monthly lead:** Its [API documentation](https://www.alphavantage.co/documentation/) marks daily adjusted history as premium and describes a monthly adjusted series. On 2026-11-11 a metadata-only call with the existing key returned 324 SPY date labels, from 1999-12-31 to 2026-11-11; the last label is an incomplete current month. The next request, for TLT, returned an HTTP 200 `Information` response rather than a series, so the check stopped without testing the other funds. The [coverage record](../../allocation/cross-asset-sleeve/_alpha_monthly_fund_coverage_20261111.json) does not retain price values or the response text; the cause of the TLT response is unknown. Neither monthly completeness nor adjustment accuracy, retention rights or five-fund coverage was verified. Monthly observations also cannot reproduce daily drawdowns or intramonth stop behavior. No return calculation was performed.

A later [metadata retry](../../allocation/cross-asset-sleeve/_alpha_monthly_fund_coverage_retry_20261111.json) returned 292 TLT dates starting 2002-08-30 and 264 GLD dates starting 2004-12-31, both through the incomplete current month. The PDBC request then returned a rate-limit message and BIL was not requested. The TLT retry establishes that its earlier informational response was not a persistent inability to retrieve that series; it does not establish the earlier response's cause. Date continuity, response values, use rights and accuracy remain unchecked; raw series were not saved.

A [subsequent retry](../../allocation/cross-asset-sleeve/_alpha_monthly_remaining_20261111.json) retrieved 144 PDBC date labels from December 2014 through November 2026, with no missing calendar-month labels; November is incomplete. The following BIL request was rate-limited. This resolves PDBC endpoint availability only: correct month-end dates, values, adjustment methodology, independent provenance and permitted use remain unchecked. No raw series was retained and no returns were calculated.

The later [BIL retry](../../allocation/cross-asset-sleeve/_alpha_monthly_bil_20261111.json) returned 234 date labels from June 2007 through November 2026, with no missing calendar-month labels; the last month is incomplete. All five symbols have now returned a monthly series at least once, with PDBC the latest observed start. These separate calls do not verify a jointly complete, consistent or reproducible five-fund panel. Daily drawdowns, values, adjustment accuracy and rights remain unresolved; no return test has been approved or run.

The published [Alpha Vantage terms](https://www.alphavantage.co/terms_of_service/) describe a personal, non-commercial platform license and identify private individual investment analysis and research in their discussion of commercial use. They also list circumstances that count as commercial use. This does not by itself resolve whether retaining response files, derived results or sharing them with an assistant is permitted under all applicable terms; confirm those specific uses before treating the endpoint as a pinned research source. The rate-limited PDBC request was not repeated during this review.

For stock selection, the separate [accruals metadata check](2026-11-10-accruals-source-metadata.md) establishes the candidate file and cell label but not permitted use, historical accounting vintages or reproducible paired inputs. It is not a validated signal and does not depend on this allocation comparison.

No investment policy, test, shadow start, broker action or change to production is authorized by these examples.

## Descriptive historical comparison — 2026-11-10

**Result: the cross-asset mix earned more than SPY/BIL but suffered a deeper worst drawdown and a larger 2022 loss.** It reduced drawdown and the 2022 loss compared with SPY alone, while ending with less wealth. These are pretax historical observations for one already-inspected period, not an approved defensive verdict, a new shadow or proof of a persistent edge. No return-sacrifice limit has been set.

| Portfolio | Ending wealth from $100,000 | Annualized return | Worst drawdown | Return in 2022 | Recovery from its own worst-drawdown peak |
| --- | ---: | ---: | ---: | ---: | --- |
| 80% SPY / 20% TLT–GLD–PDBC, equally split | $180,680 | 12.56% | −21.35% | −15.59% | 494 trading sessions; recovered 2023-12-13 |
| 80% SPY / 20% BIL | $170,234 | 11.23% | −19.77% | −14.23% | 475 trading sessions; recovered 2023-11-22 |
| 100% SPY | $184,371 | 13.02% | −24.50% | −18.18% | 489 trading sessions; recovered 2023-12-13 |

**Method:** common close-to-close window 2021-04-30 through 2026-04-30; five years, previously seen. Yahoo adjusted closes approximate reinvested distributions; weights rebalance at month-end closes. One basis point is charged per dollar bought or sold, including initial investment; no terminal sale is assumed. There are no external flows. The 2022 row compares the same calendar year. Maximum drawdown and recovery each use the portfolio's *own* peak and trough, which differ by portfolio; these recovery durations are not matched-episode comparisons. They count trading sessions from peak to recovery, not calendar days.

**Limits:** taxes, tax-lot effects, financing, real fills and terminal-liquidation costs are not modeled. Yahoo's adjustment accuracy and redistribution rights were not independently verified. Common date coverage and positive adjusted closes were checked, not every distribution or corporate action. The sample includes one main market decline and was selected after the history was known; 2022 is the only negative complete SPY calendar year in this particular window. This does not establish performance in other declines or imply statistical significance. The old cross-asset shadow remains unresolved and unchanged.

**Same decline, same dates:** SPY's largest drawdown in the window ran from its 2022-01-03 peak to its 2022-10-12 trough. Measured from the same starting close to that trough, the cross-asset mix returned **−21.05%**, SPY/BIL **−19.77%**, and SPY **−24.50%**. From that common start, the cross-asset mix and SPY each regained their own starting value on 2023-12-13 (489 trading sessions); SPY/BIL recovered on 2023-11-22 (475 sessions). This is one hindsight-selected episode, not a prospective signal or independent confirmation. Recovering one's starting value does not mean catching up with another portfolio's wealth. The [matched calculation](../../allocation/cross-asset-sleeve/code/matched_protection.py) verifies the original input and driver hashes and writes a [separate result](../../allocation/cross-asset-sleeve/_protection_matched_20261110.json), preserving the earlier artifact. Each recovery is measured after that portfolio's lowest close between the common start and SPY's trough; an unrecovered value would be reported as at least the observed duration.

**Reproduction:** [driver](../../allocation/cross-asset-sleeve/code/descriptive_protection.py), [raw result and input hashes](../../allocation/cross-asset-sleeve/_protection_descriptive_20261110.json), and local input files at `/home/flynn/ggTrader-refs/protection-20261110/`. Those private local copies were fetched from Yahoo on 2026-11-10 around 15:03 Pacific using `yfinance` with `auto_adjust=False` and `actions=True`; the driver uses their `Adj Close` columns. The local copies are not committed, so a future reader needs the identical files to reproduce the hashes. Command: `.venv/bin/python research/allocation/cross-asset-sleeve/code/descriptive_protection.py /home/flynn/ggTrader-refs/protection-20261110 research/allocation/cross-asset-sleeve/_protection_descriptive_20261110.json`.
