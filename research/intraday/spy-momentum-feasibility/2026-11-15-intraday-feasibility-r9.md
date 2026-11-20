# Approved r9 protocol: intraday SPY half-hour rule, historical screen

**Status: APPROVED for protocol and mocked implementation (approval 1), 2026-11-15.** The owner
answered "Yes" to the draft and to its two calls in section 6: accept the two-session exclusion,
and keep r8's end date of 2026-09-30. Approval 2 (data and scoring) has not been given. No price,
signal or return has been computed for this protocol. r8 stays closed as UNRESOLVED (data); r9
is a new registration, not a re-score of r8.

Correction made at approval, before any code or data: the draft's R3 said the new pull would
drop Alpaca's corporate actions. That contradicted R2's cross-check and would remove r8's
automatic split stop, so R3 now keeps them. No other text changed.

## Summary

- **What it tests.** One fixed rule on SPY, from the published intraday momentum idea (Gao and
  co-authors, 2018). If SPY is up from the prior close to 10:00 ET, buy at 15:30 and sell at the
  close. Otherwise stay out. One trade at most per day, nothing held overnight.
- **What changes from r8.** Two things only, both decided without looking at any price:
  1. Two named sessions are left out because their required price bars do not exist in the
     data: 2019-08-12 and 2019-08-13.
  2. The dividend dates come from State Street's own distribution file instead of Alpaca,
     which is missing two of them.
- **What stays the same.** The rule, the window, the price conventions, the costs, both pass
  bars, the tax ledger and every diagnostic are copied from
  [r8](2026-10-25-intraday-feasibility-r8.md) without change.
- **What a result means.** A pass says only that the rule earned enough after costs and a tax
  policy to deserve more research. It does not say the rule beats holding SPY, and it approves
  no trading. A fail does not prove the effect is absent, because the test can only see a
  fairly large edge.

## 1. Why a new protocol

r8 ended UNRESOLVED (data) on 2026-10-25 for two reasons
([validation record](2026-10-25-intraday-r8-validation-unresolved.md)):

- **Missing bars.** On 2019-08-12 the 15:30 and 15:59 minute bars are absent. The 15:59 bar is
  also the prior close for 2019-08-13. r8 allowed no missing endpoint and no dropped day.
- **Missing dividends.** Alpaca's corporate-actions data has no SPY dividend in 2016 Q1 or
  2018 Q2. The [gap investigation](2026-10-25-intraday-gap-investigation.md) found the two
  ex-dates in State Street's file: 2016-03-18 and 2018-06-15.

That investigation recommended against dropping dates and suggested asking Alpaca to explain
the gaps. The owner judged on 2026-11-15 that Alpaca will not explain them. r9 is the owner's
choice to accept a small, named exclusion in exchange for a result. That trade-off is stated
here so a reader can judge it.

## 2. Evidence the gaps are small (timestamps only)

On 2026-11-15 a fresh count of SPY regular-session minute bars (Alpaca, SIP feed, raw prices,
2016-01-01 to 2026-11-13) recorded timestamps only. No price, volume or return was stored or
read. [Script and output](../artifacts-2026-11-15-spy-minute-gap-count/).

| Check | Result |
|---|---|
| Sessions | 2,733 |
| Minute bars expected / missing | 1,062,090 / 104 |
| Sessions with any missing minute | 8 |
| Sessions missing one of r8's four endpoint bars | 1 (2019-08-12) |

It agrees with the r8 collection: the same 104 minutes on the same 8 sessions. Six of the
eight sessions miss only interior minutes, which r8 already treats as a diagnostic. Four of
those fall on 2020-03-09, 03-12, 03-16 and 03-18, the dates of the market-wide trading halts
in March 2020. That match is not yet checked against an official source.

## 3. Registered changes from r8

### R1. Fixed exclusion list

- **Excluded sessions:** 2019-08-12 (no 15:30 entry bar, no 15:59 exit bar) and 2019-08-13 (no
  prior-close bar). Both are removed from the scored series. They are not counted as flat days.
- **The list is closed.** If the r9 pull is missing any other required endpoint, or if these
  two sessions turn out to have their bars, the study ends UNRESOLVED (data). No new exclusion
  is added after the pull.
- **Scored sessions:** 2,679 under r8, so 2,677 under r9, if the calendar matches.
- **Disclosure.** The report states the exclusion in its summary, and states that it was
  chosen after the missing timestamps were known but before any price was seen.
- **Bias.** Two of 2,679 sessions is 0.07%. Losing a day for a reason linked to how the market
  moved that day could still bias the result. That effect cannot be measured without prices,
  so the report names it as a limitation.

### R2. One dividend source

- **Source:** State Street's SPDR historical-distribution workbook, the copy already archived
  at [`spdr-etf-historical-distributions.xlsx`](../artifacts-2026-10-25-intraday-gap/spdr-etf-historical-distributions.xlsx)
  (SHA-256 `6b65cbf9…3b06320`), used for every SPY cash distribution in the window.
- **Checks fixed in advance:**
  - Identity: ticker SPY and CUSIP 78462F103.
  - One ex-date per row, each on an official session.
  - Positive per-share cash amounts.
  - A blank capital-gain field means no data, never zero.
- **Cross-check with Alpaca:** compared at 6 decimal places, recorded as a diagnostic only.
  The 43 issuer ex-dates in the window, against Alpaca's 41, are the expected count.
- **Use.** Dividends feed only the non-gating diagnostics (the SPY comparison and the
  ex-dividend-flat sensitivity). The rule's own returns use raw prices, as in r8.

### R3. Fresh pull

- **One new pull** of the r8 inputs, including Alpaca's corporate actions. They are still
  needed for r8's automatic stop on an in-window SPY split, and for the dividend cross-check in
  R2. r8's check that every quarter has a dividend applies to the State Street schedule, not to
  Alpaca's records.
- **The 2026-10-25 collection is not reused.** It was gathered by an agent without approval,
  and its contents were never independently audited.

## 4. Unchanged from r8 (summary; r8 is authoritative)

- **Window.** Scored from 2016-01-05 to 2026-09-30. The 2016-01-04 close is the anchor. Half
  days are not scored.
- **Prices.** The signal is the 09:59 bar's close, the entry is the 15:30 bar's open, and the
  exit is the 15:59 bar's close. The prior close is the last minute bar before the previous
  session's official close. There are no fallbacks.
- **Daily series.** `s_t = 1[r_first > 0] × r_last − cost`. Flat days count as zero.
- **Costs.**
  - Base: 1 basis point (0.01%) a side. The verdict uses this.
  - Stress: 2 basis points a side. Reported, but it cannot change the verdict.
- **A1, the statistical bar.** A one-sided Newey-West t-statistic (a significance test that
  allows for patterns repeating over time) with 5 lags, on the mean of `s_t`. It must reach
  3.0 or more.
- **A2, the money bar.** The compound annual growth rate after r8's illustrative 24% tax
  ledger must be +3.0% a year or more.
- **Verdict.** GO for further research only if A1 and A2 both pass. Otherwise NO-GO. A data
  failure is UNRESOLVED (data).
- **Diagnostics (non-gating):**
  - subperiods 2016–19, 2020–23 and 2024–26
  - break-even cost
  - worst month, longest losing streak and maximum drawdown
  - the attribution numbers of r8 D5
  - the SPY comparison
  - the ex-dividend-flat sensitivity
- **Power (r8's planning arithmetic, assumed inputs).** A1 has about a 50% chance of passing if
  the true gross edge is about 4.9 basis points per trade. It has an 80% chance near 5.7 basis
  points.
- **After a pass.** Only drafting of a separate forward proposal follows. No orders, shadow,
  capital or production change.

## 5. Work and approvals

1. **Protocol approval (this draft).** It authorizes:
   - adding R1 and R2 to the r8 code, tested on made-up data only;
   - recording the code and protocol hashes.

   Under the delegation rules, a Claude-built change gets a Sol review and a Codex-built change
   gets an Opus review.
2. **Data-and-scoring approval (separate, after the code review).** It covers:
   - one pull;
   - one signal-blind validation;
   - one automatic score, only if validation passes, writing the results JSON and a GO/NO-GO
     report.

   There is no re-score and no choosing among results.

Neither approval touches the paper trader, cron, Docker, the database or `src/`.

## 6. Calls for the owner (all answered "Yes" on 2026-11-15)

- **Accept the two-session exclusion?** It is the only way to score this window on free data.
  Declining it leaves the forward observer as the only evidence.
- **Keep r8's window, ending 2026-09-30?** Extending it to November would be a new choice made
  after the gaps were known. This draft keeps r8's end date.
- **Keep r8's bars (t of 3.0, +3.0% a year)?** This draft keeps them unchanged.
