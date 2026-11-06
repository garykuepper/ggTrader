# Intraday r8 data-gap investigation

**Completed: bounded investigation. Verdict remains UNRESOLVED (data), not a strategy NO-GO.**
The owner accepted investigation before repairs/scoring, free sources evaluated
before use, and at most 30 targeted HTTP attempts including pages/retries.
No protocol, strategy code, frozen chunks or validation artifact was modified.
No signals, asset returns, candidate returns, performance gates or orders were computed.

## Verified findings

Parent issued seven Alpaca GETs: exact August 2019 monthly bars (two pages),
exact corporate actions, the affected regular-session day, the two absent dividend
months and a present dividend control month. All requests completed with HTTP 200,
without retries. The exact month and corporate-action bodies match the original
page SHA256s. Narrowing requests did not restore the missing records.
[Request evidence](_intraday_r8_gap_investigation.json).

The affected 2019-08-12 session has 360 regular-session minute bars. Required
15:30 and 15:59 ET bars are absent. Three further SIP trade GETs returned zero
records for 15:30 and 15:59, but 70 trades for the present-bar 15:31 control;
all responses were HTTP 200, fully paginated. Only timestamps, venues, condition
codes, counts and hashes were retained, not trade prices or sizes.
[Trade-existence evidence](_intraday_r8_gap_trade_existence.json).
The endpoint is documented in [Alpaca Historical trades](https://docs.alpaca.markets/us/reference/stocktrades-1).

This establishes reproducible absence in the accessible vendor products. It does
not prove that exchange trading stopped or that another archive has no records.
No same-source reconstruction inputs were found for the two required minutes.
Sonnet's static review found no demonstrated collector pagination/time-bound bug;
exact response fingerprints plus narrow comparisons support that conclusion for
these particular defects, not a universal certification of the collector.

## Issuer distribution verification

State Street links its historical-distribution workbook from its
[distribution page](https://www.ssga.com/us/en/institutional/resources/documents/etf-dividend-distributions).
Parent downloaded the [issuer XLSX](https://www.ssga.com/library-content/products/fund-data/etfs/us/spdr-etf-historical-distributions.xlsx),
SHA256 `6b65cbf956b5d25705d08470006a0e97e6a7d2a0d5cfd08b63f6f79783b06320`,
and retained the [source bytes](../artifacts-2026-10-25-intraday-gap/spdr-etf-historical-distributions.xlsx).
Flash extracted exact fields; parent independently parsed and asserted them:

- Row 9172: SPY, CUSIP 78462F103, ex-date 2016-03-18, record 2016-03-22,
  payable 2016-04-29, dividend USD 1.049604/share.
- Row 9163: same identity, ex-date 2018-06-15, record 2018-06-18,
  payable 2018-07-31, dividend USD 1.245568/share.
- Control row 9171: ex-date 2016-06-17, record 2016-06-21,
  payable 2016-07-29, dividend USD 1.078442/share.

Capital-gain fields are blank in these rows; blanks were not converted to zero.
The workbook has 43 unique SPY ex-dates within the research window; Alpaca has
41, missing exactly 2016-03-18 and 2018-06-15. Of 41 shared dividend amounts,
40 match as decimal values. For 2020-06-19 the issuer publishes 1.366242 and
Alpaca returns 1.3662419, a 0.0000001/share precision difference. Any future
cross-source comparison must predefine decimal precision, not choose values
based on results. No other archive-wide completeness claim follows from this.
[Verified extraction](../artifacts-2026-10-25-intraday-gap/issuer_spy_distributions.json).

## Opus assessment and parent adjudication

Opus agrees frozen r8 closes UNRESOLVED under D4/D6. Missing required endpoint
prices independently prevent scoring. Quarterly dividend presence is not an
explicit r8 performance gate; the implementation added a conservative check.
However, total-return and ex-distribution diagnostics are mandatory. Removing
the check would silently feed known-incomplete distributions into those outputs.
Retain it as a mandatory-diagnostic readiness check; do not weaken it post hoc.

For any future separately approved revision, prefer the complete issuer schedule
as one distribution source over patching only two rows, with identity, unique
session ex-dates, positive per-share cash amounts, capital-gain field semantics,
provenance and overlap precision checks fixed in advance. This does not solve
missing endpoint prices.

Two qualifications to Opus's review: the absent 15:59 close affects that day's
exit and the next session's prior-close anchor; the missing 15:30 bar separately
affects entry. One absent closing price does not determine entry. Also, empty
vendor trade responses do not establish that no free remedy exists anywhere.
No compatible alternative full-window free price source was verified here.

## Recommendation and approval boundary

Keep r8 final UNRESOLVED (data). The investigation has found vendor-response gaps,
not evidence that the trading rule fails. Do not drop dates, interpolate, use a
nearby minute, daily close, IEX patch or relaxed gap quota. No performance score
is justified by the collected inputs.

The next useful action is a vendor data-quality inquiry with the exact UTC
windows, feed, raw adjustment, request parameters and response hashes in these
artifacts. Ask whether Alpaca can restore or explain the missing bars/trades and
dividend records. No message has been sent. A correction, if supplied, still
needs a documented new source/protocol execution proposal; no silent reopening
or replacement of the frozen r8 artifacts is authorized.

No bar-from-trades implementation is warranted while the required source trades
are absent. Do not spend another broad strategy-review round on this failure.
This closes the agreed investigation before remediation or performance analysis.

## Request budget and delegated work

Direct HTTP GET attempts: ten Alpaca (seven bars/actions comparisons plus three
trade checks) and one issuer workbook download; zero retries. Public discovery
used one initial search, two issuer page reads, a failed XLSX click, a failed
Alpaca documentation read and a second search. Reserve seven slots for these
public discovery operations: 18 of 30 allocated. Search backend/internal network
requests are unobservable; this is an application-operation budget, not a certified
packet-level count. No further requests are planned. Children made no network calls.

Delegate IDs (all completed, local-file analysis only):
- Sonnet: `node:delegated-task:command%3Amcp%3Addcb1fe4-adb1-43af-b8a1-2705ab37e909%3Adelegate-task%3Aintraday-gap-plan-sonnet-static-2026-10-25-r1`.
- Flash: `node:delegated-task:command%3Amcp%3Addcb1fe4-adb1-43af-b8a1-2705ab37e909%3Adelegate-task%3Aintraday-gap-flash-issuer-extract-2026-10-25-r1`.
- Opus: `node:delegated-task:command%3Amcp%3Addcb1fe4-adb1-43af-b8a1-2705ab37e909%3Adelegate-task%3Aintraday-gap-opus-protocol-review-2026-10-25-r1`.

Parent verified Flash mechanically, checked Opus against the frozen protocol/code
and tested Sonnet's vendor hypothesis with direct requests. No code changed, so
no new test-suite run was needed. Earlier synthetic test passes are not evidence
of strategy profitability. The prior Argon scope violation remains disclosed;
this investigation does not certify what that child inspected internally.
