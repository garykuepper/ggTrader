# Verification of Opus's plan proposals

Owner requested verification of the estimates/proposal on 2026-10-25.
This checks the [Opus review](2026-10-25-intraday-overall-plan-opus-review.md),
not strategy performance. Numerical estimates are checked separately in the
[companion estimates verification](2026-10-25-intraday-opus-estimates-verification.md).
No revised protocol or order was executed.

## Auction error: cause confirmed, wording corrected

A supplementary request was frozen in the
[verification manifest](briefs/2026-10-25-intraday-opus-verification-manifest.md)
before fetching. One authenticated GET reproduced the failed January 4, 2016
closing-only request. HTTP 200, no pagination continuation; no other
authenticated endpoint was accessed. The
[metadata artifact](_intraday_opus_auction_verification_2026-10-25.json) retains
only validated structure, dates, timestamps, venues, conditions and hashes.

The body hash exactly matches the original Stage 0A failure:
`9617afc407cbf9824c50f98190cb9d9ac3ba2b7af075b3073ce4808f08273304`.
The response contains a present opening field whose value is **null**, and
two valid closing events at venues `T` and `P`, both with condition `6`.
The existing parser rejects null because it requires both fields to be lists.
Thus this is not missing closing-auction evidence. Opus's direction was right,
but its specific omitted-field explanation was wrong: the opening field exists.

The embedded official [auction OpenAPI](https://docs.alpaca.markets/us/reference/stockauctions-1)
still declares `d/o/c` required and `o/c` arrays, without null in their types.
The implementation matched that schema; the observed API response does not.
A future parser can explicitly accommodate an absent-event null while preserving
metadata validation. No parser or original artifact was changed by this check.
Stage 0A's registered verdict remains unresolved.

## Executable exit: proposal needs an additional availability check

The current [Alpaca order matrix](https://docs.alpaca.markets/us/docs/orders-at-alpaca)
excludes fractional `cls`. Whole-share market/limit `CLS` entries are `Yes*`,
and the asterisk directs users to contact sales. The same page describes the
15:50 ET broker cutoff and auction-only execution. Therefore whole shares are
necessary but not sufficient to certify this account's MOC capability.
No account-specific permission or fill was tested; no test order was sent.

[NYSE Arca's exchange timeline](https://www.nyse.com/trade/auctions) places its
closing freeze at 15:59 and auction at 16:00. That exchange deadline does not
override Alpaca's earlier broker deadline. A 15:30 entry leaves a nominal
20-minute interval before that broker cutoff, but partial fills, rejections and
exit submission still require explicit rules.

Venue alone is insufficient: the saved sample already has two `P` closing
records per later session. [Alpaca's condition definitions](https://docs.alpaca.markets/us/docs/market-data-faq)
distinguish `6` (closing trade), `M` (official close) and `9` (corrected close).
They also exclude `M` from minute/daily price updates. These observations do not
prove which later records share a price or which record matches an executable
fill. Freeze a documented selection/correction policy before price inspection.
The 15:59 bar's last qualifying trade remains a research proxy, not a guaranteed
order fill; an auction failure cannot silently switch the execution model.

## Delayed SIP: sufficient for this signal, not proof of operational readiness

The current [data-plan table](https://docs.alpaca.markets/us/docs/about-market-data-api)
and [data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq) support access
to SIP history outside the latest 15 minutes on the free plan. A signal based
on the prior close and the 09:59 bar is old enough well before a 15:30 decision.
Thus the specified signal does not require a paid real-time SIP feed or IEX.
This does not validate timely order submission, broker monitoring, slippage,
or a future signal that uses more recent observations. Dropping IEX from the
registered diagnostics remains a proposed protocol change.

## Remaining proposals: supported decisions, not empirical facts

- **Freeze first:** supported. The r7 window, price conventions, A1/A2 and
  missing-data rules remain proposed. Freeze them before viewing candidate
  prices or returns. Building/committing the scorer first improves auditability.
- **Start fallback:** supported by r7 line 69 and the retained January 4 grid.
  January 5 can be proposed as the first test day with January 4 as its input.
  The missing 2015 input need not force a new coverage probe; price correctness
  of the substitute anchor has not been inspected.
- **Forward-standard mismatch:** confirmed in the documents. A standalone
  after-tax-policy +3% research hurdle does not imply a SPY-relative forward
  hurdle will pass. Their signal/fill timing, cash treatment and order scope
  also differ. Resolve the intended research decision and any prospective
  amendment explicitly; no threshold was changed here.
- **Account isolation:** supported by the production wrapper's documented
  15:45 ET schedule and SPY cash-sweep default. The owner-approved forward
  standard already calls for a separate ledger-only shadow. Intraday broker
  orders in that same production account would not provide strategy isolation.
  Additional-account availability was not verified.
- **Validation simplification:** endpoint-focused eligibility and one bulk
  pull/validation stage are defensible proposals, not verified replacements for
  the stricter current contract. Freeze the permitted missingness, no-trade
  treatment and correction policy beforehand; preserve every calendar day and
  disclose failures. Do not select missing-data policy after inspecting returns.
- **Holdout language:** a fixed test after the original paper's sample is not
  an untouched holdout when related public results exist. Count future tests
  on the same window and keep diagnostics from implicitly selecting a winner.

The recommended next deliverable remains a documents-only r8 decision memo.
These checks support changing the plan's order, not starting price work or
silently approving Opus's suggested gates, date range or collection scope.

The parent independently reran the numerical reproduction in the estimates JSON
and matched its entire calculated object, then regenerated both offline calendar
schedules and matched their counts/hashes. These are reproducible conditional
planning calculations, not measurements of the candidate or its power.
