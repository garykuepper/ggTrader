# Parent assessment of Argon's independent plan review

Gemini 4 Argon High reviewed revision `dfeba41`. Its
[full review](2026-10-25-intraday-overall-plan-argon-review.md) supports the
revised sequence conditionally; this is approval of a research-plan direction,
not a candidate GO or owner authorization. No data, code or gate changes follow
automatically from receiving the review.

## Agreement and useful new finding

Argon agrees with Opus and the checked assessment on freezing before prices,
reconciling standalone economics with the forward standard, separating research
price proxies from execution, applying the existing start fallback under an
approved protocol, isolating a future shadow, and treating power estimates as
conditional. Those findings remain supported by the local documents and saved
verification evidence. No further authenticated request was needed.

Its useful additional warning is correct: a missing exit observation after an
intended entry cannot erase the trade as a zero-return/no-trade day. That would
condition the trading decision on later information and can bias the result.
Missing-data policy must be tied to the actual decision-time information set;
the intended position and elapsed calendar time must remain in the record.

## Corrections and recommendations not accepted as facts

1. **The 15:30 bar is not pre-entry information.** Argon groups its absence with
   prior-close and 09:59 missingness. But r7 enters at 15:30 using the 15:30
   minute's opening trade as a price proxy. That completed minute and whether
   the vendor emits its bar are not known at the entry decision; delayed SIP
   makes it available later still. Only an observable pre-entry condition may
   cause a prospective no-trade. Missing entry valuation/fill evidence after
   an intended order needs a frozen unresolved/valuation rule, not retrospective
   zero P&L. [Alpaca's bar interval and delay documentation](https://docs.alpaca.markets/us/docs/market-data-faq)
   supports this timing distinction.
2. **A bar open is also a fill proxy.** A scheduled market exit around 15:59 or
   15:55 can be proposed, but its fill is not guaranteed to equal that minute's
   opening print. It changes the holding interval and needs explicit execution
   assumptions. A last-available-bar/daily-close fallback may value an intended
   position for research only under a declared rule; it does not prove an order
   exited at that price. Do not silently convert missing exits into fills.
3. **Exposure arithmetic is wrong by a factor of two.** A 30-minute trade is
   `30/390 = 7.6923%` of a regular session; at an assumed 50% trade frequency,
   the average is `3.8462%`, not 3.8% and 1.9%. These time fractions do not
   determine the proportion of the equity premium captured or lost.
4. **No guaranteed data failure or return forecast.** The strict all-minute
   policy may be disproportionate to the endpoints, but no observed result
   proves it will inevitably fail. Requiring all minutes and endpoint validity
   is logically consistent: the former is simply stronger. A 3% sleeve's
   probability of failing a rising-market SPY comparison is also not established
   as "almost certain." The documented gate mismatch remains a design issue
   without those assertions.
5. **Auction selection remains unverified.** Argon supplies different example
   precedence orders for `6/M/9`; none is established as the right executable
   mapping. The follow-up observed an opening null only. Treating closing nulls
   as an empty list may be a future structural parser rule, but is not newly
   verified evidence of closing availability. Missing closing events cannot pass
   a required auction check. Whole-share CLS account capability remains open.
6. **No new quota or economic definition.** The suggested five-session missing
   allowance is an unsupported example, not a derived tolerance. No quota was
   adopted. A standalone 3% after-tax-policy CAGR is not interchangeable with
   3% excess over cash or SPY. Any alternate objective needs a precisely defined
   comparator and explicit owner approval, with no silent threshold change.
7. **Attribution and hypothesis count need exact meanings.** Failure to reject
   a predictive slope is not proof the signal has no effect; use "incremental
   predictive contribution not established." A label based on "similar" returns
   needs a frozen interpretation, not an improvised cutoff. Track this as one
   intended candidate and zero completed candidate return tests so far; the
   metadata probes are not scored strategy tests. A counter alone does not
   control adaptive multiple testing.
8. **Describe price access accurately.** Stage 0A API bodies inherently contained
   prices, which were discarded before persistence or analysis. "No prices
   fetched" is literally inaccurate; no candidate price analysis, signals or
   returns were performed. The original SIP-bar unresolved conclusion was due
   to the prior-close check; the auction branch was recorded separately.

## Recommendation

Keep the shared five-step direction: approved r8 protocol, mocked implementation
and code freeze, authorized whole-window collection/validation, one fixed score,
then a separately approved isolated shadow only if warranted. The immediate
deliverable is still a documents-only decision memo covering the research role,
executable versus proxy exit, statistical/economic gates, decision-time
missingness, collection scope and shadow isolation. Incorporate Argon's useful
missing-exit warning with the entry-timing correction above. No new blocker or
additional data probe is needed merely to complete this plan review.
