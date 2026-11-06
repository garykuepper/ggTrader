# Bounded verification of Opus's estimates and proposal

Owner instruction: "verify opus estimates/proposal" (2026-10-25).
This follow-up verifies the review; it does not approve its revised strategy
protocol, price-quality stage, historical pull, scoring or orders.

Before the request, freeze this supplementary metadata check:

- Exactly one logical authenticated GET to
  `https://data.alpaca.markets/v2/stocks/auctions`, reproducing the failed
  `sentinel:2016-01-04:auctions` parameters in the original Stage 0A artifact:
  SPY, SIP, asof `-`, ascending, limit 10000, UTC start
  `2016-01-04T20:59:00Z`, end `2016-01-04T21:01:00Z`.
- At most three attempts, 20-second total deadline, redirects disabled,
  no ambient proxy/configuration, minimum 0.65 seconds between starts.
  No pagination expansion; if a continuation exists, mark the check partial.
  Retry only transport/429/5xx, respecting provider delays up to 30 seconds;
  defer if a longer delay is required.
- Load existing root credentials into process environment without displaying
  or modifying them. Never query accounts, assets, positions, orders or trades.
- Persist only request parameters, attempt status/time/body hash, date,
  presence/types of the documented opening/closing arrays, event timestamps,
  venues and condition codes. Validate these strings. Discard all prices,
  sizes, unknown fields, raw payloads, page tokens and remote error text.
- Apply the existing parser to the in-memory response and record only its
  success or fixed schema-error category. Compare the new body hash to the
  original; never overwrite or reinterpret the original artifact.

Other verification uses public primary documentation, the published Gao
paper, existing metadata, repository rules and explicit arithmetic. No
candidate signals, prices or strategy returns are inspected or computed.
