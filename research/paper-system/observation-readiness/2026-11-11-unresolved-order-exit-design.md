# Working orders during a loss halt

**Owner selected the cautious exits-only design for preparation on 2026-11-11.**
It is not implemented or approved for trading. The paper trader remains stopped.
This note concerns protective exits, not a new stock signal.

## Current behavior

The trader reconciles durable order intents before reading positions or evaluating
the loss halt. An unresolved working intent stops the run, so a protective exit for
an unrelated holding is not attempted. Legacy pending orders follow a separate path:
known pending symbols are skipped while other processing continues. Neither path
cancels an earlier order; a previously submitted buy can still fill during a halt.

## Alternatives

1. **Keep the current block.** Wait for every durable intent to resolve before
   submitting any order. This avoids adding orders while the recorded state is
   unsettled, but can delay unrelated protective sells.
2. **Permit narrowly scoped exits.** If the loss halt is valid and order state is
   reliably reconciled, attempt only eligible sells for other held long positions.
   Continue blocking all strategy buys and both cash-sweep actions; skip any symbol
   with a working or uncertain order. Do not assume a previously submitted buy was
   canceled or cannot fill.

## Conditions needed for option 2

- Preserve the single-run lock and durable order-intent protocol. Check both legacy
  pending orders and intent records, and obtain a trustworthy current broker view
  of positions and open orders before deciding which symbols are unaffected.
  The current adapter can query an order by ID or client ID but does not expose an
  account-wide open-order listing; that capability and its pagination, status and
  error handling need design and validation before the option can be implemented.
- Fail closed on unreadable, inconsistent or ambiguous order identity, quantity,
  position, risk state, broker response or persistence. An uncertain submission
  remains a stop condition because it may have been accepted.
- A sell must be for a verified positive held quantity, have no conflicting open
  order for that symbol, and pass the market-session and eligibility gates. For
  this exceptional path, an unreadable market clock must stop submissions; the
  existing general path currently assumes the market is open on a clock error.
  Do not infer that a pending sell has filled or that a position slot is free.
- Persist and reconcile every new sell using the existing client-order ID protocol.
  Treat partial fills and orders that remain working as unresolved; preserve their
  status for the next run. Notify that existing orders can still fill.
- Test separate symbols, a working buy and a working sell, partial fills, unknown
  order status, stale positions, broker/DB failures, notification failure, lock
  loss and a crash after submission. Include a case proving no buy or sweep is
  submitted. Tests must be synthetic and mocked before any broker validation.

The broker view can change between checking and submission; a design must account
for that race and cannot promise immediate liquidation. Submission exceptions may
still stop later exits. This proposal does not decide whether an earlier working
buy should be canceled; cancellation requires a separate policy and validation.

## Proposed sequence

1. Under the run lock, reconcile all recorded orders without treating a known,
   verified working order as a terminal error. Any ambiguous reconciliation ends
   the run; retain the affected symbol set and working-order identities.
2. Independently obtain broker positions and a complete open-order listing, then
   reconcile the two order views. If completeness or identity cannot be established,
   submit nothing. Evaluate the existing risk state and proceed only on a valid halt.
3. Build sells only for held long symbols outside both pending sets **with an
   independently valid exit trigger**, as described below. Before each submission,
   recheck the lock and the relevant broker position and orders; any conflicting
   or unreadable state skips or stops the order according to a rule to be specified
   and tested. Never infer that the checks eliminate all races.
4. Stage each sell durably, submit with its client ID, and reconcile or leave it
   recorded as unresolved. Produce a summary distinguishing submitted, filled,
   still-working and failed exits, and state that earlier orders may still fill.

For this design, a failed lock, broker, position or order recheck stops all remaining
submissions and reports that earlier sells may already have been submitted. A
successfully identified new conflict skips only its own symbol. A broker order with
no local intent is not silently ignored: if its identity, symbol, side and working
status are interpretable, mark its symbol as pending and report it without booking
it as a project order. An unidentifiable or unsupported order stops the run. A
recorded working order absent from the broker's supposedly complete open list is
an inconsistency that also stops the run until reconciled. These are design choices,
not a claim that the broker snapshots are atomic.

No order, database, broker or production configuration was changed to prepare this
design.

### A loss halt is not a liquidation trigger

The current loss-halt path clears strategy buys; it does not turn every holding
into a sell. Separately, the trader evaluates strategy exits and, when enabled,
the catastrophe stop. The proposed exception must preserve that distinction:
passing the position, open-order, session and tradability checks only makes a
symbol eligible for an independently triggered exit. It does not create an exit
signal or authorize selling all unaffected positions. A rule that liquidates on a
portfolio-level halt would be a different proposal requiring explicit approval.

Any exceptional-path design must specify which existing exit triggers it supports
and how their decision-time inputs, configuration and corporate-action corrections
are validated. A failed signal gate or unassessable stop is not an exit trigger;
how other independently assessable exits are handled under those failures must be
specified and tested without weakening their existing gates. Test at least a valid
loss halt with one triggered holding and one untriggered holding: only the
triggered, otherwise eligible holding may be submitted. Also test a disabled
catastrophe stop, a missing or stale trigger input, and a pending order on the
triggered symbol. These are requirements for a future design, not a claim that
the proposed path has been implemented or validated.

### Signal-date boundary

**Local update, 2026-11-11:** The owner authorized a future-date fix. The worktree
bar gate rejects future bars and the trader validates every sleeve's date, while
preserving today's bar handling. This is undeployed; it does not implement the
working-order exception or prove bar completeness or information availability.

Before that update, the trader combined sleeve signals and checked the `as_of` date from the
first sleeve. It rejected unparseable dates and dates more than four calendar days
old, but a future date produced a negative age and passed that particular check.
The current production signal generator supplies one sleeve; the general blend
generator can supply more than one. A first-sleeve check would not establish the
freshness of every sleeve if that general path were used. The prior bar gate
returned early when final-bar coverage was sufficient, so coverage alone was not a
future-date rejection. This is a code inspection, not evidence that an actual
future-dated or stale signal was produced or traded.

A synthetic check through `vet_bars` confirms the coverage-gate behavior: with
the test fixture's 80 weekday bars, 40 symbols and full coverage, a final bar
dated Monday 2026-06-22 passed unchanged when the supplied current date was
Friday 2026-06-19. The bounded test passed under `scripts/run_tests.sh`; it did
not execute the signal generator, trader, or broker. It establishes the gate's
response to this invented input, not that the real loader can produce it.

A separate synthetic trader test used the existing mocked database, clock and
broker fixtures: on 2026-06-19, a fabricated single-sleeve buy signal dated
2026-06-22 passed the freshness check and reached `submit_buy` on the mock broker.
The test passed through the capped runner. It supplied the signal directly and
therefore does not prove that the live signal generator produces this input, that
any broker received an order, or that historical trading was affected. No
production code was changed.

The normal generator requests an end date based on the current UTC day. Its
`fetch_stock_ohlcv` path applies a final upper-bound filter after combining cache
and fallback results, and the cached database query also applies that bound.
This limits the path by which a later-UTC-date bar could reach the gate. The
read-only path calls `YFinanceDataLoader` directly; it requests one day beyond
the supplied end and does not itself trim the returned frame. Neither path's
requested UTC date by itself resolves a run when the New York calendar date is
still the previous day. Provider behavior, actual past inputs and whether such a
run would otherwise qualify have not been checked in these synthetic tests.

An offline loader test requested through 2026-06-19 and supplied a fabricated
2026-06-22 provider row; `YFinanceDataLoader` retained that row. The test used
a mock, made no provider request, and does not show that yfinance returns rows
outside its requested range. It confirms that the direct read-only loader relies
on the provider response rather than performing its own final upper-bound filter.

The exceptional design must validate the date and underlying availability of each
signal it uses, including an explicit future-date rejection and the relevant
exchange-session cutoff. A calendar-day age alone does not establish that a bar
was completed or available at the decision. Specify how a failed signal gate
affects independently assessed catastrophe stops before implementation; do not
silently promote a bad strategy signal into an exit. Mock a future date, mixed
sleeve dates, missing timestamps, holidays and an in-progress bar. This identifies
a limitation in the existing path; it does not modify production behavior.

### Loss-halt baseline boundary

The current drawdown check uses a persisted peak, but an absent peak row returns
`None`: `RiskGuard.check_drawdown_halt` then reports no drawdown halt, and the
trader can save the current value as the new peak. That behavior can initialize
a genuinely new risk history; the missing row by itself does not distinguish
that case from lost or replaced history. A database read error or invalid peak
is handled separately and stops the current run. The daily-loss check uses the
broker's previous-close equity and can independently detect a daily breach, but
it cannot reconstruct an absent historical drawdown peak.

For the proposed exceptional path, do not describe a missing peak as a verified
absence of a drawdown breach. Establish the origin and continuity of the risk
baseline before treating the drawdown test as assessed; distinguish a legitimate,
prospectively initialized history from missing evidence. Specify separately
whether a valid daily-loss breach permits an otherwise triggered exit when the
drawdown baseline is unknown, without converting unknown into a valid drawdown
result. Preserve the existing baseline and any uncertainty rather than silently
reinitializing it to qualify this path. Test first initialization, an unexpectedly
missing peak after prior activity, a valid daily breach with an unknown peak, and
database failures. No risk-state change is authorized by this design.

### External cash-flow boundary

The current risk formulas compare raw account values with the stored peak and
broker previous-close equity; the inspected paper path has no explicit adjustment
for deposits or withdrawals. In a hypothetical account with a $100,000 stored
peak, a $15,000 withdrawal and no market change would leave $85,000 and satisfy
the default 15% drawdown threshold. Conversely, a $20,000 deposit after a
$15,000 market loss would produce $105,000, obscuring that loss in the raw-value
comparison and potentially establishing a higher peak. These examples are
arithmetic illustrations, not evidence of a particular account transaction.

The proposed path needs a documented policy for external flows, including source,
effective timing, whether and how baselines are adjusted, and how uncertainty is
handled. Verify the broker's actual treatment of transfers and `last_equity`
before inferring how the daily check behaves. Until the policy and evidence exist,
do not claim a raw-equity change measures investment loss alone. Synthetic tests
should distinguish deposits, withdrawals and market losses, including a flow
between the prior-close baseline and current valuation; do not assume a flow is
known merely because an activity response is empty. These are design requirements,
not approval to alter risk limits, balances, or stored peaks.

### Corporate-action valuation boundary

The current trader evaluates the catastrophe stop using split-corrected position
data, but compares the broker's booked account value with the stored drawdown peak
and previous-close equity for portfolio loss checks. Its later reporting value may
also include split and dividend adjustments. Those values serve different purposes:
a corrected reporting value does not establish a corresponding broker balance or
sellable quantity, and a booked-equity halt does not establish that an individual
position's stop was triggered. The current split-evidence lookup can fail soft,
so an empty correction set alone does not prove that no correction is needed.

Before the exceptional path uses either kind of trigger, specify how it verifies
corporate-action evidence and records the exact valuation basis for each decision.
If split or position evidence needed for a stop is unavailable or ambiguous, do
not infer a valid stop or silently substitute a reporting quantity for the broker's
available quantity. Test an unapplied split, an unavailable correction source and
a broker-reported quantity that differs from the economic estimate, including
whether an independently verified exit for another symbol may continue. This
does not assert that a particular corporate action is currently unresolved or
that the existing adjustment accurately models every action.

## Broker API feasibility check

The installed `alpaca-py` exposes `TradingClient.get_orders` and
`GetOrdersRequest`. Alpaca's [SDK reference](https://alpaca.markets/sdks/python/api_reference/trading/requests.html#getordersrequest)
documents an `open` filter, a default limit of 50 and a maximum of 500. Its
`after` and `until` filters use submission timestamps, and `nested=True` rolls
multi-leg orders into a parent. The request has no page-token field in the installed
SDK. In installed `alpaca-py` 0.45.0, `get_orders` makes one `/orders` request
and does not itself paginate. The [REST reference](https://docs.alpaca.markets/us/reference/getallorders-1)
describes the time bounds as exclusive.

A cautious first implementation could request open orders without a date, side or
symbol filter and with a limit of 500. It should stop if 500 or more top-level records
are returned, the response is malformed, a status or leg cannot be interpreted,
or the returned set conflicts with durable records. This avoids assuming that a
full page is complete; a smaller response still does not prove an atomic snapshot
or establish how all order classes appear. Nested legs, replacement orders and
external orders need explicit treatment and mocked coverage. The broker may change
state after the check, so a per-symbol recheck reduces but cannot eliminate the race.

The installed SDK also names `done_for_day`, `replaced`, `pending_review` and
`suspended` statuses that the current intent reconciliation does not accept. The
new path must not silently interpret an unfamiliar status as closed. Replacement
chains and nested orders require a separately verified mapping; until then, those
cases stop the run rather than making a symbol eligible for another sell.

The installed SDK signature and published documentation were inspected without a
broker connection. Neither completeness nor exact behavior has been verified against
a paper account, and no live or paper order was submitted.

### Recovery states requiring special attention

The existing `_reconcile_order_intents` recognizes `calculated` as active, while
Alpaca describes it as completion for the day with settlement calculations still
pending; the underlying order may be filled or done for the day. Alpaca also says
`done_for_day` can receive updates on the next trading day, and `replaced` can mean
another order or a corporate-action update. See the [order reference](https://docs.alpaca.markets/us/reference/getorderbyorderid-1)
and [trade-event definitions](https://docs.alpaca.markets/us/reference/subscribetotradev2sse).
Neither a day's end nor a pending cancellation proves that a position is free to
sell. The existing recovery rejects `done_for_day` and `replaced` as unknown, and
the proposed path must retain that fail-closed behavior until those cases are mapped.

In particular, do not change `calculated` to terminal merely to make an open-order
list agree with the intent lookup. If a recorded `calculated` order is missing from
that list, stop and investigate under the existing inconsistency rule. Add mocked
cases for `calculated` both present and absent, `done_for_day`, `replaced`, and a
partially filled `pending_cancel`; none may make its symbol eligible for a new sell.
The open-list filter's actual treatment of these states remains unverified, as do
replacement-chain and later trade-correction handling. A mock establishes the
program's response, not the broker's filtering behavior.

### Adapter boundary

The current `AlpacaBroker._normalize_order` retains identity, side, quantity,
fill and status, but drops `order_class`, `legs`, `replaces` and `replaced_by`.
The installed SDK's `Order` model exposes those fields and permits a missing
parent symbol. Thus feeding its current normalized dictionaries to an account-wide
conflict check would lose evidence needed to interpret nested and replacement
orders. The existing method also converts quantities to floats, which is inadequate
for the exact availability comparison described below.

Before enabling this path, introduce and validate an order representation that
preserves the necessary broker fields and original decimal quantities. Until
nested classes and replacement chains have a verified mapping, their presence,
missing required fields or an unrecognized class must stop the run, rather than
being treated as a single harmless order. Synthetic tests should include a parent
without a symbol but with a conflicting leg, a replacement link, and an unknown
class. The existing adapter inspection does not establish how the broker populates
these fields or whether an open-order query includes every relevant leg.

## Position availability and concurrent writers

The installed SDK's `Position` model exposes both `qty` and `qty_available`, but
the current adapter returns only `qty`. Alpaca's [positions reference](https://docs.alpaca.markets/us/reference/getallopenpositions)
describes `qty_available` as shares available after open orders and covered-call
locks; it does not establish snapshot timing or atomicity with the order list.
A positive position alone does not establish that its full quantity is available
to sell. Before implementation, verify behavior for fractional quantities,
pending orders and missing values. The exceptional path must reject missing,
invalid or inconsistent availability and must not sell more than the verified
available long quantity. Do not silently fall back to `qty` if availability is
absent. Add mocked cases for reserved, zero, fractional and stale availability.

The database run lock coordinates cooperating ggTrader runs, not manual orders,
other applications or broker-side changes. Even a fresh order and position check
cannot prevent another writer from acting before submission. This residual race
must be accepted explicitly in any implementation decision, with broker rejection
or an uncertain response stopping remaining submissions and triggering durable
reconciliation. Never claim these checks guarantee an exit or prevent every
conflicting order.

The existing [`_submit_order`](../../../src/ggTrader/paper/trader.py) commits an
intent, verifies the session advisory lock, then calls the broker. Those are
separate operations: if the database session dies after verification, the lock
can be released before the broker call finishes and another cooperating run may
start. A pre-submit lock check cannot make the database lock and broker submission
atomic. The durable intent must remain recoverable by client ID, and a later run
must stop on an ambiguous or missing broker lookup rather than treat it as proof
that no order exists. Test lock loss before submission and a simulated loss after
verification with delayed or uncertain broker acceptance. A future implementation
must explicitly address the overlapping-run race; these checks reduce uncertainty
but do not guarantee that two conflicting submissions are impossible.

## Fractional quantity and submission precision

Alpaca's [fractional-trading documentation](https://docs.alpaca.markets/us/docs/fractional-trading)
says `qty` supports nine decimal places and truncates additional places. It also
says fractional orders require `day` time in force and that not every asset is
fractionable. The current adapter submits day market orders, but converts position
quantities to Python floats; the installed SDK models position `qty` and
`qty_available` as strings and `MarketOrderRequest.qty` as a float. This inspection
does not establish the exact serialized request or broker behavior for every value.

An [offline serialization probe](_fractional_sdk_serialization_20261111.json) using
installed `alpaca-py` 0.45.0 and `requests` 2.34.2 passed invented decimal strings
through `MarketOrderRequest.to_request_fields()` and a prepared JSON request, without
sending it. For example, `123456789.123456789` became the JSON number
`123456789.12345679`, an increase of 0.000000001 share; two small examples were
unchanged. The larger quantity is an extreme synthetic input, not an account holding
or a frequency estimate. An extra-precision example remained in the prepared JSON;
the probe did not simulate the broker's documented truncation. These examples do not
establish broker acceptance, fills, or behavior for all values.

Before implementing the exceptional exit, preserve broker quantity strings for
validation and compare decimal quantities without rounding upward. Establish and
test the exact SDK serialization path: the amount actually sent must be positive,
valid at the supported precision, and no greater than both the freshly verified
long position and its available quantity. If that cannot be demonstrated for a
value, stop rather than silently submit a larger sell. Report any unsellable or
unsubmitted remainder; do not describe an accepted order as a completed exit.
Check asset tradability and fractional eligibility, and add mocked tests for nine
decimal places, extra precision, floating-point boundary cases, zero after
truncation, partial availability, and a broker rejection. Confirm the account and
session rules before any paper validation; documentation and synthetic serialization
tests do not establish a fill or eliminate concurrent-writer races.

## Tradability failure boundary

The current [`resolve_untradable`](../../../src/ggTrader/paper/untradable.py)
catches broker lookup errors and returns only the configured exclusions. Therefore,
absence from its returned set does **not** prove that the broker verified an asset
as tradable. Its existing [test](../../../tests/paper/test_untradable.py) explicitly
checks that fail-soft behavior. The proposed exceptional exit path must not use
that absence as a successful eligibility check: it needs an explicit, fresh
positive broker tradability result for each candidate, with a missing, malformed,
or failed lookup stopping all remaining submissions under the recheck rule above.
A confirmed nontradable asset is skipped and reported; an earlier accepted sell
remains subject to reconciliation. Test a lookup failure with an empty configured
set and a failure partway through multiple candidate lookups so no unverified
candidate is submitted. This is a design requirement; it does not change the
existing path or establish Alpaca's live response behavior.

## Partial-fill accounting boundary

The current [`_reconcile_order_intents`](../../../src/ggTrader/paper/trader.py)
validates a working order's reported fill but leaves its intent unresolved. The
[`settle_order_intent`](../../../src/ggTrader/paper/persist.py) path writes the
associated trade and resolves the intent only after a terminal status. A working
partial fill can therefore have changed the broker's position or cash before its
trade appears in the local ledger. For the proposed exceptional path, determine
positions, available quantities and loss-halt inputs from freshly validated broker
state; do not reconstruct them by assuming the local trade ledger includes working
fills. Report an observed partial fill separately from a terminal booked fill.

Retain the cumulative terminal-fill settlement and its idempotence checks: do not
book interim fills a second time when the order becomes terminal. Add synthetic
cases in which a working partial sell changes the broker position, later cancels
with a nonzero fill, and is reconciled again after a crash. A missing or inconsistent
broker position or fill must stop remaining submissions. These requirements do not
assume broker position, cash and order snapshots are atomic, and they do not verify
later broker corrections or real execution behavior.

After an intent becomes `resolved`, the current recovery query no longer fetches
it. Repeating settlement with different fill details raises a conflict instead of
revising the trade. Alpaca's [Broker API trade-event reference](https://docs.alpaca.markets/us/reference/subscribetotradev2sse)
describes `trade_correct` and `trade_bust` events, so a terminal order is not by
itself proof that no subsequent correction is possible. That reference does not
establish whether this paper account receives those events or supports that Broker
API endpoint. Before relying on a clean ledger, independently verify the available
paper-account correction source and retention, define a reviewed reconciliation
process that preserves the original and correction, and treat detected mismatches
as unresolved. Mock a correction after resolution and a missed-event or unavailable
source; do not silently overwrite the booked trade or infer that no correction
occurred from an empty open-order list. The mechanism and any changes remain
unimplemented.

Alpaca also documents an [activity SSE reference using a paper API URL](https://docs.alpaca.markets/us/reference/subscribetoactivitiessse)
with historical queries via `since` or `since_id`; without either, it says no
historical events are returned. The reference warns that a slow client can lose
messages and that generated clients can hang. Alpaca's separate [activity guide](https://docs.alpaca.markets/us/docs/activity-sse)
describes corrections and busts and says lifecycle events without fills are not
included, but its integration instructions address Broker API partners. These
documents identify a possible investigation, not proof that this account can
access the endpoint, that paper simulations emit corrections, or that the two
documentation contexts have identical behavior. Any later integration must
verify those facts, persist and replay a cursor after durable processing, handle
dropped messages as an evidence gap, and retain a separate order-lifecycle view.
No endpoint was called or scheduled.
