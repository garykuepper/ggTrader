# Two-symbol cash-sweep design

**Status: design and future test plan only.** No paper code, configuration, order, or deployment is authorized here. The [A23 report](../../allocation/static-leverage/2026-09-30-static-fractional-leverage.md) and [worklist](../../../docs/working_task_list.md) specify a 70/30 SPY/SSO target, a 22–38% SSO band and month-end checks with trades the next day when outside the band; the advisory helper describes cash-first funding and highest-basis long-term lots. The documented initial real-money staging is 10% of intended capital. Confirm the applicable book and instruments before implementation.

## Interaction with stock-sleeve cash

Proposed invariant: a real stock-sleeve cash requirement is handled as a liquidity event, not silently as an extra monthly band rebalance. Preserve sufficient cash to meet already committed obligations and avoid unapproved shorting or leverage. Whether and how the sweep supplies an intramonth shortfall, instrument/lot priority, reserves, partial fills and fallback behavior remain **open decisions**; the existing band rule alone does not determine them. No tax-lot choice is presumed executable until broker capabilities and available long-term lots are checked.

Use fresh, timestamped prices and reconciled broker positions. Missing or stale prices, uncertain order state, and insufficient eligible lots must surface an operator decision or safe stop rather than silently assume a fill. The proposed design must be reconciled with [freeze readiness](../observation-readiness/2026-11-02-paper-freeze-readiness.md), actual funding, broker support and the registered strategy before a future implementation.

## Scenarios for a future implementation

| Scenario | Required expected behavior |
|---|---|
| Month end, SSO in band | No band rebalance; preserve separately required cash. |
| Month end, SSO outside band | Propose restoring 30% under the registered cash-first and eligible-lot rules; check rounding and available funds. |
| Mid-month stock cash shortfall | Identify liquidity need separately; stop and surface unresolved funding policy rather than treating it as a scheduled band rebalance. |
| Eligible long-term lots insufficient | Surface the shortfall and obtain the applicable approved policy; do not invent short-term sales. |
| Stale price, interrupted or rejected order | Stop dependent actions and reconcile actual broker state before any retry; no assumed fill. |

A future test suite must also cover simultaneous stock exits and sweep orders, partial fills, fractional shares, tax basis, price timestamps, and duplicate cron invocations. Policy choices require explicit recording before expected trade amounts can be specified.

## Account and implementation boundaries

1. The deployed Alpaca **paper** trader uses a single sweep symbol, normally SPY.
   It can sell that holding to fund anticipated stock buys before processing
   stock exits and buys. The current source uses bounded sell polling; it does
   not establish that every partial or rejected funding sale halts later buys,
   or that all sweep orders have cross-run idempotency protection.
2. A two-symbol **paper** sweep is a future design requiring separate
   authorization. Its interaction with daily stock funding is not specified by
   the monthly band rule.
3. The owner's real taxable A23 allocation and offline CSV advisory helper are
   separate. Actual funding and real broker are unverified. Under the
   [forward-paper standard](../../programme/evidence-standard/2026-10-23-forward-paper-standard.md), any
   separately approved future real stock pilot uses the SPY/VOO core, not the
   A23 SSO leg. No real stock-purchase path is authorized by this document.

The helper measures SSO as `SSO/(SPY+SSO)`, excluding cash and other positions
([implementation](../../../src/ggTrader/lab/a23_rebalance.py)); using whole paper
NAV would be a different rule. Its lot ordering prefers long-term lots and
then highest basis, but it can propose short-term fallback with a warning; it
is not an LT-only enforcement mechanism. The inspected paper broker adapter
accepts a symbol and quantity, with no lot identifier; this does not establish
what a different real broker or broker-assisted process supports.

## Proposed behavior pending decisions

For paper funding, compare SPY-only sales against pro-rata SPY/SSO sales; neither
is an authorized intra-month band rebalance, and pro-rata SSO funding does not
change the real-account restriction above. For example, a $1,000 funding need
from $7,000 SPY and $3,000 SSO leaves the SSO fraction at 33⅓% if SPY alone is
sold, versus 30% for a $700/$300 split. These are synthetic examples, not
instructions. Whether cash contributions buy pro rata or the underweight leg
is also open.

| Situation | Expected result or proposed fail-safe |
|---|---|
| Month-end SSO exactly 22% or 38% | Inside the inclusive band; no band order. |
| Month-end SSO at 40% | Calculate a next-session proposal to restore 30%; check cash, lots, prices and order capability before any separately authorized submission. |
| Mid-month stock cash shortfall | Show the shortfall and wait for the paper funding policy; do not silently turn it into a band rebalance. |
| Eligible long-term lots insufficient | The existing helper can flag a short-term fallback; proposed automatic policy is to stop and surface the shortfall, not silently substitute lots or instruments. |
| Missing/stale price | Proposed: stop dependent actions until price freshness and positions can be established. |
| Duplicate run after a partial or rejected sale | Proposed: reconcile the original intent and cumulative fills; reserve committed cash, and do not retry an uncertain or already-filled quantity. |
| New cash or a stock exit | Keep contribution allocation separate from a scheduled band sale; the paper contribution policy remains open. |

These controls are proposals, not descriptions of verified production behavior.
The [parity reconciliation](../observation-readiness/2026-11-02-paper-freeze-readiness.md) does not
establish that they have operated in live paper sessions.

A future implementation spec must also define how both sweep symbols are
excluded from stock-slot, stop and concentration calculations, and whether the
existing minimum purchase clip applies per leg or to the combined sweep. The
current implementation handles one sweep symbol and returns one sweep action;
the two-symbol treatment is not yet implemented.
