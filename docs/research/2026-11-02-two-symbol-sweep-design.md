# Two-symbol cash-sweep design

**Status: design and future test plan only.** No paper code, configuration, order, or deployment is authorized here. The [A23 report](2026-09-30-static-fractional-leverage.md) and [worklist](../working_task_list.md) specify a 70/30 SPY/SSO target, a 22–38% SSO band and month-end checks with trades the next day when outside the band; the advisory helper describes cash-first funding and highest-basis long-term lots. The documented initial real-money staging is 10% of intended capital. Confirm the applicable book and instruments before implementation.

## Interaction with stock-sleeve cash

Proposed invariant: a real stock-sleeve cash requirement is handled as a liquidity event, not silently as an extra monthly band rebalance. Preserve sufficient cash to meet already committed obligations and avoid unapproved shorting or leverage. Whether and how the sweep supplies an intramonth shortfall, instrument/lot priority, reserves, partial fills and fallback behavior remain **open decisions**; the existing band rule alone does not determine them. No tax-lot choice is presumed executable until broker capabilities and available long-term lots are checked.

Use fresh, timestamped prices and reconciled broker positions. Missing or stale prices, uncertain order state, and insufficient eligible lots must surface an operator decision or safe stop rather than silently assume a fill. The proposed design must be reconciled with [freeze readiness](2026-11-02-paper-freeze-readiness.md), actual funding, broker support and the registered strategy before a future implementation.

## Scenarios for a future implementation

| Scenario | Required expected behavior |
|---|---|
| Month end, SSO in band | No band rebalance; preserve separately required cash. |
| Month end, SSO outside band | Propose restoring 30% under the registered cash-first and eligible-lot rules; check rounding and available funds. |
| Mid-month stock cash shortfall | Identify liquidity need separately; stop and surface unresolved funding policy rather than treating it as a scheduled band rebalance. |
| Eligible long-term lots insufficient | Surface the shortfall and obtain the applicable approved policy; do not invent short-term sales. |
| Stale price, interrupted or rejected order | Stop dependent actions and reconcile actual broker state before any retry; no assumed fill. |

A future test suite must also cover simultaneous stock exits and sweep orders, partial fills, fractional shares, tax basis, price timestamps, and duplicate cron invocations. Policy choices require explicit recording before expected trade amounts can be specified.
