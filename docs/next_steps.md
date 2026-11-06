# Next Steps

The next two tasks are plans. Neither authorizes a return test or a production change.
The [working task list](working_task_list.md) contains dependencies, open decisions,
and the calendar.

## Intraday: verify first actual recordings (activated 2026-10-27)

Hermes agent supervision (DeepSeek Flash, job `39172df68638`) is scheduled weekdays
at 14:00 Pacific, saving local reports: first recordings, daily health, weekly calendar, checkpoint and final review.
See [schedule record](../research/intraday/spy-momentum-prospective-observer/_intraday_observer_agent_schedule.json). The T3 schedule is disabled to avoid duplicate runs. Scheduling did not execute
the checks; the host and Hermes gateway must be available.

The automatic hypothetical journal is activated. All four standalone timers are
enabled/active; installed-service no-op checks and ledger integrity pass. See
[activation record](../research/intraday/spy-momentum-prospective-observer/2026-10-27-intraday-observer-activation.md).

1. Verify the October 29 anchor archive and October 30 first inputs, decision and
   archive. Check durable timestamps, source references and integrity; preserve
   faults and never backfill a missed decision. Perform the daily local health check.
2. At slot 20, November 30, 2026, review operational reliability under the frozen
   checkpoint rules. The final slot is November 3, 2027; no performance pass is implied.

The journal uses an isolated fixed release and dedicated runtime, not the paper
account. No production cron, Docker, DB, account positions or broker orders changed.
Historical r8 remains UNRESOLVED (data). The dividend comparison stays provisional.

## Current research decision (2026-10-23)

1. **Plan paper parity and forward evidence.** Resolve how to treat the persistent
   untradable AVB holding, which shadow book is in scope, and what starts or resets
   a clean observation window. Define daily evidence and a way to preserve future
   decision inputs and outcomes. The review date is unknown. A calendar date alone
   does not lift the production freeze. See the [readiness review](../research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md).
2. **Plan accruals feasibility in parallel.** Resolve source usage, accounting
   revisions, portfolio metadata, benchmark, and costs. Register a candidate brief
   only if the screen is feasible. Do not examine returns before registration. See
   the [feasibility review](../research/stock-signals/feasibility/2026-11-02-accruals-feasibility.md).

Plan the [two-symbol paper sweep](../research/paper-system/cash-sweep/2026-11-02-two-symbol-sweep-design.md)
after the parity work clarifies the paper environment. Implementation requires a
separate decision after the freeze and readiness gates are satisfied.

## Standing research rules

- Use the [forward-paper standard](../research/programme/evidence-standard/2026-10-23-forward-paper-standard.md).
  Part A requires long free history, Holm-corrected t of at least 3, and at least
  +1.0 percentage point per year net since 1994. B0 is a kill-only replay. B1 is
  a forward shadow book of at least 18 months. Before real money, require six
  months of clean Alpaca paper trading for mechanics.
- The production paper trader remains frozen. The observation window's qualifying
  start and any reset are unresolved. Continue observation and document unknowns.
- Momentum, the classic factor screen, and net issuance are closed NO-GO under
  their registered tests. The fixed-rule historical stock model is also NO-GO:
  14.19% CAGR versus SPY at 14.83%, with Newey-West t of -0.69. Exact historical
  live parity is unverified and the older holdout is inconclusive. See the
  [research snapshot](../research/RESEARCH_SNAPSHOT.md) for evidence and limitations.
- PEAD and insider studies are parked. No paid data will be purchased. The insider
  timestamp amendment remains a draft and does not block the current plans. See the
  [decision memo](../research/audits/event-timing/2026-10-23-event-timing-audit-decision.md).
- The account is taxable only. Follow the [owner instructions](../AGENTS.md#owner-instructions-2026-10-20).

## A23 allocation and calendar

The owner's 70% SPY / 30% SSO allocation and existing terms remain in force. Check
at month-end after the position is funded. Rebalance only if SSO leaves the 22–38%
range, using cash first and then highest-basis long-term lots. The
[advisory helper](cli_reference.md) proposes trades and does not place orders.
Funding status is unverified.

The next annual review is **2027-09-30**. The century study's registered proposal
keeps 30%, despite a -91.4% synthetic full-window drawdown that trips its -90%
ruin flag. The 15% arm avoids that flag at a 0.54 percentage point annual
cost in after-tax CAGR versus 30%. Review the trade-off at the scheduled date.
See the [century study](../research/allocation/century-leverage/2026-10-23-century-leverage.md).

## History

Prior operational sequences, superseded decisions, and resolved investigations are
preserved in the [archived worklist](archive/next-steps-history-2026-11-03.md).
