# Paper-trading freeze readiness

**Status:** Evidence framework, not a clearance or deployment authorization. The existing observation window began with the 2026-09-29 deploy; the earliest possible review is approximately 2026-12-29, subject to confirmation of the start and any parity reset. See [current rules](../next_steps.md) and the [working task list](../working_task_list.md).

## Qualification and evidence

For each trading session, the person responsible for paper operations should record the observation date, reviewer, evidence location, and **pass, incident, or unknown**. An undocumented session is unknown, not clean. The owner or their designated operator is responsible for maintaining the ledger; the designation remains unconfirmed.

| Criterion | Evidence to inspect | Frequency and treatment |
|---|---|---|
| Stock names ≤ `MAX_POSITIONS` | Broker holdings/snapshots, effective config and `paper_snapshots`; exclude sweep assets only as defined in the deployed rule | Each session; missing holdings or config is unknown |
| No missed exits | Timestamped expected signals and decision inputs, broker orders/fills and `paper_trades` | Each session and after reconciliation; missing expected signals is unknown |
| No orphan or untradable holdings | Broker positions, tradability and signal/position reconciliation | Each session; missing broker state is unknown |
| Shadow weights and rebalances follow the rule | Timestamped shadow decisions, target weights, rule version, prices and shadow book records | Each scheduled decision and session-level exception review; unavailable shadow inputs are unknown |

Database rows alone do not establish expected signals, broker completeness or the rule in effect. Preserve dated evidence and reconciliation outputs. The [historical audit artifact](_replica_parity_20261019.json) does not establish exact historical live parity; absence of decision inputs and cleanup orders should not be treated as a signal failure or as proof of successful replication.

## Incident ledger

No new incident is asserted by this template; an empty ledger is not evidence that none occurred. The operator must reconstruct the period from the available evidence before making a clearance recommendation.

| Detected | Incident and session | Evidence link | Reviewer | Reset decision and rationale | New window start | Earliest review |
|---|---|---|---|---|---|---|
| Not yet reconstructed | Unknown | Outstanding | Unassigned | Pending review | Unconfirmed | Unconfirmed |

Any parity incident resets the clock under the current rule. Record the actual incident and effective restart date; do not assume the original calendar date still applies. Unknown days require investigation and cannot be counted as clean without evidence.

## Pre-lift checklist

- [ ] Reconcile the full window and its incident ledger; verify current effective dates and obtain a documented readiness decision.
- [ ] Verify current SSO fund expenses, waiver status and closure risk from current fund documents; the prior waiver expiry is not evidence of present terms.
- [ ] Obtain actual SSO distribution tax character and compare with the modeled bounds; retain sources and mark unresolved items.
- [ ] Establish and test an operator-readable rollback plan, including previous image/configuration and broker reconciliation.
- [ ] Before any separately authorized deployment, verify the documented push, published image build, `docker compose pull`, and `docker compose up -d` sequence and confirm the deployed image. See [project deployment guidance](../../AGENTS.md).
- [ ] Review the [sweep design](2026-11-02-two-symbol-sweep-design.md) and its unresolved decisions. A completed checklist is not itself authorization to change production.
