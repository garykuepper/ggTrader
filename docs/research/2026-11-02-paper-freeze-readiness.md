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

The populated ledger and its unresolved reset disposition are in
[Incident and clock disposition](#incident-and-clock-disposition) below. Any
parity incident resets the clock under the current rule. Unknown days require
investigation and cannot be counted as clean without evidence.

## Pre-lift checklist

- [ ] Reconcile the full window and its incident ledger; verify current effective dates and obtain a documented readiness decision.
- [ ] Verify current SSO fund expenses, waiver status and closure risk from current fund documents; the prior waiver expiry is not evidence of present terms.
- [ ] Obtain actual SSO distribution tax character and compare with the modeled bounds; retain sources and mark unresolved items.
- [ ] Establish and test an operator-readable rollback plan, including previous image/configuration and broker reconciliation.
- [ ] Before any separately authorized deployment, verify the documented push, published image build, `docker compose pull`, and `docker compose up -d` sequence and confirm the deployed image. See [project deployment guidance](../../AGENTS.md).
- [ ] Review the [sweep design](2026-11-02-two-symbol-sweep-design.md) and its unresolved decisions. A completed checklist is not itself authorization to change production.

## Image provenance recorded before integration push

Read-only inspection on 2026-11-02 found the running `ggtrader_live` image ID
`sha256:31fcff2829203764bb5ff9c5ae80bc92565b321496b78e126c3e859f16f549ab`,
registry digest
`ghcr.io/garykuepper/ggtrader@sha256:0f2403f1bdee3148081982faf06713a6fec94542f6c2ab1388b993a27cbf37c7`,
and revision label `291e6dd4c8628033623a640cf086ab02fbe5a1ff`. Container
creation was 2026-09-30 00:12:09 UTC. These identify the observed image; registry
retention and rollback execution have not been tested. The cron uses `docker compose run --rm`, creating a fresh container from the
local image on each run. A push to `main` publishes an image and can move remote
`:latest`; pulling it would change the image used by subsequent cron runs even
without recreating the standing container. Do not pull or recreate production
during the freeze; a rollback plan must verify immutable image availability.

## Evidence reconciliation through 2026-11-02

**Status: not demonstrated clean.** The 2026-09-29 12:45 PT run preceded the
container recreation at approximately 17:12 PT and belongs to the pre-freeze
cleanup, not a frozen-image observation. The first scheduled session after it
was 2026-09-30. The window start and eligibility must be adjudicated against
actual events; approximately 2026-12-29 is not an established clearance date.

A read-only database query found 24 daily `paper_snapshots` dated 2026-09-30
through 2026-11-02 and `paper_trades` on 17 dates in that range. Local daily
logs exist for all 24 weekdays in the interval. Logs and snapshots are not a
complete decision-input archive or a substitute for broker fills and position
reconciliation. Local evidence references below are paths on the host, not
published account data.

| Criterion | 2026-09-30 through 2026-11-02 | Evidence and limits |
|---|---|---|
| Position cap | Unknown per session | Daily snapshots exist, but the contemporaneous effective configuration and complete broker reconciliation were not independently established. The deployment record expected four tradable stock names against a cap of three on 2026-09-30, pending natural exits; verify the actual runtime configuration and whether that transition met the literal criterion. The deployed code excludes broker-untradable holdings from its strategy count; that does not decide the written criterion. |
| No missed exits | Unknown per session | Expected decision-time signals and original bars were not established for each day. [Historical audit](_replica_parity_20261019.json) leaves one October sell and one October buy unverified, and covers orders only through 2026-10-19. Replaying revised data cannot prove original decisions. |
| No orphan or untradable holdings | Incident / unresolved persistent condition | All 24 snapshots contained AVB, and all 24 daily logs recorded its exclusion as untradable. The broker-frozen holding was documented before the window ([history](../next_steps.md)). Its presence conflicts with the literal no-untradable-holdings criterion, although deployed code excludes it from strategy slots and sells. This is one persistent condition, not an inference of 24 separate reset decisions. |
| Shadow weights and rebalances | Unknown per session | The nearby [worklist](../next_steps.md) describes a static cross-asset shadow; the separate prospective B1 stock shadow is a different book. The CSV `results/shadow_cross_asset.csv` contains rows only through 2026-10-16 and is recomputed/overwritten by `scripts/shadow_cross_asset.py`; it is not a contemporaneous decision archive. Confirm which book the criterion covers and obtain underlying decision evidence before classifying. |

Sessions audited (all 2026): Sep 30; Oct 1, 2, 5–9, 12–16, 19–23, 26–30; Nov 2. Each has the criterion dispositions in the table; ranges here contain weekdays only.

The local daily log pattern is `/home/flynn/logs/paper_trade_YYYYMMDD.log`;
raw account records, order identifiers and financial amounts are not published.
Individual logs contain data-error messages, so their presence alone does not
prove a complete market-data input. The cap, exits and shadow fields remain
unknown on each date; the AVB condition is evidenced for each of the 24 dates.
The owner or designated operator has not been independently identified for
ongoing observation.

### Incident and clock disposition

| Detected | Condition | Evidence | Reset disposition | Earliest review |
|---|---|---|---|---|
| 2026-09-30 (first observed post-deploy session) | Persistent broker-frozen AVB; documented before deploy and observed in every post-deploy snapshot/log in this audit | Local logs and `paper_snapshots`; [pre-deploy history](../next_steps.md) | Unresolved: written criterion and deployed exclusion conflict. No exception or restart decision is inferred. | Unknown until the condition and qualifying start are adjudicated. |

The deployed code checks the broker market clock and normally skips when closed,
but treats clock errors as market open. A 12:45 PT cron can occur after an
early close; a normal skip is not by itself evidence of a parity incident.
Classify future sessions from observed clock, signals and orders, including
clock failures, rather than treating a scheduled run as a fill or incident.

A proposed **future**, host-side evidence-capture tool could save timestamped
signal decisions, input data provenance, code/config hashes and a sanitized
run outcome to an append-only local record without submitting orders. It would
need read-only broker/database access, restricted storage, explicit handling of
partial failures and data revisions, and its own operational review. Nothing
here installs, schedules or authorizes that tool.
