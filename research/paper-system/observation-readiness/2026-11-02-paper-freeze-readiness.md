# Paper-trading freeze readiness

**Status:** Evidence framework, not a clearance or deployment authorization. The
qualifying observation-window start and any reset remain unresolved; the 2026-09-29
run preceded the frozen-image deployment, and a calendar date does not establish
readiness. The paper trader has been stopped since 2026-11-06. See [current rules](../../../docs/next_steps.md)
and the [working task list](../../../docs/working_task_list.md).

**Owner decision, 2026-11-11:** The broker-frozen AVB holding blocks the start of
a clean future observation window until the condition is resolved. This decision
does not establish a start date, waive any other criterion, restart the trader or
authorize an order.

**Owner decision, 2026-11-11:** The shadow component of a future paper-trading
parity observation is the original 80/20 cross-asset shadow. The B1 stock-candidate
shadow is separate. This scope choice does not start either book, settle the
cross-asset shadow's governing rule or start/reset conditions, approve a study,
restart paper trading or authorize an order.

**Owner schedule choice, 2026-11-11:** Prepare a three-month mechanics review
followed by a 12-month performance review for that cross-asset shadow. The choice
does not approve the detailed performance thresholds or establish a start, reset,
review date or authorization to run.

**Merger reference:** AvalonBay's [closing filing](https://investors.avalonbay.com/sec-filings/all-sec-filings/content/0001104659-26-097833/tm2623381d1_8k.htm)
describes an August 17, 2026 exchange of 2.793 Vivmark shares per eligible AVB
share, with cash in lieu of fractional shares if applicable. The surviving
company's [8-K](https://investors.vivmarkresidential.com/sec-filings/all-sec-filings/content/0001140361-26-033377/ef20080318_8k.htm)
states that its ticker changed from EQR to VMRK on August 18. Older local notes
refer to EQR; use VMRK when asking about the successor and verify the broker's
actual corporate-action treatment rather than assuming a conversion. An
[unsent support draft](2026-11-11-avb-broker-inquiry-draft.md) requests the
necessary reconciliation without authorizing an account reset or any order.

## Qualification and evidence

For each trading session, the person responsible for paper operations should record the observation date, reviewer, evidence location, and **pass, incident, or unknown**. An undocumented session is unknown, not clean. The owner or their designated operator is responsible for maintaining the ledger; the designation remains unconfirmed.

| Criterion | Evidence to inspect | Frequency and treatment |
|---|---|---|
| Stock names ≤ `MAX_POSITIONS` | Broker holdings/snapshots, effective config and `paper_snapshots`; exclude sweep assets only as defined in the deployed rule | Each session; missing holdings or config is unknown |
| No missed exits | Timestamped expected signals and decision inputs, broker orders/fills and `paper_trades` | Each session and after reconciliation; missing expected signals is unknown |
| No orphan or untradable holdings | Broker positions, tradability and signal/position reconciliation | Each session; missing broker state is unknown |
| Shadow weights and rebalances follow the rule | Timestamped shadow decisions, target weights, rule version, prices and shadow book records | Each scheduled decision and session-level exception review; unavailable shadow inputs are unknown |

Database rows alone do not establish expected signals, broker completeness or the rule in effect. Preserve dated evidence and reconciliation outputs. The [historical audit artifact](../../audits/production-replica/_replica_parity_20261019.json) does not establish exact historical live parity; absence of decision inputs and cleanup orders should not be treated as a signal failure or as proof of successful replication.

### Proposed session-by-session reconciliation

This is a review procedure, not a change to the qualification rule. Record the
actual deployed image and configuration, session and market-clock result, original
signal inputs and their availability, signal and risk-gate decisions, positions and
working orders before and after the run, durable intents, broker acknowledgments and
terminal fills, and the corresponding ledger entries. Bind each item to its source,
capture time and content hash; protect account details in restricted local storage.
Record corrections separately with their discovery time. A hash alone does not prove
when evidence existed, and a later snapshot cannot reconstruct an original decision.

For each held symbol, compare the expected exit under the **effective deployed rule**
with what the run actually attempted and what the broker subsequently did:

| Evidence situation | Proposed finding | Next evidence needed |
|---|---|---|
| Required decision inputs, rule version, or initial position missing | Unknown | Contemporaneous input and position evidence; a replay on revised data is not a substitute. |
| Verified exit required and eligible, but no submission is evidenced | Potential missed-exit incident | Reconcile the run's recorded gates, pending orders and any accepted broker order before deciding. |
| Exit submitted but still working or partially filled | Unresolved execution | Broker order and remaining position through a verified terminal state; submission is not a fill. |
| Exit blocked by a recorded gate or pending order | Record the actual block; do not assume either a pass or a missed exit | Compare the block with the applicable rule and investigate whether earlier orders can still fill. |
| No exit required and all required evidence reconciles | Pass for this criterion only | Other criteria and the session-level review remain separate. |

Clock errors need explicit evidence because the currently inspected local trader
assumes the market is open if its clock request fails. Dry-run intentions are not
broker executions. If reconciliation fails after an order may have been accepted,
preserve that ambiguity; do not report that no order was placed. A recorded incident
resets the clock under the existing rule; whether AVB's persistent pre-existing
condition prevents a qualifying start was resolved by the owner: it remains a
blocker until resolved. This does not change the separate reset rule.

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
- [ ] Before any separately authorized deployment, verify the documented push, published image build, `docker compose pull`, and `docker compose up -d` sequence and confirm the deployed image. See [project deployment guidance](../../../AGENTS.md).
- [ ] Review the [sweep design](../cash-sweep/2026-11-02-two-symbol-sweep-design.md) and its unresolved decisions. A completed checklist is not itself authorization to change production.

**SSO expense evidence, checked 2026-11-11:** The SEC-filed [September 28, 2026
summary prospectus](https://www.sec.gov/Archives/edgar/data/1174610/000117461026001012/f46053d1.htm)
reports annual operating expenses of 0.88% before and 0.84% after a 0.04%
waiver. It describes an advisory-fee waiver and a separate 0.95% expense cap
through September 30, 2027, subject to the terms in the prospectus. Transaction
and derivatives financing costs are not included in the displayed expense table.
This document supersedes the older September 2026 waiver date as evidence of
those disclosed terms; it does not establish that no later supplement changed
them or resolve fund closure risk. Keep the checklist open and recheck current
documents before any readiness decision.

**SSO distribution-tax evidence, checked 2026-11-11:** The [May 31, 2026
financial report](https://web.archive.org/web/20261001001755id_/https://www.proshares.com/globalassets/proshares/documents/annual-reports/annual_proshares_geared_financial_statement.pdf)
discusses qualified dividend income for the trust tax year ended October 31, 2025
on printed page 390 (PDF page 393), but the displayed text does not provide an SSO
qualified-dividend percentage. It also refers to reduced tax rates in 2024 while
describing the 2025 trust tax year, so its wording needs clarification. Its
long-term tax-equalization figure is a fund-level amount, not a shareholder
distribution. The local archived copy is
`/home/flynn/ggTrader-refs/proshares-sso-20261001/annual_geared_20260531.pdf`
(SHA-256 `023caf84103b52366cbe67f5c7d76d1c0aa73dd26a9e88c85c832c251301fae9`).
This does not establish the owner's calendar-year distribution character or quantify
the effect of the century study's zero-distribution assumption. Keep
the tax checklist open; actual broker reporting and applicable holding-period
requirements are separate evidence.

**Correction and further detail:** The same report's financial-statement notes,
printed page 372 (PDF page 375), provide a broad fund-level tax classification:
for the trust tax year ended October 31, 2025, SSO reports $49,935,027 in total
distributions, all in the ordinary-income column, and dashes in the net long-term
capital-gain and tax-return-of-capital columns. The previous statement that the
report did not resolve distribution character was too broad: it supplies this
classification, but not the qualified-dividend fraction or an individual
shareholder's tax outcome. Its trust tax year differs from a calendar tax year.
This is further evidence that the synthetic 2× leg's zero-distribution assumption
does not reproduce actual SSO distributions for that interval; it does not
quantify any effect on the study's after-tax results or establish future payouts.

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
| No missed exits | Unknown per session | Expected decision-time signals and original bars were not established for each day. [Historical audit](../../audits/production-replica/_replica_parity_20261019.json) leaves one October sell and one October buy unverified, and covers orders only through 2026-10-19. Replaying revised data cannot prove original decisions. |
| No orphan or untradable holdings | Incident / unresolved persistent condition | All 24 snapshots contained AVB, and all 24 daily logs recorded its exclusion as untradable. The broker-frozen holding was documented before the window ([history](../../../docs/archive/next-steps-history-2026-11-03.md)). Its presence conflicts with the literal no-untradable-holdings criterion, although deployed code excludes it from strategy slots and sells. This is one persistent condition, not an inference of 24 separate reset decisions. |
| Shadow weights and rebalances | Unknown per session | The owner selected the original 80/20 cross-asset shadow for future parity; the prospective B1 stock shadow is a different book. The CSV `results/shadow_cross_asset.csv` contains rows only through 2026-10-16 and is recomputed/overwritten by `scripts/shadow_cross_asset.py`; it is not a contemporaneous decision archive. Obtain underlying decision evidence before classifying the historical period. |

The original September 28 worklist entry (git commit `d3da809e`) introduced the
parity criterion alongside a three-month static 80/20 cross-asset shadow; the
separate [B1 standard](../../programme/evidence-standard/2026-10-23-forward-paper-standard.md)
was adopted later, on October 23. This chronology supports interpreting the
original reference as the cross-asset shadow, but the rule itself did not name the
book. The owner has now chosen that shadow as the future parity scope; the
remaining governing rules and observation start are still unresolved. The original
historical advice about account type and retirement is superseded by current owner
instructions.

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
| 2026-09-30 (first observed post-deploy session) | Persistent broker-frozen AVB; documented before deploy and observed in every post-deploy snapshot/log in this audit | Local logs and `paper_snapshots`; [pre-deploy history](../../../docs/archive/next-steps-history-2026-11-03.md) | Owner chose to keep the condition blocking a future clean start until resolved; no exception or restart is authorized. | Unknown until the condition is resolved and the remaining start criteria are adjudicated. |

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
