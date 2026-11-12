# Next Steps

The next two tasks are plans. Neither authorizes a return test or a production change.
The [working task list](working_task_list.md) contains dependencies, open decisions,
and the calendar.

## Intraday: verify first actual recordings (activated 2026-10-27)

Hermes agent supervision (DeepSeek Flash, job `39172df68638`) is **paused** as of
the 2026-11-11 local check; its last recorded run was November 5. The reason for
the pause is unknown. Its configured weekday 14:00 Pacific schedule is not running.
The [schedule record](../research/intraday/spy-momentum-prospective-observer/_intraday_observer_agent_schedule.json)
is an activation-time record; the T3 schedule is disabled. The observer's four
recording timers are separate from the paused supervisor. Until supervision is
restored, the operator must perform the checks described below.

The automatic hypothetical journal is activated. All four standalone timers are
enabled/active; installed-service no-op checks and ledger integrity pass. See
[activation record](../research/intraday/spy-momentum-prospective-observer/2026-10-27-intraday-observer-activation.md).

1. Continue the daily local health check and weekly/notice-driven calendar review;
   preserve faults and never backfill a missed decision. The October 29 anchor and
   October 30 first stages passed a local record, source-hash and integrity check on
   November 11; details and limitations are in the activation record.
2. At slot 20, November 30, 2026, review operational reliability under the frozen
   checkpoint rules. The final slot is November 3, 2027; no performance pass is implied.

The journal uses an isolated fixed release and dedicated runtime, not the paper
account. No production cron, Docker, DB, account positions or broker orders changed.
Historical r8 remains UNRESOLVED (data). The dividend comparison stays provisional.

## Current research work

1. **Plan paper parity and forward evidence.** The owner chose to keep the persistent
   untradable AVB holding blocking a clean future observation start until resolved.
   The owner selected the original 80/20 cross-asset shadow for future parity and
   chose to prepare a three-month mechanics review plus a 12-month performance
   review. Resolve the detailed governing rule and what starts or resets a clean
   observation window. Define daily evidence and a way to preserve future
   decision inputs and outcomes. A proposed [session reconciliation](../research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md#proposed-session-by-session-reconciliation)
   distinguishes missing evidence, blocked exits and unresolved fills. The review
   date is unknown. A calendar date alone does not lift the production freeze.
2. **Check whether expected-growth inputs can support individual-stock research.**
   The reviewed portfolio screens did not establish a validated stock picker. The
   [research guide](../research/programme/RESEARCH_GUIDE.md#stock-selection-research)
   summarizes their results; the [input pilot](../research/stock-signals/feasibility/2026-11-11-expected-growth-stock-input-pilot.md)
   records the current identity, timing, accounting and market-value gaps. Verify
   those inputs across a sufficiently complete dated universe and enough history for
   the forecasting model. A new return study needs its own registered question and
   owner approval. Report evidence and uncertainty without new GO/NO-GO labels.

Plan the [two-symbol paper sweep](../research/paper-system/cash-sweep/2026-11-02-two-symbol-sweep-design.md)
after the parity work clarifies the paper environment. Implementation requires a
separate decision after the freeze and readiness gates are satisfied.

## Membership database

The S&P 500 membership list lives in PostgreSQL schema `universe` as versioned
vintages (1 the frozen file import, 2 the upstream download to 2026-08-18, 3 a curated
gap-fill to 2026-11-04) and is read through one reader, `MembershipSource`. Research
names its source with `--membership` (or `GGT_MEMBERSHIP`); there is no default. The old
file `data/universe/sp500_constituents_history.csv.gz` is the immutable vintage-1
fixture. The live trader's gate refuses to trade on a list that is unreadable, empty,
unmatched against the SPY fund holdings, mismatched, stale or future-dated, and a live
run reads only the latest database vintage. See the
[architecture guide](architecture.md#membership-database-and-the-single-reader) and the
[CLI reference](cli_reference.md).

Open, each owner-gated or unverified:

1. The weekly job (`ggt db membership weekly`) is ENABLED (owner's go, 2026-11-09): user timer
   `ggtrader-membership-weekly.timer`, Saturdays 06:00 Pacific, host-virtualenv variant (the
   `ggtrader_live` image is untouched). First live run sent its heartbeat on 2026-11-09.
2. The second free source is the S&P DJI press releases themselves (PR Newswire), the official
   dates. A proposal now carries the release date and clears a row's flag when it agrees with
   another source; a human still saves the release page as evidence before any `load-sp500 --commit`.
3. The spec's open checks (Appendix C6) are closed (2026-11-09, see the
   [evidence note](../research/programme/membership-database/2026-11-09-open-checks-closed-with-press-releases.md)).
   They found one data gap, 13 current members with no or stale bars in the price table,
   which was filled the same day with `scripts/backfill_missing_members.py` (19,491 rows inserted,
   4 partial bars refreshed; details in the evidence note).

## Standing research rules

- Use the [forward-paper standard](../research/programme/evidence-standard/2026-10-23-forward-paper-standard.md).
  Part A requires long free history, Holm-corrected t of at least 3, and at least
  +1.0 percentage point per year net since 1994. B0 is a kill-only replay. B1 is
  a forward shadow book of at least 18 months. Before real money, require six
  months of clean Alpaca paper trading for mechanics.
- The production paper trader has been stopped since 2026-11-06 (its cron lines are
  commented out); the owner will say when it returns. The
  observation window's qualifying start and any reset are unresolved. Document unknowns.
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
