# Next Steps

## Status at a glance

Development and research are paused as of 2026-11-12 with a clean slate. Nothing below is urgent except the intraday checkpoint on 2026-11-30.

| Status | What |
|---|---|
| Done | Membership database, weekly check, press-release source and price backfill are merged on main. All 503 members have bars. |
| Done | Eight stock screens, the verdict register and the [category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md) are committed on the branch. |
| Waiting on you | Approve or decline the proposed paper-trader halt policy ([design note](../research/paper-system/observation-readiness/2026-11-11-unresolved-order-exit-design.md)). |
| Waiting on you | Two unsent drafts: an [AVB broker inquiry](../research/paper-system/observation-readiness/2026-11-11-avb-broker-inquiry-draft.md) and a [data-permission request](../research/stock-signals/feasibility/2026-11-11-listing-data-permission-draft.md). |
| Waiting on you | One open ruling on factor-scope wording ([details](working_task_list.md#open-decisions)). |
| Waiting on review | Branch `feature/verdict-register` (four commits: engine, paper trader, research, category report) is not merged or pushed. It awaits a Sol review. |
| Waiting on review | Opus found no blocking issues in the engine and paper commits. Main holds one unpushed, undeployed commit, `783c7267`. |
| Running alone | Weekly membership check, Saturdays 06:00 Pacific, with a heartbeat message. |
| Running alone | Four intraday observer recording timers. |
| Stopped | Paper trader, since 2026-11-06. Cron lines are commented out. |
| Paused | Intraday supervision job, since 2026-11-06 09:18 Pacific. The reason is unknown and no alert exists. |
| Parked | PEAD and insider-trade studies, and crypto. |

Terms:

- **bps**: basis points. 100 bps is one percentage point a year.
- **AVB**: the paper account's frozen holding, which the broker marks untradable.
- **Parity**: the paper trader doing what the backtest says it would.
- **WFO**: walk-forward optimization, rolling tests that tune on past data and score on later data.
- **Part A**: the first stage of the evidence standard. It needs long free history, a strict significance test and at least +1.0 percentage point a year.
- **B0 and B1**: a replay that can only rule an idea out, and a forward paper-only copy ("shadow") kept for at least 18 months.
- **Holm-corrected t and Newey-West t**: significance tests that allow for trying many ideas and for returns that repeat patterns over time.
- **A23**: the owner's 70% SPY / 30% SSO allocation. **PEAD**: drift in a stock's price after earnings news.

### To resume

1. Merge branch `feature/verdict-register` only after a Sol review, no unresolved blocking findings and the owner's explicit approval of the halt policy.
2. Before the 2026-11-30 checkpoint, do the manual intraday checks, because supervision is paused. Look for Saturday's weekly heartbeat.

Two gates are separate. Stock-level research waits on verified inputs and data rights. A clean paper-trading start waits on the AVB holding.

Full state: [where we left it](where-we-left-it-2026-11-12.md). Detail is in the sections below and in the [working task list](working_task_list.md).

## Intraday observer: checkpoint on 2026-11-30 (activated 2026-10-27)

- The observer keeps a hypothetical journal on an isolated release. It touches no paper account, cron, Docker, database or orders.
- All four recording timers are enabled and active. See the [activation record](../research/intraday/spy-momentum-prospective-observer/2026-10-27-intraday-observer-activation.md).
- Hermes supervision (job `39172df68638`) is **paused**. Its job record says since 2026-11-06 09:18 Pacific, with no reason and no external alert.
- Until it is restored, the operator does a daily local health check and a weekly or notice-driven calendar review.
- Limits: historical r8 stays unresolved for lack of data, and the dividend comparison stays provisional.
- Preserve faults and never backfill a missed decision.
- The October 29 anchor and October 30 first stages were checked on 2026-11-11 (record, source hash, integrity); limits are in the activation record.
- Checkpoint: slot 20 on 2026-11-30 reviews operational reliability under the frozen rules. The final slot is 2027-11-03. No performance result is implied.

## Current research work

Both items are paused with the project. Neither authorizes a return test or a production change.

1. **Paper parity and forward evidence.** The untradable AVB holding blocks a clean observation start until resolved.
   - Owner choices of 2026-11-11: the original 80/20 cross-asset shadow for future parity, plus a three-month mechanics review and a 12-month performance review.
   - A shadow is a paper-only copy that places no orders.
   - A proposed reconciliation separates missing evidence, blocked exits and unresolved fills.
   - Open: the governing rule, what starts or resets a clean window, daily evidence, and how to preserve future decision inputs and outcomes.
   - The review date is unknown. A date alone does not lift the freeze. See the [proposed reconciliation](../research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md#proposed-session-by-session-reconciliation).
2. **Can expected-growth inputs support individual-stock research?** No validated stock picker exists. See the [research guide](../research/programme/RESEARCH_GUIDE.md#stock-selection-research) and the [input pilot](../research/stock-signals/feasibility/2026-11-11-expected-growth-stock-input-pilot.md).
   - Gaps: identity, timing, accounting and market-value inputs across a dated universe with enough history.
   - A new return study needs its own registered question and owner approval.
   - Report evidence and uncertainty without new GO/NO-GO labels.

Later: plan the [two-symbol paper sweep](../research/paper-system/cash-sweep/2026-11-02-two-symbol-sweep-design.md) after parity work. Building it needs a separate decision, after the freeze and readiness gates are met.

## Membership database (done, merged on main)

- The S&P 500 membership list lives in PostgreSQL schema `universe` as versioned vintages, read through one reader, `MembershipSource`.
- Research names its source with `--membership` (or `GGT_MEMBERSHIP`); there is no default.
- The live trader's gate refuses a list that is unreadable, empty, unmatched against the SPY fund holdings, mismatched, stale or future-dated.
- A live run reads only the latest database vintage.
- The weekly job (`ggt db membership weekly`) is enabled by the owner's go of 2026-11-09: timer `ggtrader-membership-weekly.timer`, Saturdays 06:00 Pacific.
- Official S&P DJI press releases are the second source; a human saves the release page before any `load-sp500 --commit`.
- Open checks (Appendix C6) are closed; see the [evidence note](../research/programme/membership-database/2026-11-09-open-checks-closed-with-press-releases.md), the [architecture guide](architecture.md#membership-database-and-the-single-reader) and the [CLI reference](cli_reference.md).

## Standing research rules

- Follow the [forward-paper standard](../research/programme/evidence-standard/2026-10-23-forward-paper-standard.md).
- Part A: long free history, Holm-corrected t of at least 3, and at least +1.0 percentage point per year net since 1994.
- B0 is a kill-only replay. B1 is a forward shadow book of at least 18 months.
- Real money needs six months of clean Alpaca paper trading first.
- The paper trader has been stopped since 2026-11-06, and the owner will say when it returns.
- The observation window's qualifying start and any reset are unresolved. Document unknowns.
- Closed NO-GO under their registered tests: momentum, the classic factor screen, net issuance. Each applies only to its registered test. See the [research snapshot](../research/RESEARCH_SNAPSHOT.md).
- The owner retired new GO/NO-GO determinations. Older classifications stay as history.
- The fixed-rule stock model is also closed NO-GO: 14.19% a year against SPY's 14.83%, with a Newey-West t of -0.69.
- Live parity is unverified, and the older holdout is inconclusive.
- This week's eight screens did not clear both hurdles. No category shows proven alpha ([category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md)).
- PEAD and insider studies are parked; the insider timestamp amendment is a draft. No paid data. See the [decision memo](../research/audits/event-timing/2026-10-23-event-timing-audit-decision.md).
- The account is taxable only. Follow the [owner instructions](../AGENTS.md#owner-instructions-2026-10-20).

## A23 allocation and calendar

- Terms: the owner's 70% SPY / 30% SSO allocation. Funding status is unverified.
- Check at month-end once funded. Rebalance only if SSO leaves the 22-38% range, using cash first, then highest-basis long-term lots.
- The [advisory helper](cli_reference.md) proposes trades and places no orders.
- Next annual review: **2027-09-30**.
- The century study's registered proposal keeps 30%, despite a -91.4% synthetic full-window drawdown that trips its -90% ruin flag.
- The 15% arm avoids that flag at a 0.54 percentage point annual cost in after-tax CAGR versus 30%. Review the trade-off then.
- See the [century study](../research/allocation/century-leverage/2026-10-23-century-leverage.md).

## History

- Branch commits: engine `acb4eac5`, paper trader `47d8a3c9`, research `51b55492`, category report `0ae8739f`.
Prior sequences, superseded decisions and resolved investigations are in the [archived worklist](archive/next-steps-history-2026-11-03.md). Older detail kept here:

- Membership vintages: 1 is the frozen file import (`data/universe/sp500_constituents_history.csv.gz`, immutable), 2 the upstream download to 2026-08-18, 3 a curated gap-fill to 2026-11-04.
- On 2026-11-09 the price-table backfill (`scripts/backfill_missing_members.py`) filled 13 current members with no or stale bars: 19,491 rows inserted, 4 partial bars refreshed.
- The weekly job's first live run sent its heartbeat on 2026-11-09. It runs from the host virtualenv; the `ggtrader_live` image is untouched.
- A proposal now carries the press-release date and clears a row's flag when it agrees with another source.
- The intraday journal is activated: installed-service no-op checks and ledger integrity were clean. Historical r8 remains UNRESOLVED (data); the dividend comparison stays provisional.
- Hermes supervision had a configured weekday 14:00 Pacific schedule; its [schedule record](../research/intraday/spy-momentum-prospective-observer/_intraday_observer_agent_schedule.json) is an activation-time record and the T3 schedule is disabled.
- The insider timestamp amendment does not block the current plans.
