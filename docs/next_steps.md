# Next Steps

Forward collection is prepared but its timers are not installed or enabled. Daily bars and weekly shares now use separate jobs; catch-up marks days without an original listing. No forward rule is registered: wait until a rule survives choices 1–4 of the research agenda and collection covers its intended scoring period.

## Status at a glance

Status as of 2026-11-28: database consolidation, the dead-stock studies and the early-decades test are complete. The paper trader remains stopped; the next research runs await the owner’s choices.

| Status | What |
|---|---|
| Done | Membership database, weekly check, press-release source and price backfill are merged on main. All 503 members have bars. |
| Done | Eight stock screens, the verdict register and the [category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md) are merged on main (2026-11-12). |
| Done | The exits-only halt policy was approved on 2026-11-12 ([design note](../research/paper-system/observation-readiness/2026-11-11-unresolved-order-exit-design.md)). The branch `feature/verdict-register` was merged and pushed to `main` the same day. The code is not deployed and the trader stays stopped. |
| Done, deploy open | The refactor and its follow-ups are merged into `main` and pushed (2026-11-20, `8c0c4540`): phases 0 to 2 (`abe8cefa`), phases 3 to 5, the lint rules, the docs audit, the snapshot and completion II (the eight seam forwards retired, a live run refuses a non-durable ledger, `run_wfo` raises `WfoNotRun`, the membership CLI and weekly job tidy-ups, the persist writers on `engine.begin()`, docstring rules for `src/ggTrader`) (all on `origin/main`). The trader stays stopped and nothing is deployed. Left for the owner: the deploy (image rebuild), and an optional re-run of the real-database persist characterization, whose fixture is kept outside the repository (evidence folder `p6/t7/`). See the [refactor design](superpowers/specs/2026-11-17-refactor-design.md), the [completion II plan](superpowers/plans/2026-11-20-refactor-completion-ii.md) and the [changelog](changelog.md). |
| Done | Database consolidation, including a dump and restore test, data and labels from the SEC (the US Securities and Exchange Commission), price-table edit protection, a side-by-side price view and a frozen Yahoo copy, is merged and pushed to `origin/main` through `ec42f9ca`. See the [inventory](database-inventory-2026-11-26.md). |
| Done | Dead-stock studies are closed and merged: [first test](../research/stock-signals/dead-stock-first-test/2026-11-24-dead-stock-first-test.md), [fresh window](../research/stock-signals/dead-stock-fresh-window/2026-11-24-dead-stock-fresh-window.md), [random baseline](../research/stock-signals/dead-stock-random-baseline/2026-11-25-dead-stock-random-baseline.md), [momentum deaths](../research/stock-signals/dead-stock-momentum-deaths/2026-11-25-dead-stock-momentum-deaths.md), [payout rerun](../research/stock-signals/dead-stock-payout-rerun/2026-11-24-dead-stock-payout-rerun.md), [early filings](../research/stock-signals/dead-stock-early-filings/2026-11-25-dead-stock-early-filings.md), and [data integrity](../research/stock-signals/dead-stock-data-integrity/2026-11-24-dead-stock-data-integrity.md). |
| Done | Fresh dump `ggtrader_20261126.dump` taken after the SEC loads, frozen Yahoo copy and SPY table; its test restore on 2026-11-27 matched all 78 row counts and 6 checksums. The old `attempts` folder was deleted at the owner's request. The Sunday 04:30 timer still runs. |
| Done | Free Yahoo SPY for 1993–1999 added as the frozen table `market.spy_daily_yahoo_1993_1999` (1,749 rows). |
| Waiting on you | Choose one to three candidates from the [research agenda](../research/programme/next-research-agenda-2026-11-26.md); the [window-use ledger](../research/programme/window-use-ledger.md) records dates already used. |
| Done | Early-decades test scored once on 1969-02-03 to 1998-07-31 (2026-11-28): the first test's rule monthly, yearly and skipping 20% one-day jumpers, 18 settings, against the CRSP market return then SPY. Before 1993 it beats the comparison at 7 of 18 settings (6 on a continuous-account check), nearly all with dead stocks assumed to pay their last price in full; after 1993 it trails SPY at all 18 as registered; see the [report](../research/stock-signals/early-decades/2026-11-28-early-decades.md). Window spent; the lock files stay until the owner decides. Branch `t3/early-decades-test-tasks-1-5` is not merged or pushed. |
| Checkpoint | Intraday observer slot 20 is 2026-11-30; supervision remains paused and manual checks continue. |
| Waiting on you | Two unsent drafts: an [AVB broker inquiry](../research/paper-system/observation-readiness/2026-11-11-avb-broker-inquiry-draft.md) and a [data-permission request](../research/stock-signals/feasibility/2026-11-11-listing-data-permission-draft.md). |
| Waiting on you | One open ruling on factor-scope wording ([details](working_task_list.md#open-decisions)). |
| Done | The paper-trader order-recovery and halt changes (including `783c7267`) were reviewed by Sol and Opus on 2026-11-12 with no open findings and are on `main`, not deployed ([details](archive/where-we-left-it-2026-11-12.md)). |
| Running alone | Weekly membership check, Saturdays 06:00 Pacific, with a heartbeat message. |
| Running alone | The user timer `ggtrader-db-dump.timer` schedules a database dump Sundays at 04:30 Pacific; its next run is 2026-11-29. |
| Running alone | Four intraday observer recording timers. |
| Stopped | Paper trader, since 2026-11-06. Cron lines are commented out. |
| Paused | Intraday supervision job, since 2026-11-06 09:18 Pacific. The reason is unknown and no alert exists. |
| Parked | PEAD and insider-trade studies, and crypto. |

Terms:

- **bps**: basis points. 100 bps is one percentage point.
- **AVB**: the paper account's frozen holding, which the broker marks untradable.
- **Parity**: the paper trader doing what the backtest says it would.
- **WFO**: walk-forward optimization, rolling tests that tune on past data and score on later data.
- **Part A** (retired for new work): the former first stage of the evidence standard. It needed long free history, a strict significance test and at least +1.0 percentage point a year.
- **B0 and B1** (retired for new work): a replay that could only rule an idea out, and a forward paper-only copy ("shadow") that was kept for at least 18 months.
- **Holm-corrected t and Newey-West t** (retired as required bars for new work): significance tests that allowed for trying many ideas and for returns that repeat patterns over time.
- **A23**: the owner's 70% SPY / 30% SSO allocation. **PEAD**: drift in a stock's price after earnings news.

### To resume

1. The consolidation, inventory, ledger and agenda are merged and pushed (`origin/main` at `b295e821`). Deploying the paper changes and restarting the trader are separate owner decisions.
2. Before the 2026-11-30 checkpoint, do the manual intraday checks, because supervision is paused. Look for Saturday's weekly heartbeat.

New studies need verified inputs and any necessary data rights. A clean paper-trading start waits on the AVB holding.

Full state: [where we left it](archive/where-we-left-it-2026-11-12.md). Detail is in the sections below and in the [working task list](working_task_list.md).

## Intraday observer: checkpoint on 2026-11-30 (activated 2026-10-27)

- The observer keeps a hypothetical journal on an isolated release. It touches no paper account, cron, Docker, database or orders.
- All four recording timers are enabled and active. See the [activation record](../research/intraday/spy-momentum-prospective-observer/2026-10-27-intraday-observer-activation.md).
- Hermes supervision (job `39172df68638`) is **paused**. Its job record says since 2026-11-06 09:18 Pacific, with no reason and no external alert.
- Until it is restored, the operator does a daily local health check and a weekly or notice-driven calendar review.
- Limits: historical r8 stays unresolved for lack of data (a fresh r9 protocol scored the same rule once on 2026-11-15: neither threshold met, [result](../research/intraday/spy-momentum-feasibility/2026-11-15-intraday-half-hour-rule-result.md)), and the dividend comparison stays provisional.
- Preserve faults and never backfill a missed decision.
- The October 29 anchor and October 30 first stages were checked on 2026-11-11 (record, source hash, integrity); limits are in the activation record.
- Checkpoint: slot 20 on 2026-11-30 reviews operational reliability under the frozen rules. The final slot is 2027-11-03. No performance result is implied.

## Current research work

Paper readiness remains paused. The research agenda awaits the owner’s choice of studies; neither item authorizes a production change.

1. **Paper parity and forward evidence.** The untradable AVB holding blocks a clean observation start until resolved.
   - Owner choices of 2026-11-11: the original 80/20 cross-asset shadow for future parity, plus a three-month mechanics review and a 12-month performance review.
   - A shadow is a paper-only copy that places no orders.
   - A proposed reconciliation separates missing evidence, blocked exits and unresolved fills.
   - Open: the governing rule, what starts or resets a clean window, daily evidence, and how to preserve future decision inputs and outcomes.
   - The review date is unknown. A date alone does not lift the freeze. See the [proposed reconciliation](../research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md#proposed-session-by-session-reconciliation).
2. **Can expected-growth inputs support individual-stock research?** No validated stock picker exists. See the [research guide](../research/programme/RESEARCH_GUIDE.md#stock-selection-research) and the [input pilot](../research/stock-signals/feasibility/2026-11-11-expected-growth-stock-input-pilot.md).
   - Remaining input limits and available data are described in the [inventory](database-inventory-2026-11-26.md) and [research agenda](../research/programme/next-research-agenda-2026-11-26.md).
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

- **Reset 2026-11-20:** the owner wiped the old evidence rules (Part A, the t-stat bars, B0/B1, the six-month wait) for new work. New work follows four plain rules: test on data the strategy never saw; use only what was known each day and include stocks that later died; subtract costs and taxes and compare to SPY; paper trade first, and the owner decides when real money is earned. See [research instructions](../CLAUDE.md#research). The older [forward-paper standard](../research/programme/evidence-standard/2026-10-23-forward-paper-standard.md) stays as history.
- The paper trader has been stopped since 2026-11-06, and the owner will say when it returns.
- The observation window's qualifying start and any reset are unresolved. Document unknowns.
- Closed NO-GO under their registered tests: momentum, the classic factor screen, net issuance. Each applies only to its registered test. See the [research snapshot](../research/RESEARCH_SNAPSHOT.md).
- The owner retired new GO/NO-GO determinations. Older classifications stay as history.
- The fixed-rule stock model is also closed NO-GO: 14.19% a year against SPY's 14.83%, with a Newey-West t of -0.69.
- Live parity is unverified, and the older holdout is inconclusive.
- The eight screens of 2026-11-10 and 2026-11-11 did not clear both hurdles. No category shows proven alpha ([category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md)).
- PEAD and insider studies are parked; the insider timestamp amendment is a draft. No paid data. See the [decision memo](../research/audits/event-timing/2026-10-23-event-timing-audit-decision.md).
- The account is taxable only. Follow the [owner instructions](../CLAUDE.md#production-freeze-and-owner-instructions).

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
