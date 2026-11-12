# Working Task List

## Status at a glance

Development and research are paused as of 2026-11-12 with a clean slate.

| Status | What |
|---|---|
| Done | Membership database, weekly check, press-release source and price backfill are merged on main. All 503 members have bars. |
| Done | Eight stock screens, the verdict register and the [category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md) are committed on the branch. |
| Waiting on you | Approve or decline the proposed paper-trader halt policy ([design note](../research/paper-system/observation-readiness/2026-11-11-unresolved-order-exit-design.md)). |
| Waiting on you | Two unsent drafts: an [AVB broker inquiry](../research/paper-system/observation-readiness/2026-11-11-avb-broker-inquiry-draft.md) and a [data-permission request](../research/stock-signals/feasibility/2026-11-11-listing-data-permission-draft.md). |
| Waiting on you | One open ruling on factor-scope wording ([details](#open-decisions)). |
| Waiting on review | Branch `feature/verdict-register` (four commits: engine, paper trader, research, category report) is not merged or pushed. It awaits your approval of the halt policy. |
| Waiting on review | Sol's two blocking issues in the paper commit are fixed. Sol's final re-check and Opus's review are clean ([details](where-we-left-it-2026-11-12.md)). Main holds one unpushed, undeployed commit, `783c7267`. |
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

1. Merge branch `feature/verdict-register` only with the owner's explicit approval of the halt policy. Sol's blocking findings are fixed, and both Sol and Opus have re-checked the fixes. The plain-words policy summary is in the where-we-left-it note.
2. Before the 2026-11-30 checkpoint, do the manual intraday checks, because supervision is paused. Look for Saturday's weekly heartbeat.

Two gates are separate. Stock-level research waits on verified inputs and data rights. A clean paper-trading start waits on the AVB holding.

Full state: [where we left it](where-we-left-it-2026-11-12.md). Detail is in the sections below and in [next steps](next_steps.md).

**Updated:** 2026-11-12 · **Status:** paused, clean slate. The immediate actions are in [next steps](next_steps.md); evidence stays in the linked reports.

## Contents

- [Current priorities](#current-priorities)
- [Open decisions](#open-decisions)
- [Conditional and later work](#conditional-and-later-work)
- [Calendar](#calendar)
- [Completed work](#completed-work)
- [Standing rules](#standing-rules)
- [History](#history)

## Current priorities

These are evidence and planning drafts. None approves a new study or a production change.

| # | Work | Where it stands | Next outcome |
|---|---|---|---|
| 1 | **Paper parity and forward evidence** | No clean window or review date: the untradable AVB holding blocks a start, and historical decision-time inputs are incomplete. Owner chose the original 80/20 cross-asset shadow. | Settle AVB, the shadow's rule and reset rules; then design a separately reviewable way to preserve signals, inputs, configuration and outcomes. [Readiness review](../research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md) |
| 2 | **Expected-growth stock inputs** | The portfolio screen gap is +208 bps with t of 2.728, under the bar of 3. It does not pick current stocks. | Resolve source rights and verify dated input panels (identity, market equity, filing availability, enough training history) before proposing a separately approved stock-level study. [Input pilot](../research/stock-signals/feasibility/2026-11-11-expected-growth-stock-input-pilot.md); [screen results](../research/stock-signals/feasibility/2026-11-11-expected-growth-results.md) |
| 3 | **Two-symbol paper-sweep design** | Preliminary design; funding and order-handling policies are proposals. | After parity, plan account boundaries, daily cash needs, partial fills, position controls and tax-lot scenarios. [Sweep design](../research/paper-system/cash-sweep/2026-11-02-two-symbol-sweep-design.md) |

- The paper trader stays stopped. A plan or design does not authorize a return test, installation, schedule, container change or order.
- A parity incident resets the window; elapsed time alone does not establish readiness.
- Models: follow the [delegation guidance](../AGENTS.md#delegation-preference-2026-11-08). Check quota with the owner first; keep private account records out of delegate prompts.

## Open decisions

- **Factor-scope wording (A1):** the previous report left "alternate factor definitions" undefined.
  - The [recommendation](../research/stock-signals/feasibility/2026-11-02-family-scope-review.md#7-wording-only-recommendation-2026-11-02) favors a narrow reading; the owner ruling is open.
  - It touches valuation, quality and q-factor variants, not the separate momentum closure; the variants are not yet defined or test-ready.
- **Insider timestamp amendment (A1):** still a [draft](../research/audits/event-timing/2026-10-23-insider-acceptance-time-amendment.md). Leaving it unapproved blocks neither priority plan.
- **Paper halt policy:** owner chose the cautious exits-only design for preparation on 2026-11-11; it is not approved for trading. See the [design note](../research/paper-system/observation-readiness/2026-11-11-unresolved-order-exit-design.md).
- **Unsent drafts:** an [AVB broker inquiry](../research/paper-system/observation-readiness/2026-11-11-avb-broker-inquiry-draft.md) and a [listing-data permission request](../research/stock-signals/feasibility/2026-11-11-listing-data-permission-draft.md).
- These are decision records, not full plans. None changes a historical closure.

## Conditional and later work

- **Tax-lot validation and integration (A6):** after a candidate meets Part A, validate `TaxLotBook` against a hand-checked year of lots. Then scope the candidate's B0/B1 integration, point-in-time fundamentals, tax funding and audit requirements. It is not yet wired into a strategy or checked against a broker ledger. [Requirements](../research/paper-system/tax-lots/2026-11-02-after-tax-lot-engine-requirements.md)
- **Walk-forward policy (A7):** keep defaults until a study proposes using WFO; then review the [fallback semantics](../research/programme/wfo-methods/2026-11-02-wfo-fallback-semantics.md) and [synthetic evidence](../research/programme/wfo-methods/2026-11-02-wfo-policy-synthetic.md).
- **Repository housekeeping (A8):** inspect old branches and the stash before any cleanup; some are unmerged or attached to worktrees. Never delete `data/`, including the local event-timing cache.

## Calendar

| When | Task | Condition |
|---|---|---|
| Saturdays 06:00 Pacific | Weekly membership check | Runs on its own; look for the heartbeat message. |
| 2026-11-30 | Intraday observer slot-20 review | Operational reliability only. Supervision is paused, so do the manual checks. Final slot 2027-11-03. |
| After each month's last NYSE close | A23 rebalance check (B1) | Once the real position is funded. The [advisory helper](cli_reference.md) places no orders; funding status is unverified. |
| Date unknown | Paper freeze-readiness review (B2) | Resolve AVB, evidence gaps, shadow scope and the start/reset rule. Require a documented readiness decision. |
| After the next closed research arc | Refresh the research snapshot (B3) | Use the `research-snapshot` skill. |
| 2027-09-30 | A23 annual review (B4) | Registered allocation, 15% versus 30% trade-off, rate clause, fund risks. [Century study](../research/allocation/century-leverage/2026-10-23-century-leverage.md) |

## Completed work

- **Membership database:** versioned vintages, one reader, fail-closed live gate, weekly check, press-release second source and price backfill. Merged on main.
- **This week's research (on the branch):** eight stock screens, input pilot, verdict register and the [category report](../research/programme/research-summary/2026-11-12-strategy-categories-alpha-range.md).
- **A3:** three tax-lot and WFO test-helper fixes integrated in `9eaf0389`.
- **A5:** the old event-data question is settled: PEAD and insider studies are parked, and no data is purchased.

## Standing rules

- Taxable account only; no paid datasets or subscriptions.
- Preserve the production freeze, paper observations and parked event studies.
- A closed result applies to its specified test and evidence; it does not authorize a project shutdown or account change.
- Meeting Part A permits further research, not capital. B0 is a kill-only replay and B1 a forward shadow; rules are in the [forward-paper standard](../research/programme/evidence-standard/2026-10-23-forward-paper-standard.md).
- Status and history: [research index](../research/RESEARCH_INDEX.md), [snapshot](../research/RESEARCH_SNAPSHOT.md) and Git history.

## History

- Branch commits: engine `acb4eac5`, paper trader `47d8a3c9`, research `51b55492`, category report `0ae8739f`.
- A3 detail: 117 targeted tests and Ruff checks ran clean during integration.
- The earlier [task list](superpowers/plans/2026-11-02-research-reliability-next-steps.md) and the [2026-11-02 follow-up plan](superpowers/plans/2026-11-02-paper-evidence-and-accruals-follow-up.md) hold prior context; the follow-up documents the completed reconciliation and design work.
- The accruals screen was run and reviewed earlier: [result](../research/stock-signals/feasibility/2026-11-02-accruals-feasibility.md#first-screen-result--2026-11-10). It did not use WFO.
- Some proposed WFO options would need code.
