# Working Task List

**Updated:** 2026-11-03 · **Status:** active worklist

The immediate actions are in [next steps](next_steps.md). This page explains the priorities, dependencies and calendar. Detailed evidence stays in the linked reports.

## Contents

- [Current priorities](#current-priorities)
- [Open decisions](#open-decisions)
- [Conditional and later work](#conditional-and-later-work)
- [Calendar](#calendar)
- [Completed work](#completed-work)
- [Standing rules](#standing-rules)

## Current priorities

Three **separate plans are proposed but not yet written**. Start the paper-parity plan and prepare the accruals plan in parallel. Prepare the sweep plan after the parity work clarifies the paper environment.

| Priority | Plan to write | Current position | Next planning outcome |
|---|---|---|---|
| 1 | **Paper parity and forward evidence** | A clean window and review date are **not established**: AVB remains untradable, and historical decision-time inputs are incomplete. | Address AVB, the relevant shadow book, incident/reset rules and ongoing observation; then design a separately reviewable way to preserve future signals, inputs, configuration and outcomes. [Readiness review](../research/paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md) |
| 2 | **Accruals feasibility and potential screen** | **Unresolved:** source usage, accounting revisions and exact portfolio metadata need work. No returns have been examined. | Define how to settle the feasibility questions; only then prepare a registered brief and consider a test under the existing standard. [Feasibility review](../research/stock-signals/feasibility/2026-11-02-accruals-feasibility.md) |
| 3 | **Two-symbol paper-sweep design** | A preliminary design exists; funding and order-handling policies remain proposals. | After parity clarifies the paper environment, plan the account boundaries, daily cash needs, partial fills, position controls and tax-lot scenarios. [Sweep design](../research/paper-system/cash-sweep/2026-11-02-two-symbol-sweep-design.md) |

The production paper trader remains frozen. A plan or design does not authorize a return test, installation, scheduling, container change or order. A parity incident resets the window; elapsed calendar time does not establish readiness.

**Models:** follow the [delegation guidance](../AGENTS.md#delegation-preference-2026-11-08). Check quota with the owner before writing a delegation plan. Keep private account records out of delegate prompts.

## Open decisions

- **Factor-scope wording (A1):** The previous report did not define “alternate factor definitions.” The [recommendation](../research/stock-signals/feasibility/2026-11-02-family-scope-review.md#7-wording-only-recommendation-2026-11-02) favors a narrow reading; the owner ruling remains open. It affects valuation, quality and q-factor variants, not the accruals task or the separate momentum closure. Even under a narrow reading, the variants are not yet defined or test-ready.
- **Insider timestamp amendment (A1):** The [amendment](../research/audits/event-timing/2026-10-23-insider-acceptance-time-amendment.md) remains a draft. PEAD and insider research are parked; leaving the amendment unapproved is valid and blocks neither priority plan.

These are decision records, not additional full plans. Neither changes a historical NO-GO.

## Conditional and later work

- **Tax-lot validation and integration (A6):** After a candidate passes Part A, validate the standalone `TaxLotBook` against a hand-checked year of lots, then scope the candidate's B0/B1 integration, point-in-time fundamentals, tax funding and audit requirements. It is not yet wired into a strategy or verified against a broker ledger. [Requirements](../research/paper-system/tax-lots/2026-11-02-after-tax-lot-engine-requirements.md)
- **Walk-forward policy (A7):** Keep existing defaults until a study proposes using WFO. At that point review the [fallback semantics](../research/programme/wfo-methods/2026-11-02-wfo-fallback-semantics.md) and [synthetic evidence](../research/programme/wfo-methods/2026-11-02-wfo-policy-synthetic.md); some proposed options would need code. The proposed accruals Part A screen does not use WFO.
- **Optional repository housekeeping (A8):** Inspect old branches and the stash before any cleanup; some branches are unmerged or attached to worktrees. Never delete `data/`, including the local event-timing cache. The earlier [task list history](superpowers/plans/2026-11-02-research-reliability-next-steps.md) and Git history retain prior context.

## Calendar

| When | Task | Condition |
|---|---|---|
| After each month's last NYSE close | A23 rebalance check (B1) | Once the real position is funded. The [advisory helper](cli_reference.md) does not place orders; funding status is unverified. |
| Date unknown | Paper freeze-readiness review (B2) | Resolve AVB, evidence gaps, shadow scope and the qualifying start/reset; require a documented readiness decision. |
| After the next closed research arc | Refresh the research snapshot (B3) | Use the designated `research-snapshot` skill. |
| 2027-09-30 | A23 annual review (B4) | Review the registered allocation, 15% versus 30% trade-off, rate clause and fund risks. [Century study](../research/allocation/century-leverage/2026-10-23-century-leverage.md) |

## Completed work

- **A3:** Three tax-lot and WFO test-helper fixes were integrated in `9eaf0389`; 117 targeted tests and Ruff checks passed during integration.
- **A5:** The old event-data buy-or-park question was resolved: both PEAD and insider studies are parked, with no paid data purchase.
- The [2026-11-02 follow-up plan](superpowers/plans/2026-11-02-paper-evidence-and-accruals-follow-up.md) documents the completed reconciliation and design work; it is not one of the three new plans above.

## Standing rules

- Taxable account only; no paid datasets or subscriptions.
- Preserve the production freeze, paper observations, parked event studies and historical NO-GOs. A NO-GO applies to its specified test and evidence; it does not authorize a project shutdown or account change.
- A Part A pass permits further research, not capital. B0 is a kill-only replay and B1 is a forward shadow; their full rules and statistical thresholds are in the [forward-paper standard](../research/programme/evidence-standard/2026-10-23-forward-paper-standard.md).
- Model review and historical research status are documented in the [research index](../research/RESEARCH_INDEX.md), [snapshot](../research/RESEARCH_SNAPSHOT.md) and Git history.
