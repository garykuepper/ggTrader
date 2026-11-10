# ggTrader research guide

This is the table of contents for the current work on **what ggTrader should aim for, how to judge evidence, and what to investigate next**. Start here when you want the answer, then follow the links for details. The original studies remain in their existing folders; this page is a guide, not a second copy of their findings.

## Contents

- [Start here](#start-here) — the verdicts, audit, objectives and current assessment.
- [Current research questions](#current-research-questions) — stock selection, protection, attribution, the shadow and the next study.
- [How decisions are recorded](#how-decisions-are-recorded) — the verdict framework, existing rules and study templates.
- [Sources and further navigation](#sources-and-further-navigation) — primary sources, claim checks and the wider research library.

## Start here

1. **What have we tried?** Open the [verdict register](../VERDICT_REGISTER.md). It lists historical decisions alongside evidence and action annotations. Coverage is **incomplete**; included rows also contain summaries and are not a count of independent tests. Original reports remain authoritative.
2. **What did the major audit find?** Read the [executive summary](../audits/root-cause-2026-11-04/2026-11-05-executive-summary.md), then the [plain-English findings](../audits/root-cause-2026-11-04/findings/2026-11-05-findings-plain-english.md).
3. **What are we trying to achieve?** Read the [consolidated objectives and benchmarks report](benchmark-objectives/2026-11-10-objectives-benchmarks-consolidated.md). It discusses beating SPY, managing losses, sleeve attribution and realistic evidence; it does not approve a benchmark or strategy.
4. **What is under consideration now?** Read the [return and protection assessment](verdict-framework/2026-11-10-return-and-protection-assessment.md). Both objectives and all three protection measures are being explored. No new study or numerical target has been approved.

## Current research questions

- **Finding stocks:** [Accruals source check](verdict-framework/2026-11-10-accruals-source-metadata.md) records the proposed source, cell label and remaining data-vintage and usage questions. The earlier [feasibility assessment](../stock-signals/feasibility/2026-11-02-accruals-feasibility.md) has the full background. This is a possible research question, not evidence of an edge.
- **Managing losses:** [Three ways to study protection](verdict-framework/2026-11-10-protection-measures-explained.md) explains deepest fall, a bad year and recovery time. The comparator, study horizon and acceptable cost of protection remain open.
- **Understanding the previous stock sleeve:** The [replica artifact inventory](benchmark-objectives/2026-11-10-replica-attribution-artifact-inventory.md) explains what can be reconstructed from saved results and what is missing. It is a feasibility inventory, not an attribution result.
- **The cross-asset shadow:** The [decision brief](verdict-framework/2026-11-10-cross-asset-shadow-decision-brief.md) separates the historical shadow-only GO from unresolved shadow rules and observation records. It does not start a shadow or change the production freeze.
- **Choosing the next study:** The [objective decision brief](verdict-framework/2026-11-10-next-research-objective-brief.md) brings the open choices together. It is a proposal; no return test is authorized by the brief.

## How decisions are recorded

- The [research verdict framework design](../../docs/superpowers/specs/2026-11-10-research-verdict-framework-design.md) proposes separating what evidence shows, whether a registered rule was met, and what action follows. Check its stated approval status before treating any part as policy.
- The [forward-paper standard](evidence-standard/2026-10-23-forward-paper-standard.md) records the existing approved rules for its stated scope; the document also has known stale draft wording. Its governing resolutions and any future amendments matter. Do not automatically apply its stock-candidate rules to the earlier cross-asset shadow.
- [Return](verdict-framework/TEMPLATE-return-brief.md) and [defense](verdict-framework/TEMPLATE-defense-brief.md) templates help write a future study brief. A template or completed proposal does not grant approval to run it.
- The [retrospective review](verdict-framework/2026-11-10-retrospective-review.md) and the `review-*` / `discovery-*` files in [verdict-framework](verdict-framework/) document how historical records were checked. They are supporting audit material; use the register to find a verdict first.

## Sources and further navigation

- [Source and PDF record](benchmark-objectives/sources/README.md) identifies the primary materials used for the benchmark report and how to find archived copies in DeepThought. Its research-paper library and regulatory collection are separate; a saved source is not an endorsement or proof it is current.
- [Claim verification](benchmark-objectives/2026-11-10-claim-verification.md) records what was checked and remaining limits for the consolidated report.
- [Research home](../README.md) explains the wider folder layout. [Research index](../RESEARCH_INDEX.md) finds studies across the project, and [next steps](../../docs/next_steps.md) is the project worklist.

**Keeping this guide useful:** Link to a new document here only when it answers a current question or changes an important decision. Keep historical reports and verdicts in place, mark proposals as proposals, and update an existing overview when possible instead of creating another summary of the same material.
