# Research process audit: design

**Status:** Draft for owner review, 2026-11-01. This specifies the audit; it does
not itself conduct the audit, authorize a return test, reopen a study, change a
verdict, or modify the production freeze.

## Purpose and intended outcome

Determine whether ggTrader's research understood each strategy, translated its
mechanism faithfully, tested it with appropriate data and machinery, and judged
the result using a decision standard suited to the question. Explain which
negative conclusions are well supported, which are narrower than their wording,
and where missing evidence or process defects prevent a reliable conclusion.
Identify research-process gaps and possible future questions without choosing a
strategy to pursue now.

The owner wants credible, actionable research for a taxable account, primarily
NYSE/NasDAQ stocks with crypto also in scope. The audit should be skeptical of
both false discoveries and premature dismissal, and preserve honest negative
results rather than seeking a positive verdict.

## Scope and boundaries

- Review the research record broadly, including registered strategies,
  non-registry levers, allocation comparisons, research/data audits and the
  web-candidate backlog. Give priority to stock selection and stock-trading
  evidence; cover crypto as a separate mechanism/data track and other assets
  when they bear on the evidence, comparison or funding leg. Distinguish
  implemented and tested work from literature-only ideas.
- Begin with [research index](../../../research/RESEARCH_INDEX.md),
  [snapshot](../../../research/RESEARCH_SNAPSHOT.md),
  [worklist](../../next_steps.md), original reports, briefs, amendments,
  results and code. The index is a navigation aid, not proof of completeness;
  independently reconcile registry entries, dated reports and backlog records.
  The index is tracked alongside this design for a durable handoff.
- No paid datasets or subscriptions; no recommendation to change the taxable
  account, retire the stock sleeve, end paper trading or shut down the project.
  Preserve owner instructions in [AGENTS.md](../../../AGENTS.md).
- Preserve production freeze, paper observation, parked event studies, existing
  stops, approved protocols and historical verdicts. Audit classifications are
  assessments of evidence, not automatic changes to research authorization.
- This design authorizes read-only examination and planning. Any rerun,
  acquisition, software change, live/paper order or amendment needs its own
  explicit scope and any applicable existing authorization. The audit may
  describe proposed tests without running them.

## Approaches considered

1. **Inventory first:** classify the full record before deep examination. Strong
   coverage; risks superficial diagnoses and repetition of existing summaries.
2. **Deep cases first:** trace a handful of studies from original mechanism to
   verdict. Better for uncovering defects; can miss systemic or unique issues.
3. **Hybrid (proposed design):** make a lightweight inventory, examine shared
   infrastructure, use diverse deep-dive cases to calibrate a rubric, then
   apply that rubric across the record. This balances coverage and depth and
   makes shared assumptions visible before selecting further deep dives.

The owner requested a specification after the hybrid approach was presented;
this choice remains part of the draft for the owner to confirm in review.

## Distinguish the questions

For each study separate: (1) whether the source's proposed mechanism exists;
(2) whether the project's particular rule implements that mechanism and has an
edge; and (3) whether its actual portfolio improves the displaced alternative
under the specified risk, costs, tax and account constraints. An inexpensive
proxy can be a useful screen without being a replication. A result concerning
one of these questions does not automatically answer the others.

Separate a **decision** from an **evidentiary conclusion**. A registered hurdle
can legitimately cause a NO-GO even if the estimate is too uncertain to rule
out a useful effect. Failing to reject zero is not proof of no edge; conversely,
a noisy positive estimate is not proof of an edge. Strongly negative observed
performance and registered failures must be reported plainly, with uncertainty
and scope, not recast as positive potential.

## Audit method and deliverables

### 1. Coverage inventory and claim ledger

Create a durable, dated audit ledger under `research/` with one entry per
material tested claim or grouped identical mechanism. Link related strategy
variants instead of counting them as independent tests. Include literature-only
ideas as candidates, separate from performed tests; use group entries when the
backlog is large, but link each group to its existing register and disclose the
coverage boundary.

For each tested entry record, where recoverable: research question, mechanism,
source and source verification, tested rules and deviations, data and universe,
period, point-in-time/timing and corporate-action limitations, benchmark and
funding leg, execution/cost/tax assumptions, risk, gates, observed results and
uncertainty, artifacts and run provenance, original verdict and subsequent
amendments, shared-infrastructure dependencies, and whether the window was
previously inspected. Mark missing information **unknown**, not passed. Record
coverage counts by category and unresolved sources; they are inventory counts,
not independent statistical trials.

Outputs: a machine-readable ledger and human-readable index with direct links to
supporting records. Specify the schema and exact paths during implementation
planning; retain original results, document uncertainty, and do not hand-edit
the generated research snapshot.

### 2. Shared-machinery audit

Trace the research harness as used at the time of each run, not just the current
code. Cover anchor construction, gates and halt/release behavior; parameter
selection vs actual deployed fallback; fold overlap, WFE and statistical units;
signal and rebalance timing; cash, leverage and sizing; universe and security
identity; corporate actions; costs; benchmark and annualization conventions,
including whether 252-day metrics reached seven-day crypto results. Record the
dates/commits of relevant fixes and map affected studies to their actual versions
when evidence permits. Separate verified mechanisms from hypotheses about impact.

Evaluate whether synthetic positive and negative controls, sensitivity to a
predefined economically meaningful effect, and power or minimum detectable
effect calculations would reveal failure modes. Specify assumptions and
limitations; do not infer power from a non-significant result or equate a passing
control with proof that market data are clean. Propose any computational checks
for separate authorization; no market-return reruns are included here.

### 3. Representative deep dives and rubric calibration

Start with three cases that test distinct machinery, not because they appear
promising: pairs (WFO and cadence), momentum (external long-history analogue),
and the fixed-rule production replica (portfolio vs selection and parity).
Include the corrected FOMC event-study statistics as a statistical-unit check;
add a crypto/data-feasibility case if available evidence allows. These are audit
cases, not candidates selected for new trading research.

For each, trace primary source to documented rule, implementation, data,
portfolio construction, benchmark, outcome and original verdict. Test the audit
rubric against both evidence supporting and limiting the conclusion. Check the
known pairs rerun anomaly, proxy correspondence, cost assumptions, anchor usage,
replica exposure and parity, and event-day clustering against primary artifacts;
do not accept reviewer assertions as facts without verification. List other
studies affected by each confirmed shared issue, distinguishing documented
impact from possible impact. Select additional deep dives by declared criteria:
coverage of previously unexamined mechanism and harness type, materiality of a
potential defect, and availability of evidence, not past attractive returns.

### 4. Apply the rubric and evaluate NO-GO standards

For every in-scope claim assign independent fields for:

- **Historical decision:** original GO/NO-GO/stop and amendment, unchanged.
- **Question and fidelity:** faithful / materially changed / unknown, with
  rationale and sources.
- **Evidence validity:** usable for its stated question / qualified / invalid /
  unresolved, with the defect's direction and impact unknown when not measured.
- **Inference:** effect not demonstrated / evidence against a stated meaningful
  effect / adverse point estimate / inconclusive / not assessed. Do not claim
  an effect is excluded unless a suitable pre-specified interval or test supports
  it under stated assumptions; disclose multiple-testing and selection history.
- **Possible follow-up:** none identified / documentary clarification /
  bounded technical diagnosis / a separately scoped new or corrected study.
  A follow-up is a proposal, not permission to reopen or run it.

For each decision standard ask whether it matches the economic objective,
benchmark, risk, measurement horizon, realistic taxable implementation and
availability of evidence. Assess false-positive and false-negative tradeoffs,
the family of attempted hypotheses across studies, confidence intervals or
minimum detectable effects where feasible, regime and sample dependence,
proxy limitations, and whether costs/taxes were measured or assumed. Review
the existing Part A / B0 / B1 standard, including its explicitly limited forward
power, without silently changing it or applying its stock-specific thresholds
automatically to crypto. Distinguish a registered economic hurdle from a
statistical test and from a paper-trading mechanics gate.

### 5. Synthesis and owner handoff

Produce a dated report that explains root-cause hypotheses and supported
findings, compares the competing explanations for many NO-GOs, identifies
research gaps and unresolved limitations, and groups conclusions by confidence
and study scope. Include the strongest counterargument to every material finding,
negative evidence that survives review, and a list of potential future research
questions with dependencies and data feasibility—not a ranking of strategies or
a recommendation to trade. Present any suggested changes to the research
standard as alternatives for owner discussion, with costs and error tradeoffs.
Keep research ideas, diagnostics and any subsequent execution plan separate.

## Evidence and quality requirements

- Cite file paths, sections, artifact keys and code revisions; distinguish
  observed facts, calculations, inference and unresolved claims. Check any
  external primary-source claims needed to evaluate strategy fidelity; clearly
  label source access failures and unverified claims.
- Preserve raw evidence, stopped/invalid runs and NO-GOs; don't cherry-pick
  favorable metrics, treat earlier test windows as fresh OOS, or invent
  reproducibility for missing inputs. Explain non-comparable windows, baselines
  and data vintages. New cited quantitative computations require a reproducible
  artifact if later authorized.
- Inventory prior variants and investigations. A changed cadence, hedge,
  universe or benchmark is a new hypothesis; repeated attempts accumulate
  selection bias even across separately registered studies.
- Validate ledger links and coverage against the registry, reports and backlog;
  independently check representative claims against raw artifacts and code.
  Review contradictions and unresolved gaps rather than overwriting them.

## Acceptance criteria

The audit is ready for owner review when its ledger and report cover the
inventory and disclosed exclusions; shared dependencies and known fixes are
mapped with evidence; representative cases have source-to-verdict traces; each
material claim has a scoped historical decision and separate evidentiary
assessment; NO-GO standards and power limitations are assessed; and every
proposed second look has a specific rationale, missing evidence, known risks and
an explicit authorization boundary. The report must preserve unfavorable
results, production freeze and standing owner decisions, and must not imply
that an audit alone validates or authorizes a trading strategy.
