# Research Verdict Framework: Return and Defense

**Date:** 2026-11-10  
**Status:** Draft for owner review; no policy change or test authorized.  
**Inputs:** [consolidated benchmark research](../../../research/programme/benchmark-objectives/2026-11-10-objectives-benchmarks-consolidated.md), [claim verification](../../../research/programme/benchmark-objectives/2026-11-10-claim-verification.md), [current forward-paper standard](../../../research/programme/evidence-standard/2026-10-23-forward-paper-standard.md), and independent Opus, Sol, Astra and Fable critiques. The lead has checked the cited project standard; review comments are advice, not adopted evidence.

## Intent and constraints

The owner wants to grow wealth, seek programmatically selected winners, manage losses and evaluate performance in falling markets. This framework records what the evidence says and what the project will do without equating failure to advance with proof that an idea has no edge. It supports research, not a live-trading change. Use free data, a taxable-account model, auditable costs and point-in-time inputs; preserve the production freeze and all previous verdicts. No recommendation to retire the stock sleeve, end paper trading, change account type or shut down the project follows.

## Approaches considered

1. **Clarify the existing GO/NO-GO only:** least disruption but leaves evidence uncertainty mixed with the action decision.
2. **GO/NO-GO/INCONCLUSIVE:** more expressive but still mixes statistical and practical reasons for halting.
3. **Evidence and action labels with a recorded protocol result:** the owner selected two labels; record whether the original registered rule was met as a factual field, not a third judgment label. Adopt this as the proposed design. It preserves decisions while separating statistical conclusions from operational or budget decisions.

The owner asked to design **both** return and defensive objectives. Either is a study's primary route, selected before evaluation. A prespecified joint objective is allowed with explicit co-primary success requirements. Multiple alternative routes to success need a preregistered multiplicity plan; report all secondary findings without promoting them to gates after seeing results.

## 1. Verdict record

Each candidate-stage record contains: study and specification version/hash; timestamp and owner-approved brief; primary objective; data and model vintage; benchmark; start/end dates and data availability cutoff; outcome and unit; economic hurdle and equality rule; point estimate and uncertainty or reason unavailable; validity and power assessment; **evidence label**; **registered protocol result**; **action label**; reason; risk/operational state; and evidence links. Record preliminary results and changes append-only with dates and provenance. A hash identifies content, not the time the content existed; use a separately specified timestamp witness before any forward decision where the governing standard requires it.

### Evidence label

- **SUPPORTED:** valid evidence satisfies the preregistered statistical criterion for the stated material benefit. Name the estimand and threshold; do not equate point-estimate success with support. A shadow or operational screen alone is not automatically statistical support.
- **EVIDENCE AGAINST THE REGISTERED BENEFIT:** valid evidence meets a preregistered criterion to rule out the stated material benefit. This does not necessarily mean return is negative or no smaller edge exists.
- **INCONCLUSIVE:** valid evidence does not satisfy either evidentiary criterion, including a test too weak to distinguish useful benefit from noise or a defensive evaluation lacking its registered stress coverage. Report bounds and data limitations, not just the label.
- **UNRESOLVED:** data, timing, accounting, execution or implementation fault prevents a valid inference. Preserve the fault and stop any affected progression; never report it as a pass.
- **DESCRIPTIVE:** post-hoc analyses or tests whose hypothesis, endpoint or decision rule was selected after inspecting their evaluation data. They can motivate a new hypothesis but cannot themselves establish prospective support. Reading a properly frozen result after its test does not retroactively make it descriptive. Record separately when a published hypothesis was developed on the same historical sample; a preregistered test on that sample is not necessarily an untouched temporal holdout.

Where an interval `[L,U]` is appropriate for the actual preregistered estimand and material benefit `m`, an example rule is `L > m` for support and `U < m` for evidence against that material benefit; otherwise inconclusive. The brief must decide confidence level, one/two-sided treatment, equality, error control and estimator *before* results. If the statistical test concerns mean arithmetic excess and the economic rule concerns CAGR or after-tax wealth, report them as separate quantities; neither test nor interval automatically proves the other. Model and power assumptions remain visible.

### Registered protocol result: factual field

Record **MET**, **NOT MET**, **STOPPED**, or **UNRESOLVED**, with the exact original rule and its version. A point estimate may fail an existing threshold while evidence is inconclusive. A stopped or unresolved test remains recorded; a stop is not proof of a negative expected edge. When a data fault affects validity, do not convert it into a valid NOT MET outcome.

### Action label

- **CONTINUE RESEARCH:** a specific feasible next question and evidence plan have already been authorized; cite the approval and its scope. The label itself grants no authorization.
- **ELIGIBLE FOR NEXT STAGE:** the registered criteria and all applicable evidence and operational prerequisites are met; this is eligibility, not an order or spending authorization.
- **HOLD / PARK THIS VERSION:** no progression now because of a stop, unresolved fault, inadequate evidence, an unapproved next question, or an explicit practical constraint; name the reason and what would change it.
- **CLOSE THIS SPECIFICATION:** its registered rule calls for closure or a documented decision ends this particular version; preserve evidence and keep other stock-picking research open.

No action label approves broker orders, changes production, starts cron, or authorizes capital. A safety fault halts the affected operation regardless of the research label. Do not discard the stopped outcome, hide its elapsed time, or reset a failed clock.

## 2. Return path

**Question:** Does the specified candidate produce a material benefit against its preregistered comparator, with acceptable risk and implementation costs?

The candidate brief fixes its universe and timing, portfolio or sleeve capital base, benchmark total return, cost/financing and tax conventions, primary endpoint, economic threshold, statistical test, decision rule, power/minimum detectable effect, dates and trial family. SPY total return remains a visible comparator. For an active-stock study also use a prespecified self-financing SPY-replacement sleeve counterfactual when inputs permit; define how idle cash, rebalancing, distributions and trading are handled. This is separate from whole-account value and not a claim of causal attribution without the required data.

The approved 2026-10-23 standard remains operative for stock candidates under its scope until explicitly amended. Its Part A, B0 and B1 registered rules and historical decisions are not silently replaced by this design. A forward B1 pass is an economic/operational screen under that standard, not automatic proof of a small statistical edge. For stages governed by the existing standard, stage eligibility follows its original registered gates: a B0 that meets its kill-only policy rule may be eligible for the next authorized research stage even when its evidence label is INCONCLUSIVE; a valid B1 policy pass may be eligible for the next specified stage without a new statistical-support gate, subject to all existing prerequisites. A STOPPED or UNRESOLVED stage cannot progress under that attempt. No label changes the original thresholds or grants capital authorization. Statistical significance in any short period depends on effect and noise; do not claim it is mathematically impossible, but do not infer support for a modest edge from the policy pass.

## 3. Defensive path

**Question:** Does the candidate improve a specifically defined downside outcome at an acceptable cost in wealth, compared with a suitable alternative?

A valid advancement rule must have **both** a material improvement in one registered downside endpoint and a separately specified wealth noninferiority/opportunity-cost condition, plus operational/risk guardrails. The brief defines a minimum acceptable benefit and the owner's maximum acceptable return/wealth sacrifice before evaluating results. Nonsignificant harm is not proof of acceptable harm. Report SPY in every case so opportunity cost is visible.

The primary economic comparator is the owner-selected fixed investable passive policy appropriate to the question; the brief specifies instruments, starting allocation, rebalance, costs, funding, dates, cash flows and tax treatment. A separately preregistered lagged risk-matched rule can be primary only when it answers the specific hypothesis, and must specify estimator, lookback and warm-up, information lag, rebalance, assets, caps, cash yield, cost and missing-data handling. A changing match can absorb the timing decision being evaluated; show fixed-policy and SPY results alongside it. Full-sample matching is descriptive, not a prospective benchmark.

The brief selects a single downside endpoint, stress definitions, coverage minimum and inference method. Maximum drawdown is useful context but a single path is fragile as the sole inferential endpoint. Define negative-month, sustained-decline, fast-crash, recovery and whipsaw measures without treating overlapping days as independent crises. A live signal uses only information then available. Historical regimes identified with hindsight may be descriptive or support a carefully preregistered conditional analysis; the prospective defense amendment must decide their permitted evidentiary role, account for known and previously inspected episodes, and prevent them from masquerading as live signals or untouched data. If the forward record has no qualifying adverse regime or insufficient coverage, defensive **evidence remains INCONCLUSIVE** even if the book satisfies an operational recordkeeping check. Never extend the registered endpoint opportunistically to find a favorable episode.

A defensive stop is an action/risk control, not a statistical verdict. Register relative and absolute limits separately; a relative-wealth stop does not cap absolute losses or guarantee execution. Continue to preserve the stopped outcome and its subsequent valuation according to the brief.

## 4. Whole-account, tax and implementation checks

At each stage report sleeve and whole-account outcomes when inputs support them; otherwise mark each unavailable with the missing inputs and do not fabricate a result. These comparisons are supplemental reporting, not new advancement prerequisites for the existing Part A/B0/B1 standard. Any mandatory new gate requires an approved prospective amendment. Where feasible, compare the candidate against (1) the actual allocation it would displace, where its starting state can be established; (2) the registered fixed passive policy; and (3) SPY. Label an exposure-matched or factor diagnostic separately. A small sleeve can have a meaningful measured edge but a small dollar contribution; an underpowered whole-account comparison cannot prove the sleeve has none.

For each feasible comparison use identical external cash flows and clearly specified starting holdings, basis, lots and cash—or a standardized hypothetical starting state. Report pretax results, estimated after-tax ongoing wealth, and after-tax terminal-liquidation wealth separately. The lot method, dividend, wash-sale, loss carryforward, cost, tax-payment funding and liquidation conventions must be fixed and their limitations disclosed. Resampling a path-dependent tax ledger requires re-running lot accounting on each synthetic path or clearly stating why an alternative is valid; resampling already-taxed daily returns does not automatically do so.

Before a separately authorized capital decision, apply all then-current evidence and operational gates and compare against the allocation displaced. Model estimates are not verified tax advice. No existing six-month paper requirement or other capital prerequisite is relaxed by this draft.

## 5. Evidence discipline and prior studies

A brief records whether data has already been used in exploration; once inspected, that period is not an untouched holdout. Freeze dates, candidate family, primary endpoint, comparator, costs, tax, thresholds, missing-data treatment, inference, sequential review and multiple-testing plan. Estimate candidate-specific power or minimum detectable effect and label an underpowered test accordingly; no universal minimum years or crisis count is asserted. Overlapping folds and correlated trades are not independent observations. A trial ledger records attempted variations and failed or stopped versions; a correction within one registered family does not automatically control the error rate across the entire programme.

Retesting an inconclusive or failed version needs a new dated brief identifying what is materially different, what data were already inspected, and why the new evidence is informative. New data or better measurement may be sufficient when justified; merely selecting a favorable endpoint, extending a failing window or changing benchmarks after results is not. The owner may choose not to spend further effort without claiming a hypothesis was disproved.

Historical NO-GO, GO, stopped and unresolved records stay append-only with their original scope and policy version. The required retrospective review in §5.1 adds evidence/action annotations that reference the original and distinguish policy failure from inferential power. No automatic reclassification, rerun or old study's revival follows from this draft.

### 5.1 Required retrospective review

Review past verdicts after the owner approves this design and a separate review plan, without running new return tests. Start from the [root-cause investigation](../../../research/audits/root-cause-2026-11-04/findings/2026-11-04-root-cause-investigation.md) and its verdict-audit artifacts, then reconcile the inventory against the [research snapshot](../../../research/RESEARCH_SNAPSHOT.md) and original briefs and reports. Record the inventory's cutoff date, source versions, and coverage, including excluded status-only records and missing artifacts; do not imply the audit covered every later study.

Triage the highest-value cases first: the historical production-rule replica, currently relevant stock-signal closures and defensive studies, then remaining verdicts in bounded batches. The review plan must define a complete inventory, batch order, progress ledger and independent verification before the first batch. A provisional annotation may be recorded from existing evidence. Missing power analysis is recorded as power unknown and can leave a valid measured result INCONCLUSIVE; it does not by itself invalidate the observed result or change a valid protocol outcome. Reserve UNRESOLVED for a missing brief, data, code, or other fault that prevents the particular inference or protocol determination; state exactly what cannot be checked.

**One canonical verdict register:** create `research/VERDICT_REGISTER.md` as the single maintained entry point for every identified past and future project research verdict. It is a navigable index, not a replacement for the underlying registered briefs, original verdicts, raw evidence or the research snapshot. The register links each row to its original source and dated annotation; the underlying records remain authoritative for their own history. The existing generated research snapshot stays the roster of strategies and study status and must not be hand-edited; link the register from the snapshot through its supported generation workflow or, if that workflow is unavailable, from `research/README.md` and `research/RESEARCH_INDEX.md` until it can be updated safely. Avoid maintaining a second competing evidence/action verdict table.

The register contains one row per distinct tested specification and verdict, with separate columns **strategy / study**, **original verdict**, **registered protocol result**, **evidence**, **action**, and **source / annotation**. Each row identifies its study, specification, attempt or stage, and verdict; dated annotation revisions have their own identifiers and link to the verdict they annotate. This preserves multiple stages, stopped attempts and revisions for one specification. Each row shows the date and scope of its annotation and links to the fuller dated ledger for benchmark, window, uncertainty, provenance, rationale and stop conditions. Preserve multiple distinct variants and dated verdicts; never merge them into one strategy-wide judgment. Untested ideas, components and operational or status-only items are explicitly identified or linked separately, never counted as verdicts. Publish the inventory cutoff, total identified/included/excluded/pending, reasons for exclusions, review progress and a visible incomplete marker until reconciliation is finished. Sources include the historical audit, snapshot, original records and later studies; do not pretend one audit covers all later work.

**Update rule:** whenever a new research verdict, approved amendment or new evidence changes an action or annotation, add a dated, source-linked entry or append-only revision to the register as part of that work. Keep the original result visible; a correction adds history and cites its authority. The source study remains authoritative if the index and source disagree; mark the index entry unresolved and correct the index against the source. A test or reconciliation check in the later implementation plan must catch missing links, duplicate record-level verdict or annotation-revision identifiers, broken study/specification/attempt relationships, and unaccounted inventory rows; repeated study or specification identifiers across distinct stages and revisions are valid; it must not invent a verdict where human judgment or evidence is missing. The detailed ledger and register may be generated together from a single maintained dataset, provided the original studies remain independent records and the generation/verification process is specified before implementation.

For each reviewed result record: study and original question; vintage, benchmark, dates and original policy; original verdict and its source; registered protocol result; evidence label with the applicable materiality criterion, uncertainty and data quality; action label with rationale and any condition for future work; and links to existing evidence and the audit. Existing results may receive a retrospective **DESCRIPTIVE** annotation where new criteria were chosen after the data were seen. Do not retroactively apply a newly chosen threshold as if it had been preregistered. Preserve the replica's distinct model NO-GO, holdout INCONCLUSIVE, and live parity UNVERIFIED.

Annotations are new, dated records: never overwrite historical verdicts, alter raw artifacts, claim data are stronger than they are, automatically reopen a closed study, or treat a policy failure as proof of no edge. `CONTINUE RESEARCH` records an already authorized scope and approval reference, not authorization to rerun; an unapproved possible next question accompanies `HOLD` instead. Any materially different candidate or new evidence requires its own dated registration and applicable approval. The owner retains control of priorities; no retirement, shutdown, account-type change or production action follows.

## 6. Standard transition and resolution of ambiguities

The existing approved return standard is unchanged while this document is a draft. Implementing the new design requires a **separate dated prospective amendment** that names precisely which studies and stages the defense path governs, how its historical and forward gates work, what remains from Part A/B0/B1, its stop policy and operational prerequisites. Past candidates retain their original brief unless an explicit authorized amendment says otherwise. New statistical intervals cannot retroactively transform B1's policy pass or fail into evidence of an edge.

The 2026-10-23 document has stale draft/unapproved passages alongside its approved header and R1–R5 resolution. A future amendment should identify the operative approval rather than quietly rewriting history. Its formula says `< −0.15`, but approved R5 says **15% or more behind**: the operative inclusive relation is `relative_wealth − 1 ≤ −0.15`; preserve the original and record the correction. The B0 −1 pp threshold is a noisy screening policy, not proof that a candidate is a clear loser. Changes to any governing gates need explicit owner approval.

## 7. Registration templates and acceptance cases

Every candidate brief must provide the fields in §1 and §5 and the route-specific fields below. Missing fields are unresolved, not silently defaulted.

**Return template:** define return/wealth estimand; materiality and confidence or other inference rule; SPY/appropriate comparator; gross and net conventions; sleeve and whole-account capital bases; power check; stage-specific gates and operational stops.

**Defense template:** define the downside estimand and meaningful improvement; wealth noninferiority margin and inference; chosen fixed passive comparator plus the SPY and risk-match diagnostics; stress coverage; risk and opportunity-cost limits; stage-specific gates and operational stops.

Acceptance examples for a later implementation plan:

1. A valid underpowered result below its registered return threshold records `INCONCLUSIVE / NOT MET / CLOSE or HOLD`, with the action dictated by the applicable brief; it does not assert absence of an edge.
2. An invalid price history records `UNRESOLVED / UNRESOLVED / HOLD`, not a pass or valid strategy failure.
3. A defensive candidate that holds cash and improves only against SPY, but not against its registered fixed alternative, cannot pass the defensive benefit gate.
4. A low-volatility candidate with statistically unresolved wealth sacrifice cannot satisfy a registered noninferiority requirement by failing to reject harm.
5. A forward window without the registered stress event records defensive evidence `INCONCLUSIVE`; its operational status is reported independently.
6. Two paths chosen after results cannot be combined into a pass. A preregistered joint claim or alternative-path rule must disclose and control its multiple-testing implications.
7. The canonical `research/VERDICT_REGISTER.md` tracks every identified historical verdict or explicit exclusion, includes separate evidence and action columns, discloses incomplete coverage, and receives future dated updates. The old replica keeps its original model NO-GO, holdout inconclusive, and live parity unverified; its dated annotation is additive and no return rerun occurs.
8. Two dated annotations of the same verdict have distinct revision identifiers and preserve their shared study, specification, stage and verdict relationships; neither overwrites the original.
9. The exact −15% relative-wealth boundary is inclusive under R5; a stop is an action, not a statement of statistical evidence.

## 8. Owner decisions before any amendment or candidate brief

- Relevant horizon and acceptable drawdown for each objective.
- Acceptable loss of after-tax wealth or annual return versus the chosen passive alternative for a meaningful defensive benefit.
- The primary defensive downside measure and the passive alternative appropriate for that study; whether and how hindsight-classified historical regimes may support inference.
- Evidence/materiality thresholds and operational/forward stop tolerances, informed by a candidate-specific power and feasibility analysis; no numbers are asserted here.
- Which stages, if any, a defensive candidate may enter under a future amendment, and whether the same minimum shadow period and paper mechanics requirements apply.

This draft does not prescribe an allocation, move capital, run a return test, or authorize a new rule. The owner reviews the design before an implementation plan or prospective amendment is written.
