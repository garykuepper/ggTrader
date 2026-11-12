# Cross-asset shadow: owner decision brief

**2026-11-10 — Proposal; no authorization.** The governing shadow rule, qualifying observations, start and any reset remain **UNRESOLVED**. This brief does not start a shadow, change the production freeze, authorize broker paper, or authorize capital. Select and document the rules prospectively; preserve the history even if a new attempt is chosen.

**2026-11-11 scope decision:** The owner selected the original 80/20 cross-asset
shadow for the future paper-parity criterion. This identifies the book only; it
does not choose the detailed governing rules, start or reset a window, approve a study, or
authorize paper trading.

**2026-11-11 schedule choice:** The owner selected a draft with a three-month
mechanics review followed by a 12-month performance review. This is separate from
the proposed 2027–2029 loss-protection comparison. Endpoints, performance measures,
reset handling and the other governing details remain to be specified and reviewed;
this choice does not start an observation or authorize trading.

## What the evidence establishes

The [original report](../../allocation/cross-asset-sleeve/2026-09-28-cross-asset-sleeve.md) records a static 80/20 GO **for shadow eligibility**, not deployment. Its frozen 2007–2011 holdout passed, but the report calls that pass weak and notes the crisis-dominated window; the principal historical windows had already been inspected. The trend arm was rejected. The [register's shadow-stage annotation](annotations.json) (`discovery-review-fe12b892259fd8fc`) records **UNRESOLVED / HOLD / PARK THIS VERSION**, with no structured approval captured; it records no shadow-stage verdict; it does not undo the static GO or weak holdout pass. A short operational record would not by itself establish a durable edge. The historical second read reported that, when cash earned the T-bill rate, dropping GLD or PDBC failed its leg-removal comparison; it also reported that 80% SPY / 20% BIL reduced drawdown about as much as the sleeve and that 80/20 SPY/GLD beat the three-leg book, while cautioning against switching legs on that seen-data diagnostic and characterized the sleeve as an allocation choice, not a validated edge. These are historical reported diagnostics; no committed artifact for those figures was located.

## Rules that need reconciliation

- The September report proposes approximately three months of shadowing before a decision, with 80% SPY and the balance equally in TLT, GLD (or IAU) and PDBC, rebalanced at each month-end close. Its suggested comparisons are the shadow, SPY and core plus SPY sweep. This is a recommendation in a report, not evidence of owner approval.
- The [archived decision plan](../../../docs/archive/next-steps-history-2026-11-03.md#real-money-plan-and-decision-rules-written-2026-09-28-before-any-shadow-data) instead specifies months 0–3 as parity-only, any parity incident resetting the clock, and a month-12 check requiring T-bill-excess Sharpe at least SPY's and drawdown no worse. Its drawdown definition and comparison implementation need clarification; the owner has since selected the cross-asset shadow for future parity. The plan labels itself written before shadow data and reports an independent second read; that label is not independent proof of timing or an explicit owner approval of this rule.
- The same plan records the owner's taxable-only decision and historically expected the sleeve was unlikely to pass after tax; that is an expectation, not a measured forward outcome. It calls for an after-tax comparison with short/long-term gains, gold tax character and **drift-band**, rather than monthly, rebalancing. It does not specify the cross-asset band here. The band was described for the after-tax funding comparison, not necessarily as a replacement for the monthly shadow. A changed shadow rule requires a separate dated prospective owner-approved amendment, with the original attempt preserved.
- The plan records the owner's later A23 adoption of 70/30 SPY/SSO and its own band and review terms. It does not resolve how the cross-asset sleeve would be funded or interact with that allocation. Do not assume it replaces, modifies or borrows the SSO leg. The [current worklist](../../../docs/next_steps.md) retains the freeze and calls for parity and start/reset decisions.

The [2026-10-23 standard](../evidence-standard/2026-10-23-forward-paper-standard.md) attests owner approval for Part B replacement for **future stock candidates**. Its B1 requires at least 18 months and a specific prospective ledger; it is not automatically retroactive to this allocation sleeve. Its opening status says approved, while its authority paragraph still describes a draft; that internal wording conflict does not establish an amendment for this sleeve. Preserve Part A/B0/B1 for their existing scope.

The owner chose to prepare a three-month operational review followed by a 12-month performance review. This schedule choice does not approve the archived performance thresholds or settle the conflicting documents. The current tracker calculates return and maximum drawdown and overwrites its CSV; it does not implement the archived T-bill-excess Sharpe comparison or maintain a contemporaneous decision ledger. A prospective amendment would need to resolve those gaps, define the comparator and clock, and retain the previous attempt as unresolved. A short performance check is a screen, not proof of an edge.

## Draft review clock and evidence

The following is a proposal for review, not an approved rule. It makes the two
reviews concrete without choosing a start date or resolving the historical rule:

1. **Establish a prospective start.** Before any clock starts, record the dated
   rule, the starting hypothetical portfolio and comparator, and the evidence that
   all start conditions are met. AVB remains a blocker until resolved. A later
   reconstruction cannot establish that an earlier start was clean.
2. **Mechanics review at three calendar months.** Propose measuring from the
   qualifying start date to its three-month anniversary, using the last day of the
   target month if that day number does not exist. If the anniversary is not an
   exchange session, use the next session's close to complete the review. For
   example, a hypothetical January 31 start would point to April 30; a closed
   anniversary moves the review to the next session. These are examples, not
   scheduled dates. Reconcile each required session and rebalance against the
   contemporaneous rule, inputs, targets, fills and exceptions. Record missing or
   late evidence as unknown; do not count it clean just because a report exists.
3. **Performance review at 12 calendar months.** Propose measuring from the same
   qualifying start, not adding 12 months after the mechanics review. Apply the
   same anniversary and non-session convention. Compare the shadow and SPY using
   consistent starting value, dates, external flows, distributions, fees and
   valuation; document the cash-rate source and drawdown definition before a run.
   Report elapsed coverage, missing valuations, return and drawdown separately.
   Sharpe and any threshold remain unapproved; insufficient or unreliable evidence
   means no performance conclusion, not an automatic pass or fail.
4. **Incidents and resets.** Preserve the initial start and every incident in a
   dated ledger. The existing paper-parity rule treats a parity incident as a
   reset; whether that also resets the shadow's performance clock, and how a
   missing record affects either clock, remain owner decisions. Report the
   affected interval as unresolved until adjudicated. Do not silently backfill,
   erase the earlier attempt, or infer that a mechanics review authorizes trading.

Before use, the amendment also needs to define the start valuation and execution
convention, what happens if an endpoint price is unavailable, the evidence store
and who performs the review. A later review date does not turn an unverified
session into a qualifying one.

## Observation and timing limits

The [readiness reconciliation](../../paper-system/observation-readiness/2026-11-02-paper-freeze-readiness.md) says the shadow CSV was recomputed and overwritten, with rows only through 2026-10-16 at that review; it is not a contemporaneous decision archive. Shadow inputs and per-session qualification remain unknown. The owner selected the cross-asset book for future parity and kept the persistent AVB condition blocking a clean start until resolved; the qualifying start and reset remain open. Unknown is not clean; neither a calendar date nor a recomputation proves a clean window. The tracker defaults to a 2026-09-29 start, but that date does not establish a qualifying start or a completed clean window. The owner stopped paper-trader cron on 2026-11-06; neither the proposed sleeve nor the A23 allocation is established here as current paper exposure. No private account values were inspected for this brief.

Git history corroborates document existence and ordering, not decision-time observations or approval: `9a79115a` added the tracker at 20:29 PT on September 28; `014dfedc` recorded caveats at 21:15; `d3da809e` added the parity/12-month plan at 21:28; `291e6dd4` recorded taxable-only at 21:35. These are commit timestamps, not an independent timestamp witness for when data were first seen or a record of an owner decision on shadow rules. The tracker's `to_csv` overwrites its output. The tracker’s simulation path was replaced on 2026-11-06 (`27fff069`); its code comment claims bit identity with the old simulator, which this brief has not verified. The tracker rebases to live NAV; a standardized hypothetical starting state avoids relying on private account data. Previously inspected data must be treated as seen.

## Decisions for the owner

The owner chose to prepare the three-month mechanics and 12-month performance schedule. Its precise rules require a dated prospective amendment and review; the choice does not retroactively qualify an earlier period. Resolve the rule and evidence first, without assigning a start date.

1. **Governing rule:** adjudicate the historical conflict and its authority using any contemporaneous approval evidence, or choose a new prospective rule in a dated amendment. If choosing a different duration, performance gate, or parity/reset rule, specify the exact endpoint, equality, drawdown definition and treatment of missing observations. Do not rewrite or retrospectively pass the earlier attempt.
2. **Comparison:** retain SPY and decide whether to include the historical core-plus-sweep comparison and a separately defined A23/displaced-allocation counterfactual. Specify starting holdings, external flows, valuation, cash and timing consistently. These are proposed comparisons selected after historical data were seen, not preregistered evidence.
3. **Observation and ledger:** decide who records each session and scheduled rebalance, and where immutable, timestamp-witnessed rule version, decision inputs, targets, prices, hypothetical fills, exceptions and corrections are retained. Define missing or late records as unknown and how they are resolved; a hash alone does not prove timing. A shadow ledger does not prove broker parity. Any tool or schedule needs separate authorization and review.
4. **Start and reset:** the owner has selected the original 80/20 cross-asset shadow for future parity and kept AVB blocking a clean start until resolved. Decide which incidents require resets, address missing evidence, and specify how a qualifying prospective start and any reset are witnessed. Start date: **[owner decision pending]**. Reset treatment: **[owner decision pending]**. Preserve prior unknowns and any stopped attempt; do not backfill a clean clock.
5. **Taxable accounting and rebalance:** choose whether the prospective hypothetical shadow keeps monthly rebalancing, changes to a specified drift band, or stays monthly for parity while a separate after-tax comparison models a band. Fix band/tolerance **[pending]**, decision and execution timing, costs, starting basis and lots (or standardized hypothetical state), lot selection, distributions and gold tax character assumptions, wash sales, loss carryforwards, tax-payment funding and liquidation. Report pretax, estimated ongoing after-tax and terminal-liquidation wealth separately, with missing inputs and limitations. A band or other changed shadow rule requires the dated prospective amendment above; tax modeling does not establish actual tax treatment.

No start, reset or review dates, numerical tolerances, after-tax results or approval are supplied by this proposal. Independent review of this brief and the owner's dated decisions are still required before presenting a final governing rule; neither alone authorizes a run or a production change.

The historical 12-month plan was reviewed by Fable, a Claude model, and was not cross-family independent from the Claude-authored plan. The leg-removal and BIL diagnostics were credited separately to research-report-review skill tests. The archived section was subsequently edited; Git commit `d3da809e` retains the original 12-month rule wording. The holdout used DBC as a proxy for PDBC.
