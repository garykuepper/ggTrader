# Research index

**Reviewed:** 2026-11-01. This is a navigation and classification aid, not a new
validation or a change to any research or production decision. For the current
strategy roster and qualifications use [RESEARCH_SNAPSHOT.md](RESEARCH_SNAPSHOT.md);
for current decisions use [next steps](../next_steps.md). Study reports and their
registered amendments govern their own scope. The snapshot's last full regeneration
was 2026-10-19 and its last incremental update was 2026-10-23; later work is
included below where identified.

For the process review and evidence limitations, see the [2026-11-01 research process audit](2026-11-01-research-process-audit.md) and its [claim ledger](_research_process_audit_ledger.json).

## How to read the classifications

- **NO-GO**: a specified test failed its registered or documented standard. It
  applies to that test and its data, not automatically to every related idea.
- **Historical NO-GO**: a negative historical result with material tape, timing,
  universe or other limitations; it is not a clean modern revalidation.
- **Inconclusive / unresolved**: the evidence cannot answer the specified question.
  A data-feasibility audit is not a strategy return test.
- **Conditional / allocation decision**: an outcome applies only to its stated
  conditions or an owner allocation decision. It is not proof of stock alpha.
- **Methods / operations**: research about data, measurement, implementation or
  paper-trading reliability, not a strategy performance verdict.
- **Candidate / feasibility**: literature, data or a proposed protocol; no
  project strategy GO follows without the required tests.

Historical metrics across different windows are not directly comparable. Earlier
Deflated Sharpe Ratio gate counts are not valid DSR evidence. Point-in-time
membership does not certify security identity or the whole price tape. Where a
later audit corrected or limited an older report, both are linked below.

## Strategy and allocation research

| Family | Work performed | Classification and scope | Primary record |
|---|---|---|---|
| Stock selection / production model | Fixed-default ensemble plus SPY sweep | **NO-GO** on the primary daily-bar model: 14.19% CAGR vs SPY 14.83%, excess −0.64 percentage points/year, Newey–West t −0.69. Historical live parity is unverified; stopped holdout performance is non-citable. | [Production-rule audit](2026-10-19-production-replica-audit-nogo.md), [results](_production_replica_results.json), [parity](_replica_parity_20261019.json) |
| Stock selection / historical core | PIT core, IC and Kelly variants | **NO-GO** on the historical PIT rebaseline; this WFO construction is not the live fixed-default portfolio. The preceding non-PIT rebaseline is invalid. | [PIT rebaseline](2026-09-23-pit-rebaseline-core-nogo.md), [invalid prior run](2026-09-23-ic-kelly-rebaseline-invalid.md), [results](_pit_rebaseline_20260923.json) |
| Stock selection / historical blend | Selected-parameter core plus SPY | **Historical KEEP, superseded for the fixed-rule model**; do not apply its conclusion to production sizing or live parity. | [Blend scope correction](2026-09-24-core-spy-blend-keep.md), [later audit](2026-10-19-production-replica-audit-nogo.md) |
| Momentum | Cross-sectional and dual momentum; French long-history screen | **NO-GO** on the registered idea-level screen; both sizes fail the multiple-testing and modern-return bars. The two implementations were not independent tests on the corrected replay. | [Momentum screen](2026-10-20-momentum-french-screen-nogo.md), [results](_momentum_french_screen_results.json) |
| Stock factors | Profitability, value, investment, value-plus-momentum | **NO-GO**: all four fail the registered multiple-testing and post-1994 economic bars. | [Factor screen](2026-10-21-factor-screen-nogo.md), [results](_factor_screen_results.json) |
| Stock factors | Net issuance | **NO-GO** on the long-only screen; weakness of heavy issuers is a short-side diagnostic, not a long-only validation. | [Net-issuance report](2026-10-23-net-issuance-screen-nogo.md), [results](_net_issuance_screen_results.json) |
| Stock anomalies | Index deletion, MAX / lottery demand, pairs mean reversion | **Historical NO-GO**, except pairs also has a corrected-tape NO-GO rerun. Historical data limitations remain relevant to the others. | [Deletion](2026-07-17-index-deletion-fade-nogo.md), [MAX](2026-07-17-max-effect-nogo.md), [pairs](2026-07-17-pairs-stat-arb-nogo.md) |
| Stock events | PEAD, insider cluster buying | **Historical NO-GO; current data feasibility unresolved and studies parked.** Do not turn an unresolved source audit into a fresh strategy verdict. | [PEAD](2026-07-17-pead-nogo.md), [insider](2026-07-19-insider-cluster-buy-nogo.md), [event decision](2026-10-23-event-timing-audit-decision.md), [current owner decision](../next_steps.md) |
| Stock information / sentiment | Congressional trades, short interest, short-volume ratio, headline sentiment | **Historical NO-GO** with tape caveats for the first three; headline sentiment is **provisional NO-GO**, based on only three usable folds. | [Congress](2026-07-19-congress-trades-nogo.md), [short interest](2026-07-17-short-interest-nogo.md), [short volume](2026-07-19-short-volume-ratio-nogo.md), [headlines](2026-07-24-headline-sentiment-nogo.md) |
| Bonds / calendar | FOMC drift, month-end Treasury sleeve | **NO-GO**: corrected FOMC evidence fails net of costs; the month-end sleeve fails against its SPY funding leg. | [Original FOMC report](2026-07-20-fomc-drift-nogo.md), [corrected clustered results](_fomc_clustered_event_study_20261020.json), [month-end report](2026-09-24-month-end-treasury-sleeve-nogo.md) |
| Macro / diversifiers | Commodity trend, Treasury curve, dynamic FX hedge | **Historical NO-GO** against their documented static or other comparators. | [Commodity](2026-07-20-commodity-trend-nogo.md), [Treasury curve](2026-07-20-treasury-curve-nogo.md), [FX](2026-07-20-fx-hedge-overlay-nogo.md) |
| Cross-asset allocation | Static 80/20 SPY plus diversified sleeve; trend variant | **Conditional shadow-only GO** for the static allocation under its own bar; trend adds nothing. Pinned CAGR 13.61% vs SPY 14.70%; Sharpe 0.973 vs 0.896. This is diversification, not validated stock-selection alpha. | [Cross-asset report](2026-09-28-cross-asset-sleeve.md), [primary results](_cross_asset_sleeve_results.json), [holdout](_cross_asset_sleeve_holdout_results.json) |
| Leveraged ETF timing | Breadth rotation and long-only trend across three indices | **NO-GO**: rotation experienced severe drawdowns and persistent defensive halts; trend lost to buy-and-hold of its own leveraged ETF. | [Rotation](2026-07-16-leveraged-index-rotation-nogo.md), [trend](2026-07-16-leveraged-trend-following-nogo.md) |
| Leveraged allocation | Static SPY/SSO A23; A24 timing rider | **Owner allocation decision** for A23, not a Sharpe-based strategy GO; higher-funding after-tax margin falls below its original bar. **A24 NO-GO** as an after-tax return improvement. | [A23/A24](2026-09-30-static-fractional-leverage.md), [results](_static_leverage_aftertax_results.json), [funding sensitivity](_static_leverage_aftertax_rf_plus0.025.json), [distribution bounds](_static_leverage_aftertax_dist_bounds.json) |
| Leveraged allocation / stress | Century synthetic stress of A23 | **Registered rule keeps 30%, with a ruin flag**: full-window 30% drawdown −91.4% vs −84.6% at 0%; 15% avoids the flag at a 0.54 percentage-point annual after-tax CAGR cost vs 30%. Proposal for the scheduled owner review, not a new strategy GO. | [Century study](2026-10-23-century-leverage.md), [results](_century_leverage_results.json), [current decision](../next_steps.md) |
| Ensemble levers | ML entry filter, exit tuning, weighted voting, extra voters and related variants | **Historical NO-GO** or insufficient independent modern evidence, depending on the lever. Components and registry membership do not establish independent edges. | [Research snapshot roster](RESEARCH_SNAPSHOT.md), [historical optimization plan](2026-06-28-alpha-optimization-execution-plan.md), [strategy audit](2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md) |
| Position sizing | Regime and sizing comparisons | **Inconclusive** for generalization: single ungated window and a mismatch between live and historical sizing. | [Sizing report](2026-08-19-position-sizing-regimes.md), [results](_position_sizing_wfo_results.json) |

## Data, methods and paper-trading reliability

| Topic | Classification and finding | Records |
|---|---|---|
| WFO anchor leakage and reproducibility | **Methods:** the anchor leak was fixed; the old 1.12 baseline does not reproduce, and its original drifting window is unrecoverable. | [Anchor fix](2026-08-18-wfo-anchor-leakage-fix.md), [reproduction](2026-08-19-anchor-fix-reproduction.md), [results](_anchor_fix_reproduction_results.json) |
| Strategy and harness audit | **Methods:** reviews historical implementation defects and evidence limitations. Later corrections and decisions supersede old retry recommendations. | [Implementation audit](2026-07-25-strategy-implementation-audit.md), [comprehensive audit](2026-09-10-comprehensive-strategy-audit-and-retry-recommendations.md), [production audit](2026-10-19-production-replica-audit-nogo.md) |
| Event data and identity | **Unresolved / parked:** coverage, historical identity, timing, inventories and lineage remain incomplete. The canonical audit artifact has not met its publication guard. | [Readiness decision](2026-10-22-event-audit-readiness-decision.md), [resolution workbook](2026-10-22-event-evidence-resolution-workbook.md), [decision memo](2026-10-23-event-timing-audit-decision.md), [implementation record](event-timing-audit-implementation.md), [source ledger](_event_source_search_ledger.json) |
| Paper signal warm-up | **Operations:** measured indicator-window mismatch; 365-day warm-up removed the measured drift. | [Warm-up study](2026-09-28-live-indicator-warmup.md), [results](_live_indicator_warmup_results.json) |
| Crypto execution venues | **Feasibility:** historical venue comparison, not a tested crypto strategy or a current price quote. | [Venue comparison](2026-07-06-crypto-execution-venues.md) |
| Intraday sources and preflight | **Candidate / feasibility:** source and replication review plus protocol design, not a project return test. The Stage 0A contract is separately documented; no strategy edge follows. | [Feasibility review](2026-10-24-intraday-trading-feasibility.md), [source audit](2026-10-24-intraday-replication-source-audit.md), [preflight](2026-10-25-intraday-preflight-delegation.md), [Stage 0A contract](briefs/2026-10-25-intraday-stage0a-contract.md) |

## Process follow-ups (2026-11-02)

Documentary and diagnostic follow-ups to the [process audit](2026-11-01-research-process-audit.md). None changes a verdict, the production freeze or the parked event studies, and none ran a market-return test.

| Topic | Classification and finding | Records |
|---|---|---|
| WFO fold attribution | **Methods:** saved fold records split into gate failure, anchor deployed with gates passed, and winner deployed. `xs_momentum`, `dual_momentum` and `fomc_drift` replays were anchor-dominated (100%, 100%, 89%); pre- and post-DSR-fix runs are kept separate. | [Attribution](2026-11-02-wfo-fold-attribution.md), [results](_wfo_fold_attribution_results.json) |
| WFO fallback and halt semantics | **Methods:** current behavior is specified and pinned by characterization tests; opt-in `WfoPolicy` options (NaN WFE exclusion, anchor-fold WFE excluded from the breaker history) and additive `halted_entering` / `skipped_folds` reporting are implemented with defaults unchanged. Owner adoption of any alternative is pending. | [Semantics](2026-11-02-wfo-fallback-semantics.md) |
| Free-data stock feasibility | **Feasibility inventory:** no candidate recommended, no return computed; includes an analytic power note for Part A. | [Inventory](2026-11-02-free-data-stock-feasibility.md), [matrix](_free_data_feasibility_matrix.json) |
| After-tax multi-stock lot engine | **Library implemented, unvalidated:** `TaxLotBook` (HIFO, wash sales over ±30 days, character-preserving carryforwards, hash-chained ledger) with 46 tests; study-level B0/B1 controls (decision receipts, outside-holdings reconciliation) are not implemented. The original gap analysis is kept as a dated snapshot. | [Requirements and status](2026-11-02-after-tax-lot-engine-requirements.md) |
| Synthetic WFO policy study | **Preregistered synthetic study** (36 cells, 5000 reps): under the stated assumptions the legacy circuit breaker halts at least once in 87.5% of runs for a true-Sharpe-1.0 strategy; plausibly driven (by calculation, an inference) by the two-consecutive-negative-Sharpe rule. Not evidence about any real strategy or the real gates. | [Study](2026-11-02-wfo-policy-synthetic.md), [results](_wfo_policy_synthetic_results.json) |
| Family-scope review | **Documentary, definitions only:** of eight pending feasibility rows, S04 (accruals) is `distinct_mechanism`; seven are `undetermined`; none is judged covered by a closed NO-GO. No recorded verdict changed; no candidate recommended. | [Review](2026-11-02-family-scope-review.md), [extracted definitions](_family_scope_extraction.json) |

## Candidate literature and supporting artifacts

The [web research candidate register](WEB_RESEARCH_CANDIDATES.md) is a backlog and
source-triage record. It contains untested, parked, infeasible and closed ideas;
a paper's reported result is not ggTrader out-of-sample evidence. Its batches
cover equity/event anomalies, commodities and FX, Treasury/calendar effects,
crypto, exposure overlays and leveraged ETFs. Statuses in older batch summaries
can be superseded by the reports and current worklist above.

The October 25 crypto batch includes an [IBIT overnight sanity-check script](artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.py)
and [results](artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.json).
It is a descriptive check, not a walk-forward test or evidence of an executable
edge. The register records its caveats and later reversal.

Registered protocols and amendments live in [briefs](briefs/); JSON and CSV
files alongside the reports preserve many, but not necessarily all, raw study
results. Lab run records also live in TimescaleDB, outside Git. Some supporting
event-source cache material is local and untracked; do not infer that a fresh
clone contains every raw input. For report-writing conventions see the
[research template](TEMPLATE-research-report.md).
