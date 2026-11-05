# Research process audit

**As of:** 2026-11-01. **Status: scoped documentary audit; unresolved evidence
listed below.** This is not a new trading verdict or authorization. Original
verdicts, the production freeze and the owner's parked-event decision remain in
force. The accompanying [claim ledger](_research_process_audit_ledger.json)
distinguishes historical decisions from preliminary assessments. Most roster
entries received a documented-source and current-class triage, not an independent
reconstruction of their historical data and exact run code.
No backtests, database queries, authenticated data requests or live/paper
orders were performed for this report.

## Executive assessment

The project has legitimate negative results, but a count of NO-GOs cannot tell
us how many independent ideas failed. The 38 registered names include voters,
variants, multiple universes for the same mechanism, untested components and
strategies assessed with shared machinery. Likewise, a parameter fallback,
external factor proxy, fixed-rule portfolio and a data-feasibility audit answer
different questions. The research program's central vulnerability is drawing
an interpretation broader than the strategy, data and decision rule tested.

The counterargument is equally important: a strategy can fail an owner-approved
research or economic gate without proving its true edge is exactly zero. An
inconclusive statistical test is not a positive result, and strategy redesign
on already-inspected history creates another selection opportunity. Several
negative findings below remain meaningful within their documented scope.

### Verified concerns and their limits

1. **Walk-forward tests can principally measure fallback.** Failed gates or
   a halt deploy anchor parameters, not necessarily cash ([wfo.py](../../src/ggTrader/lab/wfo.py)).
   The corrected momentum replay used anchors in 17/17 folds; its DSR gate
   required roughly 2.5 in-sample Sharpe in the documented setup. Thus that
   result evaluates the complete gate/fallback policy, not independently a
   successful adaptive-optimization procedure. This does not establish the
   ungated strategy succeeds. The separately registered French screen bypasses
   that WFO gate and also fails its bars. [Audit correction](2026-10-19-production-replica-audit-nogo.md#correction-2026-10-20).
2. **Pairs tested a specific simplified construction.** The report documents
   monthly same-sector, 1:1 spread selection; the implementation applies ranking
   and capacity/shared-symbol exclusions after adding held pairs, so its stated
   minimum holding floor is not an unconditional guarantee. The code cache key
   omits data identity, and the report documents an A/B comparison polluted by
   a cached result. A corrected rerun remained negative, but its unexpectedly
   unchanged OOS results are an acknowledged open anomaly. These observations do
   not establish that a daily or hedged alternative is profitable. [Code](../../src/ggTrader/lab/strategies/pairs_stat_arb.py),
   [report §§6–7](2026-07-17-pairs-stat-arb-nogo.md).
3. **Long-history screens use proxies and assumed costs.** The French momentum
   screen's large/small value-weighted portfolios are not the project's exact
   equal-weight S&P 500 top-N rules. Big and Small failed the registered Holm
   screen and the +1 pt/year modern economic bar after stated cost haircuts; the
   actual reported post-1994 excesses are +0.08 and +0.20 pt/year. These are
   meaningful negative screen results; no tax-lot backtest of the precise stock
   implementation follows from them. Haircuts are policy assumptions, not
   measured account-specific execution costs. [Momentum report](2026-10-20-momentum-french-screen-nogo.md),
   [factor report](2026-10-21-factor-screen-nogo.md).
4. **The production replica answers a portfolio-model question.** Primary
   2021-01-31–2026-04-30 CAGR was 14.19% vs SPY 14.83%, excess −0.64
   percentage point/year, NW t −0.69. Mean stock exposure was 9.37%; the
   exposure-matched increment was +0.265 point/year, t 0.202. The registered
   fixed-rule model NO-GO stands; neither number proves a small selection effect
   impossible. The older holdout stopped for price/identity artifacts and is
   non-citable; exact live parity remains unverified, with 10/32 orders explained
   under a denominator including 19 manual liquidations. [Replica audit](2026-10-19-production-replica-audit-nogo.md),
   [results](_production_replica_results.json), [parity](_replica_parity_20261019.json).
5. **Event observations must use the independent event as the unit.** The FOMC
   gross/net t statistics 3.619/1.572 used three ETF observations per date; the
   corrected 113-date clustered gross/net statistics are 2.258/0.981. The net
   result fails the stated evidence bar; that does not prove gross alpha is zero.
   [Corrected artifact](_fomc_clustered_event_study_20261020.json),
   [audit correction](2026-10-19-production-replica-audit-nogo.md#correction-2026-10-20).
6. **Crypto annualization is a traceable risk, not yet a demonstrated bad
   historical verdict.** `curve_stats` uses 252 for annual volatility, Sharpe
   and Sortino while CAGR uses elapsed calendar days. General WFO paths call it.
   A complete seven-day crypto series needs a matching annualization convention;
   this review has not established the input calendar or a historical crypto
   result using that path. [Metrics](../../src/ggTrader/lab/metrics.py),
   [WFO](../../src/ggTrader/lab/wfo.py).

## Coverage and provenance

The ledger inventories all **38** names in `STRATEGY_REGISTRY`; all 38 occur in
snapshot roster rows. It includes **40 pre-existing** top-level dated research Markdown
files, excluding this audit, and six dated candidate-register batches. Each report is represented by a
separate inventory entry; this is not a statement that each report is one
independent strategy or that all distinct claims within each report were
assessed. The initial 38 registry rows preserve source statuses from the
2026-10-23 snapshot and require individual source verification. The candidate
register contains older headings outside the six date-labelled batches; those
remain accessible in [the register](WEB_RESEARCH_CANDIDATES.md) but are not yet
individually reconciled. The ledger inventories 27 archive Markdown files; a targeted source review covers
the three legacy stock and crypto studies discussed below. Exhaustive archive
claim assessment and TimescaleDB run-table inventory, untracked
event-cache review, and every historical variant were not performed. Tracked files and external caches are not equivalent to a complete
backup of data or experiments.

The snapshot records 127 tracked research-path files at this working-tree
revision; this is a file count, not a measure of independent evidence. Some
studies have raw JSON and a defined frozen protocol; older studies and the
historical 1.12 baseline have incomplete or unrecoverable provenance. A
missing source, exact sample, benchmark, costs or data vintage is unknown, not
an implied pass. [Anchor reproduction](2026-08-19-anchor-fix-reproduction.md).

## Shared machinery and chronology

| Issue | Evidence and confirmed consequence | Reach and uncertainty |
|---|---|---|
| Anchor leakage | `2eafab1` replaced full-history anchor with training-end-aware construction; current code uses each fold's information. | Studies run earlier may depend on the old policy; direction/magnitude for each run require an exact revision and artifact. |
| PIT membership | `c9d1abe` added historical membership to signal sweeps. | Predating the change is a dependency flag, not proof of a particular quantified bias. Correct membership alone does not establish correct historical security identity. |
| DSR units | `09f9f40` corrected annual-vs-period Sharpe and kurtosis; prior gate counts cannot establish corrected DSR passage. | The 17/17 corrected-momentum anchor finding is documented; do not generalize it to every run. |
| FOMC timing/unit | Timing correction `7320c1d` and day clustering change event inference. | Apply the later artifact, not the obsolete unclustered headline. |
| WFE reporting | [compute_wfe](../../src/ggTrader/lab/wfo.py) returns `None` when IS Sharpe < 0.4. | `n/a` WFE alone is not proof of a flat OOS book; it cannot resolve the pairs rerun anomaly by itself. |
| Crypto frequency | [curve_stats](../../src/ggTrader/lab/metrics.py) assumes 252 observations per year for volatility-based statistics. | Actual seven-day crypto call path and historical impact unresolved; total return and calendar CAGR have different calculations. |

A properly constructed synthetic control could introduce a known, fixed
signal and negative/no-signal counterpart into a deterministic tape, checking
whether the pipeline preserves the clock, portfolio and benchmark accounting.
A second control should deliberately alter input prices without changing dates
and symbols to test cache isolation. Prospective power curves should be
specified for an effect, volatility/tracking error, dependence, family size,
training/sample length and each gate, with false-pass and false-kill rates.
These are proposed diagnostics, not results, and no computation is authorized
by this report.

## Representative source-to-verdict traces

### Pairs: implementation and historical data

**Question and rule.** The [study](2026-07-17-pairs-stat-arb-nogo.md) tests
same-sector S&P 500 pairs selected on 126-day return correlation, with a
1:1 log-price spread and monthly portfolio decisions. Its 54-combination WFO
sweeps entry/exit levels, pair count and one/two-month minimum holds. This is a
deliberately simplified variant, not an established replication of Gatev or a
cointegration test. Inspecting [pairs code](../../src/ggTrader/lab/strategies/pairs_stat_arb.py)
confirms the cache excludes the input frame and that current ranking and symbol
exclusions can remove a previously held pair despite its hold floor. Whether
that happened in the historic run is unknown.

**Result and counterargument.** The July 26 rerun reports Sharpe −0.42, CAGR
−2.6% and max drawdown −28.2% vs SPY Sharpe 0.88 and CAGR 15.2%; it omits
borrow fees and margin interest. The reported 32/42 halted folds mean fallback
matters; those pre-October DSR pass counts are not evidence of corrected DSR
validation. The paper's dated data revision, current-code behavior and the
bit-identical-output anomaly must not be conflated. A market-neutral sleeve
also needs cash/funding and risk-matched comparisons alongside SPY, since
directional SPY is not an equivalent exposure. The current record is negative
for this tested policy; it cannot validate or reject all pairs methods.

### Momentum: analogue versus exact rules

**Question and rule.** [Brief](briefs/2026-10-20-momentum-retest.md) registers
French Big/Small HiPRIOR long-history idea screens; [implementation](../../src/ggTrader/lab/strategies/momentum.py)
selects an eligible top-N stock book. The proxy has a different universe,
weighting and selection breadth. The two groups and their Holm family were
frozen; the source data are not a fresh independent forward period after
publication. The later French source description confirms its breakpoints.

**Result and counterargument.** Full-window Big and Small t statistics 3.052
and 3.108 fail the registered Holm step-down despite individually exceeding
3; independently post-1994 net CAGR differences +0.08 and +0.20 pt/year fail
+1.0. Haircuts are 1%/year Big and 3%/year Small, not observed execution or
tax-lot costs ([raw JSON](_momentum_french_screen_results.json)). The
shorter corrected WFO replay's xs/dual +4.367 pp/year against SPY and t 0.761
is favorable in point estimate but inconclusive and used anchors in all 17
folds; the two return streams were identical, not two confirmations. These
results support the registered screen's NO-GO, while neither cost assumptions
nor point estimates statistically exclude every practically useful version.

### Fixed-rule production replica: portfolio, parity and coverage

**Question and rule.** [The brief](briefs/2026-10-19-production-replica-audit.md)
and its amendments test prior-completed-bar decisions, fixed production-like
defaults, 5 bp/side stocks, 1 bp/side SPY sweep and accrued cash, not WFO-picked
parameters. The normalized adjusted-close SPY benchmark excludes separately
deducted benchmark entry/liquidation fees; the report states that asymmetry.
The stock overlay and slot cap result in 9.37% average stock, 85.33% SPY and
5.30% cash. A portfolio-vs-SPY comparison asks about the full construction;
the exposure-matched comparator helps assess selection with a different,
still limited, question.

**Result and counterargument.** The primary −0.642 pp/year and NW t −0.690
fail both registered bars, and the stock-cost sensitivities remain negative
vs SPY. The source shows +0.265 pp/year, t 0.202, versus exposure matched;
neither establishes selection skill. Historical live parity is separate and
unverified, and the 2010–2020 holdout has 11 unique tainted trades with a
stop; it cannot be used as performance evidence even though member-day price
coverage passed its floor. Claims that the model replicates exact live orders
or that its estimate excludes all worthwhile positive skill exceed the source.

### FOMC: event clock and evidence unit

The [original implementation](2026-07-20-fomc-drift-nogo.md) uses the signals
path for 0–3-day Treasury holding windows, so a blanket claim that all tested
strategies were forced to monthly cadence is false. The original report's
calendar-date matching issue was fixed; the later [timing correction](briefs/2026-10-19-production-replica-fomc-timing-amendment.md)
removed pre-entry returns, and the [clustered artifact](_fomc_clustered_event_study_20261020.json)
uses 113 common event dates. Gross t 2.258 and net t 0.981 are not the
unclustered 3.619/1.572. The net hypothesis failed on the corrected calculation;
the point estimate and gross result should not be represented as proof of zero
effect. The artifact and report, not an independent fresh reproduction, are
the evidence here.

## Primary-source checks

The [French monthly size/momentum construction](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_6_port_form_sz_pr_12_2.html)
uses the median NYSE size breakpoint and the 30th/70th percentile prior-return
breakpoints. That supports the reported distinction from a fixed S&P 500 top-N
book; the page alone does not validate the project's costs or implementation.
[Gatev, Goetzmann and Rouwenhorst](https://www.nber.org/papers/w7032) describe
a daily-data pairs study. Its existence shows that pairs designs can differ in
cadence; it does not establish that ggTrader purported to reproduce that exact
paper or that a daily version would be profitable. The abstract was available
in search, while direct NBER page access returned 403; full primary-text
comparison remains incomplete. [Harvey, Liu and Zhu](https://www.nber.org/papers/w20592)
discuss higher hurdles in the context of extensive factor searches; their
abstract does not establish that one fixed hurdle has equal power for every
strategy. Full-paper access via the requested page returned 403.

## Framework for a fair second look

1. Write the original economic proposition and distinguish source-paper
   portfolio, ggTrader implementation, and the taxable funded portfolio.
2. Trace information clock, universe, holdings, hedge, rebalance, execution,
   costs and benchmark to actual code and the run's exact version. Identify
   deliberate approximations and verified defects separately.
3. Separate the historical decision from the inference: a point estimate below
   a practical bar, a statistically imprecise estimate, and a defensible test
   excluding a pre-specified meaningful effect are different outcomes.
4. Ask whether any proposed correction is identifiable without choosing it
   because it improves past returns. Freeze its question, compare original and
   corrected definitions, account for the full family of prior variants, and
   retain adverse outcomes. Reused historical data stay exploratory.
5. Label a finding **unresolved** where missing data, clocks, provenance or
   stopped tests prevent a reliable answer. A hypothetical retest is not an
   owner authorization or evidence of an edge.

The current [forward standard](briefs/2026-10-23-forward-paper-standard.md)
uses a registered Part A family/significance and +1 pt/year modern economic
bar; B0 is a kill-only replay, and B1 is at least 18 months of forward shadow
with after-tax accounting. Its short B1 record is a reality check, explicitly
not proof of significance; its R5 resolution makes a 15%-or-more relative-wealth
stop inclusive despite an inconsistent `<` formula in the body. The filter can
reduce false promotions but miss a real edge; the short replay and shadow can
also reject good candidates due to noise. Part A uses long-history proxy
availability and stated cost haircuts, and its stock-specific design does not
by itself specify an appropriate crypto protocol. Assess operating
characteristics prospectively without changing a registered threshold after
seeing performance or treating within-family Holm correction as control of all
search conducted across studies.

## Earlier archived research: stock and crypto

The registry is not the entire history. The archived [S&P 500 research](../archive/sp500_equity_strategy_research.md)
explicitly invalidates its own Sharpe 2.15 headline because of full-period
selection, overlapping parameter selection, current-constituent survivorship
and replay/indicator bugs. It is a positive-control mirage, not evidence that
the system once demonstrated deployable alpha.

The legacy [crypto edge search](../archive/edge_search_report_2026-06-08.md)
compares per-coin WFO searches, venues and fee assumptions. The all-51
Binance.US run reported +13.8% CAGR vs BTC +38.8%; a Kraken maker-fee run
reported −10.0%. These runs do not isolate fees alone: the universes and
selected coins differ, so the report's causal assertion that fees by themselves
flipped the result is not established by that comparison. Sparse holdout trades
and a selection process after many combinations further limit the conclusions.
The three named result directories exist locally, but none of their files are
tracked by Git; the exact evaluation windows and raw contents were not
independently reconstructed. This is evidence about the legacy tests, not a verdict on every
crypto signal.

The archived [funding-carry study](../archive/phase4_funding_carry.md) reports
one year of actual Kraken funding, +0.45% net return and eight trades; older
periods use a basis proxy. Its own overlap audit finds correlation 0.983 but
**zero matching entry/exit events**, so high correlation does not establish
that the proxy replicates the trading strategy. The report itself rejects
numerical use of the proxy return. Venue access, financing, capital, historical
data and tax details have not been independently verified for a current
implementation; nothing here authorizes a trade or a purchase.

## Family-level review: decisions and limits

This table is a sourced triage of the research families; the ledger has the
individual registry names. These are not independent, equally powered tests.

| Family | Existing evidence / decision | Audit interpretation and strongest qualification |
|---|---|---|
| Ensemble and variants | [PIT rebaseline](2026-09-23-pit-rebaseline-core-nogo.md) reports core Sharpe 0.59 vs SPY 0.78; IC 0.41 and Kelly 0.50. The [fixed-rule replica](2026-10-19-production-replica-audit-nogo.md) separately fails its registered primary. | Negative comparisons apply to different constructions and samples; the old selected-parameter core is not the deployed fixed-rule book. None proves all voter-level effects absent. |
| Momentum | [French screen](2026-10-20-momentum-french-screen-nogo.md) fails both registered A1 and A2 for both groups. | Registered screen NO-GO stands; proxy and cost-haircut assumptions limit conclusions about exact top-N rules. The apparent short-window WFO uplift has low precision and is not independent validation. |
| Profitability, value, investment, combo | [Factor screen](2026-10-21-factor-screen-nogo.md) fails Holm and modern hurdle for all four; H1 is +0.83 point/year net vs market while H2 is −0.62. | Results are for broad French portfolios and assumed costs, not a proof all related stock rules lack useful effects. Do not call H1's +0.83 negative. |
| Net issuance | [Screen](2026-10-23-net-issuance-screen-nogo.md) has long-only A1 t 0.26 and A2 +0.48 pt/year, failing its bars. | A heavy-issuer short-side diagnostic is a different exposure; neither it nor the long-only failure validates a usable taxable long-only strategy. |
| Event and information signals | [PEAD](2026-07-17-pead-nogo.md), [insider](2026-07-19-insider-cluster-buy-nogo.md), [Congress](2026-07-19-congress-trades-nogo.md), [short interest](2026-07-17-short-interest-nogo.md), and [short volume](2026-07-19-short-volume-ratio-nogo.md) report historical NO-GOs. | Historical price and timing limitations matter; the [later event decision](2026-10-23-event-timing-audit-decision.md) leaves PEAD and insider feasibility unresolved and parked. Positive or negative effect of any individual data repair is not automatically known. |
| Headline sentiment | [Report](2026-07-24-headline-sentiment-nogo.md) is provisional with three usable folds. | Negative performance is real for the tested sample; the limited history cannot by itself settle the broad sentiment mechanism. |
| Calendar and macro | [FOMC correction](2026-10-19-production-replica-audit-nogo.md#correction-2026-10-20) shows net clustered t 0.981; [month-end Treasury](2026-09-24-month-end-treasury-sleeve-nogo.md) fails when financed out of SPY. | A gross event premium and a funded-sleeve improvement are distinct. The displaced asset, event-day dependence and costs must match the question. |
| Leveraged timing | [Rotation](2026-07-16-leveraged-index-rotation-nogo.md) and [trend](2026-07-16-leveraged-trend-following-nogo.md) report NO-GO across three universes. | Six registry names are two mechanisms over three universes, not six unrelated experiments. Historical gates, static comparators, drawdown and persistent halts must be retained. |
| Macro approximations | [Treasury curve](2026-07-20-treasury-curve-nogo.md), [FX hedge](2026-07-20-fx-hedge-overlay-nogo.md) and [commodity trend](2026-07-20-commodity-trend-nogo.md) record historical NO-GOs. | The curve report expressly calls its three-ETF design an approximation of a four-factor source; failure of that construction is not a replication failure of the source model. |
| Cross-asset and static leverage | [80/20](2026-09-28-cross-asset-sleeve.md) static shadow-only GO under its own bar; [A23/A24](2026-09-30-static-fractional-leverage.md) records an owner allocation decision and timing rider NO-GO. | These are allocation/risk tradeoffs, not stock-selection validations; their different objectives help explain why their evaluation differs from a stock-alpha hurdle. Existing owner decisions are not reopened. |
| Crypto | [Candidate register](WEB_RESEARCH_CANDIDATES.md) and [IBIT artifact](artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.json) contain feasibility and descriptive material. | Neither a cited paper nor an ETF overnight split establishes a tradable, after-cost crypto strategy; venue, calendar and taxation must be matched to actual ownership and trading. |

## Judgment of NO-GO criteria and future alternatives

A gate should match the decision it serves. **Research advancement** can justify
a predeclared minimum economic benefit, but that minimum is a policy choice:
it is not a measurement of zero alpha or a universal minimum for every
diversifier. **Statistical evidence** should state the sampling model, one- or
two-sided hypothesis, family of tested variants and the size of an effect it
can detect. A high hurdle protects against spurious discoveries but misses more
real signals when the sample is weak. Comparing an estimated benefit with a
minimum without an interval is a decision rule, not a demonstration that the
underlying benefit is below the minimum. **Mechanics validation** should check
fills, timing, reconciliation and tax lots; an 18-month shadow is valuable for
those observations but is not a replacement for a long independent return
record.

The following are options for a future dated review of standards, not changes
to the approved rules:

1. **Retain the current stock Part A/B0/B1 gates**, while reporting expected
   detectability and confidence intervals alongside the existing decision.
   This retains prior false-positive discipline but does not open a route for
   mechanisms lacking a suitable long-history proxy.
2. **Consider a mechanism-specific evidence path in a new brief** for a future
   candidate without a meaningful French-style proxy: state the experimental
   unit, feasible sample, economic comparator, tax/execution model and a
   prospective multiple-testing policy before viewing new outcomes. This may
   reduce proxy mismatch but increases false-discovery and short-sample risk.
3. **Separate a stock-selection diagnostic from portfolio eligibility**:
   specify both an exposure-matched diagnostic and the taxable portfolio-level
   decision in advance. This can show where a rule's behavior originates, but
   a selection score that fails to survive realistic portfolio funding, costs
   or taxes is not a deployable edge.

These alternatives do not grant exceptions or reopen closed decisions. A
report should state whether the registered threshold was missed, whether
uncertainty excludes a meaningful effect, or whether evidence is invalid or
missing; those are different statements. Regime-dependent results should
retain the original window and present a matching benchmark, rather than
select a favorable subperiod after observing it.

## Documentary contradictions to retain as caveats

The [older roadmap](../roadmap.md) presents 1.12 as a validated core and
suggests the daily equity book is nearly exhausted. The later [anchor
reproduction](2026-08-19-anchor-fix-reproduction.md) says that headline's
window cannot be recovered, and the [current snapshot](RESEARCH_SNAPSHOT.md)
sets a narrower fixed-model NO-GO. Old gate counts predate the DSR correction.
The old roadmap's industry-wide interpretation does not follow from those
tests. Historical documents are not rewritten here.

The [MAX report](2026-07-17-max-effect-nogo.md) attributes different results
from runs minutes apart to an unpinned `now().date()` evaluation end. A drifting
end date is a real reproducibility risk across dates, but a date-only value
does not explain a same-day change by itself; the specific discrepancy remains
unverified. Further, a higher correlation than another sleeve's does not, on
its own, prove a prospective blend must underperform. The reported standalone
5.0–5.1% CAGR and 0.39–0.45 Sharpe versus SPY's 15.2% and 0.76–0.88 remain
unfavorable for that historical construction, subject to its tape and window
caveats.

The [short-volume report](2026-07-19-short-volume-ratio-nogo.md) openly says
its FINRA total short-volume proxy omits the paper's liquidity-supplying vs
demanding decomposition. Its reported 1.1% CAGR and 0.21 Sharpe in a
long/short book do not test that decomposition. SPY provides market context,
but is not a risk- or exposure-matched market-neutral comparator; missing
borrow and financing details require separate verification.

## Research gaps and follow-up boundaries

- **Shared-policy attribution:** record which folds used winner vs fallback and
  why, at each historical revision; estimate operating characteristics only
  with a separately scoped synthetic diagnostic. A failed positive control
  would disconfirm confidence in the machinery; a passed control would not
  certify all historical data.
- **Pairs anomaly and hold logic:** inspect contemporaneous run provenance and
  determine whether a data-identity-sensitive cache or missing hold-floor
  guarantee affected any specific run. Database access and any new replay are
  outside this completed work; a historical difference would not constitute an
  independent prospective strategy validation.
- **Mechanism translation:** identify whether a proposed source has executable
  timing, universe and funding requirements compatible with free, point-in-time
  data. A more faithful specification is a new hypothesis, not evidence the
  old negative result was false.
- **Stock selection vs portfolio exposure:** independently specify a
  selection-quality question and risk/exposure-matched baseline before any
  further measurement. Existing signal selections and observed data cannot be
  relabeled untouched holdouts.
- **Crypto:** investigate exact data calendars, market/venue eligibility,
  costs, outages and tax accounting before interpreting a metric or transfer of
  a stock decision rule. The IBIT [sanity-check result](artifacts-2026-10-25-crypto-web-batch/ibit_overnight_split.json)
  is descriptive, not an executable WFO or after-tax result.
- **Event data:** PEAD and insider source feasibility remain unresolved and
  parked under the [owner decision](../next_steps.md). The canonical audit was
  not completed; neither a new verdict nor renewed acquisition is implied.

## How follow-ups could be evaluated

The representative cases were chosen to cover distinct mechanisms and
measurement environments—monthly WFO, external long-history proxy, fixed-rule
portfolio and clustered event study—rather than attractive results. The crypto
archive was added to cover a different market and a proxy-versus-trading-clock
problem. No additional strategy was selected for a return test.

| Documentary question | What would weaken the concern? | Evidence and boundary |
|---|---|---|
| Did fallback mask parameter selection? | Fold-level records show the winning parameters, rather than the anchor, were consistently deployed for a particular historical run. | Recover that run's code, fold diagnostics and flags. A control or rerun would require a separate scope; old returns remain seen. |
| Did pairs cache or holding behavior affect a named run? | A reproducible contemporaneous trace shows distinct inputs never shared a key and all selected holds respected the intended rule. | Recover run provenance and candidate/hold traces. A differently hedged or faster variant is a new hypothesis, not a fix proven by old returns. |
| Is the stock/portfolio question sensitive to dilution or ordering? | Decision-time records and a prespecified exposure-matched calculation show conclusions are insensitive to the cap or ordering. | Immutable inputs are missing for some original live decisions; any replay must preserve the registered primary failure and disclose reused data. |
| Can the crypto metric path use a 252 convention on seven-day data? | Trace the relevant historical run and show either the series was not seven-day, or that it did not use `curve_stats`. | Locate config, input frequency and metric call in the exact run; a code-level possibility alone is not evidence of a past distorted result. |
| Do short interest or short volume reproduce the cited economic mechanism? | Source-to-rule comparison establishes that the public proxy and tested side match the intended published hypothesis at a tradable information time. | The FINRA volume proxy lacks the stated decomposition; missing free point-in-time inputs stay a limitation. No paid acquisition is proposed. |

These are evidence questions only. Nothing in this table reopens a closed
strategy, permits new data collection, or authorizes a new market-return run.

## Remaining evidence gaps

This audit does not certify every historical study or reconstruct missing run
inputs. All 38 registry entries have a current-class location and a scoped
assessment, but **25 are qualified and 13 unresolved**; these are not counts
of independent experiments or of strategies that deserve a retest. The ledger
keeps missing run revisions, historical code/data identity, measured costs and
tax treatment explicitly unknown where they were not established. Forty report
entries and six candidate batches are navigation records, not 46 additional
independently assessed strategy claims. Candidate headings are individually
linked in the ledger; their original statuses remain in the register.

The database, most archive content, raw untracked caches, and some external
full texts were not examined. The source-paper-to-code comparison is detailed
for the representative cases, while other entries carry narrower documentary
assessments. Proposed prospective power analyses and synthetic controls are
not empirical results. No claim that all NO-GOs deserve or do not deserve a
second test follows from this audit. The [execution plan](../superpowers/plans/2026-11-01-research-process-audit.md)
records the work and the authorization boundaries.

## Addendum (2026-11-02): fold attribution and follow-ups

This addendum adds pointers only; no verdict or classification above changed.

- The "walk-forward tests can principally measure fallback" concern now has counts from saved fold records: `xs_momentum` and `dual_momentum` replays deployed the anchor in 17/17 folds and `fomc_drift` in 47/53; the PIT `ensemble_ic` run did in 15/17, and the winner was deployed in most folds of the other runs reviewed. See [fold attribution](2026-11-02-wfo-fold-attribution.md). Pre- and post-DSR-fix runs are kept separate.
- Current fallback/halt behavior is specified and pinned by tests in [fallback semantics](2026-11-02-wfo-fallback-semantics.md); intended behavior is still an owner decision.
- The free-data direction question is inventoried, without a recommendation, in [free-data stock feasibility](2026-11-02-free-data-stock-feasibility.md). The B0/B1 accounting prerequisite is in [after-tax lot engine requirements](2026-11-02-after-tax-lot-engine-requirements.md).

