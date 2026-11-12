# Low-accruals portfolio: fixed research screen

**2026-11-10 — registered specification; research only.** The owner directed this project to find winning stocks programmatically, let gains grow and manage losses. This screen asks whether one proposed quality characteristic merits further investigation; it does not implement stock selection, exits or a trading system, and cannot authorize capital, broker paper or changes to production.

## Candidate and limits

One hypothesis: the published French 5×5 **BIG LoAC** value-weighted portfolio beats the French broad market after a cost assumption. Portfolio formation and eligibility are described in the [provider's detail](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/tw_5_ports_me_AC.html). Its revised historical portfolio does not supply contemporaneous stock holdings, verified accounting vintages, transaction-level costs or taxable lots. A favorable result is not proof that ggTrader can reproduce or trade it.

The family contains this one accruals hypothesis. Earlier momentum, investment-factor and net-issuance screens and their NO-GOs remain in force; their data, benchmarks and periods overlap. This screen has no cross-study multiplicity guarantee. The choice of accruals after reviewing earlier work is disclosed, and all historical market regimes are already known. No accruals portfolio returns have been viewed in this preparation; a prior check counted missing/non-finite values only.

## Frozen sources and calculations

- The source ZIPs and SHA-256 hashes are listed in the [source note](../../programme/verdict-framework/2026-11-10-accruals-source-metadata.md). The proposed comparator is French market total return (`Mkt-RF + RF`), not the S&P 500. Both files were retrieved together, which does not prove matching underlying accounting vintages.
- Full window: July 1963–April 2026, 754 months. Since-1994 window: January 1994–April 2026, 388 months. These reuse the preceding feasibility proposal's dates; the April 2026 end aligns with the earlier project screens, not an independently unseen holdout.
- Apply a 1% annual hypothetical cost haircut as `0.01/12` per month to the candidate, and show a 2% annual stress haircut. These are assumptions, not measured costs or taxes. The market comparator gets no corresponding haircut; disclose this asymmetry.
- A1 uses paired monthly candidate-minus-market returns after base cost, 12-lag Bartlett Newey–West standard error, one-sided normal-tail p-value; require **t ≥ 3 and p ≤ 0.00135**. For this single registered hypothesis Holm has one threshold. Zero standard error is unresolved rather than a pass.
- A2 requires the candidate's monthly-compounded, base-cost CAGR to exceed the market's by **at least 1.0 percentage point/year** over the since-1994 window. Both A1 and A2 must pass to meet this screen. These are the existing Part A policy gates, not guaranteed economically achievable returns or an after-tax test. Failed or unavailable data produce no pass.
- The SPY diagnostic uses the separately hash-pinned adjusted-close CSV from the earlier protection analysis: 2021-04-30 through 2026-04-30, converted to monthly returns May 2021–April 2026. It is descriptive only and cannot rescue failed primary gates. SPY is not the full-history primary comparator. Its daily calendar completeness and adjustment accuracy are not independently established.

The [driver](code/run_accruals_screen.py) checks frozen input hashes and the required column, refuses an existing output path and reports results without inventing an authorization. Its [calculator](code/accruals_comparison.py) rejects misaligned, missing, duplicate, non-finite or non-positive wealth factors; it applies the 12-lag calculation and the cost assumptions. A missing required input stops the run. The source note records the local private copies and their hashes; no raw source files are committed.

## Interpretation and next stage

Record the actual A1 and A2 values and original policy outcome separately from the evidence label and action. A miss does not prove no smaller edge. A pass, if any, establishes only a reason to investigate the published portfolio; data revisions and unknown implementation costs limit inference. Future individual-stock work requires a separate point-in-time fundamentals and security-identity pipeline, entry/exit and sizing rules, realistic costs, and independent out-of-sample evidence. No registration here starts a paper or live strategy.
