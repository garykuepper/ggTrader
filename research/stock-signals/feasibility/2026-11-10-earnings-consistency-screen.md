# Earnings consistency: exploratory historical portfolio screen

**Status: exploratory; no paper or trading authorization.** Question: did a published
portfolio of companies with consistently growing earnings outperform the broad US
market? This is a portfolio-level investigation, not an executable stock picker.

The comparison uses Open Source Asset Pricing's `EarningsConsistency` long-only
portfolio 05, the high end of its five groups. The signal averages up to five yearly
earnings-growth observations, with a sign-consistency filter on the two latest; missing
values can reduce the number included. The provider describes annual June assignment
and equal weighting. Its metadata conflicts about a $1 versus $5 stock-price filter;
the exact filter applied is unverified. This overlaps the economic theme of the closed
[profitability screen](../factor-screen/2026-10-21-factor-screen-nogo.md) and the
[earnings-surprise study](../pead/2026-07-17-pead-nogo.md), without reopening either.

Compare against French market total return (not SPY) over July 1963–December 2025
and January 1994–December 2025, with assumed candidate-only costs of 1% per year
and a 2% stress case. These windows and assumptions match the earlier 52-week-high
screen. Record annual growth, drawdown and a 12-lag Newey–West t statistic. The
existing programme's t ≥ 3 and recent-window advantage ≥ 1 percentage point per
year provide context; they do not make this a confirmatory test.

This signal was chosen after other screens and publication of its literature, and
its historical source was already accessible. The specification and result are in
the same uncommitted file; no independent record establishes that the choice preceded
the calculation. Multiple testing, size exposure, revised accounting,
different source vintages, taxes, costs and stock-level fills limit the inference.
Do not adjust windows, costs, portfolio or benchmark after seeing results to claim a
pass. Preserve the production freeze and all earlier verdicts.

Source: [provider signal implementation](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/Signals/pyCode/Predictors/EarningsConsistency.py)
and [signal documentation](https://github.com/OpenSourceAP/CrossSection/blob/9f053398c4995651d822e90d81526d0644c6f3e5/SignalDoc.csv).

## Result

**No reliable winner identified; independently recomputed.** With the 1% assumed
annual cost, the portfolio returned 11.93% a year versus the market's 10.80% over
1963–2025. The paired t statistic was 1.23 (one-sided p = 0.110), below the existing
programme's 3.0 reference. Since 1994 it returned 10.75% versus 10.81% for the
market, falling short of the 1-point reference. These are historical comparisons,
not confirmatory passes or a reason to trade.

The deepest month-end fall over the full period was 69.81%, versus 50.31% for the
market; since 1994 it was 57.62%, versus 50.31%. With 2% assumed costs, the
full-period annualized advantage shrinks to about 0.02 points (t = 0.65).
At the 2% assumed-cost stress, annual returns since 1994 were 9.66% versus 10.81%.
Costs are assumptions, not measured trading costs; taxes and intramonth losses are
excluded. The equal-weighted candidate and value-weighted market also differ in
size exposure. The long-run result does not establish that a smaller edge is absent.

The independent reviewer reproduced the calculations from the frozen files; the
provider's long–short series also matches group 05 minus group 01. This supports the
high-group mapping, not security-level timing or executability. The original paper's
1971–2002 sample begins later than this 1963 comparison. The source uses an accounting
availability lag, but revisions and backfilled EPS are possible, and delisting-return
handling was not verified. Its broad universe can include micro-caps. The portfolio
file is release `dd1054254c4501776432caa6d7ce5ecd9586f6a9`; the French market archive
is the separately downloaded 2026-11-10 copy, a different vintage, not the repo's
tracked archive. The raw [result](./_earnings_consistency_screen_20261110.json) records
source hashes and a retrospectively added dependency-hash note.

## Source-availability follow-up (2026-11-11 UTC)

The provider's separate firm-level file from the same release contains future-labelled
earnings-consistency rows. In one inspected group, 11 of 80 non-null future rows differ
from the same security's latest earlier value. The [feasibility note](2026-11-02-free-data-stock-feasibility.md)
describes a possible annual-data expansion mechanism, but the cause and historical
filing-time availability remain unverified. This observation does not show that the
separate portfolio file contains lookahead. It limits any inference about implementing
the signal in individual stocks; it does not alter the recorded comparison.
