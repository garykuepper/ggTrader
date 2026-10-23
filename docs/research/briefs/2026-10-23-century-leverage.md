# Brief: century-scale static fractional leverage in a taxable account

**Status:** DRAFT 2026-10-23, awaiting owner approval. Nothing has been computed.

**Decision fed:** one input to the owner's **2027-09-30** annual A23 review. This brief
does not change the adopted 70% SPY / 30% SSO allocation now, authorize a trade,
or create a drawdown exit. The account is taxable only. This is an allocation-risk
study, not an edge test: any higher return from leverage compensates higher risk.

## 1. Question and fixed arms

Across roughly a century that includes 1929–32, the 1940s, and the 1970s
high-rate era, which static 1x/2x-daily-fund mix maximizes modeled after-tax
terminal wealth, and how severe is its worst case? Test **only** 2x-fund target
weights **0%, 15%, 30%, 45%, and 60%**, approximately 1.0x, 1.15x, 1.3x,
1.45x, and 1.6x daily market exposure. The 0% arm is the common comparator;
30% is the adopted A23 arm. Do not search for a different weight, band, window,
or tax treatment after seeing results.

## 2. Free data and freeze before calculation

Use the free [Ken French daily Fama–French three-factor CSV archive](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip),
`F-F_Research_Data_Factors_daily_CSV.zip`. On 2026-10-23, inspection in
`/tmp` was limited to the archive member's introductory text, column labels,
and first/last date labels; **no return or statistic was inspected or computed**.
The member is `F-F_Research_Data_Factors_daily.csv`. It has no explicit section
title; its introduction says it was created with the `202608` CRSP database.
Its sole daily section has header `,Mkt-RF,SMB,HML,RF` and date labels
**1926-07-01 → 2026-08-31**. The registered endpoint is the latest month
common to the section's `Mkt-RF` and `RF` labels: **2026-08**, subject to
the frozen-file completeness checks below. The
[French factor description](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_factors.html)
defines the market as a value-weighted all-US-stock series. **It is not the
S&P 500 or SPY.**

After approval and **before any return computation**, freeze the exact raw
archive URL, download time, and SHA-256 in `data/external/french/MANIFEST.json`;
verify the checksum on every later read. No paid data, now or later. Parse
percentage fields to decimal daily returns. Set `r_mkt,t = (Mkt-RF + RF)/100`
and `rf_t = RF/100`, using only the frozen daily section. Stop on changed
headers, date gaps or duplicates, non-finite required values, or a nonpositive
daily wealth factor; report the defect rather than silently dropping dates.

## 3. Registered synthetic legs and one-time calibration

The 1x leg is `r1x,t = r_mkt,t − 0.000945/252` (SPY fee: **0.0945%/yr**).
The hypothetical daily-reset 2x leg is

`r2x,t = 2·r_mkt,t − rf_t − (0.0089 + spread)/252`.

The fixed **0.89%/yr** SSO fee is a scenario input supplied by the senior
quant, **UNVERIFIED as the current expense ratio**. The [current ProShares SSO
page](https://www.proshares.com/our-etfs/leveraged-and-inverse/sso) showed
**0.88% gross / 0.84% net** on 2026-10-23; its [current summary prospectus](https://www.proshares.com/globalassets/proshares/prospectuses/sso_summary_prospectus.pdf)
agrees. Do not silently replace 0.89% or the A23 study's 0.88% with another
value. Resolve the conflicting fee descriptions before approval or retain
0.89% explicitly as a conservative hypothetical assumption.

Calibrate **one** annual financing spread using the **2006-06-21 → 2026-04-30**
overlap only, with free yfinance real SPY and SSO daily total returns and the
same frozen French daily `RF`. Fit the synthetic 2x formula with SPY as the
market proxy against real SSO daily returns, minimizing **daily root-mean-square
return error (RMSE)** over the dates common to all three series in that fixed
window. Use an unconstrained signed constant spread, estimated once by the
mean daily synthetic-minus-real residual times 252 at zero spread; do not
clip a negative estimate or drop an aligned observation after seeing its
return. Record the aligned start/end dates, count, and any missing sessions;
freeze the fitted spread and calibration input identities **before** calculating
any long-history result. Report the fit, residual bias, and residual dispersion
without refitting on a subperiod or a century outcome. The spread cannot
remove the French-market-versus-S&P-500 basis difference outside calibration.
This is a **proposed calibration definition for owner approval**: conventional
tracking-error standard deviation is invariant to a constant spread and cannot
identify one. Because actual SPY returns are net of its own fee and real SSO
has other tracking frictions, the fitted parameter is a **composite tracking
residual**, not an identified economic financing premium. This modern overlap
is fitted/in-sample; the earlier path is a backward synthetic stress exercise.

Both legs compound **daily**; there was no 2x daily-reset ETF in 1929. The
pre-inception history is a hypothetical return path, not a tradable historical
fund and not a genuine out-of-sample validation of an ETF that did not exist.
It omits margin calls, counterparty stress, fund closure, changing swap terms,
and failures of daily exposure tracking in a crisis. Label every century
result **synthetic**; do not imply the fitted modern spread held in 1932 or
the high-rate 1970s.

## 4. Portfolio and taxable-account accounting

Start every arm and separately evaluated window from the same **$100,000**.
Use the existing [`static_leverage_aftertax.py`](../../../scripts/static_leverage_aftertax.py)
lot engine and A23's **1 bp per side**, distributions reinvested, monthly
month-end band check and next-session trade, new cash first if available, then
highest-basis long-term lots first. Rebalance to target only when the 2x
weight leaves **target ±8 percentage points**. Thus 30% retains A23's
**22–38%** band; the 0% arm is buy-and-hold 1x with no rebalance. No signal,
rate switch, or drawdown-based trade. Use the existing federal lot rules,
including more-than-one-year long-term classification, loss carryforward,
taxes paid from the portfolio the following April, and terminal liquidation.
Primary bracket: **24% short-term / 15% long-term**; sensitivity:
**40.8% / 23.8%**. The recommendation gate uses the A23 baseline
**`wash=False`**; report its approximate `wash=True` path as a sensitivity
for both brackets. Keep the A23 distribution treatment and disclose its
simplifications. Reinitialize lots at each reported subperiod and rolling
window start, then liquidate at that window's end; never infer an after-tax
rolling return by subtracting points on one full-history tax path.

**Unresolved distribution-data gap:** the A23 engine gives synthetic pre-2006
SSO **zero distributions**, deferring all synthetic gains. French's daily
market total return and `RF` do not split cash distributions from price
returns, including for the 1x leg. “Treat distributions as in A23” therefore
cannot reproduce actual annual dividend taxes over 1926–2026 from these
fields alone. Before any after-tax calculation, approve either a documented
**free** historical distribution reconstruction for both legs or an explicitly
labelled zero-distribution hypothetical bound applied consistently. Do not
present the latter as realized after-tax SPY/SSO wealth or as a rigorous bound
on the **relative** advantage of one mix. Preserve and report
the A23 engine's other known simplifications: SSO distributions default to
long-term/qualified rates, federal tax only, no $3,000 ordinary-loss offset,
and approximate wash-sale handling. Do not change the existing engine's
results while adapting it. First attempt to reproduce the committed
[`_static_leverage_aftertax_results.json`](../_static_leverage_aftertax_results.json)
from the unmodified A23 driver and its original input identities; if the
mutable yfinance inputs have changed, report the mismatch and stop rather
than silently updating the committed artifact.

## 5. Frozen output plan

Report each fixed arm for the **full 1926-07-01 → 2026-08-31** window and
these non-overlapping calendar blocks: **1926–1945, 1946–1972, 1973–1981
(high-rate), 1982–1999, 2000–2026** (last block ends 2026-08-31).
For each, report after-tax CAGR and terminal wealth, max drawdown and longest
time to regain the prior peak, and whether any drawdown reaches **−90% or
worse** (ruin flag). Define drawdown and recovery on the portfolio NAV after
annual taxes paid but before terminal-liquidation tax, following the A23
engine and including the initial $100,000 in the peak path. Unpaid accrued
tax is absent until April payment, which can make that payment look like a
market drawdown. Count recovery in **trading sessions** from a prior NAV peak
until that peak is regained; report the longest completed recovery and label
any still-unrecovered peak with its elapsed sessions, never zero. Annualize
all after-tax CAGRs by **actual calendar days / 365.2425** between the first
and last observations; recompute from terminal wealth if the reused A23
engine returns its `len(idx)/252` convention. For the full window, also
report each arm's worst rolling **10-year and 20-year**
after-tax CAGR difference versus the 0% arm and the share of those rolling
windows that trail 0%. Start a rolling window at each **calendar month-end's
last available trading session** and end it at the last trading session of
the same calendar month **120 or 240 months later**, using every such pair
inside the frozen data. Compute each as a fresh equal-capital tax-lot
simulation with terminal liquidation and matched start/end dates;
include the worst-window dates. Preserve unrounded values for all gates.
Store raw results and provenance in
`docs/research/_century_leverage_results.json`, then write a report that
shows every arm, era, and failure plainly. No result exists at draft time.

## 6. Pre-registered recommendation rule for the 2027 review

The **only** way this study changes the senior quant's advice for the
2027-09-30 A23 annual review is to recommend moving from 30% to a fixed
alternative that satisfies **all four** conditions on unrounded primary
24%/15% results: (a) full-window after-tax CAGR exceeds 30%'s by at least
**+0.5 percentage point per year**; (b) its 1973–1981 after-tax CAGR is
**not below** 30%'s; (c) its full-window max drawdown is no more than
**10 percentage points deeper** than 30%'s; and (d) it never trips the
**−90% ruin flag**. A recommendation **below 30%** additionally requires
that the 30% arm itself trips the ruin flag **or** trails the 0% arm in
full-window after-tax CAGR. Otherwise advise **keep 30%**. If multiple
alternatives pass, report all and identify the one with highest full-window
after-tax terminal wealth under this fixed rule; do not use diagnostics to
rescue a failed gate. This is advice for a dated review only, not automatic
authorization to trade or an intra-year exit.

**A23 authority conflict, unresolved:** [`docs/next_steps.md`](../../next_steps.md)
currently permits only hold, rebalance to band, or step down to 100% SPY at
the annual review; it does **not** permit changing the SSO target to 15%,
45%, or 60%. The registered research rule above could recommend those
targets. Keep both texts intact. The owner must explicitly amend the A23
action set at the 2027 review before any such recommendation can be executed.
The “reduce only if” clause is a necessary extra condition, not an automatic
reduction when 30% ruins or trails 0%; if its intended priority differs, the
owner must settle it before approval.

### Senior-quant resolutions of the flagged items (approved with this brief)

- **Fee.** The base case uses SSO's current **0.84% net** expense ratio (ProShares page and
  summary prospectus, retrieved 2026-10-23), with **0.89%** as a conservative sensitivity. The
  financing spread is calibrated jointly against real SSO, so it absorbs any residual fee
  difference. Report the calibrated spread under both.
- **Distributions.** These come from free data. For the 1× leg, use the monthly S&P Composite
  dividend yield from Robert Shiller's public long-run dataset, which is free online. Record the
  retrieval URL, date and SHA-256 at freeze. Each year's dividends are taxed at the year-end
  qualified rate, and the remaining return is treated as price. For the 2× fund, keep A23's
  zero-distribution treatment, with SSO's real distributions reported as a 2006+ check. Current
  tax law is applied throughout and labelled hypothetical; historical rates are not modelled.
  Because the S&P dividend yield is applied to an all-US index, the result is an approximation;
  say so. A zero-distribution run for both legs is reported as a sensitivity only.
- **A23 action set.** Kept as written. The research rule produces a *proposal* for the 2027-09-30
  review. Changing the SSO target from 30% needs the owner to amend A23 at that review, and the
  study never triggers a sale by itself. The "reduce only if" clause is a condition for making
  that proposal, not an automatic reduction.

## 7. Calls for the owner to check and execution boundary

1. French all-US-market returns are **not** S&P 500 returns. The study's
   historical answer is a proxy for SPY/SSO, not an exact backtest of those
   funds, and the 1929 ETF is hypothetical with missing closure/margin risk.
2. The 2006-06-21 → 2026-04-30 calibration window, identifiable tracking-error
   loss, fixed 0.89% assumption, and transfer of its spread to earlier regimes
   need explicit approval. No calibration choice may use century outcomes.
3. The ±8-point band is fixed for all targets, even 15% and 60%; this follows
   A23's 30% rule shape but is a design choice, not a historical optimum.
4. The distribution-data gap and A23 action-set conflict are resolved above; any change in
   allocation remains an owner decision at the 2027 review.

**This task is this brief only.** After approval, new code may be added only
under `src/ggTrader/lab/` and `scripts/`, reusing the A23 tax and band logic.
No data archive freeze, return computation, result JSON, or test run is
authorized by this draft. Never touch `src/ggTrader/paper/`,
`src/ggTrader/data/live/`, `.env`, or Docker. No paid data at any stage.
