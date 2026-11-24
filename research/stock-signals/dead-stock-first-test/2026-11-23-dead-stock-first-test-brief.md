# Brief: Dead Stock First Test

**Status:** FROZEN 2026-11-23, before any return was computed. The rule, the numbers and the
two date windows below do not change. Any later change is a dated amendment at the bottom of
this file, never a silent edit.
**Study:** stock-signals/dead-stock-first-test (kind: screen), written 2026-11-23.
**Decision fed:** whether a plain, fixed momentum rule over liquid NYSE and Nasdaq stocks,
dead ones included, is worth taking to paper trading. The owner makes that decision.

## 0. Registered parameters

These lines are checked by the study tool. Every cited figure in the report must use these
values or say which line it departs from.

- eval_start: 2005-01-03
- eval_end: 2024-12-31
- max_leverage: 1.0
- benchmark: SPY buy-and-hold through the same ledger, same dates, same costs and same tax rates
- costs: 6.4, 18.3 and 40 bps per side (a basis point, bps, is 0.01 percent) of traded value on every buy and sell, dead-stock exits included; see section 4 for where 6.4 and 18.3 come from
- family_size: 1
- seed: none (nothing in this study is random)

`family_size` is 1 because there is one rule and no tuning. The three loss levels, three cost
levels and two tax sets are not choices to pick from. All of them are reported, and no single
combination is picked afterwards.

The overall span is two windows with a fixed split:

- **Development window:** 2005-01-03 to 2019-12-31. This is where the code is built and
  checked, and where the replay may be looked at while it is built. The rule itself is not
  changed by what is seen here.
- **Holdout window:** 2020-01-02 to 2024-12-31. Data the rule never saw while it was being
  built. It is run **once**, after the development window and the code are final. 2024-12-31
  is the last day of the CRSP file; the study reads nothing later.

(CRSP is the university stock-price database, loaded into our own database, that includes
stocks that were later delisted.)

## 1. Question and fixed arms

**Question:** Starting with 100,000 in a taxable account, does a fixed "buy the 50 stocks that
rose the most over the past year (skipping the last month)" rule, over every liquid NYSE and
Nasdaq stock that existed at the time (dead ones included), end ahead of SPY after trading
costs and taxes?

**What counts as a result:** A result is reported at every loss level and cost level beside
SPY after the same costs and taxes; no single combination is picked afterwards.

**The arms.** There is one rule (section 3). It is reported at every combination of:

- dead-stock loss 1.0, 0.5 and 0.0 (the share of its last price a holder loses when a stock
  stops trading; 1.0 means the holder gets nothing back, and is the headline because it is the
  cautious reading);
- cost per side 6.4, 18.3 and 40 bps;
- tax set low (24 percent short-term, 15 percent long-term) and high (35 percent short-term,
  20 percent long-term);
- universe A (the primary one) and, in Task 8 only, universe B (a stress test);
- both windows, development and holdout.

Every combination is shown. None is dropped.

## 2. Data and freeze before calculation

- **Prices:** the published combined release 1 (`load_bars(source="combined")` with
  `release_id=1`, pinned in every call and never "newest"). Every output JSON records
  `data_release` and `bars_sha256`. The release was built from CRSP daily stock prices
  (1968-01-02 to 2024-12-31, 25,754 stocks, loaded 2026-11-23) joined to Yahoo bars where a
  Yahoo history exists.
- **Who is eligible each month:** a table built from the raw CRSP rows (share code, exchange
  code, price and shares outstanding), using only what was known on the previous month-end.
  Stocks that later died are in it while they were alive.
- **SPY:** read through the same release, the same dates and the same ledger as the stocks.
- **Freeze date:** this brief is committed before any return is computed. The inputs are the
  release above and the repository commit that holds this brief.
- **Dead stocks:** the lab's own dead-stock exit (`ggTrader.lab.dead_stocks`): the position is
  sold on the stock's last bar, and no later order is placed in it.

## 3. Registered model

All values are fixed. No parameter is tuned.

| Item | Value |
|---|---|
| Rule | 12-1 momentum: score = close 21 trading days ago divided by close 252 trading days ago, minus 1 (`momentum_scores(close, lookback=252, skip=21)`); hold the 50 highest scores, equal weight; decide on the last trading day of each month, trade at the **next** trading day's close. No parameter is tuned. |
| Universe A (primary) | Common stock (`shrcd` in 10, 11) on NYSE or Nasdaq (`exchcd` in 1, 3), previous month-end absolute price at least $5, and previous month-end market value (price times shares) rank 1,000 or better. |
| Universe B (stress, Task 8) | Same, but price at least $1 and no rank limit. |
| Dead-stock loss | 1.0, 0.5, 0.0. A stock that stops trading is sold on its last bar at last price times (1 minus loss); price floor `SIM_PRICE_FLOOR`. Loss 1.0 is the headline, because it is the cautious reading. |
| Start cash | 100,000 |

**Lookahead guard:** the score and the eligibility test use data through the decision day
only; the trade happens at the next trading day's close; eligibility uses the previous
month-end.

## 4. Portfolio and accounting

| Item | Value |
|---|---|
| Costs | Per side 6.4, 18.3 and 40 bps of traded value, charged on every buy and sell, dead exits included. Basis for the numbers: see "Where the cost numbers come from" below. |
| Taxes | `TaxParams(st_rate=0.24, lt_rate=0.15)` (the rates already registered for the project's standalone studies) and a high set `TaxParams(0.35, 0.20)`. Tax is assessed each year-end and paid from the book's own cash. A dead-stock exit realises its loss. Both books (the rule and SPY) are fully sold on the last day and the tax paid. |
| Benchmark | SPY bought on the first day and sold on the last, through the same `TaxLotBook` ledger, costs and tax rates. |
| Leverage | None (1.0). |

### Where the cost numbers come from

The library supports two figures. They replace the 5 and 15 bps first written into the plan,
and the 40 stays as a stress level.

- **Large US companies: 6.4 bps.** Israel, Moskowitz, Ross and Serban (AQR working paper,
  "Implementing Momentum: What Have We Learned?", December 2017), Exhibit 2 ("Decomposing
  Trading Costs", AQR momentum strategies, average trading costs, 9 July 2009 to 31 December
  2016), page 5 of the library copy: "Overall Trading Costs (bps, per $ traded)" for the U.S.
  large-cap strategy is 6.40 (0.21 commissions, fees and taxes plus 6.19 market impact).
  Library: gg-library `documents`, id `3b61939e5b`, SHA-256 `3b61939e5b55cab8e336e39a53f6d3c6234c6b79429a8bdc57f0a7138f0b2189`,
  segment-offset locator segment 4, offset 1495, length 1500.
- **Small US companies: 18.3 bps.** Same exhibit and page: the U.S. small-cap strategy is
  18.27 (0.35 commissions, fees and taxes plus 17.91 market impact).
- **40 bps: an assumption, not a library figure.** It is a stress level, roughly twice the
  small-company number, to cover stocks that are cheaper or thinner than the AQR small-cap
  bucket (universe B reaches $1 stocks), the higher spreads before about 2009, and the cost of
  exiting a stock in trouble. The library holds no single figure for it. It is labelled an
  assumption.

How far the sources go, and what they leave out:

- Those two numbers are what a large professional trader actually paid for its own live
  orders, including its own price impact, over 2009 to 2016. They are not quotes from a retail
  broker, and they were earned by a firm that manages trading cost carefully; a small
  account would pay less price impact and could pay a wider spread.
- The study's window starts in 2005. Costs were higher before decimal pricing settled (a paper
  in the library, Chen and Velikov, "Zeroing in on the Expected Returns of Anomalies",
  Journal of Financial and Quantitative Analysis, DOI 10.1017/S0022109022000874, library page 9, states that trading costs "plummet
  in the 2000s as electronic trading and decimalization have improved liquidity"). The 40 bps
  level is the part of the grid that covers that period.
- Earlier library material that also fits the range (one-way spread cost of 4 to 7 bps for
  small stocks in the 2010 preprint "Intraday Patterns in the Cross-section of Stock Returns", arXiv 1005.3535, library page 15;
  and 25 bps one-way, half of a 50 bps large-cap spread, in Chaudhuri, Burnham and Lo's 2020
  tax-loss-harvesting manuscript, page 11, citing Corwin and Schultz 2012) is not used to set
  the grid, but shows the range 6 to 25 bps is not out of line.

## 5. Frozen output plan

All of these are written, with the raw numbers behind them in the results JSON (named in the
report), pinned to `data_release` and `bars_sha256`.

- **Main table:** for each of the three loss levels, three cost levels and two tax sets, in
  each window, the rule's after-tax final value of 100,000, per-year growth rate (CAGR),
  Sharpe ratio, largest peak-to-trough fall (maximum drawdown) and yearly turnover, side by
  side with SPY under the same costs and taxes. All 18 combinations per window are shown.
- **Gap to SPY:** the rule's after-tax per-year growth rate minus SPY's, for every combination.
- **Decomposition:** how much of any gap comes from dead-stock exits, from trading costs and
  from taxes.
- **Charts:** after-tax growth of 100,000 for the rule at loss 1.0, 0.5 and 0.0 against SPY, in
  each window; and the number of stocks held and the number of dead stocks held through time.
- **Stress (Task 8 only):** universe B at the same grid.
- **Sanity checks:** how many eligible stocks per month, how many dead exits, and the share of
  held stocks that have a Yahoo history against how many are CRSP only.
- **The holdout is looked at once.** Its numbers are produced by one final run after the code
  and the development numbers are final.

## 6. Decision rule

There is no pass bar and no GO/NO-GO label for this study. The old evidence bars no longer
apply to new work (owner rules reset 2026-11-20). The four plain rules apply instead:

1. Test only on data the strategy never saw while it was being tuned: no tuning at all here,
   and a holdout window that is run once.
2. Use only what was known on each day and include stocks that later died.
3. Subtract trading costs and taxes and compare to SPY over the same dates.
4. Paper trade first. The owner decides when a strategy has earned real money.

A result is reported at every loss level and cost level beside SPY after the same costs and
taxes; no single combination is picked afterwards. If the rule trails SPY at the headline
level (loss 1.0, 18.3 bps, low tax set), the report says so plainly. If the answer changes a
lot between loss levels, the report says that too, and that is a finding, not something to
smooth over.

## 7. What this cannot tell you, and the owner's checks

**What this cannot tell you.**

- **Real-world trading size limits are not tested.** The CRSP price file has no volume in the
  copy we hold, so the replay cannot check whether a 50-stock book of the stated size could
  actually be bought or sold at the stated cost. The size screen (price and company value)
  stands in for liquidity.
- **Dividend tax is not modelled.** The prices are total-return prices, so reinvested
  dividends are in the price and not taxed separately. That makes after-tax numbers for both
  the rule and SPY slightly rosy, in the same direction for both.
- **Margin and shorting are not modelled.** The book is long-only, no borrowing.
- **Stocks that left before 1968 are not in the data.**
- **Spreads wider than the cost grid are not modelled.** A stock that is in trouble can trade
  at a far worse price than 40 bps off its last price before it stops; the dead-stock loss
  level is the only place that appears.
- **What dying stocks actually pay out is unknown.** Our data records the last price, not what
  holders recovered; the three loss levels bracket the unknown rather than measure it.
- **One rule, one period.** This is one momentum rule over one span of years. It cannot tell
  you about other rules, and the holdout is only five years.

**The owner's checks before the run.** The owner approved the cost and tax grid by telling
the team to execute (2026-11-23). The cost numbers in section 4 differ from the first draft
(5 and 15) because the library figures are 6.4 and 18.3; the owner may change them before the
code is written, and the change would be recorded as a dated amendment.

**Execution boundary.** This task adds the study folder and this brief only. New study code
goes in the study's `code/` folder; nothing in `src/` is touched, vectorbt is not imported,
the paper trader stays stopped and the database is read only (through the read-only
connection) by later tasks.

## Amendments

**2026-11-23 (before any return was computed): how the code reads two sentences of section 3.**
- The market-value rank for universe A is taken among stocks that already pass the common-stock, NYSE/Nasdaq and $5 price tests that month (so "top 1,000" means the 1,000 largest eligible stocks, not the 1,000 largest of everything).
- A stock's eligibility on a day uses the last row of the previous calendar month; if that row has no price, the stock is not eligible (an older price is never substituted).
Neither changes any number in the registered parameters.

**2026-11-23 (after the first development run, before the holdout; the development numbers were looked at, the holdout was not). Changes and readings recorded so the one holdout run is interpreted as registered:**
- **File-end censoring.** The CRSP file ends 2024-12-31. A stock still alive on that day is not a death; it is sold at its last close like any other holding. (Without this a holdout stock alive at the file end would be written down.) Added in commit c81ed88b; the development numbers are identical with and without it (checked cell by cell).
- **Price-path diagnostic.** The CRSP file has no delisting codes, so a buy-out (holders paid) cannot be told from a failure. A descriptive block shows, for each dead stock the rule held, its price one month before its last day. It chooses nothing and changes no result cell. The three loss levels remain the registered answer; this block only helps read them. In the development window all 117 held dead stocks had a flat or rising price in their last month, which is the pattern of buy-outs, so loss 1.0 is a worst-case bound rather than an estimate for this universe.
- **Tax is paid in January.** The year's tax is assessed and paid from cash on the first trading day of the next year (and on the last day of the window for the final year), not on 31 December. A revised assessment (for example a wash sale) is paid the day it appears.
- **The lookbacks count each stock's own prices.** "21 and 252 trading days ago" are 21 and 252 valid closes of that stock (the registered `momentum_scores` helper), carried forward over short gaps.
- **A later spell is not a buy-out.** `has_successor` in the release means the same company number continues in a later CRSP spell (a data break such as a halt or exchange move). Real buy-outs end a company number and appear as deaths. Report field names say "later spell" for that case.
- **First month in cash.** The first decision day is the last trading day of the first month of the window, so the rule is in cash until the first trade at the next day's close while SPY is bought on the first day. Small, in SPY's favour.
- **Skipped buys.** A buy is skipped (cash kept, no substitute) when the stock has no price on the execution day or that day is its last bar. Skipping a stock on its last day uses the knowledge that it is about to end; this slightly favours the rule. The count is reported.
- **Two share classes** of one firm have different company numbers and can both be held.
- **Losses are not carried past the end.** Unused tax losses at the end of the window are not valued, for the rule and for SPY alike.
- **Reported statistics.** Sharpe ratio and largest fall are taken from the actual account value after taxes paid; the curve that adds paid taxes back is shown separately and labelled. `after_tax_final` is net of any tax still owed.
- **Holdout run conditions.** The holdout refuses to run unless release 1, CRSP load 1, a clean working tree and no Yahoo data drift. A metadata-only audit of false deaths (a Yahoo-linked series that ends while its company lives on, for example after a ticker rename) is run first; any hit is handled by a further dated amendment before the run. The output keeps month-end account curves, holdings counts, the Yahoo versus CRSP-only split of holdings, every cell's trade list in a side file, and a gap decomposition.
