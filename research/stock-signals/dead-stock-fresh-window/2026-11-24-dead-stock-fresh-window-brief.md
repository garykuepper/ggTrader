# Brief: Dead Stock Fresh Window

**Status:** FROZEN 2026-11-24, before any return is computed for the fresh window. The rule, the numbers and the two date windows below do not change. Any later change is a dated amendment at the bottom of this file, never a silent edit.
**Study:** stock-signals/dead-stock-fresh-window (kind: screen), written 2026-11-24.
**Decision fed:** whether the first test's plain momentum rule still looks worth taking to paper trading after one more look on years it has never been run on. The owner makes that decision. There is no pass bar and no GO/NO-GO label.

## 0. Registered parameters

These lines are checked by the study tool. Every cited figure in the report must use these values or say which line it departs from.

- eval_start: 2000-01-03
- eval_end: 2004-12-31
- max_leverage: 1.0
- benchmark: SPY buy-and-hold through the same ledger, same dates, same costs and same tax rates
- costs: 6.4, 18.3 and 40 bps per side (a basis point, bps, is 0.01 percent) of traded value on every buy and sell, dead-stock exits included; the figures and where they come from are in section 4 of the first test's brief
- family_size: 1
- seed: none (nothing in this study is random)

`family_size` is 1 because there is one rule and no tuning. The three loss levels, three cost levels and two tax sets are all reported and none is picked afterwards.

**The two windows.**

- **Fresh window:** 2000-01-03 to 2004-12-31. The rule has never been run on it. It is run **once**.
- **Check window:** 2020-01-02 to 2024-12-31. This is the first test's holdout, already seen. It is run only to prove that the code in this study reproduces the first test's saved numbers to the cent. It is a check on the code, not a test of the rule.

2000-01-03 is the first day our database holds SPY. A window that starts earlier cannot be compared with SPY, so 1995 to 1999 is left out.

## 1. Question and fixed arms

**Question:** Starting with 100,000 in a taxable account, does a fixed "buy the 50 stocks that rose the most over the past year (skipping the last month)" rule, over every liquid NYSE and Nasdaq stock that existed at the time (dead ones included), end ahead of SPY after trading costs and taxes over 2000 to 2004?

**The arms.** Exactly the first test's, on universe A only: dead-stock loss 1.0, 0.5 and 0.0 (the share of its last price a holder loses when a stock stops trading); cost 6.4, 18.3 and 40 bps per side; tax set low (24 percent short-term, 15 percent long-term) and high (35 percent, 20 percent). That is 18 strategy cells and 6 SPY cells. The headline cell is loss 1.0, 18.3 bps, low tax, the same cell the first test named. Every cell is shown; no cell is dropped or chosen after the run.

**Not in this study:** universe B, any tuning, any new parameter, and any SEC filing evidence about what dying stocks paid (the filing lists for 1999 to 2004 have not been downloaded; only the three loss readings are used).

## 2. Data and freeze before calculation

- **Prices:** published combined release 1, pinned (`--release-id 1`), with the raw CRSP load 1 for eligibility (`--crsp-load 1`). CRSP is the university stock-price database, loaded into our own database, that includes stocks that were later delisted. Every output records the data release and the price hash.
- **SPY:** the same release, dates and ledger as the stocks. Its first row is 2000-01-03.
- **Dead stocks:** the lab's own dead-stock exit: the position is sold on the stock's last bar at last price times (1 minus loss), and no later order is placed in it.
- **Freeze date:** this brief is committed before any code of this study exists. The code is the first test's engine files, copied with a prefix and checked against the originals by hash (the study tool forbids two studies defining the same module name).
- **The check comes first.** The check window is run before the fresh window. The fresh run is refused unless the check reproduced the first test's holdout cells and unless the code that passed the check is the code that runs the fresh window.
- **Run once.** The fresh run claims its window before reading any fresh data. A second run, or a run after a crash that left the claim, is refused; only the owner can decide what to do then.

## 3. Registered model

The first test's model, unchanged: 12-1 momentum (score = close 21 trading days ago divided by close 252 trading days ago, minus 1); hold the 50 highest scores, equal weight; decide on the last trading day of each month, trade at the next trading day's close; universe A = common stock (`shrcd` 10 or 11) on NYSE or Nasdaq (`exchcd` 1 or 3), previous month-end absolute price at least $5 and previous month-end market value (price times shares) among the 1,000 largest of the stocks passing the other tests; start cash 100,000. The first test's two dated readings of those sentences (the rank taken among stocks already passing, and no older price substituted) apply. The score and the eligibility test use data through the decision day only.

## 4. Portfolio and accounting

Exactly the first test's section 4 and its dated amendments: costs per side on every buy and sell; taxes assessed each year and paid in January from the book's own cash, a dead-stock exit realises its loss, both books fully sold on the last day; SPY through the same tax-lot ledger; no leverage; file-end handling (not reached here, the window ends in 2004).

## 5. Frozen output plan

Written to the results file, with the data release and price hash:

- All 18 strategy cells and 6 SPY cells: after-tax final value, per-year growth rate (CAGR), Sharpe ratio, largest peak-to-trough fall, yearly turnover.
- The gap to SPY for every cell: growth-rate gap in points a year, fall gap, Sharpe gap.
- The headline: gap at 18.3 bps and low tax for loss 1.0 / 0.5 / 0.0 (worst / middle / best reading).
- The first test's coverage and sanity blocks, the gap decomposition and the dead-stock price-path block (each dead stock's price one month before its last day), passed through unchanged.
- The report compares this window's range with the ranges already seen (2005 to 2019 and 2020 to 2024, in the first test and rerun reports), in words.

## 6. Decision rule

None, as in the first test: the owner's four plain rules apply (test only on data the rule never saw; use only what was known on each day and include stocks that later died; subtract costs and taxes and compare to SPY over the same dates; paper trade first, the owner decides). If the rule trails SPY at the headline cell the report says so plainly. If the answer changes a lot between loss levels, that is reported as a finding.

## 7. What this cannot tell you, and the owner's checks

- **One five-year window, one market.** 2000 to 2004 is the dot-com bust and the recovery after it. It cannot tell you how the rule does in other conditions.
- **The momentum idea is not unseen.** Earlier studies tested published momentum portfolios on other data covering these years. What is new is this exact stock-level rule, universe and cost grid, run on stocks that later died.
- **Costs before 2001 were higher.** Prices moved in fractions of a dollar and spreads were wider. The 40 bps column is the relevant reading for these years; none of the three is chosen afterwards.
- **What dying stocks paid is unknown, and in these years more of them were bankruptcies.** The first test's description of its held dead stocks as mostly buy-outs, which made losing everything a worst case, was measured on 2005 to 2019 and is not carried over. The report shows this window's own price-path block and says what it does and does not show.
- **No filing evidence for these deaths.** The loss readings bracket the unknown rather than measure it.
- **Size limits are not tested.** The CRSP copy has no volume. Dividend tax is not modelled. Margin and shorting are not modelled. A stock in trouble can trade at a far worse price than the cost grid.
- **The window is used up by one run.** A second run would not be fresh.

**Execution boundary.** New study code goes in this study's `code/` folder; nothing in `src/` is touched, vectorbt is not imported, the paper trader stays stopped and the database is read only.

## Amendments

None.
