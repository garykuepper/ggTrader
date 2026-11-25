# Dead-stock fresh window: the first test's rule, run once on 2000 to 2004

**Date:** 2026-11-24
**Study:** `research/stock-signals/dead-stock-fresh-window` (registered plan frozen before any code: [2026-11-24-dead-stock-fresh-window-brief.md](2026-11-24-dead-stock-fresh-window-brief.md))
**Question:** Does the first test's fixed momentum rule, run on every liquid NYSE and Nasdaq stock including ones that later died, end ahead of SPY (the S&P 500 index fund) after trading costs and taxes over 2000 to 2004, five years it had never been run on?

## 1. In short

- **The rule trailed SPY at every one of the 18 settings, by a wide margin.** At the registered headline setting (18.3 bps cost per side, where a basis point, bps, is 0.01 percent of the traded value; low tax set; the holder of a dead stock loses everything) the rule lost 20.3 percent a year after cost and tax, against SPY's loss of 2.3 percent a year: a gap of **-18.0 points a year**. With the dead stocks assumed to pay their last price in full (the kindest reading) the gap is still **-7.3 points a year**.
- **The result is worse than anything seen before.** The first test's range at the same cost and tax settings was -16.4 to -2.0 in 2005 to 2019 and -7.3 to +1.6 in 2020 to 2024. This window runs from -18.0 to -7.3.

| Window | Worst reading (dead stock pays nothing) | Middle (half) | Best (last price in full) |
|---|---:|---:|---:|
| 2000 to 2004 (this study, run once) | **-18.0** | -12.8 | -7.3 |
| 2005 to 2019 (first test, seen) | -16.4 | -8.8 | -2.0 |
| 2020 to 2024 (first test, seen) | -7.3 | -2.8 | +1.6 |

  Numbers are the rule's yearly growth after cost and tax minus SPY's, in percentage points a year, at 18.3 bps and the low tax set; a minus sign means behind SPY.
- **Dead stocks are not the main cause here.** Even if every dead stock paid its last price in full and there were no costs and no taxes, the rule lost 7.9 percent a year against SPY's 2.2 percent: stock picking alone cost 5.7 points a year. Dead stocks added 10.9 points a year of loss at the worst reading, and trading costs 1.5.
- **The fall was deep.** The rule's account lost 82 percent from a peak to a low at the headline setting (73.5 percent at the best reading), against 47.5 percent for SPY, and its Sharpe ratio (yearly return per unit of daily ups and downs; higher is better) was negative.
- **This is one five-year window of one market, the dot-com bust and the recovery after it.** It says how the rule did in those years. It does not say why, and it is not a forecast. The window is now used: a second run would not be fresh.
- **There is no go or no-go label.** Paper trading comes first and the owner decides.

## 2. What was done

- **The rule, universe, costs and taxes are the first test's, unchanged.** Each month-end, buy the 50 stocks that rose the most over the past year, skipping the last month (known as 12-1 momentum), equal weight, trading at the next day's close. The universe (called A) is common stock on the NYSE or Nasdaq priced at least $5 and among the 1,000 largest by company value, judged on the previous month-end. Start cash 100,000. Costs per side 6.4, 18.3 and 40 bps. Tax sets: low (24 percent short-term, 15 percent long-term) and high (35 percent, 20 percent). Dead stocks are sold on their last day at the last price times (1 minus the loss); the loss is 1.0, 0.5 or 0.0. SPY goes through the same ledger.
- **The window.** 2000-01-03 to 2004-12-31, the first day our database holds SPY to the last day of 2004. About 1,000 stocks qualified on every day (978 to 1,000). 1,922 stock series were read, from August 1998 for the lookback; 448 of them ended inside the window. The price source is the published combined release 1 of our own database (the university stock-price file CRSP, joined to Yahoo prices where a Yahoo history exists), and the raw CRSP load 1 for the size and price tests.
- **Checks before the run.** (1) The code was first run on 2020 to 2024, the first test's holdout, and matched all 24 saved cells (18 for the rule, 6 for SPY) with no difference at all in four measures each (final value, yearly growth, deepest fall, Sharpe). The fresh run refuses to start unless that check passed on exactly the same code. (2) Before the run, a read-only look at counts and dates only (no returns) showed that SPY and the stocks start and end on the same days, SPY has no missing days, and there was no Yahoo price drift.
- **Checks in the run.** SPY with no costs matches its own price change exactly; the final value never rises as the dead-stock loss grows; no company was held through two columns at once; every cell's trade ledger reconciles; no account value went to zero or below; no unpaid tax; the audit for stocks wrongly treated as dead (a Yahoo series that ends while the company lives on) found none among 3,316 checked.
- **One run.** The window was claimed before any 2000 to 2004 data was read. The run took 11 minutes 52 seconds and 0.5 GB of memory. Raw numbers: [_fresh-A.json](_fresh-A.json); run notes: [_run_notes.txt](_run_notes.txt).

## 3. Results

Every combination is shown. None was picked afterwards. "Gap" is the rule's yearly growth after tax minus SPY's, in percentage points. The low and high tax sets give identical numbers here because both books ended with net losses and so paid no tax (a loss left over at the end is not valued, for the rule and for SPY alike).

| Dead-stock reading | Cost per side | Rule ends at | Rule per year | SPY ends at | SPY per year | Gap, points a year | Deepest fall, rule / SPY | Sharpe, rule / SPY |
|---|---|---:|---:|---:|---:|---:|---|---|
| 1.0 (nothing back) | 6.4 bps | 34,199 | -19.3% | 89,381 | -2.2% | -17.1 | -81.7% / -47.5% | -0.35 / -0.00 |
| 1.0 (nothing back) | **18.3 bps** | **32,248** | **-20.3%** | **89,168** | **-2.3%** | **-18.0** | -82.3% / -47.5% | -0.38 / -0.01 |
| 1.0 (nothing back) | 40 bps | 28,978 | -22.0% | 88,782 | -2.4% | -19.6 | -83.5% / -47.5% | -0.43 / -0.01 |
| 0.5 (half back) | 6.4 bps | 46,987 | -14.0% | 89,381 | -2.2% | -11.8 | -77.6% / -47.5% | -0.19 / -0.00 |
| 0.5 (half back) | 18.3 bps | 44,305 | -15.0% | 89,168 | -2.3% | -12.8 | -78.3% / -47.5% | -0.22 / -0.01 |
| 0.5 (half back) | 40 bps | 39,808 | -16.8% | 88,782 | -2.4% | -14.5 | -79.5% / -47.5% | -0.27 / -0.01 |
| 0.0 (last price in full) | 6.4 bps | 64,200 | -8.5% | 89,381 | -2.2% | -6.3 | -72.6% / -47.5% | -0.03 / -0.00 |
| 0.0 (last price in full) | 18.3 bps | 60,531 | -9.6% | 89,168 | -2.3% | -7.3 | -73.5% / -47.5% | -0.06 / -0.01 |
| 0.0 (last price in full) | 40 bps | 54,381 | -11.5% | 88,782 | -2.4% | -9.1 | -75.0% / -47.5% | -0.12 / -0.01 |

The bold row is the registered headline. The rule trailed SPY in all 18 settings (the table shows the low tax half; the high tax half is identical).

**Where the gap comes from** (headline cost and low tax, points a year, from the study's own step-by-step breakdown, which adds up to the gap exactly):

| Step | Worst (1.0) | Middle (0.5) | Best (0.0) |
|---|---:|---:|---:|
| Stock picking (rule with dead stocks paid in full, no cost, no tax, against SPY with none) | -5.7 | -5.7 | -5.7 |
| Dead-stock losses | -10.9 | -5.6 | 0.0 |
| Trading costs | -1.5 | -1.6 | -1.7 |
| Taxes | 0.0 | 0.0 | 0.0 |

SPY's own cost and tax drag (0.07 points a year) closes the small remaining difference.

**What the dead stocks looked like.** At the headline setting the rule held 31 stocks on the day they stopped trading, from a total of 448 series that ended in the window; at the worst reading, writing them down removed 33,857 of account value over the five years. The study's descriptive block (never used in any result) looks at each stock's price in the month before its last day: of the 31, none had fallen to half its price or less, 3 had fallen by 10 to 16 percent, 20 were within 10 percent either way, and 8 had risen by 10 percent or more. That is close to the pattern the first test saw in 2005 to 2019 (all 117 held dead stocks flat or rising in the last month), and it is the pattern of a buy-out, but prices alone cannot tell a buy-out from a failure and no filing evidence was gathered for these years, so what holders received is unknown. The brief did not assume this window would look different or the same.

**Comparison with the seen windows, in words.** At the headline setting the rule trailed SPY in all three windows (-18.0 here, -16.4 in 2005 to 2019, -7.3 in 2020 to 2024). Only in 2020 to 2024, at the best reading (+1.6), did it end ahead. Its deepest fall at the headline setting was 81.8 percent in 2005 to 2019, 54.1 percent in 2020 to 2024 and 82.3 percent here; SPY's was 55.2, 33.7 and 47.5 percent.

## 4. What this cannot tell you

- **One window, one market.** 2000 to 2004 is the end of the dot-com boom, the bust, and the recovery. It cannot tell you how the rule does in other conditions. The report did not look at which stocks the rule held or why they fell, so the cause of the loss in stock picking is not established.
- **The momentum idea was not unseen.** Earlier studies tested published momentum portfolios on other data covering these years. What is new here is this exact stock-level rule, universe and cost grid, run on stocks that later died.
- **Costs before 2001 were probably higher, and none of the three levels has been measured for these years.** Prices moved in fractions of a dollar and spreads were wider. The two library figures (6.4 and 18.3 bps) are from 2009 to 2016 and 40 bps is an assumption. Costs at the high end would make the gap larger: -19.6 at the worst reading.
- **What dying stocks paid is unknown.** The three readings bracket the unknown. Even the kindest, which assumes full payment of the last price, leaves the rule well behind SPY.
- **The rule was in cash in January 2000** (its first decision is at the end of that month) while SPY was bought on the first day and fell about 4 percent in the month. That small difference favours the rule. A buy is skipped when the stock has no price that day or the day is its last, which slightly favours the rule too; the count was 6.
- **Not modelled:** trade size limits (the CRSP copy has no volume), dividend tax, borrowing and shorting, and prices worse than the cost grid for a stock in trouble.
- **The window is used.** A second run on 2000 to 2004 would not be a fresh test.

## 5. Decision

There is no pass bar and no label. At the registered headline setting the rule trailed SPY in all three windows it has now been run on, and in this window it trailed SPY at every setting. Paper trading comes first and the owner decides.
