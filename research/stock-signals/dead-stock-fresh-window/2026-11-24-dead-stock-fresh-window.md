# Dead-stock fresh window: the first test's rule, run once on 2000 to 2004

**Date:** 2026-11-24
**Study:** `research/stock-signals/dead-stock-fresh-window` (plan written down and frozen before any code: [2026-11-24-dead-stock-fresh-window-brief.md](2026-11-24-dead-stock-fresh-window-brief.md))
**Question:** Does the first test's fixed rule (each month, buy the 50 stocks with the strongest price rise over the past year, known as momentum), run on the qualifying NYSE and Nasdaq stocks including ones that later died, end ahead of SPY (the S&P 500 index fund) after trading costs and taxes over 2000 to 2004, five years it had never been run on?

## 1. In short

- **The rule trailed SPY at every one of the 18 settings, by a wide margin.** At the headline setting named in the plan before the run (18.3 bps cost per side, where a basis point, bps, is 0.01 percent of the traded value; low tax set; the holder of a dead stock loses everything) the rule lost 20.3 percent a year after cost and tax, against SPY's loss of 2.3 percent a year: a gap of **-18.0 points a year**. With the dead stocks assumed to pay their last price in full (the kindest reading) the gap is still **-7.3 points a year**.
- **This window's gap is larger than the averages of the two earlier windows, but the windows are not alike.** SPY itself lost 2.3 percent a year here, against gains of 8.1 and 12.5 percent a year before, so the gap here means the rule lost far more than SPY: 100,000 became 32,248 against SPY's 89,168. The 2005 to 2019 figure is a fifteen-year average that includes 2008, so a five-year stretch inside it could have been as bad. The deepest fall was about the same as in 2005 to 2019 (82.3 against 81.8 percent). The first test's range at the same cost and tax settings was -16.4 to -2.0 in 2005 to 2019 and -7.3 to +1.6 in 2020 to 2024; this window runs from -18.0 to -7.3.

| Window | Worst reading (dead stock pays nothing) | Middle (half) | Best (last price in full) |
|---|---:|---:|---:|
| 2000 to 2004 (this study, run once) | **-18.0** | -12.8 | -7.3 |
| 2005 to 2019 (first test, seen) | -16.4 | -8.8 | -2.0 |
| 2020 to 2024 (first test, seen) | -7.3 | -2.8 | +1.6 |

  Later, the payout rerun used company filings that show which held dead stocks were bought out and narrowed the worst reading to -7.2 (2005 to 2019) and -0.9 (2020 to 2024); the best readings stayed -2.0 and +1.6. No filings were gathered for 2000 to 2004, so this window's worst reading (-18.0) has no such correction and is not comparable with -7.2 or -0.9. The first test called its worst reading an outer limit because its dead stocks' prices looked like buy-outs; this study does not assume that for 2000 to 2004. The like-for-like column is the best reading, where every dead stock is paid its last price: -7.3 here, -2.0 and +1.6 before.

  Numbers are the rule's yearly growth after cost and tax minus SPY's, in percentage points a year, at 18.3 bps and the low tax set; a minus sign means behind SPY.
- **The rule trails SPY even when dead stocks are assumed to cost nothing.** With every dead stock paid its last price in full and no costs or taxes, the rule lost 7.9 percent a year against SPY's 2.2 percent, a gap of 5.7 points (this step mixes which stocks were chosen, equal weighting of 50 names, and the list of 1,000 stocks instead of the S&P 500; the study did not separate them). At the headline setting, where a dead stock pays nothing, dead stocks are the largest step: 10.9 points a year. Trading costs add 1.5. These shares depend on the order in which the steps are taken, and none of them says why the rule lost.
- **The fall was deep.** The rule's account lost 82 percent from a peak to a low at the headline setting (73.5 percent at the best reading), against 47.5 percent for SPY, and its Sharpe ratio (yearly return per unit of daily ups and downs; higher is better) was negative.
- **This is one five-year window of one market, the dot-com bust and the recovery after it.** It says how the rule did in those years. It does not say why, and it is not a forecast. The window is now used: a second run would not be fresh.
- **There is no go or no-go label.** Paper trading comes first and the owner decides.

## 2. What was done

- **The rule, universe, costs and taxes are the first test's, unchanged.** Each month-end, buy the 50 stocks that rose the most over the past year, skipping the last month (known as 12-1 momentum), equal weight, trading at the next day's close. The universe (called A) is common stock on the NYSE or Nasdaq priced at least $5 and among the 1,000 largest by company value, judged on the previous month-end. Start cash 100,000. Costs per side 6.4, 18.3 and 40 bps. Tax sets: low (24 percent short-term, 15 percent long-term) and high (35 percent, 20 percent). Dead stocks are sold on their last day at the last price times (1 minus the loss); the loss is 1.0, 0.5 or 0.0. SPY goes through the same bookkeeping of purchases, sales, costs and taxes.
- **The window.** 2000-01-03 to 2004-12-31, the first day our database holds SPY to the last day of 2004. About 1,000 stocks qualified on every day (978 to 1,000). 1,922 stock price histories were read, starting in August 1998 so that there is a year of prices before the first decision; 448 of them ended inside the window. The price source is the first published version (release 1) of the combined prices in our own database (the university stock-price file CRSP, joined to Yahoo prices where a Yahoo history exists), and the raw CRSP load 1 for the size and price tests.
- **Checks before the run.** (1) The code was first run on 2020 to 2024, the years the first test kept aside and ran once, and matched all 24 saved cells (18 for the rule, 6 for SPY) with no difference at all in four measures each (final value, yearly growth, deepest fall, Sharpe). The fresh run refuses to start unless that check passed on exactly the same code (the program's own refusal tests the final values to half a cent and the code's fingerprints; the exact match on all four measures was checked separately). (2) Before the run, a read-only look at counts and dates only (no returns) showed that SPY and the stocks start and end on the same days, SPY has no missing days, and the Yahoo prices had not changed since the release was published.
- **Checks in the run.** SPY with no costs matches its own price change exactly; the final value never rises as the dead-stock loss grows; no company was counted as two separate stocks at once; in every setting the trades add up to the final balance; no account value went to zero or below; no unpaid tax; the audit for stocks wrongly treated as dead (a Yahoo price history that ends while the company lives on) found none among 3,316 checked.
- **One run.** The run wrote a lock file recording that this window had been used before its own price read, so a second run is refused. An earlier availability check had read the inputs but reported only counts and dates, and computed no returns. The run took 11 minutes 52 seconds and 0.5 GB of memory. The rule traded about 9.7 times its account value a year (SPY, one purchase: 0.2). Raw numbers: [_fresh-A.json](_fresh-A.json); run notes: [_run_notes.txt](_run_notes.txt).

## 3. Results

Every combination is shown. None was picked afterwards. "Gap" is the rule's yearly growth after tax minus SPY's, in percentage points. The low and high tax sets give identical numbers here because both books ended with net losses and so paid no tax; the two tax sets test nothing in this window. A loss left over at the end is not valued, for the rule and for SPY alike (see section 4).

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

The bold row is the headline named in the plan. The rule trailed SPY in all 18 settings (the table shows the low tax half; the high tax half is identical).

**Where the gap comes from** (headline cost and low tax, points a year, from the study's own step-by-step breakdown, which adds up to the gap exactly):

| Step | Worst (1.0) | Middle (0.5) | Best (0.0) |
|---|---:|---:|---:|
| Stock picking (rule with dead stocks paid in full, no cost, no tax, against SPY with none) | -5.7 | -5.7 | -5.7 |
| Dead-stock losses | -10.9 | -5.6 | 0.0 |
| Trading costs | -1.5 | -1.6 | -1.7 |
| Taxes | 0.0 | 0.0 | 0.0 |

SPY's own cost and tax drag (0.07 points a year) closes the small remaining difference.

**What the dead stocks looked like.** At the headline setting the rule held 31 stocks on the day they stopped trading, from a total of 448 price histories that ended in the window; at the worst reading, writing them down removed 33,857 of account value over the five years. The study's descriptive block (never used in any result) looks at each stock's price in the month before its last day: of the 31, none had fallen to half its price or less, 3 had fallen by 10 to 16 percent, 20 were within 10 percent either way, and 8 had risen by 10 percent or more. Most of the 31 were flat or rising in their last month, which a buy-out would show, but a stock removed from an exchange for failure can show it too. In 2005 to 2019 all 117 held dead stocks were within 10 percent either way or had risen more than 10 percent; here 3 of the 31 fell 10 to 16 percent. No filing evidence was gathered for these years, so what holders received is unknown. The plan did not assume this window would look like the earlier ones.

**Comparison with the seen windows, in words.** At the headline setting the rule trailed SPY in all three windows (-18.0 here, -16.4 in 2005 to 2019, -7.3 in 2020 to 2024). Only in 2020 to 2024, at the best reading with a cost of 18.3 bps and the low tax set (+1.6), did it end ahead. Its deepest fall at the headline setting was 81.8 percent in 2005 to 2019, 54.1 percent in 2020 to 2024 and 82.3 percent here; SPY's was 55.2, 33.7 and 47.5 percent.

## 4. What this cannot tell you

- **One window, one market.** 2000 to 2004 is the end of the dot-com boom, the bust, and the recovery. It cannot tell you how the rule does in other conditions. The report did not look at which stocks the rule held or why they fell, so the cause of the loss in stock picking is not established.
- **The momentum idea was not unseen.** Earlier studies tested published momentum portfolios on other data covering these years. What is new here is this exact stock-level rule, universe and cost grid, run on stocks that later died.
- **Costs before 2001 were probably higher, and none of the three levels has been measured for these years.** Prices moved in fractions of a dollar until 2001 and spreads were wider. The two library figures (6.4 and 18.3 bps) are from 2009 to 2016 and 40 bps is an assumption. At 40 bps the gap is -19.6 at the worst reading, and actual costs in these years could have been higher still. The rule replaced about 9.7 times its account value each year, and each extra 0.1 percent per trade side cost about 0.75 points a year here.
- **What dying stocks paid is unknown.** The three readings bracket the unknown. Even the kindest, which assumes full payment of the last price, leaves the rule well behind SPY.
- **The rule was in cash in January 2000** (its first decision is at the end of that month) while SPY was bought on the first day and fell from 100,000 to about 95,800 in the month. That favours the rule by roughly 0.8 points a year, small beside -18.0. Its first decision was on 31 January 2000 and its first purchase on 1 February, near the market top: the account reached about 134,000 in February and the 82 percent fall is measured from there; a different start month would give different numbers. A buy is skipped when the stock has no price that day or the day is its last; the count was 6, the study does not split them by reason, and their effect on the result was not measured.
- **Unused tax losses are not counted.** Both accounts ended with net losses, so no tax was paid. The rule's loss (about 68,000) is much larger than SPY's (about 11,000). In a taxable account a loss like that can offset later gains, and a limited amount of other income, so its real worth depends on the owner's other income. Counting it would narrow the gap; the study counts it for neither.
- **Not modelled:** trade size limits (the CRSP copy has no volume), dividend tax, borrowing and shorting, and prices worse than the cost grid for a stock in trouble.
- **The window is used.** A second run on 2000 to 2004 would not be a fresh test.

## 5. Decision

There is no pass bar and no label. At the headline setting named in the plan the rule trailed SPY in all three windows it has now been run on, and in this window it trailed SPY at every setting. At the best reading it trailed by 7.3 points a year here and 2.0 in 2005 to 2019, and led by 1.6 in 2020 to 2024. Paper trading comes first and the owner decides.
