# Early decades: the first test's rule and two variants, run once on 1969 to 1998

**Date:** 2026-11-28
**Study:** `research/stock-signals/early-decades` (plan written down and committed before any scoring: [2026-11-27-early-decades-brief.md](2026-11-27-early-decades-brief.md))
**Question:** In the only stock-level years nobody has scored (1969-02-03 to 1998-07-31), does buying recent winners, each month or once a year, end ahead of the market after trading costs and taxes, with stocks that later died included?
**Dates pinned:** `--eval-start 1969-02-03 --eval-end 1998-07-31` on every result below. Raw results: `_early-A.json`, `_early-B.json`, `_early-C.json` in this folder. Code version `a21e4650`, rule hash `a5698ce0c29e…`, combined price release 1, raw CRSP load 1.

## 1. In short

- **It depends almost entirely on what a dead stock is worth.** When every stock that stopped trading is assumed to pay out its last price in full (the "0.0" reading, the one this study's plan named as its headline; the first test had registered 1.0, nothing back, as its headline and later read 0.0 as the closest to what takeovers pay, from prices alone), all three versions finish ahead of the comparison. When half of a dead stock's last value is lost, only one setting comes close, and when it is all lost, every setting is far behind. The data hold no final return at death, so the truth lies somewhere in that range and this study cannot narrow it.
- **The comparison is the whole market, not a fund, for most of these years.** From 1969-02-03 to 1993-01-29 the comparison is the CRSP value-weighted market return (all stocks counted by company size; a stand-in, not something you could buy). From 1993-02-01 to 1998-07-31 it is SPY (the S&P 500 index fund). It grew $100,000 into $2,508,055 after one final sale and tax, about 11.5% a year. The owner agreed this choice on 2026-11-27, before scoring.
- **Ahead before 1993 at 7 of 18 settings; behind after at all 18.** In 1969 to early 1993 the winning-stocks rule beat the market at 7 of its 18 settings (all six with dead stocks paid in full, plus one with half lost). In February 1993 to July 1998, when the comparison is SPY, it was behind at all 18 settings (gaps from -0.4 to -13.4 points a year). This split depends on how tax is counted at the 1993 boundary; on a continuous-account basis the count is 6 ahead before 1993 and 2 ahead after (section 3).
- **At the headline setting the numbers are large.** With a trading cost of 0.183% per purchase or sale (a basis point, bps, is 0.01%; so 18.3 bps), the 24% / 15% tax rates and dead stocks paid in full, the monthly rule (A) ended at $5,607,914 (14.6% a year), the yearly version (B) at $3,275,294 (12.6%) and the version that skips stocks with a recent 20% one-day jump (C) at $6,831,100 (15.4%). The comparison ended at $2,508,055 (11.5%).
- **The result is fragile in three ways.** (1) The dead-stock reading moves a setting by 7 to 17 points a year. (2) A higher trading cost of 0.5% per side cuts the monthly rule by about 2.5 points a year. (3) There are 18 settings here, three arms times two costs times three dead-stock readings, so a good-looking one is likelier to appear by luck.
- **The falls were deep.** The worst peak-to-low drops ranged from -47% to -94% depending on setting, against -48% for the comparison. At the headline setting the monthly rule fell -49%, slightly deeper than the comparison.
- **There is no go or no-go label.** Paper trading stays stopped. The window is now used: a second run would not be fresh.

## 2. What was done

- **The rule is the first test's, unchanged.** Each month-end, buy the 50 stocks with the strongest price rise over the past year, skipping the latest month (12-1 momentum), equal weight, from the 1,000 largest eligible stocks (universe A: NYSE and Nasdaq common stock, priced at least $5, with a market value). It uses only information known on each date and keeps stocks that later died. Average eligible stocks per month were about 988 to 998 in every year.
- **Three arms, all reported.** A is the first test's monthly rule (the control). B rebalances once a year: it trades only when the next trade date is more than one calendar year after the last one, so gains can qualify for the lower long-term tax rate, and the rate depends on the real holding time of each purchase. C trades monthly but skips any stock that rose 20% or more in one day during the prior 252 trading days, using only earlier data.
- **Costs and taxes.** 0.183% per purchase or sale (from the first test), and 0.5% per side as a second, higher cost reading because trading in these decades cost more. Tax is 24% on gains held a year or less and 15% on longer ones. The account ends with a full sale and the last tax payment.
- **Dead stocks.** Three readings, as in the first test: 0.0 (the dead stock sells at its last price, nothing lost), 0.5 (half of that value lost) and 1.0 (all of it lost).
- **The comparison.** The CRSP (university stock-price database) market return up to the 1993-01-29 close, then SPY from the 1993-02-01 return, bought at the start and held with no trading costs, taxed once on the final sale at 15%. Its risk figures use its own daily curve, which has no tax until the final sale; the strategies' risk figures use their real account after tax paid.
- **Checks before scoring.** The same code was first run on 2020-01-02 to 2024-12-31 for arm A and reproduced all 18 saved cells from the first test's holdout exactly (I compared them again by hand after the run). Linked-price drift checks and a clean-code check ran before each arm's lock file was written. The registration was committed before scoring.
- **Checks in the run.** In every setting the trade chain adds up to the final balance, no account went to zero or below, no tax was left unpaid, no company was counted as two stocks at once, and the final value never rises as the dead-stock loss grows.
- **The run.** Arms A, B and C ran one after another, 2026-11-28 05:30 to 06:02, under the 6 GB memory cap, one run each. Lock files `_early-{A,B,C}.claim` stay in place.

## 3. Results

Every setting is shown; none was picked afterwards. "Gap" is the account's yearly growth after cost and tax minus the comparison's 11.5%, in percentage points. Per-year numbers are compounded growth over the 29.5 years. Deepest fall and Sharpe ratio (yearly return per unit of daily ups and downs; higher is better) are for each account after the tax it actually paid along the way; the comparison's two figures come from its daily curve before the tax on its final sale, so they are not cut by that last tax the way the strategies' last day is (that tax would take the comparison's Sharpe from 0.90 to about 0.85 and leave its deepest fall unchanged). The lower cost is 0.183% per side, the higher 0.5%.

| Dead-stock reading | Cost per side | Arm | Ends at ($100,000 start) | Per year | Gap, points a year | Deepest fall | Sharpe |
|---|---|---|---:|---:|---:|---:|---:|
| (comparison) | none | Market, then SPY | 2,508,055 | 11.5% | 0.0 | -48.1% | 0.90 |
| 0.0 (last price in full) | 0.183% | A monthly | 5,607,914 | 14.6% | +3.1 | -49.4% | 0.75 |
| 0.0 | 0.183% | B yearly | 3,275,294 | 12.6% | +1.0 | -54.3% | 0.69 |
| 0.0 | 0.183% | C skip jumpers | 6,831,100 | 15.4% | +3.9 | -46.6% | 0.79 |
| 0.0 | 0.5% | A monthly | 2,878,793 | 12.1% | +0.5 | -52.3% | 0.65 |
| 0.0 | 0.5% | B yearly | 2,833,071 | 12.0% | +0.5 | -55.0% | 0.67 |
| 0.0 | 0.5% | C skip jumpers | 3,505,664 | 12.8% | +1.3 | -49.4% | 0.68 |
| 0.5 (half lost) | 0.183% | A monthly | 605,944 | 6.3% | -5.2 | -67.8% | 0.40 |
| 0.5 | 0.183% | B yearly | 1,315,533 | 9.1% | -2.4 | -55.3% | 0.53 |
| 0.5 | 0.183% | C skip jumpers | 2,162,227 | 11.0% | -0.6 | -50.9% | 0.60 |
| 0.5 | 0.5% | A monthly | 311,427 | 3.9% | -7.6 | -73.5% | 0.29 |
| 0.5 | 0.5% | B yearly | 1,136,359 | 8.6% | -3.0 | -56.1% | 0.51 |
| 0.5 | 0.5% | C skip jumpers | 1,112,219 | 8.5% | -3.0 | -55.0% | 0.50 |
| 1.0 (nothing back) | 0.183% | A monthly | 53,462 | -2.1% | -13.6 | -91.3% | 0.01 |
| 1.0 | 0.183% | B yearly | 480,224 | 5.5% | -6.1 | -60.8% | 0.36 |
| 1.0 | 0.183% | C skip jumpers | 663,779 | 6.6% | -4.9 | -64.4% | 0.41 |
| 1.0 | 0.5% | A monthly | 24,245 | -4.7% | -16.2 | -94.5% | -0.11 |
| 1.0 | 0.5% | B yearly | 415,117 | 4.9% | -6.6 | -61.7% | 0.33 |
| 1.0 | 0.5% | C skip jumpers | 339,389 | 4.2% | -7.3 | -69.4% | 0.30 |

### The two sub-periods

The strategy account is one continuous account across the whole window, with taxes paid as they fell due; its split at 1993-01-29 is a reading of that account, not two separately sold accounts. The comparison assumes a sale at each period end. So a sub-period reading of the strategy can include gains not yet sold and the tax on them not yet paid, while the comparison has already paid tax on its gains at that point.

Comparison, as registered: 9.5% a year in 1969-02 to 1993-01 (growth 8.88 times after tax, deepest fall -48.1%) and 18.8% a year in 1993-02 to 1998-07 (growth 2.58 times, deepest fall -11.2%).

| Dead-stock reading | Cost | Arm | 1969-02 to 1993-01: per year | Gap | 1993-02 to 1998-07: per year | Gap |
|---|---|---|---:|---:|---:|---:|
| 0.0 | 0.183% | A monthly | 13.9% | +4.4 | 17.7% | -1.0 |
| 0.0 | 0.183% | B yearly | 11.6% | +2.0 | 16.9% | -1.8 |
| 0.0 | 0.183% | C skip jumpers | 14.7% | +5.2 | 18.3% | -0.4 |
| 0.0 | 0.5% | A monthly | 11.4% | +1.9 | 15.1% | -3.7 |
| 0.0 | 0.5% | B yearly | 11.0% | +1.5 | 16.4% | -2.4 |
| 0.0 | 0.5% | C skip jumpers | 12.2% | +2.6 | 15.7% | -3.1 |
| 0.5 | 0.183% | A monthly | 5.1% | -4.4 | 11.8% | -7.0 |
| 0.5 | 0.183% | B yearly | 8.0% | -1.5 | 14.1% | -4.7 |
| 0.5 | 0.183% | C skip jumpers | 10.1% | +0.6 | 14.9% | -3.9 |
| 0.5 | 0.5% | A monthly | 2.6% | -6.9 | 10.0% | -8.8 |
| 0.5 | 0.5% | B yearly | 7.5% | -2.1 | 13.6% | -5.2 |
| 0.5 | 0.5% | C skip jumpers | 7.7% | -1.9 | 12.3% | -6.5 |
| 1.0 | 0.183% | A monthly | -4.4% | -13.9 | 8.4% | -10.4 |
| 1.0 | 0.183% | B yearly | 4.2% | -5.4 | 11.2% | -7.5 |
| 1.0 | 0.183% | C skip jumpers | 5.5% | -4.0 | 11.5% | -7.3 |
| 1.0 | 0.5% | A monthly | -6.9% | -16.4 | 5.4% | -13.4 |
| 1.0 | 0.5% | B yearly | 3.7% | -5.9 | 10.7% | -8.1 |
| 1.0 | 0.5% | C skip jumpers | 3.2% | -6.4 | 9.1% | -9.7 |

**Continuous-account check, added after the run.** The registered comparison is sold and taxed at 15% at the end of each sub-period, so its first period is shown after tax (9.5% a year) and its second restarts with a fresh account (18.8% a year). The strategy's 1993 value carries no tax on gains not yet sold, and its later period pays the deferred tax at the final sale. That helps the strategy before 1993 and hurts it after. Holding the comparison as one continuous account, untaxed at the 1993 split and taxed once at the end, gives 10.2% a year before 1993 and 17.6% a year after (the full window is unchanged at 11.5%). This check is not perfectly even either: the monthly arms have paid nearly all their pre-1993 tax by the split (the yearly arm still held about four months of unsold gains), while the comparison's whole 24-year tax is pushed into the 5.5 later years, so the two readings probably bracket the truth from opposite sides. On that basis the rule is ahead before 1993 at 6 of 18 settings (C with half lost drops out, at -0.1 points) and ahead after at 2 of 18 (A and C at dead stocks paid in full and the lower cost, +0.1 and +0.7 points). The registered reading above stays the formal result; this check changes how clean the line at 1993 looks, not the full-window result.

**The comparison starts about a month before the rule.** The comparison is invested from the 1969-02-03 close. The rule's first purchases are on 1969-03-03, and the comparison had fallen about 5% by then (to 0.948 of its start), so the rule starts that far ahead of a holder who bought on day one. That is worth about 0.2 points a year over the window. The registration fixes this start.

**Rounding.** A gap is computed from unrounded growth, so it can differ by 0.1 from the rounded per-year figures beside it.

### What the arms did

- **Yearly trading (B) cuts costs and shifts gains to the lower tax rate, but earns less when dead stocks are free.** At the headline setting B made 2,938 trades against A's 24,269, paid $71,218 in costs against $549,936, and realized $3.4 million of long-term gains against $2.1 million for A. It also held far fewer dead stocks at their death (102 exits against 269). B's yearly spacing means more than a calendar year between trades, so its real gaps were about 13 months, and it still realized $360,821 of short-term gains at the headline setting, mostly from the forced sale of everything on the last day (about 55%) and from dead-stock exits (about 44%); January tax-funding sales were negligible. The gains shown are sale-time gains before later wash-sale adjustments (a tax rule that disallows a loss when the same stock is bought back soon after). B falls least as the dead-stock reading worsens (about 7 points from full payout to nothing back, against 9 for C and 17 for A), ending at 4.9% to 5.5% a year even when nothing is recovered; C still ends higher than B at that reading, at 4.2% to 6.6%.
- **Skipping 20% jumpers (C) did slightly better than A at the optimistic setting and clearly better at the half-lost one.** C ended with 142 dead exits (stocks that stopped trading while held) against A's 269. The saved results do not count how many stocks the 20% filter itself excluded; their "skipped buys" field (C 36, A 70, B 8) counts targets that could not be bought because they had no price that day or their price history had ended. A bounce between quoted bid and ask midpoints can falsely trigger C's filter, and C bridges up to ten missing sessions when comparing prices. This agrees in direction with the earlier momentum-deaths study, which found that dying picks tended to have had 20% jumps, but this study cannot say why.
- **The tax bill was large at the optimistic setting.** At the headline setting A paid $1,503,185 in tax and $549,936 in costs, about 7.5 times its holdings traded each year.

## 4. What this cannot tell you

- **The dead-stock reading is the study.** The data have no final return at death. In these decades many deaths were takeovers (likely near the last price) and others were bankruptcies (likely near zero). A later rerun of the first test used SEC filings to find which held dead stocks were bought out, for 2005 to 2024; it did not identify failures or measure payouts, and no filings were gathered for 1969 to 1998. A setting that wins only at 0.0 is a setting that assumes no dead stock ever lost anything.
- **The comparison is not what a holder could have bought for most of these years.** The market return for 1969 to early 1993 is a stand-in. It weights companies by size, while the rule holds 50 stocks equally weighted from the 1,000 largest. Part of any gap may be that size tilt and not the winners rule. A like-for-like comparison, such as an equal-weighted version of all 1,000 stocks, was not part of the registered plan, and scoring it now would reuse this window.
- **Prices before the 1990s are weaker.** Many early prices are quote midpoints, the middle of the quoted bid and ask, not trades. There is no trading volume, so the study cannot say how many shares could have traded, and only about 1,203 NYSE common stocks qualified on 1969-02-03 before the cut to the 1,000 largest, and Nasdaq stocks are absent from the early years. The 0.5% cost reading is a guess at older trading costs, not a measurement.
- **18 settings from one window.** Three arms, two costs and three readings were all reported, with no tuning. Because the window offered 18 chances, an individual setting that looks good should not be read as a discovery. What can be read across the whole set is the pattern: 7 of 18 settings lead in the early years and none lead in the later ones as registered (6 and 2 on the continuous-account basis in section 3).
- **The sub-period split is a reading, not a pair of tests.** Section 3 explains the mismatch in taxes at the 1993 boundary and shows the continuous-account check.
- **Dividends and tax on them.** The comparison includes dividends (the CRSP market return does, and SPY's saved prices are adjusted). The stock prices are built from CRSP's return series, which includes dividends, but I did not separately confirm this for every stock history. Neither side pays yearly tax on dividends; tax is taken on sales and once at the end. That favours the comparison: dividends inside the strategy's prices are taxed at the short-term rate whenever its fast trading sells them, while the comparison compounds them untaxed for 29.5 years and pays 15% once.
- **It is one stretch of the market.** It includes 1973 to 1974, 1987 and the early 1990s. It says how the rule would have done, not why, and it is not a forecast.

## 5. Decision

No go or no-go label. Paper trading stays stopped, and the owner decides what follows. The result has two readings that this window cannot separate: the rule helped before 1993 and faded after, or the pre-1993 lead is mostly the optimistic dead-stock reading and a size tilt. Both fit the earlier windows only loosely: in 2000 to 2004 (-18.0 to -7.3 points at 18.3 bps and the low tax set) and 2005 to 2019 the rule trailed SPY, but in 2020 to 2024 it was ahead at its best reading (+1.6). The early decades are now spent for this rule and these three variants.

## 6. Reproduce

From the repository root with a clean tracked tree, one at a time, through the capped-run wrapper: the regression proof first (`ed_run.py --window holdout --arm A --eval-start 2020-01-02 --eval-end 2024-12-31`), then `ed_run.py --window early --arm X --eval-start 1969-02-03 --eval-end 1998-07-31 --regression-proof research/stock-signals/early-decades/_regression.json --out research/stock-signals/early-decades/_early-X.json` for X in A, B, C. The lock files must be cleared by the owner first. Details: [README.md](README.md).
