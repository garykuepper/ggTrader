# Dead-stock first test: a fixed momentum rule over liquid NYSE and Nasdaq stocks, dead ones included

**Date:** 2026-11-24
**Study:** `research/stock-signals/dead-stock-first-test` (brief frozen 2026-11-23 before any return was computed)
**Question:** Starting with 100,000 in a taxable account, does a fixed rule, "buy the 50 stocks that rose the most over the past year, skipping the last month", over every liquid NYSE and Nasdaq stock that existed at the time (dead ones included), end ahead of SPY (the S&P 500 fund) after trading costs and taxes?

## 1. The short answer

- **The answer depends on a number we do not have.** The data holds the last price of a stock that stopped trading, but not what shareholders received. The study tried three readings: they got nothing back (loss 1.0), half (0.5), or the full last price (0.0). The result swings from far behind SPY to ahead of it depending on the reading.
- **The 1.0 and 0.5 readings look wrong for this universe.** In both windows every dead stock the rule held, bar none, had a flat or rising price in its last month (141 of 141 in the main universe: 24 in the holdout and 117 in the development window), and none had collapsed. That is the usual look of a takeover, where holders are paid about the last price, not of a failure. So the 0.0 reading is the most realistic of the three, and 1.0 is a worst-case bound. This is a description of the prices, not a payout record.
- **At the 0.0 reading the rule does not consistently beat SPY.** In the 2005 to 2019 development window it trailed SPY by about 2 points a year after tax at the middle cost (6.1% a year against 8.1%). In the 2020 to 2024 holdout, run once after the code was fixed, it led by about 1.6 points a year (14.1% against 12.5%), but the lead disappears at the highest cost (40 bps, 0.4 percent of each trade's value) and shrinks to 0.5 points under the higher tax set. It also had a larger fall (40.7% against 33.7%) and a lower return per unit of risk (Sharpe ratio 0.55 against 0.66).
- **The wider stress universe was clearly worse.** Allowing every stock priced at $1 or more, with no size limit, the rule trailed SPY by more than 10 points a year in both windows even at the 0.0 reading.
- **This is a first look at one rule over two windows.** It is evidence, not a verdict. Paper trading comes before any real money, and the owner decides.

## 2. What was tested

- **The rule.** On the last trading day of each month, rank every eligible stock by its price change from about a year ago to about a month ago, buy the top 50 in equal amounts at the next day's close, and sell what drops out. No setting was tuned.
- **Who counts as eligible.** On the previous month's last day: an ordinary common stock, listed on the NYSE or Nasdaq, priced at $5 or more, and among the 1,000 largest by market value of the stocks that passed those tests (universe A, the main universe). Stocks that later died are included while they were alive. The stress universe B drops the size limit and lowers the price floor to $1.
- **Data.** CRSP daily prices from our own database (25,754 stocks, 1968 to 2024), joined to Yahoo prices where a Yahoo history exists (published release 1). CRSP has no volume, so real trading size limits are not tested.
- **Trading cost.** Charged on every buy and sell, dead-stock exits included: 6.4 bps (large US stocks) and 18.3 bps (small US stocks) from one large trader's published realised costs over 2009 to 2016, and 40 bps as a labelled stress level. A basis point (bps) is 0.01 percent.
- **Taxes.** Two sets, short-term and long-term gains at 24% and 15% (low) or 35% and 20% (high), through a tax-lot ledger that tracks each purchase, sells highest-cost lots first and applies the 30-day wash-sale rule. Tax is paid from the account's own cash. SPY runs through the same ledger. Both are sold on the last day and the final tax paid.
- **Windows.** Development 2005-01-03 to 2019-12-31, where the code was built and looked at. Holdout 2020-01-02 to 2024-12-31, run once. The program refuses a second run: the file `holdout-A.claim` records that it was started, and only the owner may delete it.
- **Dead-stock exit.** A stock that stops trading is sold on its last day at its last price times (1 minus the loss). A stock still alive at the end of the data file (2024-12-31) is not treated as dead.

## 3. Results for the main universe (A)

Every combination is shown. None was picked afterwards. "Gap" is the rule's yearly growth after tax minus SPY's, in percentage points. "Low tax" means 24% and 15%; "high tax" means 35% and 20%.

### Holdout, 2020 to 2024 (run once)

| Dead-stock loss | Trading cost per side | Ends at (low tax) | Per year after tax (low tax) | Gap to SPY, points a year | Ends at (high tax) | Per year after tax (high tax) | Gap, high tax |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.0 (nothing back) | 6.4 bps | 134,561 | +6.1% | -6.4 | 129,857 | +5.4% | -6.5 |
| 1.0 (nothing back) | 18.3 bps | 128,547 | +5.2% | -7.3 | 124,362 | +4.5% | -7.4 |
| 1.0 (nothing back) | 40 bps | 117,830 | +3.3% | -9.0 | 114,364 | +2.7% | -9.1 |
| 0.5 (half back) | 6.4 bps | 164,951 | +10.5% | -2.0 | 155,726 | +9.3% | -2.6 |
| 0.5 (half back) | 18.3 bps | 158,267 | +9.6% | -2.8 | 150,167 | +8.5% | -3.4 |
| 0.5 (half back) | 40 bps | 146,872 | +8.0% | -4.4 | 140,656 | +7.1% | -4.7 |
| 0.0 (full last price) | 6.4 bps | 201,575 | +15.1% | +2.6 | 185,922 | +13.2% | +1.3 |
| 0.0 (full last price) | 18.3 bps | 193,187 | +14.1% | +1.6 | 179,028 | +12.4% | +0.5 |
| 0.0 (full last price) | 40 bps | 178,899 | +12.3% | -0.0 | 167,247 | +10.8% | -0.9 |
| SPY | 6.4 bps | 180,186 | +12.5% | n/a | 175,469 | +11.9% | n/a |
| SPY | 18.3 bps | 179,793 | +12.5% | n/a | 175,099 | +11.9% | n/a |
| SPY | 40 bps | 179,079 | +12.4% | n/a | 174,428 | +11.8% | n/a |

Dead stocks held at death: 24. Price one month before the end versus the last price: under 0.5: 0; 0.5 to 0.9: 0; 0.9 to 1.1: 21; 1.1 or more: 3.
Skipped buys 5; mean yearly turnover 8.0; largest fall (loss 0.0, 18.3 bps, low tax) -40.7%; SPY largest fall -33.7%; Sharpe 0.55 vs SPY 0.66.

### Development, 2005 to 2019

| Dead-stock loss | Trading cost per side | Ends at (low tax) | Per year after tax (low tax) | Gap to SPY, points a year | Ends at (high tax) | Per year after tax (high tax) | Gap, high tax |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.0 (nothing back) | 6.4 bps | 31,577 | -7.4% | -15.5 | 31,577 | -7.4% | -15.2 |
| 1.0 (nothing back) | 18.3 bps | 27,108 | -8.3% | -16.4 | 27,108 | -8.3% | -16.1 |
| 1.0 (nothing back) | 40 bps | 20,532 | -10.0% | -18.1 | 20,532 | -10.0% | -17.8 |
| 0.5 (half back) | 6.4 bps | 104,227 | +0.3% | -7.8 | 103,494 | +0.2% | -7.6 |
| 0.5 (half back) | 18.3 bps | 89,929 | -0.7% | -8.8 | 89,529 | -0.7% | -8.5 |
| 0.5 (half back) | 40 bps | 68,669 | -2.5% | -10.5 | 68,632 | -2.5% | -10.2 |
| 0.0 (full last price) | 6.4 bps | 275,398 | +7.0% | -1.1 | 247,599 | +6.2% | -1.6 |
| 0.0 (full last price) | 18.3 bps | 243,212 | +6.1% | -2.0 | 221,880 | +5.5% | -2.3 |
| 0.0 (full last price) | 40 bps | 194,178 | +4.5% | -3.5 | 181,880 | +4.1% | -3.7 |
| SPY | 6.4 bps | 321,026 | +8.1% | n/a | 308,024 | +7.8% | n/a |
| SPY | 18.3 bps | 320,298 | +8.1% | n/a | 307,340 | +7.8% | n/a |
| SPY | 40 bps | 318,976 | +8.0% | n/a | 306,095 | +7.7% | n/a |

Dead stocks held at death: 117. Price one month before the end versus the last price: under 0.5: 0; 0.5 to 0.9: 0; 0.9 to 1.1: 108; 1.1 or more: 9.
Skipped buys 28; mean yearly turnover 7.9; largest fall (loss 0.0, 18.3 bps, low tax) -66.9%; SPY largest fall -55.2%; Sharpe 0.35 vs SPY 0.51.

### Where the gap comes from

The steps are in order, so each step's number includes the effect of the earlier steps. They add up to the gap exactly.

#### Holdout

| Dead-stock loss | Stock picking (loss 0, no cost, no tax, minus SPY) | Dead-stock loss | Trading cost | Tax | SPY's own cost and tax | Total gap |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | +5.45 | -11.37 | -1.70 | -1.47 | +1.79 | -7.30 |
| 0.5 | +5.45 | -5.78 | -1.79 | -2.50 | +1.79 | -2.83 |
| 0.0 | +5.45 | +0.00 | -1.89 | -3.72 | +1.79 | +1.63 |

Growth per year: SPY untaxed and free 14.25%; rule (loss 0.0) untaxed and free 19.70%; with cost 17.81%; with cost and tax 14.09%; SPY after cost and tax 12.46%.

#### Development

| Dead-stock loss | Stock picking (loss 0, no cost, no tax, minus SPY) | Dead-stock loss | Trading cost | Tax | SPY's own cost and tax | Total gap |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | +0.35 | -16.17 | -1.45 | +0.00 | +0.86 | -16.42 |
| 0.5 | +0.35 | -8.34 | -1.57 | -0.07 | +0.86 | -8.78 |
| 0.0 | +0.35 | +0.00 | -1.71 | -1.46 | +0.86 | -1.97 |

Growth per year: SPY untaxed and free 8.93%; rule (loss 0.0) untaxed and free 9.28%; with cost 7.57%; with cost and tax 6.11%; SPY after cost and tax 8.08%.

Before any cost or tax the stock picking beat SPY in both windows (a little in 2005 to 2019, a lot in 2020 to 2024). The rule trades about eight times its value each year (mean yearly turnover 7.9 and 8.0). Costs took about 1.7 to 1.9 points a year at 18.3 bps. Taxes took a further 1.5 points in the development window and 3.7 points in the holdout, because the rule realises short-term gains every month; SPY, held throughout, pays far less. So to match SPY after costs and tax the rule has to beat SPY by about 2 to 4 points a year before them, and it did that in one of the two windows.

## 4. What the dead stocks were

The CRSP file in our database has no delisting codes, so a takeover cannot be told from a failure directly. As a descriptive check (added after the first development run, recorded in the brief's amendments, and used to choose nothing) the study looked at each dead stock the rule held and compared its last price with its price 21 trading days earlier.

| Window | Dead stocks held | Under 0.5 of earlier price | 0.5 to 0.9 | 0.9 to 1.1 | 1.1 or more |
|---|---:|---:|---:|---:|---:|
| Holdout, A | 24 | 0 | 0 | 21 | 3 |
| Development, A | 117 | 0 | 0 | 108 | 9 |
| Holdout, stress B | 35 | 1 | 0 | 33 | 1 |
| Development, stress B | 107 | 2 | 3 | 92 | 10 |

In the main universe not one held dead stock collapsed before the end. That suggests takeovers, where holders are normally paid close to the last price, which makes the 0.0 reading the most realistic one. It does not prove it: a takeover paid partly in shares or delayed over months, or a stock halted at the end, could pay differently, and the study cannot see that. In the stress universe a few real collapses appear (1 and 5 of 35 and 107), as expected among $1 stocks.

## 5. Stress universe B (price at least $1, no size limit)

Same rule, same costs, same taxes; 7,658 stocks over the development read and 5,101 over the holdout read. Every cell is shown.

### Holdout, 2020 to 2024

| Dead-stock loss | Trading cost per side | Ends at (low tax) | Per year after tax (low tax) | Gap to SPY, points a year | Ends at (high tax) | Per year after tax (high tax) | Gap, high tax |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.0 (nothing back) | 6.4 bps | 55,977 | -11.0% | -23.5 | 53,126 | -11.9% | -23.8 |
| 1.0 (nothing back) | 18.3 bps | 53,042 | -11.9% | -24.4 | 50,436 | -12.8% | -24.7 |
| 1.0 (nothing back) | 40 bps | 48,096 | -13.6% | -26.0 | 45,892 | -14.4% | -26.2 |
| 0.5 (half back) | 6.4 bps | 78,695 | -4.7% | -17.2 | 74,121 | -5.8% | -17.7 |
| 0.5 (half back) | 18.3 bps | 74,538 | -5.7% | -18.2 | 70,335 | -6.8% | -18.7 |
| 0.5 (half back) | 40 bps | 67,534 | -7.6% | -19.9 | 63,942 | -8.6% | -20.3 |
| 0.0 (full last price) | 6.4 bps | 110,153 | +2.0% | -10.6 | 102,994 | +0.6% | -11.3 |
| 0.0 (full last price) | 18.3 bps | 104,290 | +0.8% | -11.6 | 97,686 | -0.5% | -12.3 |
| 0.0 (full last price) | 40 bps | 94,417 | -1.1% | -13.5 | 88,730 | -2.4% | -14.1 |
| SPY | 6.4 bps | 180,186 | +12.5% | n/a | 175,469 | +11.9% | n/a |
| SPY | 18.3 bps | 179,793 | +12.5% | n/a | 175,099 | +11.9% | n/a |
| SPY | 40 bps | 179,079 | +12.4% | n/a | 174,428 | +11.8% | n/a |

### Development, 2005 to 2019

| Dead-stock loss | Trading cost per side | Ends at (low tax) | Per year after tax (low tax) | Gap to SPY, points a year | Ends at (high tax) | Per year after tax (high tax) | Gap, high tax |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.0 (nothing back) | 6.4 bps | 9,405 | -14.6% | -22.7 | 9,295 | -14.7% | -22.5 |
| 1.0 (nothing back) | 18.3 bps | 7,947 | -15.5% | -23.6 | 7,865 | -15.6% | -23.4 |
| 1.0 (nothing back) | 40 bps | 5,843 | -17.3% | -25.3 | 5,797 | -17.3% | -25.1 |
| 0.5 (half back) | 6.4 bps | 27,690 | -8.2% | -16.3 | 27,269 | -8.3% | -16.1 |
| 0.5 (half back) | 18.3 bps | 23,400 | -9.2% | -17.3 | 23,093 | -9.3% | -17.1 |
| 0.5 (half back) | 40 bps | 17,198 | -11.1% | -19.1 | 17,020 | -11.1% | -18.9 |
| 0.0 (full last price) | 6.4 bps | 79,121 | -1.6% | -9.6 | 77,193 | -1.7% | -9.5 |
| 0.0 (full last price) | 18.3 bps | 66,991 | -2.6% | -10.7 | 65,580 | -2.8% | -10.6 |
| 0.0 (full last price) | 40 bps | 49,490 | -4.6% | -12.6 | 48,751 | -4.7% | -12.4 |
| SPY | 6.4 bps | 321,026 | +8.1% | n/a | 308,024 | +7.8% | n/a |
| SPY | 18.3 bps | 320,298 | +8.1% | n/a | 307,340 | +7.8% | n/a |
| SPY | 40 bps | 318,976 | +8.0% | n/a | 306,095 | +7.7% | n/a |

At the 0.5 and 1.0 readings it is far worse still (holdout, 18.3 bps, low tax: minus 5.7% and minus 11.9% a year). The rule behaves very differently once cheap, small stocks are allowed: the largest fall reaches 64.7% in the holdout and 77.2% in the development window (loss 0.0, 18.3 bps), and the 18.3 bps cost is very probably too low for a $1 stock. Universe B is a stress test and is reported beside A, never instead of it.

## 6. What this cannot tell you

- **What dying stocks paid is unknown** (section 4). The three readings bracket it; they do not measure it.
- **Real-world trading size limits are not tested.** The data has no volume. Costs are the three levels above, not a model of a 50-stock book of a given size.
- **Dividend tax is not modelled.** Prices are total-return prices, so reinvested dividends sit in the price and are not taxed separately; that flatters the rule and SPY alike.
- **No margin, no shorting, no stocks that left before 1968, no spreads wider than the cost grid.** A failing stock can trade far below its last quoted price before it stops.
- **One rule over two windows.** The holdout is five years. Two windows pointing in different directions (a loss to SPY, then a win) are what noise can produce.
- **Unused tax losses at the end are not valued**, for the rule or for SPY.
- **Small details that favour the rule slightly:** the rule skips a buy on a stock's last day (5 buys in the holdout, 28 in development), and it sits in cash until the first trade at the end of its first month while SPY is bought on day one.
- **Counting rules.** "21 and 252 trading days" are the stock's own valid closes, and "later spell" in the output means the same company continues under a later data record, not a verified takeover.

## 7. How the run went

This is recorded so nobody has to guess what was fixed after seeing numbers.

- **Before any run:** the rule, windows, costs, tax sets and universe were frozen in the brief. A cost grid of 5, 15 and 40 bps in the plan was replaced by 6.4, 18.3 and 40 after the library gave sourced figures.
- **First real run (development):** crashed on a rounding error smaller than a billionth of a dollar in the cash check. Fixed; no result was affected.
- **After the first development results were seen:** two additions were made, recorded as dated amendments in the brief. Stocks alive at the file end are not written down (needed only for the holdout). A descriptive price-path check was added. The development numbers did not change when the first was added (checked cell by cell).
- **Two independent pre-holdout reviews** (Codex Sol and Claude Opus) found problems that were fixed before the holdout ran: tax owed after a wash sale could stay unpaid until the next January, Sharpe and largest fall were taken from a curve that added paid tax back, the saved output lacked the curves and trade lists the brief promised, a final value ignored unpaid tax, and the holdout lacked checks for Yahoo data drift and for pinned release, load and a clean working tree. All were fixed and re-reviewed. The development numbers moved by under 0.01 percent as a result.
- **Holdout run once per universe.** Its claim files remain. A metadata-only audit found no confirmed Yahoo-linked series that ended while the company lived on (1,539 checked, 0 suspects).
- **Known small item left as it is:** one ledger test is named for three rounds of tax funding but does not exercise that limit.

## 8. Files

All under `research/stock-signals/dead-stock-first-test/`. Results: `_dev-A.json`, `_dev-B.json`, `_holdout-A.json`, `_holdout-B.json`, `_holdout-A-audit.json`. Every trade of every cell is in four side files in the git-ignored folder `results/dead_stock_first_test` (dev and holdout, universe A and B) (git-ignored, too large to commit). Their SHA-256 values:. Code is in `code/`, the frozen brief and its amendments in `2026-11-23-dead-stock-first-test-brief.md`. Rebuild with `.venv/bin/python research/stock-signals/dead-stock-first-test/code/run.py --window dev --universe A --release-id 1`; the holdout command refuses to run again.

- `dev-A-trades.csv.gz` 7db2e9d2a779592584dbca503f8a42fb19ef91e9c2d7aee2aaffe6ea439404a1
- `dev-B-trades.csv.gz` dba6c13e121d8eb1fef21e0074344bd49bf076b706c4ce30e442821853807714
- `holdout-A-trades.csv.gz` 647f4b61690d5052a3c5699a2b24518ddda85dd26c7dd95addeea37ea4a7ae6e
- `holdout-B-trades.csv.gz` aa4631a60d5bfa369eb1afc6d4e579b2c7ef60144a20bffc8c32476e6478b45f

