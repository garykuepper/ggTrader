# Dead-stock payout rerun: the first test again, with what the filings tell us about each dead stock

**Date:** 2026-11-24
**Study:** `research/stock-signals/dead-stock-payout-rerun` (registered plan frozen before any rerun code: [2026-11-24-dead-stock-payout-rerun-brief.md](2026-11-24-dead-stock-payout-rerun-brief.md))
**Question:** In the first test, the answer swung from far behind SPY (the S&P 500 index fund) to ahead of it depending on one guess: how much of its last price a holder of a stock that stopped trading got back (loss 1.0 = nothing, 0.5 = half, 0.0 = all of it). If the stocks for which our SEC filings show an acquisition-type event are sold at their last price, and only the other dead stocks keep the three guesses, how far behind or ahead of SPY is the same rule after trading costs and taxes?

## 1. In short

- **This is a re-score of windows both already seen, not a fresh test.** (A stock is "pinned" when its sale price comes from the payout table instead of from one of the three guesses.) The rule and both windows (2005 to 2019, and 2020 to 2024) were run in the first test; the data file ends 2024-12-31, so no unseen period exists. Nothing was tuned. The first test's one-run lock files were not touched.
- **The best case cannot change, and it did not.** The stocks with filing evidence are sold at their last price (loss 0.0), which is the first test's best reading. The rerun's best-case numbers equal the first test's 0.0 reading to the cent (checked by the program before anything was published).
- **The worst case got less bad, because of an added assumption, not because the rule did better.** Nothing about the rule or the market changed: 71 (development) and 17 (holdout) dead stocks moved from "one of three guesses" to "assume holders got the last price". Only the worst case is new information, and it rests on that assumption. At 18.3 bps and the low tax set, with the stocks we hold a checked filing for sold at their last price and every other dead stock written off completely (1.0):

| Window | Worst case (others get nothing) | Middle (others get half) | Best case (others get all) | First test's range (1.0 to 0.0) |
|---|---:|---:|---:|---|
| 2005 to 2019 (development) | **-7.2** | -4.6 | -2.0 | -16.4 to -2.0 |
| 2020 to 2024 (holdout) | **-0.9** | +0.4 | +1.6 | -7.3 to +1.6 |

  Numbers are the rule's yearly growth after cost and tax minus SPY's, in percentage points a year (a minus sign means behind SPY). The range is narrower in both windows, and the middle is between the two ends in both. The first test's own price check found that none of its dead stocks fell more than about 7 percent in its last month (the look of takeovers), so the bold worst case, where every not-pinned dead stock is written off completely, is a floor, not an expectation; a realistic reading sits nearer the best column. That still leaves the rule behind SPY in 2005 to 2019.
- **How many stocks get the last-price assumption.** In the headline cell (`confirmed` set, reading 1.0, 18.3 bps, low tax), 71 of the 117 dead stocks the rule held in 2005 to 2019 (60.7 percent) and 17 of the 24 held in 2020 to 2024 (70.8 percent) were pinned, meaning sold at their last price because a checked filing shows an acquisition-type event. The other 46 and 7 keep the three guesses; section 3 names them.
- **Risk, which does not depend on the guesses.** In both windows and at every reading the rule had a deeper fall from a peak and a lower Sharpe ratio (yearly return per unit of ups and downs; higher is better) than SPY: holdout 0.49 to 0.55 against 0.66, largest fall 40.7 to 45.6 percent against 33.7; development 0.17 to 0.35 against 0.51, 66.9 to 70.3 percent against 55.2. A lead of a point or two does not make up for that.
- **Do not lean on the friendlier window.** In 2005 to 2019 the rule trailed SPY at every reading, even with every dead stock paid in full. The 2020 to 2024 result owes almost all of its lead to stock picking in those five years (+5.45 points a year before costs and tax, against +0.35 in 2005 to 2019; section 3), and that stretch has now been looked at twice. The holdout figure is not evidence of an edge.
- **Two stocks are probably not dead at all.** JAZZ PHARMA and SPRINT NEXTEL are in the not-pinned list: the earlier phase found a company of the same name that kept filing annual and quarterly reports for years, which fits a change of name or legal form rather than a death (not verified). The rerun writes them off at reading 1.0; they would need a different fix (follow the holding into the successor), not a payout guess.
- **What the numbers say, and do not say.** Even when the filing-evidence stocks are assumed to recover their full last price, the rule trailed SPY after cost and tax in 2005 to 2019 at every reading. In 2020 to 2024 it ended close to SPY at the harshest reading (0.9 points behind) and ahead at the best (1.6 points), and at the best reading the lead disappears at the highest cost (40 bps: 0.0 points) and shrinks to 0.5 points under the higher tax set. That is a narrower range, not a verdict, and by the four plain rules (paper trading first, the owner decides) nothing here is a reason to put money in: the 0.0 for an acquisition is an assumption the filings do not measure, and both windows were already seen.

## 2. What was done

- **The rule, universe, windows, costs and taxes are the first test's** (each month, buy the 50 stocks that rose the most over the past year, skipping the last month, and trade at the next day's close; known as 12-1 momentum; the main list: common stock on the NYSE or Nasdaq, price at least $5, 1,000 largest by value; costs 6.4, 18.3 and 40 bps per side, a basis point (bps) being 0.01 percent of each trade's value; tax sets 24 and 15 percent, and 35 and 20 percent, the first rate applying to gains on holdings sold within a year and the second to older holdings; start cash 100,000). The first test's engine files are copied into this study with `ft_` in front of each name; a test checks that each copy equals the original apart from three import lines. Universe B (the $1 stress list) was not rerun.
- **The one change.** A dead stock whose row in the payout table has a loss figure is sold at its last price times (1 minus that figure). Today every such figure is 0.0 (an acquisition-type filing; the amount holders received was not measured). Every other dead stock is sold at the reading, as before.
- **Two evidence sets.** `confirmed` (the headline): the 88 stocks whose Form 25 notice (the exchange's removal notice) was read and checked. `all` (a sensitivity): those 88 plus 15 identified by company name only. In 2020 to 2024 the two sets are the same 17 stocks, so their results are identical there.
- **Checks that had to pass before anything was published.** (1) Reproduction: with nobody pinned, the rule's final value at readings 1.0 and 0.0 and SPY's final value equal the first test's saved cells to the cent in both windows. (2) Best-case identity: for every evidence set, cost and tax set, the reading-0.0 cell equals the first test's loss-0.0 cell. Both passed in both windows. The program is built to stop before any evidence-set result is computed if the first one fails; an independent reviewer found, and it was fixed before the real runs, that the code let a failed reproduction check carry on into the evidence-set cells although the registered plan said to stop.
- **The runs.** One development run (about 25 minutes, about 1 GB of memory) and one holdout run, each reading the published combined price release of our own database that the first test used (release 1, built from the university stock-price file CRSP plus Yahoo prices), with the payout table's fingerprint checked against the registered one. Both ran from code commit `fa944179` with the tracked files unchanged; the holdout run's broader git check also counted an untracked file, which was the development run's own result file, so its saved record says the tree was not clean in that wider sense. Timing: the development run took 24 minutes 39 seconds and peaked at 1.0 GB of memory (system timer, saved in `_run_notes.txt`); the holdout run was not timed.

## 3. Results

Every combination is shown. None was picked afterwards. "Gap" is the rule's yearly growth after tax minus SPY's, in percentage points. "Low tax" is 24 and 15 percent; "high tax" 35 and 20 percent. The reading is what the stocks not pinned by a filing are assumed to return: 1.0 nothing, 0.5 half, 0.0 all.

### Development, 2005 to 2019

#### Headline evidence set (`confirmed`: 88 stocks with a checked identity)

| Reading for unpinned dead stocks | Cost per side | Ends at (low tax) | Per year after tax (low tax) | Gap to SPY, points a year | Ends at (high tax) | Per year after tax (high tax) | Gap, high tax |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.0 (nothing back) | 6.4 bps | 130,016 | +1.8% | -6.3 | 127,196 | +1.6% | -6.2 |
| 1.0 (nothing back) | 18.3 bps | 114,706 | +0.9% | -7.2 | 113,581 | +0.9% | -6.9 |
| 1.0 (nothing back) | 40 bps | 89,159 | -0.8% | -8.8 | 89,159 | -0.8% | -8.5 |
| 0.5 (half back) | 6.4 bps | 189,606 | +4.4% | -3.7 | 177,860 | +3.9% | -3.9 |
| 0.5 (half back) | 18.3 bps | 167,810 | +3.5% | -4.6 | 159,834 | +3.2% | -4.6 |
| 0.5 (half back) | 40 bps | 133,945 | +2.0% | -6.1 | 130,766 | +1.8% | -5.9 |
| 0.0 (full last price) | 6.4 bps | 275,398 | +7.0% | -1.1 | 247,599 | +6.2% | -1.6 |
| 0.0 (full last price) | 18.3 bps | 243,212 | +6.1% | -2.0 | 221,880 | +5.5% | -2.3 |
| 0.0 (full last price) | 40 bps | 194,178 | +4.5% | -3.5 | 181,880 | +4.1% | -3.7 |

#### Sensitivity set (`all`: 103 stocks, 15 identified by name only)

| Reading for unpinned dead stocks | Cost per side | Ends at (low tax) | Per year after tax (low tax) | Gap to SPY, points a year | Ends at (high tax) | Per year after tax (high tax) | Gap, high tax |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.0 (nothing back) | 6.4 bps | 166,803 | +3.5% | -4.6 | 158,996 | +3.1% | -4.7 |
| 1.0 (nothing back) | 18.3 bps | 147,444 | +2.6% | -5.5 | 142,548 | +2.4% | -5.4 |
| 1.0 (nothing back) | 40 bps | 117,612 | +1.1% | -7.0 | 116,401 | +1.0% | -6.7 |
| 0.5 (half back) | 6.4 bps | 214,373 | +5.2% | -2.9 | 198,345 | +4.7% | -3.1 |
| 0.5 (half back) | 18.3 bps | 189,510 | +4.4% | -3.7 | 177,918 | +3.9% | -3.9 |
| 0.5 (half back) | 40 bps | 151,579 | +2.8% | -5.2 | 146,163 | +2.6% | -5.2 |
| 0.0 (full last price) | 6.4 bps | 275,398 | +7.0% | -1.1 | 247,599 | +6.2% | -1.6 |
| 0.0 (full last price) | 18.3 bps | 243,212 | +6.1% | -2.0 | 221,880 | +5.5% | -2.3 |
| 0.0 (full last price) | 40 bps | 194,178 | +4.5% | -3.5 | 181,880 | +4.1% | -3.7 |

#### SPY, same costs and taxes

| Cost per side | Ends at (low tax) | Per year (low tax) | Ends at (high tax) | Per year (high tax) |
|---|---:|---:|---:|---:|
| 6.4 bps | 321,026 | +8.1% | 308,024 | +7.8% |
| 18.3 bps | 320,298 | +8.1% | 307,340 | +7.8% |
| 40 bps | 318,976 | +8.0% | 306,095 | +7.7% |

#### Risk and trading at 18.3 bps, low tax (headline set)

| Reading | Sharpe ratio | Largest fall from a peak | Yearly turnover |
|---|---:|---:|---:|
| 1.0 | 0.17 | -70.3% | 8.4 |
| 0.5 | 0.26 | -68.6% | 8.2 |
| 0.0 | 0.35 | -66.9% | 7.9 |
| SPY | 0.51 | -55.2% | n/a |

#### Where the gap comes from (18.3 bps, low tax), points a year

| Evidence set | Reading | Stock picking (no loss, cost or tax) | Dead-stock effect | Trading cost | Tax | SPY's own cost and tax | Total gap |
|---|---|---:|---:|---:|---:|---:|---:|
| confirmed | 1 | +0.35 | -6.57 | -1.60 | -0.18 | +0.86 | -7.16 |
| confirmed | 0.5 | +0.35 | -3.31 | -1.66 | -0.80 | +0.86 | -4.56 |
| confirmed | 0 | +0.35 | +0.00 | -1.71 | -1.46 | +0.86 | -1.97 |
| all | 1 | +0.35 | -4.45 | -1.64 | -0.57 | +0.86 | -5.45 |
| all | 0.5 | +0.35 | -2.23 | -1.67 | -1.02 | +0.86 | -3.72 |
| all | 0 | +0.35 | +0.00 | -1.71 | -1.46 | +0.86 | -1.97 |

#### Dead stocks held, and how many were pinned (18.3 bps, low tax)

| Evidence set | Reading | Held | Pinned | Not pinned |
|---|---|---:|---:|---:|
| confirmed | 1.0 | 117 | 71 | 46 |
| confirmed | 0.5 | 117 | 71 | 46 |
| confirmed | 0.0 | 117 | 71 | 46 |
| all | 1.0 | 117 | 86 | 31 |
| all | 0.5 | 117 | 86 | 31 |
| all | 0.0 | 117 | 86 | 31 |

Not pinned in the headline cell (46): FOREST LABS (FRX), BARR INC (PERMNO-11415), ATHLON ENERGY IN (PERMNO-14062), FOUNDATION (PERMNO-14160), JUNO (PERMNO-15134), SPARK (PERMNO-15183), no name (PERMNO-16544), KERR-MCGEE (PERMNO-25769), INTL RECTIFIER (PERMNO-27705), SPRINT NEXTEL (PERMNO-39087), TITAN CP (PERMNO-48696), ESTERLINE TECH (PERMNO-50788), ANHEUSER BUSCH (PERMNO-59184), PHARMACEU RESOUR (PERMNO-61138), ALPHARMA INC-A (PERMNO-65832), VALEANT PHARMA (PERMNO-68340), CABLEVISION SYS (PERMNO-68857), OREGON STEEL (PERMNO-75308), BURLINGTON RES (PERMNO-75333), LYONDELL (PERMNO-75444), no name (PERMNO-75900), VINTAGE PETE INC (PERMNO-76272), MAVERICK TUBE CO (PERMNO-76598), MEDIMMUNE INC (PERMNO-76656), VERITAS DGC INC (PERMNO-76769), EL PASO CO (PERMNO-77481), PHILA CONSOL HLD (PERMNO-79617), CONTL AIRLINES (PERMNO-79641), MCMORAN EXPLOR (PERMNO-80557), RSA SECURITY (PERMNO-81178), NATIONWIDE FINL (PERMNO-84644), ABGENIX INC (PERMNO-86235), VARIAN SEMI (PERMNO-86791), STANCORP FINL (PERMNO-86809), SPINNAKER EXPLOR (PERMNO-87254), no name (PERMNO-87283), CEPHEI (PERMNO-88331), UNITED DEFENSE (PERMNO-89260), PREMCOR INC (PERMNO-89374), DADE (PERMNO-89680), PHARMION CORP (PERMNO-89918), PACIFICARE HEALT (PERMNO-90879), JAZZ PHARMA (PERMNO-92096), no name (PERMNO-92182), no name (PERMNO-92552), no name (PERMNO-93247)

### Holdout, 2020 to 2024

#### Headline evidence set (`confirmed`: 88 stocks with a checked identity)

| Reading for unpinned dead stocks | Cost per side | Ends at (low tax) | Per year after tax (low tax) | Gap to SPY, points a year | Ends at (high tax) | Per year after tax (high tax) | Gap, high tax |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.0 (nothing back) | 6.4 bps | 180,308 | +12.5% | +0.0 | 168,461 | +11.0% | -0.9 |
| 1.0 (nothing back) | 18.3 bps | 172,935 | +11.6% | -0.9 | 162,368 | +10.2% | -1.7 |
| 1.0 (nothing back) | 40 bps | 160,370 | +9.9% | -2.5 | 151,948 | +8.7% | -3.0 |
| 0.5 (half back) | 6.4 bps | 190,645 | +13.8% | +1.3 | 176,955 | +12.1% | +0.2 |
| 0.5 (half back) | 18.3 bps | 182,780 | +12.8% | +0.4 | 170,473 | +11.3% | -0.6 |
| 0.5 (half back) | 40 bps | 169,379 | +11.1% | -1.2 | 159,392 | +9.8% | -2.0 |
| 0.0 (full last price) | 6.4 bps | 201,575 | +15.1% | +2.6 | 185,922 | +13.2% | +1.3 |
| 0.0 (full last price) | 18.3 bps | 193,187 | +14.1% | +1.6 | 179,028 | +12.4% | +0.5 |
| 0.0 (full last price) | 40 bps | 178,899 | +12.3% | -0.0 | 167,247 | +10.8% | -0.9 |

#### Sensitivity set (`all`: 103 stocks, 15 identified by name only)

| Reading for unpinned dead stocks | Cost per side | Ends at (low tax) | Per year after tax (low tax) | Gap to SPY, points a year | Ends at (high tax) | Per year after tax (high tax) | Gap, high tax |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.0 (nothing back) | 6.4 bps | 180,308 | +12.5% | +0.0 | 168,461 | +11.0% | -0.9 |
| 1.0 (nothing back) | 18.3 bps | 172,935 | +11.6% | -0.9 | 162,368 | +10.2% | -1.7 |
| 1.0 (nothing back) | 40 bps | 160,370 | +9.9% | -2.5 | 151,948 | +8.7% | -3.0 |
| 0.5 (half back) | 6.4 bps | 190,645 | +13.8% | +1.3 | 176,955 | +12.1% | +0.2 |
| 0.5 (half back) | 18.3 bps | 182,780 | +12.8% | +0.4 | 170,473 | +11.3% | -0.6 |
| 0.5 (half back) | 40 bps | 169,379 | +11.1% | -1.2 | 159,392 | +9.8% | -2.0 |
| 0.0 (full last price) | 6.4 bps | 201,575 | +15.1% | +2.6 | 185,922 | +13.2% | +1.3 |
| 0.0 (full last price) | 18.3 bps | 193,187 | +14.1% | +1.6 | 179,028 | +12.4% | +0.5 |
| 0.0 (full last price) | 40 bps | 178,899 | +12.3% | -0.0 | 167,247 | +10.8% | -0.9 |

#### SPY, same costs and taxes

| Cost per side | Ends at (low tax) | Per year (low tax) | Ends at (high tax) | Per year (high tax) |
|---|---:|---:|---:|---:|
| 6.4 bps | 180,186 | +12.5% | 175,469 | +11.9% |
| 18.3 bps | 179,793 | +12.5% | 175,099 | +11.9% |
| 40 bps | 179,079 | +12.4% | 174,428 | +11.8% |

#### Risk and trading at 18.3 bps, low tax (headline set)

| Reading | Sharpe ratio | Largest fall from a peak | Yearly turnover |
|---|---:|---:|---:|
| 1.0 | 0.49 | -45.6% | 8.0 |
| 0.5 | 0.52 | -42.3% | 8.0 |
| 0.0 | 0.55 | -40.7% | 8.0 |
| SPY | 0.66 | -33.7% | n/a |

#### Where the gap comes from (18.3 bps, low tax), points a year

| Evidence set | Reading | Stock picking (no loss, cost or tax) | Dead-stock effect | Trading cost | Tax | SPY's own cost and tax | Total gap |
|---|---|---:|---:|---:|---:|---:|---:|
| confirmed | 1 | +5.45 | -3.25 | -1.84 | -3.03 | +1.79 | -0.87 |
| confirmed | 0.5 | +5.45 | -1.63 | -1.86 | -3.38 | +1.79 | +0.37 |
| confirmed | 0 | +5.45 | +0.00 | -1.89 | -3.72 | +1.79 | +1.63 |
| all | 1 | +5.45 | -3.25 | -1.84 | -3.03 | +1.79 | -0.87 |
| all | 0.5 | +5.45 | -1.63 | -1.86 | -3.38 | +1.79 | +0.37 |
| all | 0 | +5.45 | +0.00 | -1.89 | -3.72 | +1.79 | +1.63 |

#### Dead stocks held, and how many were pinned (18.3 bps, low tax)

| Evidence set | Reading | Held | Pinned | Not pinned |
|---|---|---:|---:|---:|
| confirmed | 1.0 | 24 | 17 | 7 |
| confirmed | 0.5 | 24 | 17 | 7 |
| confirmed | 0.0 | 24 | 17 | 7 |
| all | 1.0 | 24 | 17 | 7 |
| all | 0.5 | 24 | 17 | 7 |
| all | 0.0 | 24 | 17 | 7 |

Not pinned in the headline cell (7): AMERN NATL INS (PERMNO-13507), SANTANDER CONS (PERMNO-14414), GBT (PERMNO-15638), no name (PERMNO-16939), PRINCIPIA (PERMNO-18126), PROMETHEUS BIO (PERMNO-20870), no name (PERMNO-64899)

### Coverage in every cell

The counts and the names of pinned and not-pinned dead stocks are the same in all 18 cells of each evidence set within a window (checked in the saved results for all 72 cells): development `confirmed` 117 held, 71 pinned, 46 not pinned; development `all` 117 held, 86 pinned, 31 not pinned; holdout `confirmed` and `all` both 24 held, 17 pinned, 7 not pinned. No held dead stock was missing from the payout table in any cell. The not-pinned names of the headline set are listed under each window above. The 31 not pinned in the development `all` set are: FOREST LABS (FRX), BARR INC (PERMNO-11415), FOUNDATION (PERMNO-14160), JUNO (PERMNO-15134), SPARK (PERMNO-15183), no name (PERMNO-16544), SPRINT NEXTEL (PERMNO-39087), PHARMACEU RESOUR (PERMNO-61138), VALEANT PHARMA (PERMNO-68340), OREGON STEEL (PERMNO-75308), LYONDELL (PERMNO-75444), no name (PERMNO-75900), VINTAGE PETE INC (PERMNO-76272), MAVERICK TUBE CO (PERMNO-76598), VERITAS DGC INC (PERMNO-76769), PHILA CONSOL HLD (PERMNO-79617), NATIONWIDE FINL (PERMNO-84644), ABGENIX INC (PERMNO-86235), VARIAN SEMI (PERMNO-86791), STANCORP FINL (PERMNO-86809), SPINNAKER EXPLOR (PERMNO-87254), no name (PERMNO-87283), CEPHEI (PERMNO-88331), UNITED DEFENSE (PERMNO-89260), PREMCOR INC (PERMNO-89374), DADE (PERMNO-89680), PHARMION CORP (PERMNO-89918), JAZZ PHARMA (PERMNO-92096), no name (PERMNO-92182), no name (PERMNO-92552), no name (PERMNO-93247).

### What the tables show

- **Development window.** Selling the 71 pinned stocks at their last price lifts the harshest reading from 16.4 points behind SPY to 7.2 behind (9.3 points). What is left of the dead-stock effect at that reading, -6.6 points, comes from the 46 stocks still under the guess (the first test: -16.2). Even at the best reading the rule trails SPY by 2.0 points.
- **Holdout window.** At the harshest reading the rule is 0.9 points behind SPY at 18.3 bps (1.7 behind with the higher tax); at the middle reading it is 0.4 ahead; at the best reading 1.6 ahead. At 40 bps and low tax the best reading nearly matches SPY (0.02 points behind, 178,899 against 179,079) and the other two trail it.
- **Risk.** In both windows the rule has a larger fall from a peak and a lower Sharpe ratio than SPY at every reading (holdout 0.49 to 0.55 against 0.66).

## 4. What this cannot tell you

- **0.0 for an acquisition is an assumption.** The filings show that an acquisition-type event was reported, not what holders were paid. A stock-for-stock deal, or a deal paid late, could differ. The rerun narrows the range only to the extent that assumption holds.
- **Most of the unknown is not gone.** The not-pinned stocks (46 in development, 7 in holdout; names above) still carry the three guesses; of the 141 stocks, 38 have no loss figure at all (the 46 and 7 above also include the 15 name-only matches, which have a figure but are not pinned in the headline set): 8 have no company name in our data, 22 had no company found by name or a match that was rejected, 4 show only an announced merger vote, and 4 matched a company without an acquisition item.
- **A second look at windows already seen.** Nothing here was tested on data the rule had never met; one rule over two windows is what it was in the first test.
- **Why pinning is not hindsight in the rule's choices.** The filings found years later only set the sale price at the moment a stock dies; they never change which stocks the rule buys on any day.
- **For cash takeovers 0.0 is a reasonable central estimate** (holders usually receive about the deal price, close to the last trade); the larger uncertainty is in stock-for-stock deals, in deals paid late, and in the stocks not pinned.
- **The 15 name-only identities** are leads; the headline excludes them, and the `all` set shows what including them does (development worst case -5.5 instead of -7.2).
- **Cash path.** A higher sale price changes the cash for later buys, so the dead stocks held can differ slightly between readings; the counts above are for the cell named.
- **Everything the first test could not tell you still applies:** no trading-size limits (no volume in the data), no dividend tax, no margin, spreads beyond the cost grid, a five-year holdout.

## 5. Files and how to reproduce

Results `_rerun-dev-A.json` and `_rerun-holdout-A.json` (every cell, the checks, the coverage per cell, the decomposition, provenance); `_run_notes.txt` (timing of the development run). Every trade of every cell is in `results/dead_stock_payout_rerun/dev-A-trades.csv.gz` and `holdout-A-trades.csv.gz`, which git does not track. Code in `code/`. Run from the repository root with a clean working tree, one after the other, only through the capped-run wrapper:

    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-payout-rerun/code/run.py --window dev --release-id 1 --crsp-load 1
    scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-payout-rerun/code/run.py --window holdout --release-id 1 --crsp-load 1

The first test's files it reads: `research/stock-signals/dead-stock-first-test/_dev-A.json` and `_holdout-A.json`; the payout table: `research/stock-signals/dead-stock-data-integrity/_payouts_v2.csv`.

## Appendix A. Risk and trading for every cell

Sharpe ratio is the yearly return per unit of ups and downs (higher is better). Largest fall is the worst drop from a peak to a later low. Yearly turnover is how many times the account's value is bought and sold in a year.

### Development, 2005 to 2019

| Evidence set | Reading | Cost per side | Tax set | Sharpe ratio | Largest fall from a peak | Yearly turnover |
|---|---|---|---|---:|---:|---:|
| confirmed | 1.0 | 6.4 bps | low | 0.20 | -70.0% | 8.4 |
| confirmed | 1.0 | 6.4 bps | high | 0.20 | -70.0% | 8.4 |
| confirmed | 1.0 | 18.3 bps | low | 0.17 | -70.3% | 8.4 |
| confirmed | 1.0 | 18.3 bps | high | 0.17 | -70.3% | 8.4 |
| confirmed | 1.0 | 40 bps | low | 0.11 | -70.8% | 8.4 |
| confirmed | 1.0 | 40 bps | high | 0.11 | -70.8% | 8.4 |
| confirmed | 0.5 | 6.4 bps | low | 0.29 | -68.3% | 8.1 |
| confirmed | 0.5 | 6.4 bps | high | 0.28 | -68.3% | 7.9 |
| confirmed | 0.5 | 18.3 bps | low | 0.26 | -68.6% | 8.2 |
| confirmed | 0.5 | 18.3 bps | high | 0.25 | -68.6% | 8.0 |
| confirmed | 0.5 | 40 bps | low | 0.21 | -69.2% | 8.3 |
| confirmed | 0.5 | 40 bps | high | 0.20 | -69.2% | 8.2 |
| confirmed | 0.0 | 6.4 bps | low | 0.38 | -66.6% | 7.8 |
| confirmed | 0.0 | 6.4 bps | high | 0.36 | -66.6% | 7.6 |
| confirmed | 0.0 | 18.3 bps | low | 0.35 | -66.9% | 7.9 |
| confirmed | 0.0 | 18.3 bps | high | 0.33 | -66.9% | 7.6 |
| confirmed | 0.0 | 40 bps | low | 0.30 | -67.5% | 8.0 |
| confirmed | 0.0 | 40 bps | high | 0.28 | -67.5% | 7.8 |
| all | 1.0 | 6.4 bps | low | 0.26 | -68.6% | 8.2 |
| all | 1.0 | 6.4 bps | high | 0.25 | -68.6% | 8.1 |
| all | 1.0 | 18.3 bps | low | 0.23 | -68.9% | 8.3 |
| all | 1.0 | 18.3 bps | high | 0.22 | -68.9% | 8.2 |
| all | 1.0 | 40 bps | low | 0.18 | -69.5% | 8.4 |
| all | 1.0 | 40 bps | high | 0.18 | -69.5% | 8.4 |
| all | 0.5 | 6.4 bps | low | 0.32 | -67.6% | 8.0 |
| all | 0.5 | 6.4 bps | high | 0.30 | -67.6% | 7.8 |
| all | 0.5 | 18.3 bps | low | 0.29 | -67.9% | 8.1 |
| all | 0.5 | 18.3 bps | high | 0.28 | -67.9% | 7.9 |
| all | 0.5 | 40 bps | low | 0.24 | -68.5% | 8.2 |
| all | 0.5 | 40 bps | high | 0.23 | -68.5% | 8.1 |
| all | 0.0 | 6.4 bps | low | 0.38 | -66.6% | 7.8 |
| all | 0.0 | 6.4 bps | high | 0.36 | -66.6% | 7.6 |
| all | 0.0 | 18.3 bps | low | 0.35 | -66.9% | 7.9 |
| all | 0.0 | 18.3 bps | high | 0.33 | -66.9% | 7.6 |
| all | 0.0 | 40 bps | low | 0.30 | -67.5% | 8.0 |
| all | 0.0 | 40 bps | high | 0.28 | -67.5% | 7.8 |
| SPY | n/a | 6.4 bps | low | 0.51 | -55.2% | n/a |
| SPY | n/a | 6.4 bps | high | 0.50 | -55.2% | n/a |
| SPY | n/a | 18.3 bps | low | 0.51 | -55.2% | n/a |
| SPY | n/a | 18.3 bps | high | 0.50 | -55.2% | n/a |
| SPY | n/a | 40 bps | low | 0.51 | -55.2% | n/a |
| SPY | n/a | 40 bps | high | 0.49 | -55.2% | n/a |



### Holdout, 2020 to 2024

| Evidence set | Reading | Cost per side | Tax set | Sharpe ratio | Largest fall from a peak | Yearly turnover |
|---|---|---|---|---:|---:|---:|
| confirmed | 1.0 | 6.4 bps | low | 0.51 | -44.2% | 8.0 |
| confirmed | 1.0 | 6.4 bps | high | 0.47 | -46.2% | 7.7 |
| confirmed | 1.0 | 18.3 bps | low | 0.49 | -45.6% | 8.0 |
| confirmed | 1.0 | 18.3 bps | high | 0.45 | -47.5% | 7.8 |
| confirmed | 1.0 | 40 bps | low | 0.45 | -48.1% | 8.0 |
| confirmed | 1.0 | 40 bps | high | 0.42 | -49.8% | 7.8 |
| confirmed | 0.5 | 6.4 bps | low | 0.55 | -41.1% | 8.0 |
| confirmed | 0.5 | 6.4 bps | high | 0.50 | -43.2% | 7.8 |
| confirmed | 0.5 | 18.3 bps | low | 0.52 | -42.3% | 8.0 |
| confirmed | 0.5 | 18.3 bps | high | 0.48 | -44.3% | 7.8 |
| confirmed | 0.5 | 40 bps | low | 0.48 | -44.9% | 8.0 |
| confirmed | 0.5 | 40 bps | high | 0.44 | -46.8% | 7.8 |
| confirmed | 0.0 | 6.4 bps | low | 0.58 | -39.9% | 8.0 |
| confirmed | 0.0 | 6.4 bps | high | 0.53 | -42.1% | 7.8 |
| confirmed | 0.0 | 18.3 bps | low | 0.55 | -40.7% | 8.0 |
| confirmed | 0.0 | 18.3 bps | high | 0.51 | -42.8% | 7.8 |
| confirmed | 0.0 | 40 bps | low | 0.51 | -42.1% | 8.0 |
| confirmed | 0.0 | 40 bps | high | 0.47 | -44.0% | 7.8 |
| all | 1.0 | 6.4 bps | low | 0.51 | -44.2% | 8.0 |
| all | 1.0 | 6.4 bps | high | 0.47 | -46.2% | 7.7 |
| all | 1.0 | 18.3 bps | low | 0.49 | -45.6% | 8.0 |
| all | 1.0 | 18.3 bps | high | 0.45 | -47.5% | 7.8 |
| all | 1.0 | 40 bps | low | 0.45 | -48.1% | 8.0 |
| all | 1.0 | 40 bps | high | 0.42 | -49.8% | 7.8 |
| all | 0.5 | 6.4 bps | low | 0.55 | -41.1% | 8.0 |
| all | 0.5 | 6.4 bps | high | 0.50 | -43.2% | 7.8 |
| all | 0.5 | 18.3 bps | low | 0.52 | -42.3% | 8.0 |
| all | 0.5 | 18.3 bps | high | 0.48 | -44.3% | 7.8 |
| all | 0.5 | 40 bps | low | 0.48 | -44.9% | 8.0 |
| all | 0.5 | 40 bps | high | 0.44 | -46.8% | 7.8 |
| all | 0.0 | 6.4 bps | low | 0.58 | -39.9% | 8.0 |
| all | 0.0 | 6.4 bps | high | 0.53 | -42.1% | 7.8 |
| all | 0.0 | 18.3 bps | low | 0.55 | -40.7% | 8.0 |
| all | 0.0 | 18.3 bps | high | 0.51 | -42.8% | 7.8 |
| all | 0.0 | 40 bps | low | 0.51 | -42.1% | 8.0 |
| all | 0.0 | 40 bps | high | 0.47 | -44.0% | 7.8 |
| SPY | n/a | 6.4 bps | low | 0.66 | -33.7% | n/a |
| SPY | n/a | 6.4 bps | high | 0.63 | -33.7% | n/a |
| SPY | n/a | 18.3 bps | low | 0.66 | -33.7% | n/a |
| SPY | n/a | 18.3 bps | high | 0.63 | -33.7% | n/a |
| SPY | n/a | 40 bps | low | 0.66 | -33.7% | n/a |
| SPY | n/a | 40 bps | high | 0.63 | -33.7% | n/a |


