# Dead-stock random baseline: the first test's rule against 50 stocks drawn by chance

**Date:** 2026-11-25
**Study:** `research/stock-signals/dead-stock-random-baseline` (plan written down and frozen before any code: [2026-11-25-dead-stock-random-baseline-brief.md](2026-11-25-dead-stock-random-baseline-brief.md); four dated notes at its end record what changed along the way)
**Question:** The first test's fixed rule (each month, buy the 50 stocks that rose the most over the past year, skipping the last month; called momentum) trailed SPY (the S&P 500 index fund) in all three windows we have run it on, at the worst dead-stock reading (at the best reading it was ahead in 2020 to 2024). Is that because of how the rule picks stocks, or would any 50 stocks drawn by chance from the same list, bought and sold with the same costs and taxes, trail SPY by about as much?

## 1. In short

- **Chance does not give one answer. The rule's picks were worse than chance in one window, better in another, and in between depending on what dead stocks paid.** A "draw" below means one portfolio of 50 stocks chosen at random each month from the stocks the rule could have held that day. 100 draws were made per version and window.
  - **2000 to 2004:** at every dead-stock reading, every one of the 100 draws of both versions did better than the rule. The rule lost 18.0 points a year to SPY at the worst reading; the middle draw was about 1 point behind (fresh draws) or 1.6 points ahead (matched draws).
  - **2005 to 2019:** at the worst and middle readings every draw did better than the rule. At the best reading (dead stocks paid their last price in full) the rule, 2.0 points behind SPY, did better than 99 of 100 fresh draws and about level with the matched draws (better than 39 of 100).
  - **2020 to 2024:** the rule did better than most draws at every reading. At the worst reading it beat 95 of 100 fresh draws and 73 of 100 matched draws; at the middle and best readings it beat 98 to 100 of 100.
- **What picking added or took away, in points a year (the rule's gap to SPY minus the middle draw's gap):**

| Window | Worst reading (dead stock pays nothing) | Best reading (last price in full) |
|---|---|---|
| 2000 to 2004 | -17.0 (fresh draws), -19.6 (matched) | -10.7, -12.9 |
| 2005 to 2019 | -8.9, -11.3 | +1.9, -0.2 |
| 2020 to 2024 | +3.2, +0.7 | +9.7, +7.1 |

  Matched draws replace as many names each month as the rule did; fresh draws start over each month and trade much more (see section 3). The size of the swing, from about -20 to +10 points a year depending on the window and the dead-stock reading, is the main finding: the rule's picks do not add a steady amount in either direction.
- **The rule's shortfall in 2005 to 2024 was not unique to its picks.** In those two windows (run separately, not as one stretch), almost every chance portfolio also trailed SPY: 0 of 100 fresh draws beat SPY at any reading; 0 to 3 of 100 matched draws did. A 50-stock, equal-weight portfolio from this list, trading monthly with these costs, was a hard standard to meet under these settings, whoever chose the stocks. This study did not split that shortfall into its parts (equal weighting, the list, costs, dead-stock exits). In 2000 to 2004 it was not hard: chance portfolios beat SPY in 35 to 94 of 100 fresh draws and 75 to 100 of 100 matched draws (the lowest numbers are at the worst reading), because SPY itself fell 2.3 percent a year.
- **In 2005 to 2019 the rule held 117 stocks on the day they stopped trading; the draws held about 33 (fresh) or 31 (matched).** In the other two windows the rule held 31 and 24, against about 13 and 8. The rule held two to four times as many stocks that later died. This study does not say why. Whether the rule bought stocks that had jumped before a buy-out is one guess it did not test. This matters because what dead stocks paid is the largest unknown (section 4).
- **The random draws behave as chance should.** With dead stocks paid in full and no costs or taxes, the average of 20 draws was within about one point a year of an equal-weight hold of every eligible stock in all three windows (section 3).
- **Nothing here is a label.** Paper trading comes first and the owner decides. These are the same three windows already seen, so this adds no new market sample.

## 2. What was done

- **The rule, list, costs, taxes and dead-stock handling are the first test's, unchanged.** Each month-end the eligible list is the first test's universe A: common stock on the NYSE or Nasdaq, price at least $5, among the 1,000 largest by company value on the previous month-end. The rule takes the top 50 by momentum; the draws take 50 at random from the same list on the same day (a stock whose price history had already ended is excluded for both). Equal weight, trade at the next day's close, start cash 100,000, cost 18.3 bps per side (a basis point, bps, is 0.01 percent of the traded value), the low tax set (24 percent short-term, 15 percent long-term). A dead stock is sold on its last day at its last price times (1 minus the loss); the loss is 1.0 (nothing back, the worst reading), 0.5 (half back, the middle) or 0.0 (last price in full, the best).
- **Two versions of the dice.** *Fresh draws:* 50 stocks chosen at random each month with no memory of last month. *Matched draws:* each month keep the names held last month and replace as many as the rule actually bought that month, taking the replacements at random. The counts come from the reproduced rule's worst-reading cell and are used at every dead-stock reading. This gives a comparison with a similar amount of trading (it is not an exact match of turnover): the fresh draws turned over about 22 times the account value a year, the matched ones 7.6 to 9.5 times, the rule 7.9 to 9.7.
- **The windows.** 2000-01-03 to 2004-12-31, 2005-01-03 to 2019-12-31 and 2020-01-02 to 2024-12-31. 100 draws per version per window, seeds fixed by a rule in the brief. Only the 18.3 bps cost and the low tax set were run; the first test's other cost and tax settings were not rerun. Data: the first test's inputs (release 1 of the combined prices in our own database, raw CRSP load 1), read only.
- **Check first, on every window.** Before any draw the study rebuilt the rule's own three headline results (one per dead-stock reading) and SPY's with this study's copy of the engine and compared them with the saved results of the first test (and of the fresh-window test for 2000 to 2004): all within half a cent in all three windows. No draw runs if one cell differs.
- **Engine copy.** The first test's five engine files were copied under new names and checked against the originals by hash. There is one change besides import lines (below), recorded and pinned by a test.
- **Sanity check of the dice.** With dead stocks paid in full, no cost and no tax, the mean yearly growth of the first 20 fresh draws should be close to holding every eligible stock equally: 5.75 against 6.37 percent (2000 to 2004), 9.99 against 10.07 (2005 to 2019), 11.39 against 10.62 (2020 to 2024). Each difference was inside two standard errors of the dice mean or 0.5 points a year, whichever is larger (rule fixed before any draw).
- **Three things went wrong on the way, and are in the brief's notes.** (1) The first run of 2020 to 2024 showed matched draws trading far less than the rule (1.1 against 8.1 times a year). The cause was a bug of mine: the ledger stamps trades without a time zone and the plan's days carry one, so every purchase count came out zero. The output of that run was set aside, not used, and the code now refuses to run when a purchase does not fall on a plan day. I had looked at that discarded 2020 to 2024 output; its fresh-draw numbers are the same as those reported here, because the bug touched only the matched version. (2) The 2000 to 2004 window then stopped on a rounding error in the copied ledger: a buy of half a cent was over the cash by 0.00000000003 dollars, and the original's test measured that against the tiny buy. The check now also needs the excess to be over one hundred-millionth of a dollar. This is the one non-import change and it cannot alter a case where the original did not stop. (3) Because the code changed, all three windows were rerun from the start on one commit (`2f351fd0`), so every number here comes from the same code. The earlier 2005 to 2019 attempt had also been cut off by the wrapper's two-hour limit after all draws; that attempt too was set aside. These last two statements come from the run logs, which are not kept in the repository.
- **Independent reviews.** Codex Sol reviewed the code twice before any real run; it found the resume and purchase-count problems, a missing output table and weak tests, all fixed.
- **Run times.** 17 minutes (2000 to 2004), 20 minutes (2020 to 2024), 2 hours 53 minutes (2005 to 2019). Raw numbers: [_early.json](_early.json), [_dev.json](_dev.json), [_holdout.json](_holdout.json): every draw, the check, the sanity block and the summary.

## 3. Results

"Gap" is the yearly growth after cost and tax minus SPY's, in points a year at 18.3 bps and the low tax set. "Fresh" and "matched" are the two versions of the draws. Numbers in brackets are the 5th and 95th percentile of the 100 draws. "Rule beat" is the share of the 100 draws that ended with a lower gap than the rule.

| Window | Reading | Rule gap | Fresh draws, middle (5th to 95th) | Matched draws, middle (5th to 95th) | Rule beat, fresh / matched |
|---|---|---:|---|---|---|
| 2000 to 2004 | worst (1.0) | -18.0 | -1.0 (-5.9 to +3.3) | +1.6 (-2.4 to +5.5) | 0% / 0% |
| 2000 to 2004 | middle (0.5) | -12.8 | +1.4 (-2.8 to +5.4) | +3.7 (+0.6 to +7.1) | 0% / 0% |
| 2000 to 2004 | best (0.0) | -7.3 | +3.4 (-0.1 to +7.9) | +5.6 (+3.0 to +9.1) | 0% / 0% |
| 2005 to 2019 | worst (1.0) | -16.4 | -7.5 (-9.2 to -5.7) | -5.2 (-6.7 to -3.6) | 0% / 0% |
| 2005 to 2019 | middle (0.5) | -8.8 | -5.8 (-7.0 to -4.2) | -3.4 (-4.7 to -2.1) | 0% / 0% |
| 2005 to 2019 | best (0.0) | -2.0 | -3.9 (-5.1 to -2.3) | -1.7 (-3.0 to -0.3) | 99% / 39% |
| 2020 to 2024 | worst (1.0) | -7.3 | -10.5 (-14.4 to -7.6) | -8.0 (-11.2 to -4.9) | 95% / 73% |
| 2020 to 2024 | middle (0.5) | -2.8 | -9.2 (-12.6 to -7.0) | -6.7 (-9.6 to -3.9) | 100% / 98% |
| 2020 to 2024 | best (0.0) | +1.6 | -8.0 (-10.9 to -5.3) | -5.5 (-8.2 to -2.3) | 100% / 100% |

The rule's gaps are the first test's and the fresh-window test's saved numbers, reproduced by the check. They are not the later payout rerun's narrower worst readings (-7.2 in 2005 to 2019, -0.9 in 2020 to 2024); no filing evidence exists for 2000 to 2004, so the three readings are used throughout, as in the plan.

**Account values at the worst reading** (100,000 at the start; middle draw; the rule; SPY): 2000 to 2004, 84,867 (fresh) and 96,520 (matched), 32,248, 89,168. 2005 to 2019: 109,273 and 153,973, 27,108, 320,298. 2020 to 2024: 110,105 and 124,389, 128,547, 179,793.

**Deepest fall at the worst reading** (middle draw, fresh / matched; the rule; SPY): 2000 to 2004, -54 / -51 percent, -82, -47.5. 2005 to 2019, -65 / -64, -82, -55.2. 2020 to 2024, -41 / -40, -54, -33.7.

**How many draws beat SPY** (of 100, worst / middle / best reading): 2000 to 2004, fresh 35 / 67 / 94, matched 75 / 99 / 100. 2005 to 2019, fresh 0 / 0 / 0, matched 0 / 0 / 3. 2020 to 2024, fresh 0 / 0 / 0, matched 0 / 0 / 1.

**Trading volume (times the account value a year, middle draw; the rule, at the worst reading):** fresh 22.4, 22.1, 21.5; matched 9.5, 7.9, 7.9; rule 9.7, 8.4, 8.1 for 2000 to 2004, 2005 to 2019, 2020 to 2024.

**Stocks held on the day they stopped trading (middle draw; the rule, at the worst reading):** fresh 13, 33, 8; matched 12, 31, 8; rule 31, 117, 24.

**Sanity block** (dead stocks paid in full, no cost, no tax; first 20 fresh draws against holding every eligible stock equally): 5.75 against 6.37 (difference -0.62, allowed 1.23), 9.99 against 10.07 (-0.08, allowed 0.52), 11.39 against 10.62 (+0.78, allowed 1.06); all passed.

## 4. What this does not say

- **It does not say why.** The draws show what chance did on the same list and rules. They do not show what part of the difference comes from momentum itself, from the kind of stocks momentum picks (for example recent winners that were then bought out), or from timing.
- **It is not a test of the rule on new years.** All three windows had been run before. Nothing here is tuned, and the draws are a yardstick, so no window needs to be unseen, but the study adds no new market sample.
- **What dead stocks paid is still unknown for 2000 to 2004** and only partly known for the later windows. The rule held two to four times as many dead stocks as chance, so the result at the worst reading depends heavily on that unknown. The three readings are assumptions about payout; this study does not establish bounds on what holders actually received.
- **With 100 draws, the reported shares move in steps of one percentage point; the underlying probability is less certain** (a 73% share has a sampling error of about 4 points if the draws are independent). A "0%" means none of the 100 did worse than the rule, not that none could.
- **Costs before 2001 were probably higher than the cost used here, and only 18.3 bps was run.** The first test's 6.4 and 40 bps were not rerun for the draws. The rule's gap in 2000 to 2004 at 40 bps was -19.6 at the worst reading.
- **Taxes:** the saved first-test books of the rule and SPY paid no tax in 2000 to 2004. This does not show that the random portfolios paid none, and the result files do not record their tax. In the two later windows both paid tax. Only the low tax set was run for the draws. Dividend tax is not modelled.
- **Equal weight against SPY's size weighting is part of every gap** and cannot be separated here from the list.
- **The fresh draws trade about 2.5 times as much as the rule**, so they pay more in costs; the matched draws exist to answer that. Both are shown.
- **One rule of one kind.** This says nothing about other rules (earnings, value, quality, machine learning), the daily design, or any holding period other than one month.

## 5. Files and how to reproduce

- Brief with four dated notes: [2026-11-25-dead-stock-random-baseline-brief.md](2026-11-25-dead-stock-random-baseline-brief.md)
- Results: [_early.json](_early.json), [_dev.json](_dev.json), [_holdout.json](_holdout.json); partial files (resumable, every draw) in `results/dead_stock_random_baseline/` (not tracked by git). After the runs a `provenance` block, copied from fields already in each file, was added so the study checker finds one; nothing else in the files changed.
- Code: [code/](code/); driver [code/run.py](code/run.py); random planners [code/random_plans.py](code/random_plans.py); summary logic [code/dead_stock_random_baseline.py](code/dead_stock_random_baseline.py); engine copies `code/rb_*.py` with their hash record [code/frozen_copies.json](code/frozen_copies.json).
- To reproduce one window, from the repository root with a clean tracked tree, through the capped-run wrapper (the 2005 to 2019 window needs `GGT_RUN_MAX_SECONDS=14400`):

      scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-random-baseline/code/run.py --window holdout --release-id 1 --crsp-load 1

  The driver refuses if the output file exists, if the tree is dirty, if the partial file belongs to a different commit, lock, data or draw set, and if the check against the first test's saved numbers fails.
