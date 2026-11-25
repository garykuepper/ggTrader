# Brief: Dead Stock Momentum Deaths

**Status:** FROZEN 2026-11-25, approved by the owner ("yes look into why the rule held about 3 times as many dead stocks"). Frozen before any code of this study exists; later changes are dated amendments at the bottom, never silent edits.
**Study:** stock-signals/dead-stock-momentum-deaths (kind: audit), written 2026-11-25.
**Decision fed:** which kind of rule to test next, and whether the first test's momentum results should be read at the worst, middle or best dead-stock reading. The owner makes that decision. There is no pass bar and no GO/NO-GO label.

## 0. Registered parameters

These lines are checked by the study tool. Every cited figure in the report must use these values or say which line it departs from.

- eval_start: 2000-01-03
- eval_end: 2024-12-31
- max_leverage: 1.0
- benchmark: none (this study counts and describes stocks; no account is simulated, so no SPY comparison)
- costs: 0 bps per side (no trades are simulated)
- family_size: 1
- seed: none (nothing in this study is random)

`family_size` is 1: one question, one fixed set of measures, no tuning. Nothing is picked after the run.

**The three windows** (the first test's, the fresh-window test's, all already seen): 2000-01-03 to 2004-12-31, 2005-01-03 to 2019-12-31, 2020-01-02 to 2024-12-31.

## 1. Question and fixed measures

**Question, in plain words.** The random-baseline study found that the first test's momentum rule held 31, 117 and 24 stocks on the day they stopped trading in the three windows, against about 13, 33 and 8 for portfolios of 50 stocks drawn at random from the same list. Why does buying the 50 strongest risers end with two to four times as many stocks that later stop trading?

This study looks at the prices and the list only. It does not run the ledger and does not simulate any account.

**Hypotheses it can speak to** (it can show whether the data fit them; it cannot show a cause):
- H1. Strong recent risers are more likely to stop trading soon than other stocks on the list (the stock-month death rate is higher for the rule's picks than for all eligible stocks).
- H2. The deaths among the rule's picks are stocks that jumped in one day (the kind of move a buy-out announcement makes) inside the year the momentum score looks at, more often than other picks and other stocks.
- H3. The excess is not concentrated in a few months or years.

**The measures** (all computed from the same inputs as the first test's rule; each is fixed here):

**A. Death rates by momentum band.** Use the first test's decision days (the last trading day of each month in the window) and its list (universe A) with the first test's planner conditions: eligible that day, a finite momentum score, series not already ended before the decision day. Rank the usable stocks by score, highest first, with the planner's stable order. The *holding month* of a decision day starts on its execution day (the next trading day) and ends the day before the next decision day's execution day (for the last decision day, at the window end). A stock *dies in the holding month* if its series' last bar falls inside that period, it has no successor series (the first test's `has_successor` flag, which marks a data break rather than a real end), and the last bar is not a file-end row (the first test's censoring rule: series ending on the last day of the CRSP file are not deaths). For each window, report:
  1. Stock-months and deaths in total among usable stocks, and the death rate per stock-month.
  2. The same for four bands of score rank: 1 to 50 (the rule's picks), 51 to 150, 151 to 500, and 501 and below.
  3. The rate ratio of the top 50 to all usable stocks, with a 95 percent interval from resampling months (2,000 resamples of months with replacement; the seed is the fixed number 20261125 for this step only).
  4. The expected number of deaths among a random 50 each month, `sum over months of min(50, usable) x deaths among usable / usable`, next to the rule's actual count of top-50 stocks that died in their holding month. Also next to the random-baseline study's fresh-draw medians (13, 33, 8), read from its result files.
  5. By year: usable stock-months, deaths among the top 50, expected deaths among a random 50.

**B. Profile of the dying stocks.** For each stock that dies in a holding month, using the decision day that opens that month, report for two groups, *in the rule's top 50* and *usable but not in the top 50*: how many; the median and quartiles of the momentum score and of its rank as a share of the usable list; the share by size of the final-month price move (last close divided by the decision-day close: below 0.5, 0.5 to 0.9, 0.9 to 1.1, 1.1 and above; the first test's buckets); and the median and quartiles of the 63-trading-day return before the decision day.

**C. One-day jumps inside the scoring window.** For each stock-month in the three groups *top 50 that died*, *top 50 that did not die*, and *all usable stocks*, flag whether the stock had a one-day close-to-close gain of 20 percent or more between 252 and 21 trading days before the decision day (the span the score uses). Report the share flagged in each group. A flag is not a buy-out: earnings and news also cause jumps. It is a descriptive count, and the report says so.

**D. Check against the first test.** For the reference setting the first test used for its held-at-death count (loss 1.0, 18.3 bps, low tax), compare the count of the rule's top-50 stocks that died in their holding month with the first test's held-at-death number in its saved results (`sanity.held_at_death`), and state the difference. The two can differ for buys the ledger could not fill; the report explains any difference. If the difference is larger than 10 percent of the first test's number, the report says the audit's picture of the rule may not match the ledger's and does not draw conclusions from section 3 until the difference is explained.

**Not in this study:** any simulation of an account, costs, taxes, what the dying stocks paid, filings, other rules, other windows.

## 2. Data and freeze before calculation

- **Prices and list:** the first test's inputs, read only: published combined release 1 (`--release-id 1`) and raw CRSP load 1 (`--crsp-load 1`), universe A, the three windows above, with the first test's warm-up reading. Every output records the release and price hashes. Reads go through copies of the first test's reader files with a `dm_` prefix, checked against the originals by hash (the study tool forbids two studies defining the same module name).
- **The rule's numbers** (the first test's held-at-death counts) come from its saved result files, read only. The random-baseline study's medians come from its three result files, read only.
- **Check first.** Before any measure, the study rebuilds the rule's plan with the copied planner for the window and confirms that the study's own top-50 set on every decision day equals the planner's targets (the same check the random-baseline study made). The run stops if one day differs.

## 3. Registered model

There is no model. The measures of section 1 are counts and summaries over the stock-months of the first test's rule. Only information through the decision day is used for the score, the band and the flags; the death and the final-month move look forward from the decision day, because they are the outcome being described, and are never used to choose anything.

## 4. Portfolio and accounting

None. No account, cost or tax is simulated.

## 5. Frozen output plan

A results JSON per window with the data release and price hashes and every number of section 1 (A1 to A5, B, C, D), and a short report in plain words with one table per measure and a comparison across the three windows. The report states what the numbers show and does not state a cause.

## 6. Decision rule

None. The owner's four plain rules apply. If the numbers do not support H1 to H3, the report says so.

## 7. What this cannot tell you, and the owner's checks

- **It cannot show a cause.** A higher death rate among strong risers, or more one-day jumps among them, fits a story about buy-outs; it does not prove one. Filings (what each death was) exist only for 141 held stocks in two windows and are not used here.
- **A one-day jump of 20 percent is not a buy-out.** Earnings, news and bankruptcies' dead-cat bounces also jump.
- **Same three windows as before:** this adds no new market sample.
- **Rates for stocks near the end of a window** may be understated, because a stock that ends just after the window is not counted as dying inside it.
- **The universe is the first test's:** the 1,000 largest stocks with a price of at least $5. Smaller or cheaper stocks are not covered.

**Execution boundary.** New study code goes in this study's `code/` folder; nothing in `src/` is touched, vectorbt is not imported, the paper trader stays stopped and the database is read only. One heavy job at a time through the capped-run wrapper.

## Amendments

**2026-11-25 (after an independent code review, before any real run; no number computed).** Measure C counts trading days on each stock's own valid closes, not on the market calendar. The score reads the close 21 and 252 valid closes back from the stock's last valid close on or before the decision day (a stock with holes in its history therefore looks further back in calendar terms); C now flags a one-day jump of 20 percent or more between those two bars, both included. A jump never bridges a missing close. Section 1 C said "252 and 21 trading days before the decision day (the span the score uses)"; this makes the second half of that sentence true when a stock has missing closes. For stocks with complete histories nothing changes.
