# Brief: Dead Stock Payout Rerun

**Status:** FROZEN 2026-11-24, before any rerun code exists and before any return is computed. Nothing below changes without a dated amendment at the bottom of this file.
**Study:** stock-signals/dead-stock-payout-rerun (kind: screen), written 2026-11-24.
**Words used.** SPY is the S&P 500 index fund, the benchmark. CRSP is the university database of US stock prices (it keeps stocks that later stopped trading) that we loaded into our own database. A tax-lot ledger records every purchase separately so each sale is taxed on its own gain or loss; the wash-sale rule disallows a loss if the same stock is bought again within 30 days. Short-term tax applies to gains on holdings sold within a year, long-term tax to older holdings. Sharpe ratio is yearly return per unit of ups and downs (higher is better); yearly turnover is how many times the account's value is bought and sold in a year; leverage is borrowing to hold more than the account is worth (none here). A dead stock is one that stopped trading before the end of the data. A pinned stock is one whose sale price comes from the payout table (section 2).
**Decision fed:** how wide the first test's answer still is once the dead stocks we hold filing evidence for are sold at their last price. The owner decides what, if anything, to do next.

## 0. Registered parameters

These lines are checked by the study tool. Every cited figure in the report must use these values or say which line it departs from.

- eval_start: 2005-01-03
- eval_end: 2024-12-31
- max_leverage: 1.0
- benchmark: SPY buy-and-hold through the same ledger, same dates, same costs and same tax rates
- costs: 6.4, 18.3 and 40 bps per side (a basis point, bps, is 0.01 percent) of traded value on every buy and sell, dead-stock exits included; the same three levels as the first test
- family_size: 1
- seed: none (nothing in this study is random)

`family_size` is 1: it is the first test's one rule, unchanged. Nothing is tuned and nothing is picked afterwards; every cell is shown.

The two windows are the first test's: **development** 2005-01-03 to 2019-12-31 and **holdout** 2020-01-02 to 2024-12-31 (the last day of the CRSP price file).

## 1. Question and fixed arms

**What this is, said first.** This is a **re-score of windows both already seen**, not a fresh test. The rule and both windows were run in the first test (the development window while the code was built, the holdout once); the data file ends 2024-12-31, so no unseen period exists. The first test's one-run lock files `results/dead_stock_first_test/holdout-A.claim` and `holdout-B.claim` and all its result files are read as data only and are never written, moved or deleted.

**Question.** In the first test, the answer swung from far behind SPY to ahead of it depending on one guess: how much of its last price a holder of a stock that stopped trading got back (loss 1.0 = nothing, 0.5 = half, 0.0 = all of it). If the stocks for which our filings show an acquisition-type event are sold at their last price (loss 0.0) and only the remaining dead stocks keep the three guesses, how far behind or ahead of SPY is the same rule, after costs and taxes?

**The one change from the first test.** A dead stock that has a loss figure in the payout table is sold at its last price times (1 minus that figure). Every other dead stock is sold at the reading (1.0, 0.5 or 0.0), as before. Nothing else differs: same rule, same universe, same decision days, same costs, same taxes, same ledger.

**Arms (all reported, none dropped), universe A only:**

- evidence set: `confirmed` (the headline) or `all` (a sensitivity), defined in section 2;
- reading for every dead stock without a pinned figure: 1.0, 0.5, 0.0;
- cost per side 6.4, 18.3, 40 bps;
- tax set low (24 percent short-term, 15 percent long-term) or high (35 and 20 percent);
- both windows.

That is 2 x 3 x 3 x 2 = 36 strategy cells per window, plus 6 SPY cells (SPY does not depend on the evidence set or the reading). A third set, `none` (nobody pinned), is run only for the reproduction check in section 5, at one cost and one tax set, and is not part of the results tables.

Universe B (the $1 stress list) is **not** rerun: the payout table covers only the main list's stocks.

## 2. Data and freeze before calculation

- **Prices, universe, eligibility, ledger:** exactly as in the first test's brief (`research/stock-signals/dead-stock-first-test/2026-11-23-dead-stock-first-test-brief.md`, sections 2 to 4, including every dated amendment at its end: tax paid in January, valid-close lookbacks, the first month in cash, skipped last-day buys, the handling of later records, and the file-end rule), release 1, CRSP load 1, the same code, copied (section 7).
- **The payout table:** `research/stock-signals/dead-stock-data-integrity/_payouts_v2.csv`, SHA-256 `7c4a40db56db09d5c661a458e2505f6d8f8256910d6dabdd2dc1d6de99beef63`, 141 rows. The driver refuses to run on any other file. Every row is a dead stock the first test held in universe A.
- **Evidence sets.** A stock is **pinned** when its row has a numeric `suggested_loss`; its loss is that number. Today all 103 such rows say 0.0 (basis: the brief's assumed scenario for an acquisition; a payment amount was never measured). Rows with a blank `suggested_loss` (the 4 `deal_forms_only` rows, the matched rows with no acquisition item, the rejected and unmatched rows, the 8 unnamed rows) are never pinned.
  - `confirmed` (headline): rows with a numeric `suggested_loss` **and** `match_confidence == "confirmed"` (88 rows today: a Form 25 notice for the right company was read and checked).
  - `all` (sensitivity): every row with a numeric `suggested_loss` (103 today: the 88 plus 15 identified by company name only).
  - **Baseline counts of the table itself** (not of any cell's holdings): 141 rows, 117 development and 24 holdout; numeric loss 103 (86 and 17), all 0.0; of those 88 confirmed (71 and 17) and 15 name-only; blank loss 38; so in `confirmed` 53 rows are unpinned and in `all` 38. The pinned figure is an assumption, not a measured payment: no payment was measured for any of the 141.
- **Matching a column to a row.** The price frame's columns are `PERMNO-<n>`, Yahoo tickers, or either with a `~k` later-spell suffix. A column's base symbol (suffix removed) is looked up in the table's `symbol`. A held dead stock whose base symbol is not in the table is unpinned and falls back to the reading; the run counts and prints those names.
- **Freeze date:** this brief is committed before any rerun code exists.

## 3. Registered model

Identical to the first test: 12-1 momentum (`momentum_scores(close, lookback=252, skip=21)`), 50 highest scores, equal weight, decide on the last trading day of each month, trade at the next trading day's close, universe A (common stock on NYSE or Nasdaq, previous month-end price at least $5, market value rank 1,000 or better), start cash 100,000, a stock still alive on 2024-12-31 is not treated as dead. The lookahead guard is the first test's.

## 4. Portfolio and accounting

Identical to the first test: costs per side 6.4, 18.3, 40 bps on every buy and sell, dead exits included; taxes through the tax-lot ledger (`TaxParams` 24/15 and 35/20, wash-sale rule, tax paid from the book's own cash); SPY through the same ledger; both books fully sold on the last day and the final tax paid; no leverage.

## 5. Frozen output plan

- **Reproduction check, before any evidence-set result is computed or inspected.** With the evidence set `none`, cost 18.3 bps and the low tax set, the rule's final value at readings 1.0 and 0.0 and SPY's final value must each equal the first test's saved cell in `_dev-A.json` and `_holdout-A.json` (the saved final-value field of that cell) after rounding both to cents. A mismatch stops the run: the cause is found first, no tolerance is widened.
- **Identity check on the best case.** Today every pinned figure is 0.0, so at reading 0.0 pinning changes no sale price: for each window, evidence set (`confirmed`, `all`), cost and tax set, the reading-0.0 cell must equal the first test's loss-0.0 cell for the same cost and tax set (final value after tax and gap to SPY, rounded to cents). On a mismatch the run stops before anything is published and the cause is found first. There is no exception for other pinned figures: a different payout file needs a dated amendment.
- **Main tables** per window: for each evidence set, reading, cost and tax set, the rule's final value of 100,000, yearly growth after tax, gap to SPY in percentage points a year, Sharpe ratio, largest fall from a peak and yearly turnover, beside SPY.
- **The three lead numbers**, at 18.3 bps and the low tax set, headline evidence set, for each window: the **worst case** (every unpinned dead stock at reading 1.0), the **middle** (0.5) and the **best case** (0.0), each as the gap to SPY in points a year, printed beside the first test's old range (development: 16.4 behind to 2.0 behind; holdout: 7.3 behind to 1.6 ahead). The three are fixed payment scenarios; the report says how the range changed and says so plainly if it got wider or if the middle one is not between the other two. No cell is relabelled worst or best after the fact.
- **Coverage block, for every one of the 36 cells per window:** the number of distinct dead stocks held (counted by base symbol, the price-column name with any `~k` suffix removed), how many of them were pinned and how many were not, and the names of the unpinned ones (so the reader sees which stocks the guesses still apply to); the held dead stocks missing from the table are a subset of the unpinned and are listed separately. The first page quotes the percentage pinned from the cell: `confirmed`, 18.3 bps, low tax, reading 1.0.
- **Gap decomposition** as in the first report's section 3, in the same fixed order (stock picking with no loss, no cost and no tax, minus SPY; then dead-stock loss; then trading cost; then tax; then SPY's own cost and tax), for both evidence sets, both windows and all three readings, at 18.3 bps and the low tax set. The first step gives every dead stock its full last price; the second step introduces the pinned figures and the reading together. The steps must add to that cell's reported gap.
- **Results JSON:** `_rerun-dev-A.json` and `_rerun-holdout-A.json`, each with a provenance block (commit, clean-tree flag, code hashes, the payout table's SHA-256, release, load, the anchor values).
- The report says on its first page that this is a re-score of seen windows, that the best case cannot change by construction, and what share of the held dead stocks is pinned.

## 6. Decision rule

There is no pass bar and no GO/NO-GO label (owner rules reset 2026-11-20). The four plain rules apply: test only on data the strategy never saw while it was being tuned (here: nothing was tuned, and both windows are disclosed as seen), use only what was known on each day and include stocks that later died, subtract trading costs and taxes and compare to SPY over the same dates, paper trade first and the owner decides. The report states the three lead numbers and says plainly whether even the worst case trails SPY, whether the best case leads it, and how wide the range is compared with the first test's. It does not pick one cell.

## 7. What this cannot tell you, and the owner's checks

- **0.0 for an acquisition is an assumption.** The filings do not give a payment; 103 table rows carry that assumption (in the `all` set; 88 in the headline set), and how many of them are held in a given cell is reported per cell. The 15 name-only identities in the `all` set are leads, which is why the headline is `confirmed`.
- **Most of the unknown stays unknown.** Only part of the held dead stocks are pinned (counts per cell in the report); the rest keep the three guesses. The three readings are fixed payment scenarios, not measurements; the rerun reports how the resulting range changes, which could be a narrower or, through cash and tax effects, a different range.
- **Cash-path effects.** A different sale price changes the cash for later buys, so the set of held dead stocks can differ a little between cells; each cell is looked up on its own.
- **Everything the first test could not tell you still applies:** no trading-size limits (no volume in the data), no dividend tax, no margin, spreads beyond the cost grid, one rule over two windows.
- **Code copies.** The first test's engine files (`replay.py`, `ledger_run.py`, `bars_reader.py`, `universe_frames.py`, `dead_stock_first_test.py`) are copied into this study with `ft_` put in front of each file name (`ft_replay.py` and so on), because the study tool forbids two studies defining a module of the same name and forbids importing another study's code. The only change inside the copies is the three lines that import a sibling copy (`from replay import` becomes `from ft_replay import`, and likewise for `ledger_run`); a test checks that each copy equals the original with exactly those substitutions and nothing else, and pins every hash. Only `payout_prices.py`, `payout_rerun.py` and `run.py` are new code.
- **Tied scores and the price floor** stay as the copied engine has them; the new code adds no tie rule and no zero-price treatment.
- **Execution boundary:** new study code only. No engine, no paper-trading code, no database writes, no download. The first test's claim files and results are not touched. Heavy runs only through `scripts/run_capped.sh`.
- **The owner's checks before the run:** that the headline uses only the 88 checked identities; that the first test's windows are used again and labelled seen; that universe B is left out.
