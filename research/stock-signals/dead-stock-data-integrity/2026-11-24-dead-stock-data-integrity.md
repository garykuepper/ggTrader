# Dead-stock data integrity: Phase 1 findings (2026-11-24)

**Study:** stock-signals/dead-stock-data-integrity (an audit; no strategy was rerun).
**Source of every number below:** `_integrity.json` and `_roster.csv` in this folder, written by
one run of `code/run.py` on 2026-11-24. No number here was measured any other way.

## Short version

- The price database we loaded (CRSP) has very few damaged rows. Of 20,379,285 daily rows
  checked, three of the six checks found nothing at all (duplicate rows, missing prices,
  missing share counts). The other three found 345 one-day spikes, 251 long gaps and 249
  stretches of an unchanged price.
- Only **two** flags, out of 845, fall on a day the first test actually held that stock. Both are
  spikes in the "wider universe" (B) holdout run. The main (A) runs and the B development run
  have none. (An earlier version of this scan also showed a held gap; it was an error of the
  scan and is explained under "Limits".)
- Of the 141 dead stocks the first test held, 133 have a company name. All 141 have a price
  label; 107 look "calm like a deal" in their last month and 34 are "unclear". None looks like a
  collapse.
- This note does not say how much any result would change. That was not measured.

## Words used here

- **CRSP:** a university database of US stock prices that keeps stocks that were later closed
  out or bought. We loaded it into our own database (load 1, release 1).
- **McGill:** a second table (load 2) that gives each stock's company name.
- **Permno:** CRSP's permanent number for one stock. It stays the same when a ticker changes.
- **The first test:** the closed study `dead-stock-first-test`. It "held" a stock on a day when
  its simulated portfolio owned shares of it. Its A runs use the main stock list; its B runs use
  a wider, harsher list (the stress universe). "Dev" is the development period, "holdout" the
  later period it was scored on once.
- **Flag:** a row or stretch of rows that one of six fixed checks marks as suspicious. A flag is
  not proof of an error: it says "look here". The rules were frozen in the brief before the run.
- **Inside a holding:** the flag's date lies between the day the first test bought the stock and
  the day it sold it (the `loss0_cost18.3bps_taxlow` setting of its trade file, the same one for
  all four files).

## What was scanned

- 9,422 stocks and 20,379,285 daily rows, read in 19 batches of up to 500 stocks, from
  2004-01-01 to 2024-12-31 (the end of the CRSP file). A stock is scanned when it has at least
  one row since 2004-01-01 as an ordinary common share on the NYSE or Nasdaq; for those stocks
  every row since 2004-01-01 is read, whatever exchange it was on that day.
- The trading calendar used for the gap check has 5,285 dates, each a day on which at least
  1,000 NYSE or Nasdaq ordinary shares have a row.

## The 141 dead stocks

- **Names:** 133 of 141 have a company name from McGill. 8 have none (all stay in the list and
  are labelled).
- **Price labels** (from each stock's last month of returns up to its last day; descriptive only,
  they never set a payout): `calm_like_deal` 107 (dev 91, holdout 16), `unclear` 34 (dev 26,
  holdout 8). No stock is labelled `collapse_like`, `drifting_down` or `too_short`.
- What this means: for most of these stocks the price did not crash before it stopped trading.
  That fits a buyout or merger, but it does not prove one. The label cannot tell a cash buyout
  from a quiet delisting. Evidence from company filings is Phase 2 and has not started.

## The six checks

| Check | Flags | Stocks | Inside a holding: dev-A | dev-B | holdout-A | holdout-B |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Duplicate rows | 0 | 0 | 0 | 0 | 0 | 0 |
| Price missing or zero | 0 | 0 | 0 | 0 | 0 | 0 |
| Spikes that reverse next day | 345 | 253 | 0 | 0 | 0 | 2 |
| Gaps of 5 or more missing days | 251 | 222 | 0 | 0 | 0 | 0 |
| Same price on 10 rows in a row | 249 | 158 | 0 | 0 | 0 | 0 |
| Share count missing at month end | 0 | 0 | 0 | 0 | 0 | 0 |

Positions the first test opened (reference setting): dev-A 2,982, dev-B 3,268, holdout-A 993,
holdout-B 1,072. They cover 1,366, 1,805, 609 and 727 different stocks. The two B files are the
stress universe and are shown separately; no files were merged.

How the trade files were matched to stocks: some trade rows use a Yahoo ticker such as `AAPL`
and some use `PERMNO-<n>`. Tickers were turned into permnos through the release's symbol map,
using the date of the trade (the same rule the roster uses). All ticker rows were mapped: 5,659
(dev-A), 1,192 (dev-B), 2,197 (holdout-A) and 445 (holdout-B), with **0 unmapped**. Two
details: 45, 183, 0 and 8 rows use a name like `PERMNO-38420~1` (a later period of the same
stock) and were folded into that stock; and 266 dev-A and 189 dev-B ticker rows fell on a date
that no map entry covered, so the nearest entry was used (0 in the holdout files).

### What each check could and could not change

- **Duplicates, missing prices, missing share counts (all 0):** a clean result. Nothing here
  could move a result.
- **Spikes (345):** the day's return is probably a bad print or a data error, because the price
  came straight back. A spike inside a holding could change one day's profit or loss for that
  position, and it could change which stocks rank high that day. The data here does not say
  whether the first test's trades were affected.
- **Gaps (251):** days with no row. Many are long (the seven largest are over 3,000 trading
  days, roughly twelve years or more, where a stock disappears from the file and returns).
  None falls inside a holding, so none could have touched the first test's trades.
- **Same price on 10 rows (249):** can be an error, but can also be real: a stock waiting to be
  bought out often trades at almost the same price for weeks. Flagged only; not judged.
- **None of these checks** says what a holder received when a stock was closed out. That is the
  question Phase 2 (company filings) is for.

## The largest flags inside holdings

Only two exist, so these are all of them. "Inside a holding" is taken from the first test's
trade files; this output does not give the position's start, end or size, so the effect on its
profit is **not measured**. Both are in the B (stress) holdout file.

1. **Spike, permno 16680, 2024-07-24**, return +0.5632 then -0.3237 the next day.
2. **Spike, permno 14873, 2020-06-10**, return +0.5173 then -0.3029 the next day.

The gap on permno 92449 that an earlier version of this note listed as a held flag is gone. It
was never a hole in the data (see "Limits").

For comparison, the largest flags anywhere (held or not) are a spike on permno 14153 on
2013-10-03 (return +4.57, then -0.82), a gap on permno 77263 from 2005-12-27 (4,253 days
missing) and an unchanged price of 10.595 on permno 22582 for 34 rows from 2023-07-21. The
largest spike and stale examples are the same as in the earlier version; the gap list changed
(permno 81761, 3,865 days, no longer appears among the largest, and permnos 89986 and 88937 now do).

## Limits of this note

- **An earlier version of the scan was wrong about gaps and was corrected the same day.** It
  kept only the rows where a stock was an ordinary share on the NYSE or Nasdaq, so a stock that
  spent some years on another exchange (for example NYSE American) lost those days and the gap
  check reported the hole. That version counted 220 gaps on 195 stocks (and 342 spikes and 239
  stale runs), and listed one gap inside a holding (permno 92449, "1,600 days missing"), which
  was exactly its years on NYSE American. The corrected scan reads every row of each selected
  stock: 251 gaps on 222 stocks, 345 spikes and 249 stale runs, and no held gap. The gap count
  went up because stocks are now followed over their whole history, not just the filtered
  years; I did not separately trace each new gap. The brief carries a dated clarification.
- The stocks scanned are still chosen by the brief's rule (ordinary shares, NYSE or Nasdaq).
  A stock that never had such a row since 2004 is not scanned at all.
- Permno 77263 (flagged 4,253 days missing from 2005-12-27) was looked up with no filter: it
  has rows in 2005 and then none until 2022, so that hole is real in the file. The other large
  gaps were not looked up one by one.
- A flag inside a holding says only that the date overlaps. It is not an estimate of harm.
- The numbers come from a run on a committed state of the code (`git_dirty` is false in the
  provenance); the SHA-256 of each code file is recorded too.
- Each CRSP read used the table directly with the load id after the same "load is complete"
  test that `daily_of()` makes; `daily_of()` itself took over 12 minutes per call because it
  copies the whole load every time.
- 266 dev-A and 189 dev-B ticker rows fell on a date that no symbol-map entry covered, so the
  nearest entry was used; the effect was not measured.

## Decision for the owner

Phase 1 uses only our own database and files. Phase 2 (downloading company filings from the
SEC's free EDGAR site) starts only if you say go.
