# Dead-stock data integrity: Phase 1 findings (2026-11-24)

**Study:** stock-signals/dead-stock-data-integrity (an audit; no strategy was rerun).
**Source of the numbers:** every number in the tables and counts comes from the current
`_integrity.json` and `_roster.csv` in this folder, written by one run of `code/run.py` on
2026-11-24. The few figures that do not (figures from an earlier, wrong version of the scan, and
checks made by hand in the database) are listed in the last section, "Earlier measurements and
checks made by hand", each with how it was obtained. The commands and outputs for those are saved
in [`_hand_checks.txt`](_hand_checks.txt).

## Words used here

- **CRSP:** a university database of US stock prices that keeps stocks that were later closed
  out or bought. We loaded it into our own database (load 1, release 1).
- **McGill:** a second table (load 2) that gives each stock's company name.
- **Permno:** CRSP's permanent number for one stock. It stays the same when a ticker changes.
- **Ordinary common share:** a normal share of stock in a company (not a preferred share, a
  fund or a warrant). CRSP marks these with share codes 10 and 11.
- **Exchange code:** CRSP's code for where a stock trades: 1 is the New York Stock Exchange, 2 is
  the American exchange (NYSE American) and 3 is Nasdaq.
- **The first test:** the closed study `dead-stock-first-test`. It "held" a stock on a day when
  its simulated portfolio owned shares of it. It ran two stock lists: **A**, the main list, and
  **B**, a wider and harsher list (the "stress universe": any stock priced at least $1, with no
  size limit). It scored two periods: **dev** (the development period, used while the rule was
  being built) and **holdout** (the later period, scored once). So "dev-A" means the main list
  in the development period and "holdout-B" means the wider list in the holdout period.
- **Flag:** a row or stretch of rows that one of six fixed checks marks as suspicious. A flag is
  not proof of an error: it says "look here". The rules were frozen in the brief before the run.
- **Inside a holding:** the flag's date lies between the day the first test bought the stock and
  the day it sold it. The trade files hold many settings; this note uses one, named
  `loss0_cost18.3bps_taxlow`: the setting where dead stocks pay back their full last price, with
  a trading cost of 18.3 bps (basis points; one basis point is one hundredth of a percent, so
  18.3 bps is 0.183 percent) per side and the lower tax rates. It is the same one for all four
  files. Only the flag's **one date** is tested: for a gap that is the first missing day and for
  an unchanged-price stretch it is the first day of the stretch. A gap or stretch that starts
  outside a holding and runs into one is not counted, so the "inside a holding" counts for gaps
  and unchanged prices are a lower bound (at least this many; the true count could be higher).
- **Symbol map:** the table that says which ticker belonged to which permno on which dates. It is
  how a trade file's ticker (such as `AAPL`) is turned into a permno.
- **Commit:** a saved version of the code in its version-control history.

## Short version

- **What was scanned.** 20,379,285 daily rows of 9,422 stocks (ordinary common shares that traded
  on the NYSE or Nasdaq at some point from 2004-01-01 to 2024-12-31) in the CRSP data. That is not
  "the whole database": the Yahoo side of our combined price data was **not** scanned at all, and
  neither were stocks that never traded as an ordinary share on those two exchanges since 2004.
- **What the six checks found.** Duplicate rows, missing or zero prices and missing share counts:
  nothing, by these narrow checks (see the next point). The other three found 345 one-day spikes,
  251 gaps and 249 stretches of an unchanged price. Gaps and unchanged-price stretches are
  counted as **runs**, not as days. A gap run can be very long: the 20 largest gap examples alone
  add up to 52,895 missing stock-days (each of the 20 is at least 1,798 days). So 251 is a count
  of events and says little about how many days are missing.
- **What "found nothing" means.** The missing-price check looks only at rows where the price is
  empty or zero **and** the day's return is present. The missing-share-count check looks only at
  the last present row of each stock-month that has a valid price. "Clean" here means "nothing
  found by these narrow checks", not "the data is correct". A duplicate row cannot exist in the
  loaded table by construction (its primary key is load, permno and date), so that zero is not
  evidence of anything.
- **The file looks pre-cleaned.** A direct count over all 21,392,301 rows from 2004-01-01 found
  no empty prices, no zero prices, no empty returns and no empty or non-positive share counts
  (see the last section); the importer also counted 0 non-finite values. Real price histories
  normally contain some missing values, so this looks like a file from which such rows were
  removed before upload. That is an inference from zero counts, not something shown.
  If true, a short hole of a few missing days is invisible to us, because the days are simply not
  in the file.
- **Held flags.** Only **two** flags, out of 845, fall on a day the first test actually held that
  stock. Both are spikes in the wider-list (B) holdout run. The main (A) runs and the B
  development run have none. (An earlier version of this scan also showed a held gap; it was a
  mistake in our own scan, explained in the last section.)
- **Which stocks are labelled.** The 141 dead stocks listed here are those held in the **main
  (A) lists only** (117 in dev and 24 in holdout). The first test also held more dead stocks in
  the wider (B) lists: 107 in dev-B and 35 in holdout-B (its report, section 4:
  [2026-11-24-dead-stock-first-test.md](../dead-stock-first-test/2026-11-24-dead-stock-first-test.md)).
  Those were not named or labelled here.
- **Names and labels.** Of the 141, 133 have a company name. All 141 have a price label: 107 look
  "calm like a deal" in their last month and 34 are "unclear". None looks like a collapse.
- **What the labels are not.** "Calm like a deal" only describes how the price moved in the last
  month (little change, small daily swings). It is **not** evidence of what holders were paid.
  The plain facts: the 141 last prices run from $7.18 to $310.00 (middle value $61.28), so they
  are ordinary prices, not pennies. The 34 "unclear" stocks all had daily swings above 1 percent
  (the smallest was 1.06 percent) and a last price of at least 0.947 times the price 21 trading
  days earlier (the highest was 1.52 times); 12 of them ended more than 10 percent above the
  earlier price.
- This note does not say how much any result would change. That was not measured.

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
| Price missing or zero (with a return present) | 0 | 0 | 0 | 0 | 0 | 0 |
| Spikes that reverse next day | 345 | 253 | 0 | 0 | 0 | 2 |
| Gaps of 5 or more missing days | 251 | 222 | 0 | 0 | 0 | 0 |
| Same price on 10 rows in a row | 249 | 158 | 0 | 0 | 0 | 0 |
| Share count missing at month end | 0 | 0 | 0 | 0 | 0 | 0 |

Positions the first test opened (reference setting): dev-A 2,982, dev-B 3,268, holdout-A 993,
holdout-B 1,072. They cover 1,366, 1,805, 609 and 727 different stocks. The two B files are the
wider ("stress") list and are shown separately; no files were merged.

How the trade files were matched to stocks: some trade rows use a Yahoo ticker such as `AAPL`
and some use `PERMNO-<n>`. Tickers were turned into permnos through the symbol map, using the
date of the trade (the same rule the roster uses). All ticker rows were mapped: 5,659 (dev-A),
1,192 (dev-B), 2,197 (holdout-A) and 445 (holdout-B), with **0 unmapped**. Two details: 45, 183,
0 and 8 rows use a name like `PERMNO-38420~1` (a later period of the same stock) and were folded
into that stock; and 266 dev-A and 189 dev-B ticker rows fell on a date that no map entry
covered, so the nearest entry was used (0 in the holdout files).

### What each check could and could not change

- **Duplicates, missing prices, missing share counts (all 0):** nothing found by these narrow
  checks (see "What 'found nothing' means" above); not a proof that the data is correct.
- **Spikes (345):** the price jumped by more than half in one day and came straight back the
  next, which may mean a wrong price was recorded for that day. A spike inside a holding could
  change one day's profit or loss for that position, and a spike could change which stocks rank
  high that day, whether or not the stock was held. Whether it did was not measured. The largest
  spikes repeat on a few permnos with very similar sizes: in the 20 largest examples, permnos
  13958 and 15293 appear four times each, and 14153, 16465 and 47387 twice each; every one is a
  rise of about 413 to 457 percent followed by a fall of about 80 to 82 percent. That may be one
  recurring quirk in the data rather than twenty separate events; the cause was not checked.
- **Gaps (251):** days with no row. Many are long (the seven largest are over 3,000 trading
  days, roughly twelve years or more, where a stock disappears from the file and returns). A
  trading halt or time away from the exchanges of 5 or more trading days would also show up as a
  gap, if the file has no rows for those days. No gap starts on a day inside a recorded holding.
  Missing days could still matter in ways this overlap test does not see (for example a gap that
  begins before a holding, or missing history that changed which stocks were picked), and any
  effect on the first test's trades or results was not measured.
- **Same price on 10 rows (249):** can be an error, but can also be real: some stocks trade at
  an almost fixed price for weeks for real reasons (a stock waiting to be bought out is the usual
  example). The longest examples cluster near $10 to $11: of the 20 longest stretches, 12 are at
  prices between $9.575 and $11.065 (the longest, 34 rows, is permno 22582 at $10.595; next 31
  rows at $11.05 on permno 23451, 29 rows at $13.815 on permno 12221, 24 rows at $10.735 on
  permno 22574). The cause of these particular ones was not checked. None starts inside a
  holding; any effect on trades or results was not measured.
- **None of these checks** says what a holder received when a stock was closed out. That is the
  question Phase 2 (company filings) is for.

## The largest flags inside holdings

Only two exist, so these are all of them. "Inside a holding" is taken from the first test's
trade files; this output does not give the position's start, end or size, so the effect on its
profit is **not measured**. Both are in the B (wider list) holdout file.

1. **Spike, permno 16680, 2024-07-24**: the price rose 56.3 percent, then fell 32.4 percent the
   next day.
2. **Spike, permno 14873, 2020-06-10**: the price rose 51.7 percent, then fell 30.3 percent the
   next day.

The gap on permno 92449 that an earlier version of this note listed as a held flag is gone. It
was never a hole in the data (see the last section).

For comparison, the largest flags anywhere (held or not) are a spike on permno 14153 on
2013-10-03 (a rise of 456.8 percent, then a fall of 81.9 percent), a gap on permno 77263 from
2005-12-27 (4,253 days missing) and an unchanged price of 10.595 on permno 22582 for 34 rows from
2023-07-21. The largest spike and stale examples are the same as in the earlier version; the gap
list changed (permno 81761, 3,865 days, no longer appears among the largest, and permnos 89986
and 88937 now do).

## Limits of this note

- The stocks scanned are still chosen by the brief's rule (ordinary common shares, NYSE or
  Nasdaq). A stock that never had such a row since 2004 is not scanned at all, and the Yahoo
  side of the combined data was not scanned.
- A flag inside a holding says only that the flag's one date falls in the holding. It is not an
  estimate of harm, and an absence of such flags does not show that the first test's results
  were unaffected.
- **The spike check can never flag a stock's final trading day**, because it needs the next
  day's return. That matters for dead stocks, whose last day is the day that counts most. A bad
  last price would not be caught by this check.
- **The spike and unchanged-price checks compare neighbouring rows, not neighbouring trading
  days.** If days are missing from the file, a gap can join two prices that were not on
  consecutive days, and a change across it is treated like a one-day move.
- The large gaps were not all looked up one by one; one was (see the last section).
- 266 dev-A and 189 dev-B ticker rows fell on a date that no symbol-map entry covered, so the
  nearest entry was used; the effect was not measured.
- The counts come from a run made on a committed version of the code (a saved version in the
  version-control history). The file records that version's identifier, a "files changed since
  that saved version" flag (false for this run) and a fingerprint (SHA-256, a code that changes
  if a single character of a file changes) for each code file.

## Earlier measurements and checks made by hand

These figures are **not** in the current `_integrity.json`. The commands and outputs are in
[`_hand_checks.txt`](_hand_checks.txt); the section numbers below are that file's.

- **The earlier, wrong scan (section 4 there).** The first version of the scan kept only the rows
  where a stock was an ordinary share on the NYSE or Nasdaq on that day, so a stock that spent
  some years on another exchange lost those days and the gap check reported the hole. It was
  corrected the same day (a dated clarification is in the brief). Its figures, read from the
  saved file of that version (commit `ba4b7f83`): 220 gaps on 195 stocks, 342 spikes on 251
  stocks and 239 unchanged-price stretches on 152 stocks, and one gap inside a holding (permno
  92449, "1,600 days missing"), and permno 81761 with "3,865 days missing" among the largest.
  The corrected scan has 251 gaps on 222 stocks, 345 spikes and 249 unchanged-price stretches,
  and no held gap. The gap count went up because stocks are now followed over their whole
  history, not only the filtered years; I did not trace each new gap.
- **Permno 92449 (section 1 there).** A direct database query with no row filter shows 2,230
  rows on exchange code 3 (Nasdaq, 2007-11-28 to 2023-02-15) and exactly 1,600 rows on exchange
  code 2 (NYSE American, 2012-12-31 to 2019-05-09). The old "1,600 days missing" starting
  2012-12-31 is those rows, so it was a mistake caused by our own filter and not a hole in the
  data.
- **Permno 77263 (section 2 there).** The same kind of unfiltered query shows rows every year
  from 1992 to 2005 (248 in 2005), none from 2006 to 2021, then 30 rows in 2022 and full years
  after. So the 4,253-day gap in the current file is a stretch with no rows in the file. The cause
  was not determined: it may have left the exchanges and come back under the same permanent
  number, or the file may simply lack those days. I do not call it a data error.
- **Missing and zero values (sections 5 and 6 there).** Over all 21,392,301 rows from 2004-01-01
  (every stock, any exchange): 0 empty prices, 0 zero prices, 0 empty returns, 0 empty share
  counts and 0 share counts at or below zero. The importer's own count of "nulled values" for
  load 1 is 0: the importer replaces any number that is not a finite number (NaN or infinity)
  with an empty value and counts them, so 0 means the uploaded file had none. The scan in this
  note read 20,379,285 of those rows (the 9,422 selected stocks).
- **The sum of the 20 largest gap examples (section 7 there).** 52,895 missing stock-days, added
  up from `_integrity.json`; only the 20 largest are stored, so the true total over all 251 gaps
  is at least this.
- **Reading the price table (section 3 there).** A first probe, run through the database's
  read-by-load function (`daily_of()`) with a filter on the stock-spans table, was still running
  after 12 minutes 15 seconds and 12 minutes 19 seconds when I looked at the database's list of
  running queries (`pg_stat_activity`), so I stopped it. Rerun later with a 60-second limit it was
  stopped at 60 seconds; the same question without the spans filter finished in 25.4 seconds, and
  reading the table directly by load id finished in 0.7 seconds. The cause of the slowness was
  not determined, so "over 12 minutes" is a lower bound (at least that long) from one
  observation. The scan reads the table by load id, after the same "load is complete" test that
  `daily_of()` makes.

## Decision for the owner

Phase 1 uses only our own database and files. Phase 2 (downloading company filings from the
SEC's free EDGAR site) starts only if you say go.
