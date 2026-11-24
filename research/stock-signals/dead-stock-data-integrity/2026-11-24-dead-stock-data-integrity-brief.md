# Brief: Dead Stock Data Integrity

**Status:** FROZEN 2026-11-24, before any real run. The rules, thresholds and tables below do
not change. Any later change is a dated amendment at the bottom of this file, never a silent
edit.
**Study:** stock-signals/dead-stock-data-integrity (kind: audit), written 2026-11-24.
**Decision fed:** which dead-stock loss figures a later rerun of the first test should use for
each stock, and whether the price rows around those stocks are clean enough to trust. The owner
makes that decision.

## 0. Registered parameters

These lines are checked by the study tool. Every cited figure in the report must use these
values or say which line it departs from.

- eval_start: 2005-01-03
- eval_end: 2024-12-31
- max_leverage: 1.0
- benchmark: none needed; no strategy is rerun by this study (SPY is the benchmark of the first test it audits)
- costs: 0 percent of each trade's value; no trades are simulated by this study (the first test charged 0.064, 0.183 and 0.40 percent of each trade's value per side)
- family_size: 1
- seed: none (nothing in this study is random)

`eval_start` and `eval_end` are the span of the first test (2005-01-03 to 2024-12-31; 2024-12-31
is the last day of the CRSP file). This study reads nothing later.

## 1. Question and fixed arms

**Question:** For the dead stocks the first test held, what do our CRSP rows and SEC filings say
about why they stopped trading, and how clean are the price rows around them?

(CRSP is the university stock-price database loaded into our own database; it includes stocks
that were later delisted. The SEC is the US regulator whose free EDGAR site holds company
filings.)

**Arms.** There is one list and one set of fixed rules. No strategy is rerun by this study.

**The roster.** The 141 dead-stock exits in the closed study's `_dev-A.json` (117) and
`_holdout-A.json` (24), read at the key path `result.death_price_paths.exits`. Of the 141, 138
are keyed `PERMNO-<n>` (a PERMNO is CRSP's permanent number for one stock) and 3 are Yahoo
tickers (`ARG`, `FRX`, `HSP`). The 3 Yahoo-ticker exits, and the permno-keyed stocks that have
no company name in the McGill table (8 of the 138 were counted on 2026-11-24), are kept in the
roster and labelled (`name_source = "none"`), never dropped. Names come from the McGill table
(`kaggle_chars.mcgill_of(2)`): the row with the latest date on or before the exit date; if
the stock has only later rows, its earliest row, labelled `name_source = "mcgill_later"`.
Tickers get their PERMNO from `ref.symbol_map` (release 1).

## 2. Data and freeze before calculation

- **Prices and names:** published combined release 1; CRSP load 1; McGill load 2. All are read
  through pinned function calls, never "newest". The database is read-only for this study.
- **Result files of the first test:** `_dev-A.json`, `_holdout-A.json` and the four trade files
  in `results/dead_stock_first_test/`, read as data. Nothing in the first test's folder is
  edited and none of its code is imported.
- **SEC filings (Phase 2):** free EDGAR files, downloaded to
  `/media/thesix/ggTrader-data/sec-delistings/` with a recorded SHA-256 for every file. Every
  request carries a User-Agent that names who is asking (the code reads it from the shell
  environment; it is never written into the repository).
- **Freeze date:** this brief is committed before any real run.

## 3. Registered model (the fixed rules)

### 3.1 Price-path labels (Task 2)

Descriptive labels for the last month of each dead stock's prices. They are assumptions written
here before any run and **cannot set a payout**. Constants: `WINDOW = 21`, `CALM_STD = 0.01`,
`COLLAPSE_BELOW = 0.5`, `DRIFT_BELOW = 0.9`, `CALM_BAND = (0.9, 1.1)`.

- Drop missing closes first (the "own valid closes" rule of the first test).
- `ratio` = last close divided by the close 21 valid closes earlier; `daily_std` = standard
  deviation of the last 21 daily simple returns.
- `too_short`: fewer than 22 valid closes.
- `collapse_like`: ratio below 0.5.
- `drifting_down`: ratio from 0.5 up to (not including) 0.9.
- `calm_like_deal`: ratio inside the calm band (0.9 to 1.1) and `daily_std` at or below 0.01.
- Anything else: `unclear`.

### 3.2 Integrity checks and thresholds (Task 3)

Each check works on rows with `permno, date, prc, ret, shrout`. CRSP prices can be negative
(a negative price means a bid-ask midpoint was used), so checks compare `abs(prc)` where sign
matters. No check changes its input.

- **Duplicates:** the same permno and date twice.
- **Bad price:** `prc` missing or zero while `ret` is not missing. A negative price is not bad.
- **Spikes:** a one-day return beyond plus or minus 0.5 whose compounded two-day return stays
  inside 0.9 to 1.1 (the move reversed at once); the spike day is flagged.
- **Gaps:** trading days missing inside a stock's own first-to-last span, in a run of at least
  5; the first missing day is flagged ("N days missing"). The calendar is the dates on which at
  least 1,000 stocks have a row.
- **Stale:** the same absolute price on 10 rows in a row; the first date of the run is flagged.
- **Shares:** the last row of a calendar month has a valid price but shares outstanding missing
  or zero or less (market value, and so the size ranking, would be unknown).
- **Holdings overlap:** a flag counts as inside a holding when its date lies inside an interval
  in which the first test's reference cell held that stock (a position opens when the running
  quantity first exceeds 1e-9 and closes when it returns to zero within 1e-9), for each of the
  four trade files (dev-A, dev-B, holdout-A, holdout-B). The reference cell is
  `loss0_cost18.3bps_taxlow`, confirmed against the trade file before use.
- The CRSP scan reads shares code 10 and 11, exchange codes 1 and 3, dates from 2004-01-01, in
  batches of 500 permnos.

### 3.3 SEC match window and evidence classes (Tasks 6-7)

- **Form 25 match (Task 6):** a Form 25 is the notice that a stock is taken off an exchange.
  A notice counts for a roster stock when its filer's normalized name is **exactly** equal to
  the stock's normalized McGill name and its filing date lies in the window **last bar minus 30
  days to last bar plus 180 days** (both edges included). Normalizing means uppercase,
  punctuation to spaces, collapsed spaces, and dropping trailing `INC`, `CORP`, `CORPORATION`,
  `CO`, `COMPANY`, `LTD`, `LLC`, `LP`, `PLC`. **No fuzzy matching in version one.** One distinct
  filer number is `matched`; two or more is `ambiguous` (never a pick); none, or a stock with
  no name, is `none`. Only notices whose security text names common or ordinary shares and does
  not name preferred, warrant, note, debenture or rights count.
- **8-K events (Task 7):** an 8-K is the filing a company makes when something material
  happens. For each uniquely matched stock, 8-K and 8-K/A filings within 90 days of the exit are
  read, and the evidence class is applied in this order:
  1. any item 1.03 (bankruptcy or receivership) gives `bankruptcy_filed`;
  2. else any item 2.01 (completion of an acquisition or disposition of assets), 3.03 (material
     change to the rights of shareholders) or 5.01 (change in control) gives `acquired_or_merged`;
  3. else item 3.01 (notice of delisting) gives `delisted_other`;
  4. else `no_8k_found`.
  The remaining classes are `no_filing_match` (Form 25 match status `none` or `ambiguous`) and
  `name_unknown` (no company name). Filings with empty item lists are kept and counted as
  `items_missing`.

## 4. Portfolio and accounting

No portfolio and no accounting: this study simulates no trades and computes no return. It lists
stocks, labels prices, flags rows and reads filings. The suggested losses of section 5 are
inputs for a later plan and are not applied here.

## 5. Frozen output plan

**Suggested-loss table (Task 8).** The loss is the share of the last price a holder is assumed
to lose when the stock stops trading.

| Evidence class | Exchange code | Suggested loss | Basis |
| :--- | :--- | ---: | :--- |
| `acquired_or_merged` | any | 0.00 | filing_evidence |
| `bankruptcy_filed` | 1 (NYSE), 2 (AMEX) | 0.30 | literature_fill |
| `bankruptcy_filed` | 3 (Nasdaq) | 0.55 | literature_fill |
| any other class | any | none; the later rerun keeps all three readings (1.0, 0.5, 0.0) | none |

The two literature figures are the missing-delisting-return fills of Shumway (1997) and
Shumway and Warther (1999); the report cites them with locators from `gg-library`.

**Outputs.** `_roster.csv` (the roster plus `ratio, daily_std, price_label`); `_integrity.json`
(per check: rows flagged, stocks flagged, 20 example rows, and counts of flags inside holdings
for each of the four trade files); `_payouts.csv` and `_payouts.json` (one row per roster stock:
evidence class, filing accessions, price label, suggested loss and basis, and a cross-check
that reads `conflict` when a `bankruptcy_filed` stock has label `calm_like_deal` or an
`acquired_or_merged` stock has `collapse_like`); a findings note after Phase 1; and one report
after Phase 2 in plain words, summary first, with no GO or NO-GO label.

**Statement: no strategy is rerun by this study.** The table is the input to a later plan, and
nothing here computes a return, a Sharpe ratio or a comparison with SPY.

## 6. Pre-registered decision rule

There is no pass bar and no GO or NO-GO label: this is an audit. The study succeeds when every
one of the 141 roster stocks appears in the final table with an evidence class and a reason
for any gap. A stock with no matching notice is reported as such, not as a failure. A check
that flags nothing is a clean result and is reported as one. Flags inside holdings are reported
with the stock and date; this study does not say how much any result would move unless the
data itself shows it.

## 7. Calls for the owner to check and execution boundary

- Calls for the owner: the suggested-loss table of section 5 (literature figures, not measured
  here), the exact-name rule (no fuzzy matching) and the 30 days before to 180 days after
  window.
- May touch: new study code and files in this study's folder only. No engine code, no
  paper-trading code, no database writes, no edits to the first test's folder. Heavy jobs run
  only through `scripts/run_capped.sh`, one at a time. The trader stays stopped.
- Phase 1 (Tasks 1-4) uses only our own database and result files. Phase 2 (Tasks 5-8, the SEC
  downloads) starts only after the owner's go.

## Dated clarifications

- **2026-11-24 (roster labels).** Every roster row gets a name-source label: `mcgill` (a McGill
  row on or before the exit date), `mcgill_later` (only later McGill rows exist) or `none`.
  Only rows with no McGill name get `none`. The three Yahoo-ticker exits (`ARG`, `FRX`, `HSP`)
  are not unnamed by default: each is turned into a PERMNO through `ref.symbol_map` (release 1)
  and is named when McGill has that PERMNO. In the real data all three are named (Airgas,
  Forest Labs, Hospira). The sentence in section 1 that groups them with the unnamed stocks
  means only that they are kept in the roster and labelled, never dropped.
