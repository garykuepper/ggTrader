# Data held on 2026-11-26

This is a dated inventory, not a promise that every source is suitable for every study.
A coverage label is a tag saying which stocks a load includes. A load number identifies one
import. The [measurements](database-inventory-2026-11-26-measurements.txt) give the exact
read-only queries and outputs. Database sizes include lookup structures that speed up
searches; row counts from load records are accepted rows, not a recount of compressed data.

CRSP is the university stock-price database from which the held copy came. The SEC is the
US Securities and Exchange Commission, which collects company filings.

## In the database

| Name and load | What it holds | Rows | Dates | Coverage | How to read | Stored size |
|---|---|---:|---|---|---|---:|
| `kaggle_crsp.daily`, 1 | Daily US stock prices and returns | 68,879,302 | 1968-01-02–2024-12-31 | `includes_dead_stocks`: includes stocks that later stopped trading; no final return after removal from the market | `kaggle_crsp.daily_of(1)` or `load_bars(source="combined")` | 2,019,598,336 bytes |
| `kaggle_chars.mcgill`, 2 | Monthly stock characteristics | 273,373 | 2000-01-31–2023-12-29 | `large_stocks_only`: large stocks only | `kaggle_chars.mcgill_of(2)` | 372,580,352 bytes |
| `market.daily_bars`, 3 | Prices derived from CRSP | 68,879,302 | 1968-01-02–2024-12-31 | `includes_dead_stocks`: includes stocks that later stopped trading | `market.prices_daily_sources` with a source filter; `load_bars(source="combined")` | 914,341,888 bytes |
| `public.ohlcv` Yahoo stocks, no load number | Daily Yahoo prices | 5,777,834 | 2000-01-03–2026-11-05 | Yahoo stock rows; not a complete dead-stock history | `load_bars()` or `market.prices_daily_sources` with a source filter | Stock-only size not measured; shared table 13,345,120,256 bytes |
| `market.yahoo_daily_frozen_20261126`, no load number | Frozen copy of Yahoo stocks | 5,777,834 | 2000-01-03–2026-11-05 | Same row count and dates as current Yahoo stocks; contents not compared here | Direct `SELECT` on the frozen table; no helper exists | 780,673,024 bytes |
| `sec` loads 4–9 | Saved SEC responses, filing lists, headers and study labels | 24,381,694 accepted across loads | Filing list 1999-01-04–2025-03-31; headers 2000-02-08–2004-11-19; labels 2000-02-01–2024-07-05 | `not_a_stock_source`: cached material is partial | `sec.*_of(load_id)`; use `sec.index_entries_complete` for filtered filing-list reads | Filing list 5,755,314,176 bytes; other tables in measurements |
| `universe` loads 1–3 | S&P 500 membership history | Latest load records 1,269 membership periods | Latest load 1996-01-02–2026-11-04 | S&P 500, not the whole market | `MembershipSource`, selecting `db` or `db:<load_id>` | Membership table 294,912 bytes; other tables in measurements |
| `market.stock_spans` | Stock history spans in a published release | 29,283 | 1968-01-02–2026-11-05 | Includes linked and unlinked stocks | Read alongside the published release | 4,587,520 bytes |
| `ref` | Import, coverage, release and stock-link records | 6,955 across ten tables; individual counts in measurements | Import records started 2026-11-23 and completed through 2026-11-26 | Import records, not a stock source | Release and load records support the readers above | Individual tables in measurements |

The SEC saved-response table has 706 records fetched from 2026-11-24 to 2026-11-25.
A database table's lookup structures speed searches and contribute to its stored size.
The `hypertable_size()` database function includes storage for all time-split pieces;
`approximate_row_count()` estimates their rows, including compressed pieces. The shared
Yahoo table size also includes other data providers. `market.prices_daily_sources` is a
saved database query that keeps price sources separate; filter by source to avoid mixing them.
The [import runbook](kaggle_import_runbook.md) explains release limits and the guard that
prevents certain reads after the end of the CRSP history.

## Downloaded but not loaded

The listed sizes are apparent folder sizes in bytes, measured with `du -sb`; the exact
commands and paths appear in the [measurements](database-inventory-2026-11-26-measurements.txt).
The planned next import stages B–D require their backup and owner-approval gates; the separate
stage 0b backup setup was outside the completed stage-A plan and its completion is not verified
here. The table explains each source's limitations and planned stage, not a claim that a
specific limitation alone caused deferral. The [exploration audit](/media/thesix/ggTrader-data/market-data/kaggle/explore-2026-11-22/README.md)
records the counts below; the [market-data audit](/media/thesix/ggTrader-data/market-data/README.md)
describes price-source coverage.

| Files | Bytes | Context and planned handling |
|---|---:|---|
| Gu–Kelly–Xiu, a set of stock characteristics | 3,836,203,389 | 4,117,300 stock-month rows; stage B candidate. |
| Insider trades, reports of company insiders' transactions | 41,796,717 | About 3.47 million trades; dates need cleaning; stage B candidate. |
| SEC company facts, company financial disclosures | 332,316,785 | About 82 million facts; coverage is partial; stage B candidate. |
| SPY holdings, the stocks held by the SPY fund | 411,011 | 26 yearly holdings files for cross-checking ([design](superpowers/specs/2026-11-25-database-consolidation-design.md#53-stage-b-the-kaggle-files-already-downloaded)); stage B candidate. |
| S&P 500 daily membership and features | 41,718,301 | Membership and events are a cross-check; some removed stocks have missing prices; stage B candidate. |
| Quandl WIKI, a historical price collection | 1,797,003,576 | Earlier dead stocks are missing and the copy's licence and origin were not verified; stage D candidate. |
| Stooq price files | 4,011,824,790 | Examined stock files only include stocks surviving to the download date; prices already reflect later splits and dividends; stage D candidate. |
| Arandkei delisted sample | 22,036,069 | Small unaudited sample of stocks removed from trading; stage D candidate. |
| Finzdata sample | 2,648,290 | Small unaudited sample of stocks removed from trading; stage D candidate. |
| Fama–French series, academic market and portfolio return series | 1,501,616 | Reference files; stage C candidate. |
| Shiller series, historical market and economic data | 1,675,613 | Reference files; stage C candidate. |
| Factor-zoo benchmarks, a separate collection of claimed market factors | 1,656,404 | Contains invented extras; the audit says not to use it. |
| SPY r8 and r9, two collections of prices during the trading day | 222,380,322 and 222,381,438 | Research files, not listed as a planned database import; r8 is under `/media/thesix/ggTrader-data/worktree-leftovers-2026-11-20/intraday_momentum_r8`. |

Other small exploration folders and their limits are listed in the exploration audit; the
planned stages do not imply that every folder will be imported. The database also holds
`public.form4_transactions`, a separate insider-trade table with 783,982 records.

## Data not available for free

The held CRSP copy has no delisting returns, the final return when a stock leaves the market;
the project has not found a free complete source. Actual historical bid–ask spreads, the gap
between real buy and sell quotes, and full-market daily trading volume are also absent from
the held free sources; this inventory does not establish whether any other free source exists. Neither `kaggle_crsp.daily` nor `market.daily_bars` has a volume
column for any year; Yahoo's daily stock table has a volume column and price rows from 2000-01-03 onward, with
incomplete market coverage. See the [market-data audit](/media/thesix/ggTrader-data/market-data/README.md)
and [research design](superpowers/specs/2026-11-21-all-stocks-honest-test-bench-design.md).

## Limits

Yahoo rows are overwritten in place when refreshed; a frozen copy exists. The import log
normally limits running or completed loads to two per dataset; the import code has an explicit
override, and the saved-response dataset `sec-cache` has used
both slots in loads 4 and 5. The import logic enforces this limit in
[loadlog.py](../src/ggTrader/data/kaggle_import/loadlog.py).

The edit guards on `kaggle_crsp.daily` and `market.daily_bars`, which store compressed price
history, and `kaggle_chars.mcgill`, an ordinary uncompressed table, are ordinary triggers:
these database rules reject normal edits but a privileged restore session can bypass them.
The shared `public.ohlcv` table has no user-defined edit-guard trigger. See the
[runbook](kaggle_import_runbook.md#price-tables-side-by-side-and-their-edit-protection).
