# Data held on 2026-11-26

This is a dated inventory, not a promise that every source is suitable for every study.
A coverage label is a tag saying which stocks a load includes. A load number identifies one
import. The [measurements](database-inventory-2026-11-26-measurements.txt) give the exact
read-only queries and outputs. Sizes below include indexes; row counts from the load records
are accepted rows, rather than a recount of compressed data.

## In the database

| Name and load | What it holds | Rows | Dates | Coverage | How to read | Stored size |
|---|---|---:|---|---|---|---:|
| `kaggle_crsp.daily`, 1 | Daily US stock prices and returns | 68,879,302 | 1968-01-02–2024-12-31 ([source audit](archive/database-inventory-2026-11-23.md)) | Includes stocks that later stopped trading; no final delisting return | `kaggle_crsp.daily_of(1)` or `load_bars(source="combined")` | 2,019,598,336 bytes |
| `kaggle_chars.mcgill`, 2 | Monthly stock characteristics | 273,373 | 2000-01-31–2023-12-29 | Large stocks only | `kaggle_chars.mcgill_of(2)` | 372,580,352 bytes |
| `market.daily_bars`, 3 | Prices derived from CRSP | 68,879,302 | As CRSP load 1 | Includes stocks that later stopped trading | `market.prices_daily_sources` with a source filter; `load_bars(source="combined")` | 914,341,888 bytes |
| `public.ohlcv` Yahoo equities, no load number | Daily Yahoo prices | 5,777,834 | 2000-01-03–2026-11-05 | Yahoo equity rows; not a complete dead-stock history | `load_bars()` or `market.prices_daily_sources` with a source filter | Equity-only size not measured; shared table 13,345,120,256 bytes |
| `market.yahoo_daily_frozen_20261126`, no load number | Frozen copy of Yahoo equities | 5,777,834 | 2000-01-03–2026-11-05 | Same row count and dates as current Yahoo equities; contents not compared here | Inspect the frozen table; see [runbook](kaggle_import_runbook.md) for price readers | 780,673,024 bytes |
| `sec` loads 4–9 | Saved SEC responses, filing indexes, headers and study labels | 24,381,694 accepted across loads | Index 1999-01-04–2025-03-31; headers 2000-02-08–2004-11-19; labels 2000-02-01–2024-07-05 | Not a stock-price source; cached material is partial | `sec.*_of(load_id)`; use `sec.index_entries_complete` for filtered index reads | Index 5,755,314,176 bytes; other tables in measurements |
| `universe` loads 1–3 | S&P 500 membership history | Latest load records 1,269 intervals | Latest load 1996-01-02–2026-11-04 | S&P 500, not the whole market | `MembershipSource`, selecting `db` or `db:<load_id>` | Membership table 294,912 bytes; other tables in measurements |
| `ref` | Import, coverage, release and stock-link records | Not measured as a total | Not measured | Import records, not a stock source | Release and load records support the readers above | Individual tables in measurements |

For compressed tables, sizes use `hypertable_size()` and approximate counts use
`approximate_row_count()`, which accounts for compressed chunks. The shared Yahoo table size
also includes other venues. The stock price source view keeps sources separate; specify the
source to avoid mixing them. See the [import runbook](kaggle_import_runbook.md) for release
limits and the CRSP end-date guard. The [older inventory](archive/database-inventory-2026-11-23.md)
remains historical evidence.

## Downloaded but not loaded

The database load records above show what was imported. Sizes below are cited where an audit
states one; otherwise size is **not measured**. The exploration notes describe these as
candidates or limited sources, not approved database loads.

| Files | Size | Reason remaining as files |
|---|---|---|
| Gu–Kelly–Xiu characteristics | Not measured | Candidate for later use; not among completed loads. |
| Insider-trade download | Not measured | Candidate for later use; dates need cleaning. |
| SEC company facts | Not measured | Partial company coverage; the official bulk files are more complete. |
| SPY holdings | Not measured | Intended as a cross-check. |
| Quandl WIKI | 1.8 GB | Older dead stocks are missing and the copy's licence and provenance were not verified. |
| Stooq | US archive 629 MB | Stock files examined were survivors only and prices were already adjusted. |
| Small delisted samples | Not measured | Small, unaudited samples. |
| Fama–French and Shiller series | Not measured | Reference files; not among completed loads. The factor-zoo copy has invented extras and should not be used. |
| Intraday SPY r8 and r9 | 213 MB each (historical cited estimates) | Research files, not among completed loads; the later set is described in the [data folder README](/media/thesix/ggTrader-data/README.md). |

The [market-data audit](/media/thesix/ggTrader-data/market-data/README.md) documents Quandl,
Stooq and the small samples. The [Kaggle exploration](/media/thesix/ggTrader-data/market-data/kaggle/explore-2026-11-22/README.md)
documents the characteristics, insider, company-facts, holdings and factor-zoo files. These
file audits predate the completed database loads, so their statements about loaded status are
not current.

## Known unavailable data

The CRSP price-history copy lacks delisting returns, the final return when a stock leaves the market. Actual
bid–ask spreads, the gap between real buy and sell quotes, are not available in this copy.
CRSP volume before 2000 is unavailable in the held copy. See the
[market-data audit](/media/thesix/ggTrader-data/market-data/README.md) and
[import runbook](kaggle_import_runbook.md).

## Limits

Yahoo rows are overwritten in place when refreshed; a frozen copy exists. The SEC import
system retains at most two loads per kind of data. The edit guards on the three compressed
price tables are ordinary triggers, meaning database rules that reject normal edits but can be
bypassed by a privileged restore session. See the [runbook](kaggle_import_runbook.md#price-tables-side-by-side-and-their-edit-protection).

The intraday size estimates come from the [r8 validation report](../research/intraday/spy-momentum-feasibility/2026-10-25-intraday-r8-validation-unresolved.md)
and the [consolidation design](superpowers/specs/2026-11-25-database-consolidation-design.md).
The database also holds a separate insider-trade table; this page's downloaded-file entry does
not describe that table.
