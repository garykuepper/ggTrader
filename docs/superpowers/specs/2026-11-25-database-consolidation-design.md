# Putting the research data in the database: design

Date: 2026-11-25. Status: **draft for the owner's approval. Nothing is built, loaded or changed.** Building, loading and every edit to the backup settings need the owner's go-ahead (section 6).

How it was made: from the owner's 2026-11-25 brainstorming session ("we should have all relevant data in a database for fast querying and to maintain data integrity"), built on the approved 2026-11-22 import design ([kaggle-import-design](2026-11-22-kaggle-import-design.md)) and its runbook ([kaggle_import_runbook](../../kaggle_import_runbook.md)). That design's load log, quarantine, checks and "version two" list are reused here, not rewritten.

## At a glance

- **What:** move the research data that still lives only in files into the `ggtrader` database, in four stages, each loaded, checked and reported on its own: **A** the SEC filing data and the per-stock labels the studies produced, **B** the downloaded Kaggle files not yet loaded, **C** the small reference series, **D** the extra price files (Quandl, Stooq).
- **Where:** the database stays on the main drive (fast SSD). The raw downloads stay on thesix, hash-checked. Backups go to thesix.
- **First, backups (stage 0):** the database is not in the nightly backup today, and the raw downloads have one copy. Stage 0 fixes both before any large load.
- **Every load gets a coverage label** ("includes stocks that later died", "survivors only", "unaudited sample"), so nobody mistakes a partial source for a complete one. A source is never left out for lacking dead stocks.
- **Nothing here changes a study result.** The studies keep reading their own pinned files. The database holds copies with a record of where each came from.

## 1. Intent and why

The owner wants the data a research question might touch to sit in one database, so that questions like "what happened to this stock?" or "what did analysts expect that month?" are a query, not a hunt through folders; and so that the data cannot quietly go wrong (a file edited, a load half done, a label with no source).

Today the 2026-11-22 import put the CRSP prices and McGill table in the database with full load records. The SEC downloads (about 700 files, 352 MB, two cache folders on thesix) and the labels the studies built from them are files. Each study carries its own copy of the SEC download code. The Kaggle sets named "version two" in the earlier design were downloaded but never loaded. The small reference series (Fama-French, Shiller, the leveraged-SPY inputs, the intraday SPY bars) sit as files in the repo or on thesix.

## 2. Goals, non-goals, anti-goals

Goals
- Every data set in stages A to D is in the database, traceable to the SHA-256 of the file it came from.
- A cheap, repeatable check proves the database still matches its files and its own records.
- Common questions return in about a second (targets in section 5.8).
- The database and the raw downloads are both backed up, and a restore has been tried.

Non-goals
- Changing any study result, re-scoring any study, or editing the studies' frozen SEC copies and their pins.
- Using any new data in a signal or a test. Loading is not exposing (section 5.9).
- Moving the database off the main drive; an off-site backup; any paid data; the paper trader (stays stopped).
- Reading the McGill measures or settling their date meaning (a separate small study).
- Joining Quandl or Stooq prices into the combined bars reader.

Anti-goals (a failure even if it loads)
- A database row whose file is missing or has changed.
- A survivors-only source that looks complete.
- A label with no study, method version or evidence behind it.
- The database filling the drive, or the nightly backup breaking.

## 3. Constraints

- **Disk.** The database is on the main drive's Docker volume: 455 GB drive, 143 GB free on 2026-11-25. The load library's free-space guard (default margin 40 GB) stays. Sizes of the new tables are not measured: the CRSP file of 68.9 million rows compressed to 1.9 GB, which is the only yardstick. Each stage is sized on a pilot before it loads fully.
- **Backup today.** Borg runs nightly at 02:00 and passed on 2026-11-25 (17 daily copies, stored on thesix at `/media/thesix/backups`). Its settings skip `/var/lib/docker/volumes/*` (the database) and `/media/*` (thesix, on the grounds that RAID is redundant). RAID survives one failed disk; it does not survive a deleted folder or a corrupted file. The only database dump is the one-off 2026-11-22 dump (9.7 GB), never restored.
- **Backup config edits** are a system change that needs `sudo`. The password is something only the owner can give; ask through the secret card at that moment.
- **Time stamps.** No new `timestamp without time zone` column (the 2026-08-20 day-shift lesson); dates are `date`, instants are `timestamptz`; SEC "accepted" times are parsed as `America/New_York` explicitly.
- **SEC politeness.** Requests carry the owner's contact line from `~/.config/ggtrader/sec_contact` and stay under five a second. The contact line is never written to the repo, the database, a log or a delegate's prompt.
- **One heavy job at a time**, through the capped-run wrapper; tests only through `scripts/run_tests.sh` on small paths. Research code reads the database read-only; the importer is the only writer.

## 4. Approaches

**How much to load.** (A) SEC only; (B) A plus the version-two Kaggle files; (C) B plus the small reference series; plus (D) the price files with open, high, low and volume. *Picked by the owner: all of it, in stages, each labeled with its coverage.* I first recommended A alone, then dropped the focus on dead-stock coverage when the owner pointed out that every set can help.

**Where the database lives.** (1) main drive, backed up to thesix; (2) whole database on thesix; (3) split, with the big raw tables on thesix. *Picked by the owner: (1).* thesix is three spinning disks in RAID5: roomy and failure-tolerant, but slower for the scattered reads a database does, and slower to write.

**How to back up.** The borg nightly run already exists, has a retention history and a consistency check, and stores on thesix. *Picked: add a database dump to the borg run and add the raw data folders, rather than build a second backup system.* Fallback if nightly dumps do not shrink well in borg: a weekly dump kept on thesix.

**How to load.** Reuse the 2026-11-22 load library (`ggt db import`, load log, chunks, quarantine) for each dataset rather than write a loader per study. *My call.*

**Raw SEC filings.** Keep the files on thesix (hash-named, as now); the database holds the path, hash and parsed fields. Putting full filing text in the database would add tens of gigabytes to the fast drive for text read once. *My call.*

## 5. The design

### 5.0 Stage 0: backups first

1. **Measure.** Time and size a full dump of the live database (a TimescaleDB database needs its special pre-restore and post-restore steps when restored; the builder follows the product's procedure). The one-off 2026-11-22 dump of the whole 17 GB database was 9.7 GB. About 12 GB of the database is the parked crypto bars; they are included in the nightly dump (they may be the only copy). If the dump is too heavy for nightly use, the crypto bars move to the weekly dump and the owner is told.
2. **Database into borg.** Use borgmatic's built-in PostgreSQL dump so each nightly archive carries a consistent dump (the database runs in Docker, so the dump command goes through `docker exec`). Check on two consecutive nights how much the second night adds. If it adds nearly a full copy, switch to a weekly dump on thesix with the last four kept.
3. **Raw data into borg.** Add the folders that hold the only copy of a download: `market-data`, `sec-delistings`, `sec-early-window`, `intraday_momentum_r9`, `refactor-evidence`. Leave out the old dumps (`ggtraderdb-*`), `project-cleanup-*` and `worktree-leftovers-*`. Sizes measured 2026-11-25: `market-data` 17 GB, `sec-delistings` 265 MB, `sec-early-window` 87 MB, `intraday_momentum_r9` 213 MB, `refactor-evidence` 102 MB, about 18 GB in all, against 4.3 TB free on thesix (the borg repository is there too). Borg shrinks repeated copies, so retention does not multiply the 18 GB. The settings currently exclude `/media/*`, so the builder finds the include mechanism that works with borgmatic's patterns.
4. **Prove it.** Run borgmatic's dry run first, then one watched real run. Confirm the 02:00 job still ends with "no problems". Keep the old config as `.bak` for rollback.
5. **Restore drill.** Restore the newest dump into a scratch database (never over the live one), compare row counts for every schema and a sample of hashes with the live database, record the result, drop the scratch database.
6. **Freshness.** Add a line to the existing health check (FlynnRX) that warns when the newest database dump is older than two days, if that fits without a rewrite. The builder proposes.

Stage 0 ends when a restore has worked. Large loads (stages B to D) do not start before that.

### 5.1 Coverage labels

A small append-only table `ref.load_coverage (load_id, coverage, note)`, written by every importer, instead of changing the existing load log (which has append-only protection). Values are a short fixed list: `includes_dead_stocks`, `survivors_only`, `large_stocks_only`, `unaudited_sample`, `not_a_stock_source`. Loads 1 to 3 get their rows too: CRSP "includes_dead_stocks; no delisting returns", McGill "large_stocks_only, 2000 to 2023". A load without a coverage row fails the integrity check.

### 5.2 Stage A: the SEC data and the labels

**Scope: load what is already downloaded and labeled; make no new SEC requests.** Stage A covers the roughly 700 cached files (352 MB) and the stocks the two studies labeled (141 held stocks from the 2005 to 2024 study, 251 stocks that stopped trading in 2000 to 2004), not all 25,754 CRSP stocks. Fetching more (further quarterly indexes, the merger-document search, other stocks) is a later study with its own brief and the owner's approval; the shared downloader below is built so that study can use it, but stage A does not run it.

A new schema `sec` with:
- `sec.fetched_files`: one row per downloaded file (hash, URL, fetch time, status, size, path on thesix, kind: quarterly index, company submission list, filing header, filing document).
- `sec.index_entries`: the quarterly index rows (company number, name, form type, date filed, file name, quarter) for the cached quarters (1999 to 2006, 32 files).
- `sec.filing_headers`: parsed header facts (accession number, company number, form, filed date, the 8-K item list).
- `sec.company_matches`: for each labeled stock (those 141 and 251), the company number found, how (name only, or confirmed by a filing's own header), and whether it was ambiguous.
- `sec.death_labels`: one row per stock, label and method version. Loaded from the two studies' result files (`dead-stock-data-integrity`, `dead-stock-early-filings`), each row carrying the study folder, the result file's hash and the evidence accession numbers. Stocks are identified as in the existing studies (CRSP stock number plus last trading day) and must exist in the stock list. The suggested loss figure is stored as a scenario, with its label, exactly as the studies defined it.

Labels are append-only: a later method (for example a search of merger documents) adds rows under a new method version, never overwrites.

One shared SEC downloader goes under `src/ggTrader/data/` for new loads, writing the same hash-named cache layout; the studies' frozen copies are left alone.

### 5.3 Stage B: the Kaggle files already downloaded

As scheduled in the 2026-11-22 design, in this order, each with its own load, pilot and checks:
1. **Gu-Kelly-Xiu characteristics** (4.1 million rows, 94 measures per stock-month, 1957 to 2021) → `kaggle_chars.gkx`.
2. **SEC company facts** (82 million facts, 2009 to 2026, 6,876 companies) → `kaggle_sec.facts` and `kaggle_sec.filings`, keyed as the earlier design says (`adsh` is not unique).
3. **Insider trades** (3.47 million) → `kaggle_insider.form4`; the builder states how it relates to the existing `public.form4_transactions` (784k rows, untouched).
4. **S&P 500 daily membership and changes** → `kaggle_sp500`; cross-check of the membership list.
5. **SPY holdings by year** → `kaggle_sp500.spy_holdings`.

### 5.4 Stage C: small reference series

- **Fama-French** monthly and daily factors and the portfolio files, parsed from the zipped files into one long table.
- **Shiller** monthly series from 1871.
- **Leveraged-SPY inputs.** Load only what the database does not already have. The 3-month Treasury rate (`DTB3`) is already in `fred_series` to 2026-09-28, and SPY is already in `ohlcv`; for overlaps store a comparison check, not a second copy.
- **Intraday SPY bars** (Alpaca, 2016 to 2026, manifest-pinned) → a minute-bar table.
- **Study results index** (optional, last): one row per result file (study, path, SHA-256, commit, date). It lists results; it does not copy them.

### 5.5 Stage D: price files with open, high, low and volume

- **Quandl WIKI** (1.8 GB, 3,199 tickers to 2018, raw and adjusted, with dividends and splits) → `kaggle_quandl`.
- **Stooq US daily** (9,450 stocks and about 3,900 ETFs; prices already adjusted, no raw series) → `stooq`. Coverage `survivors_only`. The world and macro files wait.
- **The small sets** (delisted samples, the 2008 to 2009 named-company file) → `unaudited_sample`.

These are separate source tables, not merged into the combined bars. Linking them to the CRSP stocks is a later study's job.

### 5.6 How a load works and what proves it

Each load follows section 8 of the 2026-11-22 design: file hash against the download's checksum list, chunked loading with a complete-or-invisible rule, quarantine of bad rows with the raw value and reason, `parsed = accepted + quarantined`, a free-space reserve, and a load-log row with the loader's commit. Each importer adds: a coverage row, and a stored check result.

### 5.7 The integrity check

`ggt db verify-data` (read-only) re-runs the cheap checks and stores the result: counts in each table equal the load log; every file the database names still has its recorded hash; no incomplete load; every load has a coverage row; every `sec.death_labels` stock exists in the stock list; every cited filing exists in `sec.fetched_files`. The ordinary run is cheap: it checks that every named file exists with its recorded size, and re-hashes only the files loaded in that run. A separate `--deep` run re-hashes every file the database names (the large Kaggle and Stooq downloads sit on spinning disks, so it takes a long time). The deep run takes the one-heavy-job lock, runs monthly at low priority outside the 02:00 backup window, and is a timer the builder proposes.

### 5.8 Speed targets (mine, to be measured and reported)

- All filings of one company for one year, and one stock's label with its evidence: under 1 second.
- Company facts known on a given date for one company: under 1 second.
- One month of characteristics for every stock: under 5 seconds.
If a target misses, the builder adds the index that fixes it and reports the before and after times.

### 5.9 When the information became usable

Loaded is not exposed. Each set keeps the 2026-11-22 design's section 7 rules: GKX measures usable from the first trading session of the month they predict; SEC facts by the filing's accepted time (units still undecided, so none exposed); insider trades and S&P files not exposed until their date meaning is shown; McGill unchanged. SEC labels are descriptions of how a stock ended, not signals. Quandl and Stooq carry no timing risk but no exposure to signals without a study naming them.

### 5.10 Documents to update

`docs/cli_reference.md` (new import commands, `verify-data`), the database map in `CLAUDE.md`, `docs/architecture.md`, `docs/changelog.md`, the thesix `README.md` (what is backed up and how), and the import runbook (stages, owner gates).

## 6. Order, gates and reporting

Order: stage 0, then A, B (1 to 5 in order), C, D. The owner approves each stage's start. Each stage ends with a short plain-words report: rows loaded, rows quarantined, database size and free space, check results, timings, what each tool's use cost. The plan sizes itself to the owner's quota (ask before planning).

Gates: (G-backup) the owner approves the backup edit and supplies `sudo` through the secret card; (G-restore) the owner sees the restore drill result before the first large load; (G-load) the owner approves each stage's real load after its pilot.

**Stage A may load before the restore drill.** It is small (about 350 MB of files and a few thousand label rows) and can be rebuilt from the cached files and the studies' result files in the repo, so losing it costs little. Stages B to D wait for the restore drill, because they are large and cannot be rebuilt quickly. Stage A therefore does not sit idle behind the `sudo` wait; it still needs the owner's go-ahead for its real load.

## 7. Left to the builder

Schema and column names beyond those named here; column types (measure the real key of the SEC facts); chunk sizes and compression; the CLI shape; the SEC name-match storage details; how to parse the Fama-French and Shiller files (multi-section text, missing-value markers); the borgmatic include mechanism; whether `verify-data` runs on a timer; how the SEC downloader shares the cache with the studies' copies.

## 8. Calls I made (for the owner to check)

- Reuse the existing import library and load log.
- Raw SEC files stay on thesix; the database holds hashes, headers and labels.
- Coverage labels in a new table, not by editing the append-only load log.
- Load only what is not already in the database; compare overlaps.
- Stage order, and the order inside stage B.
- A dump through borgmatic first, weekly dump as the fallback.
- The study results index is optional and last.
- The studies' frozen SEC copies and results are untouched and nothing is re-scored.
- No off-site copy (a known gap, not planned here).
- Stage A loads only what is cached and labeled; no new SEC requests until a later study asks (5.2).
- The nightly dump includes the parked crypto bars, with a weekly fallback for them (5.0).
- The ordinary integrity check is cheap; the full re-hash is a separate monthly `--deep` run (5.7).
- Stage A may load before the restore drill; stages B to D may not (section 6).

## 9. Open items that are the owner's

- Approve this design.
- At stage 0: the backup edit and `sudo`, and the restore result.
- Whether to keep the study results index (5.4) at all.
- Say if the old 14 GB February dump (`ggtraderdb-dump-20260218`) can ever be retired; this design leaves it alone.
