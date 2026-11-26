# Putting the research data in the database: design

Date: 2026-11-25 (version 2, after independent review). Status: **draft for the owner's approval. Nothing is built, loaded or changed.** Building, loading and every edit to the backup settings need the owner's go-ahead (section 6).

How it was made: from the owner's 2026-11-25 brainstorming session ("we should have all relevant data in a database for fast querying and to maintain data integrity"), built on the approved 2026-11-22 import design ([kaggle-import-design](2026-11-22-kaggle-import-design.md)) and its runbook ([kaggle_import_runbook](../../kaggle_import_runbook.md)). Version 1 was read by a fresh builder-check reviewer, then reviewed by Gemini 4 Neon (accuracy against the files and the machine) and GPT-6.2 Astra (what could go wrong). The assistant checked their main findings against the files before accepting them. They changed the backup plan, the SEC scope, the Stage B order and several rules; section 8 lists what changed.

## At a glance

- **What:** move the research data that still lives only in files into the `ggtrader` database, in stages, each loaded, checked and reported on its own: **0** backups, **A** the SEC filing data and the per-stock labels from the studies, **B** the Kaggle files not yet loaded, **C** the small reference series, **D** the extra price files (Quandl, Stooq).
- **Where:** the database stays on the main drive (fast SSD). The raw downloads stay on thesix, hash-checked. Backups go to thesix.
- **First, backups.** There is no dump of today's database (the only dump predates the Kaggle load) and the raw downloads have one copy. Stage 0a takes a fresh full dump and proves a restore. Nothing loads before that.
- **Every load gets a coverage label** ("includes stocks that later died", "survivors only", "partial", "unaudited sample"), so nobody mistakes a partial source for a complete one. A source is never left out for lacking dead stocks.
- **Nothing here changes a study result.** The studies keep reading their own pinned files. The database holds copies with a record of where each came from.

## 1. Intent and why

The owner wants the data a research question might touch to sit in one database, so that questions like "what happened to this stock?" or "what did analysts expect that month?" are a query, not a hunt through folders; and so that the data cannot quietly go wrong (a file edited, a load half done, a label with no source).

Today the 2026-11-22 import put the CRSP prices (CRSP is the university-built price history of all US stocks, dead ones included) and the McGill table in the database with full load records. The SEC downloads (1,415 files, 352 MB, in two cache folders on thesix) and the labels the studies built from them are files. Each study carries its own copy of the SEC download code. The Kaggle sets named "version two" in the earlier design were downloaded but never loaded. The small reference series sit as files in the repo or on thesix.

## 2. Goals, non-goals, anti-goals

Goals
- Every data set in stages A to D is in the database, traceable to the SHA-256 (a fingerprint of a file's exact bytes) of each file it came from.
- A repeatable check proves the database still matches its files and its own records.
- Common questions return in about a second (targets in 5.8).
- The database and the raw downloads are both backed up, and a restore has been tried.

Non-goals
- Changing any study result, re-scoring any study, or editing the studies' frozen SEC copies and their pins.
- Using any new data in a signal or a test. Loading is not exposing (5.9).
- Moving the database off the main drive; an off-site backup; any paid data; the paper trader (stays stopped).
- Reading the McGill measures or settling their date meaning (a separate small study).
- Joining Quandl or Stooq prices into the combined bars reader.
- Still deferred from the earlier "version two" list, as before: translating old tickers to current ones, immutable copies of the Yahoo rows, and stricter database roles (but see the open item on roles in section 9).

Anti-goals (a failure even if it loads)
- A database row whose file is missing or has changed.
- A survivors-only source that looks complete.
- A label with no study, method version or evidence behind it, or a label attached to the wrong stock history.
- The database filling the drive, two heavy jobs running together, or the nightly backup breaking.

## 3. Constraints and facts found on 2026-11-25

- **Disk.** The database is on the main drive's Docker volume: 455 GB drive, 143 GB free. thesix has 4.3 TB free. The load library's free-space guard (default margin 40 GB) stays and is checked against the database's own drive before every chunk (not the download drive). Peak use during a load (old and new versions, indexes, temporary files, write-ahead log) is more than the final size, so each stage is sized on a pilot first.
- **The nightly backup.** Borg (the backup program) runs at 02:00 from `/etc/borgmatic/config.yaml`, a file only root can read. It passed on 2026-11-25 (17 archives kept, stored on thesix at `/media/thesix/backups`). It took 41 minutes and the service has a 2-hour limit. **The assistant had read a different file, `~/borgmatic_config.yaml` (not the one the timer uses), so the active file's exclusion rules are unverified** until it is read with `sudo`. The home copy skips Docker volumes (the database), `/media/*` (thesix) and `~/ggtraderdb`.
- **No dump of today's database.** The 2026-11-22 dump (9.7 GB, `pg_dump` custom format, took about 28 minutes) was taken before any Kaggle data was loaded; it cannot recover the CRSP, McGill, derived-bars tables or the published release. The older February dump is older still. A full restore has never been run.
- **The importer's own rule.** For the live `ggtrader` database, `ggt db import` refuses `--no-backup-check` and requires a `*.dump` file of at least 1 MB, at most 7 days old, in a named folder (`guards.backup_fresh`). A dump held only inside Borg does not satisfy it, and the 2026-11-22 dump goes stale on 2026-11-29.
- **Borgmatic's restore command** passes `--clean --if-exists --dbname ggtrader`, which drops live objects first. A restore through borgmatic must never be used for a drill.
- **Backup config edits** need `sudo`. The password is something only the owner can give; ask through the secret card at that moment.
- **Time stamps.** No new `timestamp without time zone` column; dates are `date`, instants are `timestamptz`; SEC "accepted" times are parsed as `America/New_York`.
- **SEC politeness.** Requests carry a contact line from the environment variable `SEC_USER_AGENT`, else from `~/.config/ggtrader/sec_contact`, with at least 0.2 seconds between requests. The contact line is never written to the repo, the database, a log or a delegate's prompt.
- **One heavy job at a time**, through the capped-run wrapper and its lock (also for dumps, restores, deep checks and imports); tests only through `scripts/run_tests.sh` on small paths. Research code reads the database read-only; the importer is the only writer. Nothing heavy starts in the window around 02:00.

## 4. Approaches

**How much to load.** (A) SEC only; (B) A plus the version-two Kaggle files; (C) B plus the small reference series; plus (D) the price files with open, high, low and volume. *Picked by the owner: all of it, in stages, each labeled with its coverage.*

**Where the database lives.** (1) main drive, backed up to thesix; (2) whole database on thesix; (3) split. *Picked by the owner: (1).* thesix is three spinning disks in RAID5: roomy and failure-tolerant, but slower for the scattered reads a database does.

**How to back up.** Version 1 proposed streaming the database dump into the nightly borg run through borgmatic's built-in PostgreSQL feature. Review found that it breaks the importer's `.dump` rule, produces compressed dumps that borg cannot shrink across nights, risks the 2-hour limit, runs `docker exec` inside a restricted service nobody has tested, and invites a restore that wipes the live database. *Picked instead:* a scheduled `pg_dump` that writes a dated `.dump` file to a folder on thesix (which the importer's guard can see, and which the health check can read without root), and borg backs up that folder and the raw data folders. A weekly dump kept four weeks, plus a fresh dump before every large load. *Owner picked borg for backups; this keeps borg as the history keeper.*

**How to load.** Reuse the 2026-11-22 load library (`ggt db import`, load log, chunks, quarantine) rather than write a loader per study, extended with a table that lists every file in a load (5.6). *My call.*

**Raw SEC filings.** Keep the files on thesix (named by the hash of their web address, as now); the database holds the path, content hash and parsed fields. *My call.*

## 5. The design

### 5.0 Stage 0: backups first

**0a (no `sudo` needed; first, before any load):**
1. **A fresh full dump.** A user-level job (`pg_dump -Fc`, run from the host or inside the container) writes `ggtrader_YYYYMMDD.dump` and a `SHA256SUMS` line to a new folder on thesix (for example `ggTrader-data/db-dumps/`), under the heavy-job lock, outside 02:00. Record size, time and per-schema row counts at the moment of the dump. The crypto bars are part of this dump (crypto and equity rows share `public.ohlcv`, so they cannot be split).
2. **A restore drill, by hand.** Restore that dump into a disposable PostgreSQL 16 container with the matching TimescaleDB version, with its own empty volume **on thesix** (so it cannot fill the SSD), with no route or credentials to the live database. Create the extension, run `timescaledb_pre_restore()`, restore with one worker and stop on the first error, run `timescaledb_post_restore()`. Compare row counts and sample hashes against the numbers recorded at dump time, for every schema, including compressed price history, the published release and the paper-trading tables. Remove only the verified disposable container and volume. The existing `scripts/kaggle_restore_drill.py` is the starting point; it is hard-wired to the 2026-11-22 dump and five tables, needs `3 x dump size + 40 GB` free on the main drive, and the builder reads it for safety before reuse.
3. **A schedule.** A user timer takes the weekly dump (keeping four) and a fresh dump before each large load (the importer demands one no older than 7 days). The health check reads the newest dump's file date directly (no `root`, no look inside borg).

**0b (needs `sudo` and the owner's go-ahead):**
4. **Find the active file** (`/etc/borgmatic/config.yaml`), read it with `sudo`, and record what it really excludes. Reconcile it with the home copy.
5. **Add to borg:** the dump folder, and the folders that hold the only copy of a download: `market-data` (17 GB), `sec-delistings` (265 MB), `sec-early-window` (87 MB), `intraday_momentum_r9` (213 MB), `refactor-evidence` (102 MB), about 18 GB in all, against 4.3 TB free. Leave out the older dumps (`ggtraderdb-dump-20260218`, `ggtraderdb-backup-20261122`), `project-cleanup-*`, `worktree-leftovers-*`, `library-reading`, `library-staging` (not ggTrader data) and the borg repository itself. The settings currently exclude `/media/*`, so the builder finds the include mechanism that works with borgmatic's patterns.
6. **Prove it.** Run borgmatic's dry run first, then one watched real run. Confirm the 02:00 job still ends with "no problems" inside its 2-hour limit. Require the new archive to contain the dump and an inventory of each named folder; extract and hash-check a sample from each; confirm the repository and the excluded folders are absent. Keep the old config as `.bak` for rollback.

Large loads (B to D) also wait for 0b. The raw files and their backups sit on the same disk array: losing the array loses both; there is no off-site copy (a known gap).

### 5.1 Coverage labels

A small append-only table `ref.load_coverage (load_id, coverage, note)`, written by every importer in the same transaction that completes the load. Rules: exactly one row per completed load, a real `ref.loads` id, a value from a fixed list, no update, delete or truncate (the same protections as `ref.loads`), and a load without a coverage row cannot count as complete. Values: `includes_dead_stocks`, `survivors_only`, `partial_dead_stocks`, `large_stocks_only`, `unaudited_sample`, `not_a_stock_source`. Loads 1 to 3 get their rows too (written without changing their completed records): CRSP "includes_dead_stocks; no delisting returns", McGill "large_stocks_only, 2000 to 2023", derived bars (load 3) "includes_dead_stocks; derived from load 1". Quandl WIKI is `partial_dead_stocks` with the note "dead stocks from 2014 to 2018 only; none before 2014".

### 5.2 Stage A: the SEC data and the labels

**Scope: load what is already downloaded and labeled; make no new SEC requests.** Read both cache folders **read-only** and verify each body's recorded hash first. That is 1,415 files: 706 downloaded pages (699 unique web addresses; seven quarterly indexes sit in both folders) plus their notes, `README`, checksum list and a run record. Fetching more (other stocks, the merger-document search) is a later study with its own brief and the owner's approval.

New schema `sec`:
- `sec.fetched_files`: one row per unique downloaded file (web-address hash, content hash, URL, fetch time, status, size, path on thesix, kind). Kinds as they exist: quarterly index, company submission list, EDGAR directory listing, Form 25 notice (XML), complete filing text, one PDF. Two folders holding the same address become one row.
- `sec.index_entries`: the quarterly index rows (company number, name, form type, date filed, file name, quarter) for **all 105 unique cached quarters, 1999 Q1 to 2025 Q1** (80 quarters sit in `sec-delistings`, 32 in `sec-early-window`, seven in both): more than 20 million rows, so this is a medium load and gets a pilot and a size check.
- `sec.filing_headers`: parsed header facts for the filings whose bodies are cached (accession number, company number, form, filed date, 8-K item list).
- `sec.company_matches`: for each labeled stock, the company number (CIK, the SEC's company id) where the study recorded it (the 2005–2024 study's result file has none and the early study has it only inside its free-text evidence), how it matched, and whether it was ambiguous. Missing stays empty; the loader does not re-run the name matching.
- `sec.death_labels`: one row per study, stock, last trading day and method version, loaded from the two result files exactly as the studies stored them (label, confidence, evidence text verbatim, suggested loss including empty ones). Each row carries the study folder, the result file's hash and the source row's own hash. The stock is pinned to the release (release 1) and to the exact stock and last trading day the study used; a label that does not match a stock spell exactly is rejected. Where a row cites an accession number (the 2005–2024 file cites many, all from submission lists or quarterly indexes rather than downloaded filings), the check is that the number exists in `sec.index_entries` or a cached submission list, not that a body was downloaded. Where a row cites none, none is invented. A retry that loads the identical label is harmless; a conflicting label for the same key is refused.

Labels are append-only: a later method adds rows under a new method version, never overwrites.

One shared SEC downloader goes under `src/ggTrader/data/` for later use. It writes to a **separate** cache folder with immutable, content-versioned files; it never writes into the two existing caches (the frozen client names files by the hash of the address, so a re-fetch could silently replace a body). Stage A does not run it.

### 5.3 Stage B: the Kaggle files already downloaded

As scheduled in the 2026-11-22 design, **in this order** (SEC last, because it is the largest and riskiest), each with its own load, pilot and checks:
1. **Gu-Kelly-Xiu characteristics** (4.1 million rows, 94 measures per stock-month, 1957 to 2021) → `kaggle_chars.gkx`.
2. **S&P 500 daily membership and changes** → `kaggle_sp500` (cross-check of the membership list; the price file is skipped, as before).
3. **SPY holdings by year** (26 yearly files, 2000 to 2025) → `kaggle_sp500.spy_holdings`.
4. **Insider trades** (3.47 million) → `kaggle_insider.form4`; the builder states how it relates to `public.form4_transactions` (783,982 rows, untouched).
5. **SEC company facts** (82.2 million facts, 2009 to 2026) → `kaggle_sec.facts` and `kaggle_sec.filings`, keyed as the earlier design says (`adsh` is not unique).

Downloaded in the same folder but **not loaded** (judged minor or unusable in the 2026-11-22 exploration; they stay as files and in borg): `darkmatter` (a company reference list of 16,918 companies), `sec-edgar-filings-index`, `optionable-us-equity-universe-2019-2026`, `eventwren-sec-8k-events`, `survivorship-bias-free-s-and-p-500-fundamentals`, `sec-edgar-cik-ticker-exchange`, `persistfolio-sp500-membership-history-examples` (a sample of a paid product), and `fama-french-factor-zoo-benchmarks` (marked do not use).

### 5.4 Stage C: small reference series

New schema `refseries`.
- **Fama-French** monthly and daily factors and portfolio files, from the eight pinned zips in `data/external/french` (the extra unzipped monthly file in `crsp-kaggle/raw` is compared, not loaded).
- **Shiller** monthly series from 1871; the source is a binary Excel 97 workbook (`ie_data.xls`), read as the century-leverage study reads it.
- **Leveraged-SPY inputs** (`data/external/a23_inputs`): SPY (from 1993, unadjusted, with a dividends column), SSO and BIL (unadjusted, with dividends), and the 3-month Treasury rate. These are not the same as the database's adjusted `ohlcv` rows, so they are loaded as the study's own frozen inputs in their own columns; the Treasury rate (ends 2026-10-22) is compared with `fred_series` (ends 2026-09-28), not merged into it.
- **Intraday SPY bars** (Alpaca, 2016-01-04 to 2026-09-30): 129 monthly minute-bar files plus daily bars, calendar and corporate actions (132 files), manifest-pinned. The live intraday observer timers write elsewhere and are out of scope.
- **Study results index** (optional, last): one row per result file (study, path, SHA-256, commit, date). It lists results; it does not copy them.

### 5.5 Stage D: price files with open, high, low and volume

- **Quandl WIKI** (1.8 GB, 3,199 tickers to 2018, raw and adjusted, dividends and splits) → `kaggle_quandl`, coverage `partial_dead_stocks`.
- **Stooq US daily** (9,450 stocks and about 3,900 ETFs; prices already adjusted, no raw series) → `stooq`, coverage `survivors_only`. It is read straight from the zip, whose hash is recorded, rather than from 13,000 extracted files. The world and macro files wait.
- **The small sets** (`arandkei-delisted`, `finzdata-2008-2009`) → `kaggle_small`, coverage `unaudited_sample` (file names in `arandkei-delisted` contain anomalies, such as a double extension and reversed dates; they are recorded, not repaired).

These are separate source tables, not merged into the combined bars. Linking them to the CRSP stocks is a later study's job. Adjusted prices change when a later split or dividend happens, so a stored adjusted price is not "what the market showed that day"; they carry that note in their coverage row.

### 5.6 How a load works

Each load follows section 8 of the 2026-11-22 design: file hash against the download's checksum list, chunked loading with a complete-or-invisible rule, quarantine of bad rows with the raw value and reason, `parsed = accepted + quarantined`, a free-space reserve, and a load-log row with the loader's commit. Added here:
- **`ref.load_files (load_id, path, sha256, bytes)`**, because the load log holds one file per load and most of these sets are many files. A derived load (like load 3, whose source is another load) records its parent load instead of a file.
- **A freeze step for sets with no checksum list:** `sec-early-window`, `arandkei-delisted` and the extracted Stooq files have none. The importer hashes them once, records the list with the date, and the owner sees it; lineage is then "as first hashed on that date", not "as published". (SEC cache files already carry their own hash in their notes.)
- **A coverage row** (5.1) and a stored check result, written with the completion.
- **A dump no older than 7 days** and the free-space check, as the importer already demands.

### 5.7 The integrity check

`ggt db verify-data` reads the database and the files and **changes no data table**. It prints a report and, with `--record`, inserts one row into a new insert-only table `ref.integrity_checks`. The ordinary run checks: per-load row counts agree with the load log (from the log's own counts, not a full `count(*)` over the large tables); every file named in `ref.load_files` exists with its recorded size; files loaded in that run are re-hashed; no incomplete load; every completed load has exactly one coverage row; every `sec.death_labels` stock matches a stock spell in release 1; every cited accession exists as described in 5.2. Loads whose source is another load are checked against that load, not against a file. The report says exactly what each check did and did not verify.

A separate `--deep` run re-hashes every file the database names (the big downloads sit on spinning disks, so it takes a long time) and recomputes each load's content hash from the stored rows, so an altered value is caught even when counts and files agree. It takes the heavy-job lock, runs monthly at low priority outside the 02:00 window, as a timer the builder proposes.

### 5.8 Speed targets (mine, to be measured and reported)

- All filings of one company for one year, and one stock's label with its evidence: under 1 second.
- Company facts known on a given date for one company: under 1 second.
- One month of characteristics for every stock: under 5 seconds.
If a target misses, the builder adds the index that fixes it and reports the before and after times.

### 5.9 When the information became usable

Loaded is not exposed. The new tables sit in their own schemas, and a study may read one only if its brief names it. Each set keeps the 2026-11-22 design's section 7 rules: GKX measures usable from the first trading session of the month they predict; SEC facts by the filing's accepted time (units still undecided, so none exposed); insider trades and S&P files not exposed until their date meaning is shown; McGill unchanged. SEC labels describe how a stock ended; they are outcomes, never inputs to a signal for a date before the ending. Because research sessions log in as one superuser with a read-only flag, they could still select a raw table, so the plan adds, and the builder tests: named reader functions that take a decision date and return only what was known by then, and a check (in the study checker) that refuses a study whose brief does not name a table it reads. Separate database roles are an open item for the owner (section 9).

### 5.10 Documents to update

`docs/cli_reference.md` (new import commands, `verify-data`), the database map in `CLAUDE.md` (also reconcile its stale counts: `form4_transactions` has 783,982 rows, and `~/ggtraderdb` now holds only a compose file), `docs/architecture.md`, the top-level `README.md`, `docs/changelog.md`, the thesix `README.md` (what is backed up and how), and the import runbook (stages, owner gates). Terms are explained in plain words on first use.

## 6. Order, gates and reporting

Order: 0a, 0b, then A, B (1 to 5 in order), C, D. The owner approves each stage's start. Each stage ends with a short plain-words report: rows loaded, rows quarantined, database size and free space, check results, timings, and what each tool's use cost. The plan sizes itself to the owner's quota (ask before planning).

Gates: (G-backup) the owner approves the backup edit and supplies `sudo` through the secret card (0b); (G-restore) the owner sees the restore drill result (0a) before the first load; (G-load) the owner approves each stage's real load after its pilot. **No stage is exempt from the fresh dump or the restore drill**: Stage A is a medium load (20 million index rows) and adds tables that touch the load log's neighbours.

## 7. Left to the builder

Schema and column names beyond those named here; column types (measure the real key of the SEC facts); chunk sizes and compression; the CLI shape; the SEC name-match storage details; how to parse the Fama-French files (multi-section text, missing-value markers); the borgmatic include mechanism; where `db-dumps` lives and its retention; whether `verify-data --deep` runs on a timer; how the SEC downloader's new cache is laid out.

## 8. What changed from version 1, and calls I made

Changes after review: the backup plan (dump files plus borg history instead of a dump streamed inside borg); the real backup config file; the fresh dump and restore drill first, with no exemption for Stage A; SEC scope (all 105 cached quarters, labels loaded as stored, no invented accessions); the Stage B order (SEC last); a multi-file load table; a freeze step for sets with no checksum list; the corrected Stage C facts; the folders left out, named; Quandl's coverage value; exposure rules and a reader-function test; the label and coverage rules; the SEC caches read-only.

Calls I made, for the owner to check:
- Reuse the existing import library and load log.
- Raw SEC files stay on thesix; the database holds hashes, headers and labels.
- A weekly dump plus a fresh one before each large load; borg keeps the history.
- Coverage labels in a new table, not by editing the append-only load log.
- Load only what is not already in the database; compare overlaps.
- Stage order, and the order inside stage B.
- The study results index is optional and last.
- The studies' frozen SEC copies and results are untouched and nothing is re-scored.
- The small Kaggle sets named in 5.3 are not loaded.
- No off-site copy (a known gap, not planned here).

## 9. Open items that are the owner's

- Approve this design.
- At 0b: the backup edit and `sudo`.
- **Roles.** Do you want a separate research database login that can read new data only through the approved reader functions? It is the only way to enforce "loaded is not exposed" instead of checking it by convention, but it changes how research sessions connect, which the 2026-11-22 design put under the owner's say.
- Whether to keep the study results index (5.4) at all.
- Whether any of the small Kaggle sets left out in 5.3 should be loaded after all.
- The old 14 GB February dump (`ggtraderdb-dump-20260218`) is left alone; say if it can ever be retired.
