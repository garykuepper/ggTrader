# Kaggle import runbook

**Live status (2026-11-23): loaded.** CRSP daily is load 1, McGill-FIAM load 2, and derived CRSP bars load 3; release 1 is published. The live steps were: take the pre-load baseline with main’s code, apply the schemas, load and compress CRSP, load McGill, derive and compress bars, build the release, then run the closing checks against that baseline. All eight data checks passed. If a release is wrong, use the withdrawal command in step 8 and investigate before publishing a replacement. The live database volume is treated as unbacked-up until the owner confirms otherwise. The gates below apply before further writes.

Before the first load, save the closing checks' baseline on the target with the original code.
After the load, supply it with `--compare-baseline PATH` and describe its origin with
`--baseline-origin TEXT`. Saving a baseline after the change cannot establish the before/after
comparison. Saving is a baseline-only operation: it writes the file before any release exists,
does not run the release checks, and exits nonzero until a later comparison is made.

1. Check the backup: `backup_fresh` requires a dump at least 1 MiB and no older than seven days in the supplied backup directory. A freshness check does not prove a restore works. G1 requires the owner to accept the 2026-11-22 backup or a successful restore drill before any write to `ggtrader`.
2. Apply the schemas after G1: `ggt db import apply-ddl --use-configured-db --backup-dir BACKUP_DIR --yes`. The command refuses to run without a recent dump in the backup directory (`--no-backup-check` exists for scratch databases only and is refused for `ggtrader`). It re-applies the whole Kaggle schema file as well as the newer tables; both are safe to repeat on a database that already has compressed data (tested), and the command gives up after 15 seconds instead of queueing behind a reader that holds a lock. For a scratch target, use `--db-url-env NAME` instead.
3. Run the pilot on a disposable database server using an environment variable containing its connection: `scripts/run_capped.sh .venv/bin/python scripts/kaggle_pilot.py --scratch-url-env GGT_SCRATCH_URL --file CRSP.parquet --sums SHA256SUMS.txt --out results/kaggle_pilot/pilot.json`. The optional `--year` selects a year. The pilot creates and drops a scratch database and measures a load, compression and reload. At G2 the owner approves projected size, peak disk use and free-space margin before a full load.
4. After G1 and G2, load CRSP: `ggt db import crsp --use-configured-db --file CRSP.parquet --sums SHA256SUMS.txt --backup-dir BACKUP_DIR --db-free-path DB_VOLUME --yes`. The full load automatically runs `compress_kaggle_chunks` after completion; it also vacuums the affected chunks to reclaim space. Row-group subsets and `--year` pilots do not automatically compress. Run compression only after all row groups have landed, since each group can touch every year. Allow time and disk space for the vacuum.
5. Load McGill: `ggt db import mcgill --use-configured-db --file MCGILL.csv --sums SHA256SUMS.txt --backup-dir BACKUP_DIR --db-free-path DB_VOLUME --yes`. The backup and free-space guards run again. `--no-backup-check` is available for scratch work; it does not satisfy G1.
6. Derive and compress bars: `ggt db import derive-bars --crsp-load N --calendar-file CRSP.parquet --backup-dir BACKUP_DIR --db-free-path DB_VOLUME --use-configured-db --yes`. Record the derived load ID.
7. Generate the approved links with `scripts/kaggle_find_links.py --crsp-file CRSP.parquet --out LINKS.csv` and the Yahoo connection selected by `GGT_YAHOO_URL`. Keep the resulting `LINKS.csv.manifest.json` next to the CSV. After G3 and G5 approval, publish: `ggt db import release build --crsp-bars-load N --mcgill-load M --links LINKS.csv --use-configured-db --yes`. The manifest must match the CSV, approved settings, and the raw CRSP file behind the derived load. Review the counts, demoted links and fingerprint. Use `ggt db import release list --use-configured-db` to check the final status and counts.
8. If a release is wrong, withdraw it with `ggt db import release withdraw --release-id N --use-configured-db --yes`; investigate and publish a corrected release.

A load marked `failed` is terminal: correct the cause and start a new load. A running load can resume with the same file and parser; do not read partial data. Use `kaggle_crsp.daily_of(load_id)` and `kaggle_chars.mcgill_of(load_id)` to read completed loads.

Derived-load `file_sha256` identifies its derivation inputs rather than a physical file. Its
`checks.derivation_identity` records the parent load, raw file and parser, a SHA-256 of the ordered
calendar timestamps, the derivation version, and chunk layout. A running derivation resumes only
with the same identity; to use a changed calendar or chunk layout, first mark the old running
load failed, then start a new derivation.

To exercise the backup restore, set an administrative connection to the server hosting the `ggtrader_db` container in an environment variable and run `scripts/run_capped.sh .venv/bin/python scripts/kaggle_restore_drill.py --scratch-url-env GGT_SCRATCH_URL --out results/kaggle_restore_drill.json`. Use `--dump`, `--expected`, and `--sums` to select the backup and its evidence when needed. The drill creates a scratch database on that server and runs `pg_restore` inside the `ggtrader_db` container, so the connection and container must refer to the same server. It can consume substantial disk space; the live database is not the restore target.

Further owner gates: G3 approves the written link table after calibration; G4 approves wiring the dead-stock rule into the engine; G5 approves the first release before publication. See the [implementation plan](superpowers/plans/2026-11-22-kaggle-import-version-one.md) for the evidence at each gate.

A published release reads Yahoo bars only through the date recorded in its Yahoo fingerprint.
Release publication first builds the release and its records in one transaction, then seals it as
published. A building release is unavailable to readers; published and withdrawn releases cannot
gain new child records.
Rows added after that date are excluded; Yahoo rows within the horizon remain mutable and can
trigger the release's drift check.

## Known limits of version one

- Check 4 can accept CRSP rows before the first Yahoo date in a column that should contain only
  Yahoo data after a link is rejected.
- Check 5 compares most named stocks with the release's own records. It can pass when optional
  names are absent, so it is not an independent confirmation of every historical identity.
- Checks 9 through 13 are labelled `covered_by_tests`; the closing-check script does not run
  those suites. Run the listed tests separately.
- The guard against running beyond the end of the CRSP file uses each column's last valid bar
  date. A stock that ended on that date can be refused, while a file-end stock with missing last
  prices can escape that guard.
- The link manifest's Yahoo fingerprint is recorded but not checked again when building a release.
  Yahoo symbols without an end-of-life record have no forced exit.
- A combined-data read has a fixed cost of roughly three to five seconds per call on the
  rehearsal database.
- A manually committed `building` release cannot be read, but can still be published by a later
  direct database update. Do not publish or add records to such a release by hand.

## Dumps and restore drill (stage 0a)

A dump is a database backup file. The live volume is treated as unbacked-up until the owner
confirms otherwise; a recent dump alone does not prove it can be restored. The owner must approve
live database writes and accept the backup or a successful restore drill before those writes.
The SEC loaders require a dump at least 1 MiB and at most seven days old in `--backup-dir`,
including on scratch databases. Confirm sufficient free space on the database drive and pass its
path as `--db-free-path` before loading.

Create a dump with `scripts/run_capped.sh .venv/bin/python scripts/db_dump.py --out-dir /media/thesix/ggTrader-data/db-dumps/`.
The output directory must already exist. The script writes `ggtrader_YYYYMMDD.dump.part` while
working and renames it to `ggtrader_YYYYMMDD.dump` on success. If the before and after snapshots
(the recorded counts and content digests) differ, it keeps `ggtrader_YYYYMMDD.dump.mismatch`.
The exit codes are 0 for success, 1 for a failure, 2 for an invalid output directory or
insufficient space, and 3 for a snapshot mismatch. Digests are computed with the database
session time zone pinned to UTC. The directory also holds `SHA256SUMS` and dated counts files,
and an `attempts/` subfolder with the superseded and failed runs from the first drill. The
installed user timer at
`~/.config/systemd/user/ggtrader-db-dump.timer` runs Sunday at 04:30 and keeps four dumps.

Run a restore drill (a test restore into a disposable container) with:

```text
scripts/run_capped.sh .venv/bin/python scripts/db_restore_drill.py --dump DUMP --counts COUNTS_JSON --out REPORT_JSON
```

`--sums PATH` selects a checksum file; otherwise the drill uses `SHA256SUMS` beside the dump.
`--image IMAGE` selects the container image, and `--keep` retains the scratch container and files.
The report's `ok` says whether all recorded counts and digests matched; `stage` identifies where a
failure occurred and `error` names its error type. `server_message`, when present, is the database
server's primary error message. `catalog_version_aligned` says whether the drill updated the
stored TimescaleDB version note. The live database records TimescaleDB 2.25.0 in its catalog
(the database's internal records), although 2.25.1 is installed. A real restore must align that
note to the installed version before `SELECT timescaledb_post_restore()`:

```sql
UPDATE _timescaledb_catalog.metadata SET value = '<installed version>' WHERE key = 'timescaledb_version';
```

The drill performs this alignment automatically and reports whether it changed a row.

## SEC imports and records

The four SEC loaders record saved responses, quarterly filing index entries, filing headers, and
frozen study labels respectively. `ref.load_files` records source paths, hashes (content
fingerprints) and sizes; `ref.load_coverage` records what each load covers; and
`ref.integrity_checks` stores integrity-check results. The SEC tables hold fetched-file records,
index entries, headers and death labels. A coverage label describes the scope of stock data:
`includes_dead_stocks` includes stocks that later ended; `survivors_only` includes only survivors;
`partial_dead_stocks` includes some ended stocks; `large_stocks_only` covers large stocks;
`unaudited_sample` is a sample whose coverage has not been checked; and `not_a_stock_source`
means the data is not a stock source. SEC loads use `not_a_stock_source`. See the
[CLI reference](cli_reference.md) for the commands and their dry runs. These new SEC loads have
not yet been run against the live database.

### Recovering from an interrupted SEC load

A load that stopped halfway stays `running`. Run the same command again with the same inputs and it resumes where it stopped. If the cause has to be fixed by changing the inputs (a different `--only-quarter`, a repaired cache), the new run is refused because the old one is still open; close the old one first:

```text
ggt db import fail-load --load-id N --reason "why" (--db-url-env NAME | --use-configured-db) --yes
```

Only a `running` load can be closed this way; a finished load is never changed. `ggt db verify-data` lists any load that is still `running`.

Each retained load counts toward a limit of two per kind of data. A rerun of an identical cache, index or header load stops with "already loaded"; an identical label file returns the existing load number.

The new SEC tables can be read only through `sec.fetched_files_of(load_id)`, `sec.index_entries_of(load_id)`, `sec.filing_headers_of(load_id)` and `sec.death_labels_of(load_id)`, which refuse a load that is not complete, so a half-finished or failed load never shows up in a study. A direct `SELECT` on the tables is not blocked (the research session logs in as the database owner); the study checker is expected to insist on these readers.

A command that finds nothing to load (a wrong `--folder`, a label file with no readable rows) stops with an explanation instead of recording an empty load. Rows that cannot be read are set aside with their reason, are counted in the preview, and are reported by `ggt db verify-data`.
