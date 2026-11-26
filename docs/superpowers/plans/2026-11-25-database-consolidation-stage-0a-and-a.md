# Database consolidation, stage 0a and stage A: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take a fresh full dump of the live database and prove a restore (stage 0a); then load the cached SEC data and the two studies' labels into new `sec` tables with load records, coverage labels and an integrity check (stage A).

**Architecture:** Reuse the 2026-11-22 import library (`ref.loads`, chunks, quarantine, guards) and add three bookkeeping tables (`ref.load_files`, `ref.load_coverage`, `ref.integrity_checks`) plus a `sec` schema. Dump and restore are plain scripts that run through the capped-run lock. The restore goes into a disposable container on thesix, never the live server.

**Tech Stack:** Python 3.12, psycopg2, pyarrow/zipfile, PostgreSQL 16 with TimescaleDB 2.25.1 in Docker, pytest (`integration` + `db` markers for scratch-database tests).

**Spec:** [2026-11-25-database-consolidation-design.md](../specs/2026-11-25-database-consolidation-design.md) (sections 5.0, 5.1, 5.2, 5.6, 5.7). Read it and [2026-11-22-kaggle-import-design.md](../specs/2026-11-22-kaggle-import-design.md) section 8.

## Global Constraints

- Delegates (builders and reviewers) **never** run a real-database command, an importer against the live database, a SEC request, `docker` against `ggtrader_db`, or anything under `sudo`. Tests use fakes, temp folders and, for `integration` tests, a scratch database that **the controller** runs with `GGT_TEST_DB_URL` set. The controller alone runs the real dump, drill and loads.
- Tests only through `scripts/run_tests.sh <small targeted paths>`; never a whole tier or `tests/paper/`. If it prints `waiting for another heavy ggTrader job`, wait. One heavy job at a time.
- No new `timestamp without time zone` column; dates are `date`, instants `timestamptz`. Ruff-clean (`line-length = 100`), type hints, module docstrings, `argparse` and `--help` on every script.
- The two SEC cache folders (`/media/thesix/ggTrader-data/sec-delistings`, `.../sec-early-window`) are opened **read-only**; no code in this plan writes into them. The studies' frozen files, `research/**/PIN.json` and result files are not edited.
- The SEC contact line is never written to the repo, a log, the database or a prompt. This plan makes no SEC request.
- No row of `ref.loads`, `ref.load_chunks` or any existing table is changed. New bookkeeping tables are append-only (reuse `ref.reject_change`, `ENABLE ALWAYS TRIGGER`, statement-level no-truncate), as in `sql/kaggle_schema.sql`.
- Every importer: file hash checked, chunked complete-or-invisible, quarantine with raw value and reason, `parsed = accepted + quarantined`, free-space check against the **database** drive (`--db-free-path`), a dump no older than 7 days (`guards.backup_fresh`), a coverage row written in the same transaction as completion.
- Plain words in every owner-facing report; no rule numbers.

## Lanes and order

| Task | Builder | Reviewer |
|---|---|---|
| 1 dump script + snapshot helpers + timer templates | Codex Terra | Gemini Neon (short) |
| 2 disposable-container restore drill | Codex Terra | Gemini Neon (short) |
| C1 controller run: dump, drill, report | Sonnet | owner gate |
| 3 schema, load-log additions, coverage | Codex Sol | Gemini Neon |
| 4 cache scanner and loader | Codex Terra | Gemini Neon |
| 5 quarterly index loader | Codex Terra | Gemini Neon |
| 6 filing header parser and loader | Codex Terra | Gemini Neon |
| 7 label loaders | Codex Sol | Gemini Neon |
| 8 CLI wiring and docs | Codex Terra | Gemini Neon (short) |
| 9 `verify-data` | Codex Sol | Gemini Neon |
| C2 controller run: pilot, loads, checks, report | Sonnet | owner gates |

Order: 1, 2, C1, then 3; then 4; then 5, then 6 (one implementer at a time; both need Task 4's `CacheEntry`); then 7, 8, 9, C2. Only one Antigravity task at a time. A reviewer is never the builder's vendor; Sonnet (the controller) re-reads every diff and reruns every check before moving on. Fix rounds: at most three per task, then stop and ask the owner. Usage report to the owner after C1 and after C2.

## Review Focus

Most likely first. Each line has a test in the owning task.

1. A dump that is truncated, or taken while something writes, looks fine (Task 1: counts before and after must match; size floor; `.part` rename).
2. A restore that "works" but is missing hypertable data or a schema (Task 2: every table's count and the digests, not five tables).
3. Two folders holding the same SEC address with different bytes (Task 4: conflict is an error, not a silent pick).
4. A `master.idx` line with a bad date or a company name containing `|` (Task 5: quarantine, never drop).
5. A truncated filing header, or a filing with several companies, loading as nothing or one row (Task 6).
6. A label whose stock is not in release 1 or whose last day differs (Task 7: whole load refused with the list); a rerun with a changed label (conflict error).

---

### Task 1: Dump script, snapshot helpers and timer templates

**Files:**
- Create: `src/ggTrader/utils/db_snapshot.py`, `scripts/db_dump.py`, `deploy/db_dump/ggtrader-db-dump.service`, `deploy/db_dump/ggtrader-db-dump.timer`, `deploy/db_dump/README.md`
- Test: `tests/utils/test_db_snapshot.py`, `tests/scripts/test_db_dump.py`

**Interfaces:**
- Produces (`ggTrader.utils.db_snapshot`): `DIGEST_TABLES: tuple[str, ...]` = `("ref.loads", "ref.releases", "universe.loads", "universe.membership", "public.paper_trades", "public.lab_runs")`; `table_counts(conn) -> dict[str, int]` (exact `count(*)` of every base table outside `pg_catalog`, `information_schema` and `_timescaledb_*`, key `"schema.table"`); `table_digests(conn) -> dict[str, str]` (md5 of `string_agg(t::text, '|' ORDER BY t::text)` per `DIGEST_TABLES`, empty table gives md5 of empty string); `snapshot(conn) -> dict` = `{"counts": ..., "digests": ...}`.
- Produces (`scripts/db_dump.py`): `main(argv: list[str] | None = None) -> int`; `dump_filename(day: date) -> str` = `ggtrader_YYYYMMDD.dump`; `build_dump_command(container: str = "ggtrader_db") -> list[str]` = `["docker","exec","-u","postgres",container,"pg_dump","-Fc","-Z","1","--no-owner","ggtrader"]`; `prune(folder: Path, keep: int) -> list[Path]` (removes only files matching `^ggtrader_\d{8}\.dump$` and their `counts_YYYYMMDD.json`, oldest first, never the newest).
- Arguments: `--out-dir PATH` (required, must be under `/media/thesix/`), `--keep N` (default 4), `--db-url-env NAME`. Exit codes: 0 ok; 1 failure (including dump under 1 MB); 2 refusal (out dir missing or outside thesix, free space on thesix below 2 × the newest earlier dump or 30 GB); 3 counts before and after differ.

- [ ] **Step 1: Write failing tests for `db_snapshot`** (fake connection returning canned rows): `test_table_counts_skips_system_and_timescale_schemas`, `test_table_digests_covers_every_digest_table`, `test_snapshot_has_counts_and_digests_keys`.
- [ ] **Step 2: Run them; expect FAIL (module missing).** `scripts/run_tests.sh tests/utils/test_db_snapshot.py`
- [ ] **Step 3: Implement `ggTrader.utils.db_snapshot`** as specified.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Write failing tests for `scripts/db_dump.py`** with an injected runner and connection: `test_refuses_out_dir_outside_thesix` (exit 2), `test_refuses_when_free_space_low` (exit 2), `test_writes_part_then_renames_and_records_sha256`, `test_counts_before_after_mismatch_exits_3_and_keeps_part_file_out_of_final_name`, `test_dump_under_one_megabyte_exits_1`, `test_counts_json_records_timescaledb_version`, `test_prune_keeps_newest_and_ignores_other_files`, `test_help_exits_zero`.
- [ ] **Step 6: Run; expect FAIL.** `scripts/run_tests.sh tests/scripts/test_db_dump.py`
- [ ] **Step 7: Implement `scripts/db_dump.py`.** Flow: validate folder and space; `snapshot()` before; stream `build_dump_command` stdout into `<name>.part`; `snapshot()` after; compare counts and digests (mismatch exit 3, leave the `.part` renamed to `.mismatch` for inspection); rename to the final name; append `<sha256>  <name>` to `SHA256SUMS`; write `counts_YYYYMMDD.json` = `{"day","dump_bytes","seconds","timescaledb","before","after"}` where `timescaledb` is the live `extversion` read from `pg_extension`; `prune`. It does not take the heavy-job lock itself; it is meant to be started through `scripts/run_capped.sh`.
- [ ] **Step 8: Run; expect PASS.** Also `ruff check` on the new files.
- [ ] **Step 9: Write the timer templates** (pattern of `deploy/membership_weekly/`: header comment `TEMPLATE, NOT INSTALLED, NOT ENABLED`; service runs `scripts/run_capped.sh .venv/bin/python scripts/db_dump.py --out-dir /media/thesix/ggTrader-data/db-dumps`; timer weekly Sunday 04:30 local time, not near 02:00); README says to install only after the owner says go.
- [ ] **Step 10: Commit.** `git add` the new files; message `feat: database dump script, snapshot helpers and timer templates`.

### Task 2: Disposable-container restore drill

**Files:**
- Create: `scripts/db_restore_drill.py`
- Test: `tests/scripts/test_db_restore_drill.py`

**Interfaces:**
- Consumes: `ggTrader.utils.db_snapshot.snapshot`, `DIGEST_TABLES` (Task 1); `ggTrader.data.kaggle_import.guards.verify_sha256`.
- Produces: `main(argv) -> int`; `compare(expected: dict, actual: dict) -> dict` returning `{"ok": bool, "tables": {name: {"expected","actual","ok"}}, "digests": {...}}` where a table missing on either side is a failure; `container_name(token: str) -> str` = `ggt_restore_drill_<token>`; `refuse_unsafe(name: str, scratch_dir: Path, live_container: str = "ggtrader_db") -> None` raising `ValueError` unless the name starts with `ggt_restore_drill_`, differs from the live container, and the scratch dir lies under `/media/thesix/ggTrader-data/db-restore-scratch/`.
- Arguments: `--dump PATH`, `--image NAME`, `--sums PATH` (default `SHA256SUMS` beside the dump), `--counts PATH` (the `counts_YYYYMMDD.json` of Task 1, compared against its `after` block), `--out PATH` (report JSON, required), `--keep` (leave container and folder for inspection).

- [ ] **Step 1: Write failing tests** with injected command runner: `test_refuses_name_not_ours`, `test_refuses_scratch_dir_outside_scratch_root`, `test_compare_flags_missing_table_and_changed_count`, `test_compare_checks_digests`, `test_extension_version_mismatch_aborts_before_restore`, `test_cleanup_removes_only_its_own_container_and_folder_even_after_failure`, `test_report_has_no_password_or_url`, `test_help_exits_zero`.
- [ ] **Step 2: Run; expect FAIL.** `scripts/run_tests.sh tests/scripts/test_db_restore_drill.py`
- [ ] **Step 3: Implement.** Flow: verify dump hash; read the TimescaleDB version from the counts file's `timescaledb` value (Task 1 writes it) and the image name from the `--image` argument (default `timescale/timescaledb:latest-pg16`); start `docker run -d --name ggt_restore_drill_<hex> -p 127.0.0.1::5432 -v <scratch>:/var/lib/postgresql/data -e POSTGRES_PASSWORD=<random>` from the same image; wait ready; create database `ggtrader`; `CREATE EXTENSION timescaledb`; abort if its version differs from the `timescaledb` value in the counts file; `SELECT timescaledb_pre_restore()`; `docker exec -i ... pg_restore --exit-on-error --jobs 1 --no-owner -d ggtrader` fed from the dump file; `SELECT timescaledb_post_restore()`; `ANALYZE`; `snapshot()` on the scratch connection; `compare`; write the report; remove its container and scratch folder unless `--keep`. It never issues `docker rm`, `docker stop` or `docker exec` against any other name.
- [ ] **Step 4: Run; expect PASS.** Ruff.
- [ ] **Step 5: Commit** `feat: disposable-container restore drill`.

### C1 (controller): fresh dump, restore drill, owner gate

- [ ] Confirm nothing else is running (no heavy job, trader stopped, not near 02:00). Create `/media/thesix/ggTrader-data/db-dumps/`.
- [ ] Run `scripts/run_capped.sh .venv/bin/python scripts/db_dump.py --out-dir /media/thesix/ggTrader-data/db-dumps` (first with a dry check of free space; allow `GGT_RUN_MAX_SECONDS=14400`). Expect exit 0, a dump of several GB, `counts` before equal after. Exit 3 means something wrote during the dump: find out what, rerun.
- [ ] Run `scripts/run_capped.sh .venv/bin/python scripts/db_restore_drill.py --dump ... --counts ... --out .../drill_report.json`. Expect `ok: true`, container and scratch folder gone.
- [ ] Report to the owner in plain words: dump size and time, drill result, thesix free space, usage so far. Owner gate **G-restore**: approve before any load. The weekly timer is installed only if the owner says so (copy the templates to `~/.config/systemd/user/`; user level, no `sudo`). Stage 0b (borg, `sudo`) is not part of this plan.

### Task 3: Schema additions and load-log extensions

**Files:**
- Create: `sql/consolidation_schema.sql`, `src/ggTrader/data/sec_import/__init__.py`, `src/ggTrader/data/sec_import/ddl.py`
- Modify: `src/ggTrader/data/kaggle_import/loadlog.py` (add `record_files`; add optional `coverage` argument to `complete_load`), `src/ggTrader/cli/cmd_db.py` (`apply-ddl` also applies this file)
- Test: `tests/data/sec_import/test_ddl.py` (integration, db), `tests/data/kaggle_import/test_loadlog.py` (extend)

**Interfaces:**
- Produces: `ggTrader.data.sec_import.ddl.apply_ddl(conn) -> None` (idempotent, one transaction); `loadlog.record_files(conn, load_id: int, files: Sequence[tuple[str, str, int]]) -> None` (path, sha256, bytes; insert-only, idempotent on retry); `loadlog.complete_load(conn, load_id, checks, *, expected_chunks=None, coverage: tuple[str, str] | None = None)` where `coverage = (value, note)` is inserted into `ref.load_coverage` inside the same transaction, before the status update; `COVERAGE_VALUES = ("includes_dead_stocks","survivors_only","partial_dead_stocks","large_stocks_only","unaudited_sample","not_a_stock_source")` in `sec_import/ddl.py`.
- Tables (all `IF NOT EXISTS`): `ref.load_files(load_id bigint, path text, sha256 text, bytes bigint, PRIMARY KEY(load_id, path))`; `ref.load_coverage(load_id bigint PRIMARY KEY REFERENCES ref.loads(load_id), coverage text NOT NULL CHECK (coverage IN (...)), note text NOT NULL, recorded_at timestamptz NOT NULL DEFAULT now())`; `ref.integrity_checks(check_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, ran_at timestamptz NOT NULL DEFAULT now(), mode text NOT NULL CHECK (mode IN ('ordinary','deep')), ok boolean NOT NULL, report jsonb NOT NULL)`; schema `sec` with `fetched_files(load_id, url_hash text, url text, content_sha256 text, fetched_at timestamptz, status int, bytes bigint, path text, kind text CHECK (kind IN ('quarterly_index','submission_list','directory_listing','form25_xml','filing_text','other')), PRIMARY KEY(load_id, url_hash))`, `index_entries(load_id, quarter text, cik bigint, company_name text, form_type text, date_filed date, filename text)` with indexes on `(load_id, cik, date_filed)` and `(load_id, form_type, date_filed)`, `filing_headers(load_id, accession text, cik bigint, form_type text, filed_date date, items text[], PRIMARY KEY(load_id, accession, cik))`, `death_labels(load_id, study text, method_version text, permno bigint, last_date date, release_id bigint, label text, confidence text, evidence_text text, suggested_loss double precision, accessions text[], source_file text, source_file_sha256 text, source_row_sha256 text, source_fields jsonb, PRIMARY KEY(load_id, study, method_version, permno, last_date))`.
- Protections: `ref.reject_change` row triggers (`ENABLE ALWAYS`) and statement no-truncate triggers on `ref.load_files`, `ref.load_coverage`, `ref.integrity_checks`, `sec.fetched_files`, `sec.filing_headers`, `sec.death_labels`; statement no-truncate only on `sec.index_entries` (too large for row triggers). Trigger `ref.require_coverage` BEFORE UPDATE on `ref.loads`: when `OLD.status = 'running'` and `NEW.status = 'complete'` and `NEW.dataset` not in `('crsp-daily','mcgill-fiam','crsp-derived-bars')` and no `ref.load_coverage` row exists, raise `coverage label required`. Backfill: insert coverage rows for loads of those three datasets (`includes_dead_stocks` with note "CRSP; no delisting returns", `large_stocks_only` "McGill-FIAM, 2000 to 2023, large stocks", `includes_dead_stocks` "derived from the CRSP load"), idempotent (`WHERE NOT EXISTS`), without touching `ref.loads`.

- [ ] **Step 1: Write failing integration tests** (`pytestmark = [pytest.mark.integration, pytest.mark.db]`, scratch database as in `tests/data/kaggle_import/test_loadlog.py`): `test_apply_ddl_is_idempotent`, `test_bookkeeping_tables_reject_update_delete_truncate`, `test_complete_without_coverage_fails_for_new_dataset`, `test_complete_with_coverage_writes_row_in_same_transaction`, `test_legacy_datasets_complete_without_coverage`, `test_coverage_rejects_unknown_value_and_second_row`, `test_record_files_is_idempotent`, `test_backfill_adds_rows_for_existing_loads_and_leaves_loads_unchanged`.
- [ ] **Step 2: Run on a scratch database (controller sets `GGT_TEST_DB_URL`); expect FAIL.** Builders without that variable see the tests skip; the unit-style loadlog test additions (`test_complete_load_passes_coverage_through`) use a fake connection and run for everyone.
- [ ] **Step 3: Implement the SQL file, `apply_ddl`, `record_files`, the `coverage` argument, and the `apply-ddl` hook in `cmd_db.py`** (it applies `kaggle_schema.sql` first, then `consolidation_schema.sql`).
- [ ] **Step 4: Run; expect PASS.** Existing `tests/data/kaggle_import/test_loadlog.py` and `test_import_cli.py` still pass.
- [ ] **Step 5: Commit** `feat: load files, coverage labels, integrity checks and sec tables`.

### Task 4: SEC cache scanner and loader

**Files:**
- Create: `src/ggTrader/data/sec_import/cache.py`
- Test: `tests/data/sec_import/test_cache.py`

**Interfaces:**
- Consumes: `begin_load`, `commit_chunk`, `complete_load`, `record_files` (Task 3), `guards.backup_fresh`, `guards.free_space_ok`.
- Produces: `@dataclass(frozen=True) CacheEntry(url_hash: str, url: str, content_sha256: str, fetched_at: datetime, status: int, bytes: int, body_path: Path, kind: str)`; `classify_url(url: str) -> str` (rules: path ending `/master.zip` under `full-index` is `quarterly_index`; `data.sec.gov/submissions/CIK` and `.json` is `submission_list`; a URL ending `/` under `Archives/edgar/data/` is `directory_listing`; `.xml` is `form25_xml`; `.txt` under `Archives/edgar/data/` is `filing_text`; else `other`); `scan_cache(folder: Path) -> list[CacheEntry]` (pairs `<h>.body` with `<h>.meta.json`, read-only, raises `CacheCorrupt(path)` if a body's sha256 differs from the note's `sha256`, or a pair is incomplete); `merge_caches(*groups: list[CacheEntry]) -> list[CacheEntry]` (dedupe on `url_hash`; same hash with different `content_sha256` raises `CacheConflict`); `manifest_sha256(entries) -> str` (sha256 of sorted `url_hash:content_sha256` lines); `load_cache(conn, folders: Sequence[Path], *, backup_dir: Path, db_free_path: Path, margin_bytes: int | None = None) -> int` returning the load id (dataset `sec-cache`, parser `sec-cache-1`, 200 entries per chunk keyed `fetched-NNNN`, coverage `("not_a_stock_source", "SEC download cache, not a stock source")`).

- [ ] **Step 1: Write failing unit tests** on a `tmp_path` cache with fake body/meta pairs: `test_classify_url_covers_every_kind`, `test_scan_cache_reads_pairs_and_checks_hash`, `test_scan_cache_raises_on_hash_mismatch`, `test_scan_cache_raises_on_orphan_body`, `test_merge_dedupes_identical_url_hash`, `test_merge_conflict_on_different_content`, `test_manifest_is_order_independent`, `test_scan_never_opens_a_file_for_writing` (patch `open`/`Path.open` and assert only `rb`/`r` modes).
- [ ] **Step 2: Run; expect FAIL.** `scripts/run_tests.sh tests/data/sec_import/test_cache.py`
- [ ] **Step 3: Implement.** Integration test (scratch database): `test_load_cache_registers_files_and_coverage_and_is_resumable` (marked integration, db).
- [ ] **Step 4: Run unit tests; expect PASS.** Controller later runs the integration test.
- [ ] **Step 5: Commit** `feat: SEC cache scanner and loader`.

### Task 5: Quarterly index loader

**Files:**
- Create: `src/ggTrader/data/sec_import/index.py`
- Test: `tests/data/sec_import/test_index.py`

**Interfaces:**
- Consumes: `CacheEntry`, `load` helpers (Tasks 3, 4).
- Produces: `parse_master_idx(text: str, quarter: str) -> tuple[list[tuple], list[tuple[str, dict, str]]]` (rows `(quarter, cik, company_name, form_type, date_filed, filename)`; quarantine tuples `(row_ref, raw, reason)`); `read_master_zip(path: Path) -> str` (member `master.idx`, latin-1); `quarter_of(url: str) -> str` (`.../full-index/2005/QTR2/master.zip` gives `2005Q2`); `load_index(conn, entries: Sequence[CacheEntry], *, backup_dir: Path, db_free_path: Path, only_quarters: Sequence[str] | None = None, margin_bytes: int | None = None) -> int` (dataset `sec-index`, parser `sec-index-1`, one chunk per quarter keyed by the quarter, `COPY` writer; coverage `("not_a_stock_source", "SEC quarterly filing index")`).
- Format: header text, a line of dashes, then `CIK|Company Name|Form Type|Date Filed|Filename`; dates `YYYY-MM-DD`.

- [ ] **Step 1: Write failing tests:** `test_parse_valid_lines`, `test_parse_quarantines_bad_date_with_reason`, `test_parse_quarantines_wrong_field_count_and_keeps_raw_line`, `test_parse_handles_company_name_with_extra_pipe_by_quarantine`, `test_quarter_of_urls`, `test_read_master_zip_reads_member`, `test_load_index_skips_quarters_not_requested` (fake cursor).
- [ ] **Step 2: Run; expect FAIL.** **Step 3: Implement.** **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** `feat: SEC quarterly index loader`.

### Task 6: Filing header parser and loader

**Files:**
- Create: `src/ggTrader/data/sec_import/headers.py`
- Test: `tests/data/sec_import/test_headers.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) HeaderRow(accession: str, cik: int, form_type: str, filed_date: date, items: tuple[str, ...])`; `parse_header(text: str) -> list[HeaderRow]` (one row per company block in the `<SEC-HEADER>`; `items` are the raw `ITEM INFORMATION:` values, kept as written; raises `HeaderIncomplete` when the standalone line `</SEC-HEADER>` is missing, when accession, form or filed date is missing, or when a company block has no `CENTRAL INDEX KEY`); `read_header_text(path: Path) -> str` (reads only up to and including the `</SEC-HEADER>` line); `load_headers(conn, entries: Sequence[CacheEntry], *, backup_dir: Path, db_free_path: Path, margin_bytes: int | None = None) -> int` (kind `filing_text` entries only; dataset `sec-headers`, parser `sec-headers-1`; a header that raises `HeaderIncomplete` goes to quarantine with the path as `row_ref`; coverage `not_a_stock_source`).
- No era mapping and no label logic: items are stored raw.

- [ ] **Step 1: Write failing tests** with small inline headers: `test_parse_single_company_8k_with_items`, `test_parse_two_companies_gives_two_rows_same_accession`, `test_parse_filed_date_formats_yyyymmdd`, `test_truncated_header_raises`, `test_quoted_closing_tag_inside_text_is_not_the_end`, `test_missing_cik_raises`, `test_read_header_text_stops_at_closing_line`, `test_load_headers_quarantines_incomplete_header`.
- [ ] **Step 2: Run; expect FAIL.** **Step 3: Implement.** **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** `feat: SEC filing header parser and loader`.

### Task 7: Label loaders

**Files:**
- Create: `src/ggTrader/data/sec_import/labels.py`
- Test: `tests/data/sec_import/test_labels.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) LabelRow(study: str, method_version: str, permno: int, last_date: date, label: str, confidence: str | None, evidence_text: str | None, suggested_loss: float | None, accessions: tuple[str, ...], source_fields: dict, source_row_sha256: str)`; `read_early_labels(csv_path: Path) -> list[LabelRow]` (study `dead-stock-early-filings`, method `early-v1-1`; `last_date` from `last_trading_day`; empty `suggested_loss` becomes `None`; `evidence` verbatim; whole CSV row as `source_fields`); `read_payout_labels(json_path: Path) -> list[LabelRow]` (study `dead-stock-data-integrity`, method `payouts-v2`; label from `evidence_class`; `last_date` from `exit_date`; `evidence_text` from `loss_note`; `confidence` from `match_confidence` or `None`; accessions from `form25_accession`, `event_accessions` and `deal_form_accessions` split on `;`, empty parts dropped; the whole row as `source_fields`); `unmatched_against_stock_list(conn, rows: Sequence[LabelRow], release_id: int) -> list[LabelRow]` (a row matches when `market.stock_spans` has the `permno` in that release with `last_date` equal to the row's `last_date`); `class LabelMismatch(Exception)` (carries the unmatched list); `class LabelConflict(Exception)`; `load_labels(conn, rows, *, source_file: Path, release_id: int = 1, backup_dir: Path, db_free_path: Path, margin_bytes: int | None = None) -> int` (dataset `sec-labels:<study>`, parser `sec-labels-1`; refuses the whole load with `LabelMismatch` if any row is unmatched; for a key `(study, method_version, permno, last_date)` already present in an earlier complete load with a different `source_row_sha256` raises `LabelConflict`, an identical row is skipped; coverage `("not_a_stock_source", "labels from <study>, copied as stored")`).
- A source row's hash is the sha256 of its canonical JSON (sorted keys). Nothing is invented: absent accessions stay an empty tuple, empty loss stays `None`.

- [ ] **Step 1: Write failing tests** on tiny fixtures that use the real column names: `test_read_early_labels_maps_columns_and_empty_loss_to_none`, `test_read_early_labels_keeps_evidence_verbatim`, `test_read_payout_labels_splits_accessions_on_semicolon`, `test_read_payout_labels_empty_confidence_is_none`, `test_row_hash_is_stable_for_key_order`, `test_unmatched_against_stock_list_reports_wrong_last_date` (integration, db), `test_load_labels_refuses_whole_load_on_mismatch` (integration), `test_load_labels_conflict_on_changed_row_and_skip_on_identical` (integration).
- [ ] **Step 2: Run; expect FAIL.** **Step 3: Implement.** **Step 4: Run; unit tests PASS (integration ones skip without a scratch URL).**
- [ ] **Step 5: Commit** `feat: SEC label loaders`.

### Task 8: CLI wiring and docs

**Files:**
- Modify: `src/ggTrader/cli/cmd_db.py`, `docs/cli_reference.md`, `docs/changelog.md`, `docs/kaggle_import_runbook.md`, `research/RESEARCH_SNAPSHOT.md` (only if a row applies; otherwise leave)
- Test: `tests/data/kaggle_import/test_import_cli.py` (extend)

**Interfaces:**
- Produces: `ggt db import sec-cache --folder PATH [--folder PATH ...]`, `sec-index --folder PATH ...`, `sec-headers --folder PATH ...`, `sec-labels --study {early,payouts} --file PATH`; each with `--backup-dir PATH` (required for the live database, as the other importers), `--db-free-path PATH`, `--db-url-env NAME | --use-configured-db`, `--yes` (without it, print what would be done and write nothing). Handlers load through `scan_cache` + `merge_caches` and call the loaders of Tasks 4 to 7.

- [ ] **Step 1: Write failing CLI tests** (parser level and dry-run, no database): `test_sec_commands_require_explicit_target`, `test_sec_commands_without_yes_write_nothing`, `test_sec_commands_refuse_no_backup_check_for_ggtrader`, `test_sec_labels_requires_study_and_file`.
- [ ] **Step 2: Run; expect FAIL.** **Step 3: Implement the subcommands.**
- [ ] **Step 4: Docs:** `docs/cli_reference.md` (four commands and `verify-data` once Task 9 lands), `docs/changelog.md` entry "2026-11-25 (database consolidation, stage 0a and A)", the runbook (stage 0a steps, owner gates, fresh-dump rule). Plain words, terms explained on first use.
- [ ] **Step 5: Run; expect PASS.** `scripts/run_tests.sh tests/data/kaggle_import/test_import_cli.py`
- [ ] **Step 6: Commit** `feat: SEC import commands and docs`.

### Task 9: `verify-data`

**Files:**
- Create: `src/ggTrader/data/sec_import/verify.py`
- Modify: `src/ggTrader/cli/cmd_db.py` (`ggt db verify-data [--deep] [--record] [--out PATH]`)
- Test: `tests/data/sec_import/test_verify.py`

**Interfaces:**
- Produces: `run_checks(conn, *, deep: bool = False) -> dict` returning `{"ok": bool, "checks": {name: {"ok": bool, "detail": str}}}`; check names (exactly): `load_row_counts_match_log`, `load_files_exist_with_recorded_size`, `no_incomplete_loads`, `every_complete_load_has_one_coverage_row`, `labels_match_stock_spells_in_release`, `cited_accessions_exist`, and with `deep=True` also `load_files_rehash` and `content_sha256_matches_stored_rows`. Row counts come from `ref.load_chunks` sums against `ref.loads.rows_accepted`, not a full `count(*)`; a load whose `file_path` starts with `load:` is checked against its parent load, not a file; the legacy datasets (`crsp-daily`, `mcgill-fiam`, `crsp-derived-bars`) are checked for counts only; `cited_accessions_exist` passes a cited accession if it appears in `sec.filing_headers` or in `sec.index_entries.filename` (the accession is part of the file name); an accession with no hit is listed, not hidden. `record(conn, report: dict, mode: str) -> int` inserts one `ref.integrity_checks` row. The command changes no data table and only writes `ref.integrity_checks` when `--record` is given. The report states what each check did and did not verify.

- [ ] **Step 1: Write failing tests** (scratch database for the DB checks; pure for the rest): `test_ordinary_run_passes_on_clean_database`, `test_detects_changed_file_size`, `test_detects_complete_load_without_coverage` (insert via a legacy dataset name to bypass the trigger, then use a fresh load) , `test_detects_label_with_wrong_last_date`, `test_derived_load_checked_against_parent`, `test_deep_detects_changed_file_bytes`, `test_deep_detects_altered_stored_row`, `test_record_inserts_one_row_and_nothing_else_changes`, `test_cli_help_and_no_write_without_record`.
- [ ] **Step 2: Run; expect FAIL.** **Step 3: Implement.** **Step 4: Run; PASS.** Ruff.
- [ ] **Step 5: Docs.** Add the `ggt db verify-data` lines to `docs/cli_reference.md` and one line to the `docs/changelog.md` entry Task 8 created. Plain words.
- [ ] **Step 6: Commit** `feat: verify-data integrity check`.

### C2 (controller): pilot, owner gate, stage A loads, report

- [ ] Run the full gate once: `scripts/run_tests.sh --full` (expect all pass), then the scratch-database integration tests of Tasks 3 to 9 with `GGT_TEST_DB_URL` set to a scratch server.
- [ ] Confirm a dump no older than 7 days exists in `/media/thesix/ggTrader-data/db-dumps/` (take a fresh one with C1's command if not) and the restore drill passed.
- [ ] **Owner gate G-load (apply and pilot).** Tell the owner what will be written, then, on their yes: `ggt db import apply-ddl --use-configured-db --yes` (adds the new tables only); pilot the index loader on one quarter (`2005Q2`); record time, rows, size, free space on the main drive.
- [ ] Owner sees the pilot numbers; on their yes, load in this order, each with `--yes`, one at a time: `sec-cache` (both folders), `sec-index` (all 105 quarters), `sec-headers`, then the two `sec-labels` loads (early file, then payouts file). Stop at the first refusal (`LabelMismatch` lists unmatched stocks; show the list to the owner, do not force).
- [ ] `ggt db verify-data --record` (ordinary), then `--deep --record` (heavy-job lock, outside 02:00). All checks pass or each failure is explained.
- [ ] Speed checks from the design (5.8) that apply to stage A: one company's filings for a year and one stock's label with its evidence, each under one second; report before and after times.
- [ ] Report to the owner in plain words: rows and size by table, quarantined rows and why, free space, check results, timings, what changed from the plan, and usage by lane. Update the changelog and memory. Do not merge or push without the owner's say.

## Self-review

- **Spec coverage:** 5.0 steps 1 to 3 (Tasks 1, 2, C1); 5.1 coverage rules (Task 3); 5.2 scope, tables, read-only caches, labels (Tasks 3 to 7; `company_matches` dropped, recorded in the design); 5.6 multi-file loads and coverage with completion (Task 3); 5.7 ordinary and deep checks (Task 9); 5.8 stage A targets (C2); 5.10 docs (Task 8); gates G-restore and G-load (C1, C2). Stage 0b, stages B to D, the shared SEC downloader and the research login are outside this plan by decision.
- **Names:** `CacheEntry`, `HeaderRow`, `LabelRow`, `snapshot`, `record_files`, `complete_load(..., coverage=...)`, dataset names and parser versions are used identically wherever they appear.
- **Not in the spec but decided here:** the `.part`/`.mismatch` file names, exit codes, the weekly Sunday 04:30 timer time, `early-v1-1` and `payouts-v2` method names, the legacy-dataset list for the coverage trigger.
