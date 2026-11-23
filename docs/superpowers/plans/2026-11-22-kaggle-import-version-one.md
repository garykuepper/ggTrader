# Kaggle Import, Version One Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the Kaggle CRSP daily file and the McGill sample into the `ggtrader` database, derive stock-spell bars from CRSP, link them to the Yahoo tape, and publish them as numbered releases that `load_bars(source="combined")` reads, with an exit rule for stocks that stop trading.

**Architecture:** A small shared library (`ggTrader.data.kaggle_import`) does hash check, free-space and backup guards, chunked transactional `COPY` and a load log. Dataset modules load files as delivered into `kaggle_*` schemas. Pure functions build spells, closes, links and splice windows; a release row pins one compatible set; a plain-SQL function `market.ohlcv_daily_combined` reads one release. All development runs on throwaway databases; the live `ggtrader` database is written only in Task 13.

**Tech Stack:** Python 3.12, pandas, pyarrow, psycopg2, PostgreSQL 16 + TimescaleDB 2.25, pytest (only through `scripts/run_tests.sh`).

**Spec:** `docs/superpowers/specs/2026-11-22-kaggle-import-design.md` (sections cited as "spec N"). Plan for change zero (the `load_bars` default flip and the bars hash): `docs/superpowers/plans/2026-11-22-load-bars-db-default.md`, which must land first. Version two (GKX, SEC, Form 4, S&P files, ticker translation) is a later plan.

## Global Constraints

- Nothing is written to the `ggtrader` database before Task 13, and Task 13 needs the owner's go-ahead (gates below). Tests use throwaway databases via `tests/data/scratch_db.py` (`GGT_TEST_DB_URL`; tests marked `integration` and `db`).
- `ohlcv` and its `yfinance` rows are never touched. No new `timestamp without time zone` column anywhere: dates are `date`, instants `timestamptz` (spec 3).
- Placeholder symbols are `PERMNO-<permno>`; spell labels `PERMNO-<permno>~<k>`; both pass `normalize_yf_ticker` unchanged (spec 6.1).
- Defaults: at least 40 GB free after the projected peak; a backup dump no older than 7 days; at most two loads per dataset kept; a load a readable release depends on is never deleted (spec 3, 6.5).
- Gap rule values: tolerance 0.02, longest bridged gap 10 sessions, calendar = the file's own 14,347 sessions (spec 6.1). Expected file counts: 21,024 gap rows; 18,234 short; 248 short rows cut (214 market-value disagreement, 34 zero or missing shares); 2,790 long, all cut.
- Link numbers (sessions, correlation, margin) are **not** fixed here; Task 9 calibrates them and the owner approves the table (spec 6.3).
- Heavy data work only through `GGT_RUN_MEM_MAX=3G scripts/run_capped.sh .venv/bin/python -I <script>`; never read the 69-million-row CRSP file whole; read by `year` filter or per stock with pyarrow `columns=`/`filter=`. One heavy job at a time; if a script prints that it is waiting for another heavy ggTrader job, wait. Never bypass the scripts, never kill another process.
- Tests run only as `scripts/run_tests.sh <small targeted paths>`. Never run `tests/paper/`, a whole tier or an oracle driver in a task; the orchestrator runs those, one at a time.
- Ruff-clean, type hints, 100-column lines, module docstrings, `argparse` and `--help` on every script, long jobs resumable (spec 3; AGENTS.md section 4).
- Raw inputs (read-only): `/media/thesix/ggTrader-data/market-data/crsp-kaggle/cleaned/crsp_1968_2024_daily_cleaned.parquet`, `.../crsp-kaggle/raw/crsp_data_dump_daily.csv`, `.../kaggle/explore-2026-11-22/mcgill-fiam/hackathon_sample_v2.csv`, hash lists `SHA256SUMS` beside them.

## Owner gates (the plan stops and asks)

- **G1** before any write to `ggtrader`: the owner accepts the 2026-11-22 backup, or a restore drill has passed (Task 6 supplies the drill script).
- **G2** after the pilot (Task 6): the owner approves the projected size, peak and free-space margin.
- **G3** after calibration (Task 9): the owner approves the written link table.
- **G4** before wiring the dead-stock rule into `src` (Task 11).
- **G5** before the first release is published (Task 13).

## Review Focus

- A stock with a split inside a gap (price halves, return near zero, one missing session before it) must be cut by the market-value test only when shares did not change by the split ratio; a split on consecutive sessions must never cut.
- Take-Two (84761), Comverse (10942), Super Micro (91907), Visteon (88319), stock 10128 and AAPL each yield the exact columns in spec 6.1 (tests pin them).
- A load interrupted mid-chunk and resumed, a repeated chunk, and the same bytes with a new parser version each give exactly the right row counts and never a double.
- Reading a pinned release after Yahoo levels are rescaled (a dividend) shows no jump at the join and no drift flag; a changed Yahoo return in the join window sets the drift flag.
- `combined` on a bankrupt stock with an open position exits it on the last valid bar and takes no later order, for signal and weight strategies, with the same result for two different window end dates.

---

### Task 1: Schema DDL and `ggt db import apply-ddl`

**Files:**
- Create: `sql/kaggle_schema.sql`, `src/ggTrader/data/kaggle_import/__init__.py`, `src/ggTrader/data/kaggle_import/ddl.py`
- Modify: `src/ggTrader/cli/cmd_db.py` (register `import` group with `apply-ddl`)
- Test: `tests/data/kaggle_import/test_ddl.py` (marked `integration`, `db`)

**Interfaces:**
- Produces: `apply_ddl(conn) -> None` in `ddl.py` (runs `sql/kaggle_schema.sql`, idempotent); schemas `kaggle_crsp`, `kaggle_chars`, `market`, `ref`; tables:
  - `ref.loads(load_id bigint identity pk, dataset text, kaggle_slug text, file_path text, file_sha256 text, parser_version text, content_sha256 text, rows_parsed bigint, rows_accepted bigint, rows_quarantined bigint, loader_git_sha text, loaded_by text, started_at timestamptz, completed_at timestamptz null, status text check in ('running','complete','failed','withdrawn'), checks jsonb)`
  - `ref.load_chunks(load_id, chunk_key text, rows bigint, committed_at timestamptz, pk(load_id, chunk_key))`
  - `ref.quarantine(load_id, chunk_key, row_ref text, raw jsonb, reason text)`
  - `ref.releases(release_id bigint identity pk, created_at timestamptz, settings jsonb, status text check in ('published','withdrawn'), yahoo_fingerprint jsonb)`, `ref.release_loads(release_id, load_id, role text)`
  - `ref.symbol_map(permno bigint, symbol text, valid_from date, valid_to date, source text, release_id bigint)`
  - `ref.splice_links(release_id bigint, permno bigint, symbol text, first_yf_date date, link_day_returns_hash text, agreement jsonb, pk(release_id, permno))`
  - `kaggle_crsp.daily(load_id bigint, permno bigint, date date, shrcd int, exchcd int, prc double precision, ret double precision, shrout double precision, vwretd double precision, year int, month int, pk(load_id, permno, date))` as a hypertable on `date`, one-year chunks, compression by `permno`.
  - `kaggle_chars.mcgill` (columns copied from the CSV header; `load_id` first; dates as `date`).
  - `market.daily_bars(load_id bigint, permno bigint, spell int, date date, close double precision, abs_prc double precision, midpoint_flag boolean, gap_return_approx boolean, final_flag boolean, pk(load_id, permno, date))` hypertable, one-year chunks, compression by `permno`; `market.stock_spans(release_id, symbol text, permno bigint, spell int, first_date date, last_date date, linked boolean, pk(release_id, symbol))`.
- Append-only: triggers (for every role, as in `sql/universe_schema.sql`) on `ref.loads` (no delete; no update once `status='complete'` except to `'withdrawn'`), `ref.releases`, `ref.symbol_map`, `ref.splice_links`. None on the hypertables (spec 8).

- [ ] **Step 1: Write failing tests:** `test_ddl_is_idempotent` (apply twice, no error); `test_no_timestamp_without_time_zone_columns` (query `information_schema.columns` for the four schemas, expect none); `test_complete_load_is_immutable` (insert a load, set `complete`, an update of `rows_parsed` raises, setting `withdrawn` succeeds); `test_hypertables_exist` (`timescaledb_information.hypertables` lists both).
- [ ] **Step 2: Run** `GGT_TEST_DB_URL=<scratch server> scripts/run_tests.sh tests/data/kaggle_import/test_ddl.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the SQL (model it on `sql/universe_schema.sql`: `IF NOT EXISTS`, `CREATE OR REPLACE`, `DROP TRIGGER IF EXISTS`) and `apply_ddl`, then the CLI command `ggt db import apply-ddl`.
- [ ] **Step 4: Run** the same command. Expected: PASS. **Step 5: Commit** `feat: kaggle import schemas and apply-ddl`.

### Task 2: Guards (hash, free space, backup age)

**Files:**
- Create: `src/ggTrader/data/kaggle_import/guards.py`
- Test: `tests/data/kaggle_import/test_guards.py`

**Interfaces:**
- Produces: `verify_sha256(path: Path, sums_file: Path) -> str` (returns the hex digest, raises `HashMismatch` naming file and both digests); `free_space_ok(path: Path, needed_bytes: int, margin_bytes: int = 40 * 1024**3) -> bool`; `backup_fresh(backup_dir: Path, max_age_days: int = 7, now: datetime | None = None) -> Path` (returns the newest `*.dump`, raises `BackupStale`).

- [ ] **Step 1: Failing tests** with `tmp_path` fixtures: a correct and a corrupted file against a one-line `SHA256SUMS`; `free_space_ok` with a monkeypatched `shutil.disk_usage` at exactly margin, margin minus 1 byte; `backup_fresh` with dumps of age 6 and 8 days and an empty directory.
- [ ] **Step 2: Run, expect FAIL. Step 3: Implement** (stream the hash in 1 MiB blocks; `SHA256SUMS` lines are `digest  relative/path`). **Step 4: Run, PASS. Step 5: Commit.**

### Task 3: Load log and transactional chunks

**Files:**
- Create: `src/ggTrader/data/kaggle_import/loadlog.py`
- Test: `tests/data/kaggle_import/test_loadlog.py` (integration, db)

**Interfaces:**
- Consumes: Task 1 tables.
- Produces: `begin_load(conn, *, dataset: str, kaggle_slug: str, file_path: Path, file_sha256: str, parser_version: str, content_sha256: str) -> int` (new load id; if a `running` load with the same `(file_sha256, parser_version)` exists, returns its id so it can resume; if a `complete` one exists raises `AlreadyLoaded`; if two `complete` loads of the dataset already exist raises `RetentionLimit` unless `allow_third=True`, which the owner must request); `commit_chunk(conn, load_id: int, chunk_key: str, rows: int, writer: Callable[[cursor], int]) -> bool` (runs the writer and inserts the `ref.load_chunks` row in one transaction; returns `False` and writes nothing if the chunk key already exists); `quarantine_rows(conn, load_id, chunk_key, rows: Sequence[tuple[str, dict, str]]) -> None`; `complete_load(conn, load_id: int, checks: dict) -> None` (requires every expected chunk, equality `parsed == accepted + quarantined`); `can_delete_load(conn, load_id: int) -> bool` (false while a published release names it); `LoadFailed`, `AlreadyLoaded`.

- [ ] **Step 1: Failing tests:** `test_third_load_refused_without_owner_flag`; `test_interrupted_load_resumes` (commit chunks "1999", "2000", raise before "2001", call `begin_load` again: same id, `commit_chunk("1999")` returns False, "2001" commits); `test_repeated_chunk_never_doubles_rows`; `test_same_bytes_new_parser_version_is_a_new_load`; `test_complete_requires_parsed_equals_accepted_plus_quarantined`; `test_can_delete_load_false_while_release_depends`; `test_nan_goes_to_null_not_quarantine` is in Task 4.
- [ ] **Step 2: Run, FAIL. Step 3: Implement** with plain `psycopg2`, one connection, `conn.commit()` after each chunk. **Step 4: Run, PASS. Step 5: Commit.**

### Task 4: CRSP loader (as delivered)

**Files:**
- Create: `src/ggTrader/data/kaggle_import/crsp.py`; modify `cmd_db.py` (`ggt db import crsp --file PATH --yes`)
- Test: `tests/data/kaggle_import/test_crsp_load.py` (integration, db; builds a 3-stock Parquet in `tmp_path` with the real schema from `crsp-kaggle/parquet_schemas.json`)

**Interfaces:**
- Consumes: Tasks 2-3.
- Produces: `PARSER_VERSION = "crsp-1"`; `load_crsp(conn, parquet_path: Path, sums_file: Path, *, backup_dir: Path | None, years: Sequence[int] | None = None, margin_bytes: int = 40 * 1024**3) -> int` (load id). One chunk per `year`; each chunk read with a pyarrow `filter` on `year` and only the file's columns; `shrcd`/`exchcd` cast double to int (refuse a non-integral value, fatal); NaN/inf stored NULL and counted; a `date` that is not midnight is fatal; any date outside 1968-01-01..2024-12-31 is quarantined.

- [ ] **Step 1: Failing tests:** `test_rows_as_delivered` (negative `prc` kept negative); `test_shrcd_double_becomes_int_and_fractional_is_fatal`; `test_year_chunks_and_resume`; `test_quarantine_out_of_window_date` (1900-01-01 row to quarantine, counts reconcile); `test_refuses_when_hash_mismatch`; `test_refuses_when_free_space_short` (monkeypatch `free_space_ok`).
- [ ] **Step 2: Run, FAIL. Step 3: Implement.** Use `COPY ... FROM STDIN` through `cursor.copy_expert` with an in-memory CSV of one year (about 1.2 million rows). **Step 4: Run, PASS. Step 5: Commit.**

### Task 5: McGill loader (raw only)

**Files:**
- Create: `src/ggTrader/data/kaggle_import/mcgill.py`; modify `cmd_db.py` (`ggt db import mcgill`)
- Test: `tests/data/kaggle_import/test_mcgill_load.py`

**Interfaces:**
- Produces: `PARSER_VERSION = "mcgill-1"`; `load_mcgill(conn, csv_path: Path, sums_file: Path, *, backup_dir: Path | None) -> int`. Every column declared explicitly (no type inference); `ret_eom` and `date` stored as `date`; ticker, CUSIP and name kept as text; nothing exposed to signals (spec 7).

- [ ] **Step 1: Failing tests** against a 10-row CSV with the real header: `test_declared_types_not_inferred`; `test_null_ticker_rows_kept`; row-count reconciliation. **Steps 2-5** as Task 4.

### Task 6: Pilot, backup drill and the G1/G2 gates

**Files:**
- Create: `scripts/kaggle_pilot.py`, `scripts/kaggle_restore_drill.py` (adapt `scripts/membership_restore_drill.py` to restore `ggtrader_20261122.dump` into a scratch database and compare row counts with `rowcounts_before.txt` in the backup folder)
- Test: `tests/scripts/test_kaggle_pilot_cli.py` (`--help` and argument parsing only)

**Interfaces:**
- Produces: `kaggle_pilot.py --year N --scratch-url URL --out results/kaggle_pilot/pilot.json`: creates a throwaway database on the same server, applies the DDL, loads the densest year (default: the year with most rows, found from the `year` column counts), compresses it, loads it again as a second version, records `hypertable_size` after each step, free space at each step and the elapsed time, projects full CRSP size and the reload peak (rows by year from the file, bytes per row measured), drops the database, and exits 0 with the JSON.

- [ ] **Step 1: Failing CLI test** (`--help` exits 0; missing `--scratch-url` exits 2). **Step 2-4:** implement, pass.
- [ ] **Step 5 (orchestrator, G1):** run the restore drill under `scripts/run_capped.sh`, one heavy job at a time; or record the owner's acceptance of the backup. Ask the owner.
- [ ] **Step 6 (orchestrator, G2):** run the pilot; give the owner the JSON in plain words (size, peak, free after, margin). **Stop until the owner approves.** Commit the scripts only.

### Task 7: Spells (pure functions)

**Files:**
- Create: `src/ggTrader/data/kaggle_import/spells.py`, `tests/data/kaggle_import/fixtures/crsp_cases.csv` (rows for permnos 10066, 10124, 10128, 10669 around the events, and 12490 (IBM), 14593 (AAPL), 10107 (MSFT) around two splits each; built by `scripts/kaggle_make_fixture.py` from the raw/cleaned files, under `run_capped.sh`, under 300 KB)
- Test: `tests/data/kaggle_import/test_spells.py`

**Interfaces:**
- Produces: `assign_spells(stock: pd.DataFrame, calendar: pd.DatetimeIndex, *, tolerance: float = 0.02, max_gap: int = 10) -> pd.DataFrame`. Input: one stock sorted by `date`, columns `date, prc, ret, shrout`. Output: input plus `session` (position on `calendar`), `spell` (int, 0 = oldest), `chained_return` (float, NaN on a spell's first row), `gap_return_approx` (bool). Rules (spec 6.1): previous session present means chain `ret`; after a missing session, bridge only if gap <= `max_gap` and shares non-zero on both sides and `abs((mv_t/mv_prev - 1) - ret) <= tolerance` with `mv = abs(prc) * shrout`, and then `chained_return = mv_t/mv_prev - 1` and `gap_return_approx = True`; otherwise start a new spell. `calendar_from_file(parquet_path: Path) -> pd.DatetimeIndex` reads only the `date` column.

- [ ] **Step 1: Failing tests:**
  - `test_consecutive_split_is_not_cut` (IBM, AAPL, MSFT fixtures: one spell each).
  - `test_10066_cut_where_the_gap_exceeds_ten_sessions` (the 1990-01-29 to 1999-03-19 and 2002 breaks; 4 spells with the Nov 2001 to Feb 2002 break, as measured).
  - `test_10128_cut_april_1986` (4/11 to 4/23: market-value ratio -37.8% against reported -8%, new spell).
  - `test_10669_bridged_with_price_return` (assert `chained_return` on 1987-05-08 equals `18.00/18.25 - 1` within 1e-9, not 0.0, and `gap_return_approx` is True).
  - `test_zero_shares_after_gap_starts_new_spell`.
  - `test_full_file_counts` (marked `slow`, reads the real files by `columns=`; asserts 21,024, 18,234, 248 (214 + 34) and 2,790).
- [ ] **Step 2: Run, FAIL. Step 3: Implement** vectorised (groupby over sessions; no row loops). **Step 4: Run (fast tests), PASS; the orchestrator runs the slow one capped. Step 5: Commit.**

### Task 8: Derived bars and stock spans

**Files:**
- Create: `src/ggTrader/data/kaggle_import/derive.py`; modify `cmd_db.py` (`ggt db import derive-bars --crsp-load N`)
- Test: `tests/data/kaggle_import/test_derive.py` (pure part; integration part for the DB write)

**Interfaces:**
- Consumes: `assign_spells`.
- Produces: `build_spell_closes(spelled: pd.DataFrame) -> pd.DataFrame` adding `close` (per spell: last row's `abs(prc)`, earlier rows by dividing by `1 + chained_return` of the following row), `abs_prc`, `midpoint_flag` (`prc < 0`), `final_flag` (last row of a stock's last spell). A `chained_return <= -1` raises `ValueError` naming permno and date (none exists in the file; fail loud, spec 6.4). `derive_bars(conn, crsp_load_id: int) -> int` fills `market.daily_bars` per year chunk through `commit_chunk` and returns a new load id.

- [ ] **Step 1: Failing tests:** `test_close_is_anchored_to_spell_last_price`; `test_rescaled_close_has_correct_day_to_day_returns` (compare `close.pct_change()` with `chained_return`); `test_abs_prc_and_midpoint_flag`; `test_total_loss_row_raises`; `test_split_does_not_change_returns` (AAPL fixture).
- [ ] **Step 2-5** as above. The DB write test loads the three-stock fixture into a scratch DB and checks row counts and compression.

### Task 9: Links, calibration and the G3 gate

**Files:**
- Create: `src/ggTrader/data/kaggle_import/links.py`, `scripts/kaggle_calibrate_links.py`
- Test: `tests/data/kaggle_import/test_links.py`

**Interfaces:**
- Produces: `LinkSettings(min_sessions: int, min_sessions_short: int, agreement_bar: float, margin: float, window_start: date, hold_first_days: int)` (frozen dataclass; no defaults for the calibrated numbers); `agreement(yahoo_ret: pd.Series, crsp_ret: pd.Series) -> dict` (shared sessions, correlation, share of days within 0.5 points, first-half and second-half values split by date); `find_links(yahoo: Mapping[str, pd.Series], crsp: Mapping[int, pd.Series], first_rows: Mapping[...], settings: LinkSettings) -> pd.DataFrame` (columns `symbol, permno, status in {linked, unlinked, ambiguous, duplicate}, agreement, reason`). Rules (spec 6.3): enough shared sessions or the short-listing rule (CRSP's first row on Yahoo's first day or at most one session after); agreement above the bar in both halves; best beats second by `margin`; a tie is ambiguous; a second Yahoo symbol claiming the same permno is `duplicate`; a link is refused when CRSP has more than 20 sessions of data after Yahoo's last day.
- `kaggle_calibrate_links.py` writes `results/kaggle_calibration/table.json`: for the independently known matches taken from the McGill dated tickers (AAPL, MSFT, IBM, XOM, META, GOOG/GOOGL, PX/LIN, old and new GM) and a held-out set of stocks not used for calibration, the agreement values and the proposed `LinkSettings`.

- [ ] **Step 1: Failing tests** with synthetic series: `test_identical_returns_link`; `test_noisy_first_year_still_links_on_late_window` (noise in 2000 only, window starting 2005 links; whole-window correlation fails); `test_two_candidates_within_margin_is_ambiguous`; `test_second_yahoo_symbol_for_same_permno_is_duplicate`; `test_crsp_outliving_yahoo_is_refused`; `test_start_day_match_allows_one_session_later`; `test_join_check_uses_first_k_sessions_after_the_join_not_the_join_day` (a wrong-company Yahoo series passes the join-day return check but fails the first-K comparison); `test_noisy_2000_join_uses_loose_report_only_tolerance`.
- [ ] **Step 2-4:** implement; pass.
- [ ] **Step 5 (orchestrator, G3):** run the calibration capped, present the table in plain words (which known matches link, how many false matches on the held-out set), and **stop until the owner approves the numbers**. Commit.

### Task 10: Releases and the combined reader (SQL)

**Files:**
- Create: `src/ggTrader/data/kaggle_import/release.py`; extend `sql/kaggle_schema.sql` with `market.ohlcv_daily_combined`
- Test: `tests/data/kaggle_import/test_release.py` (integration, db)

**Interfaces:**
- Produces: `build_release(conn, *, crsp_bars_load: int, mcgill_load: int | None, settings: LinkSettings) -> int` (finds links against the live `ohlcv` Yahoo rows, writes `ref.symbol_map`, `ref.splice_links` (with `link_day_returns_hash` over the first `hold_first_days` Yahoo returns after the join), `market.stock_spans`, a `ref.releases` row with the Yahoo fingerprint (SHA-256 over those windows) and `ref.release_loads`); `withdraw_release(conn, release_id) -> None`; `stocks_alive_on(conn, release_id: int, day: date) -> pd.DataFrame` (symbol, permno, spell from `market.stock_spans`, the dead-stock-safe universe of spec 6.2); `ReleaseUnavailable`; CLI `ggt db import release --build | --publish | --withdraw N` in `cmd_db.py`. Link demotion (a zero or missing Yahoo close on the join day, spec 6.4) happens here, at build time only.
- SQL: `market.ohlcv_daily_combined(p_release_id bigint, p_symbols text[], p_start date, p_end date)` returns `(date date, symbol text, open double precision, high double precision, low double precision, close double precision, volume double precision, source text, abs_prc double precision, midpoint_flag boolean, possible_double boolean, drift boolean)` in one statement, so bars and flags share one snapshot. Naming table of spec 6.1; scale = Yahoo close / CRSP close on the first common day within 5 sessions, computed from the same `ohlcv` rows; Yahoo rows used only from the release's `first_yf_date`; a missing join-day Yahoo row raises an error named `ReleaseInputMissing`; the cheap drift check compares the join-window returns hash; the function raises `release unavailable` for a withdrawn release and returns zero rows (no error) for no matching symbols.

- [ ] **Step 1: Failing tests** on a scratch DB seeded with small hand-made CRSP and Yahoo fixtures:
  - `test_ttwo_trace` (spell A before 2000-01-03 scaled then Yahoo; spell B unreadable; no `~1` column).
  - `test_comverse_trace`; `test_super_micro_no_splice_all_crsp_hidden`.
  - `test_visteon_linked_and_unlinked` (exact three-column unlinked result, `VC` and `PERMNO-88319` flagged `possible_double`).
  - `test_scale_survives_yahoo_rescale` (multiply all Yahoo closes by 0.9: no jump at the join, `drift` False).
  - `test_changed_join_window_return_sets_drift`.
  - `test_backfilled_yahoo_rows_are_ignored_and_flagged`.
  - `test_missing_join_row_raises_release_input_missing`.
  - `test_withdrawn_release_is_unavailable_but_empty_query_is_not`; `test_zero_yahoo_anchor_demotes_link_at_build_not_at_read`; `test_stocks_alive_on_includes_dead_stocks`; `test_mcgill_tickers_recorded_not_used_as_names`.
  - `test_no_stock_readable_under_two_names_same_day` (except flagged doubles).
  - `test_delete_blocked_while_release_depends` (via `can_delete_load`).
  - `test_pushdown` (`EXPLAIN` shows the date and symbol filters reach `market.daily_bars`; if it does not, implement the materialized fallback of spec 6.2 and rerun).
- [ ] **Step 2-5** as above.

### Task 11: Dead-stock rule (pure functions, then the G4 gate)

**Files:**
- Create: `src/ggTrader/lab/dead_stocks.py`
- Test: `tests/lab/test_dead_stocks.py`

**Interfaces:**
- Produces: `forced_exit_frames(close: pd.DataFrame, spans: pd.DataFrame) -> DeadStockFrames` (frozen dataclass: `exits: pd.DataFrame` bool, `entry_ok: pd.DataFrame` bool, `weight_ok: pd.DataFrame` bool; `spans` has `symbol, last_date`; the last valid bar comes from `last_date`, **not** from trailing empty cells, so the result does not depend on the window end); `apply_weight_rule(weights: pd.DataFrame, frames: DeadStockFrames) -> pd.DataFrame` (0.0 on the last valid bar, NaN after); `haircut_close(close: pd.DataFrame, spans: pd.DataFrame, loss: float) -> pd.DataFrame` (final bar close times `1 - loss` for stocks whose data ends without a successor; used for the 0%, 50%, 100% sensitivity runs).

- [ ] **Step 1: Failing tests:** `test_exit_on_last_valid_bar_and_no_entries_after`; `test_weights_zero_then_no_order`; `test_window_end_independence` (two windows ending 30 and 90 days after a stock's last bar give identical exits through the common dates); `test_temporary_outage_is_not_dead` (a stock with a later spell is not exited); `test_haircut_scales_only_the_last_bar`.
- [ ] **Step 2-4:** implement; pass; commit the pure module.
- [ ] **Step 5 (G4):** read `src/ggTrader/lab/wfo/run.py` and `src/ggTrader/engine/portfolio.py` and write a half-page proposal (`docs/superpowers/plans/dead-stock-wiring-proposal.md`) naming where `forced_exit_frames`/`apply_weight_rule` plug in so every `combined` run applies them and records the optimism in provenance (sensitivity runs at 0%, 50%, 100%); state the fallback (`combined` refuses weight strategies with a clear error). **Stop for the owner's approval** because it touches `src`.
- [ ] **Step 6 (after approval):** implement the wiring with tests `test_combined_run_applies_forced_exit` (signal) and `test_combined_run_weight_strategy_exits_dead_stock` (or `test_combined_refuses_weight_strategy_if_unwired`), a bankrupt-stock fixture with an open position and an order after the end, expected cash, holdings and returns, two end dates agreeing through common history. Commit.

### Task 12: `load_bars(source="combined")`, flags and provenance

**Files:**
- Modify: `src/ggTrader/data/bars.py` (`SOURCES` gains `"combined"`; `load_bars(..., release_id: int | None = None, ...)`; `Bars` gains optional `flags: pd.DataFrame | None = None`), `src/ggTrader/engine/provenance.py` (a differing `data_release` or `tape_sha256` classifies `TAPE_DIFF`), `src/ggTrader/lab/wfo/cli_support.py` and `lab/blend.py` (record `data_release` in the run provenance and persisted blend record), `src/ggTrader/lab/cli.py` (`--source combined`, `--release-id N`)
- Test: `tests/data/test_bars_combined.py`, `tests/engine/test_provenance_release.py`, `tests/lab/test_cli.py`

**Interfaces:**
- Consumes: `market.ohlcv_daily_combined` (Task 10), `bars_sha256` (change zero).
- Produces: `_read_combined_long(release_id: int | None, symbols, start, end) -> tuple[pd.DataFrame, pd.DataFrame, int]` (flat bars, flags, resolved release id; one SQL call); `Bars.provenance` gains `source="combined"`, `data_release`, `drift` (bool), `bars_sha256`.

- [ ] **Step 1: Failing tests:** `test_combined_returns_five_fields_and_flags_from_one_call`; `test_missing_symbols_lists_unknown_permno_and_k` (`PERMNO-84761`, `PERMNO-1~7`); `test_default_release_is_newest_and_id_is_recorded`; `test_drift_flag_reaches_provenance`; `test_different_release_is_tape_diff_not_comparable`; `test_validate_long_still_enforces_weekday_and_duplicates_on_combined`; `test_lab_cli_source_combined_parses`; a test that the saved WFO JSON and the persisted blend record both carry `data_release`; `test_smoke_close_only_strategy_on_explicit_symbols` (spec 6.7: `run_wfo` on an explicit list of one placeholder, one linked and one dead stock; asserts eligible-stock count is not zero and exact expected trades, cash, holdings and returns; the lab's universe choices are not changed).
- [ ] **Step 2-5** as above.

### Task 13: Closing checks and the first full load (G1, G2, G5)

**Files:**
- Create: `scripts/kaggle_closing_checks.py` (implements spec 9 checks 1-13 as a report with pass/fail per check and writes `results/kaggle_checks/<date>.json`)
- Test: `tests/scripts/test_kaggle_closing_checks_cli.py` (`--help`, argument parsing)

- [ ] **Step 1 (orchestrator, gates):** confirm G1, G2 and G3 are recorded. Run `scripts/run_tests.sh --fast`, then `--full` once, alone. Expected: PASS. Confirm `backup_fresh` passes (re-dump if the backup is older than 7 days).
- [ ] **Step 2 (orchestrator, the one write to the live database):** under `run_capped.sh`: `ggt db import apply-ddl`; `ggt db import crsp --yes`; `ggt db import mcgill --yes`; `ggt db import derive-bars --crsp-load <id>`; check free space after each year chunk (the library already stops at the margin).
- [ ] **Step 3:** build links with the approved settings and **stop for G5**: show the owner the link counts, flagged doubles, ambiguous list and the closing-check report; publish the release only on approval (`ggt db import release --publish`).
- [ ] **Step 4:** run `scripts/kaggle_closing_checks.py` against the live database (read-only) and file the JSON. Verify check 4: `load_bars(source="db")` for the current Yahoo symbols is byte-identical before and after.
- [ ] **Step 5: Docs:** update `AGENTS.md` (Database map: new schemas and tables, `combined` source, release id, dead-stock caveat), `docs/cli_reference.md`, `docs/architecture.md`, `docs/changelog.md`; update the memory note. Commit `docs: kaggle import version one`.

---

## Delegation and quota (CLAUDE.md, 2026-11-08)

The orchestrator cannot see plan usage. **Before starting, ask the owner for current Claude and Codex headroom** and size the work to it. Default phased version if unknown:
- **Phase 1:** change-zero plan, then Tasks 1-6 (stops at G1/G2).
- **Phase 2:** Tasks 7-9 (stops at G3).
- **Phase 3:** Tasks 10-13 (stops at G4 and G5).
Builders: Codex Terra for Tasks 1-5, 7-10; Claude Sonnet for Tasks 11-12 and the docs. Reviewers: Claude output goes to Codex Sol, Codex output to Claude Opus; anything headed for a merge gets both. The orchestrator re-reads every diff and reruns the checks itself, runs the heavy tests one at a time, and reports usage after each phase. Astra and Argon are reserved for critical judgments and need the owner's say.
