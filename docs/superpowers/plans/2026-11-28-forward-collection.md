# Forward-Check Data Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start collecting free daily stock prices and a daily list of listed stocks, kept in append-only form with stocks that stop trading preserved, so a rule can later be scored on data that arrived after it was registered.

**Architecture:** One small daily collector reads a free listing file and free daily prices after the close, writes them to a new `forward` schema that refuses updates and deletes, and records every gap. A user timer runs it on trading days. A weekly health report shows coverage. Choosing the price source (Task 1) and approving writes to the live database (Task 2) are owner gates.

**Tech Stack:** Python, pandas, Postgres/Timescale (new `forward` schema), systemd user timer, `scripts/run_tests.sh`.

**Spec:** Agenda choice 5 in `research/programme/next-research-agenda-2026-11-26.md` and the ledger note in `research/programme/window-use-ledger.md` (held Yahoo rows stop on 2026-11-05; no timer refreshes them; normal coverage is about 623 symbols).

## Global Constraints

- Free data only; no paid source. Taxable account; the paper trader stays stopped; no orders.
- Information known on each date; keep stocks that later stop trading (never overwrite or drop a symbol that disappears).
- New tables follow the project's import rules: append-only triggers for every role, idempotent DDL, timezone-aware timestamps, naive UTC for bar timestamps, one loads row per collection with source, raw and normalized SHA-256, code version and time.
- Writes to the live database need the owner's explicit yes. A dump now exists and is test-restored (`ggtrader_20261126.dump`); a fresh dump must be taken before the first write.
- This plan collects data only. It starts no rule test: a rule must first survive choices 1-4, and none has. A rule registered later is scored only on data dated after its registration commit.
- Plain words in owner-facing text. No GO/NO-GO labels. Tests only through `scripts/run_tests.sh`; the collector runs light, so it needs no overnight slot.

## Review Focus

- A symbol that vanishes from the listing or from the price source. Expected: recorded as a gap, kept, and flagged `ended` only after five consecutive missing trading days, never deleted.
- A rerun on the same day. Expected: no duplicate rows (idempotent), same loads hash recorded as a repeat.
- A holiday or half day. Expected: no false gaps; the NYSE calendar decides which days must have bars.
- A symbol that changes ticker or relists. Expected: stored by the vendor's raw symbol with the listing snapshot beside it, no silent merge.
- Source returns fewer rows than listed. Expected: the run succeeds, writes what it got, and the health report shows the coverage drop; a coverage below 90% sends an alert.

---

### Task 1: Pilot the free sources for one day (read-only)

**Files:**
- Create: `research/stock-signals/forward-check/` via `scripts/study.py new stock-signals forward-check --kind screen --placement 9`
- Create: `code/fc_pilot.py`, test `code/tests/test_fc_pilot.py`; output `_source_pilot.json`

**Interfaces:**
- Produces: `fetch_listing() -> pd.DataFrame` (columns `symbol, exchange, is_etf, is_test`); `fetch_bars(symbols: list[str], day: date) -> pd.DataFrame` (columns `symbol, open, high, low, close, volume`); `coverage(listed: pd.DataFrame, bars: pd.DataFrame) -> dict` with keys `listed`, `returned`, `missing`, `share`; `price_agreement(bars: pd.DataFrame, reference: pd.DataFrame, tol: float) -> float` (share of symbols within `tol`).

- [ ] **Step 1: Write failing tests on toy frames:** `test_coverage_counts_missing_symbols`, `test_price_agreement_uses_tolerance`, `test_listing_filter_drops_etf_and_test_issues`.
- [ ] **Step 2:** Run `scripts/run_tests.sh research/stock-signals/forward-check/code/tests/test_fc_pilot.py`; expect FAIL.
- [ ] **Step 3:** Implement the functions for three free sources: the Nasdaq Trader symbol-directory files for the listing, and for prices Alpaca's free daily bars and Yahoo's bulk download. Run once on the last full trading day; compare each source's prices with the frozen Yahoo copy `market.yahoo_daily_frozen_20261126` on a survivor day and count how many symbols each returns. Record whether either source returns a symbol that has delisted, and whether shares outstanding is available free at all.
- [ ] **Step 4:** Rerun the tests; PASS. Commit with `_source_pilot.json`.
- [ ] **Step 5 (owner gate):** Show the owner, in plain words, the coverage of each source and a recommendation. Ask which source, and what "eligible list" means going forward if shares outstanding is not free (proposal: rank by price times volume instead of company size, and say plainly that this is a different list from the first test's).

### Task 2: Storage schema and the write approval

**Files:**
- Create: `sql/forward_schema.sql`; test `tests/db/test_forward_schema.py` (against a disposable database container, the way `scripts/db_restore_drill.py` builds one)
- Modify: `docs/database-inventory-2026-11-26.md` (add the schema once created)

**Interfaces:**
- Produces tables `forward.loads(load_id, day, source, raw_sha256, normalized_sha256, code_commit, actor, created_at)`, `forward.listing_snapshots(load_id, day, symbol, exchange, is_etf, is_test)`, `forward.daily_bars(load_id, day, symbol, open, high, low, close, volume)`, `forward.gaps(load_id, day, symbol, reason)`, `forward.symbol_status(symbol, first_seen, last_seen, ended_on)`; append-only triggers on all of them.

- [ ] **Step 1: Write failing tests:** `test_update_is_refused`, `test_delete_is_refused`, `test_ddl_is_idempotent` (running the file twice succeeds), `test_bars_unique_per_day_symbol_load`.
- [ ] **Step 2:** Run them against the disposable container; expect FAIL.
- [ ] **Step 3:** Write the DDL with triggers for every role.
- [ ] **Step 4:** Rerun; PASS.
- [ ] **Step 5 (owner gate):** Ask for a yes to create the schema in the live database. Take a fresh dump first and state that plainly. Commit; apply only after the yes.

### Task 3: The daily collector

**Files:**
- Create: `scripts/forward_collect.py` (has `main()`, `argparse`, working `--help`, `--dry-run`), test `tests/scripts/test_forward_collect.py`

**Interfaces:**
- Consumes: the Task 1 source functions (copied into `scripts/` helpers, since scripts must not import a study's private code) and the Task 2 tables.
- Produces: `collect(day: date, dry_run: bool) -> dict` returning counts `listed, bars, gaps, newly_ended`; `update_symbol_status(conn, day: date) -> list[str]` returning symbols newly flagged `ended`.

- [ ] **Step 1: Write failing tests with an in-memory fake source and the disposable database:** `test_rerun_same_day_adds_no_rows`, `test_missing_symbol_is_recorded_as_gap`, `test_symbol_ended_only_after_five_missing_trading_days`, `test_holiday_makes_no_gaps`, `test_low_coverage_returns_alert_flag`.
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement `collect` and `update_symbol_status`; hash the raw files before parsing; skip non-trading days using the NYSE calendar already used by `research/stock-signals/early-decades/code`'s checks (copy the helper, do not import it).
- [ ] **Step 4:** Rerun; PASS; run `ruff check`.
- [ ] **Step 5:** Dry-run for five trading days with `--dry-run` writing nothing; show the owner the counts. Commit.

### Task 4: Timer, heartbeat and weekly health report

**Files:**
- Create: `scripts/forward_health.py`, user timer files `ggtrader-forward-collect.timer`/`.service` (install steps documented in `docs/cli_reference.md`), test `tests/scripts/test_forward_health.py`

**Interfaces:** Produces `weekly_report(conn, end_day: date) -> dict` with keys `days`, `mean_coverage`, `worst_day`, `gaps`, `ended`.

- [ ] **Step 1: Write failing tests:** `test_weekly_report_computes_mean_coverage`, `test_report_lists_ended_symbols`.
- [ ] **Step 2:** Run; expect FAIL; implement; rerun; PASS.
- [ ] **Step 3:** Write the timer to run on weekdays about 90 minutes after the close with a Telegram heartbeat, matching the existing weekly membership timer's pattern. Do not enable it.
- [ ] **Step 4 (owner gate):** Ask the owner to enable the timer. Record the start date; that date is the earliest any later rule's data can begin.
- [ ] **Step 5:** Commit; update `docs/next_steps.md`, `docs/cli_reference.md`, changelog.

### Task 5: Registration template for the later check

**Files:** `research/stock-signals/forward-check/2026-MM-DD-forward-check-registration-TEMPLATE.md`, test `code/tests/test_registration_template.py`

- [ ] **Step 1: Write `test_template_requires_pinned_dates_and_comparison`:** asserts the template has `eval_start`, `eval_end`, `family_size`, the cost and tax lines, the comparison with SPY on the same dates, and the statement that only data dated after the registration commit is scored.
- [ ] **Step 2:** Write the template; PASS; commit.
- [ ] **Step 3:** Record in `docs/next_steps.md`: no rule is registered, and none may be until one survives choices 1-4 and the collection has run long enough to cover the chosen scoring period.

## Self-Review

- Coverage: source choice (1), safe storage (2), collector with ended-stock handling (3), schedule and health (4), later registration rule (5).
- Open risks: free sources may not return delisted stocks or shares outstanding, so the first pilot decides how close the later list can get to the first test's; elapsed time, not code, is the real cost of this candidate.
