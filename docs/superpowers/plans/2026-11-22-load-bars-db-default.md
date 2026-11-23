# Change Zero: `load_bars` Reads the Database by Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `load_bars`, `load_run_bars` and `ggt lab` from defaulting to a frozen Parquet file that is not on disk; they read the database (Yahoo rows only) unless a snapshot is named, and every database run records a hash of the bars it received.

**Architecture:** `source` becomes `str | None` on `load_bars` and `load_run_bars`: `None` resolves to `"snapshot"` when a snapshot path is given and `"db"` otherwise; an explicit `source="db"` with a snapshot path is an error. A new `bars_sha256(bars)` fills the run's `tape_sha256`, so the existing provenance comparison (`TAPE_DIFF`) separates two database runs whose data changed. Nothing in the import is needed.

**Tech Stack:** Python 3.12, pandas, psycopg2, pytest (run only through `scripts/run_tests.sh`).

**Spec:** `docs/superpowers/specs/2026-11-22-kaggle-import-design.md`, sections 6.5 (run provenance) and 6.6 (three explicit sources; "change zero"). The `combined` source is added by the version-one plan (`2026-11-22-kaggle-import-version-one.md`), not here.

## Global Constraints

- The new default equals today's explicit `source="db"` on pinned inputs. It is **not** equal to the old default: the snapshot manifest ends 2026-10-30, the database reaches 2026-11-05 (spec 6.6).
- An explicit snapshot path selects the snapshot unless another source is also given explicitly; a conflicting source and path raises `ValueError`.
- Research code stays read-only; no network, no writes (`load_bars` opens a server-enforced read-only session). `tests/data/test_bars.py` scans `bars.py` for write verbs; keep it passing.
- Tests run only as `scripts/run_tests.sh <small targeted paths>`. Never run `tests/paper/`, a whole tier or an oracle driver in a task; the orchestrator does, one at a time. If a run prints "waiting for another heavy ggTrader job", wait. Never bypass the scripts.
- Ruff-clean (`ruff check .`, `ruff format .`), type hints, line length 100, module docstrings, no `timestamp without time zone`.
- The snapshot code, `data/snapshots/MANIFEST.json` and the oracle numbers are not edited.

## Review Focus

- `load_bars(["AAPL"], snapshot_path=p)` with no `source` must read the snapshot, not the database (an existing `--snapshot` run must not silently change source).
- `source="db", snapshot_path=p` must raise, with a message naming both.
- A database run before and after a Yahoo refresh must compare as `TAPE_DIFF`, not "same stack".
- The 23 `load_bars(` calls in `tests/data/test_bars.py` that stub the snapshot reader must not reach a database after the default flips.
- An empty result (no symbols found) must still hash without raising.

---

### Task 1: Source resolution in `load_bars`

**Files:**
- Modify: `src/ggTrader/data/bars.py` (`SOURCES`, `load_bars` signature and body near lines 252-318)
- Test: `tests/data/test_bars.py`

**Interfaces:**
- Produces: `resolve_source(source: str | None, snapshot_path: Any | None) -> str` (module-level, in `bars.py`); `load_bars(..., source: str | None = None, ...)`.

- [ ] **Step 1: Write failing tests** in `tests/data/test_bars.py` (the file already has a fixture that routes the snapshot reader to an in-memory frame, and a way to stub `_read_db_long`; reuse both):
  - `test_default_source_is_db`: stub `_read_db_long`, call `load_bars(["AAA"])`, assert the stub was called once and `bars.provenance["source"] == "db"`.
  - `test_snapshot_path_alone_selects_snapshot`: call `load_bars(["AAA"], snapshot_path=tmp)`, assert the snapshot reader was called and the DB stub was not.
  - `test_explicit_db_with_snapshot_path_raises`: `pytest.raises(ValueError, match="source.*snapshot_path")`.
  - `test_unknown_source_still_raises` (existing behaviour, `source="yfinance"`).
  - `test_resolve_source_table`: parametrized over `(None, None) -> "db"`, `(None, "x.parquet") -> "snapshot"`, `("snapshot", None) -> "snapshot"`, `("db", None) -> "db"`.
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/data/test_bars.py -k "source" -q`. Expected: FAIL (`resolve_source` missing).
- [ ] **Step 3: Implement** `resolve_source` and wire it as the first line of `load_bars`'s validation (the `SOURCES` check stays; keep `SOURCES = ("snapshot", "db")`). Update the docstring's `source` line.
- [ ] **Step 4: Run** the same command. Expected: PASS. The other tests in the file that call `load_bars` without a source now fail; Task 3 fixes them.
- [ ] **Step 5: Commit** `git commit -m "feat: load_bars resolves its source (db by default)"`.

### Task 2: Bars hash in provenance and `tape_sha256`

**Files:**
- Modify: `src/ggTrader/data/bars.py` (add `bars_sha256`; set `provenance["bars_sha256"]` for the `db` source), `src/ggTrader/lab/wfo/cli_support.py:~174` and `src/ggTrader/lab/blend.py:~122` (use the helper below)
- Test: `tests/data/test_bars.py`, `tests/lab/test_blend.py` or a new `tests/lab/wfo/test_tape_hash.py`

**Interfaces:**
- Produces: `bars_sha256(bars_frames: Mapping[str, pd.DataFrame]) -> str` (hex SHA-256 over every field frame, columns sorted, index and values as bytes, NaN as NaN); `tape_sha256_of(bars: Bars) -> str | None` returning the snapshot's `sha256` if present, else `bars.provenance.get("bars_sha256")`.
- Consumes: `Bars.provenance` (existing dict).

- [ ] **Step 1: Write failing tests:** `test_bars_sha256_is_stable_and_order_independent` (same frames, columns shuffled, same hash); `test_bars_sha256_changes_with_one_value`; `test_empty_bars_hash_does_not_raise`; `test_db_provenance_carries_bars_sha256`; `test_tape_sha256_of_prefers_snapshot_hash`; and a provenance test (`tests/engine/` has the comparison tests; copy the nearest) that two records differing only in `tape_sha256` compare as `TAPE_DIFF`, not comparable.
- [ ] **Step 2: Run** the new tests. Expected: FAIL.
- [ ] **Step 3: Implement** both helpers; call `bars_sha256` inside `load_bars` after the frames are built (database source only) and store it under `provenance["bars_sha256"]`. In `cli_support.py` and `blend.py`, replace `tape = (bars.provenance or {}).get("snapshot") or {}` / `tape.get("sha256")` with `tape_sha256_of(bars)`.
- [ ] **Step 4: Run** the tests; also `scripts/run_tests.sh tests/lab/test_blend.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: database runs record a hash of the bars they read"`.

### Task 3: `load_run_bars`, the lab CLI, and every caller

**Files:**
- Modify: `src/ggTrader/lab/wfo/cli_support.py` (`load_run_bars` default `source: str | None = None`; `run_vbt_wfo` and the blend entry point accept and pass `source`), `src/ggTrader/lab/blend.py` (`run_blend(...)` accepts `source: str | None = None` and passes it), `src/ggTrader/lab/cli.py` (new `--source {db,snapshot}` next to `--snapshot`, lines ~102-106; pass `source=args.source` at lines ~159 and ~181)
- Modify (explicit `source="snapshot"` where the call depends on the frozen tape or stubs the snapshot reader): every hit of `grep -rn "load_bars(\|load_run_bars(" --include=*.py .`. The 2026-11-22 list: `scripts/p5_wfo_validation.py`, `scripts/p7_live_golden.py`, `scripts/p3_pit_parity.py`; tests `tests/data/test_bars.py` (23 calls), `tests/lab/wfo/helpers.py`, `tests/lab/wfo/test_cli_engine.py`, `tests/lab/wfo/test_frozen_evidence_seams.py`, `tests/lab/test_cli.py`, `tests/lab/test_blend.py`, `tests/lab/test_universe.py`, `tests/paper/test_live_golden.py`; research drivers `research/programme/vectorbt-engine/p2b-evidence/window_convention_check.py`, `p4-evidence/chunker_experiment.py`, `p4-evidence/chunker_memory.py`, `p10-evidence/dsr_one_combo.py`, and `p5c-evidence/capture_legacy_replica_panel.py` and `compare_ported_replica_panel.py` (already explicit). The archived `legacy-scripts/p4_strategy_parity.py` and `p5_equity_parity.py` get a one-line docstring note: valid only against their original revision.
- Test: `tests/lab/test_cli.py` (argument parsing), `tests/lab/wfo/test_cli_support_source.py` (new)

**Interfaces:**
- Consumes: `resolve_source` (Task 1).
- Produces: `load_run_bars(universes, cfg, eval_start, eval_end, snapshot: str | None = None, source: str | None = None, *, membership=None) -> Bars`.

- [ ] **Step 1: Write failing tests:** `test_cli_source_flag_parses` (`--source db`, `--source snapshot`, default `None`); `test_load_run_bars_default_reads_db` (stub `load_bars`, assert it receives `source=None` and `snapshot_path=None`); `test_load_run_bars_snapshot_path_reaches_load_bars`; `test_cli_conflicting_source_and_snapshot_is_error` (`--source db --snapshot x` exits non-zero with the `ValueError` message).
- [ ] **Step 2: Run** `scripts/run_tests.sh tests/lab/test_cli.py tests/lab/wfo/test_cli_support_source.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the signature changes and the CLI flag. Then go through the caller list: for each hit decide `source="snapshot"` (anything that reads the frozen tape, stubs the snapshot reader, or is an oracle/golden check) or leave it (a test that stubs the database). Re-run the grep at the start; the plan lists every hit it finds, not only the list above.
- [ ] **Step 4: Run** `scripts/run_tests.sh tests/data/test_bars.py tests/lab/test_cli.py tests/lab/test_blend.py tests/lab/test_universe.py tests/lab/wfo -q`. Expected: PASS. (The orchestrator runs `tests/paper/test_live_golden.py` and the oracle drivers separately, one at a time.)
- [ ] **Step 5: Commit** `git commit -m "feat: ggt lab and load_run_bars default to the database; snapshot is explicit"`.

### Task 4: Pin test, documentation and the real-database check

**Files:**
- Modify: `AGENTS.md` ("Database map" load_bars paragraph: the snapshot is now opt-in; the default reads Yahoo rows from the database), `docs/cli_reference.md` (`--source`), `docs/changelog.md`
- Test: `tests/data/test_bars_default_pin.py` (new, marked `integration` and `db`, skipped without `GGT_TEST_DB_URL`)

- [ ] **Step 1: Write the pin test:** `test_new_default_equals_explicit_db`: on a scratch database seeded with a small `yfinance` 1d fixture (helpers in `tests/data/scratch_db.py`), `load_bars(["AAA", "BBB"], "2024-01-01", "2024-01-31")` and `load_bars(..., source="db")` return frames equal under `pd.testing.assert_frame_equal` and carry the same `bars_sha256`.
- [ ] **Step 2: Run** with `GGT_TEST_DB_URL` pointing at a scratch server (see `tests/data/test_universe_db_integration.py` for the pattern). Expected: PASS. Without the variable: SKIPPED.
- [ ] **Step 3: Update the three docs.** The AGENTS paragraph states: snapshot file absent on 2026-11-22; default is `db`; `--snapshot` or `source="snapshot"` selects the frozen tape; the default covers dates up to the latest Yahoo row, not 2026-10-30.
- [ ] **Step 4: Real-database check (read-only; the orchestrator runs it):** `python -c "from ggTrader.data.bars import load_bars; b = load_bars(['SPY','AAPL'], '2025-01-02', '2025-03-31'); print(b.provenance['bars_sha256'], len(b.close))"` twice; both hashes must be identical.
- [ ] **Step 5: Run** `scripts/run_tests.sh --fast`. Expected: PASS. Then the orchestrator runs `scripts/run_tests.sh --full` once, alone.
- [ ] **Step 6: Commit** `git commit -m "docs: load_bars default is the database; pin test"`.
