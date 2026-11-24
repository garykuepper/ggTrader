# Task 5 report: scale probe and batched close reader

Status: DONE_WITH_CONCERNS (real probe not run; orchestrator runs it).

RED: both new test files failed at collection (`No module named 'bars_reader'`).
GREEN: 11 passed (`scripts/run_tests.sh .../test_scale_probe.py .../test_bars_reader.py -q`).
Ruff check and format clean; `scripts/study.py check` OK; `scale_probe.py --help` works.

Files (all under research/stock-signals/dead-stock-first-test/code/):
- bars_reader.py: `read_release_spans`, `symbol_batches`, `read_close_batched` (+ `SPAN_COLUMNS`).
- scale_probe.py: one measurement per invocation; writes a "started" record first, then a "finished" record.
- tests/test_bars_reader.py, tests/test_scale_probe.py.

Command for the orchestrator (one run per measurement, each through run_capped.sh):
`scripts/run_capped.sh .venv/bin/python research/stock-signals/dead-stock-first-test/code/scale_probe.py --n 500 --mode batched --out results/dead_stock_first_test/scale_probe_500.json`
(same with `--n 1500`, and `--n all --mode one_call` then `--n all --mode batched`).
The plan's single `scale_probe.json` is replaced by one file per measurement (per ruling 2).

Concerns / decisions:
1. `market.stock_spans` has no `has_successor` or `total_loss` columns (load_bars derives them), so `read_release_spans` returns exactly symbol, permno, spell, first_date, last_date, kind, possible_double.
2. `Bars.spans` holds only ENDED series (and only symbol, last_date, has_successor, total_loss), so the spans returned by `read_close_batched` are those; eligibility must use `read_release_spans`.
3. Extra optional arg `provenance_out` on `read_close_batched` (fills data_release from the first batch, bars_sha256 per batch, n_batches); the tuple return is as ruled.
4. one_call mode = one batch of all selected columns. `peak_memory_mb` is the whole-process VmHWM, so it includes the picking step (release spans, month-end fields, eligibility over all ~29k columns); that step's own peak is recorded separately as `setup_peak_memory_mb`.
5. Month-end fields are read from one month before `--start` because eligibility uses the previous month-end.
6. If the process is killed by the cap, the JSON stays as `{"status": "started", ...}`.

## Fix round 1 (Codex Sol review of 67836941)

Changes (scale_probe.py):
1. The `{"status": "started", ...}` marker is now written right after argument parsing, before any
   database read or eligibility work, and rewritten with `n_columns` once the columns are picked.
   Any exception after that writes `{"status": "failed", "error_type", "error_message", ...}` and
   re-raises, so an old `finished` file can never look current. Setup moved into `pick_columns(args)`.
2. `peak_memory_mb()` raises `PeakMemoryUnavailable` when the status file is unreadable or has no
   `VmHWM`. `main` catches it through `safe_peak_mb()`: the record then has `peak_memory_mb: null`,
   `ok: false` and a `measurement_error` string (same for the setup peak via
   `setup_peak_memory_mb: null` and `setup_measurement_error`). `build_record` never reports ok for a
   missing peak.
3. `crsp_load` is recorded in the started, finished and failed records beside `release_id`.
CLI and the existing JSON field names are unchanged; fields were only added.

Covering tests (test_scale_probe.py): `test_peak_memory_unavailable_is_an_error_not_zero` (replaces
the zero test), `test_missing_peak_is_never_ok`, `test_setup_failure_replaces_old_finished_file`
(pre-existing finished file is `started` during setup, `failed` with error type and message after),
`test_main_writes_finished_record_with_crsp_load`, `test_main_without_peak_data_is_not_ok`.

Command: `scripts/run_tests.sh .../test_scale_probe.py .../test_bars_reader.py -q`
Output: `15 passed in 0.23s` (waited for the orchestrator's heavy job lock first). Ruff check and
format clean; `scripts/study.py check` prints `OK research/stock-signals/dead-stock-first-test`.
