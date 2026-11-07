# P1 parity: old stack vs new stack, real data

Date 2026-11-06. Driver: `p1_parity_driver.py` (this folder). Raw results: `parity_results.json`,
`runs/*/` (per-run JSON), `raw_array_sha256.json` (sha256 of the full equity arrays kept in the
session scratchpad), `coldstart_results.jsonl` and `coldstart_summary.json`.

## Profile

- Window 2021-01-31 to 2022-07-31, 2 folds, `ensemble` 48-combo grid, plus the `xs_momentum` weight
  path (6 combos), 150 SP500 symbols (5 of them have price gaps inside the window) plus SPY.
- Data: one read-only extract from the production TimescaleDB (session option
  `default_transaction_read_only=on`, verified `SHOW transaction_read_only = on`) into
  `flat.parquet` (129,405 rows, sha256 `2bff21e4...`, see `extract_meta.json`). Every run then loads
  the same rows through the real `load_ohlcv` -> `CachedYFinanceLoader._db_fetch` pivot code with
  `pandas.read_sql` and the DB connection replaced, so no run touches the database. The
  P0 frozen tape did not exist yet (`data/snapshots/` empty at the start), so this extract is
  temporary and lives in the session scratchpad, not in the repo.
- Four runs: **A** original source (exported from `HEAD`) on the old stack (pandas 2.3.3, numpy 2.2.6,
  numba 0.61.2, vectorbt 0.28.5); **A2** the P1 source on the old stack; **B1** the P1 source on the new stack
  (pandas 3.1.1, numpy 2.5.4, numba 0.68.1, vectorbt 1.2.0) with `engine=numba`; **B2** same with `engine=rust`.
  A vs A2 isolates the code edits; A2 vs B isolates the stack.

## Result

| Quantity | A vs A2 (code only) | A vs B1 (numba) | A vs B2 (Rust) | B1 vs B2 |
|---|---|---|---|---|
| 48 signal equity curves | 48/48 identical | 48/48 identical | 48/48 identical | 48/48 identical |
| 6 weight-path curves | 6/6 | 6/6 | 6/6 | 6/6 |
| OOS equity (run_wfo) | identical | identical | identical | identical |
| PIT entry mask | identical | identical | identical | identical |
| Per-combo metrics (7 per combo) | 0 diffs | 0 diffs | 0 diffs | 0 diffs |
| Per-fold results, winners, gates, WFE | 0 diffs | 1 float: `dsr_value` fold 1 0.7190648267528605 vs ...604 (1.1e-16) | same 1 float | 0 diffs |
| Live params, printed table | identical | identical | identical | identical |

"Identical" means `np.array_equal` (max abs diff 0.0) on float64 arrays and equal nanosecond index.
The only change in the WFO path is the documented 1-ulp `dsr_value` from pandas 3 `skew()/kurtosis()`.
Winners, gates (all FAIL at the NDH/DSR gates in both folds, WFE 0.31) and OOS Sharpe 0.60 / CAGR 4.4% /
MaxDD -3.2% are unchanged. The OOS curve is not flat in this profile (unlike S3's 60-symbol one), so it is a
real check of trades, stops and fold stitching.

## pct_change-affected paths on real data

- A call-site probe on the old stack (A) recorded which `pct_change` calls saw non-leading NaN input:
  `event_study.py`, `idio_vol.py`, `max_effect.py`, `feature_gate.py` (9 of 18 calls) did; the equity-curve
  sites (`metrics.py`, `wfo.py`, `overlay.py`, `simulate.py`) never did. That is the evidence behind
  using `fill_method=None` on curves and `padded_pct_change` on price panels.
- Close panels, 507 symbols (2021-22) and 533 symbols (2025-26): pandas 2's padded `pct_change()` equals
  `padded_pct_change()` bit for bit on all four runs. A plain pandas-3 call would differ in 453 and 2,937 cells
  (5 and 20 symbols with non-leading gaps).
- Strategy and live-path probes (event study, `idio_vol`/`max_effect` selections at 4 as-of dates,
  `extract_features` on 12 symbols x 2 dates, `compute_weights_and_scale`, `curve_stats`): A equals A2 exactly.
  A vs B differs only by float noise or tie order, none of it in the ensemble or WFO path:
  - `event_study` statistics differ by 2e-19 to 9e-16. Cause isolated: returns and masks are bit-identical on both
    stacks; the per-date `DataFrame.mean(axis=1)` row reduction differs in the last bit on pandas 3.
  - `idio_vol.select` at 2022-07-29 returns the same 30 symbols with equal weights but lists three delisted,
    padded-to-zero-variance names (DISCK, PBCT, XLNX) in a different order: a tie broken by 1e-19 noise.
    That is a pre-existing artefact of padding delisted symbols to zero returns, not a regression.

## Timing (indicative only: box load ~4 on 4 cores, runs overlapped with the test suite)

| Run | PIT mask | 48-combo sweep (incl. JIT) | 6-combo weight sweep | `run_wfo` 2 folds | peak RSS |
|---|---|---|---|---|---|
| A old stack | 50.9 s | 16.1 s | 27.8 s | 109.8 s | 1.67 GB |
| A2 old stack, P1 code | 67.2 s | 14.1 s | 29.9 s | 110.3 s | 1.67 GB |
| B1 new, numba | 59.6 s | 22.0 s | 29.3 s | 104.2 s | 1.63 GB |
| B2 new, Rust | 45.4 s | 8.1 s | 12.5 s | 77.4 s | 2.21 GB |

Block consolidation micro-benchmark (RSI-style chain, 150 columns, pandas 3.1.1): 120.8 ms on the raw concat,
18.1 ms after `.copy()`; pandas 2.3.3: 13.3 ms. Rust uses about 0.5 GB more peak RSS here.

## Cold start (fresh process, import + first `simulate_signals` with ATR stops + first `simulate_weights`)

Median of 3 runs each (`coldstart_summary.json`). Rust never uses the numba cache.

| Stack | cold (empty cache) | warm cache |
|---|---|---|
| New, Rust | 4.0 s (3.9 to 4.2) | 3.9 s |
| New, numba | 29.2 s (29.2 to 30.4) | 8.8 s |
| Old, numba (production today) | 22.0 s (21.9 to 22.4) | 8.0 s |

## Limits

- The DB read path is not part of this comparison (the parquet extract replaces it); S3 compared it separately
  (7 symbols x 250 bars bit-identical) and the DB write path was not exercised. Index resolution after the
  parquet round trip is `datetime64[us, UTC]` on both stacks, so the ns/us/s resolution change is not covered.
- Timings are from an overlapped, shared box; only the ratios and the cold-start rows (3 repeats, tight spread) carry weight.
- No 17-fold run (P0 owns the 17-fold legacy reference; compare against it once P0 delivers).
