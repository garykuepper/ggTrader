# P1 17-fold parity: legacy walk-forward code on the new stack vs the P0 legacy reference

Date 2026-11-06. Closes the P1 exit criterion left open in `P1-completion.md` ("17-fold: not run").
Raw results: [`p1-17fold-parity.json`](p1-17fold-parity.json) (every difference, equity hashes, engine proof,
timings) and `runs17fold/{numba,rust}/` (per-run JSON and OOS equity CSV). Runner
`scripts/new_stack_reference_runs.py` (reuses `run_wfo_case` of `scripts/legacy_reference_runs.py`
unchanged), comparer `scripts/new_stack_parity_compare.py`.

Terms: **WFO** = walk-forward optimization. **OOS** = out-of-sample. **DSR** = deflated Sharpe ratio gate.
**NDH** = plateau-density gate. **Anchor** = the defensive fallback parameter set deployed when a fold's gates fail.

## What ran

- Profile: the P0 references `core17_ensemble_sp500` (default gates; anchor in 17/17 folds) and
  `core17_ensemble_sp500_dsr0` (DSR threshold 0: winner-deployment path, 0 anchor folds). `ensemble`, sp500 PIT
  universe, eval 2021-01-31 to 2026-04-30 (17 folds, 48-combo grid), `STOCK_BASE_CONFIG`, max leverage 1.0.
- Tape: frozen `data/snapshots/ohlcv_20261106.parquet` (hash-checked by the runner against `MANIFEST.json`); database
  and network blocked as in P0. `sp500_constituents_history.csv.gz` sha256 `b30739a3...` equals the file at the legacy tag.
- Code: a copy of `src/ggTrader` taken 13:53 PT after `git diff --stat c08d4ebb` for `lab/{wfo,sweep,simulate,metrics,gates,harness}.py`
  showed only the intended P1 edits (`pct_change(fill_method=None)` in `wfo.py` x1 and `metrics.py` x3). Other P1 edits in
  the copy: `"1D"` freq strings, `.copy()` after concat in `indicators.py`. Run with `PYTHONPATH` on the copy, so concurrent edits
  could not reach it. SHA list of the copy: scratchpad `snap/src_sha.txt`.
- Stack: Python 3.12.3, pandas 3.1.1, numpy 2.5.4, numba 0.68.1, vectorbt 1.2.0 (+ vectorbt-rust 1.2.0). P0 side:
  pandas 2.3.3, numpy 2.2.6, numba 0.61.2, vectorbt 0.28.5.
- Engine set with `vbt.settings["engine"] = "numba" | "rust"` (default is `auto`, which would pick Rust). Proof is in the JSON
  (`engine_proof`): all 676 engine resolutions returned the requested engine; Rust kernel calls (wrapped `*_rs` functions):
  numba runs 0, Rust runs 676 (52 `simulate_from_signals_rs`, 104 `col_map_rs`, 52 each of 10 other kernels).
- Same P0 memory hygiene (inert `gc.collect` + `malloc_trim`), 6 GB cgroup, `MemorySwapMax=0`; none was OOM-killed.

## Result: no difference beyond `dsr_value`

Comparison covers the whole result JSON (666 numeric/string leaves per run: per-fold train and test windows, winner combo and params, scores,
NDH/DSR/gate verdicts, NDH density, anchor use, WFE, in- and out-of-sample Sharpe, halts, live params, printed table, OOS
Sharpe/CAGR/MaxDD/Sortino/vol, SPY stats, bar counts and first/last dates) plus the full OOS equity CSVs (1,065 bars).

| Comparison | OOS equity (CSV text, `np.array_equal`, sha256) | JSON differences | Class |
|---|---|---|---|
| core17: P0 legacy vs new numba | identical, max abs diff 0.0, sha256 `978ef3a9...` | 7, all `dsr_value`, max 1.7e-16 | bit-identical except `dsr_value` last digits |
| core17: P0 legacy vs new Rust | identical, 0.0 | the same 7 | same |
| core17: new numba vs new Rust | identical, 0.0 | 0 | bit-identical |
| dsr0: P0 legacy vs new numba | identical, 0.0, sha256 `4814c730...` | 7, all `dsr_value`, max 1.7e-16 | bit-identical except `dsr_value` last digits |
| dsr0: P0 legacy vs new Rust | identical, 0.0 | the same 7 | same |
| dsr0: new numba vs new Rust | identical, 0.0 | 0 | bit-identical |

Headline numbers are unchanged: core17 OOS Sharpe 0.3905, CAGR 3.49%, MaxDD -9.58%, gates 1/17, anchor 17/17;
dsr0 Sharpe 0.6224, CAGR 3.31%, MaxDD -5.28%, gates 17/17, anchor 0/17. Winners, test windows, gate verdicts and anchor use are
identical in all 17 folds of both variants. The only input-description difference is `base_config.FREQ` `"1d"` vs `"1D"` (the intended P1 edit).

`dsr_value` differences (folds 5, 6, 8, 10, 13, 15, 16 in 1-based numbering; the same folds in both variants and both engines): 1, 1, 2, 2, 3, 4
and 6 ulps (absolute 2.8e-17 to 1.7e-16, relative at most 1.2e-15). The documented P1 cause is pandas 3 `skew()/kurtosis()`
last-bit change feeding the DSR formula; the pass/fail flags derived from it never flipped. Note the 2-fold note called this
"1 ulp": here it reaches 6 ulps, which is still last-digit noise but is wider than "1-ulp" literally.

## Not covered / limits

- Trade counts per fold are not stored in the P0 references (`fold_table` keeps no trade fields), so they are not compared directly;
  bit-identical OOS equity over 1,065 bars (including the stitched fold boundaries and stops) is the indirect evidence.
- Per-combo sweep metrics are not in the reference JSON, so they are covered only through winners, scores and gates.
- The DB read path is not exercised (frozen tape, as in P0).
- Runs are monolithic (one `run_wfo` call, about 22 to 25 minutes); resume works at run level (an `ok` JSON is skipped), not inside a run.

## Timings and memory (shared 4-core box, indicative)

| Run | Wall total | `run_wfo` incl. load | Peak RSS (runner / `time -v`) |
|---|---|---|---|
| P0 legacy core17 (old stack) | 1,421 s | 1,421 s | 4,971 MB |
| P0 legacy dsr0 (old stack) | 1,363 s | 1,363 s | 4,968 MB |
| new numba core17 | 1,334 s | see JSON | 5,454 MB / 5.58 GB |
| new numba dsr0 | 1,413 s | see JSON | 5,466 MB / 5.54 GB |
| new Rust core17 | 1,388 s | see JSON | 5,413 MB / 5.54 GB |
| new Rust dsr0 | 1,528 s | see JSON | 5,414 MB / 5.60 GB |

Peak RSS sits within about 0.4 GB of the 6 GB cap, so a smaller cap would fail. Rust shows no wall-time gain on this workload (the
fold sweeps are dominated by the Python/pandas signal generation and by the other agent's load on the box), unlike the 2-fold
run where Rust was faster; treat the timings as noisy.

## Verdict

The P1 exit criterion "17-fold bit-for-bit except documented DSR last digits" is met for both references and both engines. Nothing
beyond `dsr_value` differs; numba and Rust are bit-identical to each other. The one caveat is that the `dsr_value` noise spans up to 6 ulps
rather than 1.
