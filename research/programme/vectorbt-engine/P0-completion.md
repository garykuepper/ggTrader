# P0 completion note: save the research, freeze the tape, record the legacy references

**Date:** 2026-11-06. **Phase:** P0 of the [vectorbt engine rebuild](../../../docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md) (section 9).
**Branch:** `vbt12/engine` (nothing committed, tagged or pushed by this phase). **Legacy tag:** `legacy-engine-vbt0285` = `c08d4ebb68dc24cdf12e9333c146f0f73c980684`.
**Status:** all P0 artefacts exist. The historical 17-fold figure does **not** reproduce; a fresh legacy baseline replaces it (spec Appendix D, question 8: not a stop condition). The difference is attributed below.

Terms: **WFO** = walk-forward optimization. **OOS** = out-of-sample. **PIT** = point-in-time. **NDH** = the plateau-density gate. **DSR** = deflated Sharpe ratio gate. **Anchor** = the defensive fallback parameter set deployed when a fold's gates fail. **Tape** = the daily bars table.

## 1. What was saved

| Item | Where | Notes |
|---|---|---|
| Research reports A, B, C, D, F and the S1 to S8 spec-phase reports, the section 8/9 draft | [`reports/`](reports/) | dated names, original names in the [README](README.md) |
| Evidence scripts and key outputs (B-bench, D-examples, F-checks, S2 prototypes, S3, S5 signal-diff arms, S6 incl. `pit.py` and the `vbt.Data` prototype `tsdata.py`, S8 tool tests, lead checks) | [`evidence/`](evidence/) | scripts keep their scratchpad paths: records, not runnable tools |
| Profiling of the legacy 2-fold WFO | [`profiling/`](profiling/) | `prof_wfo.py`, `timing.json` (load 22.5 s, WFO 439.9 s, 48 combos), `run.log`, `wfo.prof` (1.99 MB, kept) |
| Index of every file, one line each | [`README.md`](README.md) | generated from each file's docstring or first line |
| Legacy environment | `requirements-legacy.lock`, `scripts/legacy_env.sh` | see section 2 |
| Frozen tape | `data/snapshots/ohlcv_20261106.parquet` (git-ignored), `data/snapshots/MANIFEST.json` (tracked), `scripts/make_tape_snapshot.py` | see section 3 |
| Legacy reference runs | [`legacy-references/`](legacy-references/), `scripts/legacy_reference_runs.py`, `scripts/legacy_reference_compare.py` | see section 4 |
| Index row | `research/RESEARCH_INDEX.md` (study-dossier table) | one row added |

Files saved by P0 under this folder: 15 reports, 418 evidence files (28 MB), 4 profiling files (2.0 MB, includes the 1.99 MB `wfo.prof`), 36 legacy-reference files (1.9 MB), plus this note and the README, all indexed one line each in the README. Hash of the tracked inputs at completion:

| File | sha256 |
|---|---|
| `requirements-legacy.lock` | `a17ab713fb8d053f8700528d359c2933e065a21072a34f562aaeef5b86ae9df0` |
| `data/snapshots/MANIFEST.json` | `fd7b9b4797d4b079a3c6b3f844b8b482cc9076b02bf88962db23c8e821faa4cb` (changes if the snapshot is rebuilt) |

## 2. Legacy environment

- `requirements-legacy.lock`: exact `pip freeze` of `/home/flynn/ggTrader/.venv` (Python 3.12.3, vectorbt 0.28.5, pandas 2.3.3, numpy 2.2.6, numba 0.61.2), 171 pins, editable `ggTrader` line removed.
- `scripts/legacy_env.sh [--help] [TARGET_DIR]`: `git archive legacy-engine-vbt0285` into `TARGET_DIR/src`, a fresh venv from the lock in `TARGET_DIR/venv`, then an import check of `ggTrader.lab.simulate`. No `git worktree` is used; ggTrader is put on `PYTHONPATH`, not pip-installed.
- **Verified:** built in the scratchpad in 86.7 s; output `legacy env ok: 0.28.5 2.3.3 2.2.6 ggTrader.lab.simulate`; the new venv's `pip freeze` is identical to the lock (diff empty). All reference runs below used that rebuilt environment, not the production venv.

## 3. Frozen tape

`scripts/make_tape_snapshot.py` reads the `ohlcv` table with plain SQL on a server-enforced read-only session (`default_transaction_read_only=on`, checked before use), venue `yfinance`, interval `1d`. No loader, no network, no DDL, no writes.

| Field | Value |
|---|---|
| File | `data/snapshots/ohlcv_20261106.parquet` (zstd, mode 0444, git-ignored) |
| sha256 | `6573e84065d54796a6240051c3d83a68b32a079241665fb44001ec0cbc789b2e` |
| content sha256 (writer-independent) | `a1dc6b58181ec9a23521e6edd167c3e46c28c61d20e9ed2db713ae7c6c3fd70c` |
| Rows / symbols / trading days | 1,110,874 / 605 / 1,929 |
| Dates | 2019-03-01 to 2026-10-30 |
| Request | SP500 PIT union for 2021-01-31 to 2026-04-30 (612 symbols, via `equity_universe_between`) plus SPY, TLT, GLD, PDBC, BIL = 617 requested |
| Not in the database (12) | ABC, ANTM, BLL, COG, FBHS, FRC, GPS, HFC, MMC, PKI, RE, WLTW (delisted or renamed tickers; absent from the table, so the old loader would have tried to fetch them from Yahoo and Tiingo) |
| Conventions | naive-UTC midnight `timestamp`; the reader localises to UTC, pivots exactly as `CachedYFinanceLoader._db_fetch` does, sorts fields close/high/low/open/volume, applies `collapse_daily_duplicates` (it dropped **0** rows: the tape has 0 duplicate dates, 0 non-midnight stamps, 0 weekend stamps, 0 null closes) |
| Boundary rule | the database compares naive stamps with tz-aware bounds in the session zone (America/Los_Angeles here), so a request ending 2026-04-30 00:00 UTC excludes the 2026-04-30 bar. `load_snapshot_wide` reproduces this from the manifest field `db_session_timezone` |
| Verification | (a) 60 random symbols (112,585 rows) re-read from the database and compared bit for bit: identical; (b) `load_snapshot_wide` compared with the real `CachedYFinanceLoader._db_fetch` + `collapse_daily_duplicates` for 12 requested symbols (11 present, one deliberately absent) over two windows (to 2026-04-30 and to 2022-07-31) with `assert_frame_equal(check_exact=True)`: identical, including the end-bar exclusion (an open-ended third window differed only by database bars after 2026-10-30, as expected, so it was not a valid comparison); (c) SPY OOS Sharpe of the fresh 17-fold run equals the 2026-09-23 record to 16 digits (0.7792168856067855) |
| Tape sanity flags (manifest `sanity`) | jumps over 10x: PARA (1, 2021-02-12), SBNY (1, 2024-08-15), SIVB (5, from 2023-03-28); identical-close runs of 20 or more bars: ANSS 76, HBI 76, SIVB 73, DISCA 26, DISCK 25. These are delisted or acquired names; the reference runs include them exactly as the old pipeline did |
| Last bar coverage | 554 of 605 symbols on 2026-10-30 (no ragged final bar in this window) |

To rebuild: `PYTHONPATH=<checkout with .env>/src python scripts/make_tape_snapshot.py --tag 20261106` (needs database access; `--verify-only <file>` re-checks an existing file). The recorded `creation_command` omits the `PYTHONPATH` prefix. The database credentials come from the production checkout's `.env` read by `get_db_connection_string()` inside the code; they were never printed.

## 4. Legacy reference results (frozen tape, legacy code at the tag, vectorbt 0.28.5 / pandas 2.3.3)

Runner: `scripts/legacy_reference_runs.py` (resumable per run: an existing `"ok": true` JSON is skipped unless `--force`). Each run writes `legacy-references/_legacy_reference_<run>.json` with strategy, window, config, versions, tape sha, wall seconds, fold table, OOS Sharpe/CAGR/MaxDD against SPY, gates and winners per fold, plus `<run>_oos_equity.csv`. Database and network access are blocked in the process (`psycopg2.connect` and the Yahoo loader raise).

| Run | Window (eval) | Folds | OOS Sharpe | CAGR % | MaxDD % | SPY Sharpe (same OOS days) | Gates | Anchor folds | Wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `smoke2_ensemble_sp500` | 2021-01-31 to 2022-07-31 | 2 | 0.62 | 7.54 | -6.98 | -0.52 | 0/2 | 2 | 363 |
| **`core17_ensemble_sp500`** | 2021-01-31 to 2026-04-30 | 17 | **0.39** | **3.49** | **-9.58** | 0.78 | **1/17** | 17 | 1,421 |
| `core17_ensemble_sp500_dsr0` (winner-path variant) | same | 17 | 0.62 | 3.31 | -5.28 | 0.78 | 17/17 | 0 | 1,363 |
| `ref_cross_asset_sleeve` | 2021-01-31 to 2022-07-31 | 2 | -0.59 | -13.32 | -16.85 | -0.52 | 0/2 | 2 | 17 |
| `ref_cross_asset_sleeve_17fold` | 2021-01-31 to 2026-04-30 | 17 | 0.93 | 13.09 | -18.54 | 0.78 | 4/17 | 15 | 21 |
| `ref_bb_reversion` | 2-fold | 2 | 1.02 | 12.02 | -4.89 | -0.52 | 0/2 | 2 | 185 |
| `ref_rsi_reversion` | 2-fold | 2 | 0.33 | 4.42 | -7.14 | -0.52 | 0/2 | 2 | 185 |
| `ref_ema_cross` | 2-fold | 2 | -0.60 | -10.95 | -14.83 | -0.52 | 0/2 | 2 | 167 |
| `ref_macd_divergence` | 2-fold | 2 | -0.29 | -8.33 | -13.90 | -0.52 | 0/2 | 2 | 172 |
| `ref_volume_bb_reversion` | 2-fold | 2 | 0.31 | 2.96 | -6.39 | -0.52 | 0/2 | 2 | 175 |
| `ref_xs_momentum` | 2-fold | 2 | -0.62 | -18.42 | -20.79 | -0.52 | 0/2 | 2 | 182 |
| `ref_conviction_bb` | 2-fold | 2 | 0.29 | 2.82 | -6.68 | -0.52 | 0/2 | 2 | 326 |
| `ref_retail_attention` | 2-fold | 2 | n/a (flat) | 0.00 | 0.00 | -0.52 | 0/2 | 2 | 259 |

JSON file for each run: `legacy-references/_legacy_reference_<run>.json`. Wall times are with two to three jobs sharing 4 cores and another agent's load (load average 4 to 7), so they are not benchmark timings; the quiet-machine figures are in `profiling/timing.json` (WFO 439.9 s for the 2-fold ensemble, which matches 363 s here within load noise).

**SPY buy-and-hold (JSON `_legacy_reference_spy_bh_*`):**

| Window | Bars | Sharpe | CAGR % | MaxDD % |
|---|---:|---:|---:|---:|
| 17-fold eval window, first to last bar (2021-02-01 to 2026-04-29) | 1,317 | 0.886 | 14.49 | -24.50 |
| 17-fold OOS days only (the 1,065 days the WFO equity covers; SPY column above) | 1,065 | 0.779 | 12.96 | -22.09 |
| Smoke eval window (2021-02-01 to 2022-07-29) | 377 | 0.501 | 7.79 | -23.01 |
| Smoke OOS days only | 125 | -0.515 | -15.17 | -20.56 |

(Both smoke-window folds and the smoke-day SPY window are short and bear-market dominated, so the smoke SPY Sharpe is negative.)

### 4.1 The historical 17-fold figure does not reproduce

Spec section 7.3 legacy values: Sharpe 0.592, CAGR 3.08%, MaxDD -5.28%, SPY 0.779, gates 17/17 (`research/audits/rebaseline/_pit_rebaseline_20260923.json`, key `ensemble`). Fresh frozen-tape legacy run: **Sharpe 0.391, CAGR 3.49%, MaxDD -9.58%, SPY 0.779, gates 1/17**. Comparison file: `legacy-references/_legacy_reference_core17_vs_20260923.json` (made by `scripts/legacy_reference_compare.py`, fold by fold).

| Check | Result |
|---|---|
| SPY OOS Sharpe | identical to 16 digits (same OOS days, same tape for SPY) |
| Training winner per fold | same combo in 15 of 17 folds (differs in folds 15 and 16) |
| NDH gate per fold | same in 17 of 17 |
| DSR gate per fold | same in **1** of 17 (old: pass 17/17; now: fail 16/17) |
| Anchor deployed | old 0 of 17 folds; now 17 of 17 |

Attribution: commit `09f9f401` (2026-10-19, after the 2026-09-23 record) fixed the DSR gate, which "used annualized Sharpe against per-day thresholds"; the research index already warns that earlier DSR gate counts are not valid evidence. With the corrected gate the ensemble fails DSR in 16 of 17 folds, so the anchor set is deployed in all 17 folds and the OOS curve is the anchor's. This was **tested**: the same code and tape with the DSR threshold set to 0 (so DSR always passes, `core17_ensemble_sp500_dsr0`) gives gates 17/17, no anchor, **Sharpe 0.622, CAGR 3.31%, MaxDD -5.278%** against the 2026-09-23 record's 0.592, 3.08%, -5.278% (MaxDD equal to 6e-7). Fifteen of the 17 folds match the old record in winner, gates and OOS Sharpe to the printed digits (comparison file `_legacy_reference_core17_dsr0_vs_20260923.json`; JSON `null` vs `NaN` in folds 6 to 8 is the same missing value). The two folds that differ, 15 and 16 (test windows starting 2025-07-31 and 2025-10-31), have a different training winner and OOS Sharpe 2.31 vs 1.72 and 0.69 vs 3.45. Cause not established; a plausible one is tape repairs made after 2026-09-23 (the APH/MNST tape fix of 2026-09-29), which falls inside those training windows; the 12 symbols absent from the database are another candidate. This is a hypothesis, not tested.

So the old figure reproduces to within two folds once the DSR fix is undone, and the fresh run is the old figure plus the documented DSR correction plus a small tape drift.

**Working baseline for later phases (E0, legacy):** the frozen-tape row above, JSON `_legacy_reference_core17_ensemble_sp500.json`. It is anchor-dominated, so it hardly exercises the winner-deployment path; `core17_ensemble_sp500_dsr0` (DSR threshold 0, diagnostic only, not a production configuration) is recorded to cover that path and is the closest reproduction of the old 0.592 / 17-of-17 record. Which of the two is the E0 comparator for P1 and P5 is for the lead to decide; P1's bit-for-bit test should run both. Neither run is a strategy claim; Sharpe 0.62 in the dsr0 variant is still below SPY's 0.78 on the same days; the verdict versus SPY (0.39 vs 0.78 Sharpe) is unchanged from the NO-GO already on record.

### 4.2 Signal golden (task 4d)

`legacy-references/signal_golden_legacy.npz` (sha256 `6389ab4d3d23192bce4265f7dc9ce3c85764c3d5501fd30104baf8048ea3101b`) with metadata in `_legacy_reference_golden_ensemble_signals.json`: entry and exit boolean frames for the five voters (bb, rsi, ema, macd, vbb at the deployed parameters bb 20/2.0, rsi 14/30/50, ema 20/50, macd 12/26/9 with divergence 20, vbb vol 20 x 2.0) and for `EnsembleSignal(LabConfig())` defaults (min_agree 2, min_agree_exit 2), on the first 150 symbols (alphabetical: A, AAL, AAP, ...) over all 1,929 bars of the snapshot, with per-array packed-bit sha256 values and signal counts (ensemble: 4,673 entries, 24,322 exits). The golden was produced twice and the 12 array hashes matched. The bb and vbb exits hash identically (both are the close crossing back above the middle band), which is expected, not a bug.

### 4.2a Live golden (spec 7.3 item 4)

`legacy-references/live_golden_legacy.json` (1.2 MB, sha256 `6fea09e02c7185431af6fe1703c9f3676fd226e6a1a3671442ebd58fd2fa63c1`; metadata `_legacy_reference_golden_live_signals.json`): for each of the last 120 sessions of the snapshot (2026-05-12 to 2026-10-30), what `paper/signal_runner.generate_signals` computes with live defaults: PIT SP500 members of that day, bars from 365 calendar days earlier through that day (the last bar is the session itself; the live code's end bound excludes the same-day bar, so live runs see the previous session as last bar), `LabConfig(min_history_bars=60)`, default `EnsembleSignal`; per session `buys`, `sells` and `last_exit` (symbol to last exit date, from the whole window, which feeds `_missed_exits`). The ML gate is off by default and skipped. Totals: 1,028 buys and 5,519 sells over 120 sessions, 500 universe symbols in the last session, 100 symbol-sessions with 60 to 199 bars of history (the short-history case the spec asks to include). Run time 135 s.

### 4.3 Per-strategy references (task 4e)

One pinned window each: the 2-fold window 2021-01-31 to 2022-07-31 (the smallest multi-fold window; a fold needs 12 months of training and 3 of test, and 2 folds is the smallest window that also exercises the expanding anchor). All finished in 17 to 326 s of the 15-minute allowance. Notes:

- **Gates:** every 2-fold run fails its gates in both folds and deploys the anchor, so these references pin the sweep, ranking, anchor and the anchor's out-of-sample curve, not winner deployment. The same holds for `core17_ensemble_sp500` (anchor 17/17). Use the `_dsr0` variant for the winner path.
- **`cross_asset_sleeve`:** its universe is the five-ETF list (SPY, TLT, GLD, PDBC, BIL), not the SP500. The cheap 17-fold run is also recorded; the 2026-09-28 research recorded a different pinned window (2016 start), so the numbers are not comparable with it.
- **`retail_attention`:** the database table `google_trends_interest` exists but holds **0 rows** (checked read-only), so the strategy never selects anything and the book is flat cash (Sharpe undefined, CAGR 0). The runner supplies an empty frame (a read-only session would refuse the strategy's `CREATE TABLE IF NOT EXISTS` anyway). It needed the joblib threading backend so that the in-process patch reaches the workers. This reference only proves the pipeline runs with no data.
- **`conviction_bb`, `xs_momentum`, voters:** ran on the full PIT universe (519 loaded symbols for the 2-fold window).
- **Voters not run on their own as 17-fold:** not required by the spec.

## 5. Problems, findings and deviations

1. **Out-of-memory at the 6 GB cap, first 17-fold attempt (finding).** `core17_ensemble_sp500` was killed by the OOM killer at fold 7 after 14.5 min under `MemoryMax=6G`, `MemorySwapMax=0` (note: `legacy-references/_legacy_reference_core17_attempt1_oom6g_note.json`, log beside it). The legacy walk-forward keeps vectorbt portfolio objects alive between steps. The runner now calls `gc.collect()` and `malloc_trim(0)` after each anchor and sweep step (numerically inert: no value is touched); the second attempt finished in the same 6 GB cap with steady resident memory about 0.7 GB and a **peak of 4.97 GB** (`peak_rss_mb` in the JSON, with a per-step `rss_trace`). The cap was not raised. Two reading notes: (a) the 4.97 GB peak is the legacy comparator for the P5 criterion "peak RSS under 6 GB"; (b) without the hygiene hook the legacy path does not fit in 6 GB. Peak RSS was recorded only for the later runs (macd 969 MB, volume_bb 1,110 MB, xs_momentum 807 MB, conviction_bb 2,304 MB, retail_attention 754 MB).
2. **Weight-strategy runs use joblib worker processes**, so the runner's in-process patches (database block, empty interest frame) do not reach them; this only matters for `retail_attention`, where the threading backend is forced. The database block is belt-and-braces: the runner's environment has no database credentials at all (the credentials are only visible to the snapshot script), and no run touched the database.
3. **Runner status line bug:** the chain wrapper scripts in the scratchpad logged `exit=0` regardless of outcome (they read `$?` after a `date` call); the per-run JSON `ok` field is authoritative. The OOM run left no JSON; the note above covers it.
4. **`ref_retail_attention` first failed** (`ValueError: Database configuration not found`, caused by the unpatched worker loading the real data function) and was re-run after the fix; the failure JSON was overwritten.
5. **Concurrent changes in the worktree:** while P0 ran, another process (presumably P1) modified `pyproject.toml`, `Dockerfile`, CI, several `src/` and `scripts/` files, and added `uv.lock`, `src/ggTrader/utils/returns.py`, tests and `.venv` in the worktree. P0 did not touch these. All reference runs used the **extracted tag** (`git archive`) and the legacy venv, never the worktree's `src/`, so they are unaffected. P0's own changes: `.gitignore` (one rule for `data/snapshots/*.parquet`), `research/RESEARCH_INDEX.md` (one row), `requirements-legacy.lock`, `scripts/legacy_env.sh`, `scripts/make_tape_snapshot.py`, `scripts/legacy_reference_runs.py`, `scripts/legacy_reference_compare.py`, `data/snapshots/`, and this folder. Note that `make_tape_snapshot.py` imports `ggTrader.lab.data` (universe helper, `collapse_daily_duplicates`) and so follows whatever `src/` is on its path when rebuilt.
6. **Not done in P0:** the "fix the changelog defect" item of the section 9 P0 row (not in the task list given). The live golden of spec 7.3 item 4 was added beyond the list (section 4.2a). No unit tests were added for the new scripts (the full suite was not run; nothing in `src/` or `tests/` changed).
7. **Reading `.env`:** never read directly. The snapshot script and the exploratory read-only queries obtained the connection string through `get_db_connection_string()` by running with the production checkout's `src` on `PYTHONPATH` (read-only use; its virtualenv interpreter was used unchanged).
8. **Absolute paths:** evidence scripts and some JSON `spec`/`tape` fields refer to scratchpad locations; the reference JSON uses repo-relative tape paths.

## 6. Skipped (not copied)

- Virtualenvs and caches: `venv0285`, `venv100`, `venv100b`, `venv120` (scratchpad), `vbt-spec/S8-evidence/venv` (1.7 GB), `__pycache__`, `.ruff_cache`, `nc_numba` (numba cache, 7.9 MB).
- Third-party package copies and wheels: `vbt-research/v12/` (vectorbt 1.2.0 source tree), `vbt-research/wheels/` (vectorbt wheel), `vbt-spec/S8-evidence/vectorbt*` and `.dist-info`, `vbt-spec/S6-evidence/shim/` (psycopg2 and dotenv stubs with binary libraries).
- Downloaded third-party text: `vbt-spec/S8-evidence/dl/` (a downloaded skill pack; untrusted content kept out of the repo), `mlruns`.
- Files over 2 MB: `vbt-spec/report.json` (2.1 MB), `vbt-spec/lead-checks/up.csv` (5.5 MB), `vbt-spec/S5-evidence/tape_close.parquet` (2.06 MB), `vbt-spec/S6-evidence/tape_{open,high,low,close,volume}.parquet` (7.7 to 10.5 MB; superseded by the frozen snapshot), `vbt-research/B-bench/drift_venv{0285_none,100_numba,120_numba,120_rust}.npz` (4 x 33.5 MB).
- Not research artefacts: the scratchpad `gh/` (919 MB), `x/`, `vbt120/`, crontab backups.
- All scripts and outputs that were small enough, including the three `.prof` files under 2 MB, were copied.

## 7. Timing summary

| Step | Time |
|---|---|
| Legacy environment build and import check | 86.7 s |
| Snapshot export plus 60-symbol verification | 21.9 s |
| `smoke2_ensemble_sp500` | 363 s |
| `core17_ensemble_sp500` (attempt 1, killed at 6 GB) | 831 s to kill |
| `core17_ensemble_sp500` (attempt 2) | 1,421 s |
| `core17_ensemble_sp500_dsr0` | 1,363 s |
| Nine strategy references | 17 to 326 s each (table above) |
| Signal golden | 8 s |
| Live golden | 135 s |
