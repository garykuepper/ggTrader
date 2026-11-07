# S3: What moving ggTrader to the vectorbt 1.2.0 stack takes (hands-on)

Date 2026-11-06. Worktree `.claude/worktrees/agent-abed3ab263d5272c0` (nothing committed; `/home/flynn/ggTrader` untouched; DB access read-only).
Evidence: `.../vbt-spec/S3-evidence/` (file names in brackets below). Old stack = `/home/flynn/ggTrader/.venv` (pandas 2.3.3, numpy 2.2.6, numba 0.61.2, vectorbt 0.28.5, SQLAlchemy 2.0.48, yfinance 1.2.0). New stack = fresh py3.12 venv in the worktree (pandas 3.1.1, numpy 2.5.4, numba 0.68.1, llvmlite 0.50.0, vectorbt 1.2.0 + vectorbt-rust 1.2.0).

## 0. Bottom line

1. **The migration is feasible and small in code: 4 source edits, all verified.** Resolver passes once `pandas-ta` is deleted. Full suite on the new stack: 1770 passed / 0 failed / 0 skipped (old stack baseline also 1770 / 0 / 0). Two edits are needed to get there (`event_study.py`, `pyproject.toml`); a third (`indicators.py`) is a performance fix, not a test fix.
2. **Numbers do not move.** On real cached SP500 data (100 symbols, 860 bars, real PIT mask) the 48-combo ensemble signal sweep and 6 xs_momentum weight combos give **bit-identical equity curves and metrics** old vs new (max abs diff 0.0, includes the Rust engine) [41, 57]. The only numeric change seen anywhere: a 1-ulp (1e-16) change in `dsr_value` from pandas 3 `skew()/kurtosis()` last digits [25].
3. **Hidden performance regression found and fixed in one line.** Unpatched, the 48-combo sweep is **3x slower** steady-state (13.1 s vs 4.3 s). Root cause is not vectorbt or Rust: pandas 3 leaves `pd.concat({sym: series}, axis=1)` as 100 unconsolidated blocks, so every later frame op is ~5x slower (66 ms vs 13 ms per RSI chain) [54, 53]. `.copy()` after the concat in `extract_close/open/volume` restores it: 4.6 s, still bit-identical [55, 57]. Earlier research (B, F) predicted "1.2.0 is slower" from numba kernels; the real cause was this.
4. **Rust removes the JIT cold start that every cron run pays today.** Cron runs `docker compose run --rm` (fresh container, no numba disk cache): today's first-call cost is **19.6 s** (measured, old stack); new stack with Rust **4.5 s** (3.1 s of that is `import vectorbt`); new stack without Rust 29.4 s [50]. 2-fold/60-symbol WFO end to end in a fresh process: 52 s old vs 32 s new patched [58].
5. **Biggest silent traps (unfixed they change research numbers, not live orders):** `event_study.split_event_returns` (`.stack()` keeps NaN in pandas 3; 3 tests fail), 44 bare `pct_change()` sites (2 on the live path, both on gap-free equity curves or gated off), and `engine="auto"` flipping to numba silently if the Rust wheel is missing (25 s extra JIT, no error).
6. Effort ESTIMATE: ~9 agent-hours code + verification, ~3 more for the 17-fold re-baseline; deploy waits for the ~2026-12-29 freeze.

## 1. Resolver outcome and dependencies

Procedure [01-06]: `pip install --dry-run -e . -r target_constraints.txt` (vectorbt==1.2.0, vectorbt-rust==1.2.0, pandas>=3.0.3, numpy>=2.4.6, numba>=0.66).

| Step | Change to pyproject.toml | Result |
|---|---|---|
| 1 | none | `ResolutionImpossible`: ggTrader pins `vectorbt<0.29,>=0.28.5` (`pyproject.toml:39`) |
| 2 | remove vectorbt pin only | `ResolutionImpossible`: `pandas-ta 0.4.71b0` and `0.4.67b0` both require `numba==0.61.2`, vectorbt 1.2.0 needs `numba>=0.66` |
| 3 | also remove `pandas-ta` (`pyproject.toml:22`) | **resolves**; real install 83 s, 141 packages, `pip check` clean; venv 1.3 GB vs 1.4 GB old |

`pandas_ta` (and `ta`) are imported nowhere in `src/`, `scripts/`, `tests/`, `ggt.py` (AST scan of every import) [01-03]. **Remove `pandas-ta`**; no other blocker. Plotly 7.1.0 also imports cleanly with vectorbt 1.2.0 (tested), so the `plotly<7` cap (pyproject.toml:25) can go.

Every pyproject dependency (full table with wheel-metadata bounds: [06_dep_table.md]). "Supports pandas 3 / numpy 2.4": all resolved on the new stack and passed the suite; the metadata column shows what each wheel declares.

| Dependency | old | new (resolved) | pandas3/numpy2.4 release? / note |
|---|---|---|---|
| numpy / pandas / numba | 2.2.6 / 2.3.3 / 0.61.2 | 2.5.4 / 3.1.1 / 0.68.1 | numba 0.68.1 caps `numpy<2.6`, so pin numpy `<2.6` |
| vectorbt (+ vectorbt-rust) | 0.28.5 | 1.2.0 (+1.2.0) | needs numpy>=2.4.6, pandas>=3.0.3,<4, numba>=0.66; rust wheel must equal vectorbt minor |
| pandas-ta | 0.4.71b0 | removed | **blocker, never imported** |
| ta | 0.11.0 | 0.11.0 (sdist build) | never imported; propose removal |
| scikit-learn / scipy | 1.8.0 / 1.17.1 | 1.9.1 / 1.18.1 | OK; scipy is imported by src (3 files) but undeclared (transitive) |
| lightgbm / hmmlearn | 4.6.0 / 0.3.3 | 4.7.0 / 0.3.3 | OK; `models/ensemble_gate.joblib` loads, predictions identical to 12 digits [26]; hmmlearn never imported |
| yfinance | 1.2.0 | 1.7.0 | OK; index becomes `datetime64[s]` (see 3) |
| SQLAlchemy / psycopg2-binary | 2.0.48 / 2.9.11 | **2.1.8** / 2.9.13 | uncontrolled minor jump; 13 live `paper/persist.py` read getters return identical results [23] |
| pandas-market-calendars / exchange_calendars | 5.5.0 / 4.13.2 | 5.6.0 / 4.13.2 | OK (`pandas>=2.3`) |
| alpaca-py | 0.43.4 | 0.45.0 | OK; order path not exercised (no orders by rule) |
| joblib, matplotlib, plotly, pyarrow, optuna, seaborn, mplfinance, xlrd, xlsxwriter, fsspec, tabulate, tqdm, ccxt, pytrends, typer, pydantic, requests, defusedxml, pypdf, python-dotenv, typing-extensions | all resolved (matplotlib 3.11.2, optuna 5.0.0, pyarrow 26.0.1, plotly 6.9.0, ccxt 4.5.92...) | | OK. Statically imported: joblib, matplotlib (1 script, 1 test), requests, tqdm, typer, defusedxml, pypdf, pytrends, dotenv, ccxt (1 script). **Zero static imports**: optuna, seaborn, mplfinance, xlsxwriter, tabulate, fsspec, hmmlearn, ta, pandas-ta (candidates for removal after owner check) |
| beautifulsoup4 | 4.14.3 | 4.15.0 | imported by `lab/event_timing_sources.py` but **undeclared** (rides in via yfinance); declare it |

Keep `pyarrow`: pandas 3's default `str` dtype is Arrow-backed only when pyarrow is installed; dropping it changes string behaviour.

## 2. Test suite (scripts/run_tests.sh, cap 8G, swap 0 kept; only change: `PYTHON="${GGT_PYTHON:-...}"` so one script can drive both interpreters)

| Run | Result | Log |
|---|---|---|
| Old stack baseline (`/home/flynn/ggTrader/.venv`) | **1770 passed, 0 failed, 0 skipped**, 30 integration deselected, 1 warning, 379 s (shared box, load 8-16) | [11] |
| Old stack, `-W error::FutureWarning` | 1770 passed: the tests never hit the `pct_change` gap case, so green tests are blind to that hazard | [13] |
| New stack as resolved | 1765 passed, **5 failed**, 0 errors, 0 skipped, 174 warnings, 363 s | [10] |
| New stack + 2 fixes | **1770 passed, 0 failed**, 174 warnings, 244 s (final, with indicators patch, re-run) | [12, 14] |

Failures by root cause:

| # | Test | Root cause | Fix |
|---|---|---|---|
| 1-2 | `tests/lab/test_event_study.py::TestSplitEventReturns::test_separates_event_and_nonevent_cells`, `::test_empty_mask_gives_empty_event_series` | **pandas-3 behaviour**: `DataFrame.stack()` keeps NaN rows (old default dropped them). `src/ggTrader/lab/event_study.py:48-49` | `.stack().dropna()` (works on both stacks; 28 tests pass on 2.3.3 with it) |
| 3 | `tests/lab/test_fomc_drift.py::test_event_signal_session_semantics` | same function (`event_study.py:48`), returns 10 rows instead of 1 | same fix |
| 4-5 | `tests/scripts/test_ruff_excludes_research.py` (two tests) | **environment, not pandas**: `.venv/bin/ruff` missing because the first install used `.[test]`; `.[dev]` carries `ruff==0.15.12` | install `.[test,dev]` (as CI does, `ci.yml`) |

No vectorbt-1.x API failure, no removed-API failure, no flaky test, no dependency-import failure. `scripts/*.py --help` smoke on 36 argparse scripts (24 non-argparse scripts skipped on purpose: `--help` would execute backfills): identical on both stacks; only `update_universe_ccxt.py` / `update_venue_listings.py` fail, with the same pre-existing `ModuleNotFoundError ggTrader.data.core.venue_listings` on old and new [60].
All 174 new-stack warnings are `Pandas4Warning`/deprecations, none breaking: `inplace=True` (16 sites in `src/`, e.g. `tiingo_loader.py:105,106,144,148`, `cached_yfinance_loader.py:202`, `yfinance_loader.py:82,86,95`, `timescaledb_loader.py:106`); `Timestamp.dayofweek` (`paper/feature_gate.py:92`, 140 warnings); `read_csv(float_precision=)` (`scripts/static_leverage_aftertax.py:70`); xlrd engine deprecated (`lab/century_leverage.py:97`); vectorbt's own `pd.Timedelta("1d")` triggered by `FREQ "1d"` (`lab/data.py:23,44`, `strategies/signals.py:65`: use `"1D"`).

## 3. pandas-3 hazard audit (src/ + scripts/)

| Hazard | Count / location | Live path? | Verdict |
|---|---|---|---|
| bare `pct_change()` (no `fill_method`) | **44 verified** (src 13 code + 1 comment line, scripts 30) | `paper/overlay.py:88` (sleeve equity curves, gap-free) and `paper/feature_gate.py:50` (single symbol; only when `ML_GATE_ENABLED`, off). The module chain `paper.trader` imports `lab.simulate`, `strategies.*`; none of them use bare `pct_change` | equity-curve sites (metrics.py:15,46,47; wfo.py:592; blend.py:39,170,173; significance.py:63; overlay.py:88) are gap-free: `fill_method=None` is identical. Price-panel sites (`event_study.py:122`, `strategies/idio_vol.py:72`, `max_effect.py:52`, `feature_gate.py:50`) need `.ffill()` first to keep old numbers. Measured on real cached SP500 closes: 13 of 520 symbols (2021-22) and 20 of 533 (2025-26) have gaps where old padding gave 0.0 and pandas 3 gives NaN (2,434 and 2,920 cells) [24]. Scripts: 30 research-driver sites, mechanical |
| `agg(np.std)` / `apply(np.std)` ddof change | 0 hits in src, scripts | no | clear |
| `stack()` NaN | `event_study.py:48,49` (only 2 `.stack(` in src) | no (research) | **real, fixed**, caused 3 failures |
| groupby | 9 `groupby(` in src; string/period keys, loops `for sym, g in df.groupby("symbol")`; `observed=` not needed | no | clear (suite + bit-identical sweeps) |
| string dtype | DB frames now have `str` column levels (was object); `df[sym]["close"]`, `pd.concat({sym: ...})`, `from_product` all work; `dtype == object` appears only as constructors (`regime.py:33`, `leveraged_rotation.py:44`) | symbols everywhere | clear (measured) [21] |
| copy-on-write / chained assignment | greps for `df[a][b] = x`, `.values[...] = x`, in-place `to_numpy` writes: 0 DataFrame hits (only dict chains) | no | clear; numba/vbt accept read-only inputs |
| **block fragmentation** (new, not on any prior list) | `pd.concat({..}, axis=1)` in `strategies/indicators.py:15,21,234` (also `simulate.py:217-226`, `overlay.py:50`, `signal_runner.py:79`, `train_gate.py:164`) | feeds every sweep | **3x slowdown, fixed** with `.copy()`; live sites run once per run, not worth touching |
| datetime resolution | DB frames `datetime64[us, UTC]` (was ns); yfinance `[s, UTC]` (was ns). Epoch conversions (`astype("int64")`, `.view`, `.asi8`, `.value`): **0 in src**, 1 in scripts | write path | values/instants equal; see below |
| freq strings | `period_range(freq="M")`, `Period freq M` ok; no `resample("M")`, no `freq="H"`, no `applymap`, `fillna(method=`, `groupby(axis=`, `iteritems`: 0 hits | no | clear |

**DB read path, READ-ONLY on `CachedYFinanceLoader._db_fetch`, 7 symbols x 250 bars, pandas 2.3.3 vs 3.1.1** [db_read_smoke.py, 21]: values **bit-identical** (max abs diff 0.0); same instants; same shape (250 x 35); no NaN; all weekdays Mon-Fri (the 2026-08-20 day-shift fix is intact). Index dtype `datetime64[ns, UTC]` -> `datetime64[us, UTC]`; column levels `object` -> `str`. `pd.read_sql` with a raw psycopg2 connection works but emits the UserWarning "only supports SQLAlchemy" (`cached_yfinance_loader.py:193`), same as before.
**DB write path** (`_cache_to_db`), executed with `execute_values` and `_connect` stubbed so nothing is written: the captured records are **identical** old vs new (naive `datetime.datetime`, `tzinfo=None`, NaN -> None, symbol `str`), and identical for ns/us/s indexes. The CLAUDE.md naive-UTC rule holds under pandas 3 because the code strips tz before `to_pydatetime()` (`cached_yfinance_loader.py:234-235`). Live yfinance probe (2 symbols, 8 days): identical closes; `YFinanceDataLoader` output index `[ns, UTC]` -> `[s, UTC]`, columns `str` [22]. Not verified: a real INSERT against a scratch Postgres (no scratch DB; writes forbidden).

## 4. Real-data end-to-end old vs new [e2e_wfo.py, e2e_sweep.py, 41, 57, 58]

Read-only guard: yfinance loader and cache writers replaced, DB session `readonly=True`. Window 2021-01-31 .. 2022-07-31, SP500 PIT universe.
- **Signal sweep** (100 symbols, 48 ensemble combos, real `pit_entry_mask`): 48/48 equity curves **bit-identical**, all 7 metrics diff 0.0, n_trades identical, PIT mask identical. New stack ran `engine=auto` (Rust).
- **Weight path** (xs_momentum, 6 combos, `simulate_weights` -> `from_orders` targetpercent): 6/6 bit-identical.
- **`run_wfo`** (60 symbols, 2 folds, 48 grid): `fold_results` equal except `dsr_value` 0.2616441817035998 vs ...977 (1 ulp), attributed to pandas 3 `skew()/kurtosis()` arithmetic (isolated probe: skew -0.03764809723018072 vs ...077) [25]; `live_params` equal. Caveat: this config trades nothing out of sample (both stacks: OOS equity flat 10000, anchor "MaxDD 0.0%, CAGR 0.0%", DSR fails), so the per-combo sweeps above are the real comparison, not the OOS curve.
- **Wall clock** (idle box, load ~2, fresh process, includes JIT): old 52.2 / 51.6 s, new patched 32.3 / 32.3 s [58]. Steady-state signal sweep (3 reps, warm): old 4.3 s; new unpatched 13.1 s (Rust or numba, same); new patched **4.6 s** (numba 4.6-4.8) [51, 55]. Memory: peak RSS 1.1 GB old vs 1.2 GB new (60 symbols). The 35 s PIT-mask build (`eligible_at`, O(T^2)) is identical on both and untouched.
- Not run: a 17-fold oracle. F's plan (Phase 0 oracle on pinned window) still applies; run it on the old stack first, then new, and diff.

## 5. Docker / CI

- `Dockerfile:2` `python:3.12-slim` already satisfies `>=3.11`. All 141 packages came as binary wheels (the single sdist, `ta`, is pure Python and goes away with removal). vectorbt-rust 1.2.0 ships `cp312 manylinux_2_17_x86_64` (917 KB; installed size 2.2 MB, B), slim's glibc 2.36 is fine; aarch64 wheel also exists. `build-essential`/`libpq-dev` stay needed only for the unpinned-psycopg2 fallback; psycopg2-binary is a wheel.
- Size/build: venv 1.3 GB vs 1.4 GB (measured on disk). Docker image size and build time **ESTIMATE**: image size within +/-5%, `docker-build.yml` ~2.5 min -> ~2.5-3 min (pip step ~80 s measured here). I did not build the image.
- **Cron model matters**: `scripts/paper_trade.sh` uses `docker compose run --rm`, a fresh container per run, so numba never gets a warm disk cache. Measured cold-start matrix, fresh process, fresh `NUMBA_CACHE_DIR` [50, coldstart.py runs `simulate_signals` with ATR stops, then `simulate_weights`]:

| Stack | import vectorbt | first `simulate_signals` | first `simulate_weights` | total | warm-cache total |
|---|---|---|---|---|---|
| old 0.28.5 numba (**production today**) | 2.9 s | 14.9 s | 1.2 s | **19.6 s** | 7.0 s |
| new 1.2.0 + Rust (`auto`) | 3.1 s | **0.77 s** | 0.04 s | **4.5 s** | 3.8 s |
| new 1.2.0 numba only | 3.3 s | 22.9 s | 2.6 s | 29.4 s | 8.7 s |
| new numba, `NUMBA_CPU_NAME=generic` | 2.7 s | 20.7 s | 2.5 s | 26.4 s | 8.0 s |

  So Rust turns a 15-20 s JIT tax into near zero. Options: (a) **Rust (recommended)**: no cache needed; (b) if Rust is absent, the silent numba fallback costs +25 s per run: add a build-time and run-time guard (`python -c "import vectorbt._engine as e; assert e.is_rust_available()"` in the Dockerfile next to the existing `import ggTrader.paper.trader` check, `Dockerfile:25`) and pin `vectorbt-rust==1.2.0` exactly equal to vectorbt (minor mismatch => one warning and numba fallback); (c) a baked numba cache (`NUMBA_CACHE_DIR` in the image, `NUMBA_CPU_NAME=generic` so the build runner's CPU does not matter) cuts the fallback from 26 to 8 s, but I did not test baking inside an image: UNVERIFIED, not needed if (a) holds.
- CI (`ci.yml`): Python 3.12, `pip install -e ".[test,dev]"`, ruff + pytest; works unchanged after the pin edits. `docker-build.yml` runs no tests; consider a smoke step. Without a constraints file the 2026-09-23 failure mode (unpinned rebuild pulls a breaking major) remains; today's resolve already drags SQLAlchemy 2.1.8, optuna 5.0.0, scikit-learn 1.9.1, pyarrow 26. Generate the constraints file from the validated freeze [05_new_stack_freeze.txt] and install with `-c`.

## 6. Ordered migration patch list (files, one-line change, effort ESTIMATE in agent-hours)

A. Safe now, valid on both stacks (can land before the freeze ends; do not touch behaviour): 
1. `pyproject.toml:22` delete `pandas-ta`; optionally `ta` and the six other zero-import deps; add `beautifulsoup4`, `scipy` (undeclared direct imports). 0.5 h.
2. `lab/event_study.py:48-49` `.stack()` -> `.stack().dropna()` (verified on both). 0.25 h.
3. `lab/strategies/indicators.py:15,21,234` wrap concat result in `.copy()` (helper `_consolidated`); no numeric change (bit-identical on 54 curves); on pandas 2 it is a no-op. 0.25 h.
4. 44 `pct_change()` sites: `fill_method=None` on gap-free curves (9 src sites + scripts that use curves), `.ffill().pct_change(fill_method=None)` on the 4 price-panel src sites; unit tests with a NaN gap that pass on 2.3.3 so the old numbers are preserved. 2.5 h incl. tests (scripts 30 sites mechanical, ~1 h of that).
5. Deprecation cleanup (no result change): 16 `inplace=True`, `feature_gate.py:92` `dayofweek`->`day_of_week`, `FREQ "1d"`->`"1D"` (3 sites, confirm identical vectorbt freq), `century_leverage.py:97` xlrd engine, `static_leverage_aftertax.py:70`. 1.5 h.
6. `scripts/run_tests.sh:21` `PYTHON="${GGT_PYTHON:-$REPO/.venv/bin/python}"` (needed for A/B; optional). 0.1 h.
7. Constraints file from [05_new_stack_freeze.txt]; `Dockerfile` `pip install -c`. 0.5 h.

B. The flip (single commit on a branch; deploy only via `ggtrader-deploy` after ~2026-12-29):
8. `pyproject.toml`: `vectorbt==1.2.0`, `vectorbt-rust==1.2.0`, `pandas>=3.0.3,<4`, `numpy>=2.4.6,<2.6`, `numba>=0.66,<0.69`; drop `plotly<7` cap and its stale comment (`:23-25`) and the vectorbt cap comment (`:38-39`). 0.5 h.
9. `Dockerfile:25`: add the Rust-available assertion and one tiny `Portfolio.from_signals(...).value()` smoke call. 0.5 h.
10. Rebuild venv, `.[test,dev]`, full suite (4-6 min). 0.5 h.
11. 17-fold oracle on old then new stack, diff fold table JSON; live golden (entries/exits/sizes over the last ~60 trading days, 365-day lookback) old vs new. 3 h agent time + ~2 h wall (ESTIMATE; the sweeps measured here took 1.5 min per 100 symbols x 1 fold).
12. Docs: `docs/changelog.md` entry, `docs/architecture.md`, `docs/installation*.md`, CLAUDE.md Docker/vectorbt notes. 1 h.
Total ESTIMATE ~11-12 agent-hours, of which only #8-#9 touch what the live image installs.

## 7. Owner-visible behaviour changes (anything that changes numbers)

| Change | Effect | Condition |
|---|---|---|
| Portfolio simulator, indicators, sweeps, PIT mask | **none** (bit-identical on real data) | measured, 54 curves |
| `dsr_value` and any gate threshold that falls within 1e-16 of the edge | 1-ulp noise (pandas `skew/kurtosis`) | theoretically can flip a gate at exactly the boundary; none seen |
| `event_study.split_event_returns` (FOMC and event studies) | event counts and means would change (NaN rows kept) | only if edit A2 is skipped; 3 tests fail |
| Bare `pct_change()` on gappy price panels (`idio_vol`, `max_effect`, `event_study`) | gaps become NaN instead of 0.0 returns; changes cross-sectional ranks for delisted/stale symbols (13-20 of ~530) | only if sites are changed to plain `fill_method=None` instead of `.ffill()` first |
| Cold-start per cron run | -15 s (19.6 s -> 4.5 s) | with Rust; +10 s worse (29 s) if Rust silently absent |
| Index dtype / epoch units | `[ns]` -> `[us]` (DB), `[s]` (yfinance); instants unchanged | matters to any future `astype("int64")` code (none in src) |
| Dependency drift | yfinance 1.2.0->1.7.0, SQLAlchemy 2.0->2.1, alpaca-py 0.43->0.45, lightgbm 4.6->4.7, sklearn 1.8->1.9.1, scipy 1.17->1.18 | pin via constraints; ML gate model predictions identical; model file pickled with sklearn 1.9.0 warns on both |
| Re-baselining | accepted by owner; because simulation is bit-identical, the re-baseline mainly confirms rather than changes figures | |

## 8. Unverified / limits

- Orders and Alpaca broker path under new alpaca-py 0.45 not exercised (no orders by rule). The `paper/persist.py` write functions and `lab/persist.py` not run (writes forbidden); read getters verified.
- 30 integration tests (DB/network) were deselected on both stacks and not run.
- No real INSERT test of `_cache_to_db`, only captured records; no scratch Postgres.
- No 17-fold run; the e2e WFO config trades nothing OOS, so equality evidence rests on the 54 per-combo curves.
- Image build not done (size/time are estimates). Baked numba cache not tested in an image.
- Timings: box shared (load 2-16); the clean ones (idle load ~2-4) are [51, 55, 58]; ratios under ~10% are noise. Cold-start rows [50] ran at load 4.5-8.5; CPU-time columns are in the JSON.
- `engine="auto"` switching to Rust is verified by wall time and cold-start behaviour here, and by B's kernel-call counters; I did not re-count kernel calls.
- Patches live only in the worktree (`70_worktree_patches.diff`: pyproject, run_tests.sh, event_study.py, indicators.py). Full list of those I applied to get green: event_study.py:48-49, pyproject (vectorbt cap + pandas-ta removed), `pip install .[dev]`; plus indicators.py for speed. No test was edited.
