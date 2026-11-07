# P2 completion: the `ggTrader.engine` adapter, contract tests, import boundary and upgrade tooling

Date 2026-11-06. Branch `vbt12/engine` in `/home/flynn/ggTrader-vbt12`. Nothing committed, staged, pushed or tagged.
No Docker, crontab, Alpaca, Telegram or database action (no database access at all; the one real-shape input is
the frozen tape `data/snapshots/ohlcv_20261106.parquet`). The `provenance` migration is written, not run.

## Verdict against the P2 exit criteria

| Criterion | Result |
|---|---|
| L0-L3 green on both engines | `tests/engine`: 264 tests, all pass with the default engine `rust` and with `GGT_VBT_ENGINE=numba`; engine-parametrised tests run both engines inside each run |
| Rust dispatch probe passes | `python -m ggTrader.engine.selfcheck`: `rust_calls: 1`, `year_freq: 252 days`; tests fail if Rust is not dispatched (and prove numba makes 0 Rust calls) |
| Boundary lint green, only legacy importers allow-listed | `ruff check .` clean (TID251); AST test green; 7 allow-listed files (below) |
| Ruff clean | `ruff check .` and `ruff format --check .` clean |
| Whole suite via `scripts/run_tests.sh` | 2630 passed, 0 failed (see "Suite") |
| Parity with `lab/simulate.py` | `run_signals` / `run_weights` equity bit-identical (`np.array_equal`), synthetic and frozen tape, both engines |
| Configured indicators reproduce legacy values | yes, far inside class (ii); table below |

## Files

New: `src/ggTrader/engine/{__init__,config,selfcheck,indicators,signals,portfolio,stats,windows,provenance,_compat}.py`,
`src/ggTrader/utils/results.py`, `tests/engine/*` (15 test modules, `panels.py`, `scenarios.py`, `golden_io.py`,
`defaults_probe.py`, `golden/engine_scenarios.json`, `golden/vbt_defaults.json`), `tests/lab/test_persist_provenance.py`,
`tests/scripts/test_vbt_latest.py`, `scripts/vbt_latest.py`, `scripts/engine_parity_evidence.py`,
`.github/workflows/vbt-upgrade-watch.yml`, `.github/dependabot.yml`, `.claude/skills/vbt-upgrade/SKILL.md`,
`research/programme/vectorbt-engine/p2-evidence/p2_engine_parity_20261106.json`, this note.

Modified: `ruff.toml` (TID251 + allow-list), `scripts/check_vbt_engine.py` (delegates to `selfcheck`; same API),
`tests/conftest.py` (`--update-golden`, `--golden-reason`), `src/ggTrader/lab/persist.py` (additive provenance column
and optional argument), `docs/architecture.md`, `docs/changelog.md`.

Transitional allow-list (`ruff.toml` and `tests/engine/test_boundary.py`, must reach zero in P5):
`lab/simulate.py`, `lab/strategies/signals.py`, `scripts/position_sizing_regimes.py`, `scripts/position_sizing_wfo.py`,
`scripts/leveraged_trend_research.py`, `scripts/legacy_reference_runs.py` (P0; runs under the legacy 0.28.5 interpreter, so
it cannot use the engine), `tests/scripts/test_check_vbt_engine.py`.

## Configured indicators against the legacy pandas formulas (150 symbols x 1,929 bars, frozen tape)

Max abs difference after each symbol's own 200-bar burn-in (a global 200 is wrong for late listings; the first version of
the driver used it and showed a spurious 1.3e-2 MACD gap):

| Indicator | Max abs diff | Note |
|---|---|---|
| Wilder RSI (`RSI(window=2n-1, ewm=True)`) | 0.0 | first valid bar 27 not 14 (1,950 early cells legacy-only) |
| EMA 10/20/50 | 0.0 | vectorbt NaN-pads `w-1` bars |
| MACD line | 0.0 | |
| MACD signal and histogram | 2.8e-17 | signal seeded at the first valid MACD bar; max 0.83 inside the burn-in |
| Bollinger (`ddof=1`) | 1.1e-12 | |
| ATR (`ewm=False`) | 2.1e-14 | vectorbt valid one bar earlier (152 cells) |

Voter cells (engine presets + vectorbt crossovers against legacy `bb/rsi/ema/macd_signals`): 0 differing cells after
burn-in except 1 Bollinger exit that a tie audit explains (flip within 1e-9 of the threshold on the flip bar or the bar
before). All other differences (RSI 27+83, EMA 183+180, MACD 328+6) sit inside the burn-in.

## Findings

1. **`ddof` is exposable**: `BBANDS.run(..., ddof=1)` works through `kwargs_to_args` (not in the signature, so the defaults
   tripwire also pins the behaviour). No maths of ours.
2. **DSR configuration differs from the spec.** `deflated_sharpe_ratio(nb_trials=len(grid), var_sharpe=252/(T-1))`
   reproduces `gates.dsr_check` to 1.6e-17 only with **`bias=False`** (the legacy inputs are pandas' unbiased `skew()` /
   `kurtosis()`). Spec 5.4 item 6 says `bias=True`; that differs by up to 5.7e-6 (relative up to 4e-5) on 8 seeded
   fixtures. S5's `1/(T-1)` is confirmed wrong (spec Appendix A). Recorded in `stats.dsr`'s docstring and a test.
3. **Rust dtype traps**: an int threshold in `crossed_*` and `fshift` on a *boolean* frame both raise under `engine="rust"`.
   `signals.py` coerces to float64; tests pin the raw failures so an upstream fix is noticed.
4. **Indicator output frames are rebuilt on each attribute access** (`obj.lower` is a new DataFrame), so
   `obj.lower.columns = ...` is silently lost. I hit this myself; assign to a local. Worth a line in the P4 strategy guide.
5. The 5-voter flip numbers reproduce S5's class (ii) result on a different symbol sample.
6. `hide_params` removes the constant flags from column levels, but the RSI window level still reports `2n-1` (27).
7. ATR stop: an entry on bar 13 gets a stop from vectorbt and none from legacy (legacy ATR starts a bar later), which changes
   that trade; the parity test for `atr_mult` therefore blanks entries before bar 30. Irrelevant for real histories (>=60 bars).

## Parity evidence (`p2-evidence/p2_engine_parity_20261106.json`)

12 simulator comparisons per engine (signals plain and 5% trailing on tape and synthetic, weights on tape and synthetic):
`bit_identical: true`, `max_abs_diff: 0.0`, trade counts equal, on both `rust` and `numba`. Engine run time on the tape
panel 0.1 s against 0.1-0.2 s legacy (the legacy numba first call 4.8 s of JIT).

## Tests added

`tests/engine`: 264 tests in 15 modules: config 14, selfcheck 14, indicators 34 (independent numpy references plus legacy
forms, both engines), signals 26, portfolio 33, stats 34, windows 13, provenance 23, compat 4, boundary 27 (incl. ruff
cross-check), golden 11, defaults 11, warnings 2 (`-W error` lane), lock consistency 10, upgrade tooling 8. Elsewhere:
`test_persist_provenance` 5, `test_vbt_latest` 12. Mutation check: nine deliberate breakages (ddof 0, RSI window n, SMA MACD,
EWM ATR, 365-day year, DSR bias True, DSR var 1/(T-1), no finite guard, no float coercion) were each caught by a test.

## Suite

First full run (`scripts/run_tests.sh tests/`): 2629 passed, 1 failed in 9 min 39 s. The failure was P1's own guard
`test_src_has_no_bare_pct_change` flagging `value.vbt.pct_change()` in `portfolio.py`; replaced with
`pct_change(fill_method=None)` (bit-identical). Final run after that fix and the last test edits: **2630 passed**, 30 integration deselected, 0 failed, 9 min 25 s, peak under the 8 GB cap.

## Open problems and uncertainties

- Workflow and Dependabot YAML parse (checked with the system PyYAML) and the shell steps were exercised individually
  (`sed` bump on a copy of `pyproject.toml`, `scripts/vbt_latest.py` against live PyPI), but the workflow has not run on
  GitHub. Unverified: the action versions `upload-artifact@v5` / `download-artifact@v5`, the Dependabot `uv` ecosystem
  handling of the `[rust]` extra (the engine stack is ignored anyway), and the repository setting "Allow GitHub Actions to
  create pull requests".
- Allow-list holds a P0 script that runs on the legacy interpreter; archive it after P6.
- The L5 oracle (17-fold, live golden) does not exist yet; the runbook says so.
- Persist: the SQL is unit-tested against a fake connection only; the `ALTER TABLE ... IF NOT EXISTS` statements have not
  been run on a database.
- `run_signals` ffills `close` like the legacy function; `vol_target` sizing and `compute_vol_scalar` stay in lab (callers pass
  `size=`).
- Goldens are produced by this implementation (cross-checked numba == Rust and, for equity, against the legacy simulator),
  so they pin behaviour rather than prove it; the independent references in `test_indicators.py` / `test_stats.py` do that.
- `CLAUDE.md` still lacks the engine rules suggested by S2; I did not edit the owner's instruction file.
