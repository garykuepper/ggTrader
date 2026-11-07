# P8b: fixes for the defects the P8 rehearsal found in the vectorbt upgrade procedure

Date 2026-11-06. Branch `vbt12/engine`, worktree `/home/flynn/ggTrader-vbt12`. Nothing committed, staged, pushed or tagged; no Docker, crontab, Alpaca, Telegram or database action; no `.env` read; nothing placed an order. Source note: `P8-completion.md` (sections 6 and 7).
Terms: **tripwire** = a test that fails when a vectorbt upgrade changes a definition. **Reference side** = the independent formula or recorded upstream behaviour a tripwire compares with. **Welford** = the numerically stable running-variance method. **DSR** = deflated Sharpe ratio. **Pair** = `vectorbt` and `vectorbt-rust` at the same version.

## 1. The 16 defects

| # | Defect | Status | What was done |
|---|---|---|---|
| R1 | `test_defaults` compared the vectorbt version string, so every bump failed | **fixed** | `tests/engine/test_defaults.py`: `BUILD_KEYS = ("vectorbt_version",)` dropped from both sides before `_diff`; the version stays in the probe output and golden, and in provenance. New tests: version-only difference is not a default change; a changed default is still reported next to a bump; the golden still records its version. The failure message now names the installed pair and the version the golden was made in. |
| R2 | `test_selfcheck` hard-coded 1.1.9 / 1.2.9 | **fixed** | `tests/engine/test_selfcheck.py`: the mismatch and patch cases derive from the installed version; added a different-major case and fixed-literal parametrised pairs that cannot go stale. |
| R3 | `git diff -U0 uv.lock \| grep` printed no package names | **fixed** | Runbook step 2 and the workflow report now use uv's own delta: `grep -E '^(Added\|Removed\|Updated) ' lock.log`. Checked on a real `uv lock` to 1.1.2 in scratch: `Updated vectorbt v1.2.0 -> v1.1.2`, `Removed nbformat v5.11.1` and six more. |
| R4 | Step 5 pointed to `research/audits/engine-migration/_*.json` and a database session | **fixed** | Step 5 names the real artifacts (`p5-evidence/_p5_compare_*.json`, `p7-evidence/p7_live_golden.json`), the commands (`p5_wfo_validation.py --list`, `--runs`, `--out-dir`; `p7_live_golden.py --out`) and says the drivers read only the tape. `p5_equity_parity.py` is not listed: it compares with legacy code that P5c removes. |
| R5 | Parity script never run; its default `--out` overwrote committed evidence | **fixed** | New runbook step 3b with `--out "$SCRATCH/parity_new.json"` and the diff rule. `scripts/engine_parity_evidence.py`: `--out` now defaults to `DEFAULT_OUT` in the system temp directory (a test pins it outside `research/`). The runbook also says to pass a scratch path to `p5_wfo_validation.py --out-dir` and `p7_live_golden.py --out`, whose defaults are the evidence folders (those two scripts are P5c's and were not edited). |
| R6 | `MSTD` listed under `indicators.py`; no list of unreachable items | **fixed** | Step 1 lists `RSI, MA, BBANDS, MACD, ATR`; a "Not relevant to ggTrader" table (section 3) lists the items the adapter cannot reach. |
| R7 | No procedure for an unreleased commit | **fixed** | New "Pre-release look-ahead" section: compare URL, scratch venv from `uv export` minus the vectorbt lines, `--no-deps` git install, Rust wheel variants (a) and (b), record the sha, run from a copy, never touch the pin. |
| R8 | No close-PR step on REJECT, no ready step on ACCEPT | **fixed** | Step 6: `gh pr close <n> --delete-branch --comment ...`, `gh issue close`, `gh pr checkout` / `gh pr ready`. The workflow's PR body now says REJECT closes the PR. |
| R9 | `docs/vbt_upgrade_rejected.txt` / `vbt_upgrade_log.md` do not exist | **fixed** | Step 6 says "create" for both; `vbt_latest.py` already tolerates a missing file (now also pinned by an end-to-end test). |
| R10 | Step 7 claimed the deploy pre-flight builds the image and self-checks as root and `appuser` | **fixed** | Reworded: the deploy skill runs `selfcheck --engine rust` in the pulled container; the image build does the root and `appuser` smoke steps (`Dockerfile` lines 46 and 59, `docker-build.yml`). A test pins the corrected wording. |
| R11 | `LOCKED` assigned and never used | **fixed** | Used in the compare URL of step 1 and in the pre-release section. |
| R12 | `vbt_latest.py` proposed an older release after the newest was rejected | **fixed** | `scripts/vbt_latest.py`: any version at or below the highest rejected one is skipped (`--force` still names any). Reproduces the drill's case (lock 1.1.1, rejected 1.2.0: was `1.1.2`, now empty). Tests in `tests/scripts/test_vbt_latest.py`; the old test that asserted the buggy behaviour was rewritten. |
| R13 | A bare `pytest -q` collects `research/` | **fixed** | `pyproject.toml` `[tool.pytest.ini_options] testpaths = ["tests"]` (nothing else in the file touched); `ci.yml` and `vbt-upgrade-watch.yml` now run `pytest tests -q`. Tests pin both. |
| R14 | No failure-table rows for version-only / hard-coded-literal failures or for a collection error | **fixed** | Rows added, plus rows for each new tripwire. |
| R15 | The report's `sed` printed only the version line | **fixed** | Resolved by R1; the `sed` pattern now matches the new message prefix (`vectorbt defaults changed`) and a test keeps the two in step. |
| R16 | No triage entry when the new vectorbt value agrees with the independent reference | **fixed** | Triage item 2 split into 2a (new value departs from the reference: flag or RE-BASELINE) and 2b (it agrees: ACCEPT, update the expectation with a reason). |

## 2. The seven gaps from P8 section 6

| Gap | Result | Test |
|---|---|---|
| G1 Bollinger on real tape | **test written** | `test_tripwires.py::test_bollinger_bands_match_pandas_rolling_std_on_the_real_tape`: 60 tape symbols, lower and upper band against pandas `rolling(20).std(ddof=1)`, 200-bar burn-in, `rtol 1e-10`, skipped with a reason when the tape is absent. Margin: 5e-15 on 1.2.0, 1.4e-7 on 1.1.0. |
| G2 Welford stability | **test written** | `test_rolling_std_is_numerically_stable_for_a_large_offset_with_a_small_spread` (level 1e6, spread 1e-2, `np.longdouble` reference). |
| G3 DSR with NaN and fat tails | **tests written** | `test_dsr_on_fat_tailed_returns_matches_an_independent_formula` (10, 48, 80 trials), `test_dsr_with_nan_returns_matches_an_independent_formula_per_column` (clean, scattered, leading and almost-all NaN columns), `test_dsr_nan_returns_are_omitted_not_zero_filled`, and a power check that the fat-tailed case is sensitive to the kurtosis convention. The reference is the Bailey and Lopez de Prado formula written in the test file (no vectorbt, no `ggTrader.lab`); it agrees with the legacy `gates.dsr_check` to 1e-15 on clean data (scratch check). |
| G4 numba against Rust on degenerate series | **tests written** | `test_the_two_engines_agree_on_every_statistic_of_a_degenerate_series` (raw accessor: flat zero, constant positive, no downside, single value, all NaN, trailing NaN; five statistics) and the adapter-level sibling (guards included, never inf). |
| G5 trailing-NaN window | **tests written** | `test_max_drawdown_of_a_series_that_ends_in_nan_equals_the_explicit_path` and `test_window_stats_of_a_window_that_ends_in_nan_equal_the_explicit_reference`, both engines. |
| G6 Sortino flat-curve sibling | **test written** | `test_defaults.py::test_flat_curve_sortino_is_still_inf_upstream`. |
| G7 EWM-std, OBV ties, breakeven expectancy | **no test, by design** | A "Not relevant to ggTrader" table in the runbook (10 rows incl. master items). `test_upgrade_tooling_fixups.py` greps `src/ggTrader/engine` for each row's needle and fails if the adapter starts calling it, so the table cannot go stale. A meta-test also requires every public function in `engine/indicators.py` to appear in a test file. |

Failure messages: `tests/engine/tripwire.py::explain` states the quantity, the installed value with the installed pair, the reference value and which reference it is, the relative gap, the probable upstream change and the triage item. Used by the DSR, flat-curve (Sharpe and Sortino), Bollinger, Welford, engine-agreement and trailing-NaN tripwires. Example from vectorbt 1.1.0:

```
quantity:   Bollinger lower band (window 20, alpha 2, ddof 1), worst of 103395 cells
installed:  374.29994855180394   <- vectorbt 1.1.0 / vectorbt-rust 1.1.0
reference:  374.3   <- pandas rolling(20).std(ddof=1) on the frozen tape (gap relative 1.37e-07)
probably:   vectorbt changed its rolling standard deviation (1.1.0 had the naive running variance, 1.1.1 'numerical stability' switched to Welford)
next:       run scripts/engine_parity_evidence.py (runbook step 3b) ...
```

## 3. Mutation check of the new tripwires

Method: pytest plugin `scratchpad/p8b/mut/mutplug.py` monkeypatches the vectorbt function to the old or new behaviour in a copy of the run (`MUT=<name> pytest -p mutplug tests/engine/test_tripwires.py tests/engine/test_defaults.py`, installed vectorbt 1.2.0, Rust default). Unmutated run: 39 passed.

| Mutation (what is simulated) | Tripwires that failed | Survived? |
|---|---|---|
| `welford_old`: Bollinger bands rebuilt from a naive running sum and sum of squares, patched into both the Rust and numba BBANDS kernels | the real-tape Bollinger test and the large-offset rolling-std test | no |
| `dsr_old`: `deflated_sharpe_ratio` replaced by the 1.1.0 body (NaN zero-filled, T = all rows, excess kurtosis) | 3 fat-tail DSR (10/48/80 trials), the NaN per-column test, the NaN-omitted test | no |
| `dsr_kurtosis_only`: only the kurtosis convention flipped | 3 fat-tail DSR and the NaN per-column test | no |
| `flat_nan`: zero-volatility Sharpe and Sortino return NaN (#897) | `test_flat_curve_sharpe_is_still_inf_upstream` and the new Sortino sibling | no |
| `flat_nan_sortino_only` | the Sortino sibling alone | no |
| `engine_disagree`: numba returns NaN for a flat series, Rust inf (master with the 1.2.0 wheel) | the raw-accessor engine-agreement test (`flat_zero`); the adapter-level test stays green because the finite guard hides the disagreement, as intended | no |
| `trailing_nan_dd`: returns-path drawdown is NaN when the series ends in NaN | both drawdown tests on both engines (4 failures) | no |

Against real older releases (same tests, 1.1.0 venv): 5 DSR tripwires, the Bollinger real-tape test and the Welford test fail; see section 4.

## 4. Regression drill with the fixes in place

Tests copied to scratch with the lock re-resolved by `uv lock` for each target where noted; run with `env -i ... PYTHONPATH=src` under `systemd-run ... MemoryMax=6G -p MemorySwapMax=0`. The copy was taken while P5c was editing `ruff.toml` and `tests/engine/test_boundary.py`; three `test_boundary` tests fail in the copy for that reason **on every version including the 1.2.0 control**, so they are listed once and excluded from the version comparison.

| Installed | Result | Failures that depend on the version | Verdict |
|---|---|---|---|
| 1.2.0 (worktree, `scripts/run_tests.sh tests/engine`) | 390 passed (rust default) | none | green |
| 1.2.0 control in the copy | 392 passed, 3 failed | none; the 3 `test_boundary` artifacts only | baseline |
| 1.1.2 (lock re-resolved) | 392 passed, 3 failed | none: the same 3 `test_boundary` artifacts. **No version-string test fails any more** (P8: 3). | ACCEPT-equivalent, as in P8 |
| 1.1.1 (lock still 1.1.2) | 390 passed, 5 failed | 2 `test_lock_consistency::test_installed_version_equals_the_lock[vectorbt / vectorbt-rust]` (the venv and lock disagree; the test name says so) | no numeric failure |
| 1.1.0 (plotly 6.6 forced; lock still 1.1.2) | 371 passed, 24 failed | 3 lock tests (named); 11 `test_stats.py` DSR tests; 5 new DSR tripwires; `test_bollinger_bands_match_pandas_rolling_std_on_the_real_tape`; `test_rolling_std_is_numerically_stable_...` | REJECT as in P8, now with the Bollinger and Welford change caught by pytest |

In every row the failing tests name the thing that changed. Logs: `scratchpad/p8b/out/{ctl,v110,v111,v112}_default.log`.

## 5. Files changed

`.claude/skills/vbt-upgrade/SKILL.md`; `.github/workflows/ci.yml` (one line); `.github/workflows/vbt-upgrade-watch.yml` (pytest path, dependency delta, defaults `sed`, PR body); `pyproject.toml` (`testpaths` only); `scripts/vbt_latest.py`; `scripts/engine_parity_evidence.py` (`--out` default and `import tempfile`; the file was being rewritten by P5c at the same moment and was re-formatted with `ruff format`); `tests/scripts/test_vbt_latest.py`; `tests/engine/test_defaults.py`; `tests/engine/test_selfcheck.py`; new `tests/engine/tripwire.py`, `tests/engine/test_tripwires.py`, `tests/engine/test_upgrade_tooling_fixups.py`; this note. `src/ggTrader/engine/` was **not** edited; `.github/dependabot.yml` needed no change.

## 6. Left over, uncertain

* **Changelog entry not written.** `docs/changelog.md` is in the P5c agent's write set, so the entry below is for the lead to paste (CLAUDE.md requires one).
* `tests/engine/test_stats.py` was outside my edit list: its 11 older DSR tests keep their terse `Obtained / Expected` failure text, and they still import the legacy `ggTrader.lab.gates.dsr_check` as the reference. If P5c deletes that module they need the independent formula from `test_tripwires.py::ref_dsr`, which has no `lab` dependency and agrees with it. The new tripwires carry the readable message.
* `engine/stats.py::_std` emits `RuntimeWarning: Degrees of freedom <= 0` for a window with fewer than two observations (seen with the single-value and all-NaN cases). It is harmless (the result is NaN) but noisy in production logs; one `np.errstate` fix, not made because `src/ggTrader/engine/` was out of bounds. The new test filters that one warning.
* `scripts/engine_parity_evidence.py` keeps an `OUT_DIR` constant that nothing uses now; remove it when P5c is done with the file.
* `p5_wfo_validation.py` and `p7_live_golden.py` still default their output to the committed evidence folders (P5c's files); the runbook compensates by requiring a scratch path.
* The workflow itself still cannot be run without a GitHub runner; its shell steps changed only in the report text and the pytest path.
* The three `test_boundary.py` failures in the scratch copy come from P5c's in-flight `ruff.toml` / allow-list changes, not from this work; re-run `tests/engine` after P5c merges.

## 7. Changelog entry for the lead to paste into `docs/changelog.md`

```
## 2026-11-06 (P8b)

- **P8b: fixes for the 16 defects the rehearsal of the vectorbt upgrade runbook found, and seven new tripwires.** The defaults test no longer compares the vectorbt version string (a bump alone left it red), the selfcheck pair tests derive their cases from the installed version, a bare `pytest` collects `tests/` only (`testpaths`; CI and the watch workflow name the path), `scripts/vbt_latest.py` no longer proposes an older release after the newest was rejected, `scripts/engine_parity_evidence.py` defaults its output to a scratch file, and the `vbt-upgrade` skill gains a tape pass, a pre-release procedure, REJECT/ACCEPT PR commands, a decision for "the new vectorbt value agrees with the independent reference", real oracle paths and a "not relevant to ggTrader" table. New tests: Bollinger bands against pandas on 60 tape symbols and a large-offset rolling-std test (both would have caught the 1.1.1 Welford change), the DSR with NaN and fat-tailed returns against an independent formula, numba against Rust on degenerate series, the trailing-NaN drawdown, and the Sortino sibling of the flat-curve tripwire. Failure messages name the installed vectorbt, the reference side, the gap and the probable upstream change. Mutation check: all seven simulated definition changes are caught. Details: `research/programme/vectorbt-engine/P8b-fixups.md`.
```
