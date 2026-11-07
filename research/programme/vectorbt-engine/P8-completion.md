# P8 completion: rehearsal of the vectorbt upgrade runbook (downgrades 1.1.2 / 1.1.1 / 1.1.0, unreleased master)

Date 2026-11-06. Branch `vbt12/engine`, worktree `/home/flynn/ggTrader-vbt12`. This note and `p8-evidence/` are the only files written there. Nothing committed, staged, pushed or tagged; no Docker, crontab, Alpaca, Telegram or database action; no `.env` read; nothing placed an order.
Terms: **pair** = `vectorbt` and `vectorbt-rust` at the same version. **E1** = the baseline epoch the goldens were made in (vectorbt 1.2.0). **Tape** = the frozen daily bars `data/snapshots/ohlcv_20261106.parquet`. **Live golden** = P0's 120 recorded live-signal sessions. **DSR** = deflated Sharpe ratio. **Welford** = the numerically stable running-variance method.

## 1. Bottom line

* The pair installs cleanly on the project stack for 1.1.2 and 1.1.1. 1.1.0 installs but does not import under the project's `plotly>=7`. 1.0.0 cannot be installed with `pandas>=3.0.3` (it needs `pandas<3`). Master installs from GitHub and its Rust half builds from source in 1.5 minutes.
* **The contract tests catch the DSR definition change** (1.1.1) with 11 failing tests when 1.1.0 is installed. They **do not catch** the rolling-std Welford change (1.1.1), nor any 1.2.0 item (EWM-std, OBV ties, breakeven expectancy, trailing-NaN drawdown). Reason for the 1.2.0 items: the adapter calls none of those APIs, so a silent suite is correct. Reason for Welford: it is below the 1e-9 tolerance on the synthetic panels; only the script `engine_parity_evidence.py` on real tape shows it (Bollinger lower band differs by 5.1e-5 absolute in 1.1.0), and the runbook does not mention that script.
* The tripwire built for the next known change fires on master: `test_flat_curve_sharpe_is_still_inf_upstream` (#897, zero-volatility Sharpe and Sortino now return NaN instead of inf).
* **The runbook is not executable as written by an agent that stops at "everything green": 3 tests are red on every version bump for reasons unrelated to numbers** (1 version string in a golden, 2 tests with hard-coded 1.1.9/1.2.9). 16 runbook/tooling defects in total, listed in section 7 with exact fixes.
* Numbers: simulator and statistics goldens are bit-identical (max abs diff 0.0) across 1.1.0, 1.1.1, 1.1.2, 1.2.0 and master on both engines. The live golden is 120/120 sessions equal on all six stacks.

## 2. Candidates and how they were built

All in `scratchpad/p8/` (full path `/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/p8/`). Python 3.12.3, `uv 0.11.25`. Each venv = `uv export --frozen --no-hashes --extra test --extra dev` of the project `uv.lock` minus the two vectorbt lines (`req_base.txt`, 427 lines; pandas 3.1.1, numpy 2.5.4, numba 0.68.1, plotly 7.1.0), then the pair installed with `--no-deps`. Tests ran from a copy of the worktree (no `.git`, `data/snapshots` symlinked) with `env -i PATH HOME PYTHONPATH=src` under `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`. The copy is a snapshot taken while the P5b agent was editing; it has 331 engine tests.

| Candidate | Installed as | Upstream pin check (PyPI metadata) |
|---|---|---|
| control | `vectorbt==1.2.0` + `vectorbt-rust==1.2.0` | `numpy>=2.4.6`, `pandas>=3.0.3,<4`, `numba>=0.66`; the extra adds `nbformat` |
| 1.1.2 | PyPI pair 1.1.2 | same pins (no nbformat); Rust wheel `numpy>=2.4.6` |
| 1.1.1 | PyPI pair 1.1.1 | same |
| 1.1.0 | PyPI pair 1.1.0 | same pins. **Import fails on plotly 7.1.0** (`Bad property path: scattermapbox`, the template bug fixed by 1.1.1). Run with `plotly==6.6.0` forced, outside the project bound `plotly>=7`. |
| 1.0.0 | not installed | `pandas<3.0,>=2.0`; `uv pip install --dry-run` would downgrade pandas 3.1.1 to 2.3.3. Cannot coexist with the project stack. |
| master `c923f74c` (25 commits after v1.2.0, 2026-11-03) | `uv pip install --no-deps git+https://github.com/polakowo/vectorbt.git@c923f74c...` under `env -i`; **two variants**: (a) Python master + PyPI Rust 1.2.0 wheel, (b) Python master + Rust built from the same commit (`maturin build --release`, cargo 1.96, `CARGO_HOME` in scratch, 1.5 min) | master's `__version__` is still `"1.2.0"` and `rust/Cargo.toml` is still 1.2.0, so **version strings cannot tell master from the release** |

Pair constraint check: the pair is tied by `vectorbt-rust==<same>` in the `rust` extra, and `selfcheck.check_version_pair` compares major.minor. Every released candidate satisfies the project bounds (`numpy>=2.4.6,<2.6`, `pandas>=3.0.3,<3.2`, `numba>=0.66,<0.69`) with no resolver change, except that 1.2.0 to 1.1.x removes the 7 `nbformat` dependency packages from the lock.

## 3. Per-candidate result table

`tests/engine`, 331 tests in the control. "Benign" = failures that are not numeric (section 6).

| Candidate | Install | Selfcheck (rust, `rust_calls: 1`, `year_freq 252 days`) | `tests/engine` rust default | numba default | Failures and cause | Golden drift vs 1.2.0 | Tape parity script | Live golden | Decision I would record |
|---|---|---|---|---|---|---|---|---|---|
| control 1.2.0 | ok | pass | 331 passed | 331 passed | none | n/a | baseline | 120/120 | baseline |
| 1.1.2 | ok (literal runbook steps: `uv lock` 4 s, `uv sync` 22 s) | pass (17 s) | 3 failed, 328 passed | 3 failed, 328 passed | `test_defaults::...have_not_changed` (version string only), `test_selfcheck::...rejects_a_mismatch`, `::test_patch_difference_is_allowed` (hard-coded 1.1.9 / 1.2.9). With a correctly updated lock `test_lock_consistency` is green. No numeric failure. | 0.0 (bit-identical, both engines) | 0 differences | 120/120 | Numbers: ACCEPT-equivalent. As a candidate it is a downgrade and loses the 1.2.0 fixes, so not adopted. |
| 1.1.1 | ok | pass | 5 failed, 326 passed (the 3 above + 2 lock-consistency, because the lock was not re-resolved in this venv) | same | same as 1.1.2 | 0.0 | 0 | 120/120 | same as 1.1.2 |
| 1.1.0 (plotly 6.6.0 forced) | import fails on the project stack | pass (with plotly 6.6) | 17 failed, 314 passed | 17 failed | 6 benign (version, 3 lock incl. plotly, 2 selfcheck) + **11 DSR failures** (`test_stats.py`: `test_configured_dsr_reproduces_the_legacy_statistic` x8, `test_dsr_var_sharpe_is_252_over_t_minus_1...`, `test_dsr_uses_unbiased_moments_like_pandas`, `test_dsr_frame_returns_one_value_per_column`) | 0.0 | **Bollinger lower band: max abs 5.14e-5, max rel 1.37e-7** (others 0) | 120/120 | REJECT: cannot import on the project stack and it carries the old DSR (1.1.1 "Results change") |
| 1.0.0 | cannot install | n/a | n/a | n/a | resolver: needs `pandas<3` | n/a | n/a | n/a | REJECT (stack conflict, also pre-dates pandas 3 support) |
| master (a): Python master + Rust 1.2.0 wheel | ok | pass | 331 passed | **1 failed** | `test_flat_curve_sharpe_is_still_inf_upstream` fails under numba only (#897) | 0.0 | 0 | 120/120 | see below |
| master (b): both halves from master | ok (Rust build 1.5 min) | pass | **1 failed** | 1 failed | same test, both engines | 0.0 | 0 | 120/120 | see below |

Master decision: **ACCEPT-ready, but do not adopt an unreleased commit.** Every ggTrader number is unchanged (goldens bit-identical on both engines, tape parity 0 differences, live golden 120/120). The only red test is the designed tripwire for #897; the engine's `_nan_if_nonfinite` guard already turns inf into NaN, so the application value does not move. When 1.2.1 ships: re-run, update that test with a recorded reason (accept inf or NaN), keep the guard. Two cautions found: (1) master Python with the old Rust wheel is **engine-dependent for zero-volatility Sharpe/Sortino (numba NaN, Rust inf)**; released pairs are atomic so this cannot happen in an upgrade, but it shows why the dispatch/parity checks matter. (2) The pre-release carries the release's version string (see section 2).

## 4. Upstream release-note items against what the suite and probes show

Items are from `https://api.github.com/repos/polakowo/vectorbt/releases` (v1.1.1, v1.1.2, v1.2.0) and, for master, `compare/v1.2.0...master`. "Probe" = my raw-API check `p8-evidence/probes/known_changes_probe.py` run in each venv with `python -I` (42 quantities, `probe_matrix_auto_engine.txt`). "Adapter?" = does `src/ggTrader/engine` or any `src/` code call it (grep).

| Release | Item (as in the notes) | Reproduced by probe | Adapter uses it? | Caught by pytest? | Which test, or why not |
|---|---|---|---|---|---|
| 1.1.1 | Deflated Sharpe: non-excess kurtosis, omit NaNs ("Results change") | yes. Fat-tailed 500 returns: 0.0040192 (1.1.0) to 0.0040230 (1.1.1+); with NaN returns 0.005749 to 0.005919; S2's seeded fingerprint 0.019057 to 0.019276 | yes (`stats.dsr`, the gate) | **yes, 11 tests on 1.1.0** | `test_stats.py::test_configured_dsr_reproduces_the_legacy_statistic[48-0..80-3]`: `assert 0.03223756645567962 == 0.03226673013133988 +- 3.2e-11`; also `...252_over_t_minus_1...` (0.2171 vs 0.2179) and the frame test (0.000472 vs 0.000479) |
| 1.1.1 | Welford rolling std, "numerical stability" | yes. Offset-1e6 series: sum 9.098 (1.1.0, last bar exactly 0) vs 3.713; S2 fingerprint Bollinger sums differ 1.7e-13 relative | indirectly (BBANDS) | **no** (stays below 1e-9 on synthetic panels) | `test_indicators.py` passes on 1.1.0. Real tape: Bollinger lower differs 5.14e-5 absolute (1.37e-7 relative) in 1.1.0, 1.1e-12 in 1.1.1+, seen only by `scripts/engine_parity_evidence.py` |
| 1.1.1 | `expanding_mean/std` NaN when `minp` > length | not reproduced (same value in all versions for my input) | no | n/a | out of surface |
| 1.1.1 | out-of-bounds read on empty input in expanding min/max, drawdowns | not reproducible through the public accessor | no | n/a | out of surface |
| 1.1.1 | Plotly 7 template (Mapbox entries) | yes, `import vectorbt` raises on plotly 7.1.0 with 1.1.0 | import | yes, as a collection error (exit 4) | whole `tests/engine` fails to start; message `Bad property path: scattermapbox` is clear |
| 1.1.2 | `cond_value_at_risk` ignores NaN returns | not reproduced (identical in 1.1.0 to master for my input) | no | n/a | out of surface |
| 1.1.2 | `is_sorted_nb` NaN parity, NaN trade-plot markers, datetime64 conversion, Best/Worst Trade doc | `is_sorted_nb` identical in my probe | no | n/a | out of surface |
| 1.2.0 | EWM standard deviation ddof | yes. `MSTD.run(window=20, ewm=True)`: sum 659.99 to 643.10 (-2.6 %), last 1.0822 to 1.0548 | **no** (Bollinger is `ewm=False`; RSI/MACD/MA use EWM mean, ATR is `ewm=False`) | no | nothing exercises it; correct silence |
| 1.2.0 | OBV on unchanged closes | yes. 35-bar tie series: last 250 to 60; 400-bar panel: sum 160,944,016 to -9,651,788 | no | no | correct silence |
| 1.2.0 | breakeven trades in expectancy | yes. `pf.trades.expectancy()` -2.6727 to -1.1263 | no (`wfo_v2/gates.py` computes its own return/trades) | no | correct silence |
| 1.2.0 | drawdowns when a series ends in NaN | yes. `value.vbt.drawdowns.max_drawdown()` NaN to -0.3065; drawdown records 0 to 1. The returns-accessor `max_drawdown` the adapter uses is identical (-0.3065 in 1.1.0, 1.1.2 and 1.2.0) | **no** (returns path only) | no | no test feeds a trailing-NaN window to the adapter: gap G5 |
| 1.2.0 | ulcer index / Martin ratio added | present only in 1.2.0+ | no | n/a | n/a |
| 1.2.0 | `preserve selected group membership`, `unsorted unstack labels` | not probed (base-accessor internals) | no | n/a | out of surface |
| master | zero-volatility Sharpe/Sortino return NaN or signed inf (#897) | yes: flat zero returns inf to NaN (numba and Rust-from-master); Rust 1.2.0 wheel still inf | yes (`stats.sharpe`/`sortino` guard) | **yes** | `test_defaults.py::test_flat_curve_sharpe_is_still_inf_upstream`: `assert np.False_ where np.False_ = isinf(nan)` |
| master | unsigned true-range underflow (#900), narrow-int `diff` overflow (#899) | yes in numba (uint32 ATR sum 4.2e21 to 25,396; int8 diff 56 to -200) | no (Rust path already rejects non-float64: `Rust engine requires float64 arrays and uint32 cannot be safely cast`; `signals.py` coerces) | no | correct silence |
| master | O(n) rolling min/max (#901), `data` fixes, `indicator` column selection | rolling min/max identical | no | no | performance only |

S2's prototype claim ("16 quantities: 13 identical, 2 at 1e-12, 1 DSR") reproduces exactly: S2's `contract_fingerprint.py` (15 numeric quantities plus the versions entry) gives 1.1.0 versus 1.2.0 = 3 differences (Bollinger default and ddof=1 lower at 1.7e-13 relative, DSR 0.019057 to 0.019276), and 1.1.1, 1.1.2, master (both variants) = 0 differences (`p8-evidence/s2-fingerprint/`). What the real pytest suite adds: the DSR is caught; the two 1e-13 differences are below the suite's tolerance and are not reported (on real tape the same Welford change is 5e-5, which only the parity script reports).

## 5. Do the failures name the quantity and the old and new values?

Decision need: an agent must be able to choose ACCEPT / RE-BASELINE / REJECT without reading vectorbt source.

| Failure | What the agent sees | Enough? | Missing |
|---|---|---|---|
| DSR (11 tests) | test name says the quantity; `Obtained: 0.0322375..`, `Expected: 0.0322667.. +- 3.2e-11` | mostly. Quantity and both numbers are present, but not which side is vectorbt versus the legacy reference, not the vectorbt versions, not the relative gap (9e-4 to 4 %), not the cause | add a message naming `vectorbt <installed> vs legacy gates.dsr_check`, the relative gap and "see vbt-upgrade triage item 2"; the cause (kurtosis convention, NaN omission) lives only in the release notes the agent was told to read in step 1 |
| Defaults tripwire | `vectorbt defaults changed: /vectorbt_version: '1.2.0' -> '1.1.2'` | clear but **a false alarm on every bump** (R1). The real default checks produced no failure for any candidate, so no default changed in 1.1.0 to master | version should not be compared |
| Flat curve (master) | `assert np.False_ ... isinf(nan)` | an agent who has read the runbook table row understands; a fresh reader would not | message should say "flat-curve Sharpe is now nan (was inf); upstream #897 fixed it; keep the guard; update this test with a reason" |
| Selfcheck pair tests | `DID NOT RAISE EngineError`; `vectorbt 1.1.1 and vectorbt-rust 1.2.9 differ in major.minor` | misleading: looks like a real pair mismatch but is a hard-coded literal (R2) | |
| Lock consistency | `venv is stale: run uv sync --frozen`, `assert '1.1.1' == '1.2.0'` | clear and correct | |
| Plotly template (1.1.0) | pytest exit 4 and `Bad property path: scattermapbox` | clear | |

## 6. Gaps: known upstream changes not caught (proposed contract tests, not written)

| ID | Gap | Evidence | Proposed test |
|---|---|---|---|
| G1 (medium) | Rolling-std/Bollinger accuracy on real data is checked by a script, not a test | 1.1.0 passes all pytest; tape parity shows 5.14e-5 abs / 1.37e-7 rel on `bb_lower_20_2` over 284,659 cells | `tests/engine/test_indicators.py`: Bollinger lower/upper and SMA on 60 tape symbols against pandas `rolling(20).std(ddof=1)` at `rtol=1e-9` after each symbol's burn-in, skipped when the tape is absent (as the P3 tests already do). Fails on 1.1.0 (1.4e-7). |
| G2 (low) | Numerical stability of rolling std | offset-1e6, sd-1e-2 series: 1.1.0 sum 9.098 with exact zeros, 1.1.1+ 3.713 | one synthetic test with a large offset against a `np.longdouble` reference; protects against a regression to the naive formula |
| G3 (medium) | DSR with NaN returns and with fat tails are not isolated | every DSR test uses clean `normal(0.0005, 0.01)`; probe: NaN case 0.005749 vs 0.005919, fat tail 0.004019 vs 0.004023 | add `stats.dsr` on a Student-t(3) series, and on a series with NaN, against the legacy formula |
| G4 (medium) | Engines can disagree on degenerate statistics and nothing asserts it | master Python + Rust 1.2.0 wheel: flat Sharpe numba NaN, Rust inf; `test_numba_and_rust_agree_on_every_scenario` has no degenerate input | NaN-aware numba == Rust over the stats adapter for flat zero, constant positive, all-NaN, single-value and trailing-NaN return series |
| G5 (low) | Trailing-NaN window in `max_drawdown`/`window_stats` | 1.2.0 changed the value-path result (NaN to -0.3065) while the returns path is identical; no test pins the returns path | `stats.max_drawdown` and `window_stats` on a window ending in NaN equal an explicit reference |
| G6 (low) | Sortino sibling of the flat-curve tripwire | master changes flat Sortino inf to NaN too; only Sharpe has a tripwire (same test fired) | `test_flat_curve_sortino_is_still_inf_upstream` |
| G7 (info) | 1.2.0 EWM-std, OBV ties, breakeven expectancy | reproduced; not in adapter surface | no test now. Rule for the runbook: any new adapter function must ship with an independent-reference test; add a meta-test that every public function in `engine/indicators.py` has one |

## 7. Runbook, workflow and tooling defects, with fixes (patch list for the lead)

Ranked. "Verified" = observed in this drill.

| # | Where | Defect (verified) | Fix |
|---|---|---|---|
| R1 | `tests/engine/test_defaults.py`, `golden/vbt_defaults.json` | `vectorbt_version` is part of the compared payload, so **every bump fails this test with `/vectorbt_version` only**; the runbook then forbids regenerating the golden without a reason and a changelog line, and its ACCEPT rule needs "everything green". | In `test_vectorbt_defaults_have_not_changed` drop the `vectorbt_version` key from both sides before `_diff` (keep it in the probe output). Runbook table row: "a version-only line is not a default change". |
| R2 | `tests/engine/test_selfcheck.py:80-97` | `test_major_minor_pair_check_rejects_a_mismatch` hard-codes `1.1.9`, `test_patch_difference_is_allowed` hard-codes `1.2.9`. They fail on 1.1.2, 1.1.1, 1.1.0 and will fail on 1.3.0. The runbook row for `test_selfcheck` ("Rust is not dispatching") misdirects. | Derive both: `maj, mnr, _ = real()["vectorbt"].split(".")`; mismatch = `f"{maj}.{int(mnr)+1}.0"`, patch = `f"{maj}.{mnr}.9"`. |
| R3 | SKILL step 2 | `git diff -U0 -- uv.lock \| grep -E '^[+-](name\|version) '` prints `-version = "1.2.0"` / `+version = "1.1.2"` twice with no package name (the unchanged `name` line is hidden by `-U0`). | Use uv's own delta: `uv lock 2>&1 \| tee lock.log; grep -E '^(Added\|Removed\|Updated) ' lock.log` (real output: `Updated vectorbt v1.2.0 -> v1.1.2`, `Updated vectorbt-rust ...`, `Removed nbformat v5.11.1`, 6 more). |
| R4 | SKILL step 5 | Path `research/audits/engine-migration/_*.json` does not exist. The artifacts are `research/programme/vectorbt-engine/p5-evidence/_p5_compare_*.json` and `p7-evidence/p7_live_golden.json`. It also says a read-only database session is needed; the drivers read only the frozen Parquet (`p7_live_golden.py` takes 40 s). | Replace with the real paths and the commands `scripts/p5_wfo_validation.py`, `scripts/p5_equity_parity.py`, `scripts/p7_live_golden.py --out /tmp/...`; say the tape (git-ignored) is the only thing a GitHub runner lacks. |
| R5 | SKILL steps 3 and 5 | `scripts/engine_parity_evidence.py` (15 s, tape only) is the only check that showed the Welford change on real data and the runbook never runs it. Its default `--out` overwrites the committed `p2-evidence/p2_engine_parity_20261106.json`, and `p7_live_golden.py`'s default `--out` overwrites `p7-evidence/p7_live_golden.json`, which breaks the "edit nothing else" red line. | Add step "3b tape pass": `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python scripts/engine_parity_evidence.py --symbols 150 --out /tmp/parity_X.json`, then diff against the committed JSON (any non-zero change in `indicators` or `voters` is a finding). Always pass `--out`. |
| R6 | SKILL step 1 | Lists `MSTD` under `engine/indicators.py`; the adapter has no MSTD (only the defaults probe uses it). Nothing says which release-note items cannot reach us. | Remove MSTD; add the "out of surface" list from section 4 (EWM-std, OBV, expectancy/profit_factor, drawdowns accessor, ulcer, VaR) so an agent can mark an item "not reachable" with a grep. |
| R7 | SKILL | No procedure for an unreleased commit (the stated goal of P8). | Add "Pre-release look-ahead": `gh api repos/polakowo/vectorbt/compare/v<LOCKED>...master --jq '.commits[].commit.message'`; scratch venv only; `env -i PATH HOME UV_CACHE_DIR uv pip install --no-deps git+https://github.com/polakowo/vectorbt.git@<sha>`; Rust from source: `maturin build --release -i <venv python>` in `rust/` (1.5 min, `CARGO_HOME` in scratch). Record the **sha**, because master still reports 1.2.0 in `vectorbt.__version__`, the lock check, the provenance block and the defaults golden. Do not edit the pin for a pre-release. |
| R8 | SKILL step 6 | REJECT/ACCEPT never say what to do with the draft PR and branch `vbt-upgrade/X.Y.Z` the workflow opened; ACCEPT has no "mark ready / merge" step. | REJECT: `gh pr close <n> --delete-branch --comment "<reason>"`. ACCEPT: `gh pr checkout <n>`, apply fixes, `gh pr ready`. |
| R9 | SKILL step 6 | `docs/vbt_upgrade_rejected.txt` and `docs/vbt_upgrade_log.md` do not exist. The log is "create on first use"; the rejected file is only "add". | Say "create". `vbt_latest.py` already tolerates a missing file (verified). |
| R10 | SKILL step 7 | Says the `ggtrader-deploy` pre-flight "builds the image, runs the engine self-check as root and as appuser". It does not. The `Dockerfile` does (root lines 45-46, appuser lines 58-59) and `docker-build.yml:62-63` smoke-runs it; the deploy skill only runs `selfcheck --engine rust` in the pulled container. | Reword. |
| R11 | SKILL step 1 | `LOCKED=$(...)` is assigned and never used. | Use it in the compare URL (R7) or delete. |
| R12 | `scripts/vbt_latest.py` | After the newest release is REJECTED, the next run proposes an older one. Verified: lock 1.1.1, rejected file `1.2.0`, output `version=1.1.2`. This contradicts "a burst within 14 days is evaluated by its newest member only". | In `pick_version` skip any version lower than the highest rejected one (unless `force`). |
| R13 | `.github/workflows/vbt-upgrade-watch.yml:129` and `ci.yml:60` | Bare `pytest -q` collects `research/`. Verified in the copy: 1 collection error, `research/programme/vectorbt-engine/evidence/vbt-spec/s2/test_boundary_proto.py` (untracked today), so the watch job would always report `FULL_RED` and CI turns red the day that folder is committed. | Add `testpaths = ["tests"]` to `[tool.pytest.ini_options]` (or call `pytest tests/`). |
| R14 | SKILL step 3 table | No row for "version-only" or "hard-coded literal" failures (R1, R2) and no row for a collection error from a plotly floor (1.1.0). | Add rows. |
| R15 | workflow report step | `sed -n '/vectorbt defaults changed:/,/^$/p'` prints only the version line on every release until R1 is fixed. | Resolved by R1. |
| R16 | SKILL step 4 | Decision tree has no entry for "contract test fails and the NEW vectorbt value matches the legacy/independent reference" (the DSR case in the forward direction: an upstream bug fix). It would be sent to RE-BASELINE. | Add: "if the independent reference agrees with the new vectorbt value, record ACCEPT and update the expectation with a reason; RE-BASELINE only when new vectorbt departs from the reference." |

What worked literally, unchanged: step 1 `gh api repos/polakowo/vectorbt/releases` and `releases/tags/v1.1.2` (the tag form `v<X.Y.Z>` is right; bare `1.1.2` gives 404); step 2 `sed` bump (the workflow form), `uv lock` (4 s), `uv sync --frozen --extra test --extra dev` (22 s); step 3 `selfcheck`, both `run_tests.sh tests/engine` runs, the `-W error` import (4 s), `ruff check .`, `ruff format --check .` (clean); `GGT_VBT_ENGINE=numba` before the helper took effect (99 s versus 148 s, 2 warnings versus 20); `--update-golden` / `--golden-reason`, `test_goldens_are_changelogged`, `CONTRACT_VERSION`, `BASELINE_EPOCH`, `provenance.compare` and `docs/superpowers/specs/...` sections 6 and 7 all exist under the names the runbook gives.

### `scripts/vbt_latest.py` against live PyPI

Behaves as specified. Lock 1.2.0: `version=` (nothing newer, PyPI and GitHub both show 1.2.0 as latest). `--force 1.2.0`: `version=1.2.0`; `--force 9.9.9`: `vbt_latest failed: 9.9.9 is not on PyPI`, exit 1. With the lock rewritten to 1.1.1: cooldown 3 gives 1.2.0 (uploaded 2026-10-19, 18 days ago); cooldown 18 gives 1.2.0; cooldown 20 gives 1.1.2 (1.2.0 inside the window, so the next newest); cooldown 30 gives nothing; rejected file `1.2.0` gives 1.1.2 (R12); a missing rejected file is tolerated; the workflow's empty `--force ""` is fine. `--help` works. The 3-day rule is measured from the earliest non-yanked upload of the vectorbt files; the Rust upload is only checked for existence, not for age or yank. `tests/scripts/test_vbt_latest.py` (12 tests) ran in the full suite.

## 8. Numeric drift on the golden panels, tape and live signals

| Check | Result |
|---|---|
| `scenarios.run_all()` (simulation equity, entry/exit indices, trade counts, mini-WFO) dumped per stack and engine, diffed against 1.2.0 on the same engine | max abs 0.0, max rel 0.0, 0 discrete differences for 1.1.0, 1.1.1, 1.1.2, master (a) and (b), on both numba and Rust (`golden-drift/golden_drift_vs_control.txt`). The goldens do not exercise DSR, so they could not have caught it. |
| numba versus Rust within 1.2.0 | max abs 1.1e-16, max rel 1.0e-14, 1 scalar differing (`test_max_drawdown`, last bit); simulators equal |
| `scripts/engine_parity_evidence.py --symbols 150` (150 symbols x 1,929 bars of tape) | 1.1.1, 1.1.2, master (a), master (b): identical to 1.2.0 in every field. 1.1.0: only `bb_lower_20_2` (max abs 5.14e-5, max rel 1.37e-7, 284,659 cells). Voter flips: none. |
| `scripts/p7_live_golden.py` (120 sessions, ~40 s each) | 120/120 equal on 1.1.0, 1.1.1, 1.1.2, control, master (a), (b): even the 1.1.0 Bollinger error flips no live signal |
| Probe, rust versus numba on 1.2.0 | 3 of 42 differ, all dtype/feature guards: Rust refuses uint32 and int8 (`Rust engine requires float64 arrays`) and has no `ulcer_index` |

## 9. Timing of the drill (one agent, this machine, 4 cores, other jobs running)

| Step | Time |
|---|---|
| Reading runbook, S2, P2 notes, tooling | about 5 min |
| Building 6 venvs (uv cache warm) | about 5 s each |
| Selfcheck | 11-17 s |
| `tests/engine` (331 tests) | 2 min 15 s (rust default), 1 min 35 s to 2 min (numba) |
| Master Rust build from source | 1.5 min |
| Literal runbook run on 1.1.2 (lock, sync, selfcheck, 2 engine suites, import, ruff, full suite) | 17 s + 22 s + 4 s + 148 s + 99 s + 5 s + 751 s = about 17 min, the full suite being 12.5 min |
| Probes, golden drift, parity script, live golden | about 8 min for all stacks |
| Whole drill, 16:27 to 17:14 | about 47 min wall, including writing this note and the 10-minute control full-suite run |

An agent doing a real upgrade with this runbook should budget about 20 minutes of machine time (the full suite dominates) and 10 to 15 minutes of reading and triage when anything is red; today it would also spend time on R1 and R2 before it could see a clean result.

## 10. Not done, and uncertain

* The full suite in the scratch copy has failures unrelated to vectorbt (the copy has no `.git` and lacks untracked data): section 11 compares it with the 1.2.0 control run in the same copy. Only 1.1.2 was run through the full suite; 1.1.1, 1.1.0 and master were run through `tests/engine` and the evidence scripts only.
* The actual GitHub workflow was not run (no runner); its shell steps were run individually.
* Rust-from-master was built once with `maturin build --release`; I did not run vectorbt's own test suite or benchmark the O(n) rolling min/max.
* "Reproduced by probe: no" for three 1.1.1/1.1.2 items means my input did not trigger the fix, not that the fix is absent.
* Master was one commit (`c923f74c`, 2026-11-03). The real 1.2.1 may differ.
* 1.0.0 was verified only by a resolver dry run.
* The numbers above are from a snapshot of the worktree (the P5b agent was editing at the time); a re-run after its merge may differ by its test changes.

## 11. Full-suite comparison

Same copy, same cap, whole `tests/` suite (`scripts/run_tests.sh`), 1.2.0 control venv versus the literal 1.1.2 run (fresh lock):

| Run | Result | Time |
|---|---|---|
| control 1.2.0 | 13 failed, 2,966 passed, 57 errors, 30 deselected | 10 min 30 s |
| 1.1.2, runbook followed literally (lock re-resolved) | 13 failed, 2,966 passed, 57 errors, 30 deselected | 12 min 27 s |

The two lists are the same except for the vectorbt-related lines. Outside `tests/engine` neither run has a vectorbt-attributable failure: the 57 errors are `tests/lab/test_event_timing_audit.py` (needs untracked data under `data/external/event_timing_audit/`, not in the copy), and the failures are `test_intraday_probe.py` (6), `test_research_reorg.py` (2), `test_event_timing_audit.py` (1) and `test_ruff_excludes_research.py`, all environment-bound in a copy without `.git`. Three tests fail only in the control (`test_run_capped.py` x2, `test_ruff_excludes_research.py::test_repo_ruff_config_skips_bad_research_file_but_lints_others`) and pass in the 1.1.2 run: they are load-sensitive, not version-sensitive. The 1.1.2 run adds exactly the 3 engine failures of R1 and R2. Consequence for the runbook: step 3's "whole suite" has a 70-line environment baseline in a bare copy; the runbook should say to compare the failure set with the same command on the current lock (a worktree with `.git` and the untracked data will show fewer), not to expect zero.

## Files

`p8-evidence/`: `probes/` (probe script, comparers, JSON per stack and engine, matrix text), `s2-fingerprint/`, `golden-drift/`, `parity/` (tape parity and live-golden JSON per stack), `test-logs/` (summaries per run, full logs for 1.1.0, 1.1.1, master (b), the literal 1.1.2 runbook run and the Rust build), `runsuite.sh`, `req_base_from_uv_lock.txt`.
