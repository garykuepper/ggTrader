---
name: vbt-upgrade
description: Take a new vectorbt (+ vectorbt-rust) release into ggTrader - read the release notes, bump the pair, run the contract / defaults / golden / dispatch suites on both engines, triage every difference and decide ACCEPT, RE-BASELINE or REJECT, then record it. Also covers looking ahead at an unreleased commit. Use for a vbt-upgrade pull request or issue from the upgrade-watch workflow, "new vectorbt version", "bump vectorbt", "upgrade the engine", "try vectorbt master", or any change to the vectorbt[rust] pin.
---

# vectorbt upgrade runbook

ggTrader's whole vectorbt surface is the `ggTrader.engine` package. An upgrade is one pin change,
a test run, a diff report and a decision. vectorbt is contributor-driven, minors raise
dependency floors and change numbers without a deprecation cycle (1.1.0 dropped pandas 2; 1.1.1
changed the deflated Sharpe and the rolling std; 1.2.0 changed EWM-std), so treat **every release
as result-changing until the suites say otherwise**.

Where things live: `src/ggTrader/engine/` (adapter), `tests/engine/` (contract, defaults,
golden, dispatch and tripwire suites), `.github/workflows/vbt-upgrade-watch.yml` (daily detect +
draft PR or issue), `src/ggTrader/engine/config.py` (`CONTRACT_VERSION`, `BASELINE_EPOCH`), spec
`docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md` sections 6 and 7. This runbook was
rehearsed on 1.1.0, 1.1.1, 1.1.2 and unreleased master (`research/programme/vectorbt-engine/P8-completion.md`,
fixes in `P8b-fixups.md`).

## Red lines

- Never widen a version cap, a bound or a tolerance to make CI pass.
- Never regenerate a golden without a written reason (`--golden-reason`), and never without a
  `docs/changelog.md` line containing that exact reason (`test_goldens_are_changelogged`).
- For a pure upgrade, edit nothing outside `src/ggTrader/engine/`, `tests/engine/`,
  `pyproject.toml`, `uv.lock`, the changelog and the upgrade log. Evidence scripts write to a
  scratch path (`--out` / `--out-dir`), never over a committed result.
- Never edit the crontab, and never place orders. Once the owner has re-enabled cron, never
  deploy in the 12:30-13:15 PT window. Promotion to the trader is only through `ggtrader-deploy`.
- The pair is atomic: `vectorbt[rust]==X` and `vectorbt-rust==X` move together or not at all.
- A NO-GO verdict never reopens because a re-baseline looks better.

## 0. Preconditions

Read `docs/next_steps.md` for any hold the owner placed. Work in a worktree or branch, not on
the production checkout. Run everything with `GGT_PYTHON=<venv>/bin/python` and
`scripts/run_tests.sh` (8 GB cgroup, no swap). Exit 137 means a hang or runaway, not a reason to
raise the cap. Budget about 20 minutes of machine time (the FULL suite, `scripts/run_tests.sh --full`, is
10 to 12 minutes of it). The upgrade gate is the FULL run: the default `scripts/run_tests.sh` is the fast
core and leaves out the numba engine, the tripwire tests and the real-tape oracles. `--full` also runs the
study half (the tests inside each study's `code/tests` folder and each category's `_lib/tests` folder, some of
which exercise engine and paper code); `scripts/run_tests.sh --studies -q` runs that tier on its own, and it must pass too.

## 1. Read the release notes for every version between the lock and the target

```bash
LOCKED=$(.venv/bin/python -c "import importlib.metadata as m; print(m.version('vectorbt'))")
gh api repos/polakowo/vectorbt/releases --paginate --jq '.[] | [.tag_name, .published_at] | @tsv'
gh api repos/polakowo/vectorbt/releases/tags/v<VERSION> --jq .body     # tags are v<MAJOR.MINOR.PATCH>
gh api repos/polakowo/vectorbt/compare/v${LOCKED}...v<VERSION> --jq '.commits[].commit.message'
```

Make a table: *release note item | adapter symbol it touches | expected effect*. Map items to
`engine/indicators.py` (RSI, MA, BBANDS, MACD, ATR), `engine/signals.py` (crossed_above/
below, fshift), `engine/portfolio.py` (from_signals, from_orders, trades.records),
`engine/stats.py` (sharpe, sortino, annualized, annualized_volatility, max_drawdown,
deflated_sharpe_ratio on the **returns accessor**) and `engine/windows.py` (range_split). Also
note any new floor on numpy, pandas, numba, python or plotly (1.1.0 raised them; a plotly floor
shows up as a collection error, see the failure table). Skipping a release needs a recorded reason.

### Not relevant to ggTrader: do not investigate these again

Each row is an upstream change that the rehearsal reproduced in a raw probe, in an API the adapter
never calls (`tests/engine/test_upgrade_tooling_fixups.py` greps `src/ggTrader/engine` for the
needle and fails if the adapter ever starts calling it, so this table cannot go stale quietly).
Mark such an item "not reachable" in the table of step 1; write **no** test for it. If a needle
turns up in `engine/`, the row is wrong: delete it and add an independent-reference test first.

| Release | Upstream change | Needle the adapter would call | Why it does not reach us |
|---|---|---|---|
| 1.2.0 | EWM standard deviation uses ddof | `MSTD` | Bollinger is `ewm=False` (hard-coded in `bbands`); only EWM *means* (RSI, MA, MACD) are used |
| 1.2.0 | OBV on unchanged closes | `OBV` | no OBV in any preset |
| 1.2.0 | breakeven trades counted in expectancy | `expectancy` | `wfo/gates.py` computes its own trades and returns |
| 1.2.0 | drawdowns when a value series ends in NaN | `\.drawdowns` | the adapter uses the **returns** accessor `max_drawdown` (pinned by `test_tripwires.py`) |
| 1.2.0 | ulcer index and Martin ratio added | `ulcer\|martin` | new metrics, not called |
| 1.1.2 | `cond_value_at_risk` ignores NaN returns | `cond_value_at_risk` | no VaR statistic |
| 1.1.2 | `is_sorted_nb` NaN parity, plot markers, datetime64 conversion | `is_sorted` | internals and plotting |
| 1.1.1 | `expanding_mean/std` NaN when `minp` exceeds the length; empty-input min/max | `expanding_` | no expanding window |
| master | O(n) rolling min/max (#901) | `rolling_min\|rolling_max` | performance only, results identical |
| master | unsigned true-range underflow (#900), narrow-int `diff` overflow (#899) | `uint32\|int8` | the Rust path rejects non-float64 and `signals.py` coerces |

### Pre-release look-ahead (an unreleased commit, "try master")

Never edit the pin or `uv.lock` for a pre-release, and never put one in the project venv. Use a
scratch venv outside the repo:

```bash
SCRATCH=$(mktemp -d)                                    # outside the repository
gh api repos/polakowo/vectorbt/compare/v${LOCKED}...master --jq '.commits[] | [.sha[:8], .commit.message] | @tsv'
SHA=<full sha you chose>
uv export --frozen --no-hashes --no-emit-project --extra test --extra dev \
  | grep -vE '^vectorbt(-rust)?==' > "$SCRATCH/req_base.txt"
uv venv --python 3.12 "$SCRATCH/venv"
uv pip install --python "$SCRATCH/venv/bin/python" -r "$SCRATCH/req_base.txt"
env -i PATH="$PATH" HOME="$HOME" UV_CACHE_DIR="$SCRATCH/uvcache" \
  uv pip install --python "$SCRATCH/venv/bin/python" --no-deps "git+https://github.com/polakowo/vectorbt.git@${SHA}"
uv pip install --python "$SCRATCH/venv/bin/python" --no-deps vectorbt-rust==${LOCKED}   # variant (a): old Rust wheel
```

Variant (b) builds the Rust half from the same commit (1.5 minutes): clone the commit, then in
`rust/` run `maturin build --release -i "$SCRATCH/venv/bin/python" --out "$SCRATCH/wheel"` with
`CARGO_HOME` in the scratch directory, and install the wheel with `--no-deps`. Run variant (a) and
(b); a result that differs between them (the rehearsal saw numba NaN against Rust inf for a flat
curve) is the finding, because released pairs are atomic but master is not.

Run the suites of step 3 from a **copy** of the worktree with `PYTHONPATH=src` and the scratch
python (the tests import the package from `src/`). **Record the sha**: master still reports the
release's `__version__` (1.2.0 after v1.2.0), so the version string, the lock check, the
provenance block and the defaults golden cannot tell master from the release. A pre-release
verdict is advice for the day the release ships ("ACCEPT-ready, re-run on the tag"), never an
adoption.

## 2. Bump the pair and re-lock

Edit only these two lines in `pyproject.toml` (and a bound, if the release raised a floor):

```
"vectorbt[rust]==X.Y.Z",
"vectorbt-rust==X.Y.Z",
```

```bash
uv lock 2>&1 | tee lock.log               # resolves; the dependency delta is in the log
grep -E '^(Added|Removed|Updated) ' lock.log   # e.g. "Updated vectorbt v1.2.0 -> v1.1.2", "Removed nbformat v5.11.1"
uv sync --frozen --extra test --extra dev
```

(`git diff -U0 -- uv.lock` hides the package names; use uv's own delta.) If `uv lock` cannot
resolve, name the conflicting package from the resolver message and stop (open an issue). Do not
loosen an engine bound to force it; a transitive cap that blocks the engine stack is removed or
replaced, as `pandas-ta` was. A downgrade drops the packages the newer release added (1.2.0 to
1.1.x removes `nbformat` and six more); that is expected.

## 3. Run the suites, on both engines

```bash
.venv/bin/python -m ggTrader.engine.selfcheck                      # Rust really dispatches
scripts/run_tests.sh tests/engine -q                               # L0-L3, both engines inside
GGT_VBT_ENGINE=numba scripts/run_tests.sh tests/engine -q          # numba as the default engine
scripts/run_tests.sh --full                                        # whole suite (the gate)
scripts/run_tests.sh --studies -q                                  # the study tier alone (also run by --full)
.venv/bin/python -W error::DeprecationWarning -W error::FutureWarning -c "import ggTrader.engine"
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

The whole suite is not expected to be green in every checkout: a copy without `.git` or without
the untracked data (`data/external/...`) fails 70 or so unrelated tests. Compare the failure set
with the same command on the **current** lock and judge only the difference.

### 3b. Tape pass (real data; the only check that saw the 1.1.0 Bollinger error before a test did)

```bash
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
  .venv/bin/python scripts/engine_parity_evidence.py --symbols 150 --out "$SCRATCH/parity_new.json"
```

Compare with the committed `research/programme/vectorbt-engine/p2-evidence/p2_engine_parity_20261106.json`:
any non-zero change in `indicators` or a flipped cell in `voters` is a finding. Always pass
`--out` (the default is a scratch file now, but a path under `research/` would overwrite evidence).
`tests/engine/test_tripwires.py::test_bollinger_bands_match_pandas_rolling_std_on_the_real_tape`
repeats the Bollinger part as a test whenever the tape exists.

### What each failure means

| Failing test | Meaning |
|---|---|
| `test_defaults.py::test_vectorbt_defaults_have_not_changed` | An upstream default changed. The failure lists each one. Decide per default whether an engine preset must now set an explicit flag. The version string is **not** compared: a bump alone leaves this green. |
| `test_defaults.py::test_flat_curve_sharpe_is_still_inf_upstream`, `test_flat_curve_sortino_is_still_inf_upstream` | Upstream fixed zero-volatility ratios (#897, NaN instead of inf). The adapter guard already maps non-finite to NaN, so no application value moves. Keep the guard, accept NaN in the test with a recorded reason. |
| `test_tripwires.py::test_dsr_*` (and `test_stats.py` DSR tests) | The deflated Sharpe definition changed (1.1.1 did: non-excess kurtosis, NaN omitted). The message names the installed vectorbt, the independent formula, the relative gap. See triage items 2 and 2b. |
| `test_tripwires.py::test_bollinger_bands_match_pandas_rolling_std_on_the_real_tape`, `test_rolling_std_is_numerically_stable_...` | The rolling std changed (1.1.1 Welford). Run step 3b; a signal flip is a RE-BASELINE, a change below 1e-9 without a flip is an ACCEPT with a reason. Skipped (with a reason) where the tape is absent. |
| `test_tripwires.py::test_the_two_engines_agree_...` | numba and Rust disagree on a degenerate series. Triage item 4. The `test_the_adapter_gives_the_same_answer_...` sibling stays green while the finite guard hides it: that is the guard working. |
| `test_tripwires.py::test_max_drawdown_of_a_series_that_ends_in_nan_...`, `test_window_stats_of_a_window_that_ends_in_nan_...` | Trailing-NaN handling in the returns path moved. |
| `test_indicators.py`, `test_stats.py` (independent references) | A definition changed. See triage below. |
| `test_golden.py` | Equity, trades or fold winners moved. Compare numba against Rust first. |
| `test_selfcheck.py` | Rust is not dispatching, or the pair differs in major.minor. (The pair tests derive their cases from the installed version; a failure is real.) |
| `test_warnings.py` | The adapter now triggers a pandas/vectorbt deprecation; fix the call, not the filter. |
| `test_lock_consistency.py` | The venv is stale or pyproject and the lock disagree. `uv sync --frozen`; after a downgrade in a scratch venv re-run `uv lock` first. |
| `test_boundary.py` | Something outside `engine/` now imports vectorbt. |
| A failure whose only difference is a version string or a hard-coded version literal | A test bug, not a result. Fix the test (derive from the installed version); never regenerate a golden for it. |
| Collection error, exit code 4, before any test runs (e.g. `Bad property path: scattermapbox`) | The release does not import on the project stack: a dependency floor or ceiling moved (1.1.0 under `plotly>=7`). REJECT unless a bound can move without breaking the pins in `test_lock_consistency.py`. |

## 4. Triage - the decision tree

1. **Everything green, defaults unchanged, goldens unchanged:** ACCEPT.
2. **Contract test failed (an indicator or statistic changed):** read the upstream change and ask
   which side the independent reference supports.
   - 2a. The new vectorbt value **departs** from the independent reference. If the old definition
     is recoverable through an explicit flag, set that flag in the preset and keep the epoch
     (document it in the changelog). If it is not recoverable, the number is a new definition:
     RE-BASELINE, owner decision.
   - 2b. The new vectorbt value **agrees** with the independent reference and it was the old
     expectation that was wrong (an upstream bug fix, the 1.1.1 DSR in the forward direction):
     ACCEPT, update the expectation with a recorded reason. RE-BASELINE only when vectorbt moves
     away from the reference.
3. **Golden moved but contract tests are green:** inspect which fold winners, trade counts or
   gates changed. A winner or gate flip stops the upgrade for the owner. Float noise below 1e-9
   with the same trades is expected and does not need new goldens (loosen nothing).
4. **numba and Rust disagree:** the simulator must be exact and the indicators within 1e-10. File
   an upstream issue, pin `GGT_VBT_ENGINE=numba` for research only, do not ship a Rust path that
   disagrees.
5. **Dispatch probe red:** do not proceed. Rust must run the simulator.
6. **Only deprecations or a new floor:** adapt `engine/` (or add a shim in `engine/_compat.py`
   with its still-needed test) and rerun.

## 5. Oracle runs (Encom only)

The pinned 17-fold WFO and the live golden need the frozen tape (`data/snapshots/`, git-ignored);
GitHub runners cannot run them. They read only the Parquet tape (no database session). Heavy
runs: `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`. Always write to a scratch
path, then compare with the committed results:

```bash
.venv/bin/python scripts/p5_wfo_validation.py --list
.venv/bin/python scripts/p5_wfo_validation.py --runs core17_ensemble_sp500 --out-dir "$SCRATCH/p5"
.venv/bin/python scripts/p7_live_golden.py --out "$SCRATCH/p7_live_golden.json"      # about 40 s
```

Committed results: `research/programme/vectorbt-engine/p5-evidence/_p5_compare_*.json` and
`p7-evidence/p7_live_golden.json` (120 sessions). The live golden must stay 120/120 equal.
Compare provenance blocks with `ggTrader.engine.provenance.compare()`. Since P9 the default
`--out-dir` of `p5_wfo_validation.py` and `--out` of `p7_live_golden.py` is a scratch folder
under the system temp directory; a run only touches the committed evidence when `--out` or
`--out-dir` names it, so keep passing an explicit scratch path anyway.

## 6. Decide and record

- **ACCEPT** - numbers identical within tolerance. Commit the pin and lock. For a draft PR the
  watch workflow opened: `gh pr checkout <n>`, apply any test fix, then `gh pr ready <n>` and
  merge by the normal route (the workflow never merges).
- **RE-BASELINE** - a definition changed. Bump `CONTRACT_VERSION` (and `BASELINE_EPOCH` only
  with the owner's acceptance) in `engine/config.py`, regenerate goldens with
  `scripts/run_tests.sh tests/engine --update-golden --golden-reason "<why>"`, write the
  changelog line containing that exact reason, and put a research note under
  `research/programme/vectorbt-engine/`. `compare()` refuses cross-epoch comparisons, so old and
  new numbers stay separate.
- **REJECT** - leave the pin. **Create** `docs/vbt_upgrade_rejected.txt` if it does not exist
  yet and add the version on its own line (`1.2.1  # reason, upstream issue URL`) so
  `scripts/vbt_latest.py` stops proposing it; `scripts/vbt_latest.py` also stops proposing any
  older version than the highest rejected one (rejecting the newest of a burst rules out its
  older members; use the workflow's `version` input to force one). Close the draft PR and delete
  its branch: `gh pr close <n> --delete-branch --comment "<reason, see docs/vbt_upgrade_log.md>"`
  (an issue for an unresolvable lock: `gh issue close <n> --comment "<reason>"`). Write the reason
  in the upgrade log. A newer release is still evaluated; a burst of releases within 14 days is
  evaluated by its newest member only.

Always add: a `docs/changelog.md` entry (what changed, why, results) and a row in
`docs/vbt_upgrade_log.md` (**create** it on first use; `scripts/vbt_latest.py` does not need it)
with *version, date, decision, items that mattered, suites run, who decided*. A
numbers-changing decision (RE-BASELINE) belongs to the owner.

## 7. Promotion

Research adoption comes first. The trader image takes the new pin only after: research adopted,
the live golden identical, a 7-day soak, and the `ggtrader-deploy` pre-flight, which runs the
`selfcheck --engine rust` probe inside the pulled container and verifies Rust dispatch. (The
image build itself runs the self-check as root and as `appuser`: `Dockerfile` smoke steps and
`.github/workflows/docker-build.yml`.) Never merge and pull in the same hour.
