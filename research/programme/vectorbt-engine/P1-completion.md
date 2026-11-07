# P1 completion: lock file and the pandas-3 / vectorbt 1.2.0 + Rust stack (legacy indicators)

Date 2026-11-06. Branch `vbt12/engine` in `/home/flynn/ggTrader-vbt12`. Nothing committed, staged, pushed
or tagged; no Docker build, crontab, Alpaca or Telegram action; the only database access was one
read-only extract (see `p1-parity/p1-parity-note.md`).

## Verdict against the P1 exit criteria

| Criterion (spec section 9, P1) | Result |
|---|---|
| Full suite green on the new stack | **2270 passed**, 30 integration deselected, 0 failed, 29 warnings (7.5 to 9 min depending on box load). Same worktree code on the old stack: 2265 passed + 5 errors, all five in the new `test_check_vbt_engine.py` (vectorbt-rust does not exist on the old stack). S3's "1770" predates the current tree (2,250 pre-existing tests now); every one of the 2,250 pre-existing tests passes on both stacks. |
| Signal diff A to B zero | 48 ensemble + 6 weight combo curves, PIT mask: bit-identical (`np.array_equal`) old vs new, numba and Rust |
| 2-fold bit-for-bit except documented 1-ulp DSR | Met. One float differs: `dsr_value` fold 1, 0.7190648267528605 vs 0.7190648267528604 (1.1e-16). Winners, gates, WFE, live params, table identical |
| numba vs Rust bit-identical | Met: B1 vs B2 zero differences in everything compared |
| 17-fold | **Not run** (P0 owns the 17-fold legacy reference; it was not available). Open item |
| Rust dispatch probe | `scripts/check_vbt_engine.py`: pins the engine, counts calls into `simulate_from_signals_rs` (1 call), fails on mismatch; wired into Dockerfile (root and `appuser`) and CI |

## Measured numbers

- Cold start of a fresh process (import vectorbt + trader chain + first `simulate_signals` with ATR stops + first
  `simulate_weights`), median of 3: Rust 4.0 s, new stack numba 29.2 s, old stack today 22.0 s. Warm numba cache:
  8.8 s new, 8.0 s old. Rust ignores the cache.
- 2-fold WFO profile (150 symbols, 48 combos), indicative, overlapped load: old 109.8 s, new numba 104.2 s,
  new Rust 77.4 s; 48-combo sweep 16.1 s old, 22.0 s numba (includes JIT), 8.1 s Rust. Peak RSS 1.67 GB old,
  1.63 GB numba, 2.21 GB Rust.
- pandas-3 block fragmentation: 120.8 ms vs 18.1 ms per RSI-style chain without/with `.copy()`; pandas 2: 13.3 ms.
- Real-data pct_change hazard: 453 cells (2021-22, 5 of 507 symbols) and 2,937 cells (2025-26, 20 of 533) differ
  between pandas 2's padding and a plain pandas-3 call.

## Test edits

**No existing test was edited.** New tests only: `tests/test_pandas3_compat.py` (15 tests; pass on pandas 2.3.3 and
3.1.1; the gap tests fail if the helper is replaced by a bare call, checked) and
`tests/scripts/test_check_vbt_engine.py` (5 tests; Rust only).

## Findings that differ from, or add to, S3

1. S3 listed the 5 tests that failed at first as 3 pandas-3 plus 2 ruff-environment; re-verified: with
   `.dropna()` after `stack()` and an install with the `dev` extra there are no failures. `uv sync --extra test --extra dev` is required for
   `tests/scripts/test_ruff_excludes_research.py`.
2. `idio_vol` and `max_effect` really do rank stale/delisted symbols with padded zero returns today (the real-data
   probe saw non-leading NaN in their inputs). `padded_pct_change` preserves that old behaviour on purpose; whether
   it is the right behaviour is a P6 (terminal exits, OD7) question, not P1.
3. Outside the ensemble/WFO path pandas 3 changes last bits: `DataFrame.mean(axis=1)` (event study statistics, up to
   9e-16 absolute) and tie order among near-zero-variance names in `idio_vol.select`. Both are attributed in the
   parity note; none touches an equity curve, a metric, a winner or a gate. The spec's stop condition says "any numeric
   difference beyond the documented 1-ulp DSR" halts P1; these are the same ulp class but outside the documented
   DSR item, so the lead should confirm they are accepted.
4. The `uv` venv has no `pip`; the Dockerfile and CI use `uv pip check`.

## Decisions I made where the task text and the spec disagreed or were silent

- pandas bound: `<3.2` (spec section 6) instead of the task's `<4`; the lock holds 3.1.1 either way.
- `vectorbt[rust]==1.2.0` and `vectorbt-rust==1.2.0` are both listed (the explicit second line makes a lock that
  drops the extra fail).
- `ruff.toml` `target-version` left at `py311` (changing it could reformat code); `mypy python_version` set to 3.12.
- `scripts/run_tests.sh` already supported `GGT_TEST_PYTHON`; added `GGT_PYTHON` as an alias (default unchanged).
- The pandas-3 `inplace=True` deprecation warnings (16 data-loader sites, 29 warnings in the suite) and the
  xlrd/`float_precision` deprecations were left alone: warnings only, and the loaders write the tape.
- Only `Dockerfile` and `ci.yml` install dependencies; `docker-build.yml` builds the Dockerfile, so it inherits the
  new install and the Rust smoke check without edits. The base image is not yet pinned by digest (UNVERIFIED digest;
  the uv image is pinned to the tag `0.11.25`).

## Open problems

1. 17-fold bit-for-bit comparison against the P0 legacy reference is not done (needs P0's tape and reference).
2. The Docker image was not built (rules); the Dockerfile and CI lines are untested end to end. The `uv sync` and
   `check_vbt_engine.py` steps were run locally; CI's `setup-uv` action and the uv image tag `0.11.25` are unverified.
3. The DB read path and write path were not re-exercised in P1 (S3 covered the read path; the write path was never
   run). Parity used a parquet extract, so ns/us/s index resolution of real DB frames is not covered here.
4. `.github/workflows/docker-build.yml` still pushes without running tests; the smoke check inside the Dockerfile
   is the only gate (P2 adds the proper step).
5. CLAUDE.md still says `pip install -e .`; I did not edit the owner's instruction file. README, docs/installation.md and
   docs/architecture.md were updated.
6. Ties/order noise in `idio_vol` and `DataFrame.mean(axis=1)` last-bit differences (finding 3) await acceptance.

## Files changed

`git diff --stat` (tracked files; `.gitignore` was changed by P0 and is not counted):

```
 .github/workflows/ci.yml                          | 22 +++++++++--
 Dockerfile                                        | 31 ++++++++++++----
 README.md                                         |  5 ++-
 docs/architecture.md                              |  2 +-
 docs/changelog.md                                 | 30 +++++++++++++++
 docs/installation.md                              | 15 +++++---
 pyproject.toml                                    | 45 +++++++++++------------
 scripts/audit_reruns_wfo.py                       |  2 +-
 scripts/congress_trades_correlation_check.py      |  4 +-
 scripts/core_spy_blend.py                         |  4 +-
 scripts/cross_asset_sleeve_wfo.py                 | 29 +++++++++++----
 scripts/fomc_clustered_event_study.py             |  2 +-
 scripts/fx_hedge_overlay_static_baseline_check.py |  6 ++-
 scripts/insider_cluster_correlation_check.py      |  4 +-
 scripts/max_effect_correlation_check.py           |  4 +-
 scripts/midcap_research.py                        |  2 +-
 scripts/month_end_treasury_wfo.py                 |  8 ++--
 scripts/pairs_stat_arb_research.py                |  4 +-
 scripts/pead_correlation_check.py                 |  4 +-
 scripts/position_sizing_regimes.py                |  5 ++-
 scripts/run_tests.sh                              |  3 +-
 scripts/static_leverage_aftertax.py               |  2 +-
 src/ggTrader/lab/blend.py                         |  6 +--
 src/ggTrader/lab/data.py                          |  4 +-
 src/ggTrader/lab/event_study.py                   |  9 +++--
 src/ggTrader/lab/metrics.py                       |  6 +--
 src/ggTrader/lab/significance.py                  |  2 +-
 src/ggTrader/lab/strategies/idio_vol.py           |  3 +-
 src/ggTrader/lab/strategies/indicators.py         | 15 ++++++--
 src/ggTrader/lab/strategies/max_effect.py         |  3 +-
 src/ggTrader/lab/strategies/signals.py            |  2 +-
 src/ggTrader/lab/wfo.py                           |  2 +-
 src/ggTrader/paper/feature_gate.py                |  6 ++-
 src/ggTrader/paper/overlay.py                     |  5 ++-
 34 files changed, 200 insertions(+), 96 deletions(-)
```

New files (untracked, mine): `uv.lock`, `scripts/check_vbt_engine.py`, `src/ggTrader/utils/returns.py`,
`tests/test_pandas3_compat.py`, `tests/scripts/test_check_vbt_engine.py`,
`research/programme/vectorbt-engine/p1-parity/` (driver, cold-start probe and runner, results, per-run JSON,
parity note) and this file. Temporary scratch (not in the repo): the DB extract, raw arrays and logs under the
session scratchpad `p1/` folder.
