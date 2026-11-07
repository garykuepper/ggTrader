# vectorbt 1.2.0 engine rebuild: saved research and legacy references

Programme folder for the vectorbt engine rebuild (design: [the spec](../../../docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md)).
Created in phase P0 on 2026-11-06 so that later phases build from files, not from session prose
(spec Appendix D, question 5). Nothing here changes a strategy verdict.

Placement test (research-reorganization design, section 5): the work establishes a shared
experimental standard and tooling for the whole lab, which is test 4, so it lives in
`programme/`. Reports follow `<YYYY-MM-DD>-<slug>`; machine outputs keep the `_<name>` prefix.

> **Renamed 2026-11-06 (P9):** `ggTrader.lab.strategies.v2.*` is now `ggTrader.lab.strategies.*`, `ggTrader.lab.wfo_v2` is `ggTrader.lab.wfo`, and `tests/lab/{strategies,wfo}_v2` are `tests/lab/{strategies,wfo}`. The completion notes below keep the old names as written.

> **Pruned 2026-11-06 (P10):** nine regenerable intermediates were dropped or compressed (`profiling/wfo.prof`, `evidence/vbt-spec/S5-evidence/tape_volume.parquet`, the six `evidence/vbt-research/B-bench/portfolio_venv*.npz`, `evidence/vbt-spec/s2/dep*.html` now `.html.gz`); all JSON results and notes are kept. The file inventory table below still lists them as written; the list with sizes and the way to regenerate each is in [`P10-ledger.csv`](P10-ledger.csv). `REGISTRY_V2` is now `REGISTRY`. Archived source: [`legacy-src/`](legacy-src/README.md).

## Start here

00. [`P10-completion.md`](P10-completion.md): the last cleanup before review: strategy discovery, one DSR, dead code archived, stale docs fixed, evidence pruned; ledger in [`P10-ledger.csv`](P10-ledger.csv).
0. [`P5-completion.md`](P5-completion.md): the walk-forward on the engine: equity-level parity with the legacy sweeps, definition diagnostic, timings, deletion list for the lead.
1. [`P0-completion.md`](P0-completion.md): what P0 saved, hashes, the legacy reference numbers, what was skipped, problems.
2. [Final research synthesis (F)](reports/2026-11-05-research-F-synthesis-and-verdict.md): the verified decision to move to vectorbt 1.2.0 + Rust.
3. [`legacy-references/`](legacy-references/): the frozen-tape results every later engine must reproduce or explain.

## Caveats about the evidence files

- Scripts under `evidence/` were written in a session scratchpad and contain absolute scratchpad
  paths, local virtualenv names and a pointer to the production checkout. They are saved as
  records of what was run, not as runnable tools. Rebuild runnable versions in `scripts/` or `tests/`.
- Their outputs were produced on the **legacy tape of 2026-11-05** read directly from the database,
  not on the frozen snapshot (`data/snapshots/ohlcv_20261106.parquet`, built later in P0).
- `reports/2026-11-05-research-readme-master.md` is a copy of the upstream vectorbt master README
  (third-party text, kept as a reference). `evidence/vbt-research/pro_llms.txt` is likewise upstream text.
- Skipped on purpose (virtualenvs, caches, files over 2 MB, third-party package copies, downloaded
  third-party skill text): listed in [`P0-completion.md`](P0-completion.md).

## Files

Every saved file has one row below (generated mechanically from each file's own docstring, first
line or keys; row text is descriptive, not a review).

### Reports

| Report | Original scratchpad name | Title |
|---|---|---|
| [`reports/2026-11-05-research-A-vectorbt-0285-inventory.md`](reports/2026-11-05-research-A-vectorbt-0285-inventory.md) | `vbt-research/A-vectorbt-0.28.5-inventory.md` | vectorbt 0.28.5 (installed) feature inventory |
| [`reports/2026-11-05-research-B-vectorbt-1x-and-rust.md`](reports/2026-11-05-research-B-vectorbt-1x-and-rust.md) | `vbt-research/B-vectorbt-1x-and-rust.md` | vectorbt 0.28.5 -> 1.x and the optional Rust engine: what is delivered, what it would cost |
| [`reports/2026-11-05-research-C-ggtrader-handrolled-inventory.md`](reports/2026-11-05-research-C-ggtrader-handrolled-inventory.md) | `vbt-research/C-ggtrader-handrolled-inventory.md` | C: ggTrader hand-rolled inventory (lab + paper) |
| [`reports/2026-11-05-research-D-vectorbt-wfo-and-agents.md`](reports/2026-11-05-research-D-vectorbt-wfo-and-agents.md) | `vbt-research/D-vectorbt-wfo-and-agents.md` | vectorbt for a walk-forward-optimisation (WFO) lab: how far it carries ggTrader, and how agent-friendly it is |
| [`reports/2026-11-05-research-F-synthesis-and-verdict.md`](reports/2026-11-05-research-F-synthesis-and-verdict.md) | `vbt-research/F-synthesis-and-verdict.md` | F: Verification of reports A-D and the vectorbt decision for ggTrader |
| [`reports/2026-11-05-research-readme-master.md`](reports/2026-11-05-research-readme-master.md) | `vbt-research/readme_master.md` | Upstream vectorbt master README (third-party text, reference copy) |
| [`reports/2026-11-06-spec-S1-target-architecture.md`](reports/2026-11-06-spec-S1-target-architecture.md) | `vbt-spec/S1-target-architecture.md` | S1: Target architecture for ggTrader on vectorbt 1.2.0 + vectorbt-rust |
| [`reports/2026-11-06-spec-S2-upgradeability.md`](reports/2026-11-06-spec-S2-upgradeability.md) | `vbt-spec/S2-upgradeability.md` | S2: Upgradeability, making vectorbt updates routine and safe |
| [`reports/2026-11-06-spec-S3-stack-migration.md`](reports/2026-11-06-spec-S3-stack-migration.md) | `vbt-spec/S3-stack-migration.md` | S3: What moving ggTrader to the vectorbt 1.2.0 stack takes (hands-on) |
| [`reports/2026-11-06-spec-S4-strategy-port-inventory.md`](reports/2026-11-06-spec-S4-strategy-port-inventory.md) | `vbt-spec/S4-strategy-port-inventory.md` | S4 Strategy Port Inventory (facts and classification) |
| [`reports/2026-11-06-spec-S5-validation-protocol.md`](reports/2026-11-06-spec-S5-validation-protocol.md) | `vbt-spec/S5-validation-protocol.md` | S5: Validation and re-baseline protocol for the vectorbt 1.2.0 + Rust migration |
| [`reports/2026-11-06-spec-S6-data-and-universe.md`](reports/2026-11-06-spec-S6-data-and-universe.md) | `vbt-spec/S6-data-and-universe.md` | S6: Data and point-in-time universe layer, in vectorbt 1.2.0 terms |
| [`reports/2026-11-06-spec-S7-live-trader-integration.md`](reports/2026-11-06-spec-S7-live-trader-integration.md) | `vbt-spec/S7-live-trader-integration.md` | S7: Moving the live paper trader onto the vectorbt 1.2.0 + Rust stack |
| [`reports/2026-11-06-spec-S8-complementary-tools.md`](reports/2026-11-06-spec-S8-complementary-tools.md) | `vbt-spec/S8-complementary-tools.md` | S8: Open-source tools that complement vectorbt for the ggTrader lab |
| [`reports/2026-11-06-spec-sec8-9-draft.md`](reports/2026-11-06-spec-sec8-9-draft.md) | `sec8_9.md` | 8. Trader rebuild |

### Research-phase evidence (reports A, B, C, D, F)

98 files under `evidence/vbt-research/`.

| File | Bytes | What it is |
|---|---:|---|
| `evidence/vbt-research/B-bench/bench_indicators.py` | 5677 | vbt rolling/EWM/indicator micro-benchmarks: numba vs rust vs pandas (1.2.0 venv only). |
| `evidence/vbt-research/B-bench/bench_portfolio.py` | 5208 | Portfolio micro-benchmark mirroring ggTrader's usage (from_signals / from_orders). |
| `evidence/vbt-research/B-bench/bench_small_calls.py` | 2043 | WFO-fold-sized from_orders / from_signals calls (many small calls): per-call wall time, incl. pf.value() + trades.count(). |
| `evidence/vbt-research/B-bench/check_exactness.py` | 1297 | Sanity check that the 0.0 differences in bench_indicators are real, not a harness artefact. |
| `evidence/vbt-research/B-bench/check_exactness_out.txt` | 266 | rolling_mean #elements != (numba,rust): 919926 \| #!= (numba,pandas): 919517 \| max \|x\| 3247.247676844627 |
| `evidence/vbt-research/B-bench/common.py` | 1886 | Shared helpers for the vectorbt benchmark scripts (runs on 0.28.5 and 1.2.0). |
| `evidence/vbt-research/B-bench/compat_compare.txt` | 914 | venv100_numba {'w_ret': 'identical', 'w_eq': 'identical', 's_sl_trail_ret': 'identical', 's_sl_trail_eq': 'ide |
| `evidence/vbt-research/B-bench/compat_ggtrader_simulate.py` | 2806 | Run ggTrader's own lab/simulate.py (loaded read-only from the repo by path) under a given venv. |
| `evidence/vbt-research/B-bench/compat_raw.txt` | 4692 | === venv0285 default |
| `evidence/vbt-research/B-bench/compat_venv0285_default.npz` | 66034 | npz arrays: w_ret, w_eq, s_sl_trail_ret, s_sl_trail_eq, s_plain_ret, s_plain_eq |
| `evidence/vbt-research/B-bench/compat_venv100_numba.npz` | 66034 | npz arrays: w_ret, w_eq, s_sl_trail_ret, s_sl_trail_eq, s_plain_ret, s_plain_eq |
| `evidence/vbt-research/B-bench/compat_venv100_rust.npz` | 66034 | npz arrays: w_ret, w_eq, s_sl_trail_ret, s_sl_trail_eq, s_plain_ret, s_plain_eq |
| `evidence/vbt-research/B-bench/compat_venv120_numba.npz` | 66034 | npz arrays: w_ret, w_eq, s_sl_trail_ret, s_sl_trail_eq, s_plain_ret, s_plain_eq |
| `evidence/vbt-research/B-bench/compat_venv120_rust.npz` | 66034 | npz arrays: w_ret, w_eq, s_sl_trail_ret, s_sl_trail_eq, s_plain_ret, s_plain_eq |
| `evidence/vbt-research/B-bench/drift_compare.txt` | 3376 | == venv100_numba |
| `evidence/vbt-research/B-bench/fallback_probe.py` | 2887 | Probe when vectorbt 1.2.0 engine='auto' silently uses numba vs rust, and what engine='rust' does when unsupported. |
| `evidence/vbt-research/B-bench/fallback_probe_out.txt` | 2323 | baseline float64 signals                                engine=auto  -> sim kernel rust calls=1  ok  warnings= |
| `evidence/vbt-research/B-bench/fuzz_compare.txt` | 702 | baseline 0.28.5: ok cases 102 error idx [0, 16, 18, 22, 24, 32, 34, 42, 44, 50, 56, 72, 84, 96, 104, 106, 110, |
| `evidence/vbt-research/B-bench/fuzz_parity.py` | 3829 | Randomised parity fuzz of Portfolio.from_signals / from_orders. Same seeds in every venv; arrays saved for cross-venv compare. |
| `evidence/vbt-research/B-bench/fuzz_raw.txt` | 2005 | none 0.28.5 cases ok: 102 errors: 18 {0: 'ValueError: Only SizeType.Amount, SizeType.Value, and SizeType.Perce |
| `evidence/vbt-research/B-bench/fuzz_venv0285_none.npz` | 458500 | npz arrays: v1, n1, o1, v2, n2, o2 |
| `evidence/vbt-research/B-bench/fuzz_venv0285_none.npz.errs.json` | 1681 | JSON results; keys: 0, 16, 18, 22, 24, 32 |
| `evidence/vbt-research/B-bench/fuzz_venv100_numba.npz` | 458500 | npz arrays: v1, n1, o1, v2, n2, o2 |
| `evidence/vbt-research/B-bench/fuzz_venv100_numba.npz.errs.json` | 1681 | JSON results; keys: 0, 16, 18, 22, 24, 32 |
| `evidence/vbt-research/B-bench/fuzz_venv100_rust.npz` | 458500 | npz arrays: v1, n1, o1, v2, n2, o2 |
| `evidence/vbt-research/B-bench/fuzz_venv100_rust.npz.errs.json` | 1825 | JSON results; keys: 0, 16, 18, 22, 24, 32 |
| `evidence/vbt-research/B-bench/fuzz_venv120_numba.npz` | 458500 | npz arrays: v1, n1, o1, v2, n2, o2 |
| `evidence/vbt-research/B-bench/fuzz_venv120_numba.npz.errs.json` | 1681 | JSON results; keys: 0, 16, 18, 22, 24, 32 |
| `evidence/vbt-research/B-bench/fuzz_venv120_rust.npz` | 458500 | npz arrays: v1, n1, o1, v2, n2, o2 |
| `evidence/vbt-research/B-bench/fuzz_venv120_rust.npz.errs.json` | 1825 | JSON results; keys: 0, 16, 18, 22, 24, 32 |
| `evidence/vbt-research/B-bench/indicator_drift.py` | 1624 | Do indicator/metric outputs change between vectorbt versions? Saves arrays for cross-venv compare. |
| `evidence/vbt-research/B-bench/indicators_raw_run1.txt` | 4245 | rolling_mean w=20 (500 sym) {'numba_s': 0.00423, 'numba_rust_calls': 0, 'rust_s': 0.0141, 'rust_rust_calls': 6 |
| `evidence/vbt-research/B-bench/indicators_raw_run2.txt` | 4238 | rolling_mean w=20 (500 sym) {'numba_s': 0.00478, 'numba_rust_calls': 0, 'rust_s': 0.0138, 'rust_rust_calls': 6 |
| `evidence/vbt-research/B-bench/indicators_venv120_run1.json` | 5789 | JSON results; keys: vbt, numpy, pandas, cases, shape, sweep_shape |
| `evidence/vbt-research/B-bench/indicators_venv120_run2.json` | 5790 | JSON results; keys: vbt, numpy, pandas, cases, shape, sweep_shape |
| `evidence/vbt-research/B-bench/pandas3_probe.py` | 6076 | Probe pandas 2.3.3 vs 3.1.1 behaviour for idioms used in a MultiIndex-OHLCV / groupby / SQL / yfinance codebase. Synthetic data on |
| `evidence/vbt-research/B-bench/pandas3_probe_venv0285.txt` | 3303 | pandas 2.3.3 numpy 2.2.6 |
| `evidence/vbt-research/B-bench/pandas3_probe_venv120.txt` | 2869 | pandas 3.1.1 numpy 2.5.4 |
| `evidence/vbt-research/B-bench/pip_install.log` | 24587 | Collecting vectorbt==1.2.0 |
| `evidence/vbt-research/B-bench/pip_install_0285.log` | 15702 | Collecting vectorbt==0.28.5 |
| `evidence/vbt-research/B-bench/pip_install_0285_extras.log` | 0 | empty file |
| `evidence/vbt-research/B-bench/pip_install_100.log` | 0 | empty file |
| `evidence/vbt-research/B-bench/pip_install_100b.log` | 0 | empty file |
| `evidence/vbt-research/B-bench/pip_install_120_extras.log` | 0 | empty file |
| `evidence/vbt-research/B-bench/portfolio_rerun.done` | 5 | done |
| `evidence/vbt-research/B-bench/portfolio_venv0285_none.json` | 2218 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv0285_none.npz` | 1378000 | npz arrays: signals_1group_500sym_trailSL_pct, signals_1group_500sym_nostop_pct, signals_48grp_x100sym_trailSL_pct, orders_48grp_x100sym_targetpct_auto, orders_1group_500sym_targetpct_auto |
| `evidence/vbt-research/B-bench/portfolio_venv0285_none_pass2.json` | 2474 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv0285_none_pass2.txt` | 393 | signals_1group_500sym_trailSL_pct sim=0.075s value=0.121s trades.count=0.012s |
| `evidence/vbt-research/B-bench/portfolio_venv0285_none_pass3.json` | 2467 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv0285_none_pass3.txt` | 393 | signals_1group_500sym_trailSL_pct sim=0.077s value=0.125s trades.count=0.014s |
| `evidence/vbt-research/B-bench/portfolio_venv100_numba.json` | 2466 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv100_numba.npz` | 1378000 | npz arrays: signals_1group_500sym_trailSL_pct, signals_1group_500sym_nostop_pct, signals_48grp_x100sym_trailSL_pct, orders_48grp_x100sym_targetpct_auto, orders_1group_500sym_targetpct_auto |
| `evidence/vbt-research/B-bench/portfolio_venv100_numba.txt` | 393 | signals_1group_500sym_trailSL_pct sim=0.046s value=5.269s trades.count=1.208s |
| `evidence/vbt-research/B-bench/portfolio_venv100_rust.json` | 4438 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv100_rust.npz` | 1378000 | npz arrays: signals_1group_500sym_trailSL_pct, signals_1group_500sym_nostop_pct, signals_48grp_x100sym_trailSL_pct, orders_48grp_x100sym_targetpct_auto, orders_1group_500sym_targetpct_auto |
| `evidence/vbt-research/B-bench/portfolio_venv100_rust.txt` | 393 | signals_1group_500sym_trailSL_pct sim=0.051s value=0.073s trades.count=0.003s |
| `evidence/vbt-research/B-bench/portfolio_venv120_auto.json` | 4190 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv120_auto.npz` | 1371783 | npz arrays: signals_1group_500sym_trailSL_pct, signals_1group_500sym_nostop_pct, signals_48grp_x100sym_trailSL_pct, orders_48grp_x100sym_targetpct_auto, orders_1group_500sym_targetpct_auto |
| `evidence/vbt-research/B-bench/portfolio_venv120_numba.json` | 2219 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv120_numba.npz` | 1371783 | npz arrays: signals_1group_500sym_trailSL_pct, signals_1group_500sym_nostop_pct, signals_48grp_x100sym_trailSL_pct, orders_48grp_x100sym_targetpct_auto, orders_1group_500sym_targetpct_auto |
| `evidence/vbt-research/B-bench/portfolio_venv120_numba_pass2.json` | 2459 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv120_numba_pass2.txt` | 393 | signals_1group_500sym_trailSL_pct sim=0.149s value=0.234s trades.count=0.020s |
| `evidence/vbt-research/B-bench/portfolio_venv120_numba_pass3.json` | 2452 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv120_numba_pass3.txt` | 393 | signals_1group_500sym_trailSL_pct sim=0.149s value=0.149s trades.count=0.015s |
| `evidence/vbt-research/B-bench/portfolio_venv120_rust.json` | 4185 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv120_rust.npz` | 1371783 | npz arrays: signals_1group_500sym_trailSL_pct, signals_1group_500sym_nostop_pct, signals_48grp_x100sym_trailSL_pct, orders_48grp_x100sym_targetpct_auto, orders_1group_500sym_targetpct_auto |
| `evidence/vbt-research/B-bench/portfolio_venv120_rust_pass2.json` | 4430 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv120_rust_pass2.txt` | 393 | signals_1group_500sym_trailSL_pct sim=0.112s value=0.174s trades.count=0.006s |
| `evidence/vbt-research/B-bench/portfolio_venv120_rust_pass3.json` | 4430 | JSON results; keys: vbt, numpy, pandas, engine, python, cases |
| `evidence/vbt-research/B-bench/portfolio_venv120_rust_pass3.txt` | 393 | signals_1group_500sym_trailSL_pct sim=0.110s value=0.160s trades.count=0.006s |
| `evidence/vbt-research/B-bench/profile_small_call.py` | 1141 | Python script (no docstring); import sys, cProfile, pstats, io, numpy as np, pandas as pd, vectorbt  |
| `evidence/vbt-research/B-bench/profile_small_call_out.txt` | 1035 | 1.0.0 numba |
| `evidence/vbt-research/B-bench/run_portfolio_rerun.sh` | 1024 | #!/bin/bash |
| `evidence/vbt-research/B-bench/small_calls_raw.jsonl` | 4081 | {"vbt": "0.28.5", "engine": "none", "from_orders targetpercent auto, 280d x 150cols, 3 groups": {"first_call_s |
| `evidence/vbt-research/B-bench/small_calls_table.txt` | 1728 | venv0285 vbt 0.28.5 engine=none  from_orders ta first=0.32s median=27.8ms min=25.8 p90=29.9 chk=317947.8879 |
| `evidence/vbt-research/D-examples/d1_output_20sym.txt` | 2934 | data: (1695, 20), 2020-01-01 .. 2026-06-30 |
| `evidence/vbt-research/D-examples/d1_output_500sym_mem.txt` | 1518 | data: (1695, 500), 2020-01-01 .. 2026-06-30 |
| `evidence/vbt-research/D-examples/d1_wfo_splitter.py` | 7942 | D1: ~17 rolling folds (12m train / 3m test) with vbt's own splitter + IndicatorFactory grid. |
| `evidence/vbt-research/D-examples/d2_output.txt` | 2955 | === A. param_product=True vs zipped === |
| `evidence/vbt-research/D-examples/d2_param_sweeps.py` | 4035 | D2: how parameters become columns in vbt 0.28.5 and how to pick winners. |
| `evidence/vbt-research/D-examples/d3_output.txt` | 4598 | SMA20: pandas rolling vs vbt.MA(ewm=False)                 warm(ggt/vbt)=19/19  max\|d\|=4.576e-12 corr=1.000000 |
| `evidence/vbt-research/D-examples/d3_primitive_equivalence.py` | 12898 | D3: numerically compare ggTrader hand-written primitives against vectorbt 0.28.5 equivalents. |
| `evidence/vbt-research/D-examples/d3_results.csv` | 2184 | case,warm_ggt,warm_vbt,max_abs_diff,corr,nan_mismatch_cells,cells |
| `evidence/vbt-research/D-examples/d4_output.txt` | 438 | fold  1: prefix sim to 2022-01-31 vs full sim sliced -> max \|equity diff\| = 0.000e+00; max \|Sharpe diff\| over  |
| `evidence/vbt-research/D-examples/d4_prefix_causality.py` | 2525 | D4: ggTrader's wfo.py re-simulates the 48-combo grid on ohlcv.loc[:window_end] for EVERY fold |
| `evidence/vbt-research/D-examples/d5_full_history_wfo.py` | 2522 | D5: the vbt-native shape of ggTrader's WFO at full scale (500 symbols x 48 combos x 17 folds). |
| `evidence/vbt-research/D-examples/d5_output_500sym.txt` | 579 | symbols=500 T=1695 cols=24000 groups=48 |
| `evidence/vbt-research/F-checks/d3_rerun.txt` | 4598 | SMA20: pandas rolling vs vbt.MA(ewm=False)                 warm(ggt/vbt)=19/19  max\|d\|=4.576e-12 corr=1.000000 |
| `evidence/vbt-research/F-checks/f_primitives.py` | 4375 | F: own primitive-equality checks (independent of D's script), ggTrader functions vs vbt 0.28.5. |
| `evidence/vbt-research/F-checks/f_primitives_out.txt` | 1453 | SMA20 pandas vs vbt.MA                     max\|diff\|=1.319e-11  first-valid bar g/v=19/19  only-one-NaN cells= |
| `evidence/vbt-research/F-checks/f_real_checks.json` | 1102 | JSON results; keys: eligible_at_40dates, generate_signals_1combo, simulate_signals_12groups, pit_vectorised, causality_real_sweep |
| `evidence/vbt-research/F-checks/f_real_checks.log` | 1375 | $PBCT: possibly delisted; no timezone found |
| `evidence/vbt-research/F-checks/f_real_checks.py` | 6822 | F: independent checks on real cached data (read-only use of ggTrader). |
| `evidence/vbt-research/F-checks/pd3.rst` | 125904 | .. _whatsnew_300: |
| `evidence/vbt-research/count_primitives.py` | 5566 | Read-only: measure line spans of functions classified as hand-rolled primitives. |
| `evidence/vbt-research/pro_llms.txt` | 11949 | # VectorBT PRO |
| `evidence/vbt-research/tree.txt` | 10332 | .coveragerc |

### Spec-phase evidence (S1 to S8, lead checks)

320 files under `evidence/vbt-spec/`.

| File | Bytes | What it is |
|---|---:|---|
| `evidence/vbt-spec/S3-evidence/01_resolve_unmodified_pyproject.log` | 5554 | Obtaining file:///home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0 |
| `evidence/vbt-spec/S3-evidence/02_resolve_keep_pandas_ta_plotly6.log` | 47132 | Obtaining file:///home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0 |
| `evidence/vbt-spec/S3-evidence/03_resolve_no_pandas_ta_plotly6.log` | 30237 | Obtaining file:///home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0 |
| `evidence/vbt-spec/S3-evidence/04_pip_install_real.log` | 44687 | Obtaining file:///home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0 |
| `evidence/vbt-spec/S3-evidence/05_new_stack_freeze.txt` | 2673 | aiohappyeyeballs==2.7.1 |
| `evidence/vbt-spec/S3-evidence/05_old_stack_freeze.txt` | 3143 | aiodns==4.0.0 |
| `evidence/vbt-spec/S3-evidence/06_dep_table.md` | 2450 | \| package \| declared in pyproject \| old (2.3.3 stack) \| new (resolved) \| declared pandas / numpy / numba bound |
| `evidence/vbt-spec/S3-evidence/10_tests_new.log` | 30297 | running pytest under MemoryMax=8G (no swap) |
| `evidence/vbt-spec/S3-evidence/10_tests_new.xml` | 249720 | pytest JUnit report (1770 tests, 5 failures) |
| `evidence/vbt-spec/S3-evidence/11_tests_old.log` | 2701 | running pytest under MemoryMax=8G (no swap) |
| `evidence/vbt-spec/S3-evidence/11_tests_old.xml` | 228226 | pytest JUnit report (1770 tests, 0 failures) |
| `evidence/vbt-spec/S3-evidence/12_tests_new_patched.log` | 8201 | running pytest under MemoryMax=8G (no swap) |
| `evidence/vbt-spec/S3-evidence/12_tests_new_patched.xml` | 227562 | pytest JUnit report (1770 tests, 0 failures) |
| `evidence/vbt-spec/S3-evidence/13_tests_old_futurewarning.log` | 2709 | running pytest under MemoryMax=8G (no swap) |
| `evidence/vbt-spec/S3-evidence/13_tests_old_futurewarning.xml` | 228212 | pytest JUnit report (1770 tests, 0 failures) |
| `evidence/vbt-spec/S3-evidence/14_tests_new_final.log` | 8201 | running pytest under MemoryMax=8G (no swap) |
| `evidence/vbt-spec/S3-evidence/14_tests_new_final.xml` | 227562 | pytest JUnit report (1770 tests, 0 failures) |
| `evidence/vbt-spec/S3-evidence/21_db_read_compare.txt` | 280 | shape (250, 35) (250, 35) |
| `evidence/vbt-spec/S3-evidence/22_yf_new.json` | 804 | JSON results; keys: pandas, yfinance, raw_index_dtype, raw_columns_names, raw_col_level0, loader_shape |
| `evidence/vbt-spec/S3-evidence/22_yf_old.json` | 809 | JSON results; keys: pandas, yfinance, raw_index_dtype, raw_columns_names, raw_col_level0, loader_shape |
| `evidence/vbt-spec/S3-evidence/23_persist_new.json` | 1160 | JSON results; keys: sqlalchemy, get_latest_snapshot, get_earliest_snapshot, get_latest_snapshot_positions, get_latest_snapshot_run_date, get_peak_value |
| `evidence/vbt-spec/S3-evidence/23_persist_old.json` | 1161 | JSON results; keys: sqlalchemy, get_latest_snapshot, get_earliest_snapshot, get_latest_snapshot_positions, get_latest_snapshot_run_date, get_peak_value |
| `evidence/vbt-spec/S3-evidence/24_pct_change_gap.json` | 398 | JSON results; keys: 2021-2022, 2025-2026 |
| `evidence/vbt-spec/S3-evidence/25_ulp_new.txt` | 164 | pandas 3.1.1 scipy 1.18.1 |
| `evidence/vbt-spec/S3-evidence/25_ulp_old.txt` | 165 | pandas 2.3.3 scipy 1.17.1 |
| `evidence/vbt-spec/S3-evidence/26_joblib_new.txt` | 263 | lightgbm 4.7.0 sklearn 1.9.1 LGBMClassifier |
| `evidence/vbt-spec/S3-evidence/26_joblib_old.txt` | 365 | lightgbm 4.6.0 sklearn 1.8.0 LGBMClassifier |
| `evidence/vbt-spec/S3-evidence/30_e2e_new.log` | 4500 | [data] 860 rows x 61 symbols (1d) |
| `evidence/vbt-spec/S3-evidence/31_e2e_old.err` | 1019 | Command being timed: "/home/flynn/ggTrader/.venv/bin/python /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29a |
| `evidence/vbt-spec/S3-evidence/31_e2e_old.log` | 4342 | [data] 860 rows x 61 symbols (1d) |
| `evidence/vbt-spec/S3-evidence/32_sweep_new.err` | 1066 | Command being timed: "/home/flynn/ggTrader/.claude/worktrees/agent-abed3ab263d5272c0/.venv/bin/python /tmp/cla |
| `evidence/vbt-spec/S3-evidence/32_sweep_new.log` | 267 | [data] 860 rows x 101 symbols (1d) |
| `evidence/vbt-spec/S3-evidence/33_sweep_old.err` | 1023 | Command being timed: "/home/flynn/ggTrader/.venv/bin/python /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29a |
| `evidence/vbt-spec/S3-evidence/33_sweep_old.log` | 266 | [data] 860 rows x 101 symbols (1d) |
| `evidence/vbt-spec/S3-evidence/40_e2e_all.log` | 193 | == 31_e2e_old start 00:13:54 |
| `evidence/vbt-spec/S3-evidence/41_e2e_compare.txt` | 1867 | === run_wfo (2 folds, 60 symbols) === |
| `evidence/vbt-spec/S3-evidence/50_coldstart.jsonl` | 5243 | {"label":"new_auto_cold","load1":"8.46","r":{"mode": "auto", "numba_cache_dir": "/tmp/claude-1000/-home-flynn- |
| `evidence/vbt-spec/S3-evidence/50_coldstart_new_auto_cold.err` | 0 | empty file |
| `evidence/vbt-spec/S3-evidence/50_coldstart_new_auto_warm.err` | 0 | empty file |
| `evidence/vbt-spec/S3-evidence/50_coldstart_new_numba_cold.err` | 0 | empty file |
| `evidence/vbt-spec/S3-evidence/50_coldstart_new_numba_generic_cold.err` | 0 | empty file |
| `evidence/vbt-spec/S3-evidence/50_coldstart_new_numba_generic_warm.err` | 0 | empty file |
| `evidence/vbt-spec/S3-evidence/50_coldstart_new_numba_warm.err` | 0 | empty file |
| `evidence/vbt-spec/S3-evidence/50_coldstart_old_cold.err` | 0 | empty file |
| `evidence/vbt-spec/S3-evidence/50_coldstart_old_warm.err` | 0 | empty file |
| `evidence/vbt-spec/S3-evidence/51_sweep_timing.err` | 0 | empty file |
| `evidence/vbt-spec/S3-evidence/51_sweep_timing.jsonl` | 867 | load=2.67 old_numba |
| `evidence/vbt-spec/S3-evidence/52_done` | 5 | DONE |
| `evidence/vbt-spec/S3-evidence/52_prof_new.prof` | 198702 | cProfile dump (binary; read with pstats) |
| `evidence/vbt-spec/S3-evidence/52_prof_old.prof` | 184146 | cProfile dump (binary; read with pstats) |
| `evidence/vbt-spec/S3-evidence/52_sweep_profile_new.txt` | 138 | {"vectorbt": "1.2.0", "pandas": "3.1.1", "engine": "auto", "cols": 100, "rows": 860, "wall_cpu_per_rep": [[16. |
| `evidence/vbt-spec/S3-evidence/52_sweep_profile_old.txt` | 134 | {"vectorbt": "0.28.5", "pandas": "2.3.3", "engine": "na", "cols": 100, "rows": 860, "wall_cpu_per_rep": [[9.87 |
| `evidence/vbt-spec/S3-evidence/53_prof_report.txt` | 3184 | ===== old: total 7.4s, calls 5468513 |
| `evidence/vbt-spec/S3-evidence/54_op_micro2_new.txt` | 241 | pandas 3.1.1 \| blocks in extract_close frame: 100 \| consolidated copy blocks: 1 |
| `evidence/vbt-spec/S3-evidence/54_op_micro2_old.txt` | 239 | pandas 2.3.3 \| blocks in extract_close frame: 1 \| consolidated copy blocks: 1 |
| `evidence/vbt-spec/S3-evidence/54_op_micro_new.txt` | 546 | pandas 3.1.1 |
| `evidence/vbt-spec/S3-evidence/54_op_micro_old.txt` | 546 | pandas 2.3.3 |
| `evidence/vbt-spec/S3-evidence/55_after_consolidate.jsonl` | 359 | load=3.73 new_auto_patched |
| `evidence/vbt-spec/S3-evidence/56_sweep_new_patched.log` | 267 | [data] 860 rows x 101 symbols (1d) |
| `evidence/vbt-spec/S3-evidence/57_compare_patched.txt` | 1867 | === run_wfo (2 folds, 60 symbols) === |
| `evidence/vbt-spec/S3-evidence/58_wfo_timing.txt` | 551 | load=2.54 old |
| `evidence/vbt-spec/S3-evidence/60_help_new.tsv` | 2676 | scripts/a23_rebalance_check.py	0 |
| `evidence/vbt-spec/S3-evidence/60_help_old.tsv` | 2676 | scripts/a23_rebalance_check.py	0 |
| `evidence/vbt-spec/S3-evidence/70_worktree_patches.diff` | 4049 | diff --git a/pyproject.toml b/pyproject.toml |
| `evidence/vbt-spec/S3-evidence/coldstart.py` | 2777 | Cold-start probe, fresh process: import cost + first simulate_signals / simulate_weights call. |
| `evidence/vbt-spec/S3-evidence/db_new/facts.json` | 774 | JSON results; keys: pandas, psycopg2, shape, index_type, index_dtype, index_tz |
| `evidence/vbt-spec/S3-evidence/db_new/wide.parquet` | 93877 | parquet frame 250 x 35; columns ["('AAPL', 'close')", "('AAPL', 'high')", "('AAPL', 'low')", "('AAPL', 'open')"] |
| `evidence/vbt-spec/S3-evidence/db_new/write_capture_utc_midnight.json` | 405 | JSON results; keys: n_records, first_ts, ts_type, ts_tzinfo, sample_tuple, nan_to_none_types |
| `evidence/vbt-spec/S3-evidence/db_new/write_unit_ns.txt` | 79 | True datetime.datetime(2022, 1, 3, 0, 0) datetime.datetime(2022, 12, 29, 0, 0) |
| `evidence/vbt-spec/S3-evidence/db_new/write_unit_s.txt` | 79 | True datetime.datetime(2022, 1, 3, 0, 0) datetime.datetime(2022, 12, 29, 0, 0) |
| `evidence/vbt-spec/S3-evidence/db_new/write_unit_us.txt` | 79 | True datetime.datetime(2022, 1, 3, 0, 0) datetime.datetime(2022, 12, 29, 0, 0) |
| `evidence/vbt-spec/S3-evidence/db_old/facts.json` | 782 | JSON results; keys: pandas, psycopg2, shape, index_type, index_dtype, index_tz |
| `evidence/vbt-spec/S3-evidence/db_old/wide.parquet` | 93971 | parquet frame 250 x 35; columns ["('AAPL', 'close')", "('AAPL', 'high')", "('AAPL', 'low')", "('AAPL', 'open')"] |
| `evidence/vbt-spec/S3-evidence/db_old/write_capture_utc_midnight.json` | 405 | JSON results; keys: n_records, first_ts, ts_type, ts_tzinfo, sample_tuple, nan_to_none_types |
| `evidence/vbt-spec/S3-evidence/db_old/write_unit_ns.txt` | 79 | True datetime.datetime(2022, 1, 3, 0, 0) datetime.datetime(2022, 12, 29, 0, 0) |
| `evidence/vbt-spec/S3-evidence/db_old/write_unit_s.txt` | 79 | True datetime.datetime(2022, 1, 3, 0, 0) datetime.datetime(2022, 12, 29, 0, 0) |
| `evidence/vbt-spec/S3-evidence/db_old/write_unit_us.txt` | 79 | True datetime.datetime(2022, 1, 3, 0, 0) datetime.datetime(2022, 12, 29, 0, 0) |
| `evidence/vbt-spec/S3-evidence/db_read_compare.py` | 961 | Python script (no docstring); import json |
| `evidence/vbt-spec/S3-evidence/db_read_smoke.py` | 4406 | READ-ONLY DB smoke: CachedYFinanceLoader._db_fetch + lab.data helpers, dump frame facts. |
| `evidence/vbt-spec/S3-evidence/dep_table.py` | 1712 | Per-dependency table: old resolved version, new resolved version, declared pandas/numpy bounds. |
| `evidence/vbt-spec/S3-evidence/e2e_compare.py` | 2962 | Compare old-stack vs new-stack e2e artefacts (run under the NEW venv so pandas 3 reads both). |
| `evidence/vbt-spec/S3-evidence/e2e_new/oos_equity.csv` | 4511 | timestamp,ensemble__bb_period20_bb_std2.5_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_m |
| `evidence/vbt-spec/S3-evidence/e2e_new/summary.json` | 4782 | JSON results; keys: pandas, numpy, n_symbols, symbols, rows, folds |
| `evidence/vbt-spec/S3-evidence/e2e_new/table.txt` | 1965 | WFO: ensemble \| 48 combos x 2 folds \| rolling 12mo/3mo |
| `evidence/vbt-spec/S3-evidence/e2e_old/oos_equity.csv` | 4511 | timestamp,ensemble__bb_period20_bb_std2.5_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_m |
| `evidence/vbt-spec/S3-evidence/e2e_old/summary.json` | 4604 | JSON results; keys: pandas, numpy, n_symbols, symbols, rows, folds |
| `evidence/vbt-spec/S3-evidence/e2e_old/table.txt` | 1965 | WFO: ensemble \| 48 combos x 2 folds \| rolling 12mo/3mo |
| `evidence/vbt-spec/S3-evidence/e2e_sweep.py` | 4737 | Per-combo equity comparison on real cached data (READ-ONLY): signal sweep (48 ensemble |
| `evidence/vbt-spec/S3-evidence/e2e_wfo.py` | 5998 | Small real-data end-to-end WFO (ensemble, 2 folds) on cached symbols only. READ-ONLY. |
| `evidence/vbt-spec/S3-evidence/joblib_probe.py` | 670 | Python script (no docstring); import warnings |
| `evidence/vbt-spec/S3-evidence/op_micro.py` | 1540 | Per-op micro-benchmark on a (860 x 100) frame -- locates the pandas-3 slowdown in rsi_signals. |
| `evidence/vbt-spec/S3-evidence/op_micro2.py` | 1526 | Same micro-benchmark but on a frame built the way extract_close builds it (concat of Series, axis=1). |
| `evidence/vbt-spec/S3-evidence/paper_persist_ro_smoke.py` | 1549 | READ-ONLY smoke of paper/persist.py getters (SQLAlchemy 2.0.x vs 2.1.x). Prints only types and |
| `evidence/vbt-spec/S3-evidence/pct_change_gap_probe.py` | 2025 | How many cells would a pandas-3 bare pct_change() change on REAL cached SP500 closes? READ-ONLY. |
| `evidence/vbt-spec/S3-evidence/prof_report.py` | 952 | Python script (no docstring); import pstats |
| `evidence/vbt-spec/S3-evidence/pyproject.orig.toml` | 1856 | [build-system] |
| `evidence/vbt-spec/S3-evidence/pyproject.step1_novbt_pin.toml` | 1826 | [build-system] |
| `evidence/vbt-spec/S3-evidence/pyproject.step2_no_pandas_ta.toml` | 1809 | [build-system] |
| `evidence/vbt-spec/S3-evidence/run_after_consolidate.sh` | 1551 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/run_coldstart.sh` | 1201 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/run_e2e_all.sh` | 1080 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/run_sweep_profile.sh` | 540 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/run_sweep_timing.sh` | 751 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/run_tests_final.sh` | 901 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/run_tests_last.sh` | 372 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/run_tests_old.sh` | 511 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/run_wfo_timing.sh` | 713 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/scripts_help_sweep.sh` | 870 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/sweep_new/pit_mask.parquet` | 51672 | parquet frame 860 x 100; columns ['A', 'ABMD', 'ADM', 'AES'] |
| `evidence/vbt-spec/S3-evidence/sweep_new/signal_equity.parquet` | 242051 | parquet frame 860 x 48; columns ['ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14'] |
| `evidence/vbt-spec/S3-evidence/sweep_new/signal_metrics.csv` | 17730 | combo,total_return_pct,cagr_pct,sharpe,sortino,ann_vol_pct,max_drawdown_pct,n_trades |
| `evidence/vbt-spec/S3-evidence/sweep_new/summary.json` | 229 | JSON results; keys: pandas, numpy, n_symbols, rows, pit_mask_s, sweep_signal_s |
| `evidence/vbt-spec/S3-evidence/sweep_new/weight_equity.parquet` | 34362 | parquet frame 860 x 6; columns ['xs_momentum__lookback126_skip0_top_n10', 'xs_momentum__lookback126_skip0_top_n20', 'xs_momentum__lookback126_skip0_top_n50', 'xs_momentum__lookback126_skip21_top_n10'] |
| `evidence/vbt-spec/S3-evidence/sweep_new/weight_metrics.csv` | 1024 | combo,n_trades,total_return_pct,cagr_pct,sharpe,sortino,ann_vol_pct,max_drawdown_pct |
| `evidence/vbt-spec/S3-evidence/sweep_new_patched/pit_mask.parquet` | 51672 | parquet frame 860 x 100; columns ['A', 'ABMD', 'ADM', 'AES'] |
| `evidence/vbt-spec/S3-evidence/sweep_new_patched/signal_equity.parquet` | 242051 | parquet frame 860 x 48; columns ['ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14'] |
| `evidence/vbt-spec/S3-evidence/sweep_new_patched/signal_metrics.csv` | 17730 | combo,total_return_pct,cagr_pct,sharpe,sortino,ann_vol_pct,max_drawdown_pct,n_trades |
| `evidence/vbt-spec/S3-evidence/sweep_new_patched/summary.json` | 229 | JSON results; keys: pandas, numpy, n_symbols, rows, pit_mask_s, sweep_signal_s |
| `evidence/vbt-spec/S3-evidence/sweep_new_patched/weight_equity.parquet` | 34362 | parquet frame 860 x 6; columns ['xs_momentum__lookback126_skip0_top_n10', 'xs_momentum__lookback126_skip0_top_n20', 'xs_momentum__lookback126_skip0_top_n50', 'xs_momentum__lookback126_skip21_top_n10'] |
| `evidence/vbt-spec/S3-evidence/sweep_new_patched/weight_metrics.csv` | 1024 | combo,n_trades,total_return_pct,cagr_pct,sharpe,sortino,ann_vol_pct,max_drawdown_pct |
| `evidence/vbt-spec/S3-evidence/sweep_old/pit_mask.parquet` | 51929 | parquet frame 860 x 100; columns ['A', 'ABMD', 'ADM', 'AES'] |
| `evidence/vbt-spec/S3-evidence/sweep_old/signal_equity.parquet` | 242308 | parquet frame 860 x 48; columns ['ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14'] |
| `evidence/vbt-spec/S3-evidence/sweep_old/signal_metrics.csv` | 17730 | combo,total_return_pct,cagr_pct,sharpe,sortino,ann_vol_pct,max_drawdown_pct,n_trades |
| `evidence/vbt-spec/S3-evidence/sweep_old/summary.json` | 228 | JSON results; keys: pandas, numpy, n_symbols, rows, pit_mask_s, sweep_signal_s |
| `evidence/vbt-spec/S3-evidence/sweep_old/weight_equity.parquet` | 34619 | parquet frame 860 x 6; columns ['xs_momentum__lookback126_skip0_top_n10', 'xs_momentum__lookback126_skip0_top_n20', 'xs_momentum__lookback126_skip0_top_n50', 'xs_momentum__lookback126_skip21_top_n10'] |
| `evidence/vbt-spec/S3-evidence/sweep_old/weight_metrics.csv` | 1024 | combo,n_trades,total_return_pct,cagr_pct,sharpe,sortino,ann_vol_pct,max_drawdown_pct |
| `evidence/vbt-spec/S3-evidence/sweep_timing.py` | 2211 | Controlled timing of the 48-combo signal sweep (reads a saved PIT mask; READ-ONLY DB). |
| `evidence/vbt-spec/S3-evidence/target_constraints.txt` | 76 | vectorbt==1.2.0 |
| `evidence/vbt-spec/S3-evidence/ulp_probe.py` | 434 | Python script (no docstring); import numpy as np |
| `evidence/vbt-spec/S3-evidence/with_db_env.sh` | 810 | #!/usr/bin/env bash |
| `evidence/vbt-spec/S3-evidence/yf_live_probe.py` | 1529 | Tiny network probe (no DB): YFinanceDataLoader output frame facts for 2 symbols, 8 days. |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/ENS_entries.parquet` | 96689 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/ENS_exits.parquet` | 115117 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/LIVE_entries.parquet` | 67233 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/LIVE_exits.parquet` | 67233 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/bb_entries.parquet` | 101393 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/bb_exits.parquet` | 111850 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/ema_entries.parquet` | 93918 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/ema_exits.parquet` | 93985 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/macd_entries.parquet` | 102420 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/macd_exits.parquet` | 108527 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/meta.json` | 103 | JSON results; keys: arm, tag, pandas, full_tape_s, vectorbt |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/rsi_entries.parquet` | 92699 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/rsi_exits.parquet` | 110762 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/vbb_entries.parquet` | 88925 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/A_ggt_pd2/vbb_exits.parquet` | 111850 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/ENS_entries.parquet` | 96686 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/ENS_exits.parquet` | 115114 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/LIVE_entries.parquet` | 67230 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/LIVE_exits.parquet` | 67230 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/bb_entries.parquet` | 101390 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/bb_exits.parquet` | 111847 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/ema_entries.parquet` | 93915 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/ema_exits.parquet` | 93982 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/macd_entries.parquet` | 102417 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/macd_exits.parquet` | 108524 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/meta.json` | 129 | JSON results; keys: arm, tag, pandas, full_tape_s, vectorbt, engine_setting |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/rsi_entries.parquet` | 92696 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/rsi_exits.parquet` | 110759 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/vbb_entries.parquet` | 88922 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/B_ggt_pd3/vbb_exits.parquet` | 111847 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/ENS_entries.parquet` | 96687 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/ENS_exits.parquet` | 114907 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/LIVE_entries.parquet` | 67230 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/LIVE_exits.parquet` | 67230 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/bb_entries.parquet` | 101390 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/bb_exits.parquet` | 111847 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/ema_entries.parquet` | 93649 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/ema_exits.parquet` | 93683 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/macd_entries.parquet` | 102403 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/macd_exits.parquet` | 108258 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/meta.json` | 135 | JSON results; keys: arm, tag, pandas, full_tape_s, vectorbt, engine_setting |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/rsi_entries.parquet` | 92668 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/rsi_exits.parquet` | 110624 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/vbb_entries.parquet` | 88922 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/C_vbt_matched/vbb_exits.parquet` | 111847 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/ENS_entries.parquet` | 96687 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/ENS_exits.parquet` | 114907 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/LIVE_entries.parquet` | 67230 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/LIVE_exits.parquet` | 67230 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/bb_entries.parquet` | 101390 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/bb_exits.parquet` | 111847 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/ema_entries.parquet` | 93649 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/ema_exits.parquet` | 93683 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/macd_entries.parquet` | 102403 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/macd_exits.parquet` | 108258 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/meta.json` | 145 | JSON results; keys: arm, tag, pandas, full_tape_s, vectorbt, engine_setting |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/rsi_entries.parquet` | 92668 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/rsi_exits.parquet` | 110624 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/vbb_entries.parquet` | 88922 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/D_vbt_matched_vbtcross/vbb_exits.parquet` | 111847 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/ENS_entries.parquet` | 100972 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/ENS_exits.parquet` | 115479 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/LIVE_entries.parquet` | 67230 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/LIVE_exits.parquet` | 67230 | parquet frame 120 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/bb_entries.parquet` | 102276 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/bb_exits.parquet` | 111847 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/ema_entries.parquet` | 93649 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/ema_exits.parquet` | 93683 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/macd_entries.parquet` | 105145 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/macd_exits.parquet` | 106042 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/meta.json` | 137 | JSON results; keys: arm, tag, pandas, full_tape_s, vectorbt, engine_setting |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/rsi_entries.parquet` | 101409 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/rsi_exits.parquet` | 111912 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/vbb_entries.parquet` | 89011 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/arms/E_vbt_default/vbb_exits.parquet` | 111847 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S5-evidence/dump.log` | 1471 | ['HOLX', 'DISCK', 'NLOK', 'CTLT', 'ATVI', 'MXIM', 'PBCT', 'DAY', 'ANSS', 'DFS']: possibly delisted; no timezon |
| `evidence/vbt-spec/S5-evidence/engine_metrics.json` | 3794 | JSON results; keys: vectorbt, pandas, engine, invariants, metrics, rust_kernel_names_seen |
| `evidence/vbt-spec/S5-evidence/s5_compare.py` | 5330 | S5 prototype step 3: signal-diff report across arms, with one-at-a-time attribution. |
| `evidence/vbt-spec/S5-evidence/s5_dump_tape.py` | 3193 | S5 prototype step 1: dump a read-only snapshot of the cached SP500 tape to parquet. |
| `evidence/vbt-spec/S5-evidence/s5_engine_metrics.py` | 7279 | S5 prototype step 4 (venv120): engine dispatch/parity, invariants and metric definitions. |
| `evidence/vbt-spec/S5-evidence/s5_signals.py` | 7631 | S5 prototype step 2: per-voter ensemble signals under one implementation arm. |
| `evidence/vbt-spec/S5-evidence/signal_diff_report.json` | 34490 | JSON results; keys: burnin_bars, valid_cells, pairs, attribution_ensemble, live_window |
| `evidence/vbt-spec/S5-evidence/signal_diff_report.txt` | 9270 | == A_ggt_pd2 -> B_ggt_pd3 |
| `evidence/vbt-spec/S5-evidence/tape_meta.json` | 1777 | JSON results; keys: start, end, n_symbols, n_bars, first_bar, last_bar |
| `evidence/vbt-spec/S5-evidence/tape_volume.parquet` | 1886844 | parquet frame 1968 x 150; columns ['A', 'AAPL', 'ABNB', 'ACN'] |
| `evidence/vbt-spec/S6-evidence/db_ro.py` | 345 | Read-only DB helper for S6 (never writes; session forced read-only). |
| `evidence/vbt-spec/S6-evidence/env.sh` | 237 | # usage: source env.sh ; $PY script.py |
| `evidence/vbt-spec/S6-evidence/explore_db.out` | 9 | done 124 |
| `evidence/vbt-spec/S6-evidence/explore_db.py` | 1209 | Python script (no docstring); import sys; sys.path.insert(0,'.') |
| `evidence/vbt-spec/S6-evidence/p1_load.out` | 578 | universe 662 0.39 |
| `evidence/vbt-spec/S6-evidence/p1_load.py` | 722 | Python script (no docstring); import sys, time, resource |
| `evidence/vbt-spec/S6-evidence/p2_data_vbt.out` | 1727 | download->Data 42.7 s; symbols 646 missing 17 |
| `evidence/vbt-spec/S6-evidence/p2_data_vbt.py` | 3437 | Load full SP500-union tape via TimescaleData; time Data ops; run vbt indicators; save parquet. |
| `evidence/vbt-spec/S6-evidence/p3_yfdata.out` | 38941 | /tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/venv120/lib/python3.12/s |
| `evidence/vbt-spec/S6-evidence/p3_yfdata.py` | 2118 | Evaluate vbt.YFData: tz, mixed ETF/stock, delisted, speed. Network read only; nothing persisted. |
| `evidence/vbt-spec/S6-evidence/p3b_yf_dst.out` | 287 | UTC hour-of-day counts in YFData index: {5: 171, 4: 330} |
| `evidence/vbt-spec/S6-evidence/p4_pit_verify.json` | 1068 | JSON results; keys: min_hist=400 (lab default), min_hist=60 (live), min_hist=60 + max_stocks=50, min_hist=60 + max_sector_count=3, original_per_bar_s, original_full_tape_est_s |
| `evidence/vbt-spec/S6-evidence/p4_pit_verify.out` | 1034 | tape (1933, 646) UTC |
| `evidence/vbt-spec/S6-evidence/p4_pit_verify.py` | 3719 | Verify vectorised PIT mask == eligible_at on >=200 sampled dates, 4 configs. Time + memory. |
| `evidence/vbt-spec/S6-evidence/p5_mask_in_vbt.out` | 1742 | tape (1933, 646) |
| `evidence/vbt-spec/S6-evidence/p5_mask_in_vbt.py` | 5063 | PIT mask x vbt signals: entries&mask, exit-on-removal, delisted tails, memory/time. |
| `evidence/vbt-spec/S6-evidence/p5b_param_cols.out` | 275 | entries columns: ['rsi_window', 'symbol'] (1933, 100) \| mask columns: [None] (1933, 50) |
| `evidence/vbt-spec/S6-evidence/p5b_param_cols.py` | 1116 | Gotcha: indicator outputs carry a parameter column level; the mask must be broadcast across it. |
| `evidence/vbt-spec/S6-evidence/p6_tape_quality.out` | 1467 | dow counts (0=Mon): {0: 363, 1: 400, 2: 396, 3: 388, 4: 386} \| hours: [0] |
| `evidence/vbt-spec/S6-evidence/p6_tape_quality.py` | 1734 | Tape quality: weekday stamps, split-like discontinuities in the cached adjusted closes, volume, dup rows. |
| `evidence/vbt-spec/S6-evidence/p7_coverage.out` | 397 | year  member-cells  % with a close on that day |
| `evidence/vbt-spec/S6-evidence/p7_coverage.py` | 1064 | PIT member-day coverage: how much of the point-in-time universe has a price on that day? |
| `evidence/vbt-spec/S6-evidence/pit.py` | 2880 | S6 prototype: fully vectorised point-in-time universe mask (dates x symbols, bool). |
| `evidence/vbt-spec/S6-evidence/tsdata.py` | 2948 | S6 prototype: vbt.Data subclass reading the existing TimescaleDB cache READ-ONLY. |
| `evidence/vbt-spec/S8-evidence/constraints.txt` | 27 | pandas>=3.0.3 |
| `evidence/vbt-spec/S8-evidence/gh_meta.sh` | 778 | #!/bin/bash |
| `evidence/vbt-spec/S8-evidence/gh_meta_1.tsv` | 2478 | ranaroussi/quantstats	7801	lastcommit=2026-11-02	Apache-2.0	archived=False	issues=33	rel=v0.0.86 2026-09-27	Po |
| `evidence/vbt-spec/S8-evidence/gh_meta_2.tsv` | 2713 | TA-Lib/ta-lib-python	12372	lastcommit=2026-11-05	BSD-2-Clause	archived=False	issues=87	rel=v0.8.3 2026-10-28	P |
| `evidence/vbt-spec/S8-evidence/gh_meta_3.tsv` | 3068 | mlflow/mlflow	28697	lastcommit=2026-11-06	Apache-2.0	archived=False	issues=2187	rel=v3.18.0 2026-11-04	The ope |
| `evidence/vbt-spec/S8-evidence/gh_meta_4.tsv` | 1590 | eslazarev/purged-cross-validation	58	lastcommit=2026-10-25	MIT	archived=False	issues=3	rel=v0.1.11 2026-10-16	 |
| `evidence/vbt-spec/S8-evidence/gh_meta_all.tsv` | 9849 | ranaroussi/quantstats	7801	lastcommit=2026-11-02	Apache-2.0	archived=False	issues=33	rel=v0.0.86 2026-09-27	Po |
| `evidence/vbt-spec/S8-evidence/gh_pandas3_issues.txt` | 4365 | == ranaroussi/quantstats |
| `evidence/vbt-spec/S8-evidence/gh_search.tsv` | 1498 | ### purged kfold embargo combinatorial cross validation finance |
| `evidence/vbt-spec/S8-evidence/pypi_meta.py` | 782 | Fetch PyPI latest version, upload date, licence and pandas/numpy requirement lines for a package list. |
| `evidence/vbt-spec/S8-evidence/pypi_meta.tsv` | 2720 | vectorbt	1.2.0	2026-10-19	py<3.15,>=3.11		numpy>=2.4.6; pandas<4.0,>=3.0.3; numba>=0.66 |
| `evidence/vbt-spec/S8-evidence/resolve_1.txt` | 1267 | RESOLVES  quantstats \|  + numba==0.68.1  + numpy==2.5.4  + pandas==3.1.1 |
| `evidence/vbt-spec/S8-evidence/resolve_2.txt` | 2201 | RESOLVES  quantstats \|  + pandas==3.1.1  + quantstats==0.0.86 |
| `evidence/vbt-spec/S8-evidence/resolve_3.txt` | 252 | FAILS     pandas-ta \|   × No solution found when resolving dependencies:   ╰─▶ Because only vectorbt==1.2.0 is |
| `evidence/vbt-spec/S8-evidence/resolve_check.sh` | 700 | #!/bin/bash |
| `evidence/vbt-spec/S8-evidence/t10_mlflow.py` | 467 | Python script (no docstring); import mlflow, tempfile, pandas as pd, numpy as np, os |
| `evidence/vbt-spec/S8-evidence/t10_out.txt` | 43 | OK   mlflow 3.18.0 run logged; search -> 1 |
| `evidence/vbt-spec/S8-evidence/t11_dsr_pbo.py` | 1893 | Cross-library DSR / PSR / PBO consistency on one synthetic 48-config grid (null: no skill) and one with a planted winner. |
| `evidence/vbt-spec/S8-evidence/t11_out.txt` | 592 | OK   null: best=29 per-period SR=0.0583 SR0=0.0679  DSR manual=0.3807 purgedcv=0.3811 \| jsharpe PSR(SR,SR0)=0. |
| `evidence/vbt-spec/S8-evidence/t1_analytics.py` | 2156 | Python script (no docstring); import matplotlib; matplotlib.use("Agg") |
| `evidence/vbt-spec/S8-evidence/t1_out.txt` | 7362 | See https://pandas.pydata.org/docs/dev/whatsnew/v3.1.0.html#deprecation-inplace for more details. |
| `evidence/vbt-spec/S8-evidence/t2_cv_stats.py` | 3121 | Python script (no docstring); import numpy as np, pandas as pd |
| `evidence/vbt-spec/S8-evidence/t2_out.txt` | 2209 | OK   sklearn TimeSeriesSplit(gap): [(680, 63), (743, 63), (806, 63), (869, 63), (932, 63)] |
| `evidence/vbt-spec/S8-evidence/t3_out.txt` | 694 | ggTrader folds: 17 (Timestamp('2021-01-31 00:00:00'), Timestamp('2022-01-31 00:00:00'), Timestamp('2022-04-30  |
| `evidence/vbt-spec/S8-evidence/t3_skfolio_wf_vs_ggt.py` | 1796 | Can skfolio.WalkForward reproduce ggTrader's generate_folds (12m train / 3m test / 3m step, eval 2021-01-31..2026-04-30)? |
| `evidence/vbt-spec/S8-evidence/t4_indicators.py` | 2233 | Python script (no docstring); import numpy as np, pandas as pd, vectorbt as vbt |
| `evidence/vbt-spec/S8-evidence/t4_out.txt` | 875 | talib 0.8.3 funcs 237 |
| `evidence/vbt-spec/S8-evidence/t5_out.txt` | 689 | OK   skfolio HRP: HRP w sum=1.000 |
| `evidence/vbt-spec/S8-evidence/t5_portfolio_tracking.py` | 3458 | Python script (no docstring); import numpy as np, pandas as pd, vectorbt as vbt, tempfile, os |
| `evidence/vbt-spec/S8-evidence/t6_out.txt` | 97 | OK   skfolio rolling HRP weights -> vbt from_orders targetpercent: total_return=0.365 orders=200 |
| `evidence/vbt-spec/S8-evidence/t6_weights_to_vbt.py` | 796 | Python script (no docstring); import numpy as np, pandas as pd, vectorbt as vbt |
| `evidence/vbt-spec/S8-evidence/t7_out.txt` | 491 | ggTrader file: 1996-01-02 2026-06-02 2712 \| upstream: 1996-01-02 2026-08-18 2720 |
| `evidence/vbt-spec/S8-evidence/t7_pit_compare.py` | 1072 | Compare ggTrader's committed S&P500 PIT file against upstream fja05680 'Updated' CSV (same format). |
| `evidence/vbt-spec/S8-evidence/t8_out.txt` | 250 | files=12 distinct vbt.* symbols=8 missing_in_1.2.0=1 |
| `evidence/vbt-spec/S8-evidence/t8_skill_api_check.py` | 1488 | Extract vbt.* symbols and Portfolio method/kwarg names from the marketcalls skill and test them against installed vectorbt 1.2.0. |
| `evidence/vbt-spec/S8-evidence/t9_crosscheck.py` | 1426 | Independent cross-check: 25-line numpy ledger vs vbt from_signals (single symbol, all-in, fees, next-bar? same-bar close fills). |
| `evidence/vbt-spec/S8-evidence/t9_out.txt` | 271 | OK   numpy-ledger (fee convention: shares=cash/(p*(1+fee))) vs vbt from_signals max \|equity diff\| = 1.455e-11 |
| `evidence/vbt-spec/S8-evidence/t_common.py` | 980 | Shared synthetic data + safe-run helper for S8 compat probes. |
| `evidence/vbt-spec/coldstart.py` | 1948 | Python script (no docstring); import time, sys, gc, os |
| `evidence/vbt-spec/dry.log` | 24375 | Collecting ccxt (from -r req_new.txt (line 1)) |
| `evidence/vbt-spec/lead-checks/A.parquet` | 71421 | parquet frame 1968 x 80; columns ["('e', 'AFL')", "('e', 'ANSS')", "('e', 'BAC')", "('e', 'CDNS')"] |
| `evidence/vbt-spec/lead-checks/B.parquet` | 71407 | parquet frame 1968 x 80; columns ["('e', 'AFL')", "('e', 'ANSS')", "('e', 'BAC')", "('e', 'CDNS')"] |
| `evidence/vbt-spec/lead-checks/C.parquet` | 71348 | parquet frame 1968 x 80; columns ["('e', 'AFL')", "('e', 'ANSS')", "('e', 'BAC')", "('e', 'CDNS')"] |
| `evidence/vbt-spec/lead-checks/E.parquet` | 72703 | parquet frame 1968 x 80; columns ["('e', 'AFL')", "('e', 'ANSS')", "('e', 'BAC')", "('e', 'CDNS')"] |
| `evidence/vbt-spec/lead-checks/eq.py` | 2139 | Python script (no docstring); import sys, time, numpy as np, pandas as pd |
| `evidence/vbt-spec/lead-checks/eq_new.parquet` | 538568 | parquet frame 627 x 96; columns ["('plain', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14')", "('plain', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14')", "('plain', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14')", "('plain', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14')"] |
| `evidence/vbt-spec/lead-checks/eq_old.parquet` | 538586 | parquet frame 627 x 96; columns ["('plain', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14')", "('plain', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit1_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14')", "('plain', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold25_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14')", "('plain', 'ensemble__bb_period20_bb_std2.0_divergence_window10_ema_fast10_ema_slow50_macd_fast12_macd_signal9_macd_slow26_min_agree2_min_agree_exit2_rsi_oversold30_rsi_period14_vol_mult1.5_vol_period20_weekly_rsi_exit50_weekly_rsi_oversold30_weekly_rsi_period14')"] |
| `evidence/vbt-spec/lead-checks/pit_recheck.py` | 1167 | Python script (no docstring); import sys, time, json, pathlib |
| `evidence/vbt-spec/lead-checks/reg.py` | 910 | Python script (no docstring); import collections, inspect, numpy as np, pandas as pd |
| `evidence/vbt-spec/lead-checks/sig_arm.py` | 1000 | Python script (no docstring); import sys, numpy as np, pandas as pd |
| `evidence/vbt-spec/numba_cold.txt` | 656 | versions 1.2.0 3.1.1 2.5.4 0.68.1 import_s=2.73 |
| `evidence/vbt-spec/numba_warm.txt` | 1456 | versions 1.2.0 3.1.1 2.5.4 0.68.1 import_s=12.26 |
| `evidence/vbt-spec/req_new.txt` | 373 | ccxt |
| `evidence/vbt-spec/s1_engine_selfcheck.py` | 2795 | S1 prototype: explicit engine pin + startup self-check that proves Rust dispatch. |
| `evidence/vbt-spec/s1_engine_selfcheck_out.txt` | 315 | {'vectorbt': '1.2.0', 'vectorbt_rust': '1.2.0', 'engine': 'rust', 'rust_calls': 1} |
| `evidence/vbt-spec/s1_ensemble_contract_proto.py` | 3585 | S1 prototype: ensemble in the proposed contract (built-in vbt indicators, run once per |
| `evidence/vbt-spec/s1_ensemble_contract_proto_out.txt` | 205 | cols=24000 groups=48 ind=0.2s sig=0.7s pf=6.6s total=16.0s peakRSS=3857MB |
| `evidence/vbt-spec/s1_flat_sharpe_out.txt` | 64 | flat sharpe 1.2.0: inf |
| `evidence/vbt-spec/s2/cold.py` | 478 | Python script (no docstring); import time,sys |
| `evidence/vbt-spec/s2/contract_fingerprint.py` | 2198 | Prototype: emit numeric fingerprints of vbt definitions that ggTrader would rely on. |
| `evidence/vbt-spec/s2/dep.html` | 978192 | HTML report: dependency graph |
| `evidence/vbt-spec/s2/dep2.html` | 1493361 | HTML report: dependency graph |
| `evidence/vbt-spec/s2/fp_0285.err` | 0 | empty file |
| `evidence/vbt-spec/s2/fp_0285.json` | 911 | JSON results; keys: _versions, ann_ret, ann_vol, atr14_sma, bb_ddof1_lower, bb_default_lower |
| `evidence/vbt-spec/s2/fp_120.err` | 0 | empty file |
| `evidence/vbt-spec/s2/fp_120.json` | 910 | JSON results; keys: _versions, ann_ret, ann_vol, atr14_sma, bb_ddof1_lower, bb_default_lower |
| `evidence/vbt-spec/s2/proto/pyproject.toml` | 1875 | [build-system] |
| `evidence/vbt-spec/s2/proto/req.lock` | 73291 | # This file was autogenerated by uv via the following command: |
| `evidence/vbt-spec/s2/proto/src/ggTrader/__init__.py` | 0 | Python script (no docstring);  |
| `evidence/vbt-spec/s2/proto/uv.lock` | 296288 | version = 1 |
| `evidence/vbt-spec/s2/proto_pta/pyproject.toml` | 1886 | [build-system] |
| `evidence/vbt-spec/s2/proto_pta/src/ggTrader/__init__.py` | 0 | Python script (no docstring);  |
| `evidence/vbt-spec/s2/rs.json` | 116346 | JSON results; keys: info, last_serial, ownership, releases, urls, vulnerabilities |
| `evidence/vbt-spec/s2/ruffproto/ruff.toml` | 315 | line-length = 100 |
| `evidence/vbt-spec/s2/ruffproto/scripts/a.py` | 179 | Python script (no docstring); import vectorbt as vbt |
| `evidence/vbt-spec/s2/ruffproto/scripts/b.py` | 21 | Python script (no docstring); import vectorbt_rust |
| `evidence/vbt-spec/s2/ruffproto/src/ggTrader/engine/ok.py` | 179 | Python script (no docstring); import vectorbt as vbt |
| `evidence/vbt-spec/s2/test_boundary_proto.py` | 1230 | Prototype: AST boundary test -- vectorbt may only be touched inside ggTrader/engine/. |
| `evidence/vbt-spec/s2/vbt.json` | 91638 | JSON results; keys: info, last_serial, ownership, releases, urls, vulnerabilities |
| `evidence/vbt-spec/s4_grid_count.py` | 628 | Count raw and valid sweep combos per registry strategy (read-only). |

### Profiling of the legacy 2-fold WFO

4 files under `profiling/`.

| File | Bytes | What it is |
|---|---:|---|
| `profiling/prof_wfo.py` | 2735 | Throwaway profile of the current equity WFO path (SP500 ensemble, full grid, N folds). |
| `profiling/run.log` | 3541 | folds: 2 |
| `profiling/timing.json` | 68 | JSON results; keys: load_s, wfo_s, grid |
| `profiling/wfo.prof` | 1992235 | cProfile dump (binary; read with pstats) |

### Legacy reference runs (P0 task 4)

36 files under `legacy-references/`.

| File | Bytes | What it is |
|---|---:|---|
| `legacy-references/_legacy_reference_core17_attempt1_oom6g_note.json` | 1137 | Manual note: first 17-fold attempt killed by the OOM killer at the 6 GB cap (fold 7), and the fix |
| `legacy-references/_legacy_reference_core17_dsr0_vs_20260923.json` | 7213 | JSON results; keys: new, old, spy_sharpe_new, spy_sharpe_old, gates_new, gates_old |
| `legacy-references/_legacy_reference_core17_ensemble_sp500.json` | 65911 | legacy reference run core17_ensemble_sp500; ok=True; OOS Sharpe 0.39, gates 1/17; 1421 s |
| `legacy-references/_legacy_reference_core17_ensemble_sp500_dsr0.json` | 65887 | legacy reference run core17_ensemble_sp500_dsr0; ok=True; OOS Sharpe 0.62, gates 17/17; 1363 s |
| `legacy-references/_legacy_reference_core17_vs_20260923.json` | 7261 | JSON results; keys: new, old, spy_sharpe_new, spy_sharpe_old, gates_new, gates_old |
| `legacy-references/_legacy_reference_golden_ensemble_signals.json` | 4517 | legacy reference run golden_ensemble_signals; ok=True; 8 s |
| `legacy-references/_legacy_reference_golden_live_signals.json` | 1144 | legacy reference run golden_live_signals; ok=True; 135 s |
| `legacy-references/_legacy_reference_ref_bb_reversion.json` | 7811 | legacy reference run ref_bb_reversion; ok=True; OOS Sharpe 1.02, gates 0/2; 185 s |
| `legacy-references/_legacy_reference_ref_conviction_bb.json` | 12291 | legacy reference run ref_conviction_bb; ok=True; OOS Sharpe 0.29, gates 0/2; 326 s |
| `legacy-references/_legacy_reference_ref_cross_asset_sleeve.json` | 6877 | legacy reference run ref_cross_asset_sleeve; ok=True; OOS Sharpe -0.59, gates 0/2; 17 s |
| `legacy-references/_legacy_reference_ref_cross_asset_sleeve_17fold.json` | 20234 | legacy reference run ref_cross_asset_sleeve_17fold; ok=True; OOS Sharpe 0.93, gates 4/17; 21 s |
| `legacy-references/_legacy_reference_ref_ema_cross.json` | 7835 | legacy reference run ref_ema_cross; ok=True; OOS Sharpe -0.60, gates 0/2; 167 s |
| `legacy-references/_legacy_reference_ref_macd_divergence.json` | 9213 | legacy reference run ref_macd_divergence; ok=True; OOS Sharpe -0.29, gates 0/2; 172 s |
| `legacy-references/_legacy_reference_ref_retail_attention.json` | 8053 | legacy reference run ref_retail_attention; ok=True; 259 s |
| `legacy-references/_legacy_reference_ref_rsi_reversion.json` | 9539 | legacy reference run ref_rsi_reversion; ok=True; OOS Sharpe 0.33, gates 0/2; 185 s |
| `legacy-references/_legacy_reference_ref_volume_bb_reversion.json` | 9547 | legacy reference run ref_volume_bb_reversion; ok=True; OOS Sharpe 0.31, gates 0/2; 175 s |
| `legacy-references/_legacy_reference_ref_xs_momentum.json` | 8896 | legacy reference run ref_xs_momentum; ok=True; OOS Sharpe -0.62, gates 0/2; 182 s |
| `legacy-references/_legacy_reference_smoke2_ensemble_sp500.json` | 32463 | legacy reference run smoke2_ensemble_sp500; ok=True; OOS Sharpe 0.62, gates 0/2; 363 s |
| `legacy-references/_legacy_reference_spy_bh_17fold_window.json` | 1591 | legacy reference run spy_bh_17fold_window; ok=True; 3 s |
| `legacy-references/_legacy_reference_spy_bh_smoke_window.json` | 1595 | legacy reference run spy_bh_smoke_window; ok=True; 6 s |
| `legacy-references/core17_attempt1_oom6g.log` | 3059 | Console log of the first 17-fold attempt, killed by the OOM killer at the 6 GB cap |
| `legacy-references/core17_ensemble_sp500_dsr0_oos_equity.csv` | 47761 | Out-of-sample equity curve of run core17_ensemble_sp500_dsr0 (date, value; 1065 bars) |
| `legacy-references/core17_ensemble_sp500_oos_equity.csv` | 47594 | Out-of-sample equity curve of run core17_ensemble_sp500 (date, value; 1065 bars) |
| `legacy-references/live_golden_legacy.json` | 1210221 | Live golden: buys, sells, last_exit for each of the last 120 sessions (legacy engine) |
| `legacy-references/ref_bb_reversion_oos_equity.csv` | 5605 | Out-of-sample equity curve of run ref_bb_reversion (date, value; 125 bars) |
| `legacy-references/ref_conviction_bb_oos_equity.csv` | 5570 | Out-of-sample equity curve of run ref_conviction_bb (date, value; 125 bars) |
| `legacy-references/ref_cross_asset_sleeve_17fold_oos_equity.csv` | 47347 | Out-of-sample equity curve of run ref_cross_asset_sleeve_17fold (date, value; 1065 bars) |
| `legacy-references/ref_cross_asset_sleeve_oos_equity.csv` | 5503 | Out-of-sample equity curve of run ref_cross_asset_sleeve (date, value; 125 bars) |
| `legacy-references/ref_ema_cross_oos_equity.csv` | 5514 | Out-of-sample equity curve of run ref_ema_cross (date, value; 125 bars) |
| `legacy-references/ref_macd_divergence_oos_equity.csv` | 5499 | Out-of-sample equity curve of run ref_macd_divergence (date, value; 125 bars) |
| `legacy-references/ref_retail_attention_oos_equity.csv` | 4271 | Out-of-sample equity curve of run ref_retail_attention (date, value; 125 bars) |
| `legacy-references/ref_rsi_reversion_oos_equity.csv` | 5555 | Out-of-sample equity curve of run ref_rsi_reversion (date, value; 125 bars) |
| `legacy-references/ref_volume_bb_reversion_oos_equity.csv` | 5548 | Out-of-sample equity curve of run ref_volume_bb_reversion (date, value; 125 bars) |
| `legacy-references/ref_xs_momentum_oos_equity.csv` | 5511 | Out-of-sample equity curve of run ref_xs_momentum (date, value; 125 bars) |
| `legacy-references/signal_golden_legacy.npz` | 163082 | npz arrays: index_ns, columns, bb_entries, bb_exits, rsi_entries, rsi_exits |
| `legacy-references/smoke2_ensemble_sp500_oos_equity.csv` | 5580 | Out-of-sample equity curve of run smoke2_ensemble_sp500 (date, value; 125 bars) |


### Files outside this folder created or changed by P0

| File | What it is |
|---|---|
| `requirements-legacy.lock` | Exact pip freeze of the legacy environment (vectorbt 0.28.5, pandas 2.3.3, numpy 2.2.6, numba 0.61.2), 171 pins |
| `scripts/legacy_env.sh` | Rebuild the legacy environment from the tag `legacy-engine-vbt0285` plus the lock (handles `--help`) |
| `scripts/make_tape_snapshot.py` | Export and verify the hashed read-only Parquet tape snapshot; also the snapshot reader `load_snapshot_wide` |
| `data/snapshots/MANIFEST.json` | Tracked manifest: path, rows, symbols, dates, sha256, creation command, sanity flags |
| `data/snapshots/ohlcv_20261106.parquet` | The frozen tape (git-ignored, rebuild with the script) |
| `scripts/legacy_reference_runs.py` | Resumable runner for every legacy reference run |
| `scripts/legacy_reference_compare.py` | Fold-by-fold comparison of a fresh 17-fold run with an older results JSON |
| `research/RESEARCH_INDEX.md` | One study-dossier row added for this folder |
| `.gitignore` | One rule: `data/snapshots/*.parquet` |
