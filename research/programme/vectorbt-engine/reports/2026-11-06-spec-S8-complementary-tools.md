# S8: Open-source tools that complement vectorbt for the ggTrader lab

Date: 2026-11-06. Read-only on `/home/flynn/ggTrader`. Evidence: `S8-evidence/` (all scripts, raw outputs `t*_out.txt`, metadata `gh_meta_all.tsv`, `pypi_meta.tsv`, `resolve_*.txt`, `venv_freeze.txt`).
Test venv: `S8-evidence/venv` = Python 3.12, **vectorbt 1.2.0, pandas 3.1.1, numpy 2.5.4, numba 0.68.1** (all tools below were installed *together* with these, under `-c constraints.txt` pandas>=3.0.3, numpy>=2.4.6). Stars, commit dates, licences and releases were read from the GitHub API (`gh api`) and the PyPI JSON API on 2026-11-06. "TESTED" = installed in that venv and a tiny example executed (outputs in `t*_out.txt`). Synthetic data only; none of this proves fitness on the real SP500 tape.

## 0. Findings the spec writer must see first

1. **vectorbt itself is not OSI-licensed.** `polakowo/vectorbt` LICENSE.md is "Apache 2.0 + Commons Clause" (GitHub reports `NOASSERTION`; text read via `gh api repos/polakowo/vectorbt/license`). Commons Clause bars *selling* a product whose value derives from the software. Private research and a personal paper trader are unaffected, but it strictly fails the owner's "OSI licence" rule. The owner should record this as a conscious exception. Source: https://github.com/polakowo/vectorbt/blob/master/LICENSE.md
2. vectorbt cadence is healthy: v1.0.0 2026-04-22, v1.1.0 07-05, v1.1.1 09-26, v1.1.2 10-09, v1.2.0 10-19 (https://github.com/polakowo/vectorbt/releases); master has commits to 2026-11-03 and a `docs/generated/release-notes/v1.2.1.md`; PyPI still shows 1.2.0 (https://pypi.org/project/vectorbt/). A 2026-11-03 master commit is titled "fix(returns): return NaN or signed infinity for zero-volatility Sharpe", i.e. upstream is touching the flat-curve Sharpe trap noted in F (commit title only, behaviour UNVERIFIED, not in 1.2.0).
3. **Everything in the "ADOPT/EVALUATE" lists below resolves and imports on pandas 3.1.1 / numpy 2.5.4 next to vectorbt 1.2.0.** Resolver-level failures (pandas<3 pins): `mlfinpy`, original `pandas-ta`, and `zipline-reloaded` latest (the resolver silently fell back to 3.0.4; 3.2.1 declares `pandas<3.0`, https://pypi.org/project/zipline-reloaded/).
4. **ggTrader's PIT file is a copy of fja05680/sp500 and is stale and slightly wrong.** `data/universe/sp500_constituents_history.csv.gz` vs upstream "Updated" CSV (`t7_pit_compare.py`, `t7_out.txt`): 2,711 shared dates identical except **2023-10-18** (3 tickers each way), which upstream moved to 2023-12-18 in PR #29 (2026-07-05, "Fix effective date of Dec 2023 rebalance"). Upstream has 9 extra dates and runs to 2026-08-18; ggTrader's ends 2026-06-02. Cheap, high-value refresh (section 5).
5. **Annualisation convention is the common trap across every analytics tool.** Same return series (`t9_out.txt`): vbt with `year_freq='252 days'` = empyrical = quantstats = -0.626010; ffn -0.626323; **vbt default (365) -0.753403**. The `ret.vbt.returns().qs` adapter gave -0.952 vs quantstats direct -0.791 on another series, so never use the adapter's defaults without explicit `periods`. Whatever tool is chosen, wrap it with explicit 252.

## 1. Performance/risk analytics and tear sheets

| Repo | Stars | Last commit | Licence | Latest release | pandas 3 / numpy 2.4 | Verdict |
|---|---|---|---|---|---|---|
| [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) | 7,801 | 2026-11-02 | Apache-2.0 | v0.0.86, 2026-09-27 | TESTED OK (stats, `reports.metrics` vs benchmark, `reports.html`); emits pandas 3.1 `inplace` FutureWarnings; open issue #582 (matplotlib tick warning, pandas 3.1), #584 (benchmark pairing across different periods) | **ADOPT** (report layer) |
| [stefan-jansen/empyrical-reloaded](https://github.com/stefan-jansen/empyrical-reloaded) | 132 | 2026-11-02 | Apache-2.0 | 0.7.2, 2026-11-05 | TESTED OK; 4 pandas-3 fixes closed 10-15..17 (#60, #67, #89, #109) | SKIP (duplicate; keep as optional test oracle) |
| [stefan-jansen/pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded) | 622 | 2026-11-02 | Apache-2.0 | 0.10.2, 2026-11-02 | TESTED OK (`perf_stats`, `create_returns_tear_sheet` returns a figure); its metadata demands pandas>=3.0.6 | SKIP (matplotlib tear sheet, overlaps quantstats) |
| [pmorissette/ffn](https://github.com/pmorissette/ffn) | 2,695 | 2026-11-04 | MIT | v1.4.1, 2026-10-10 | TESTED OK (`calc_stats`); Sharpe differs from the others in the 4th decimal (own annualisation) | SKIP (use only if `bt` is used) |
| [skfolio/skfolio](https://github.com/skfolio/skfolio) | 2,595 | 2026-10-29 | BSD-3 | v1.12.5, 2026-10-29 | TESTED OK (section 2, 7) | see sections 2, 7 |
| [dcajasn/Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | 4,590 | 2026-11-04 | BSD-3 | no GitHub releases (PyPI 7.4.1, 2026-11-04) | TESTED OK | SKIP (portfolio optimiser; ggTrader has none) |

- quantstats: reads a returns Series. Integration with vectorbt: `pf.returns()` -> `qs.reports.html(ret, benchmark=spy_ret, periods_per_year=252)` / `qs.reports.metrics(..., mode="full")`. vectorbt 1.2.0 already ships a `.vbt.returns().qs` adapter (`QSAdapter`, verified present) but see finding 5. Quantstats Sharpe == empyrical == vbt(252d) to 6 digits (`t9_out.txt`).
- Maintenance risk: single maintainer (ranaroussi, also yfinance), pre-1.0 version number, 33 open issues, two open correctness-flavoured ones. Mitigation: align strategy and SPY returns to one index yourself before passing them in; keep ggTrader's own metric code as the gate source of truth and use quantstats only for human-facing reports.
- Not recommended as the metric source for gates: ggTrader's Sortino/CAGR definitions differ from library defaults (F, section b).

## 2. Walk-forward / cross-validation with purging and embargo

| Repo | Stars | Last commit | Licence | Release | pandas 3 | Verdict |
|---|---|---|---|---|---|---|
| skfolio `model_selection`: `WalkForward`, `CombinatorialPurgedCV` | 2,595 | 2026-10-29 | BSD-3 | v1.12.5 | TESTED OK | **EVALUATE** |
| [eslazarev/purged-cross-validation](https://github.com/eslazarev/purged-cross-validation) (`purgedcv`) | 58 | 2026-10-25 | MIT | v0.1.11, 2026-10-16 | TESTED OK (import, CPCV/PBO/DSR) | EVALUATE (PBO/DSR only) |
| [sam31415/timeseriescv](https://github.com/sam31415/timeseriescv) | 290 | 2022-02-15 | MIT | PyPI 0.2, 2018 | installs/imports; abandoned | SKIP |
| [scikit-learn TimeSeriesSplit](https://github.com/scikit-learn/scikit-learn) | 68,022 | 2026-11-05 | BSD-3 | 1.9.1 | TESTED OK (`gap`, `test_size`) | already transitive; use for plain gaps only |
| [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab) | 4,968 | 2021-12-01 | proprietary EULA (LICENSE.txt read) | none | n/a | SKIP (not open source) |
| [baobach/mlfinpy](https://github.com/baobach/mlfinpy) | 92 | 2026-10-11 | MIT | 0.1.2, 2024-10-09 | **FAILS**: depends on pandas<3, numba<0.61 | SKIP |
| vectorbt-specific OSS splitters | none found | | | | | GitHub search and web search found none; purged/embargo CV is PRO-only (https://vectorbt.pro/features/optimization/time-series-cross-validation/) |

What was verified (`t2_out.txt`, `t3_skfolio_wf_vs_ggt.py`, `t3_out.txt`):
- `skfolio.model_selection.CombinatorialPurgedCV(n_folds=8, n_test_folds=2, purged_size=5, embargo_size=5)` on 1,000 bars: 28 splits, 7 test paths. This is real de Prado CPCV that vectorbt OSS lacks.
- `skfolio.model_selection.WalkForward` is **calendar-aware** (`freq="QS"`, `train_size=pd.DateOffset(months=12)`, `purged_size`, `expand_train`). It does NOT reproduce ggTrader's `generate_folds`: on 2021-01-31..2026-04-30 it gives **16 folds anchored to period starts vs ggTrader's 17 anchored to month-end-31**, so boundaries differ. Keep `generate_folds` (17 lines) as the fold definition; use skfolio only to add purge/embargo or CPCV.
- Integration point: both splitters yield integer positions over a DatetimeIndex frame, so each (train, test) pair slices the close/entries frames before `Portfolio.from_signals`, or feeds `close.vbt.range_split(start_idxs, end_idxs)` (F: positions, index lost).
- Risk: skfolio had a CPCV bug "never tests the trailing observations" (#413, closed 2026-10-10), so pin >=1.12.5 and unit-test fold coverage. skfolio is a heavy dependency (scikit-learn, cvxpy, plotly stack); import it lazily in a research extra.
- Honest limit: the lab selects on a *fixed monthly* walk-forward with the NDH gate. CPCV answers a different question (distribution of OOS paths); only worth adding if the owner wants PBO-style evidence, not as a replacement.

## 3. Technical indicators

| Repo | Stars | Last commit | Licence | Latest release | pandas 3 / numpy 2.4 | vectorbt wrapper (1.2.0) | Verdict |
|---|---|---|---|---|---|---|---|
| [TA-Lib/ta-lib-python](https://github.com/TA-Lib/ta-lib-python) | 12,372 | 2026-11-05 | BSD-2 | v0.8.3, 2026-10-28 | TESTED OK: PyPI wheel `ta-lib 0.8.3` imports with **no system C library**, 237 functions | `IndicatorFactory.from_talib('RSI'/'BBANDS'/'ATR')` and `vbt.talib('MACD')` TESTED OK (multi-param, multi-column) | **ADOPT as test oracle**; EVALUATE for production |
| [xgboosted/pandas-ta-classic](https://github.com/xgboosted/pandas-ta-classic) | 481 | 2026-10-31 | MIT | 0.9.0, 2026-10-28 | TESTED OK | `IndicatorFactory.from_pandas_ta('RSI')` TESTED OK: vectorbt 1.2.0 source tries `import pandas_ta` then falls back to `pandas_ta_classic` (`indicators/factory.py:3630-3634`) | EVALUATE |
| original `pandas-ta` | repo not found by API (`twopirllc/pandas-ta` 404) | | | 0.4.71b0, 2025-09-14 (beta) | **FAILS** with vectorbt 1.2.0: pins `numba==0.61.2` | n/a | SKIP |
| [bukosabino/ta](https://github.com/bukosabino/ta) | 5,250 | 2026-03-18 | MIT | PyPI 0.11.0, **2023-11-02** | TESTED OK (RSI) | `from_ta('RSIIndicator')` TESTED OK | SKIP (stale release, 160 issues) |
| [peerchemist/finta](https://github.com/peerchemist/finta) | 2,266 | 2022-07-24 | LGPL-3.0 | 1.3, 2021 | resolves | none | SKIP (archived) |
| [tulipcharts/tulipy](https://github.com/tulipcharts/tulipy) | 377 | 2019-04-11 | LGPL-3.0 | 0.4.0, 2019 | resolves | none | SKIP (unmaintained) |

Parity measured on synthetic bars (`t4_out.txt`, RSI-14, bars >200): TA-Lib vs **pandas-ta-classic 5.0e-14**; TA-Lib vs vbt `RSI(window=27, ewm=True)` 3.6e-5; TA-Lib vs `ta` 1.1e-5 (seeding differences). Consequence: if the lab wants a reference implementation to *test* its hand-written Wilder RSI/ATR/MACD against, TA-Lib is the industry oracle and costs a wheel. Moving the **live** signal path to TA-Lib is a signal change and falls under F's Phase 2c gate (0 entry/exit flips on the full tape), so do not do it without that test. ggTrader's `[bb, vbb, divergence, strength]` voters are not TA-Lib functions anyway.
Maintenance risk: TA-Lib is a thin wrapper over a C library with active 2026 releases (low). pandas-ta-classic is a community fork created after `pandas-ta` was abandoned (issue #30 in its repo asks what happened to the original); one-maintainer risk (medium).
Remove from ggTrader: the dead `pandas-ta` dependency (F: 0 imports; it also blocks numba upgrades).

## 4. Statistical validation of backtests

| Repo | Stars | Last commit | Licence | Release | pandas 3 | Verdict |
|---|---|---|---|---|---|---|
| [tschm/jsharpe](https://github.com/tschm/jsharpe) | 25 | 2026-11-03 | MIT | v0.6.3, 2026-07-09 | TESTED OK | **EVALUATE** (PSR, expected max Sharpe, min track record, FDR/Holm/Bonferroni, Sharpe variance with autocorrelation) |
| purgedcv (above) | 58 | 2026-10-25 | MIT | v0.1.11 | TESTED OK | EVALUATE (PBO via CSCV, `effective_n_trials`, DSR) |
| [bashtage/arch](https://github.com/bashtage/arch) | 1,667 | 2026-11-06 | NCSA (OSI-approved; LICENSE.md read, GitHub shows NOASSERTION) | v8.0.0, 2025-10-21 | TESTED OK (`SPA`, `StationaryBootstrap.conf_int`) | **EVALUATE** |
| [esvhd/pypbo](https://github.com/esvhd/pypbo) | 145 | 2026-07-06 | **AGPL-3.0** | not on PyPI | UNVERIFIED | SKIP (AGPL, no package) |
| [mnemox-ai/deflated-sharpe](https://github.com/mnemox-ai/deflated-sharpe) | 9 | 2026-03-21 | Apache-2.0 | v0.1.0 | imports | SKIP (see below) |
| scipy.stats.bootstrap | n/a | | BSD | | TESTED OK | already a dependency; enough for Sharpe CIs |

Cross-library check on a synthetic 48-config grid (`t11_out.txt`): best-config DSR (null / planted-edge) = manual Bailey and López de Prado formula with *non-excess* kurtosis **0.3807 / 0.9361**; purgedcv **0.3811 / 0.9363**; jsharpe PSR against the same SR0 **0.3808 / 0.9369**. So these two reproduce the textbook DSR to about 1e-3 and use the correct kurtosis (the bug F found in vbt 0.28.5). `deflated-sharpe` (mnemox) returned `(21.96, p=0.0)` for the same inputs under my best-guess annualised-SR call, so either its contract is non-obvious or it is wrong: not worth the time. PBO via CSCV: 0.59 (null, high = overfit) vs 0.15 (planted edge), directionally right.
- Integration: all take plain numpy arrays of per-period returns (`pf.returns().to_numpy()`; for the grid, a `(n_configs, T)` matrix from the sweep). They sit *beside* `gates.py`, not in vectorbt. Use them first as an **independent cross-check of ggTrader's own DSR** (`gates.py:205-210`, per-period scaling) before replacing it; PBO and SPA/MCS are new evidence ggTrader does not have for the 48-combo grid.
- `arch.bootstrap.SPA` (Hansen) tests whether the best of many rules beats a benchmark, which maps directly to "does the best of 48 combos beat SPY after data snooping". Releases lag (v8.0.0 is 12 months old) but master is active (commit 2026-11-06).
- Maintenance risk: jsharpe and purgedcv are young, small-star, mostly single-author projects. They are small pure-numpy modules, so vendoring the one function you need is a fallback.

## 5. Point-in-time membership and fundamentals/data sources

| Source | Stars | Last commit | Licence | Role | Caveats |
|---|---|---|---|---|---|
| [fja05680/sp500](https://github.com/fja05680/sp500) | 961 | 2026-09-07 | MIT | S&P 500 membership 1996 to 2026-08-18; **already ggTrader's source** | Ticker symbols only (no permanent IDs), so ticker reuse and renames are ambiguous (README note 2: a rename shows as removal). Origin is a file from Andreas Clenow's book; later maintained from Wikipedia (README notes 1, 4). No prices for delisted names; README says Yahoo "only keeps current stock history". Dates have had errors (the 2023-12-18 fix). |
| [yfiua/index-constituents](https://github.com/yfiua/index-constituents) | 67 | 2026-11-01 | Apache-2.0 | monthly snapshots since 2023/07 for sp500, nasdaq100 | too short for PIT; useful only as a *second opinion* for 2023+ |
| [hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents) | 31 | 2025-08-24 | MIT | same 1996+ idea | derivative of the same lineage; no new information |
| [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies) | 675 | 2026-11-05 | none shown | current list only | not PIT |
| [dgunning/edgartools](https://github.com/dgunning/edgartools) | 2,838 | 2026-11-05 | MIT | SEC EDGAR filings, XBRL financials, filing dates | TESTED import OK on pandas 3.1.1 (`resolve_2.txt`); needs `set_identity(email)` per SEC fair-access; 104 open issues; **filing date is the PIT stamp, never use period-end** |
| [jadchaar/sec-edgar-downloader](https://github.com/jadchaar/sec-edgar-downloader) | 720 | 2026-02-02 | MIT | raw filing download | no parsing; SKIP in favour of edgartools |
| [mortada/fredapi](https://github.com/mortada/fredapi) | 1,666 | 2026-01-28 | Apache-2.0 | FRED **and ALFRED vintages** (`realtime_start/end`), the only free PIT macro series | last release 0.5.2 in 2024-05; import OK on pandas 3.1.1; needs a free API key |
| [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) | 25,837 | 2026-11-03 | Apache-2.0 | prices (already used) | README: "intended for research and educational purposes", refer to Yahoo's terms, "API is intended for personal use only"; data is survivor-only |
| [alpacahq/alpaca-py](https://github.com/alpacahq/alpaca-py) | 1,583 | 2026-11-05 | Apache-2.0 | broker + bars (already in live path) | import resolves with pandas 3.1.1 |
| [OpenBB-finance/OpenBB](https://github.com/OpenBB-finance/OpenBB) | 75,127 | 2026-11-02 | Apache-2.0 (LICENSE read; GitHub shows NOASSERTION) | aggregator | SKIP: large dependency surface, mostly wraps sources above; many providers need keys |

Verdicts: **ADOPT** a refresh of `sp500_constituents_history.csv.gz` from upstream fja05680 (finding 4; keep it vendored, do not add a runtime dependency). **EVALUATE later** `edgartools` and `fredapi` only if a fundamentals or macro sleeve is approved (earlier factor studies were already NO-GO with free data). **Survivorship caveat unchanged:** no free source supplies delisted-name prices; PIT membership fixes the *universe*, not the *price tape*.

## 6. Experiment tracking and result storage

| Repo | Stars | Last commit | Licence | Release | Verdict for a single-machine lab |
|---|---|---|---|---|---|
| [duckdb/duckdb](https://github.com/duckdb/duckdb) + [apache/arrow](https://github.com/apache/arrow) (pyarrow 26.0.1) | 42,846 / 17,275 | 2026-11-05 / 2026-11-06 | MIT / Apache-2.0 | v2.0.0 (2026-10-21) / 26.0.1 | **EVALUATE**: TESTED OK on pandas 3.1.1: Parquet write via pyarrow, `duckdb.sql("select ... from 'x.parquet'")`, replacement scan of a pandas 3 DataFrame, `INSTALL postgres` extension. Attaching to TimescaleDB UNVERIFIED (no credentials used). |
| [optuna/optuna](https://github.com/optuna/optuna) | 15,010 | 2026-11-04 | MIT | v5.0.0, 2026-09-07 | EVALUATE only if grids outgrow 48 combos: TESTED a TPE study over `vbt.MA` crossovers + `Portfolio.from_signals` (25 trials, best Sharpe 1.08). Caution: TPE finds the best-looking cell faster, which *raises* snooping; DSR trial count must be the number of trials run. |
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | 28,697 | 2026-11-06 | Apache-2.0 | v3.18.0, 2026-11-04 | SKIP: TESTED OK (sqlite store, log params/metrics) on pandas 3.1.1 but pulls a very large dependency tree and 2,187 open issues, for a job `lab_runs` + JSON already does |
| [iterative/dvc](https://github.com/iterative/dvc) | 15,963 | 2026-11-06 | Apache-2.0 | 3.67.1, 2026-03-31 | SKIP: resolves; data versioning not needed (`data/` is off-limits and git-tracked artefacts are small) |
| [aimhubio/aim](https://github.com/aimhubio/aim) | 6,293 | 2025-12-31 | Apache-2.0 | v3.29.1, 2025-05-08 | SKIP (stale) |
| [wandb/wandb](https://github.com/wandb/wandb) | 11,312 | 2026-11-05 | MIT | v0.31.0 | SKIP (hosted-service oriented) |

Fit: TimescaleDB already holds `lab_runs`/`lab_*`. The only gap is the raw per-trial artefacts that CLAUDE.md wants next to each report (JSON). Parquet + DuckDB queries over those files fits "thin": no server, one wheel each. Do not duplicate `lab_*` tables.

## 7. Portfolio construction that consumes vectorbt returns

| Repo | Stars | Last commit | Licence | Release | Result |
|---|---|---|---|---|---|
| skfolio | 2,595 | 2026-10-29 | BSD-3 | v1.12.5 | TESTED: `HierarchicalRiskParity`, `MeanRisk(CVAR)` fit on pandas 3 returns; rolling monthly HRP weights -> `vbt.Portfolio.from_orders(size=W, size_type="targetpercent", call_seq="auto", group_by=True, cash_sharing=True)` ran, 200 orders (`t6_out.txt`) |
| Riskfolio-Lib | 4,590 | 2026-11-04 | BSD-3 | 7.4.1 | TESTED: `Portfolio.optimization(model='Classic', rm='MV', obj='Sharpe')` |
| [robertmartin8/PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) | 6,149 | 2026-07-07 | MIT | v1.6.0, 2026-02-26 | TESTED: `EfficientFrontier.max_sharpe` on prices |
| [cvxpy/cvxpy](https://github.com/cvxpy/cvxpy) | 6,388 | 2026-11-04 | Apache-2.0 | v1.9.3, 2026-09-19 | resolves; transitive of the above |

Verdict: **SKIP all for now.** ggTrader has no portfolio-optimisation problem (equal/percent sizing, vol overlay, 5% sweep sleeve; equity-only diversification is a closed NO-GO per CLAUDE.md). If one ever appears, skfolio is the single choice (already needed for CV) and the weights->`from_orders(targetpercent)` bridge above is the integration point. Do not add Riskfolio and PyPortfolioOpt as well.

## 8. Sibling engines as cross-checks (not replacements)

| Repo | Stars | Last commit | Licence | One line |
|---|---|---|---|---|
| [pmorissette/bt](https://github.com/pmorissette/bt) | 3,015 | 2026-11-05 | MIT | Weights/rebalance-oriented; TESTED on pandas 3.1.1 (monthly equal weight, `bt.run`). Good independent check of the 5%-sleeve and SPY-blend accounting, useless for per-signal entries/stops. EVALUATE (dev only). |
| [kernc/backtesting.py](https://github.com/kernc/backtesting.py) | 9,199 | 2026-08-05 | **AGPL-3.0** | Event loop, single asset; copyleft; skip |
| [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) | 2,002 | 2026-11-02 | Apache-2.0 | Event-driven, bundle ingest; latest release declares `pandas<3.0`, so it cannot share the environment. Skip |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | 31,287 | 2026-11-06 | LGPL-3.0 | Production Rust engine with live parity; huge scope for a daily-bar lab; skip |
| [mementum/backtrader](https://github.com/mementum/backtrader) | 23,670 | 2023-04-19 | GPL-3.0 | Dormant; skip |
| [vnpy/vnpy](https://github.com/vnpy/vnpy) | 46,252 | 2026-10-06 | MIT | China-market/futures trading framework; irrelevant |

**Independent cross-check that actually works (recommended):** a 25-line numpy ledger is the best oracle for vbt `from_signals`. `t9_crosscheck.py` reproduces vbt equity on a single symbol with fees to **1.5e-11** max absolute difference once vbt's fee convention is matched (shares = cash / (price x (1+fee))). My first attempt with shares = cash x (1-fee)/price was off by 1.3, which shows why such a ledger belongs in the test suite (it documents the convention). Extend it to ATR/trailing stops and the PIT mask on a handful of symbols and diff against `pf.value()`; for returns metrics, cross-check Sharpe/Sortino/CAGR across vbt(252d), empyrical and quantstats (all agree to 6 digits in `t9_out.txt`).

## 9. AI-agent-friendly assets for vectorbt

| Asset | Stars | Last commit | Licence | Assessment |
|---|---|---|---|---|
| [marketcalls/vectorbt-backtesting-skills](https://github.com/marketcalls/vectorbt-backtesting-skills) | 220 | 2026-07-12 | **none** (no LICENSE file; API 404) | Claude Code skills via `npx skills add`. Read 12 of 42 files (`S8-evidence/dl/skills`). Pinned `vectorbt==0.28.4`, `pandas==2.3.3`, `numpy==2.4.2` in requirements.txt (not 1.2.0, not pandas 3). Optimised for Indian markets: depends on `openalgo` indicators and Rupee fee models. Its walk-forward template is a Python loop over (fast, slow) picking **max total return** with no gates, purge or trial accounting; `performance-analysis.md` never mentions the 365 vs 252 annualisation default. API check against 1.2.0 (`t8_check`, `t8_out.txt`): 8 distinct `vbt.*` symbols, all exist (the one "missing" hit is a `.vbt.to_returns()` accessor that does exist); 15 `from_signals` and 11 `from_orders` kwargs all valid; 10 `pf.*()` methods all exist. So low hallucination risk on core calls, but a **methodology** risk (return-max selection) and no licence. **SKIP as installed skill; at most read for ideas.** |
| [wshobson/maverick-mcp](https://github.com/wshobson/maverick-mcp) | 719 | 2026-10-30 | MIT | MCP server (38 core tools, 12 optional backtesting tools) for Yahoo data and analysis; its README says "educational and informational use". Its backtesting engine is its own module (`maverick/backtesting/engine.py`); vectorbt use not established. Not a vectorbt asset. SKIP. |
| [OnePunchMonk/AgentQuant](https://github.com/OnePunchMonk/AgentQuant) | 220 | 2026-11-03 | not checked | "failure-aware harness for reproducible quantitative-strategy research" (repo description only). UNVERIFIED relevance; not evaluated. |
| `polakowo/vectorbt` repo | 9,700 | 2026-11-03 | Commons Clause | Tree has `docs/`, `examples/`, `rust/`, `benchmarks/`; **no** llms.txt, AGENTS.md, skill or MCP (tree grep). Agent tooling is PRO-only (consistent with D). |
| Dreamsorcerer/vectorbt | 0 | 2026-03-26 | Commons Clause | Fork described as the community edition of PRO; 0 stars, unaudited. SKIP. |

Recommendation: for agent-friendliness, write ggTrader's own short `lab/vbt_compat.py` (explicit `year_freq`, `ddof`, `ewm` flags per F 2a) plus a CLAUDE.md note listing the traps; a stale third-party skill pinned to 0.28.4 would teach exactly the defaults ggTrader needs to avoid.

## 10. Ranked shortlist (max 8)

| Rank | Tool | Verdict | ggTrader problem it solves | Integration point | Risk |
|---|---|---|---|---|---|
| 1 | **fja05680/sp500 refresh** (vendored data) | ADOPT | PIT file wrong on 2023-10-18 and ends 2026-06-02 | replace `data/universe/sp500_constituents_history.csv.gz`; rerun oracle | Ticker-only IDs; any membership change shifts historical numbers, so re-baseline |
| 2 | **quantstats** | ADOPT (reports only) | Human tear sheets vs SPY without hand-written code | `qs.reports.html(pf.returns(), benchmark=spy, periods_per_year=252)` | Benchmark-alignment issue #584, FutureWarnings; do not feed gates |
| 3 | **TA-Lib** (`ta-lib` wheel) | ADOPT as dev/test dependency | Reference oracle for hand-written RSI/ATR/MACD; `from_talib` for new indicators | tests + `vbt.IndicatorFactory.from_talib` | Live signal change needs F's 0-flip gate |
| 4 | **jsharpe** + **purgedcv** (PBO/DSR) | EVALUATE | Independent check of `gates.py` DSR; add PBO for the 48-combo grid | per-period returns arrays from the sweep | Young projects; vendor if needed |
| 5 | **arch** (`SPA`, `StationaryBootstrap`) | EVALUATE | Multiple-testing evidence that the best combo beats SPY; Sharpe confidence intervals | arrays from `pf.returns()` | Release cadence slow; NCSA licence is OSI-approved |
| 6 | **skfolio** `CombinatorialPurgedCV` / `WalkForward(purged_size)` | EVALUATE | The purge/embargo/CPCV gap that vectorbt OSS lacks | positions slice the frames before `from_signals` | Heavy deps; fold boundaries differ from `generate_folds` (16 vs 17) |
| 7 | **DuckDB + pyarrow** | EVALUATE | Query raw per-trial JSON/Parquet artefacts without new tables | write Parquet next to each report | Timescale attach UNVERIFIED |
| 8 | **pandas-ta-classic** | EVALUATE | Pure-Python indicator catalogue that vectorbt 1.2.0 wraps (`from_pandas_ta`), matches TA-Lib RSI to 5e-14 | `IndicatorFactory.from_pandas_ta` | Community fork, one maintainer |

## 11. Deliberately do NOT add

- `mlfinlab` (proprietary EULA), `mlfinpy` (pandas<3), original `pandas-ta` (pins numba 0.61.2, beta), `pypbo` (AGPL, not on PyPI), `backtesting.py` (AGPL), `backtrader` (GPL, dormant since 2023), `zipline-reloaded` (latest needs pandas<3), `finta`/`tulipy`/`ta` (stale), `deflated-sharpe` by mnemox (wrong output on my test).
- `empyrical-reloaded` and `pyfolio-reloaded` and `ffn` as runtime dependencies: they duplicate quantstats and each add their own Sharpe/CAGR conventions. Riskfolio-Lib, PyPortfolioOpt and direct cvxpy: no portfolio-optimisation need exists, and skfolio would cover one if it did.
- `mlflow`, `DVC`, `wandb`, `aim`: duplicate `lab_runs` and JSON artefacts; mlflow alone drags a very large tree.
- `OpenBB`, `nautilus_trader`, `vnpy`: platforms, not complements.
- `marketcalls/vectorbt-backtesting-skills`: no licence, 0.28.4/pandas 2 pins, return-maximising WFO, unflagged 365-day default.
- `optuna`: do not add until a grid is too large to enumerate; it increases snooping.

## 12. Limits of this report

- Compatibility was tested with synthetic data and a handful of calls per tool, on one Python (3.12.x) and one pandas/numpy pair; resolution and import were checked for everything else. `dvc`, `edgartools`, `fredapi`, `alpaca-py`, `nautilus_trader`, `zipline-reloaded`, `riskfolio` beyond one optimisation, and `openbb` were only resolved (and in some cases imported), not exercised.
- Star counts and dates are as returned by the GitHub API on 2026-11-06 and will drift. "Last commit" is the newest commit on the default branch.
- I did not read the full text of every README claim; where a claim comes only from a description or README, the table says so.
- The vectorbt master commit about zero-volatility Sharpe is known by title only.
- Nothing here was run against the real SP500 tape or ggTrader code; the PIT comparison read the committed file only.
