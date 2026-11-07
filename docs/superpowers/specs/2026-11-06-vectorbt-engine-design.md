# vectorbt 1.2.0 Engine Rebuild: Design

**Date:** 2026-11-06 (revised the same day, Appendix E) · **Status:** draft for owner review; not
approved · **Type:** architecture, migration and validation design · **Author:** lead architect
(Opus 5.5), from eight research reports (S1-S8) and an earlier verified synthesis (F), after
independent checks (Appendix A).

**Revision note (2026-11-06).** The owner stopped the paper trader, closed its stock positions and
said to drop the live-trader protections and switch completely to vectorbt. The production-freeze
premise, the shadow job and the staged live cutover are removed. Section 8 is now a short trader
rebuild plan. Appendix E lists every section changed.

Labels used throughout: **MEASURED** (someone ran it; Appendix A says whether I re-ran it),
**ESTIMATE** (judgement or extrapolation), **UNVERIFIED** (claimed, not checked). File references
are to `main` at `6774b4d3`.

## 1. Intent

### 1.1 Why the owner wants this

The owner has decided that ggTrader should stop being a hand-built backtesting engine with a
strategy collection attached, and become a place to **define strategies and test them with
vectorbt**. In the owner's terms:

1. vectorbt 1.2.0, with its Rust engine, is **the** engine.
2. Indicators, signals and statistics come from vectorbt, so ggTrader stops maintaining its own
   copies of RSI, MACD, Bollinger, Sharpe and the rest.
3. Old research numbers may be re-baselined. A large architectural effort is accepted.
4. ggTrader code is the thinnest glue that vectorbt cannot provide.
5. New vectorbt releases must be easy to take. AI agents will do the upgrades.
6. Useful open-source tools that complement vectorbt were to be surveyed (S8; section 4.2 OD10).

### 1.2 What success looks like

- One package, `ggTrader.engine`, is the only code that imports vectorbt. An upgrade is a one-line
  pin change, an automated report, and a decision, done by an agent in under a day.
- A strategy is one file: a parameter grid plus a `build()` that composes vectorbt indicators and
  signal accessors. The engine runs the whole grid in one grouped simulation.
- The walk-forward run that takes ~254 s for 2 folds today (F P2, ESTIMATE unprofiled) takes well
  under a minute, mostly from removing duplicated work, not from Rust.
- Every result carries a provenance block. Old-engine and new-engine numbers are never mixed.
- The paper trader, stopped by the owner on 2026-11-06, is rebuilt on the validated engine, runs
  first in dry-run mode, and returns to cron only when the owner says so (section 8).
- Honest NO-GO verdicts stay NO-GO. A re-baseline never reopens one by accident.

## 2. Goals, non-goals, anti-goals

**Goals**
- Move the lab onto vectorbt 1.2.0 + vectorbt-rust 1.2.0, pandas 3, numpy 2.5, Python 3.12.
- Replace the hand-written indicator, statistic and simulation plumbing with vectorbt calls behind
  one adapter package with explicit settings.
- Rewrite the walk-forward (WFO) engine so each simulation runs once (F D1/D3).
- Make upgrades routine: lock file, contract/golden/canary tests, an upgrade watch job, an agent
  runbook and provenance on every result.
- Re-baseline the active research numbers under a recorded epoch (`E1`).
- Rebuild the paper trader's signal path on the new engine after the engine and the strategy port
  are validated (section 7), then dry-run it (section 8).

**Non-goals**
- New strategies or new research questions. This program changes the engine, not the edge.
- vectorbt PRO features (purged CV, time stops, chunking, agent tooling). No paid software.
- Replacing ggTrader's policy code: point-in-time (PIT) universe, gates, anchor, circuit breaker,
  selection, persistence, tax lots, broker, risk. vectorbt has no equivalent (F section b).
- Re-porting closed NO-GO strategies by default (OD11).

**Anti-goals (failure even if it technically works)**
- Trader signals that change silently: any difference between the rebuilt trader and the validated
  lab strategy on the same frame, or between legacy and new signals, that is not pre-registered,
  measured against the live golden (7.3 item 4) and accepted by the owner.
- The trader back on cron before the engine and strategy port are validated, the dry run is done
  and the owner has said so.
- Old and new numbers compared as if they were one series, or a NO-GO verdict re-read in the light
  of a re-baseline that happens to look better.
- Bundled definition changes, so nobody can say which one moved a number.
- Dependency bloat: adding frameworks (mlflow, OpenBB, zipline) or keeping dead ones (`pandas-ta`).
- Maintaining a fork or vendored copy of vectorbt, or private `vectorbt._engine` access outside
  one file.
- A wrapper layer that re-implements vectorbt maths under ggTrader names.
- Any recommendation to retire the stock-picking sleeve, end paper trading or shut the project
  down, or any IRA or retirement-account advice (owner instructions 2026-10-20).

## 3. Constraints

| Constraint | Consequence for this design |
|---|---|
| **Licence.** vectorbt and vectorbt-rust are Apache 2.0 **with Commons Clause** (wheel `LICENSE.md`; rust METADATA `Apache-2.0 WITH Commons-Clause-1.0`). Not an OSI licence. It forbids selling a product or service whose value derives substantially from vectorbt. | Private research and a personal paper trader are unaffected. The owner records an explicit exception (OD4). Never offer ggTrader as a paid service without re-reading the licence. |
| **Free data only**, no paid data or software (owner, 2026-10-23). | No PRO, no SIP feed. Alpaca free IEX is unusable for research history or volume (S6 section 3). Yahoo + Tiingo + TimescaleDB stay. |
| **Paper trader stopped** (owner, 2026-11-06). The 12:45 PT `scripts/paper_trade.sh` and 06:00 `daily_pnl_report.sh` cron lines are commented out; stock positions were closed except the stuck AVB holding; ~114 SPY shares remain. Nothing else runs or auto-updates the old trader. | No live run to protect: work merges to `main`, and files shared with the old trader (`src/ggTrader/paper/`, its lab imports, `pyproject.toml`, `Dockerfile`, `docker-compose.yaml`) may change or be deleted. Cron stays disabled until the owner re-enables it (section 8). No phase places orders. `.env` is still not read by agents. |
| **Taxable account only.** | Tax-lot and after-tax tooling (`TaxLotBook`, A23) stays and is out of scope. |
| **Memory.** Tests via `scripts/run_tests.sh` (8 GB cgroup, no swap). Research runs under `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`. | The engine chunks grid columns to keep an estimated peak under 5 GB (section 5.5). A 137 exit is a finding, not a reason to raise the cap. |
| **Stack.** vectorbt 1.2.0 requires numpy>=2.4.6, pandas>=3.0.3,<4, numba>=0.66, Python 3.11-3.14 (wheel METADATA, checked). | The old and new stacks cannot share one image. The old image is kept by immutable tag for reproducibility only; the rebuilt trader ships on the new image. |
| **Python.** Image is `python:3.12-slim` (3.12.14). | Pin `requires-python = "==3.12.*"`. A Python bump is its own yearly upgrade event. |
| **No PRO.** | Time stops, calendar splitters, purge/embargo and chunking stay ggTrader glue or come from free libraries. |

## 4. Decisions

### 4.1 Owner decisions already made (final; not reopened here)

1. Adopt the latest open-source vectorbt, 1.2.0, with the vectorbt-rust engine as the engine.
2. Use vectorbt's own indicator, signal and statistic definitions instead of hand-written ones.
3. Accept re-baselining of research numbers and a large architectural effort.
4. ggTrader's purpose is defining, testing and backtesting strategies with vectorbt; its own code
   is the thinnest glue needed.
5. Make frequent vectorbt updates easy; AI agents will perform them.
6. Survey complementary GitHub tools (done in S8).

Earlier advice to stay on 0.28.5 (F section d, B section 7) is superseded by decision 1. Its
measurements are kept and used to design around the traps it found.

**Standing rules that bind this spec:** research, strategy evidence and paper-trading
reliability first; no retirement or IRA recommendations; no paid data; the paper trader is
stopped and stays off cron until the owner re-enables it; honest NO-GOs are preserved. The
ensemble signal path and the trader rebuild are **in scope**. `docs/changelog.md:267-268` still says the next steps "retain ... the owner
decision to retire stock picks around 2026-12-29". That line is stale and is superseded by the
2026-10-20 owner instructions; it is a **documentation defect to fix**, not a decision. (At
`6774b4d3`, `docs/next_steps.md` no longer contains it; only the changelog does.)

### 4.2 Decisions the owner still must make

Each item: options, my recommended default, and what it changes. Defaults are what the builder
does if the owner says "use the defaults".

| # | Decision | Options | Recommended default | What it changes |
|---|---|---|---|---|
| OD1 | **Indicator definitions** (all via vbt code; the question is which form). | (a) configure vbt kernels to the legacy conventions; (b) vbt defaults. | **(a) for epoch E1:** RSI = `vbt.RSI(window=2n-1, ewm=True)` (exact Wilder values, first valid bar 27 not 14); MACD = `macd_ewm=signal_ewm=True` (Appel EMA MACD); BBANDS/MSTD `ddof=1`; ATR `ewm=False`; EMA = `vbt.MA(ewm=True)`, adopting vbt's NaN warm-up; crossovers = vbt `crossed_above/below` (0 cell differences, S5). | (a): 0 entry and 215 exit cells differ on 150 symbols x 1,968 bars, all in burn-in except one float tie; live last bar 0 of 120 sessions (S5; re-run in Appendix A). (b) is a **different strategy**: 59% of entry cells differ, live differs on 95 of 120 days (S5). (b) needs the full forward-paper standard. |
| OD2 | **Statistic definitions.** | Per item: Sharpe year length; Sortino; CAGR; DSR; return ddof. | Sharpe/vol/max DD: vbt with `year_freq="252 days"`, ddof=1 (equal to 1e-15). **Never 365** (+20% Sharpe, S5). Sortino: adopt vbt (textbook downside deviation; 0.487 vs 0.431) in P6 as its own run. CAGR: adopt vbt `annualized()` at 252 bars for strategy **and** SPY, labelled "bar-annualised", in P6 as its own run. **Until then E1 scores Sortino and CAGR with the legacy forms** (`stats.window_stats(..., definitions="legacy")`, the default for epoch E1, via the transitional `engine/legacy_defs.py`; Appendix D Q6), so E1 is configured-to-legacy and no definition change is bundled into it. DSR: configure vbt to reproduce the legacy statistic: `deflated_sharpe_ratio(nb_trials=len(grid), var_sharpe=252/(T-1), bias=False)` (see Appendix A: **not** S5's `1/(T-1)`, and **not** `bias=True`). Flat-curve guard: not optional, see 5.6. | Sortino enters the composite score at weight 0.3 and can flip fold winners (F b). CAGR moves the anchor's CAGR>4% filter. Each is one P6 run with its own JSON and owner sign-off. |
| OD3 | **Volume-BB double vote.** `vbb` entry = `bb` entry AND a volume spike (`indicators.py:67-79, 237-263`), and the two exit formulas are identical. One spike is 2 entry votes, which alone meets `min_agree=2`; one SMA cross is 2 exit votes, which alone meets `min_agree_exit=2`. Verified in code and numerically (0 of 220 vbb entries outside bb; exit frames identical). | (a) keep for parity; (b) make `vbb` a volume confirmation of `bb` (one vote, or a +1 only when bb fires), a new strategy. | **(a) in E1** so the port is comparable. Register (b) as a separate research question under the forward-paper standard if the owner wants it. | (b) changes live signals and every ensemble number. |
| OD4 | **Commons Clause licence exception.** | Record it, or do not adopt vbt 1.x. | **Record it** in `docs/architecture.md` and the changelog: private research and personal paper trading only; no resale or paid service. | None technically; it is a governance record. |
| OD5 | **Refresh the stale PIT constituent file** `data/universe/sp500_constituents_history.csv.gz` from upstream fja05680/sp500 (MIT). Local file ends 2026-06-02; upstream ends 2026-08-18 with 9 extra dates; 2,711 shared dates identical except 2023-10-18, where the local file applies the Dec-2023 rebalance two months early (BLDR, JBL, UBER in; ALK, SEDG, SEE out). Re-checked against the live upstream CSV. | Refresh now; refresh after E1; never. | **Refresh after E1 parity, as its own P6 run**, then a scheduled quarterly refresh with an age check in the mask builder. `data/` is never cleaned up automatically; this is a deliberate data edit with a recorded diff. | Shifts membership on a few dates. Re-baseline per run. The trader reads the same file, so the rebuilt trader (section 8) picks the refresh up. |
| OD6 | **Exit on index removal.** Today the mask gates entries only; positions in a removed symbol run to their normal exit (`sweep.py:138-141`). | (a) keep; (b) exit at the first non-member bar (S6 variant C). | **(a) in E1.** Offer (b) as a flag and a P6 run. The weight path already exits on removal by construction (`weights.where(mask, 0)`). | (b) changes numbers by policy. |
| OD7 | **Forced terminal exits for dead tickers.** vbt holds a position at its last price forever when the close turns NaN for good (8 stranded positions in S6 test B). | (a) keep; (b) exit at each series' last valid bar and suppress a same-bar entry (S6 variant E). | **(b) as a P6 run**, labelled optimistic (last traded price overstates recovery for failures), with the dead-ticker list in every run JSON. | Small number changes; removes a silent accounting artefact. |
| OD8 | **WITHDRAWN 2026-11-06** (was: authorization A1 for a scheduled shadow job). | n/a | n/a | The trader is stopped, so there is no live run to shadow; the dry run in section 8 replaces it. |
| OD9 | **WITHDRAWN 2026-11-06** (was: A2, pin compose to a digest before merging to `main`; A3, the readiness decision gating cutover). | n/a | n/a | With cron disabled a push to `main` cannot reach a running trader, so work merges to `main`; re-enabling is OD18. Immutable image tags remain a rebuild safeguard (section 8.3), not a prerequisite. |
| OD10 | **S8 shortlist.** | Adopt / evaluate / skip per tool. | **Adopt:** fja05680 refresh (= OD5, vendored file); `quantstats` for human tear sheets only, never for gates, always `periods_per_year=252`; `TA-Lib` wheel as a **test-only** oracle. **Evaluate later, not in this program:** `jsharpe`/`purgedcv` (DSR/PBO cross-check), `arch` (SPA test), `skfolio` (purged CV), DuckDB+pyarrow (query JSON/Parquet artefacts). **Skip:** everything in S8 section 11. | Two new dependencies (quantstats in a `reports` extra; ta-lib in `test`). |
| OD11 | **Port scope for closed NO-GO strategies.** 38 registry entries; 29-30 are closed NO-GO (S4, Appendix A). | (a) port all; (b) port the active set, keep the rest reproducible at the legacy tag. | **(b).** Port: `ensemble` and its five voter strategies (`bb_reversion`, `rsi_reversion`, `ema_cross`, `macd_divergence`, `volume_bb_reversion`), `conviction_bb`, `cross_asset_sleeve` (static shadow GO), `retail_attention` (paused, no verdict) and one weight strategy (`xs_momentum`) as the weight-path reference. Others stay runnable at tag `legacy-engine-vbt0285`; a NO-GO re-test is ported only when registered. | Removes ~3,000 lines from the active tree (ESTIMATE). No verdict changes. |
| OD12 | **Trade count and expectancy per scoring window.** Today `pf.trades.count()` counts the whole simulation (F D3: 72 of 80 combos differ when sliced). | Redefine per window; keep. | **Redefine** (count trades whose entry index is inside the window). NDH gate uses only the sign, so decisions should not change (analytic; checked on the oracle in P5). | Stored expectancy values change. |
| OD13 | **`--max-leverage` default** is 2.0 (`lab/cli.py:97`, `blend.py:30,74`); production runs 1.0. | Keep 2.0; change to 1.0. | **1.0** (this trap was hit twice per CLAUDE.md). | Only blend runs that omit the flag. |
| OD14 | **Which ensemble configuration is the reference.** The grid pins `divergence_window=10, vol_mult=1.5`; live runs the defaults 20 and 2.0, which are not in the grid (`ensemble.py:120-140` vs `:67-91`; already recorded in the production replica audit, lines 33-35). | Reference = grid winners; = live defaults; both. | **Both, labelled.** The live golden, the rebuilt trader and its dry run use live defaults. WFO references use the unchanged grid. The re-baseline report adds the live defaults as a pinned control row. No grid change in this program. | Clarifies what each number describes; no strategy change. |
| OD15 | **Dead dependencies.** `pandas-ta` (0 imports; blocks numba>=0.66), `ta`, `hmmlearn`, `optuna`, `seaborn`, `mplfinance`, `xlsxwriter`, `tabulate`, `fsspec` have zero static imports (S3). `scipy` and `beautifulsoup4` are imported but undeclared. | Remove/declare. | **Remove `pandas-ta` (required). Remove the others after one owner glance. Declare scipy and bs4.** Keep `pyarrow` (pandas 3 string dtype). | Smaller lock; no behaviour change. |
| OD16 | **Upgrade cadence.** | Every release; selected releases. | **Evaluate every release after a 3-day cooldown; adopt unless the report shows an unexplained change.** In a burst, evaluate only the newest. Promotion to the trader image only after research adoption, live golden identical and a 7-day soak. | Sets the agent's standing workload (~6-9 releases a year, S2). |
| OD17 | **Stuck AVB holding** (new 2026-11-06). AVB could not be sold: the ticker is inactive after the EQR merger and the paper account still shows 4.05 shares. | (a) leave it and have the rebuilt trader treat it as untradable, as the old trader did; (b) the owner contacts the broker to have it converted or removed. | **Owner question; no default.** Whichever is chosen, the rebuilt trader must not try to trade it, and the dry run must show it handled. | Affects position counts, the cap and portfolio value in the dry run and after re-enabling (risk 15). |
| OD18 | **Cron re-enable policy** (new 2026-11-06). | Re-enable only on the owner's explicit say-so; any date-based or automatic re-enable. | **Owner's say-so only.** No date, phase exit or green check re-enables cron by itself; an agent never edits the crontab. The owner un-comments the lines (or explicitly tells an agent to). | Ends the dry run (section 8.4). |

## 5. Target architecture

### 5.1 Layers and package layout (decided)

```
ggt.py -> cli
  lab.cli (ggt lab, --blend)                 paper.trader (ggt paper)   <- rebuilt in P7 (section 8)
     |                                            |
  lab.wfo  (folds, select, gates, breaker, anchor, run, report)     paper.* (broker, risk, sweep, ...)
     |                                            |
     +---------------- lab.strategies (one file per strategy, registry, vote) ----+
                         |                    |
                  lab.universe (PIT mask)   ggTrader.engine  (ONLY importer of vectorbt)
                         |                    |
                  data.bars (read-only)     vectorbt 1.2.0 + vectorbt-rust 1.2.0
```

| Path | Status | Responsibility | Size (ESTIMATE) |
|---|---|---|---|
| `engine/__init__.py` | new | Public API; `configure()` runs on import | 30 |
| `engine/config.py` | new | `EngineSettings`; engine from `GGT_VBT_ENGINE` in {`rust`,`numba`}, default `rust`, **never `auto`**; sets `vbt.settings["engine"]` and `vbt.settings["returns"]["year_freq"]="252 days"`; turns the version-mismatch `UserWarning` into an error | 60 |
| `engine/selfcheck.py` | new | Dispatch probe: wraps `vectorbt_rust.portfolio.simulate_from_signals_rs`, runs a 2-column trailing-stop `from_signals`, raises unless the call count matches the pinned engine; checks vectorbt and vectorbt-rust share major.minor (prototype works, S1) | 60 |
| `engine/indicators.py` | new | **Presets**, one line each, every behaviour flag explicit: `wilder_rsi(close, n)` -> `vbt.RSI.run(window=2n-1, ewm=True)`; `bbands(close, window, alpha)` -> `ddof=1`; `ema`, `sma`, `macd` (both ewm flags), `atr` (`ewm=False`). Returns vbt indicator objects. No maths. | 80 |
| `engine/signals.py` | new | `cross_above/below` wrappers that coerce thresholds to float64 (an int threshold raises on Rust: re-checked), `time_stop(entries, n)` via `fshift` | 40 |
| `engine/portfolio.py` | new | `run_signals(out, close, mask)`, `run_weights(weights, close)`; one grouped call; column chunking under a 5 GB estimate; returns a frozen `SimResult` (value, returns, trade records) | 180 |
| `engine/stats.py` | new | Window stats from returns via `.vbt.returns(freq="1D", year_freq="252 days")`; finite guard; per-window trade counts; configured DSR | 120 |
| `engine/windows.py` | new | Calendar folds -> bar positions (`searchsorted`); `range_split` wrapper for the weight path | 60 |
| `engine/provenance.py` | new | `provenance()`, `compare(a, b)` (S2 section 6) | 90 |
| `engine/_compat.py` | new | Version shims, one per upstream change, each with a test that fails when it becomes unnecessary | 0 at start |
| `lab/universe.py` | new | `pit_mask`, `membership_mask`, `terminal_exits`, `removal_exits`; pure, vectorised (S6 `pit.py`, 51 lines) | 70 |
| `lab/strategies/vote.py` | new | N-of-M vote over precomputed voter arrays, independent exit, grid stacking (vbt has no vote primitive, F b) | 60 |
| `lab/strategies/<name>.py` | rewritten | One file per ported strategy | ~650 for the indicator family (S1) |
| `lab/wfo/` | rewritten | `folds` (keep `generate_folds`, 17 lines), `select` (composite, live-param pick), `gates` (NDH; DSR via engine), `breaker`, `anchor`, `run`, `report` | ~800 |
| `data/bars.py`, `data/snapshot.py` | new | `load_bars()`: read-only DB or hashed Parquet snapshot; no network, no DDL; asserts unique index, midnight stamps, weekdays; returns per-field frames and `missing_symbols` | 100 |
| `lab/persist.py` | modified | `provenance JSONB` column on `lab_runs` and `lab_sweeps` | +20 |
| `utils/results.py` | new | `write_result_json()` that stamps provenance | 30 |

**Deleted after E1 is accepted (end of P5)**; no live copies are kept, because no live run
depends on them (section 8.1): `lab/simulate.py` (304),
`lab/sweep.py` (350), `lab/metrics.py` (56), `lab/harness.py` (148; non-WFO runs become a one-fold
WFO), the WFO sweep plumbing in `lab/wfo.py`, the per-combo loops in `signals.py`/`ensemble*.py`/
`conviction.py`, `kelly.py` trade pairing (becomes `pf.trades.records`), `pit_entry_mask` and
`eligible_at` (O(T^2)), `collapse_daily_duplicates` (becomes an assert), research branches of
`fetch_stock_ohlcv`, `lab/negative_cache.py`. The old pandas indicators in
`strategies/indicators.py` are deleted at the same point, once the new presets are validated; the
old trader imports them, so `ggt paper` stays broken from then until the rebuild (P7). That is
acceptable because cron is disabled; tag `legacy-engine-vbt0285` and the legacy venv lock keep
closed research reproducible.

**Kept unchanged:** `lab/data.py` live parts, `allocation.py`, `blend.py`, `gates.py` NDH logic,
TimescaleDB store and loaders, tax lots, A23, replica, century study, screens, event tooling,
`paper/` apart from the signal-path rebuild in section 8. About 70% of `lab/` is code vbt cannot replace (F c), so the honest size result is:
engine path ~4.6k -> ~2.5-2.9k lines; `lab/` 18,591 -> ~15,300 (net about -10%, S1 section 7,
ESTIMATE). This is a maintenance and speed win, not a "ggTrader halves" win.

**Not adopted:** `vbt.Data` (OSS is a dict-of-frames holder; S6 section 2), `YFData`/`AlpacaData`
(tz-ambiguous, serial, crash on delisted symbols, IEX history too short; S6 section 3), vbt
splitters for the signal path, `vectorbt.labels` (forward-looking by design; banned by test).

### 5.2 Import rules (enforced)

1. `vectorbt` and `vectorbt_rust` are imported only under `src/ggTrader/engine/`. Enforced by ruff
   `TID251` banned-api with a per-file ignore for `engine/**` (prototyped on ruff 0.15.12, S2) and an
   AST test over `src/`, `scripts/`, `tests/`, `ggt.py`.
2. Raw indicator constructors (`vbt.RSI.run`, `BBANDS.run`, `MACD.run`, `ATR.run`, `MA.run`,
   `MSTD.run`) and `Portfolio.*` appear only in `engine/`. Strategies call `engine.indicators`
   presets and `engine.portfolio` never directly.
3. The `.vbt` pandas accessor (crossings, `fshift`, `tile`, rolling helpers) is allowed in `engine/`
   and `lab/strategies/` only. AST test bans the attribute elsewhere (once vectorbt is imported,
   `.vbt` is reachable from any module without an import, S2 0.4).
4. `engine` imports nothing from ggTrader. `lab.strategies` may import `engine`, `lab.universe`,
   `lab.strategies.vote`. `lab.wfo` may import `strategies`, `engine`, `universe`, `persist`.
5. `paper` may import `lab.strategies`, `engine`, `lab.universe`, `lab.allocation`. Nothing in `lab`
   imports `paper`, except the allow-listed `replica.py:14-27` and `train_gate.py:38`.
6. `vectorbt._engine` (private) is touched only in `engine/selfcheck.py`.

### 5.3 Strategy contract

```python
@dataclass(frozen=True)
class SignalOut:
    entries: pd.DataFrame          # bool; columns = MultiIndex(grid levels..., "symbol")
    exits: pd.DataFrame            # bool; same columns
    size: float | pd.DataFrame = 0.03          # size_type="percent"
    sl_stop: float | pd.DataFrame | None = None
    sl_trail: bool = False
    tp_stop: float | None = None
    td_stop: int | None = None     # time stop is ggTrader glue (not in OSS from_signals)

class SignalStrategy(Protocol):
    name: ClassVar[str]
    kind: ClassVar[Literal["signals"]]
    grid: ClassVar[ParamGrid]      # values + validity predicate (ports sweep.py:38-74)
    live: ClassVar[dict | None]    # deployed params; may lie outside the grid (OD14)
    def build(self, bars: Bars, grid: ParamGrid) -> SignalOut: ...

class WeightStrategy(Protocol):
    name: ClassVar[str]
    kind: ClassVar[Literal["weights"]]
    grid: ClassVar[ParamGrid]
    def build(self, bars: Bars, grid: ParamGrid, rebal: pd.DatetimeIndex) -> pd.DataFrame: ...
    # target weights, NaN = hold; engine uses from_orders(size_type="targetpercent")
```

Rules: strategies never call `Portfolio`; every threshold is a float; every indicator call goes
through a preset; the PIT mask is applied by the engine, **before** the time stop (this also
fixes the sweep path, which today masks after the time stop, `sweep.py:155-166`, F C).
Registration is a decorator filling the existing `STRATEGY_REGISTRY` dict, so the CLI choices
still derive from it (`strategies/registry.py`). Event and alt-data strategies keep a small
`select()` that builds their weight frame.

Ensemble sketch (epoch E1, OD1(a), OD3(a)):

```python
@register
class Ensemble:
    name, kind = "ensemble", "signals"
    grid = ParamGrid(min_agree=[2, 3, 4], min_agree_exit=[1, 2], bb_std=[2.0, 2.5],
                     rsi_oversold=[25.0, 30.0], ema_fast=[10, 20],
                     fixed=dict(bb_period=20, rsi_period=14, rsi_exit=50.0, ema_slow=50,
                                macd=(12, 26, 9), divergence_window=10, vol_period=20,
                                vol_mult=1.5))
    live = dict(min_agree=2, min_agree_exit=2, bb_std=2.0, rsi_oversold=30.0, ema_fast=20,
                divergence_window=20, vol_mult=2.0)          # ensemble.py:67-91 defaults

    def build(self, b, g):
        bb = ind.bbands(b.close, window=20, alpha=g.values("bb_std"))     # once per alpha
        rsi = ind.wilder_rsi(b.close, 14)
        ema_f, ema_s = ind.ema(b.close, g.values("ema_fast")), ind.ema(b.close, 50)
        hist = ind.macd(b.close, 12, 26, 9).hist
        spike = b.volume > g.fixed("vol_mult") * ind.sma(b.volume, 20).ma
        bbv = per_value(bb, lambda x: (sig.cross_below(b.close, x.lower),
                                       sig.cross_above(b.close, x.middle)))
        voters = {
            "bb":   bbv,
            "rsi":  {o: (sig.cross_below(rsi.rsi, o), sig.cross_above(rsi.rsi, 50.0))
                     for o in g.values("rsi_oversold")},
            "ema":  per_value(ema_f, lambda f: (sig.cross_above(f, ema_s), sig.cross_below(f, ema_s))),
            "macd": divergence(b.close, hist, window=g.fixed("divergence_window"), warmup=55),
            "vbb":  {k: (e & spike, x) for k, (e, x) in bbv.items()},  # parity (OD3): bb AND spike
        }
        return vote(voters, g, independent_exit="rsi", size=0.03)
```

`vote()` sums precomputed int8 arrays per combo and stacks them into one wide frame, so each
indicator runs once per unique parameter (the memoisation F Phase 1c asked for, by construction).
The MACD divergence rule and its warm-up mask are ggTrader rules (~5 lines) and stay glue.

### 5.4 Walk-forward design (decided)

1. **Folds:** keep `generate_folds` (12-month train, 3-month test, 3-month step, `wfo.py:194-210`).
   `engine.windows.to_positions` maps dates with `searchsorted` (handles a Sunday 2021-01-31).
   skfolio and vbt splitters do not reproduce these folds (16 vs 17, S8 section 2).
2. **Signal strategies: one simulation per (strategy, universe, stop group) over the full
   history**, sliced per window (anchor, train, test, live). F verified on the real `_sweep_fold`,
   120 SP500 symbols, 80 combos: equity and Sharpe max abs diff 0.0. This matches today's
   semantics, because today's signal path already simulates from the first loaded bar and carries
   positions into every window.
3. **Weight strategies: stacked windows, fresh start per (combo, fold)** via `range_split`, one
   `from_orders` call. Today's weight path starts flat per window, so slicing a full-history
   simulation would be wrong (F D3). Anchor and train share a simulation inside a fold; folds never do.
4. **Trade counts per scoring window** from `pf.trades.records` entry index (OD12).
5. **Selection reads engine stats** with the finite guard (5.6); composite, anchor, NDH, breaker
   and WFE logic are unchanged and fed the same quantities.
6. **DSR** via `engine.stats.dsr(returns, n_trials=len(grid))`, configured to reproduce
   `gates.dsr_check` (OD2): `var_sharpe=252/(T-1)`, **`bias=False`** (the legacy inputs are pandas'
   unbiased `skew()`/`kurtosis()`). This item first said `bias=True`; that was wrong (Appendix A,
   P0-P3 review): on 12 seeded series `bias=False` matches `gates.dsr_check` to 4.4e-16 and
   `bias=True` is off by up to 1.26e-4, enough to flip a gate within 1e-4 of 0.80. The code was
   right from P2.
7. **The anchor reuses the train simulation** (they are byte-identical work today, F D1).
8. **Window convention (legacy, decided 2026-11-06 after the P0-P3 review).** A window of bars
   `a..b-1` is scored exactly as `lab.metrics.curve_stats(equity.iloc[a:b])`: the equity is
   rescaled to its first bar and the returns start at the *second* bar (`returns.iloc[a+1:b]`).
   `engine.stats.window_stats(returns, a, b)` implements this and `n_obs = b - a - 1`; the
   return into bar `a` belongs to the previous window. Tested against `curve_stats` on random
   and real-tape equity (`tests/engine/test_legacy_defs.py`). Legacy folds overlap by one bar
   (`train_end == test_start`, both `closed="both"`); `fold_positions` reproduces this and it is a
   P6 definition question, not an E1 change (review MINOR-12).

### 5.5 Engine selection, startup assertion, memory

- `GGT_VBT_ENGINE` in {`rust`,`numba`}, default `rust`, unknown value raises. `auto` is rejected:
  it silently falls back to numba when the wheel is missing or minor-mismatched (B 4.2-4.3).
- `selfcheck()` runs at `ggt lab` and `ggt paper` start, in the image build (as root and as
  `appuser`), and in CI. Its result goes into provenance. It takes milliseconds with Rust.
- numba is a **research** escape hatch only (bit-identical simulator results, S5 e). It is never
  a trader fallback: a cold numba start measured 15-57 s per kernel on a loaded box (S3, S7). The
  trader has no fallback path: if the Rust check fails it places no orders (section 8.3).
- `engine.portfolio` estimates peak memory as T x columns x 8 B x (number of dense per-column
  arrays + 12 working arrays) and splits whole groups into chunks under the budget (default
  5 GiB). The working-array count was calibrated in the P2b fix-up on the full ensemble grid
  (measured about 10.7 cells per bar-column; first guess 8 was 18% low) and each finished chunk
  is copied out and released, so the budget bounds the peak instead of the sum
  (`p4-evidence/p2b_memory_summary.json`). Prototype: 24,000 columns x 1,700 bars, 48 groups, 3.86 GB peak,
  16 s (S1, synthetic).

### 5.6 Statistics and guards ggTrader must own

- **Non-finite guard:** vbt 1.2.0 returns `+inf` Sharpe for a flat curve and 4.5e16 for constant
  returns (S1 output); Sortino `+inf` with no downside. Set to NaN when return std < 1e-12 or the
  window has zero trades, before any `idxmax`, rank or composite. Test: one flat column in a
  48-column grid must never win. (Upstream master has a 2026-11-03 commit changing this, title only;
  the guard stays regardless.)
- **Selection-window degeneracy guards (P5b, 2026-11-06; the Sortino threshold is a lead call):**
  the zero-trade half of the bullet above is now wired in. A selection window (anchor, train,
  live) blanks Sharpe and Sortino of a combination with no entry inside it, and blanks the Sortino
  of one with fewer than `MIN_LOSING_DAYS` = 20 losing days (`engine.config`, overridable with
  `run_wfo(min_losing_days=...)`, 0 = off, stamped in the result and in `definitions`). A guard
  like the flat-curve guard, not a new definition: vectorbt's Sortino formula is unchanged above
  the threshold. The out-of-sample test window keeps only the flat-curve guard (it measures the
  book that was deployed). NaN is floored by the composite's existing rule. See Appendix E
  (2026-11-06, P5b).
- **pandas-3 rules:** explicit `fill_method` on every `pct_change` (44 sites, `.ffill()` first on
  the 4 gappy price-panel sites); `.copy()` after `pd.concat(axis=1)` frame builds (60-block
  fragmentation made the sweep 3x slower; re-checked: 2.21 s vs 6.71 s); `.stack().dropna()` in
  `event_study.py:48-49`; `"1D"` not `"1d"`.

### 5.7 Data layer and universe mask

- Research reads bars through `load_bars()` only: read-only DB session or a hashed Parquet snapshot
  (1.15 s for 646 symbols, S6). Research never fetches, never writes `ohlcv`, never runs DDL.
  `refresh_bars()` is the one explicit, logged research-side fetch, never called by `ggt lab`.
- The trader's loader chain (`fetch_stock_ohlcv`, `CachedYFinanceLoader`, Tiingo) moves to
  `data/live/` as the trader-only fetch during the rebuild (P7).
- `lab.universe.pit_mask(close, min_history_bars, max_stocks, max_sector_count)` replaces the
  per-day loop: 0 mismatches against `eligible_at` on 457 dates x 4 configs (S6) and on 100 fresh
  dates x 2 configs on the old stack (re-checked). Build 0.05 s vs ~334-370 s.
- Indicator outputs carry a parameter column level; the engine tiles the mask with
  `mask.vbt.tile(n, keys=...)` (S6 2).
- Every run JSON records `missing_symbols` (17 members with no bars), the early-ending list (72),
  yearly member-day coverage, the last-bar coverage, and the constituent file's age. The mask
  builder warns when the constituent file is older than 45 days (ESTIMATE threshold).

### 5.8 Persistence and provenance

- All seven `lab_*` tables stay. Add nullable `provenance JSONB` to `lab_runs` and `lab_sweeps`
  with `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in `init_schema()`. Null means legacy engine.
- Provenance schema 1 (S2 section 6): vectorbt, vectorbt_rust, engine requested and effective
  (from `selfcheck`, not settings), pandas, numpy, numba, llvmlite, scipy, python, `uv.lock`
  sha256, git sha and dirty flag, `contract_version`, `baseline_epoch`, `definitions`
  (rsi, macd, bb_ddof, atr, sortino, cagr, dsr, year_freq), tape sha256, created_utc.
- `compare(a, b)` returns SAME_STACK, ENGINE_DIFF, PATCH_DIFF, VBT_VERSION_DIFF, or
  not-comparable (PANDAS_MAJOR_DIFF, CONTRACT_DIFF, EPOCH_DIFF). `research-report-review` gains a
  "Comparable?" line.

### 5.9 CLI

`ggt lab --strategy` and `--blend` keep their flags (`lab/cli.py:57-98`); add `--engine
{rust,numba}` (debug) and `--snapshot PATH`. `--eval-end` keeps its drifting default, so the
citation rule (pin both ends) stays. `--max-leverage` default per OD13. `ggt paper` keeps its
flags; its internals are rebuilt in P7 (section 8), and `--live` stays the only way to place orders. `ggt ingest` stays a documented non-functional stub; this program does not touch it.

## 6. Upgrade mechanics

1. **Lock.** `uv` with a committed `uv.lock` (resolved 147 packages for Python 3.12 in 1.7 s,
   S2; prototype lock re-checked). `pyproject.toml` holds intent: `vectorbt[rust]==1.2.0` (the one
   line an upgrade edits; the extra pins `vectorbt-rust==1.2.0`), bounded `numpy>=2.4.6,<2.6`,
   `pandas>=3.0.3,<3.2`, `numba>=0.66,<0.69`, `plotly>=7,<8`; everything else compatible ranges,
   exact only in the lock. `uv export` gives a hashed pip file as an escape hatch.
2. **Docker and CI consume the lock.** Dependencies installed from the lock before `COPY . .`
   (today `Dockerfile:21` re-resolves every unpinned package on every source change, and
   `ci.yml:30` does the same, so CI does not test what ships). Base image pinned by digest. Image
   build runs `pip check`, `selfcheck()` and `import ggTrader.paper.trader` before push.
3. **Version policy.** OD16. The pair `vectorbt`/`vectorbt-rust` is atomic. Dependabot ignores the
   engine stack; a dedicated watch job owns it.
4. **Tests (all via `scripts/run_tests.sh`).**
   - L0 static: ruff TID251, boundary AST test, lock-consistency test, `pip check`.
   - L1 contract: each preset and stat against an **independent** reference written in plain
     numpy/pandas in the test (Wilder RSI including first valid bar and zero-loss case; BB ddof=1;
     EMA MACD; ATR first-bar rule; Sharpe x sqrt(252); Sortino; bar-annualised CAGR; DSR against
     the Bailey-Lopez de Prado formula; flat-series NaN), `rtol=1e-9`. TA-Lib as a second oracle
     where it applies (OD10).
   - L1b defaults canary: `inspect.signature` defaults of every vbt API ggTrader calls plus
     `vbt.settings["returns"]`, against checked-in JSON. An upgrade lists each changed default.
   - L2 golden: three seeded synthetic panels (NaN holes, a gap, a PIT mask) through both paths and
     a 2-fold mini WFO; equity `rtol=1e-9`, trade counts and winners exact. Regenerated only by
     `--update-golden`, which writes provenance and requires a changelog line.
   - L3 engine: dispatch probe (fails unless `rust`), numba-vs-Rust parity (simulator exact,
     indicators 1e-10), `-W error::DeprecationWarning -W error::FutureWarning` for vectorbt/pandas.
   - L4 image smoke in `docker-build.yml` before push.
   - L5 oracle (Encom only, needs the tape): pinned 17-fold WFO and the live golden (section 7.3).
5. **Watch job** `.github/workflows/vbt-upgrade-watch.yml` (S2 section 5 draft): daily detect with a
   3-day cooldown, bump and re-lock, run L0-L3 on both engines, open a PR (or an issue if the lock
   fails), never auto-merge.
6. **Agent runbook** `.claude/skills/vbt-upgrade/SKILL.md` (S2 outline): read release notes for every
   skipped version; bump only the pin line (and floors); run suites; triage (ACCEPT / RE-BASELINE /
   REJECT); run the L5 oracle on Encom; record in `docs/vbt_upgrade_log.md` and the changelog; promote
   to the trader image only through `ggtrader-deploy`. Red lines: never widen a cap to pass CI, never
   regenerate goldens without a written reason, never edit outside `engine/` for a pure upgrade,
   never edit the crontab, and, once the owner has re-enabled cron, never deploy in the 12:30-13:15
   PT window.
7. **Provenance** on every result (5.8), so an upgrade that changes a number starts a new
   `baseline_epoch` instead of silently drifting.

## 7. Validation and re-baseline protocol

### 7.1 Change classification

| Class | What | Rule |
|---|---|---|
| (i) bit-for-bit | tape load and snapshot hash; PIT mask; folds; grid; simulator mechanics for identical signals; gates/anchor/breaker for identical inputs; Sharpe/vol/max DD at 252 on the legacy window convention (5.4 item 8; vectorbt's accessors agree with `curve_stats` to a few ulp, about 1e-16, never bit-for-bit, and that ulp noise is the accepted class (ii) floor for these three) | any difference beyond ulp noise = STOP |
| (ii) tolerance, judged by signal flips | SMA, BBANDS ddof=1, MSTD, ATR `ewm=False`, rolling vol, MACD after ~200 bars, RSI 2n-1 | values within 1e-9 x scale after burn-in **and** 0 signal flips after a 200-bar burn-in, except flips a tie audit explains (`|x - threshold| < 1e-9`) |
| (iii) definition changes | Sortino, CAGR basis, DSR form, BB ddof=0, warm-up policy, exit-on-removal, terminal exits, PIT refresh, any vbt default | one change per run against the configured baseline; JSON field `changed:` names exactly one; owner signs each |

### 7.2 Test pyramid

Section 6 item 4, plus: a **signal truncation test** (signals at bar t on `close.loc[:t]` equal the
full-tape signals at t, 50 random t x all voters); a grep test banning `vectorbt.labels`; a
**prefix causality** test for the signal path (sliced full history equals per-fold prefix); a fill
convention test (fill = close of the signal bar, `indicators.py:82-93`); property tests (cash >=
-1e-9 per group, fees reconcile, no buy outside the PIT mask, equity = cash + positions x close,
no NaN in value after bar 0); and a 25-line numpy ledger that reproduces vbt `from_signals` equity
with fees (1.5e-11, S8 section 8) as an oracle for the fee convention.

### 7.3 Reference artefacts

Written to `research/audits/engine-migration/_<name>_<date>.json` with provenance:
1. `sp500_core_17fold`: `--eval-start 2021-01-31 --eval-end 2026-04-30`, leverage 1.0,
   `STOCK_BASE_CONFIG`, SP500 PIT. Legacy values to reproduce first: Sharpe 0.592, CAGR 3.08%,
   MaxDD -5.28%, SPY 0.779, gates 17/17 (`research/audits/rebaseline/_pit_rebaseline_20260923.json`,
   re-checked).
2. A 2-fold smoke variant (2021-01-31..2022-07-31).
3. SPY buy-and-hold for both windows.
4. **Live golden:** the last 120 sessions of 365-day-window `buys`, `sells` **and `last_exit`** on the
   full live universe with live defaults (OD14). S5's prototype panel compared only last-bar buys and
   sells; `last_exit` is computed from the whole window's exits (`signal_runner.py:104-106`) and feeds
   `_missed_exits`, so warm-up differences early in the window can move it. Include symbols with
   60-200 bars, the case where configured kernels can still differ on the last bar.
5. Open-decision studies (production replica, the cross-asset 80/20 shadow book, a separate study
   unrelated to the withdrawn trader shadow job) are re-run only if their path
   imports vbt or the changed indicators; otherwise marked "pandas-3 unverified".

All references read a frozen, hashed Parquet tape under a read-only DB session
(`PGOPTIONS=-c default_transaction_read_only=on`; S5 saw `load_ohlcv` attempt `INSERT`/`CREATE TABLE`
on a "read"). A tape sanity test (price jumps > 10x, long runs of identical closes; PARA is junk from
late 2021) runs before any reference.

### 7.4 Comparison rules

- **Consistent** (configured migration): identical winners 17/17, identical gate verdicts,
  |dSharpe| < 0.01, trade count within 0.5%, verdict versus SPY unchanged.
- **Materially different:** any winner or gate flip, any verdict change versus SPY, or |dSharpe| >=
  0.05. The OOS Sharpe standard error over ~4.25 test-years is about 0.5, so 0.05 is a pipeline
  tripwire, not a significance test.

### 7.5 Legacy treatment

- `git tag legacy-engine-vbt0285 <last pre-migration main sha>`, `requirements-legacy.lock`
  (`.venv` freeze: vectorbt 0.28.5, pandas 2.3.3, numpy 2.2.6, numba 0.61.2), the legacy image
  digest (`ghcr.io/garykuepper/ggtrader@sha256:0f2403f1...`, readiness review line 39) and
  `scripts/legacy_env.sh` to rebuild that venv in a worktree.
- `research/ENGINE_PROVENANCE.json`: one row per existing results JSON, `engine: legacy`.
- Old reports are **not edited** except a one-line banner on each that cites a vbt-derived figure:
  "Computed on the legacy engine (vbt 0.28.5 / pandas 2.3.3, tag legacy-engine-vbt0285); not re-run."
  The banner list is generated mechanically. `RESEARCH_SNAPSHOT.md` gains an `Engine` column via the
  `research-snapshot` skill.
- **NO-GO verdicts stay NO-GO.** A NO-GO reopens only through a registered re-test under the
  forward-paper standard, never because an E1 re-run looks better.

## 8. Trader rebuild

### 8.1 Situation (owner, 2026-11-06)

- The paper trader is **stopped**. The 12:45 PT `scripts/paper_trade.sh` cron line and the 06:00
  `daily_pnl_report.sh` cron line are commented out in the user crontab. Nothing else runs or
  auto-updates the old trader: the nightly updater does not scan `/home/flynn/ggTrader` and there
  is no watchtower.
- The owner closed the stock positions (ADM, ALB, AMZN sold at market). **AVB** could not be closed:
  the ticker is inactive after the EQR merger and the account still shows 4.05 shares (OD17).
  About 114 **SPY** shares remain.
- The intraday SPY observer timers and the Hermes "Intraday diary supervision" job are a separate
  research study that places no orders. They keep running and are not part of the trader.

Consequences for this program:
- With no live run to protect, **work merges to `main`**. There is no frozen lane, no long branch
  and no second image lane.
- Files shared with the old trader (`paper/*`, `lab.strategies.ensemble`, `strategies/indicators.py`,
  `lab.strategy`, `lab.data.fetch_stock_ohlcv`, `lab.persist.get_engine`, the loader chain,
  `pyproject.toml`, `Dockerfile`, `docker-compose.yaml`) **may be changed or deleted**.
- The old pandas indicators are **deleted once the new ones are validated** (end of P5, section
  5.1). Tag `legacy-engine-vbt0285` and the legacy venv lock (section 7.5) still reproduce closed
  research and the old trader's signals.
- The live path today calls no vectorbt function (`trader.py:519` -> `generate_core_signals` ->
  pandas indicators through `EnsembleSignal`; vectorbt is imported only via `overlay.py:12` ->
  `lab.simulate`). The rebuild is therefore new signal code, not an engine swap.

### 8.2 What is built (P7, after section 7 validation)

- `paper/signal_runner.py` calls the same `Ensemble.build()` used in research, with a
  one-combination grid of the live defaults (OD14), and extracts `buys`, `sells` and `last_exit`
  from the last bar. The live golden (7.3 item 4) is its acceptance test: buys, sells and
  `last_exit` over 120 sessions match the validated lab strategy on the same frame, and any
  difference from the legacy signals is pre-registered and explained.
- Order selection keeps the existing rules unless the owner changes them: buys sorted
  alphabetically (`signal_runner.py:100`) and the position cap `MAX_POSITIONS` from the environment
  (`risk.py:8-28`, code default 30; the changelog says `.env` sets 3; `.env` not read). One extra
  early-alphabet buy changes which names fill the slots, so the dry-run check compares intended
  orders after the cap, not only raw signals.
- `overlay.py` stops importing `lab.simulate` (removes the 2026-09-23 crash class).
- The loader chain moves to `data/live/` (section 5.7). Before any trader run that writes
  `ohlcv` (`_cache_to_db`, `refresh_benchmark_tape`), a scratch-Postgres round-trip test under
  pandas 3 proves timestamps keep their unit and stay naive UTC (UNVERIFIED today, S3). This
  protects the shared tape, not a live run, so it stays.
- AVB is handled per OD17; the trader must not try to trade it.

### 8.3 Safeguards kept (cheap engineering practice)

- **Fail closed.** `selfcheck()` runs before the data fetch; an input sanity gate (row count,
  universe size, NaN share on the last bar, duplicate or weekend days, absurd buy/sell counts;
  bounds ESTIMATE, tuned from the dry run) and any exception in signal generation mean **no orders**
  and a notifier message. There is no fallback engine and no legacy fallback path. If signals are
  ready after 12:55 PT, no orders (a DAY order after the close queues to the next open).
- **Image smoke test** in the Dockerfile and `docker-build.yml` before push: `selfcheck()` as root
  and as `appuser` asserts the Rust engine is dispatched (wrapped `simulate_from_signals_rs` call
  count), vectorbt and vectorbt-rust share major.minor, the mismatch warning is an error,
  `pip check`, `import ggTrader.paper.trader`, `ggt.py paper --help`.
- **Immutable image tags for rollback.** The trader runs an immutable sha tag, never trusts
  `:latest`; the previous tag is kept locally until the new one has run cleanly. Rollback is
  selecting the previous tag.
- **Cron stays disabled** until the owner re-enables it (OD18). No agent edits the crontab.

### 8.4 Dry run, then re-enable

1. After P7's tests and the live golden pass, run `ggt paper` **without `--live`** (the existing
   dry-run mode, which logs intended orders) for a short period the owner chooses, by hand or as the
   owner directs.
2. Before the first dry run, confirm what dry-run mode writes. A code read (`trader.py:495-526`,
   `:675-680`, `:1086-1095`) suggests it still initialises the paper schema, refreshes the
   benchmark tape (an `ohlcv` write), saves the peak value, writes a `paper_snapshots` row and sends
   Telegram messages labelled "DRY RUN" (UNVERIFIED end to end). Record which of these happen, so
   dry-run and live behaviour differ only in order submission (risk 14).
3. Each dry-run session: the run completes; the Rust probe passes; intended buys and sells after the
   cap equal the lab strategy's signals on the same frame; wall time and peak memory are logged.
4. The owner reviews the dry-run record and, only on the owner's explicit say-so, re-enables the
   cron lines. The first live run is checked with `daily-trader-check`. No date schedules this.

## 9. Phased plan

Effort is agent-hours, ESTIMATE (+/-40%); wall time for oracle runs is extra. Every phase merges to
`main`. No phase places orders or edits the crontab.

| Phase | Entry | Work | Exit criteria | Artefacts | Decides | Effort |
|---|---|---|---|---|---|---|
| **P0 Save research and legacy references** | spec approved | copy the S1-S8 reports, evidence scripts and prototypes into the repo (Appendix D Q5); legacy tag + `requirements-legacy.lock` + `legacy_env.sh`; frozen Parquet tape + sha; 17-fold and 2-fold reference re-run on legacy; per-strategy legacy references (Appendix D Q8); live golden (7.3 item 4); fix the changelog defect | 17-fold reproduces 0.592 / 17-of-17 or a documented fresh legacy baseline replaces it | `_legacy_reference_*.json`, tape sha, golden, saved reports | agent | 12 |
| **P1 Lock and pandas-3 stack** (pandas 3, numpy 2.5, vbt 1.2.0 + Rust, **legacy indicators**) | P0 | drop `pandas-ta` (OD15); `uv.lock`; `event_study` `.dropna()`; `.copy()` after concat; 44 `pct_change` sites (including `paper/feature_gate.py:50`, `paper/overlay.py:88`) with gap tests on 2.3.3 first; deprecations; `"1D"` | full suite green (S3: 1770/1770 after 2 fixes); signal diff A->B 0; 2-fold and 17-fold bit-for-bit except documented 1-ulp DSR; numba vs Rust bit-identical | `_p1_*.json`, signal report | agent | 14 |
| **P2 Engine adapter and Rust** | P1 | `engine/` config, selfcheck, provenance, `_compat`; presets, signals, portfolio (+chunker), stats (+guard, DSR), windows; boundary lint; L0-L3 tests; CI on the lock; image smoke step; watch job; Dependabot; runbook skill | all L0-L3 green on both engines; Rust dispatch probe passes; boundary test has only the legacy importers allow-listed | tests, workflow, skill | agent | 37 |
| **P3 Data and universe layer** | P2 | `data/bars.py`, `data/snapshot.py`; `lab.universe` (PIT mask, removal and terminal exits) | PIT mask 0 mismatches vs `eligible_at`; no research write to `ohlcv` | mask parity record | agent | 5 |
| **P4 Strategy port, `ensemble` first** | P3 | `vote`; new `ensemble` (OD1(a), OD3(a), OD14 control row); then the rest of the OD11 set; archive the others at the legacy tag | ensemble signal diff A->D within class (ii); each ported strategy reproduces its legacy result within class (ii) or carries an attributed diff | per-strategy JSON, tie audit | agent | 30 |
| **P5 WFO rewrite** (section 5.4: `generate_folds` mapped to bar positions, single-pass signals, `range_split` stacked windows for weights) | P4 | single-pass signal WFO; stacked weight WFO; per-window trades; compatibility shims (Appendix D Q3) removed at the end; delete legacy lab plumbing and the old pandas indicators once E1 is accepted | 17-fold "consistent" (7.4), scored with the legacy definitions (`definitions="legacy"`: E1 = legacy window convention, Sortino and CAGR); peak RSS under 6 GB; wall time recorded | `_p5_*.json`, memory record | agent proposes; **owner accepts** (it defines E1) | 15 |
| **P6 Validation: definition changes and re-baseline** | P5 | one run per change: Sortino, CAGR, (BB ddof=0 optional), PIT refresh (OD5), terminal exits (OD7), exit-on-removal (OD6), trade-count definition (OD12); then final references, `ENGINE_PROVENANCE.json`, banners, snapshot Engine column, docs; **delete `engine/legacy_defs.py`** and the `definitions="legacy"` branch once the Sortino and CAGR runs are signed (the expiry-guard test fails otherwise) | each change attributed and accepted or rejected by the owner; owner accepts E2; any verdict change vs SPY has an attributed cause | `_p6_<change>_*.json`, engine-migration report | **owner per change** | 22 |
| **P7 Trader rebuild and dry run** | P4 ensemble validated + P5 E1 accepted (P6 changes the trader only if the owner accepts a change that affects the ensemble) | section 8.2-8.4: signal path on `Ensemble.build()`, fail-closed checks, sanity gate, image smoke, immutable tags, `ggtrader-deploy` and `daily-trader-check` updates, tests, dry-run review | live golden passes; every dry-run session meets 8.4 item 3; owner reviews the record | dry-run log, deploy record | **owner** (re-enabling cron, OD18) | 14 |
| **P8 First upgrade drill** | P2 | run the runbook on the next vectorbt release (1.2.1 is in upstream master) | ACCEPT/RE-BASELINE/REJECT recorded | upgrade log row | agent; owner if results change | 4 |

**Total ~153 agent-hours (ESTIMATE, +/-40%).** The earlier ~178 (rounded to ~180) less the hours
that existed only to protect a running trader: the shadow phase (30) and the staged cutover D0-D3
(11), 41 in all; plus the trader rebuild and dry run (14, of which about 11.5 are the parts of S7's
43.5-hour live plan that survive: selfcheck and sanity gate, tests, skills and docs, image and tag
work, cleanup) and 2 for saving the research artefacts in P0. S7's own breakdown puts about 32 of
its 43.5 hours on protection-only work (shadow harness, offline O-vs-N replay, forward shadow,
authorization asks, rollback pull-test, the D1/D2 probation steps and the engine flag/crosscheck);
the spec's shadow phase had been costed higher than S7's. Re-enabling the trader has no date.

**Stop conditions (any one halts the phase and goes to the owner):** P0 legacy reference does not
reproduce and no fresh legacy baseline can be documented; P1 any numeric difference beyond the
documented 1-ulp DSR; Rust probe returns 0 calls; P4 or P5 an unexplained post-burn-in flip or a
winner/gate flip in the configured migration; any run killed at the memory cap twice after
chunking; any write to `ohlcv` from a research path; any dry-run failure, or any unexplained
difference between dry-run intended orders and the lab signals on the same frame; a new vectorbt
release that cannot be locked without loosening an engine bound.

## 10. Risks and open uncertainties (with the cheapest experiment for each)

| # | Risk / uncertainty | Cheapest experiment |
|---|---|---|
| 1 | Configured vbt kernels flip real signals on the **full** PIT universe and grid, not only on 150 sampled symbols with live defaults (S5 limit) | run the S5 signal-diff harness over all ~650 symbols and all 48 combos with the PIT mask applied (lab-only, minutes) |
| 2 | Single-pass signal WFO memory at real size (24k+ columns, dense per-column ATR stop arrays) exceeds the cap | P5 17-fold run under `MemoryMax=6G` with `/usr/bin/time -v`; calibrate the chunker |
| 3 | `last_exit` diverges under vbt warm-up rules even when last-bar signals match | add `last_exit` to the S5 live panel over 120 sessions (minutes) |
| 4 | Symbols with 60-200 bars (recent index additions) get different last-bar signals | live-golden subset of short-history symbols; count diffs |
| 5 | pandas 3 corrupts a Postgres write (timestamp units, tz) | scratch Postgres container, round-trip `_cache_to_db` and `refresh_benchmark_tape` records under pandas 3 |
| 6 | `n_trades` redefinition changes an NDH decision despite the sign argument | diff the 17-fold fold tables, legacy vs per-window counts |
| 7 | The sweep path's mask-after-time-stop ordering changed past `td_stop` results | rerun one `td_stop` grid with the mask before vs after; count changed exits |
| 8 | Upstream changes flat-curve Sharpe in 1.2.1 (commit title only) | the defaults canary and L1 flat-series test catch it on the first upgrade PR |
| 9 | vectorbt maintenance (contributor-driven, minor releases raise floors and change results, S2 1) | the watch job's cadence data; REJECT path keeps the last good pin |
| 10 | Rust parity bugs in a young engine | nightly fuzz parity (B's 102 configs) numba vs Rust |
| 11 | The trader is re-enabled on cron before the engine and strategy port are validated and the dry run is reviewed | OD18 (owner's say-so only); P7 entry requires P4/P5 validation; no agent edits the crontab |
| 12 | The legacy image digest is no longer pullable (reproducibility of closed research only; the legacy tag and venv lock are the primary path) | pull-test it once in P0 |
| 13 | quantstats/benchmark misalignment (#584) misleads a human report | align strategy and SPY indexes before calling; never feed gates |
| 14 | Dry-run and live behaviour differ silently (dry run writes, sends or skips something live does not, or vice versa), so a clean dry run does not predict a clean live run | before the first dry run, list every DB write, tape refresh and notifier call in both modes (8.4 item 2); a unit test that dry and live take the same path up to order submission |
| 15 | The stale AVB holding (4.05 shares, inactive ticker) distorts position count, the cap or portfolio value, or the rebuilt trader tries to sell it | OD17; a dry-run check that AVB is reported as untradable and never produces an order |

## 11. Left to the builder

- Exact module names inside `engine/` where this spec marks ESTIMATE; the `ParamGrid` API; the
  `Bars` container (per-field frames plus `missing_symbols`).
- The chunker's memory formula, calibrated on the P5 run.
- The exact sanity-gate bounds and the constituent-file age threshold, tuned from dry-run data.
- Digest values (base image, legacy image) at implementation time.
- How the 30 scripts that import `ggTrader.lab` are handled: port live drivers, archive the rest
  at the legacy tag.
- Whether the watch job's PR needs a personal token to trigger CI (GitHub token behaviour UNVERIFIED).
- Wording of the banner and the `ENGINE_PROVENANCE.json` schema details.

---

## Appendix A: Verification and trust

I re-checked a sample of each report against code, data and my own runs. Scratch scripts and
outputs: `/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/lead-checks/`.

| Src | Claim | Verdict | Evidence |
|---|---|---|---|
| S4 | Registry has 38 entries | VERIFIED | imported `STRATEGY_REGISTRY`: 38 |
| S4 | Class name and `file:line` for all 38 rows | VERIFIED | `inspect.getsourcelines`, 38/38 match |
| S4 | `target_kind` for all 38 rows | VERIFIED | 18 signals, 20 weights, all match |
| S4 | Grid raw/valid (8 sampled rows: ema_cross 20/16, ensemble 48, leveraged_trend 72, pairs 54, conviction_bb 36, ensemble_ic 15, mtf 32, fomc 6) | VERIFIED | `build_grid` |
| S4 | Snapshot verdict quotes (rows 1-16) | VERIFIED | `RESEARCH_SNAPSHOT.md:145-160` |
| S4 | Voter sets ALL/FIVE/DEFAULT/THREE and constructor defaults | VERIFIED | `ensemble.py:31-42, 67-91` |
| S4 | `min_agree_exit` None -> `min_agree`; RSI independent exit; mask before time stop | VERIFIED | `ensemble.py:94, 194-209` |
| S4 | vbb entries subset of bb; one spike = 2 votes; exits identical (2 votes) | VERIFIED | code `indicators.py:67-79, 237-263`; numeric: 0 of 220 vbb entries outside bb, exit frames equal |
| S4 | Live path reaches exactly 9 `indicators` functions | VERIFIED | `signal_runner.py:60-106`, `ensemble.py:142-232` |
| S4 | conviction recomputes SMA/STD; max_effect `pct_change` + rolling max; mtf weekly resample | VERIFIED | `conviction.py:56-59`, `max_effect.py:19,52`, `indicators.py:283` |
| S4 | Primary counts A5/B4/C0/D29 | PARTLY | sums to 38, but `ensemble` is put in B although S4's own rule (D for any NO-GO verdict) and its quoted verdict "Daily-bar construction NO-GO" make it D (A5/B3/D30); the exception is reasonable (it is live) but undocumented |
| S4 | Sweep axes listed | PARTLY | correct, but omits that the grid pins div 10 / vol 1.5 while live uses 20 / 2.0 (known from the replica audit; OD14) |
| S3 | Old 1770/0, new 1765/5, fixed 1770/0 | VERIFIED (logs) | `11_tests_old.log`, `10_tests_new.log`, `14_tests_new_final.log`; not re-run |
| S3 | 5 failures = `.stack()` NaN (3) + missing ruff (2) | VERIFIED | FAILED lines + `70_worktree_patches.diff` |
| S3 | Old vs new equity bit-identical | VERIFIED (independent re-run) | own script, 60 other symbols, 48 combos x {plain, 10% trailing}: 96 curves, max abs diff 0.0 |
| S3 | 3x sweep slowdown from concat fragmentation; `.copy()` fixes | VERIFIED (re-run) | `sweep_signals` 2.21 s old vs 6.71 s new; 60 blocks vs 1; RSI chain 38 ms vs 6 ms after copy |
| S3 | 1-ulp DSR difference; cold start 19.6 -> 4.5 s | COULD-NOT-CHECK | not re-run |
| S3 | Worktree context | note | worktree is at `bf5c749d`, not `6774b4d3`; patches are small and apply to HEAD files I read |
| S5 | Configured kernels: 0 entry diffs, exit diffs in burn-in except a PARA tie | VERIFIED (re-run, 40 other symbols) | 0 entry; 63 exit diffs, 1 after bar 200 = PARA 2022-01-03 |
| S5 | pandas 2 -> 3 changes 0 signals | VERIFIED (re-run) | 0/0 |
| S5 | vbt defaults are a different strategy | VERIFIED (re-run) | 1,372 entry / 4,293 exit XOR on 40 symbols |
| S5 | `year_freq` default 365; RSI/MACD defaults SMA; ATR default ewm | VERIFIED | `vbt.settings`, `inspect.signature` in venv120 |
| S5 | **DSR configured as `var_sharpe=1/(T-1)` reproduces ggTrader** | **WRONG** | vbt 1.2.0 divides `var_sharpe` by the annualisation factor; `1/(T-1)` gives 0.9920 vs `gates.dsr_check` 0.6202; `252/(T-1)` gives 0.6201666871 = legacy. Would have silently loosened the 0.80 DSR gate. |
| S5 | Legacy reference 0.592 / 3.08% / -5.28% / SPY 0.779 / 17-17 | VERIFIED | `_pit_rebaseline_20260923.json` |
| Spec 5.4 item 6 / OD2 (this spec's first draft) | DSR configured with `bias=True` | **WRONG in the spec; the P2 code is right** | Two reviewers confirmed `bias=False, var_sharpe=252/(T-1)`: the P0-P3 reviewer ran 12 seeded series (normal, t(3), skewed; T 83-790; 8-190 trials) against `gates.dsr_check` fed by `_compute_dsr_inputs` + `curve_stats`: `bias=False` matches to 4.4e-16, `bias=True` is off by up to 1.26e-4 (enough to flip a gate within 1e-4 of 0.80). Spec text corrected 2026-11-06. |
| S5 | Live panel 0 of 120 sessions differ | PARTLY | true for last-bar buys/sells; `last_exit` not compared |
| S6 | Vectorised PIT mask 0 mismatches vs `eligible_at` | VERIFIED (re-run) | 100 fresh dates (seed 20261106) x min_history 400 and 60, pandas 2.3.3: 0 mismatches |
| S6 | Constituent file ends 2026-06-02 | VERIFIED | read file |
| S6 | 17 no-bar members, coverage by year, YFData behaviour | COULD-NOT-CHECK | network and full-tape scans not repeated |
| S7 | Live path calls no vbt function; `overlay.py:12` imports `lab.simulate` | VERIFIED | `trader.py:519`, `signal_runner.py:60-200`, `overlay.py:12` |
| S7 | `MAX_POSITIONS=3`, alphabetical buys | PARTLY | alphabetical verified (`signal_runner.py:100`); 3 is an env override (code default 30, `risk.py:17,26-28`; comment `:8-9` and changelog say `.env` sets 3); `.env` not read |
| S7 | Freeze wording `next_steps.md:12-13, 30-31` (superseded 2026-11-06: owner dropped the freeze premise, Appendix E) | VERIFIED | read |
| S7 | Risk 12: changelog "retire stock picks" may decide scope | WRONG (framing) | superseded by owner instructions 2026-10-20; ensemble path is in scope (section 4.1) |
| S7 | Legacy image digest (now a reproducibility record only, not a rollback target; 2026-11-06) | VERIFIED (doc) | readiness review line 39; pull not tested |
| S8 | vectorbt licence Apache 2.0 + Commons Clause | VERIFIED | wheel `LICENSE.md`; rust METADATA; GitHub `NOASSERTION` |
| S8 | fja05680 MIT, quantstats Apache-2.0, TA-Lib BSD-2 | VERIFIED | `gh api` 2026-11-06 |
| S8 | PIT file differs only on 2023-10-18; upstream +9 dates to 2026-08-18 | VERIFIED (re-run) | downloaded upstream CSV and diffed |
| S8 | "fails the owner's OSI licence rule" | PARTLY | no OSI rule found in CLAUDE.md; the owner rule is no paid data/software. The exception is still worth recording (OD4) |
| S1 | `vbt.settings["engine"]` exists, default `auto`; int threshold raises on Rust | VERIFIED (re-run) | venv120 |
| S1 | Plan to move Sortino/CAGR/DSR to vbt defaults directly | conflict | resolved in Appendix B |
| S2 | 5 files import vectorbt; `Dockerfile:21`; `ci.yml:30`; compose `:latest` | VERIFIED | grep |
| S2 | uv lock of 147 packages, Python 3.12 | VERIFIED (file) | `s2/proto/uv.lock` |

**Scorecard:** 41 claims: **31 VERIFIED, 6 PARTLY, 2 WRONG, 2 COULD-NOT-CHECK** (one "note" and one
"conflict" row not counted).

**Trust ratings**
- S1 (Opus): HIGH on facts; its plan to adopt vbt statistics directly is superseded by S5's
  one-at-a-time rule.
- S2: HIGH; static claims all hold; release-cadence table not re-checked.
- S3: HIGH; its key measurements reproduced on my own code and different symbols.
- S4 (cheaper model): HIGH for facts and locations (every sampled row correct); MEDIUM for its
  classification counts.
- S5 (Opus): HIGH, with one WRONG configuration value that would have silently changed a gate. Every
  vbt configuration in this spec is therefore backed by a contract test, not by a report.
- S6: HIGH.
- S7: HIGH on code and mechanics; one framing error on scope.
- S8: HIGH on facts; overstates an "OSI rule".
- F: verified synthesis; recommendation superseded by owner decision 1, measurements reused.

**Independence limitation.** I am Opus checking reports partly written by Opus (S1, S5, F). That is
not independent review (CLAUDE.md delegation rule). I reduced the overlap by using different symbols,
dates and seeds and, for S3 and the DSR check, my own comparison code. For S5 and S6 I reused their
arm implementations (`s5_signals.run`, `pit.py`), so a shared bug in those functions would not be
caught. I chose claims mostly from the reports' own tables. I did not re-run the full test suite, any
17-fold WFO, B's fuzz suite, cold-start timings or network probes, and I did not read `.env`.

## Appendix B: Conflict-resolution log

| # | Conflict | Resolution | Reason |
|---|---|---|---|
| 1 | Stay on 0.28.5 (F, B) vs adopt 1.2.0 | Adopt 1.2.0 | Owner decision 1; F's measurements kept as design inputs |
| 2 | "1.2.0 is slower" (B, F: numba kernel 2.6x) vs S3 (3x slowdown is pandas-3 concat fragmentation) | Both true, different causes; fix fragmentation with `.copy()`, run Rust | Re-checked S3; production engine is Rust, so the numba kernel cost is off the hot path |
| 3 | Engine setting names: `GGT_VBT_ENGINE` (S1), `engine_requested/effective` (S2), `SIGNAL_ENGINE` (S7) | `GGT_VBT_ENGINE` {rust,numba} for the vbt engine; provenance records requested and effective. **Superseded 2026-11-06:** the live-only `SIGNAL_ENGINE` {legacy,vbt} flag is dropped; the rebuilt trader has one signal path | Merging the knobs would have made "numba" a live fallback |
| 4 | numba as escape hatch (S1, S2) vs live fallback must be legacy pandas (S7) | numba for research only. **Superseded 2026-11-06:** no legacy live fallback either; the trader fails closed (section 8.3) | Cold numba compile does not fit the pre-close window; the legacy path is deleted after validation |
| 5 | Python `>=3.12,<3.13` (S2) vs `>=3.11` (pyproject) vs 3.11-3.14 (vbt) | `==3.12.*` | Matches the image; halves the lock; one minor to test |
| 6 | Fold design: single pass (S1) vs splitters (S2 optional) vs skfolio (S8) | Keep `generate_folds`; single pass for signals; stacked fresh-start windows for weights | F verified equivalence for signals only; weight path starts flat (F D3); skfolio gives 16 folds not 17 |
| 7 | Definitions: S1 adopts vbt Sortino/CAGR/DSR directly; S5 configures first, then one change at a time | S5's order; owner decides each (OD1, OD2) | Bundled changes cannot be attributed; Sortino can flip winners |
| 8 | DSR: vbt cross-column default (S1) vs configured `1/(T-1)` (S5) | Configured with `252/(T-1)` | Re-checked; S5's value is wrong for 1.2.0 |
| 9 | Lock tooling: uv (S2) vs pip constraints (S3) vs either (S7) | uv | Universal hashed lock, fast, Dependabot support, export escape hatch |
| 10 | Strategies call `vbt.*` directly via re-export (S1) vs no vbt outside engine (S2) vs one compat module (S5, F) | Presets in `engine/indicators.py` for constructors and `Portfolio`; `.vbt` accessors allowed in strategies | The silent-default trap lives in constructors and stats (S5); accessors measured equal (0 cells). Presets are one-line configuration, not maths, so decision 4 still holds |
| 11 | Shadow: S5 13:05 PT host job, two engines on one frame vs S7 12:50 PT three-way O/O'/N | **Superseded 2026-11-06:** no shadow job; replaced by the live golden plus a dry run (section 8.4). (Was: S7 design + S5 pass criteria + `last_exit`.) | The trader is stopped, so there is no live run to shadow |
| 12 | Compose pin now (S2 "step 0") vs ask first (S7 A2) | **Superseded 2026-11-06:** OD9 withdrawn; immutable tags are part of the rebuild (8.3), not a prerequisite | With cron disabled a push to `main` cannot reach a running trader |
| 13 | S3 `.stack()`/`.copy()` fixes vs S1 deleting `indicators.py` | Apply both: fixes in P1, deletion at the end of P5 | P1 runs legacy indicators on the new stack; S1's deletion waits for validated presets (revised 2026-11-06; was "P4/post-cutover") |
| 14 | `engine_meta` (S1) vs `provenance` (S2) column | `provenance JSONB` with S2's schema plus S1's `definitions` | One name; S2's schema is fuller |
| 15 | `vbt.Data` adapter (S6 "ship it") vs not mentioned (S1) | Do not adopt | OSS `Data` is a dict holder; it adds an upgrade surface for no capability |
| 16 | Effort: S2 28 h, S3 11-12 h, S7 43.5 h, S1 size only, S5 phases | ~180 h total; revised 2026-11-06 to ~153 h after removing live-protection work (section 9) | Sum with overlap removed (lock counted once) plus the uncosted rewrite and port |
| 17 | S5 P1 "numba, no definition change" then P2 "Rust on" vs S3 running Rust from the start | P1 runs both engines and requires bit-identity | B, S3 and S5 all measured simulator bit-identity; one phase is enough |
| 18 | Ensemble scope (`changelog.md:268`, S7 risk 12) | In scope | Owner instructions 2026-10-20 supersede it; fix the line |

## Appendix C: Source reports

Scratchpad root: `/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/`.

| Id | File | Subject |
|---|---|---|
| A | `vbt-research/A-vectorbt-0.28.5-inventory.md` | 0.28.5 API inventory (cheaper model) |
| B | `vbt-research/B-vectorbt-1x-and-rust.md` (+`B-bench/`) | 1.x and Rust benchmarks, parity fuzz |
| C | `vbt-research/C-ggtrader-handrolled-inventory.md` | hand-rolled code inventory (cheaper model) |
| D | `vbt-research/D-vectorbt-wfo-and-agents.md` | WFO capabilities, primitive equality |
| F | `vbt-research/F-synthesis-and-verdict.md` (+`F-checks/`) | verified verdicts on A-D |
| S1 | `vbt-spec/S1-target-architecture.md` | target architecture |
| S2 | `vbt-spec/S2-upgradeability.md` (+`s2/`) | upgrade mechanics |
| S3 | `vbt-spec/S3-stack-migration.md` (+`S3-evidence/`; worktree `.claude/worktrees/agent-abed3ab263d5272c0`) | hands-on stack migration |
| S4 | `vbt-spec/S4-strategy-port-inventory.md` | strategy port inventory (cheaper model) |
| S5 | `vbt-spec/S5-validation-protocol.md` (+`S5-evidence/`) | validation and re-baseline |
| S6 | `vbt-spec/S6-data-and-universe.md` (+`S6-evidence/`) | data layer and PIT mask |
| S7 | `vbt-spec/S7-live-trader-integration.md` | live trader integration |
| S8 | `vbt-spec/S8-complementary-tools.md` (+`S8-evidence/`) | complementary tools |
| Lead | `vbt-spec/lead-checks/` | this spec's re-runs (Appendix A) |

## Appendix D: Builder-check resolutions (2026-11-06)

A fresh reviewer read this spec as the builder and raised eight questions. Five were settled by
the lead (marked **LEAD CALL**, change them if you disagree); three needed the owner (marked
**OWNER**). Questions 1 and 2 were answered by the owner's 2026-11-06 decision (Appendix E);
question 7 is still open.

### Lead calls

| # | Question | Resolution (LEAD CALL) |
|---|---|---|
| 3 | Which interfaces stay stable for "kept unchanged" modules (`blend.py`, `allocation.py`, `lab/cli.py`, `signals.py`)? | P2-P5 keep `run_wfo`, `WfoResult`, `combo_name`, `build_grid` and `curve_stats` importable from their current paths as thin compatibility shims over the new modules, so `--blend` and `--strategy` keep working while the rewrite lands. Adapting callers is a P5 task and the shims are deleted at the end of P5 (phase numbers revised 2026-11-06). A strategy the owner has not asked to port (closed NO-GO, OD11) fails with an explicit error naming the legacy tag; it never silently runs a different path. |
| 4 | Which function produces live signals, and where does numba compile time go? | Live signals come from the same `Ensemble.build()` called with a one-combination grid (the live defaults, OD14), plus a small extraction step in `paper/signal_runner.py` (`buys`, `sells`, `last_exit` on the last bar, then the existing alphabetical cap), built in the trader rebuild (P7, section 8). Compile time: the spec measured 0.4 s (Rust) against 19.6-190 s (cold numba), so P7 includes a fresh-container timing test that fails if any indicator used live does not dispatch to Rust; a numba cache baked into the image is the fallback only if that test finds a kernel without a Rust path. |
| 5 | Do the research reports and prototypes survive? | No: they live in the session scratchpad and are ephemeral. P0's first task is to copy the reports, evidence scripts and prototypes (PIT mask, signal-diff harness, contract-test fixtures, workflow and runbook drafts, lock file) into the repo under the location the research-placement test selects, so later phases build from files and not from prose. |
| 6 | Is the E1 baseline "configured to legacy conventions" or "after definition changes"? | E1 is the configured-to-legacy-conventions baseline, established in P5 and parity-checked against the legacy engine. E2 is the baseline after the owner-accepted P6 definition changes, established at the end of P6 (phase numbers revised 2026-11-06). `baseline_epoch` takes these two values and `compare()` refuses to compare across epochs without an explicit flag. The legacy Sortino and CAGR formulas needed to prove E1 live only in `tests/` as oracle implementations, never in `src/`, so the "no re-implemented vectorbt maths" anti-goal holds. **AMENDED 2026-11-06 (lead, after the P0-P3 review MAJOR-2):** that rule contradicted E1 being configured-to-legacy, because the engine had no way to score Sortino and CAGR the legacy way and P5 would have bundled two definition changes into E1. `src/ggTrader/engine/legacy_defs.py` is now a **time-boxed exception**: the two legacy formulas, reachable only through `stats.window_stats(..., definitions="legacy")` (the default while `BASELINE_EPOCH` is E1), stamped per statistic in provenance (differing definitions are not comparable), checked against `lab.metrics.curve_stats` on random and real-tape series, guarded by a test that fails once the epoch is E2 or later while the file exists, and **deleted at the end of P6** after the owner-approved per-definition runs. It is not a precedent: all other legacy formulas stay in tests. |
| 8 | Which legacy reference runs does P0 need? | P0 also records a reference for each strategy P4 will port (`conviction_bb`, `cross_asset_sleeve`, `retail_attention`, `xs_momentum`, the five voters). If the historical 0.592 figure does not reproduce, a fresh legacy run on the frozen tape snapshot becomes the working baseline and the difference is documented; the build does not halt for that alone. |

### Needs the owner

1. **ANSWERED 2026-11-06 (owner): no freeze; work merges to `main`.** *Question as asked:* what
   does the production freeze forbid before the readiness decision? Reading 1 allowed merges to
   `main` behind a compose digest pin; Reading 2 kept the program on a long `vbt12` branch with a
   separate legacy hotfix lane (the lead had recommended Reading 2). *Answer:* the owner stopped
   the trader and dropped the live-trader protections, so neither reading applies. Work merges to
   `main`; files shared with the old trader may change or be deleted; `lab.simulate` and
   `lab.sweep` need no parallel live copies (sections 3, 8.1).
2. **ANSWERED 2026-11-06 (owner): the defaults stand, except OD8 and OD9, which are withdrawn.**
   *Question as asked:* do the recommended defaults for OD1-OD16 stand? The two items that gated
   the shadow job and the branching no longer exist; OD11 (porting scope) and the other
   definition, licence and data defaults stand. OD17 (AVB) and OD18 (cron re-enable) are new owner
   items. The effort estimate is now about 153 agent-hours (section 9, ESTIMATE).
7. **OWNER (open): current model quota headroom.** CLAUDE.md requires the delegation plan to be
   sized to the owner's headroom, phased, with a usage report after each phase. The owner said
   earlier in this session that there is plenty of headroom for research; the build is much
   larger. Which phases may run, and where should it pause (default: stop for review after P5,
   when E1 is ready for acceptance)?

## Appendix E: Revision log

**2026-11-06, owner decisions (final).**
1. The owner stopped the paper trader: the 12:45 PT `scripts/paper_trade.sh` cron line and the
   06:00 `daily_pnl_report.sh` cron line are commented out in the user crontab.
2. The owner closed the paper-account stock positions: ADM, ALB and AMZN sold at market. AVB could
   not be closed (ticker inactive after the EQR merger; still shows 4.05 shares). About 114 SPY
   shares remain.
3. The owner said to "drop live trader protections" and "completely switch to vectorbt".
4. Nothing runs or auto-updates the old trader (the nightly updater does not scan
   `/home/flynn/ggTrader`; no watchtower). The intraday SPY observer timers and the Hermes
   "Intraday diary supervision" job are a separate research study that places no orders; they keep
   running and are not part of the trader.

**Sections changed.**
- Header: revision note.
- 1.2 success criteria (trader line); 2 goals (trader rebuild) and anti-goals (12:45-run item
  removed; silent-signal item reworded; premature re-enable added).
- 3 constraints: production-freeze row replaced by "paper trader stopped"; stack row no longer
  keeps live on the old image.
- 4.1 standing rules: "production and frozen" replaced.
- 4.2: OD8 and OD9 WITHDRAWN (numbers kept); OD5, OD14, OD16 freeze/shadow wording removed; phase
  references renumbered; OD17 (AVB) and OD18 (cron re-enable) added.
- 5.1 diagram and deletion list (old pandas indicators deleted after validation; no live copies);
  5.5 (no trader fallback, fail closed); 5.7 loader move; 5.9 `ggt paper`.
- 6 item 6: freeze red line replaced by "never edit the crontab".
- 7.3 item 5: the cross-asset shadow book is clarified as a separate study. No validation
  requirement changed.
- 8 replaced: "Live trader plan" (frozen lane, shadow design, switch and rollback, D0-D3 cutover,
  failure-mode table) became "Trader rebuild" (situation, what is built, kept safeguards, dry run
  and re-enable).
- 9 replaced: shadow phase and freeze-tied cutover removed; build order is P0 save artefacts and
  legacy references, P1 lock and pandas-3 stack, P2 engine adapter and Rust, P3 data and universe,
  P4 strategy port (`ensemble` first), P5 WFO rewrite, P6 validation and re-baseline, P7 trader
  rebuild and dry run, P8 upgrade drill. Effort ~180 -> ~153 agent-hours (ESTIMATE). Old-to-new
  phase map: old P2 + engine part of old P3 -> P2; universe part of old P3 -> P3; vote/ensemble part
  of old P3 + old P4 -> P4; WFO part of old P3 -> P5; old P5 + P6 -> P6; old P7 shadow removed; old
  P8 cutover -> P7 rebuild; old P9 -> P8.
- 10 risks: rows 11 (shadow partial bar) removed; 12 reworded to reproducibility; 14 (dry-run vs
  live difference) and 15 (stale AVB) added; 11 now premature re-enable.
- 11: phase and dry-run references.
- Appendix A: freeze and digest rows labelled superseded. Appendix B: rows 3, 4, 11, 12, 13
  labelled superseded or revised. Appendix D: Q3, Q4, Q6 phase references; Q1 and Q2 ANSWERED; Q7
  open.

**2026-11-06, P2b fix-up (lead decisions after the independent P0-P3 review).** Review:
4 MAJOR, 12 MINOR, 0 blockers, 21 mutants all killed. Lead decisions and dispositions (details and
evidence in `research/programme/vectorbt-engine/P2b-fixups.md`):

| Finding | Decision / disposition |
|---|---|
| MAJOR-1 window convention | `window_stats` follows the legacy convention (5.4 item 8): returns from the second bar, drawdown from the base bar. Tested against `curve_stats` on random and real-tape series. Fixed. |
| MAJOR-2 E1 vs Sortino/CAGR | E1 = legacy definitions. `engine/legacy_defs.py` is a time-boxed exception to "legacy formulas only in tests" (Appendix D Q6), deleted at the end of P6. Provenance stamps definitions per statistic. Fixed. |
| MAJOR-3 workflow token | Read-only `detect`/`try`, write only in `report`, SHA-pinned actions, no persisted credentials in the new-code job. Fixed. |
| MAJOR-4 P1 exit criterion | 17-fold parity on the new stack is being run by a separate job (`p1-parity/`); **the lead accepts, as class (ii) outside the WFO path, P1's extra last-digit differences**: the event-study means (9e-16, `DataFrame.mean(axis=1)` summation order) and the `idio_vol` tie order for 3 near-zero-variance delisted names. Anything beyond the documented 1-ulp DSR inside the WFO path is still a STOP. |
| MINOR-1 tape and definitions in `compare` | Fixed (`TAPE_DIFF`; any `definitions` difference is `CONTRACT_DIFF`). |
| MINOR-2 vol keys ignored | Fixed (`ValueError`). |
| MINOR-3 ATR after gaps | Documented and pinned by a test (class (ii)). |
| MINOR-4 terminal exits | Fixed (both uses of the series-end knowledge are labelled). The OD7 P6 run must report `same_bar_entries_suppressed`. |
| MINOR-5 inclusive end | Fixed (`legacy_exclusive_end`). Every legacy-parity run must use it. |
| MINOR-6 tz-aware span | Fixed. |
| MINOR-7 DSR flag in the spec | Fixed in 5.4 item 6, OD2 and Appendix A (`bias=False`). |
| MINOR-8 zero-trade guard scope | Open for P5 (docstring in `stats.sharpe`): decide zero entries vs zero exposure and apply the rule to every statistic in the composite. |
| MINOR-9 `getattr(x, "vbt")` | Fixed (plus `attrgetter`, `importlib` and `exec` forms). |
| MINOR-10 image and Dependabot | Fixed (digest pin; `python` major/minor ignored). |
| MINOR-11 changelog defect | Fixed with a bracketed supersession note (history kept). |
| MINOR-12 one-bar fold overlap | No E1 change; recorded as a P6 definition question (5.4 item 8). |
| P4 chunker finding | Fixed; `WORKING_ARRAYS` calibrated to 12 (5.5). |

Also changed: 5.4 item 6 and OD2 (`bias=False`), 5.4 item 8 (window convention), 5.5 (memory
calibration), 7.1 (class (i) ulp floor), 9 (P5 and P6 rows), Appendix A (DSR row), Appendix D Q6
(amended).

### Owner preference recorded 2026-11-06 (after P4): trust vectorbt's numbers

The owner said: "I would trust vectorbt numbers versus my numbers." Consequences for this spec:

- Where a legacy ggTrader definition and a vectorbt definition differ (Sortino, CAGR, deflated-Sharpe
  kurtosis, default indicator forms), **vectorbt's definition is the intended end state**. The
  per-definition P6 runs are therefore no longer "adopt or keep" questions; they are the measured,
  one-at-a-time route to adopting vectorbt's definitions, each with its own JSON and a note of what
  moved. The owner pre-approves adopting vectorbt's Sortino and CAGR once their runs are recorded.
- Engine-mechanics parity (E1, legacy definitions) is still built first, because it is the only way to
  show the new engine runs the same simulation; it is not a statement that the legacy numbers are
  right. `engine/legacy_defs.py` stays a time-boxed attribution tool and is deleted at the end of P6.
- Indicator forms (OD1): the plan stays staged. E1 uses the configured presets so signals can be
  proven identical to legacy; after E1 is accepted, vectorbt's default forms are run as their own
  research question (a different strategy, so judged by the forward-paper standard, not assumed
  better). The owner's preference decides which form the rebuilt trader uses if the results are
  comparable.
- Two settings are not "numbers" and are not changed by this preference: the annualisation year for
  equities is 252 trading days (vectorbt's 365-day default is a crypto setting and inflates Sharpe by
  about 20%), and the flat-curve guard (a strategy that never trades must not score an infinite
  Sharpe) stays mandatory.

### Owner decision 2026-11-06 (after P4 review): drop legacy definitions; vectorbt overrides

The owner answered the question "keep or drop the legacy-definitions step" with: **drop it; prefer
vectorbt's definitions; where there are future conflicts, vectorbt overrides.** This supersedes the
staged "E1 = legacy definitions" plan above (section 7.1 class iii, Appendix D Q6, OD2, and the
owner-preference note).

- **Baseline (E1) uses vectorbt's definitions** (Sortino, CAGR, annualised measures, deflated-Sharpe
  kurtosis, and the vectorbt default indicator forms are the intended end state).
- **Correctness proof is equity-level, not definition-level.** The new single-pass WFO must reproduce the
  legacy per-window equity curves (all combos, all folds, anchor slices) to the ulp floor; winners and
  gates are expected to move wherever the definition moved, and each such move is attributed in the
  P5 and P6 notes.
- **`engine/legacy_defs.py` is used once as a diagnostic, then deleted** (with its tests and expiry
  guard). Legacy `lab/metrics.curve_stats` stays only as an oracle for Sharpe/vol/drawdown conventions.
- **Conflict rule (standing):** when a ggTrader definition and a vectorbt definition disagree, vectorbt
  wins and ggTrader's version is removed, not kept beside it.
- **The only two exceptions, stated so the owner can overrule them:** (1) the annualisation basis for
  daily equity bars is 252 trading days (`year_freq="252 days"`); vectorbt's 365 default is a calendar
  assumption for crypto and overstates equity Sharpe by about 20%, so this is a data-calendar
  parameter, not a definition. (2) The flat-curve guard stays (a strategy that never trades must not
  score an infinite Sharpe and win selection), because it prevents a selection bug rather than
  redefining a statistic.
- Indicator forms: E1 still uses the configured presets for signal parity with legacy (P4 proof), and
  vectorbt's default forms are then run as their own research question; the owner's preference decides
  the form the rebuilt trader uses when results are comparable.

### P5 outcome recorded 2026-11-06 (builder note)

P5 built `ggTrader.lab.wfo_v2` and wired `ggt lab --wfo|--blend --engine {legacy,vbt}` (default legacy). Per the owner decision above, E1 now means vectorbt's own Sortino and CAGR; `engine/legacy_defs.py` and the `definitions="legacy"` branch were used once as a diagnostic and deleted in P5, not at the end of P6 (so section 5.1's row for `legacy_defs.py`, OD2's "legacy until P6" and Appendix D Q6's time-boxed exception no longer apply). The correctness proof is equity-level: 864 of 864 per-combination equity curves of the single full-history simulation are bit-identical to the legacy per-window simulations (18 window ends x 48 combinations, 17-fold SP500 ensemble). Weight strategies run as fresh-start stacked windows; their anchors are prefixes of one simulation from the first bar. `--max-leverage` now defaults to 1.0 (OD13). Details: `research/programme/vectorbt-engine/P5-completion.md`.

### P5b outcome recorded 2026-11-06 (review findings and disposition; builder note)

An independent review of P5 found 0 BLOCKER, 3 MAJOR, 7 MINOR (4 INFO). It confirmed the central
claims (no lookahead through the whole `run_wfo`; 420 of 420 equity curves bit-identical to legacy
on the reviewer's own windows; the 22 s run reproduces) and called the 17-fold core a clear NO-GO
against SPY. Disposition (details and tests in `research/programme/vectorbt-engine/P5b-fixups.md`):

| Finding | Disposition |
|---|---|
| MAJOR-1 degenerate Sortino wins selection; spec 5.6 zero-trade half not wired | Fixed. Zero-entry rule wired (spec 5.6 as written); **lead call:** Sortino needs `MIN_LOSING_DAYS` = 20 losing days (a guard, not a new definition; 20 is the smallest threshold that changes exactly the four flipped folds 8, 9, 16, 17). |
| MAJOR-2 a full-sample (lookahead) anchor passed all tests | Fixed with tests (independent anchor recomputation, perturbation after an interior `train_end`, DSR default pinned); the mutants are now killed. |
| MAJOR-3 per-fold anchor not recorded | Fixed (`anchor_combo`, `anchor_params`, `anchor_reason`, `deployed_combo` per fold; `anchor_history` in the JSON; separate table section). |
| MINOR-1 to MINOR-7 | Fixed or recorded in the P5b note (ties, combination order, dead anchors, DSR default, golden label, `--max-leverage` docs, symbol count). |

**E1 candidate, now guarded (17-fold SP500 ensemble, frozen tape, eval 2021-01-31 to 2026-04-30;
the earlier unguarded numbers are kept for the record):**

| Run | Variant | Gates | Anchor folds | OOS Sharpe / CAGR / MaxDD | SPY same days |
|---|---|---:|---:|---|---|
| core | unguarded (P5) | 1/17 | 17 | 0.391 / 3.50% / -9.58% | 0.779 / 13.02% / -22.09% |
| core | **guarded (E1 candidate)** | 1/17 | 17 | **0.391 / 3.50% / -9.58%** | same |
| DSR threshold 0 diagnostic | unguarded (P5) | 16/17 | 1 | 0.886 / 4.91% / -5.28% | same |
| DSR threshold 0 diagnostic | **guarded** | 17/17 | 0 | **0.637 / 3.41% / -5.28%** | same |

The core result is anchor-driven (the winner never reaches the money), so the guard does not move
it; the 0.886 diagnostic was the artefact (the fold-8 winner was a one-entry, four-losing-day
combination whose gate failure routed a lucky anchor). Neither variant beats SPY. This is a
research result, not a retirement recommendation.

### P5c, P8b and P9 recorded 2026-11-06 (builder notes)

**P5c: the legacy engine is deleted.** `ggTrader.lab.simulate`, `sweep`, `harness`, `wfo`, `metrics`,
the hand-written indicators, `STRATEGY_REGISTRY` and the 17 legacy strategy classes plus `ensemble*`,
`conviction`, `momentum`, `cross_asset_sleeve` and `retail_attention` left the main tree (git tag
`legacy-engine-vbt0285` restores them). `ggt lab` needs `--wfo` or `--blend`; `--engine`, `--sweep`
and `--max-sector-count` are gone. Size in lines of `*.py`: `src/ggTrader/lab` 27,445 to 19,912;
`src/ggTrader` 35,873 to 28,340 (tag: 28,953); `scripts/` 20,118 to 12,397; `tests/` 42,901 to 33,967.
48 test files and 34 source files were deleted, 32 closed-study drivers archived to
`research/programme/vectorbt-engine/legacy-scripts/`; the legacy comparison tests became frozen
goldens first. The vectorbt import allow-list went from eight entries to one
(`scripts/legacy_reference_runs.py`). Details: `P5c-completion.md`, `P5c-ledger.csv`.

**P8b: the upgrade runbook rehearsal fixed.** 16 runbook and tooling defects (P8) closed; seven new
tripwires (Bollinger against pandas on the tape, a large-offset rolling-std test, the DSR with NaN
and fat tails against an independent formula, numba against Rust on degenerate series, the
trailing-NaN drawdown, the Sortino sibling of the flat-curve test); every simulated definition
change of the mutation check is caught; the version-string test no longer fails on a bump alone.
The runbook is `.claude/skills/vbt-upgrade/SKILL.md`; details: `P8b-fixups.md`.

**P9: cleanup.** (a) `engine.stats._std` no longer emits `RuntimeWarning: Degrees of freedom <= 0`
for a window with fewer than two observations (same NaN, bit-identical results; a `-W error` test).
(b) `p5_wfo_validation.py`, `p7_live_golden.py`, `p3_pit_parity.py`, `legacy_reference_compare.py`,
`legacy_reference_runs.py` and `engine_parity_evidence.py` default their output to a scratch folder
under the system temp directory; writing into `research/` needs an explicit `--out`/`--out-dir`.
(c) Every live `scripts/*.py` has an `ArgumentParser` (or delegates to a library one) and a main
guard; `tests/scripts/test_scripts_have_help.py` proves this by AST and then runs `--help` in a
stripped environment. (d) The `v2` names are gone: `lab/strategies/v2/*` is `lab/strategies/*`
(`ggTrader.lab.strategies.contract`, `.registry`, `.ensemble`, ...), `lab/wfo_v2/` is `lab/wfo/`,
`tests/lab/strategies_v2` and `tests/lab/wfo_v2` are `tests/lab/strategies` and `tests/lab/wfo`.
The names `REGISTRY_V2` and `LEGACY_ONLY_STRATEGIES` are kept, and the provenance contract and
epoch fields did not change. Section 5.1's file table and every module path in sections 5 to 8
that reads `wfo_v2` or `strategies/v2` now means `wfo` or `strategies` (the older text above is a
historical record and was not rewritten). Details: `P9-cleanup.md`.

**P10: the uncontroversial part of the final audit's punch list.** (a) Strategy discovery:
`registry.discover()` scans `lab/strategies/` for classes that implement the contract (`name`,
`kind`, `grid`, `build`), in module-name order; a malformed module raises an error naming the file;
`REGISTRY_V2` is now `REGISTRY` (section 5.3's "registered with `@register`" now reads "found by
scanning the package"; `@register` stays for strategies outside it). The tests that pinned the
strategy count assert properties, `test_leak.py` runs the lookahead check over every registered
strategy, `engine.indicators.values` strips the parameter level, and `docs/adding_a_strategy.md`
has a worked example that a test runs. (b) Provenance gains `strategy_sha256`/`strategy_files`
(`strategy_sources` for a blend); `compare` ignores them like `git_sha`. `run_wfo` raises for a
signal strategy without a point-in-time mask unless `allow_unmasked=True` (section 5.4's "warning"
is now an error). (c) One deflated Sharpe: `lab/gates.py` lost its own `psr`/`expected_max_sr`/
`dsr_check` (the oracle copy is `tests/oracles/legacy_dsr.py`); `engine.stats.dsr` accepts one trial
(PSR against zero, by vectorbt with `var_sharpe=0`); the section 5.4 item 6 fallback to the legacy
function is gone; multi-combination results are unchanged, a one-combination 17-fold run moved by
at most 1.1e-16. (d) Removed or archived (research reports cite them): `ggt ingest`,
`postgres_ingestor`, `regime`, `train_gate`, `paper/feature_gate` and its hook in the live signal
path (section 8's ML-gate sentence no longer applies; the signal golden is 120/120), five loaders of
legacy-tag-only strategies with their backfill drivers, `scripts/mcp_server.py`. Kept and listed:
the intraday and event-timing tools, `tax_lots`, `a23_rebalance`, `replica`, `blend`,
`momentum_screen`, `eligible_at` (still used by the replica audit). (e) The stale statements that the
trader runs on cron are corrected in `AGENTS.md`, `README.md` and `docs/roadmap.md`. Details and
numbers: `P10-completion.md`, `P10-ledger.csv`.
