# S1: Target architecture for ggTrader on vectorbt 1.2.0 + vectorbt-rust

Date 2026-11-06. Author: Opus 5.5 (S1 workstream). Repo read-only at `6774b4d3`. Scratch evidence in this folder:
`s1_engine_selfcheck.py` (+`_out.txt`), `s1_ensemble_contract_proto.py` (+`_out.txt`), `s1_flat_sharpe_out.txt`, all run in
`../venv120` (vectorbt 1.2.0, vectorbt-rust 1.2.0, pandas 3.1.1). Prior work cited as F/D/B/C = `../vbt-research/*.md`.
Owner decisions 1-5 are taken as given. Where earlier reports (F section d, B section 7) advised staying on 0.28.5, this spec
does not reopen that question. It uses their measurements only to design around the traps they found.

## 0. Recommendation in brief

**Architecture (a), thin adapter, plus a frozen live lane.** One new package, `ggTrader.engine`, is the only code that
imports `vectorbt`. It pins the engine, the year length and the versions, proves Rust dispatch at startup, and re-exports a
configured `vbt` for strategies to use. Strategies become *declarative*: a parameter grid, plus a function that builds wide
entry/exit (or weight) frames from built-in vbt indicators and signal accessors. The adapter turns that into one grouped
`Portfolio.from_signals` / `from_orders` call over all combos. ggTrader keeps only what vbt OSS has no concept of: point-in-time
(PIT) universe, WFO policy (gates, anchor, breaker, selection), persistence, tax, and the paper trader.
**The live trader stays on its current image (pinned by SHA tag) until the production freeze lifts and a forward shadow comparison passes.**
It cannot run in place: vbt 1.2.0 needs pandas>=3.0.3 / numpy>=2.4.6 (B section 6.1), so the two stacks cannot share one image.

## 1. Three architectures compared

| Criterion | (a) Thin adapter `ggTrader.engine` + vbt-native strategies | (b) Fork / vendor vectorbt | (c) Keep structure, swap internals |
|---|---|---|---|
| Absorbing upstream releases | **Best.** One pin pair (`vectorbt[rust]==X`). Every call site is in `engine/` or uses re-exported built-ins. A defaults-canary test (section 5) names each drifted default. | **Worst.** Every release becomes a merge of a 133-commit, +99k-line diff (B section 2: v0.28.5 to v1.2.0), plus the Rust crate (~15.6k lines, B section 4.1). Agents would maintain a fork. | Medium. vbt calls stay spread across `simulate.py`, `sweep.py`, `signals.py` and `indicators.py` (C section 1), so an upgrade touches many files and silent defaults hide in each. |
| Code ggTrader owns | Least on the engine path: roughly 4.6k lines become 2.4-2.9k (section 7, ESTIMATE). Policy, data and tax code are unchanged. | Most: all of ggTrader plus about 52.7k lines of vectorbt (D section 5.4) and the Rust crate. | About today's amount. `indicators.py` shrinks, but the per-combo Python loops (`ensemble.py` sweep_signals, `sweep.py` stop groups) remain. |
| Risk to live trader | Low **if** the live lane is pinned. The live code is untouched until cutover, which is gated (section 6). | High. A forked kernel under live orders. Also licence: Commons Clause (B section 4.1). | **High.** The same files the live trader imports (`overlay.py:11-14`, `signal_runner.py:10-12`) change in place, and a `:latest` rebuild ships them (`docker-compose.yaml:3`). |
| Performance | Good. One simulation per stop group over the full history (verified exact for signals, F section D3). Indicators run once per unique parameter. Prototype: 24,000 columns x 1,700 bars, 48 groups, 16 s, 3.86 GB peak (`s1_ensemble_contract_proto_out.txt`, synthetic). | Same kernels, so same speed, unless the fork adds parallelism (not proposed). | Gains only if the refactors in F Phase 1 are also done. The O(T^2) PIT mask is outside vbt (F P1). |
| Fits owner decision 4 ("thinnest glue") | Yes | No | Partly |

A fourth variant was considered and rejected: one adapter that wraps *every* vbt indicator in a ggTrader function. It hides
vbt from strategy authors, which defeats owner decision 2 and adds a wrapper layer to maintain on every upgrade.

## 2. Layering and package layout

```
                 ggt.py / cli.main
                        |
          +-------------+--------------+
          v                            v
   lab.cli (ggt lab / --blend)     paper.trader (ggt paper)   <- live lane: FROZEN image until cutover
          |                            |
          v                            v
   lab.wfo  (folds, select, gates, breaker, anchor, run)     paper.* (broker, risk, sweep, splits...)
          |                            |
          +----------+-----------------+
                     v
            lab.strategies  (declarative specs, registry)
                     |
          +----------+-----------+
          v                      v
     lab.universe (PIT)     ggTrader.engine  (ONLY importer of vectorbt / vectorbt_rust)
          |                      |   config . selfcheck . grid . simulate . windows . stats
          v                      v
     data/, lab.data        vectorbt 1.2.0 (+ vectorbt-rust 1.2.0)
          ^
   lab.persist (TimescaleDB lab_*), lab.{tax_lots,a23_rebalance,replica,century_leverage,
   significance,event_study,*_screen,*_data,event_timing_*}  (unchanged research tooling)
```

**Import rules** (enforced by a pytest that parses imports, so no new dependency is needed):
1. `vectorbt` / `vectorbt_rust` may be imported only in `ggTrader/engine/`. Everything else uses `from ggTrader.engine import vbt`, which is the configured module. This gives one configuration point and makes the upgrade surface something grep can find.
2. `engine` imports nothing from ggTrader. `lab.strategies` may import `engine` and `lab.universe`. `lab.wfo` may import `strategies`, `engine`, `universe` and `persist`.
3. `paper` may import `lab.strategies`, `engine` and `lab.allocation` (as today: `overlay.py:82`). Nothing in `lab` imports `paper`. **Existing exceptions:** `replica.py:14-27` and `train_gate.py:38` import paper guard code. Allow-list them now. Moving the pure guards (`risk`, `cash_sweep`, `catastrophe_stop`) into a shared `ggTrader.policy` package is a later step that can happen only after cutover, because the live code is frozen.

**`ggTrader/engine/` modules (new):**

| Module | Responsibility |
|---|---|
| `config.py` | `EngineSettings` (engine, year_freq `'252 days'`, freq `'1D'`, strict flag). `configure()` sets `vbt.settings["engine"]` and `vbt.settings["returns"]["year_freq"]`. `stack_metadata()` returns versions, engine and the dispatch probe result. |
| `selfcheck.py` | Startup dispatch probe (section 5). |
| `grid.py` | `ParamGrid`: product plus validity predicate (ports `sweep.py:38-74`), producing a column `MultiIndex` of combos. |
| `simulate.py` | `run_signals(out, close, *, size, stops)` and `run_weights(weights, close)`, each one grouped call. This replaces `lab/simulate.py`'s `simulate_signals` / `simulate_weights` and the stop-group loop. |
| `windows.py` | Calendar folds to bar positions (`searchsorted`), slicing helpers, `range_split` wrapper for the weight path. |
| `stats.py` | Window stats from `pf.returns()` via the returns accessor, a non-finite guard, and per-window trade counts from `pf.trades.records`. |

**Deleted from `src/ggTrader/lab`:** `simulate.py`, `sweep.py`, `metrics.py` (Sharpe, Sortino, max DD, CAGR and vol all go to vbt),
`harness.py` (non-WFO runs become a one-fold case of `lab.wfo`), `strategies/indicators.py`, the per-combo loops in
`signals.py` / `ensemble*.py` / `conviction.py`, and the `wfo.py` sweep plumbing (`_sweep_fold*`, `_extract_grid_arrays`, the anchor
re-simulation that F D1 showed is identical to the train sweep). `kelly.py` trade pairing becomes `pf.trades.records`.
**Kept:** `strategy.py` (rewritten contract), `data.py`, `persist.py`, `allocation.py`, `blend.py`, `cli.py`, `gates.py` (NDH
logic; DSR moves to vbt, see section 4), the policy half of `wfo.py`, and all non-engine research tooling.

**What vectorbt owns and what ggTrader must own:**

| vectorbt owns | ggTrader must own, and why |
|---|---|
| Indicators MA, MSTD, BBANDS, RSI, MACD, ATR, STOCH, OBV (`dir(vbt)`, probed) | **PIT universe / membership mask.** vbt has no membership concept (D section 4). Survivorship control is core to every honest number (F section b). |
| Crossovers (`.vbt.crossed_above/below`), signal ops (`.vbt.signals.AND/OR/first/fshift/nth`, `generate_stop_exits`, STX) | **Time stop (`td_stop`).** `from_signals` 1.2.0 has no time-stop argument (signature probed: only `sl_stop`, `tp_stop`, `sl_trail`). Keep the one-line `entries.vbt.fshift(n)` OR. |
| Parameter products (`param_product=True`), column levels | **Grid validity rules** (`sweep.py:38-61`). `run_combs` cannot express them (D section 2). |
| Simulation `from_signals` / `from_orders`, stops, sizing, cash sharing, grouping | **Live position cap** (`MAX_POSITIONS`). In vbt this needs a numba callback, and callbacks never run on Rust (B section 4.3). Keep it in paper, outside the engine. |
| Returns accessor: Sharpe, Sortino, Calmar, max DD, annualised return/vol, DSR (kurtosis fixed in 1.1.1, B section 2.1) | **Selection policy:** composite score, anchor, live-param pick, NDH plateau gate, circuit breaker, shadow re-entry, WFE. None exist in vbt OSS or PRO (D section 4). These are research decisions, not math. |
| Splitters (`range_split`) for stacked fresh-start windows | **Calendar fold dates** (`generate_folds`, 17 lines). vbt splitters count bars and drop the index (D section 1.1). |
| Trade records (`pf.trades.records`) | **Persistence** to TimescaleDB plus JSON artefacts. vbt can only pickle (D section 4). |
| | **Tax lots, A23 book, replica, century study.** vbt OSS has no tax handling and no cash flows (D section 4). |
| | **Paper trader:** broker, risk, sweep, split and dividend checks. vbt has no live-trading layer. |
| | **Sleeve blend overlay** (`allocation.py`). It is roughly 100 lines of pandas over curves, and vbt has nothing equivalent (D section 4). |

## 3. Strategy contract

A strategy is a frozen dataclass registered with `@register("name")`. Discovery goes through the existing single
`STRATEGY_REGISTRY` dict, now filled by the decorator, and the CLI choices still derive from it (`registry.py`). The contract:

```python
class SignalStrategy(Protocol):
    name: str; kind: Literal["signals"]
    grid: ParamGrid                       # values + valid(combo) predicate; default = live params
    def build(self, data: Panel, grid: ParamGrid) -> SignalOut: ...
    # SignalOut(entries, exits: bool DF; columns = MultiIndex(grid levels..., "symbol")
    #           size: float|DF|None; sl_stop/tp_stop: float|DF|None; sl_trail: bool; td_stop: int|None)
class WeightStrategy(Protocol):
    name: str; kind: Literal["weights"]; grid: ParamGrid
    def build(self, data: Panel, grid: ParamGrid, rebal: pd.DatetimeIndex) -> pd.DataFrame
    # target weights (NaN = hold), same column MultiIndex; engine -> from_orders(size_type="targetpercent")
```

The engine adapter (not the strategy) does four things. It ANDs the PIT mask into entries **before** the time stop, which also
fixes the sweep/live ordering mismatch F found (F section C, second bullet). It applies `td_stop`. It tiles `close` to the grid
columns. It makes one `from_signals` call with `group_by=grid levels` and `cash_sharing=True`. Strategies never call
`Portfolio`. They also never pass a behaviour-affecting indicator argument implicitly: `window`, `ewm`, `alpha`, `ddof` and
`*_ewm` are always written out, so a changed upstream default cannot alter a strategy silently (section 5 canary).

**Ensemble sketch.** The prototype runs; its timing is in section 4.

```python
@register("ensemble")
@dataclass(frozen=True)
class Ensemble:
    grid = ParamGrid(min_agree=[2,3,4], min_agree_exit=[1,2], bb_alpha=[2.0,2.5],
                     rsi_oversold=[25.0,30.0], ema_fast=[10,20], live=dict(min_agree=2, ...))
    def build(self, d, g):
        bb   = vbt.BBANDS.run(d.close, window=20, alpha=g.values("bb_alpha"), ddof=1)  # once per alpha
        rsi  = vbt.RSI.run(d.close, window=27, ewm=True)      # vbt definition; span 27 ~= Wilder 14
        emaf = vbt.MA.run(d.close, window=g.values("ema_fast"), ewm=True)
        emas = vbt.MA.run(d.close, window=50, ewm=True)
        macd = vbt.MACD.run(d.close, 12, 26, 9, macd_ewm=True, signal_ewm=True)
        vma  = vbt.MA.run(d.volume, window=20, ewm=False)
        voters = {   # each voter: dict param_value -> (entry ndarray, exit ndarray), computed once
          "bb":  per_value(bb, "bb_alpha", lambda b: (b.close_crossed_below(b.lower), b.close_crossed_above(b.middle))),
          "rsi": {o: (rsi.rsi_crossed_below(o), rsi.rsi_crossed_above(50.0)) for o in g.values("rsi_oversold")},
          "ema": per_value(emaf, "ema_fast", lambda f: (f.vbt.crossed_above(emas.ma), f.vbt.crossed_below(emas.ma))),
          "macd": divergence(d.close, macd.hist, window=10),   # ggTrader rule, ~5 lines
          "vbb": volume_gate(voters["bb"], d.volume > 1.5 * vma.ma),
        }
        return vote(voters, g, entry_key="min_agree", exit_key="min_agree_exit",
                    independent_exit="rsi", stops=dict(sl_stop=0.10, sl_trail=True), size=0.03)
```

`vote()` is the one generic helper ggTrader owns for voting ensembles (~40 lines). It loops over combos only to sum
precomputed int8 arrays and `np.hstack` them into one wide frame, so each indicator runs once per unique parameter. This is the
memoisation F Phase 1c asked for, obtained by construction. vbt has no N-of-M vote primitive (F section b: "KEEP"), so this glue is
genuinely needed. Conviction sizing (`EnsembleConviction`) returns `size` as a per-column float frame with `size_type="percent"`.

**Weight strategy sketch (12-1 momentum):**

```python
@register("momentum")
@dataclass(frozen=True)
class Momentum:
    grid = ParamGrid(lookback=[252], skip=[21], top_n=[50])
    def build(self, d, g, rebal):
        out = {}
        for lb, sk, n in g.combos():                                 # tiny grids; vectorised inside
            score = d.close.shift(sk) / d.close.shift(lb) - 1        # pandas; no vbt equivalent
            score = score.where(d.pit_member)                        # PIT from lab.universe
            rank = score.loc[rebal].rank(axis=1, ascending=False)
            w = (rank <= n).div((rank <= n).sum(axis=1), axis=0)     # equal weight top-n
            out[(lb, sk, n)] = w.reindex(d.close.index).shift(1)     # takes effect next bar; NaN = hold
        return g.stack(out)                                          # MultiIndex columns
```

The engine then calls `from_orders(size=w, size_type="targetpercent", call_seq="auto", cash_sharing=True, group_by=levels)`.
The per-date `select()` / `Plan` loop goes away for vectorisable strategies. Event and alt-data strategies (insider, PEAD,
FOMC, and others) keep a small `select()` that builds their weight frame, since their inputs are not price panels. Note that
`from_signals` with `targetpercent` is unsupported on every engine (B section 3.1, 18 fuzz configs). Weight strategies therefore
must use `from_orders`.

## 4. Walk-forward design

1. **Folds:** keep `generate_folds` (12m/3m/3m calendar, `wfo.py:194-210`). `engine.windows.to_positions(index, folds)`
   maps dates with `searchsorted`, which handles non-trading dates such as Sunday 2021-01-31 (D section 1.1).
2. **Signal strategies: single pass.** Run one `from_signals` per (strategy, universe, stop group) over the whole history. Then
   per fold, slice `pf.value()` / `pf.returns()` rows for the anchor (start to train_end), train and test windows, and the live
   window. F verified this on the real `_sweep_fold`, 120 SP500 symbols and 80 combos: **max abs equity and Sharpe diff 0.0**
   (F section D3). Today's code already carries positions into windows, so slicing matches its semantics. No splitter is needed
   here, and it would be wrong: stacking gives a fresh start that today's signal path does not have.
3. **Weight strategies: `range_split` stacking.** Today's `_sweep_fold_weights` starts flat at each window (F section D3),
   which matches `close.vbt.range_split(start_idxs, end_idxs)` exactly: windows stacked as columns, `split_idx` level, fresh
   cash per (combo, fold). One `from_orders` call covers all folds. Anchor and train share a simulation inside each fold;
   folds never share one.
4. **Trade counts per window:** count `pf.trades.records` whose `entry_idx` falls in the scoring window. Today
   `trades.count()` covers the whole simulation, which is already definitionally wrong (F section D3: 72 of 80 combos differ).
5. **Selection and gates read vbt stats.** `stats.window(pf, a, b)` returns `rets.iloc[a:b].vbt.returns(freq="1D",
   year_freq="252 days")` → `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `annualized`, `calmar_ratio`.
   **Non-finite guard (ggTrader owns this):** vbt 1.2.0 still returns `inf` for a flat curve and **4.5e16** for constant
   returns (`s1_flat_sharpe_out.txt`). Set Sharpe to NaN when std < 1e-12 or the window has zero trades, before any `idxmax` or
   composite. Composite score, anchor (CAGR floor), NDH, breaker and WFE then run on these values unchanged.
   **DSR** uses `deflated_sharpe_ratio` over the train-window returns frame of all grid columns: `nb_trials` = column count =
   grid size, kurtosis fixed in 1.1.1. Because vbt takes `var_sharpe` across columns rather than ggTrader's null 1/(T-1)
   (D section 3), **this is a changed statistic, and the 0.80 pass bar must be re-registered before use**.
6. **Memory budget:** dense float64 per array = T x cols x 8 B. At 1,700 x 24,000 that is ~326 MB per per-column `sl_stop`/`size`
   frame (arithmetic). Measured peaks: 3.86 GB (prototype, 1.2.0/Rust, scalar stop, 24k columns), 3.06 GB (D d5, 0.28.5),
   3.64 GB for the 408,000-column 17-fold stacked layout (D section 1.3). Rule: `engine.simulate` chunks the grid columns so the
   estimated peak stays under 5 GB, leaving headroom inside the 8 GB test cgroup (CLAUDE.md). vbt OSS has no chunking; it is a
   PRO feature (D section 6). Large grids run in-process and sequentially by default.
   **Fold parallelism is not needed for signals:** per-fold slice-and-select took 0.2 s for 17 folds (D d5). For the weight
   path, a single stacked call replaces the joblib per-combo pool (`wfo.py:388`).
7. **Prototype cost** (synthetic, 500 symbols x 1,700 bars, 48 combos, Rust): indicators 0.2 s, votes 0.7 s, `from_signals`
   6.6 s, total 16 s (`s1_ensemble_contract_proto_out.txt`). Today's comparable 2-fold run is ~254 s unprofiled (F P2).
   Most of that saving comes from the vectorised PIT mask and deduplication, not from Rust.

**Numbers that will shift** (re-baselining is accepted). Every one must be recorded against the pinned-window oracle from F Phase 0:

| Shift | Why | Size (where measured) |
|---|---|---|
| RSI values / first bar | vbt RSI is span-EWM. `window=27, ewm=True` equals Wilder 14 exactly, but its first valid bar is 27, not 14 | values 0.0 diff; 104 warm-up cells (F section D5) |
| MACD | vbt EWM is seeded at the first valid value | 0.12 early, ~1e-15 after 200 bars (F) |
| EMA | vbt pads `w-1` bars with NaN | warm-up only (D section 3) |
| BB / MSTD | must pass `ddof=1` | 1.2e-9 with it; 0.196 on the lower band in my probe without it (`s1_engine_selfcheck_out.txt`) |
| ATR stop | `vbt.ATR(ewm=False)`; the first true range is H-L | 2e-13; first bar 13 vs 14 (F) |
| Sortino | vbt uses a semi-deviation over all returns | 1.17 vs 1.01; **can change fold winners** via the 0.3 composite weight (F section b) |
| CAGR (`annualized`) | bars/252, not calendar days/365.25 | 15.51 vs 14.94 (F); this also moves the anchor's CAGR>4% filter |
| DSR | cross-column `var_sharpe`, fixed kurtosis | 0.6236 vs 0.6022 on D's sample |
| n_trades / expectancy | counted per window | 72/80 combos differ; NDH sign is analytically unchanged (F) |
| Live-path signals | all of the above, in warm-up and NaN edges | UNVERIFIED on production tape (F section e2) |
| pandas 3 side effects | bare `pct_change()` (44 sites), `stack`, datetime units | B section 6.2 |

The simulator itself does not shift: it was bit-identical across 0.28.5 and 1.2.0 on both numba and Rust (B section 3.1).

## 5. Engine configuration and the self-check

- **One explicit setting:** `GGT_VBT_ENGINE` in {`rust`, `numba`}, default `rust`. **Never `auto`**: B showed that `auto`
  silently changes engine when the wheel is installed or on a minor-version mismatch (B section 4.2-4.3).
  `engine.configure()` runs at import of `ggTrader.engine`. It sets `vbt.settings["engine"]`, sets
  `vbt.settings["returns"]["year_freq"] = "252 days"` (the default is `"365 days"`, probed), and turns the mismatch
  `UserWarning` into an error.
- **Pins:** `vectorbt[rust]==1.2.0`, which pins `vectorbt-rust==1.2.0` (B section 4.3). Also pin exact pandas, numpy and
  numba, and delete the dead `pandas-ta`, which blocks numpy>=2.4 (B section 6.1).
- **Startup self-check (prototyped, works):** `s1_engine_selfcheck.py` wraps
  `vectorbt_rust.portfolio.simulate_from_signals_rs`, runs a two-column trailing-stop `from_signals`, and raises if the counted
  engine differs from the pinned one. Output: `rust_calls: 1` for the Rust pin and 0 for numba. It takes milliseconds because
  Rust needs no JIT (B section 3.2). It runs in `ggt lab` / `ggt paper` startup and its result goes into run metadata.
- **Strict mode surfaces dtype traps.** In the prototype, `h.vbt.crossed_below(0)` (an int scalar) **raised** under
  `engine="rust"`: "requires float64 arrays and int64 cannot be safely cast". Under `auto` it would silently have run on numba.
  Strict `rust` is therefore also a lint for slow paths. The house rule is float thresholds (`0.0`, `30.0`).
- **Tests (via `scripts/run_tests.sh`):**
  1. `tests/engine/test_dispatch.py` runs the probe and fails unless the result is `rust`. It is skipped with a loud reason only if `GGT_VBT_ENGINE=numba`.
  2. `test_defaults_canary.py` snapshots `inspect.signature` defaults of every vbt API ggTrader calls (MA, MSTD, BBANDS, RSI, MACD, ATR, from_signals, from_orders), plus `settings["returns"]`, against a checked-in JSON. On an upgrade it lists each changed default for the agent to review.
  3. `test_golden_engine.py`: a fixed synthetic panel run through the Ensemble and Momentum specs, with value curves and per-fold winners compared to a stored artefact (1e-9 tolerance on values, exact on winners).
  4. `test_import_rules.py` (section 2).
- **Upgrade playbook for agents** (5 steps): bump the pin pair; run the canary and update the JSON with a note per default;
  run the dispatch and golden tests; run the pinned 17-fold oracle and diff the fold-table JSON; write a changelog entry with the
  version pair. Any winner change stops the upgrade for owner review.

## 6. CLI and persistence

- **`ggt lab --strategy`**: flags unchanged (`cli.py:57-98`). Add `--engine {rust,numba}` for debugging only. `--eval-end` keeps
  its drifting default, so citation rules still apply. Non-WFO "harness" runs become a one-fold WFO.
- **`--blend`**: unchanged; `allocation.py` stays. Note that `--max-leverage` still defaults to 2.0 (`cli.py:97`, `blend.py:30,74`)
  while live runs 1.0. This rewrite is the right moment to make 1.0 the default (owner call).
- **`ggt paper`**: no change while frozen. Live lane: before the migration merges, pin `docker-compose.yaml:3` from `:latest` to
  the current SHA tag, which the workflow already publishes (`docker-build.yml:35-37`, `type=sha`). Then a push to `main`
  cannot ship the new stack. Also tag `pre-vbt12` in git so the closed NO-GO numbers stay reproducible.
  **Cutover**, after the freeze lifts (`docs/next_steps.md:11-13`: "a calendar date alone does not lift the production freeze", so
  the ~2026-12-29 date is not a trigger): `paper.signal_runner` imports the new Ensemble and calls an engine `last_bar_signals()`.
  Run new-stack signals in dry-run beside live for a pre-registered number of sessions, record the daily XOR of entries/exits as
  JSON, and cut over only with owner sign-off. Signals are *expected* to differ (section 4 table), so this measures the difference
  and does not demand zero.
- **Tables:** all seven `lab_*` tables stay (`persist.py:29-91`). Add `engine_meta JSONB` to `lab_runs` and `lab_sweeps` via
  `ADD COLUMN IF NOT EXISTS`, holding the vbt/rust/pandas/numpy versions, engine, probe result and `metric_defs: "vbt-1.2"`.
  Old rows have NULL there, which marks them as old-engine numbers that must not be mixed with new ones. `lab_plans` is
  written only for strategies that still `select()`.

## 7. Size estimate (ESTIMATE; line counts from `wc -l` at HEAD)

| Area | Today | End state | Note |
|---|---|---|---|
| Engine path: simulate 304, sweep 350, metrics 56, harness 148, indicators 391, wfo 1,076, gates 215, kelly 178, strategy 60 | 2,778 | ~1,250 (engine ~400, wfo policy ~650, gates NDH ~120, strategy ~80) | WFO policy stays by design |
| Indicator strategies: signals 754, ensemble 489, ensemble_ic 205, ensemble_kelly 183, conviction 94, ic_weights 87 | 1,812 | ~650 | vote helper and specs |
| Other 26 strategy files | ~4,000 | ~3,300 | mostly port `to_targets` → `build`; alt-data logic stays |
| Rest of lab (data loaders, audits 4,890, tax, A23, screens, replica, ...) | ~10,000 | unchanged | no vbt equivalent |
| **lab total** | **18,591** | **~15,300** | about −3,300 / +1,400 lines → **net about −1,900 (−10%)** |
| tests/lab (18,986) | | about −3,000 indicator/simulate tests, +800 engine/golden/canary | |
| scripts/ (52 of 70 import `ggTrader.lab`) | | port only live drivers; archive the rest at `pre-vbt12` | |

To be blunt: "thinnest glue" shrinks the *engine path* by about half. It does not shrink ggTrader by half, because about 70% of `lab/` is data, audit, tax and study code that vbt cannot replace (F section c).

End-state tree (engine-relevant part):
```
src/ggTrader/engine/{__init__,config,selfcheck,grid,simulate,windows,stats}.py
src/ggTrader/lab/strategies/{__init__,registry,vote}.py + one file per strategy (spec + build)
src/ggTrader/lab/wfo/{folds,select,gates,breaker,anchor,run,report}.py
src/ggTrader/lab/universe.py   (vectorised PIT mask, F Phase 1a)
src/ggTrader/lab/{data,persist,allocation,blend,cli,tax_lots,...}.py  (unchanged)
src/ggTrader/paper/...         (unchanged until cutover)
```

## 8. Top risks and mitigations

| # | Risk | Mitigation in this design |
|---|---|---|
| 1 | The live trader silently gets the new stack via a `:latest` rebuild (this has happened twice, per CLAUDE.md) | Pin the compose file to the SHA tag before merging, and gate cutover on a dry-run XOR record and owner sign-off |
| 2 | pandas 3 silent changes (`pct_change`, units, `stack`, string dtype) corrupt data or timestamps, including the naive-UTC `ohlcv` hazard | Fix the 44 `pct_change` sites on 2.3.3 first (F Phase 0.4). Run a scratch-DB round-trip test of the Postgres write path (UNVERIFIED in B). Use the oracle diff. |
| 3 | `auto` / mismatch fallback leaves runs on numba (2.6x slower kernel on the new numba, B section 3.2) | Explicit engine, no `auto`, mismatch as error, dispatch probe at startup and in tests |
| 4 | Upstream default drift goes unnoticed (365-day year, ddof, ewm, `inf` Sharpe) | Explicit arguments everywhere, the defaults-canary test, the golden test, the non-finite guard |
| 5 | Changed statistics (Sortino, CAGR, DSR) move gates and winners, and old and new numbers get compared | `metric_defs` in `engine_meta`, a NULL marker on old rows, DSR/anchor thresholds re-registered before use, and NO-GO verdicts re-run rather than inherited. Honest NO-GOs stay recorded with their engine era. |
| 6 | Memory blowup on wide grids with per-column stop or size frames | The column chunker in `engine.simulate` with a 5 GB estimate, and the 8 GB cgroup kept |
| 7 | Lost reproducibility of published results | The `pre-vbt12` tag, old-stack oracle JSON kept beside the new-stack JSON, and versions in every artefact |
| 8 | A young Rust path (parity bugs fixed through 1.1.x, B section 4.5) | Simulator parity was measured bit-exact (B section 3.1). The golden test runs on both engines in CI, and `GGT_VBT_ENGINE=numba` is an escape hatch. |
| 9 | Commons Clause licence on vbt and vbt-rust | Internal research only. Re-read before ever offering a service (B section 4.1). |
| 10 | Agents hit the accessor API (`.vbt.*` cannot be found by grep, generated level names, misleading `group_by` errors, D section 5.4) | Strategies use the `engine.vbt` re-export; `grid.py` builds the level names; `group_by` uses positions, not names |

UNVERIFIED in this workstream: real-tape signal flips under the vbt indicators; a 17-fold real-data memory peak on 1.2.0;
Portfolio row-slicing support (the design avoids it and slices returns); Rust gains from threading (Rust releases the GIL, B section 4.1).
