# Architecture Guide

This guide explains how the ggTrader codebase is structured and how its components work together. For commands and usage see the [CLI Reference](cli_reference.md); for developer-specific guidelines, see [Developer Guidelines](../AGENTS.md).

---

## Technical Concepts & Terminology (Plain English)

Before diving into the code, here are the core concepts used throughout ggTrader:

- **Backtesting (Simulating)**: Running a trading strategy on historical price data to see how much money it would have made or lost in the past.
- **Walk-Forward Optimization (WFO)**: A realistic testing method. Instead of finding one rule that fits all of history, we split history into rolling time blocks. We use an earlier block (the **In-Sample** or train period) to find the best rules, and then test those rules on the immediate next block (the **Out-of-Sample** or test period). Then we shift forward and repeat.
- **Overfitting**: A common trap where a strategy is tuned so perfectly to past data that it looks amazing on paper, but fails completely on new, unseen data (because it "memorized" the past instead of finding a general pattern).
- **Walk-Forward Efficiency (WFE)**: A score that compares a strategy's performance during testing (out-of-sample) to its performance during training (in-sample). A high WFE suggests the strategy is robust; a low WFE suggests it is overfitted.
- **Vectorization**: Traditional simulators run day-by-day in a slow loop. ggTrader uses **vectorbt**, a library that performs backtesting calculations on entire arrays of data all at once using fast math. This allows you to simulate years of trading across hundreds of stocks in seconds.
- **Survivorship Bias**: Cheating in backtests by only testing on stocks that exist *today*. We prevent this by using a **Point-in-Time Universe**, which tracks exactly which companies were in the index on any specific day in the past (including those that later went bankrupt or were acquired).
- **OHLCV Data**: The basic building blocks of price charts—stands for **O**pen, **H**igh, **L**ow, **C**lose, and **V**olume for a given time period.

---

## ggTrader at a Glance

| Component | What it does in plain English |
|---|---|
| **Core Libraries** | **Python 3.12** (language) · **vectorbt 1.2.0 + vectorbt-rust** on pandas 3 / numpy 2.5 (fast calculations; exact versions in `uv.lock`) · **TimescaleDB** (database) · **yfinance** (free Yahoo Finance data). |
| **Data Sources** | Fetches live stock data from Yahoo Finance, uses Tiingo as a backup, loads crypto data from the database, and tracks S&P 500 constituents over time in a versioned membership list (database schema `universe`, read through one reader; see "Membership database and the single reader"). |
| **Lab Output** | `ggt lab --wfo` writes a JSON record with a provenance block; `--blend` also persists a `blend:` run to `lab_runs` and the per-fold and per-sweep tables (`lab_plans`, `lab_returns`, `lab_equity`, `lab_summary`, `lab_sweeps`, `lab_sweep_combos`). |
| **Paper Trading** | Virtual trading on Alpaca. Includes a machine learning guard (ML feature gate) to filter bad trades, risk checks (guardrails), and Telegram notifications. |
| **Supported Markets** | US stock indices (S&P 500, Nasdaq-100, Russell 2000) and customizable cryptocurrency pairs. |
| **Vectorization** | All portfolio calculations run instantly using NumPy and Pandas vector operations. |

---

## Directory & File Structure

Here is where the source code lives inside `src/ggTrader/`:

```
src/ggTrader/
├── engine/                       # The ONLY package that imports vectorbt (adapter, see "Engine adapter")
│   ├── config.py                # Engine settings: rust/numba, 252-day year, daily freq, epoch/contract
│   ├── selfcheck.py             # Dispatch probe: proves the configured engine really runs
│   ├── indicators.py            # Presets with every flag explicit: wilder_rsi, bbands, ema, sma, macd, atr
│   ├── signals.py               # cross_above/cross_below (float coercion), time_stop
│   ├── portfolio.py             # run_signals / run_weights -> SimResult, column chunking
│   ├── stats.py                 # Window Sharpe/vol/drawdown, finite guard, per-window trades, DSR
│   ├── windows.py               # Calendar folds -> bar positions, range_split wrapper
│   ├── provenance.py            # provenance() stamp and compare() across epochs
│   └── _compat.py               # Version shims (empty at the start of epoch E1)
│
├── lab/                          # The Research & Simulation Engine
│   ├── cli.py                   # Command-line entry point for running simulations (ggt lab)
│   ├── data.py                  # Loads stock lists and historical prices (yfinance, TimescaleDB)
│   ├── persist.py               # Saves blend results to database tables
│   ├── strategy.py              # LabConfig (shared research settings)
│   ├── universe.py              # Vectorised point-in-time universe mask and optional exit policies
│   ├── blend.py                 # Walk-forward several sleeves and blend their out-of-sample curves
│   ├── wfo/                     # The walk-forward on the vectorbt engine (`ggt lab --wfo|--blend`), see section 3a
│   │   ├── folds.py             # generate_folds + calendar windows -> bar positions
│   │   ├── select.py            # composite score, winner pick, durable live-parameter pick
│   │   ├── gates.py             # NDH lattice glue (lab.gates.ndh_check) and the DSR gate
│   │   ├── anchor.py            # defensive anchor fit only on data to each fold's train_end
│   │   ├── breaker.py           # circuit breaker, 2-of-3 shadow re-entry, WFE, WfoPolicy
│   │   ├── scorers.py           # SignalScorer (one full-history simulation, sliced) / WeightScorer (fresh-start stacked windows)
│   │   ├── run.py / report.py   # run_wfo, WfoResult, table, curve_summary, JSON payload
│   │   └── inputs.py / cli_support.py  # Bars input, PIT masks, `ggt lab` glue
│   ├── gates.py                 # NDH plateau check (the DSR gate is engine.stats.dsr)
│   └── strategies/              # Strategies on the engine contract; any module dropped in here is found
│       ├── vote.py              # N-of-M vote over precomputed voter arrays (engine contract)
│       ├── contract.py          # ParamGrid (+ validity rules), SignalOut, tiling, weight stacking
│       ├── voters.py            # bb / rsi / ema / macd / vbb voter arrays from engine presets
│       ├── ensemble.py          # Ensemble (48-combo WFO grid + live-defaults control config)
│       ├── voter_strategies.py  # ema_cross, bb_reversion, rsi_reversion, macd_divergence, volume_bb_reversion
│       ├── conviction_bb.py, xs_momentum.py, cross_asset_sleeve.py, retail_attention.py
│       ├── leak.py              # truncation / perturbation lookahead checks
│       └── registry.py          # REGISTRY built by discover() (scans the package) and the 28 legacy-only names
│
├── paper/                       # The Live Paper-Trading Deployment
│   ├── alpaca_broker.py         # Adapter to talk to Alpaca (our paper trading broker)
│   ├── signal_runner.py         # Engine self-check, live bars, v2 ensemble signals, sanity gates (fail closed)
│   ├── live_bars.py             # Live loader frame -> engine Bars (the partial 12:45 bar)
│   ├── live_signals.py          # The one place the live ensemble frames are built (runner + overlay)
│   ├── gates.py                 # Input/output sanity gates and the 12:55 PT late-run cutoff
│   ├── untradable.py            # UNTRADABLE_SYMBOLS, broker status and the "asset is not active" 422
│   ├── overlay.py               # Blend sleeve curves through engine.run_signals
│   ├── trader.py                # Coordinates signals, risk checks, and submits orders
│   ├── risk.py                  # Safety checks (caps position sizes, halts on losses)
│   ├── notifier.py              # Sends trade updates to Telegram
│   └── persist.py               # Saves daily paper trading logs and portfolio balances
│
├── data/                        # Data Downloader & Database Loader
│   ├── bars.py                  # Read-only research bars (per-field frames) from the snapshot or DB
│   ├── snapshot.py              # Reads and hash-verifies the frozen Parquet tape (MANIFEST.json)
│   ├── core/
│   │   ├── base_loader.py       # Basic instructions for downloading data
│   │   └── stock_constants.py   # Yahoo Finance parameters and S&P 500 constants
│   ├── historical/
│   │   └── timescaledb_loader.py # Fetches historical prices from your local database
│   └── live/
│       └── yfinance_loader.py    # Downloads live stock data and saves it to a cache
│
├── utils/                       # Common Helpers
│   ├── config.py                # Global settings and defaults
│   ├── paths.py                 # Locates files relative to the project root directory
│   ├── db_engine.py             # Manages connections to the database
│   ├── io.py / results.py       # Atomic file writes; result JSON stamped with provenance
│   ├── manifest.py / binding.py # Hash records for frozen inputs; bind a result to its git commit and lock hash
│   ├── significance.py          # Newey-West excess-return statistics (shared by studies)
│   ├── fred_data.py             # FRED series loader (shared by studies)
│   └── study.py                 # Study plumbing: find the repo root, put a study's code on sys.path
│
└── cli/                         # Command-Line Subcommands
    ├── main.py                  # Root CLI setup (the main 'ggt' command)
    └── cmd_db.py                # The 'ggt db' command to manage the database
```

**Where study code lives.** `src/` holds the engine, the walk-forward lab, the data layer and the live paper path, and nothing that belongs to one study. A study's own modules, drivers and tests live inside its folder under `research/`, in a `code/` subfolder (tests in `code/tests/`), with helpers that several studies of one category share in that category's `_lib/` folder. Since the study-isolation refactor (2026-11-07) `lab/` no longer holds study modules: the stock-signal screens, the WFO attribution and synthetic-policy tools, the tax-lot library, the century-leverage study, the production-replica tools, the event-timing audit and the intraday family all moved out (every moved file is listed in `research/CODEMOVED.csv`). The shared statistics helpers (`utils/significance.py`, `utils/fred_data.py`) stayed in `src/` under `utils/`. Nothing in `src/` imports a study; a study imports `ggTrader` and calls `ggTrader.utils.study.bootstrap(__file__)` to put its own `code/` folder and its category's `_lib/` on the path. `scripts/study.py` creates and checks a study (see the [CLI Reference](cli_reference.md)).

---

## Research data layer and point-in-time universe (vectorbt engine rebuild, P3)

Research reads daily bars only through `ggTrader.data.bars.load_bars()`. It is read-only and
never touches the network: the default source is the frozen Parquet tape in `data/snapshots/`,
which `ggTrader.data.snapshot` verifies against `MANIFEST.json` (sha256, row count, date range)
and refuses to use on any mismatch; the `db` source opens a server-enforced read-only session
(`default_transaction_read_only=on`) and runs one SELECT. It returns per-field frames
(`close`, `open`, `high`, `low`, `volume`: dates x symbols, UTC-midnight index), the requested
symbols that have no bars (`missing_symbols`) and each symbol's last valid bar. Duplicate rows,
non-midnight stamps and weekend stamps raise `BarsValidationError`; this assertion replaces the
silent `collapse_daily_duplicates` heuristic (which `lab.data.load_ohlcv` still applies for the replica tools in `research/audits/production-replica/code/`).
`Bars.legacy_frame()` gives the old `(symbol, field)` MultiIndex frame (used by the replica panel and the ML-gate trainer). `vbt.Data` is deliberately not used (it is a dict of frames in the open-source build).

`ggTrader.lab.universe` builds the point-in-time universe once per run instead of once per bar:
`pit_mask(close, ...)` equals `lab.data.eligible_at` on every date (member on that date,
enough non-NaN closes, alphabetical `max_stocks` cap) in about 0.05 s and 1.2 MB for the full
window. `membership_mask` is the index-membership part alone. Two optional exit policies are off
by default because they change numbers: `removal_exits` (exit when a symbol leaves the index)
and `terminal_exits` (exit at a dead ticker's last valid bar; optimistic, see its docstring).
The mask builder warns (policy `on_stale`) when the constituent list looks stale: with a membership source, when its last verification is more than 45 days before the run's as-of date (information for research results; the live trader's gate is separate); with a bare history table, when its last snapshot is that old. The list itself comes from the membership reader described next.

---

## Membership database and the single reader

The S&P 500 membership list is data that changes, so it lives in PostgreSQL schema `universe`
instead of in a file, and every refresh is kept as a new **vintage**. Design and decisions:
`docs/superpowers/specs/2026-11-07-membership-database-design.md` (Appendix C wins where it
differs from the earlier sections). Commands and flags: [CLI Reference](cli_reference.md).

**Components**

- **Schema** (`sql/universe_schema.sql`, applied by `ggt db membership apply-ddl`, safe to repeat).
  `loads` (one row per vintage, with source, parent, hashes and the normalizer version),
  `load_blobs` (the raw bytes of each load), `snapshot_dates` and `membership` (the index
  intervals), `load_evidence` (events file and saved press-release pages of a curated vintage),
  `symbol_changes` (company continuity across a ticker change; membership itself moves by remove
  plus add), `checks` and `check_blobs` (every comparison against an outside list, with the
  fetched bytes) and `source_snapshots` (raw bytes the weekly job archived). The tables are
  append-only, enforced by triggers and role privileges; the one permitted update marks a replaced row `superseded_by_load_id`. Three vintages exist: 1 the frozen file
  import, 2 the upstream download (coverage to 2026-08-18), 3 a curated gap-fill (to 2026-11-04).
- **The one reader** (`ggTrader.data.membership`). `MembershipSource` has two backends: a
  database backend (`db` for the latest vintage, `db:<load_id>` for a pinned one, over a
  read-only session) and a fixture backend (`fixture:<name>`, a frozen file listed in
  `data/universe/membership/MANIFEST.json` and hash-checked). The source is always named, by
  `--membership` or the `GGT_MEMBERSHIP` environment variable; there is no default and no
  fallback to a file, and an unnamed source raises `MembershipSourceNotSelected`. The old file
  `data/universe/sp500_constituents_history.csv.gz` is the immutable vintage-1 fixture
  `sp500_v1_2026-06-02`.
- **Loader** (`ggt db load-sp500`). Validates a membership file (`--from-file`) or applies a
  human-confirmed events file (`--from-events`, which writes a `curated` vintage) and stores it
  as the next vintage. A dry run unless `--commit`. `membership reinstate` writes a new vintage
  with the content of an older one; `membership export` dumps the whole schema with `pg_dump`;
  `scripts/membership_restore_drill.py` restores such a dump into a scratch database and
  compares it with the source.
- **Cross-checks** (`membership_holdings.py`, `membership_upstream.py`). `check-holdings`
  matches the latest vintage against the SPDR SPY fund holdings for the holdings' own as-of
  date and records `matched`, `mismatch` or `fetch_failed` in `checks`. `check-upstream`
  compares the public upstream list and is informational only.
- **Weekly job** (`membership_weekly.py`, `membership_sources.py`, `membership_proposal.py`;
  `ggt db membership weekly`). Fetches and archives the sources, matches SPY, compares the other
  sources, applies the refuse conditions, and on a difference writes a proposed events file and
  one alert message. It writes only `checks`, `check_blobs` and `source_snapshots`, and never
  loads a vintage: a human confirms the effective dates against the S&P DJI press release and
  runs `load-sp500 --from-events FILE --commit`. Systemd unit templates are under
  `deploy/membership_weekly/`; they are not installed.

**Data flow**

```
public sources (SPY holdings, Wikipedia, chinobing CSVs)
        |  weekly job, check-holdings
        v
 universe.checks / check_blobs / source_snapshots   (comparisons and archived bytes; never the list)

 human, with S&P DJI press releases
        |  load-sp500 --from-events FILE --commit
        v
 universe.loads + membership   (a new vintage)
        |
 MembershipSource  (db, db:<id> or fixture:<name>, always named)
        |-- research: ggt lab, drivers, load_bars, pit_mask (any named vintage)
        '-- live trader: membership_gate (latest "db" vintage; also reads universe.checks)
```

**The live gate** (`ggTrader.data.membership_gate`). Before any bar is fetched, a wall-clock
live run reads the latest database vintage and refuses to trade (`SignalGateError`, no orders)
when the list is unreadable, empty, never matched against SPY, last matched longer ago than the
freshness limit, too far behind the newest attested date, dated in the future, or when the
newest holdings comparison of that vintage is a mismatch. A match belongs to one vintage, so
loading a newer vintage starts again from "never matched" until `check-holdings` matches it.
A live run (`--live`) refuses any membership source but the latest database vintage. A wall-clock
dry run on the latest `db` vintage applies the same gate; only fixtures, pinned vintages and
replays with an injected `now` skip it. Limits, override variables and error codes are in
the CLI Reference (the "Membership gate" paragraph under the `ggt db membership` commands).

---

## Strategy contract on the engine (vectorbt engine rebuild, P4)

Ten strategies were ported to the new contract in `ggTrader.lab.strategies`; `REGISTRY` (built by `registry.discover()`, see `docs/adding_a_strategy.md`) is the
only registry. The legacy `select()/to_targets()` classes, `STRATEGY_REGISTRY` and the legacy
plumbing (`lab/simulate.py`, `sweep.py`, `harness.py`, `wfo.py`, `metrics.py`) were deleted in P5c.
A signal strategy declares a `ParamGrid` (values plus validity rules as data, e.g.
`ema_fast < ema_slow`; `grid.combos()` equals the legacy `build_grid`, frozen in a test golden) and a
`build(bars, grid) -> SignalOut` that composes only engine presets and crossovers and returns
entry/exit frames with columns `(swept grid levels..., symbol)`. It never calls `Portfolio` and
never applies the PIT mask: the caller tiles the mask onto the grid columns and the engine ANDs it
into the entries before the time stop. `lab/strategies/vote.py` is the one piece of voting glue
(vectorbt has no vote primitive): each voter is computed once per distinct parameter tuple, a
combination only sums the right boolean arrays and thresholds them (`min_agree`,
`min_agree_exit`, RSI exiting on its own). A weight strategy's `build(bars, grid, rebal,
eligible)` returns target weights on the first bar after each decision date (NaN = hold).
`Ensemble.grid` is the 48-combination WFO grid (`divergence_window=10`, `vol_mult=1.5`);
`Ensemble.live` holds the deployed defaults (20 and 2.0) that are *not* in that grid, and
`Ensemble.live_grid()` is the one-combination control grid (spec OD14). The 28 other registry
names (`LEGACY_ONLY_STRATEGIES`, for example `mtf_reversion`) are closed NO-GO or untested research;
their code is not in the main tree and they run at tag `legacy-engine-vbt0285` (`scripts/legacy_env.sh`).

---

## Strategy Classification

Strategies in ggTrader belong to one of two categories:

### 1. Weight-Based Strategies (`target_kind="weights"`)
These strategies return a target percentage for each asset. For example, "allocate 5% of money to Apple, and 10% to Tesla." The simulator automatically generates buy/sell orders to match these weights at every rebalance.

| Strategy | Plain English Description |
|---|---|
| `xs_momentum` | **Cross-Sectional Momentum:** Ranks stocks by their past 12-month returns and divides portfolio money equally among the top N stocks. |
| `dual_momentum` | **Dual Momentum:** Same as above, but if the overall market trend is negative, it drops underperforming stocks and moves the money into cash for safety. |

### 2. Signal-Based Strategies (`target_kind="signals"`)
These strategies return binary flags: `True` for buy (entry) and `True` for sell (exit). The simulator executes trades based on these flags, allocating a fixed position size (default is 2% of cash) to each trade.

| Strategy | Plain English Description |
|---|---|
| `ema_cross` | **EMA Crossover:** Buys when a fast average price crosses above a slow average price (uptrend), and exits when it crosses below. |
| `wfo_tournament` | **Tournament:** Automatically selects the best performing EMA configuration from a list of combinations on recent past data. |
| `bb_reversion` | **Bollinger Band Reversion:** Buys when the price dips below the lower Bollinger Band (cheap) and sells when it returns to the middle average band. |
| `rsi_reversion` | **RSI Reversion:** Buys when the Relative Strength Index falls below 30 (oversold) and sells when it rebounds. |
| `macd_divergence` | **MACD Divergence:** Buys when price is falling but selling momentum is weakening (divergence). |
| `volume_bb_reversion` | **Volume Bollinger Band:** Reversion buy confirmed by a massive surge in volume (high activity at the bottom). |
| `mtf_reversion` | **Multi-Timeframe Reversion:** Buys only when the asset is oversold on both weekly (long-term) and daily (short-term) charts. |
| `ensemble` | **Voting Ensemble:** Combines multiple strategies and enters a trade only if a majority of them agree. |
| `ensemble_conviction` | **Conviction Voting:** Voting strategy that buys larger position sizes when more indicators agree. |
| `conviction_bb` | **Conviction Bollinger Band:** Buys larger positions the further the price plunges below the lower band. |

---

## Code Interfaces (Developer Reference)

All strategies must implement the `Strategy` protocol defined in [strategy.py](file:///home/flynn/ggTrader/src/ggTrader/lab/strategy.py):

```python
class Strategy(Protocol):
    name: str
    target_kind: str  # Must be "weights" or "signals"

    def select(self, asof, data, eligible) -> Plan:
        """
        Point-in-time selection.
        Must use only data available on or before the 'asof' date (no future peeking).
        """

    def to_targets(self, plans, data) -> DataFrame | SignalTargets:
        """
        Converts the plans generated during rebalancing into a target matrix
        (either weights or buy/sell signal matrices).
        """
```

Signal-based strategies also support parameter sweeps and walk-forward parameter updates:

```python
    def sweep_params(self) -> Dict[str, List[Any]]:
        """Returns the grid of settings to test (e.g. fast/slow EMA lengths)."""

    def sweep_signals(self, ohlcv, symbols) -> Tuple[entries, exits]:
        """Generates entry and exit signals for a specific parameter combination."""
```

---

## Core Workflows and Data Flows

### 1. Lab Walk-Forward Run (`ggt lab --wfo`)

```
User runs command: ggt lab --strategy ensemble --wfo --eval-start 2021-01-31 --eval-end 2026-04-30
    ↓
1. registry check               -> The strategy must be one of the ten ported ones; the 28 legacy-only names exit with a pointer to the legacy tag.
    ↓
2. load_bars()                  -> Frozen tape, read-only. equity_universe_between() + lab.universe.pit_mask() -> daily point-in-time membership (no survivorship bias).
    ↓
3. generate_folds()             -> Divides history into rolling folds (12 months training / 3 months testing).
    ↓
4. wfo.run_wfo()                -> One grouped Rust simulation per configuration, then per fold (see 3a).
    ↓
5. Prints the fold table and OOS summary and writes the JSON record with its provenance block.
   (A `--blend` run additionally blends the sleeves and writes a `blend:` row to `lab_runs`.)
```

### 2. Walk-Forward Optimization (what each fold does)

```
1. generate_folds()             -> Divides history into rolling folds (e.g., 12 months training / 3 months testing).
    ↓
2. For each Fold:
   - Scores every parameter combination of the strategy's grid on the 12-month training window.
   - Finds the parameter set that performed best.
   - Uses those parameters to trade during the 3-month test window.
   - Calculates Walk-Forward Efficiency (WFE) = Test Return / Train Return.
   - Circuit Breaker: Stops the test early if WFE drops below safety limits (indicating the strategy is failing).
    ↓
3. Calculates "Anchor Parameters" (settings that were stable across most folds).
    ↓
4. Prints the final test results, WFE, and recommended stable parameters.
```

### 3a. How the engine produces the numbers (`ggt lab --wfo`)

The folds, selection, gates, anchor, circuit breaker and table described above, and how the numbers
are produced (`lab/wfo/`).

```
1. load_bars()                  -> frozen tape, read-only. lab.universe.pit_mask() -> daily eligibility.
2. strategy.build(bars, grid)   -> entries/exits for EVERY grid combination (indicators run once per distinct parameter).
3. engine.run_signals()         -> ONE grouped Rust simulation over the whole history (one cash pool per
                                   combination, mask tiled onto the parameter columns, chunked under a memory budget).
4. For each fold the window statistics are SLICES of that one simulation (engine.stats.window_stats):
   anchor = first bar .. train_end, train = the 12-month window, test = the deployed combination's 3-month window.
   Causality makes this identical to simulating each window on data truncated at its end (tested bit for bit).
   Weight strategies cannot be sliced (each window starts flat with fresh cash): they run as stacked windows,
   one engine.run_weights call per window kind; only the anchor, which always starts at the first bar, is sliced.
5. Gates, anchor, breaker and OOS stitching are the unchanged ggTrader policy code.
```

The deleted legacy path ran three grouped simulations per fold plus one for the live pick (52 for 17
folds); the engine path runs one (two memory chunks on the full SP500 ensemble grid).

### 4. Paper Trading Cycle (`ggt paper`)

> The trader is **stopped by the owner** (2026-11-06) and its cron lines are commented out.
> **Cron stays disabled until the owner re-enables it** (spec OD18); the signal path below was
> rebuilt in P7 and has not run live.

```
ggt paper [--live]   (cron, when the owner re-enables it: scripts/paper_trade.sh, 12:45 PT)
    ↓
1. signal_runner.require_rust_engine() -> engine.selfcheck before any data: Rust dispatched,
                                          vectorbt/vectorbt-rust paired; otherwise no orders.
    ↓
2. loader chain (lab.data.fetch_stock_ohlcv, the 12:45 partial bar) -> paper.live_bars.Bars
                                       -> paper.gates.vet_bars (empty / duplicates / weekend /
                                          universe / history / last-bar coverage; a not-yet-complete
                                          bar dated today is dropped, anything else thin is a bad tape)
    ↓
3. paper.live_signals.live_ensemble_frames -> the v2 Ensemble.build() with the live defaults
                                       (shared with overlay.compute_sleeve_curve, which simulates
                                        through engine.run_signals); buys / sells / last_exit off the
                                        last bar; paper.gates.vet_signal_counts
    ↓
4. trader: late-run gate (after 12:55 PT no orders), freshness gate, untradable resolution
                                       (UNTRADABLE_SYMBOLS + broker status + 422 "not active"),
                                       risk guardrails, MAX_POSITIONS cap (alphabetical)
    ↓
5. trader.execute()                 -> Alpaca orders (--live only; a dry run is read-only)
    ↓
6. persist.save()                   -> ledger, snapshot, peak (--live only)
    ↓
7. notifier.send()                  -> Telegram (a read-only dry run logs instead)
```

`paper/` reaches vectorbt only through `ggTrader.engine` (`tests/engine/test_boundary.py` scans every file under `src/`; `tests/paper/test_paper_boundary.py` checks the signal path).

---

## Engine adapter (`ggTrader.engine`, vectorbt engine rebuild, P2)

`src/ggTrader/engine/` is the only code that imports `vectorbt` or `vectorbt_rust` (spec
`docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md` sections 5.1-5.9). An upgrade is one pin
change, a test run and a decision (`.claude/skills/vbt-upgrade/SKILL.md`).

**Boundary.** Ruff `TID251` bans the imports everywhere except `engine/` and `tests/engine/`
(`ruff.toml`), and `tests/engine/test_boundary.py` walks `src/`, `scripts/`, `tests/` and `ggt.py` with the
AST: imports at any depth, dynamic import strings, the global `.vbt` accessor (allowed only in `engine/` and
`lab/strategies/`), raw indicator constructors, `Portfolio` entry points and the private `vectorbt._engine`
(only `selfcheck.py`). The allow-list (`LEGACY_ONLY`, mirrored in `ruff.toml`) is down to the one documented exception,
`scripts/legacy_reference_runs.py`, which runs under the legacy 0.28.5 interpreter against the tree
extracted from tag `legacy-engine-vbt0285`; the test fails if that entry goes stale or the two lists differ.

**Configuration (`config.py`).** `GGT_VBT_ENGINE` is `rust` (default) or `numba`; `auto` is rejected because it
silently falls back to numba. Importing the package sets `vbt.settings["engine"]`, `returns.year_freq = "252
days"` (vectorbt's default is 365, +20% Sharpe) and the daily `freq`, and turns the vectorbt/vectorbt-rust
version-mismatch warning into an error. `selfcheck.py` (`python -m ggTrader.engine.selfcheck`) wraps the Rust
simulator, runs a two-column trailing-stop `from_signals` and fails unless the Rust call count matches the
configured engine.

**Presets (`indicators.py`).** vectorbt's defaults are a different strategy, so each preset writes every flag:
`wilder_rsi(n)` = `RSI(window=2n-1, ewm=True)`, `bbands` = `ddof=1`, `macd` = `macd_ewm=signal_ewm=True`,
`atr` = `ewm=False`, `ema` = `MA(ewm=True)`. On the frozen tape they equal the legacy pandas indicators to
0.0 (RSI, EMA, MACD line), 1e-12 (Bollinger) and 2e-14 (ATR) after each symbol's burn-in
(`research/programme/vectorbt-engine/p2-evidence/`). `signals.py` coerces thresholds and booleans to float64
because the Rust engine rejects ints and bools.

**Simulation and statistics.** `portfolio.run_signals` and `run_weights` make one grouped, cash-shared
`from_signals` / `from_orders` call per memory chunk (columns split under a 5 GiB estimate, groups never
split; each finished chunk is copied out and released, so the budget bounds the peak) and return a frozen
`SimResult` (value, returns, trade records). Equity curves were bit-identical to the deleted `lab/simulate.py` (its outputs are frozen in `tests/engine/golden/legacy_simulate.npz`); the legacy
`vol_target`/`vol_lookback`/`vol_cap` keys raise instead of being ignored. `stats.py` takes window statistics
from returns via vectorbt's accessor at 252 bars and ddof=1, scores a window on the legacy convention
(`window_stats(returns, a, b)` equals `curve_stats(equity.iloc[a:b])`: returns from the second bar, drawdown
from the base bar), turns flat-curve and no-downside `inf` into NaN, counts trades by entry bar inside a
window (OD12) and provides the deflated Sharpe ratio configured to reproduce `gates.dsr_check`
(`var_sharpe = 252/(T-1)`, `bias=False`). Sortino and CAGR are vectorbt's own (owner decision 2026-11-06:
where a ggTrader definition differs, vectorbt's wins); the one-off diagnostic that attributed the change is
recorded in `research/programme/vectorbt-engine/P5-completion.md`. The 252-day year and the flat-curve guard
are a calendar parameter and a safety guard, not definition overrides.

**Provenance.** `provenance.provenance()` stamps vectorbt, vectorbt-rust, pandas, numpy, numba, the engine
requested and the engine that actually ran, `uv.lock` sha256, git sha, `contract_version` and
`baseline_epoch`. `provenance.compare()` classifies two blocks and refuses to compare across baseline epochs
(E1 = the first baseline on vectorbt's definitions, E2 after a later definition change), a different tape sha, or different
per-statistic `definitions`. `utils/results.write_result_json` stamps every
result file; `lab/persist.py` gains a nullable `provenance JSONB` on `lab_runs` and `lab_sweeps`
(`ADD COLUMN IF NOT EXISTS`, NULL means a legacy-engine result).

**Licence.** vectorbt and vectorbt-rust are Apache 2.0 with the Commons Clause: private research and a personal
paper trader are unaffected; ggTrader must not be offered as a paid service without re-reading the licence
(spec OD4, owner to record the exception).

**Tests (`tests/engine/`).** L0 boundary, lock consistency, upgrade tooling; L1 contract tests against
independent numpy references (rtol 1e-9) and the legacy formulas; L1b a defaults tripwire (a clean interpreter
dumps vectorbt's defaults, compared with `golden/vbt_defaults.json`); L2 seeded golden panels on both
engines; L3 dispatch probe, numba-vs-Rust parity and a `-W error` lane. Goldens are rewritten only with
`--update-golden --golden-reason "<why>"` and the reason must appear in `docs/changelog.md`.

**Test tiers.** `scripts/run_tests.sh` runs the fast core (Rust engine only; no `slow`, `numba`, `tripwire`, `study`, `tooling` or `integration` test). The merge, deploy and vectorbt-upgrade gate is the FULL run, `scripts/run_tests.sh --full`; the tiers are defined in `tests/tiers.py`.

---

## Walk-Forward Fold Logic Details

The walk-forward ([wfo/](../src/ggTrader/lab/wfo/)) splits historical data systematically:

1. **Universe Eligibility**: the point-in-time mask (`lab/universe.py`) keeps a stock out of the entries until it is an index member with enough price history.
2. **Rolling Folds**: 12 months of training and 3 months of testing, sliding by 3 months (`generate_folds`).
3. **Warmup Period**: data is loaded from days before the first fold so indicators (like moving averages) have enough history to calculate values on day one.
4. **Aggregation**: the out-of-sample windows are stitched into one curve and summarised against SPY on the same days.

---

## Technical Indicators

There are no hand-written indicator functions any more. Strategies compose the presets in
[engine/indicators.py](../src/ggTrader/engine/indicators.py) (`wilder_rsi`, `bbands`, `ema`, `sma`, `macd`,
`atr`) and the crossovers in [engine/signals.py](../src/ggTrader/engine/signals.py), which call vectorbt with
every behaviour flag written out. The voter arrays are built in `lab/strategies/voters.py`. The old pandas
formulas live on only as test oracles (`tests/oracles/legacy_voters.py`).

---

*Back to [README.md](../README.md).*

### Event-timing audit evidence path

`event_timing_audit.py` (in `research/audits/event-timing/code/`) owns PIT denominator accounting, clocks, requirement
matrices and reporting; `event_timing_sources.py` owns bounded immutable raw
collection, and `event_timing_reconstruction.py` owns public-time Form 4 replay.
The approved 2026-10-22 amendment adds a separate `insider_filing_date_only`
assessment through the same clock, population and replay functions, with separate
preparation/replay/prefix/population progress and matrix/disposition output. Its
raw API/header filing dates must match for the issuer/accession; execution is the
first NYSE close strictly after that date, and observed availability stays null.
Original insider output and PEAD retain their existing behavior. Timing PASS
means declared convention compliance; dissemination remains UNVERIFIED. Progress
and result metadata bind the amendment identity/hash alongside code/protocol
bindings; legacy or changed-policy registrations cannot resume.
The standalone driver (`code/run.py` in the same folder) delegates to one CLI. This path uses explicit read-only
SELECTs for metadata/bar existence, never schema loaders or execution code.
Offline evidence and fixture tests precede collection/code review. Details and
remaining input requirements are in the
[implementation handoff](../research/audits/event-timing/event-timing-audit-implementation.md).

## Intraday metadata preflight

`intraday_probe.py` (in `research/intraday/_lib/`) is an isolated research utility, invoked through
`intraday_coverage_probe.py` in `research/intraday/spy-momentum-feasibility/code/`. It checks a frozen SPY calendar/sample
manifest against Alpaca SIP and IEX bars and SIP auction metadata. It uses only
read-only HTTP requests and atomic JSON checkpoints, with no database,
strategy registry, simulation or paper-trader integration. Prices are discarded
before persistence. See the [Stage 0A contract](../research/intraday/spy-momentum-feasibility/2026-10-25-intraday-stage0a-contract.md)
for request bounds and the limits of its sampled-coverage conclusion.

## Intraday r8 research pipeline

`intraday_momentum.py` (in `research/intraday/spy-momentum-feasibility/code/`) owns the frozen protocol identity, dataset validation,
pure research calculations and write-once evidence. `intraday_momentum_data.py`
collects complete, hashed, immutable request chunks and atomically checkpoints
progress for resumption. `run.py` in the same folder exposes standalone
`plan`, `collect`, `validate` and `score` commands. It has no strategy-registry,
vectorbt, database or production-paper integration.

Validation checks every required endpoint independently of the signal and binds
its PASS artifact to actual protocol, code, calendar and data hashes. Scoring
rehashes/revalidates those identities and claims the collection directory before
calculating returns. The original metadata-only Stage 0A code/artifact stays
separate. See the [r8 protocol](../research/intraday/spy-momentum-feasibility/2026-10-25-intraday-feasibility-r8.md)
and [CLI reference](cli_reference.md) for the current authorization boundary.


## Prospective intraday observer

`intraday_observer.py` (in `research/intraday/spy-momentum-prospective-observer/code/`) owns an isolated file-based observer, hypothetical
accounting, calendar slots and append-only source/decision evidence.
`intraday_observe.py` in the same folder exposes manual commands; it is not a `ggt paper`
subcommand. Safe helpers are reused from `intraday_probe.py`; the historical r8
scorer and production broker path are not imported or modified.

Each state directory contains an immutable manifest, raw source payloads,
hash-chained records, exclusive stage claims and a shared per-date GET budget.
Offline rehearsal uses a synthetic transport and simulated clock. No scheduling
or real data collection is enabled by installation/import. Timely stages require a durable publication-completion record; interrupted writes
cannot become completed decisions. Activation requires a manifest-bound readiness
artifact. Real-source sample access and the documented fifteen-minute restriction
have been checked; future availability and actual intraday publication latency
are not guaranteed. The calendar-aware `intraday_observer_job.py` wrapper (same folder)
dispatches the same core stages; standalone user timers are defined in
`deploy/intraday_observer/` and were separately installed on October 27, 2026. The deployed runtime is a
separate read-only snapshot cut at commit `280d4a0`, which still has the pre-move `scripts/` layout; only fetching units receive a private persistent
credential file under the owner-approved activation. Corrupt evidence yields unresolved diagnostics
without statistics, and cannot be waived by resume.
See the [protocol](../research/intraday/spy-momentum-prospective-observer/2026-10-25-intraday-prospective-observation.md)
and [CLI reference](cli_reference.md#prospective-intraday-practice-journal).
