# Architecture

How ggTrader is structured. For commands and usage see [CLI Reference](cli_reference.md); for codebase guidelines see [agents.md](../agents.md).

---

## At a glance

ggTrader is now a **research-first lab** — a vectorbt-based walk-forward optimization framework for backtesting trading strategies on historical data. The codebase contains only the research engine; live execution, live trading, and monthly recalibration have been removed as of 2026-06-16.

| Aspect | Detail |
|---|---|
| **Core Stack** | Python 3.10+ · vectorbt · TimescaleDB · yfinance |
| **Data Sources** | TimescaleDB (crypto OHLCV) · yfinance (equities) · CSV (SP500 constituents) |
| **Lab Output** | TimescaleDB `lab_runs` / `lab_periods` tables (timestamped, immutable) |
| **Supported Markets** | US equities (SP500 constituents) · Crypto (customizable universe) |
| **Evaluation Window** | Monthly folds (overlapping calendar months) · Daily rebalancing (configurable) |
| **Vectorization** | All calculations are fully vectorized via numpy/pandas/vectorbt — no per-bar iteration |

---

## Module structure

```
src/ggTrader/
├── lab/                      # Research engine (vectorbt-first)
│   ├── cli.py               # CLI entry point (ggt lab)
│   ├── data.py              # Universe + OHLCV loading
│   ├── harness.py           # Walk-forward driver + fold logic
│   ├── metrics.py           # Sharpe, Calmar, max DD, win rate
│   ├── persist.py           # DB persistence to lab_runs/lab_periods
│   ├── simulate.py          # Vectorized portfolio sim (vectorbt.Portfolio)
│   ├── strategy.py          # Strategy protocol + LabConfig
│   └── strategies/
│       ├── momentum.py      # Momentum-based strategies (wfo_tournament, xs_momentum, dual_momentum)
│       └── signals.py       # Signal-based strategies (ema_cross, wfo_tournament_signal)
│
├── data/                    # Data loading infrastructure
│   ├── core/
│   │   ├── base_loader.py  # Abstract loader protocol
│   │   ├── stock_constants.py  # yfinance interval mapping, SP500 symbol list
│   │   └── constants.py     # Kraken symbol mapping, quote currencies, intervals
│   ├── historical/
│   │   ├── timescaledb_loader.py  # TimescaleDB OHLCV fetch
│   │   └── postgres_ingestor.py   # DB insert helpers
│   └── live/
│       └── yfinance_loader.py     # yfinance OHLCV fetch + cache
│
├── utils/
│   ├── config.py            # Configuration schema + defaults
│   ├── paths.py             # Project root resolution
│   └── db_engine.py         # SQLAlchemy engine + connection pool
│
└── cli/
    ├── main.py              # Typer CLI app (ggt command root)
    ├── cmd_ingest.py        # ggt ingest subcommand
    └── cmd_db.py            # ggt db subcommand
```

---

## Key data flows

### 1. Lab run (research)

```
ggt lab --strategy <name> --eval-start DATE --eval-end DATE
    ↓
load_ohlcv(universe, date_range)          [yfinance for equities, TimescaleDB for crypto]
    ↓
equity_universe_between(start, end)       [SP500 constituents on eval dates]
    ↓
build_strategy(name, cfg)                 [momentum or signal strategy from registry]
    ↓
walkforward([strat], ohlcv, ...)          [monthly folds, vectorized simulation]
    ↓
foreach fold:
  - train_window = prior months (in-sample)
  - test_window = next month (out-of-sample)
  - simulate via vectorbt.Portfolio
  - compute Sharpe, Calmar, max DD, win rate per fold
    ↓
persist to lab_runs + lab_periods tables
    ↓
print summary (stdout)
```

### 2. Data ingestion (crypto)

```
ggt ingest --days N
    ↓
TimescaleDBLoader.fetch_ohlcv()  [full universe from DB]
    ↓
or: CachedExchangeLoader         [CCXT live fetch + write-through to DB]
    ↓
persist to ohlcv table
```

### 3. Database administration

```
ggt db <diag|clean|truncate|compression|export>
    ↓
diagnostics: table sizes, row counts
clean: remove malformed rows
truncate: drop specific tables
compression: enable TimescaleDB hypertable compression
export: PostgreSQL dump
```

---

## Lab strategy interface

All strategies implement the `Strategy` protocol (defined in `lab/strategy.py`):

```python
class Strategy(Protocol):
    name: str
    target_kind: str  # "weights" or "signals"

    def select(self, asof: pd.Timestamp, data: pd.DataFrame,
               eligible: List[str]) -> Plan:
        """Point-in-time selection — must use only data <= asof."""
        ...

    def to_targets(self, plans: Dict[pd.Timestamp, Plan],
                   data: pd.DataFrame) -> DataFrame | SignalTargets:
        """Whole-window target matrix from per-rebalance plans.
        Weight strategies → DataFrame (time x symbol, float weights).
        Signal strategies → SignalTargets(entries, exits) boolean frames."""
        ...
```

- **Weight strategies** (target_kind="weights"): `to_targets` returns a `pd.DataFrame` with float weights (0.0 = exit, NaN = no order). Simulated via `vbt.Portfolio.from_orders(size_type="targetpercent")`.
- **Signal strategies** (target_kind="signals"): `to_targets` returns `SignalTargets(entries, exits)` with boolean DataFrames. Simulated via `vbt.Portfolio.from_signals()`.

All returns must be fully vectorized (no per-bar loops), aligned to the OHLCV index, and NaN-safe.

---

## Vectorbt portfolio simulation

`lab/simulate.py` provides two simulation functions:

**`simulate_weights()`** — for weight-based strategies (xs_momentum, dual_momentum):
```python
pf = vbt.Portfolio.from_orders(
    close=close,                     # 2D (dates × [strategy, symbol])
    size=size,                       # target-percent weights
    size_type="targetpercent",
    init_cash=START_CASH,
    fees=FEES, slippage=SLIPPAGE,
    cash_sharing=True,
    group_by=strategy_index,         # shared cash per strategy group
    call_seq="auto",
)
```

**`simulate_signals()`** — for signal-based strategies (ema_cross, wfo_tournament):
```python
pf = vbt.Portfolio.from_signals(
    close=close,                     # 2D (dates × [strategy, symbol])
    entries=entries,                  # boolean entry signals
    exits=exits,                     # boolean exit signals
    size=SIGNAL_POSITION_SIZE,       # fraction per entry (default 0.02)
    size_type="percent",
    init_cash=START_CASH,
    fees=FEES, slippage=SLIPPAGE,
    cash_sharing=True,
    group_by=strategy_index,
)
```

Both functions batch all strategies into ONE vectorbt call via `group_by`, then split the results back out. This is the core performance advantage — no per-strategy loops.

---

## Walk-forward fold logic

The `walkforward()` harness in `lab/harness.py` implements textbook walk-forward optimization:

1. **Universe eligibility** — stocks/coins with sufficient price history (e.g., 500+ trading days)
2. **Monthly folds** — each fold is one calendar month
   - **In-Sample (IS) / Train Window** — prior 12+ months of history (configurable)
   - **Out-of-Sample (OOS) / Test Window** — the evaluation month itself
3. **Evaluation period** — `--eval-start` to `--eval-end` spans all test windows
4. **Warmup period** — data is loaded from `eval_start - warmup_days` to ensure sufficient history for indicators
5. **Per-fold metrics** — Sharpe, Calmar, max drawdown, win rate, monthly return
6. **Aggregation** — mean and std of metrics across all folds

All results are persisted to TimescaleDB:
- `lab_runs` — one row per lab invocation (strategy, config, timestamps, summary metrics)
- `lab_periods` — one row per fold (strategy, fold #, start date, end date, per-fold metrics)

---

## Where to find things

| Path | Contents |
|---|---|
| `src/ggTrader/lab/` | Lab engine: CLI, data loading, walk-forward harness, metrics, persistence, strategy registry |
| `src/ggTrader/lab/strategies/` | Strategy implementations (momentum, signals) |
| `src/ggTrader/data/` | OHLCV loaders (yfinance, TimescaleDB), data schemas, market constants |
| `src/ggTrader/utils/` | Config, paths, DB engine |
| `src/ggTrader/cli/` | CLI commands (main, ingest, db) |
| `data/universe/sp500_constituents_history.csv.gz` | Point-in-time S&P 500 membership (2,712 snapshots, 1996–present) |
| TimescaleDB `lab_runs` table | Lab run history (strategy, config, execution time, summary metrics) |
| TimescaleDB `lab_periods` table | Per-fold results (strategy, fold dates, per-fold metrics) |

---

*Back to [README.md](../README.md).*
