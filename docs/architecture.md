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
│   │   ├── sp500_constituents.csv  # S&P 500 tickers (updated infrequently)
│   │   └── constants.py     # Markets, intervals, holidays
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

All strategies implement the `Strategy` protocol:

```python
from ggTrader.lab.strategy import Strategy, LabConfig

class MyStrategy(Strategy):
    def __call__(self, universe: List[str], ohlcv: DataFrame, cfg: LabConfig) -> Tuple[DataFrame, DataFrame]:
        """
        Args:
            universe: list of stock/coin tickers available on this date
            ohlcv: multi-index DataFrame (ticker, field) → values
            cfg: LabConfig with top_n, lookback, skip, max_stocks
        
        Returns:
            (weights, signals) as DataFrames (dates × symbols)
                weights: fractional allocation per symbol each day [0, 1]
                signals: +1 (long), 0 (no position), -1 (exit)
        """
```

All returns must be:
- **Fully vectorized** — numpy arrays or pandas DataFrames, no per-bar loops
- **Aligned to OHLCV index** — matching dates and symbols
- **NaN-safe** — forward-fill or zero-fill missing values before passing to vectorbt

---

## Vectorbt portfolio simulation

The `simulate()` function in `lab/simulate.py` wraps vectorbt:

```python
import vectorbt as vbt

pf = vbt.Portfolio.from_signals(
    close=ohlcv['close'],           # 2D array (dates × symbols)
    entries=signals == 1,           # boolean (dates × symbols)
    exits=signals == -1,            # boolean (dates × symbols)
    init_cash=INIT_CASH,
    fees=TRADING_FEES,
    freq='D',                       # daily rebalancing
    group_by=True,                  # shared cash pool across symbols
)

# Compute metrics
sharpe = pf.sharpe_ratio(freq='D')
calmar = pf.calmar_ratio()
max_dd = pf.max_drawdown()
returns = pf.returns()
```

**Key configuration:**
- `group_by=True` — all symbols share one capital pool (no per-symbol independent cash)
- `freq='D'` — daily rebalancing (can be changed; monthly folds are handled by the harness, not vectorbt)
- `init_cash` and `fees` set at call time from config

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
| `data/core/sp500_constituents.csv` | S&P 500 ticker list (updated infrequently) |
| TimescaleDB `lab_runs` table | Lab run history (strategy, config, execution time, summary metrics) |
| TimescaleDB `lab_periods` table | Per-fold results (strategy, fold dates, per-fold metrics) |

---

*Back to [README.md](../README.md).*
