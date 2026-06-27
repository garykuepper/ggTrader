# Architecture

How ggTrader is structured. For commands and usage see [CLI Reference](cli_reference.md); for codebase guidelines see [agents.md](../agents.md).

---

## At a glance

ggTrader is a **research-first lab** — a vectorbt-based walk-forward optimization framework for backtesting trading strategies on historical data, with a paper-trading deployment pipeline for validated strategies.

| Aspect | Detail |
|---|---|
| **Core Stack** | Python 3.10+ · vectorbt · TimescaleDB · yfinance · Tiingo (fallback) |
| **Data Sources** | TimescaleDB (crypto OHLCV, equity backfill) · yfinance (equities) · Tiingo (delisted tickers) · CSV (SP500/Nasdaq-100/Russell 2000 constituents) |
| **Lab Output** | TimescaleDB `lab_runs` / `lab_periods` tables (timestamped, immutable) |
| **Paper Trading** | Alpaca paper via `ggt paper` · ML feature gate · risk guardrails · Telegram alerts |
| **Supported Markets** | US equities (SP500, Nasdaq-100, Russell 2000 universes) · Crypto (customizable) |
| **Evaluation Window** | Monthly folds (overlapping calendar months) · Daily rebalancing (configurable) |
| **Vectorization** | All calculations are fully vectorized via numpy/pandas/vectorbt — no per-bar iteration |

---

## Module structure

```
src/ggTrader/
├── lab/                          # Research engine (vectorbt-first)
│   ├── cli.py                   # CLI entry point (ggt lab)
│   ├── data.py                  # Universe + OHLCV loading (yfinance, Tiingo, TimescaleDB)
│   ├── harness.py               # Walk-forward driver + fold logic
│   ├── metrics.py               # Sharpe, Calmar, max DD, win rate
│   ├── persist.py               # DB persistence to lab_runs/lab_periods
│   ├── simulate.py              # Vectorized portfolio sim (vectorbt.Portfolio)
│   ├── strategy.py              # Strategy protocol + LabConfig
│   ├── sweep.py                 # Parameter sweep grid builder + runner
│   ├── wfo.py                   # Walk-forward optimization (rolling folds, WFE, circuit breaker)
│   ├── gates.py                 # Robustness gates (NDH plateau filter, DSR)
│   ├── train_gate.py            # LightGBM train gate for signal quality
│   └── strategies/
│       ├── momentum.py          # Weight strategies (xs_momentum, dual_momentum)
│       ├── signals.py           # Signal strategies (ema_cross, wfo_tournament, bb_reversion,
│       │                        #   rsi_reversion, macd_divergence, volume_bb_reversion,
│       │                        #   mtf_reversion)
│       ├── ensemble.py          # Ensemble strategies (ensemble, ensemble_conviction)
│       ├── conviction.py        # Conviction-weighted BB sizing (conviction_bb)
│       └── indicators.py        # Shared vectorized indicator functions (BB, RSI, EMA,
│                                #   MACD, volume, multi-timeframe)
│
├── paper/                       # Paper trading deployment
│   ├── alpaca_broker.py         # Alpaca TradingClient adapter (paper-only)
│   ├── signal_runner.py         # Daily ensemble signal generation
│   ├── trader.py                # Signal → order orchestration
│   ├── feature_gate.py          # LightGBM ML feature gate (precision filter)
│   ├── risk.py                  # Risk guardrails (max positions, drawdown, daily loss)
│   ├── notifier.py              # Telegram trade alerts + daily summaries
│   └── persist.py               # Trade + snapshot persistence to TimescaleDB
│
├── data/                        # Data loading infrastructure
│   ├── core/
│   │   ├── base_loader.py       # Abstract loader protocol
│   │   ├── stock_constants.py   # yfinance interval mapping, SP500 symbol list
│   │   └── constants.py         # Kraken symbol mapping, quote currencies, intervals
│   ├── historical/
│   │   ├── timescaledb_loader.py # TimescaleDB OHLCV fetch
│   │   └── postgres_ingestor.py  # DB insert helpers
│   └── live/
│       └── yfinance_loader.py    # yfinance OHLCV fetch + cache
│
├── utils/
│   ├── config.py                # Configuration schema + defaults
│   ├── paths.py                 # Project root resolution
│   └── db_engine.py             # SQLAlchemy engine + connection pool
│
└── cli/
    ├── main.py                  # Typer CLI app (ggt command root)
    ├── cmd_ingest.py            # ggt ingest subcommand
    └── cmd_db.py                # ggt db subcommand
```

---

## Strategy taxonomy

### Weight strategies (target_kind="weights")

`to_targets` returns a `pd.DataFrame` with float weights (0.0 = exit, NaN = no order). Simulated via `vbt.Portfolio.from_orders(size_type="targetpercent")`.

| Strategy | Description |
|---|---|
| `xs_momentum` | Cross-sectional momentum — rank stocks by lookback return, equal-weight top N |
| `dual_momentum` | Dual momentum — same ranking but drops negative-momentum picks to cash |

### Signal strategies (target_kind="signals")

`to_targets` returns `SignalTargets(entries, exits)` with boolean DataFrames. Simulated via `vbt.Portfolio.from_signals()`.

| Strategy | Description |
|---|---|
| `ema_cross` | EMA crossover — enter when fast EMA crosses above slow, exit on cross below |
| `wfo_tournament` | Evaluates 4 EMA combos on 70% IS window, picks best by Sharpe, generates piecewise signals |
| `bb_reversion` | Bollinger Band mean reversion — enter on lower band touch, exit at middle band |
| `rsi_reversion` | RSI mean reversion — enter when RSI < oversold threshold, exit when RSI > exit threshold |
| `macd_divergence` | MACD bearish/bullish divergence detection over configurable window |
| `volume_bb_reversion` | BB reversion confirmed by volume spike (volume > period mean × multiplier) |
| `mtf_reversion` | Multi-timeframe — weekly RSI oversold + daily BB touch for higher-conviction entries |
| `ensemble` | Majority-vote of N sub-signals (bb, rsi, ema, macd, vol_bb, mtf). Configurable `min_agree` |
| `ensemble_conviction` | Same as ensemble but sizes positions by average strength of agreeing sub-signals |
| `conviction_bb` | Conviction-weighted BB sizing — position size proportional to distance below band |

All strategies implement the `Strategy` protocol (defined in `lab/strategy.py`):

```python
class Strategy(Protocol):
    name: str
    target_kind: str  # "weights" or "signals"

    def select(self, asof, data, eligible) -> Plan:
        """Point-in-time selection — must use only data <= asof."""

    def to_targets(self, plans, data) -> DataFrame | SignalTargets:
        """Target matrix from per-rebalance plans."""
```

Signal strategies additionally support:

```python
    def sweep_params(self) -> Dict[str, List[Any]]:
        """Parameter grid for sweep mode."""

    def sweep_signals(self, ohlcv, symbols) -> Tuple[entries, exits]:
        """Generate entry/exit signals for the current parameter set."""
```

---

## Key data flows

### 1. Lab run (research)

```
ggt lab --strategy <name> --eval-start DATE --eval-end DATE [--universe sp500|nasdaq100|russell2000]
    ↓
load_ohlcv(universe, date_range)          [TimescaleDB backfill, yfinance live, Tiingo fallback]
    ↓
equity_universe_between(start, end)       [point-in-time constituents on eval dates]
    ↓
build_strategy(name, cfg)                 [from strategy registry]
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

### 2. Parameter sweep

```
ggt lab --strategy <name> --sweep [--sweep-param key=v1,v2,v3]
    ↓
strategy.sweep_params()                   [default grid or --sweep-param overrides]
    ↓
build_grid(params)                        [cartesian product of all param combos]
    ↓
foreach combo:
  - instantiate strategy with combo params
  - run walk-forward
  - collect OOS metrics
    ↓
format_results_table()                    [ranked by Sharpe, stdout]
```

### 3. Walk-forward optimization (WFO)

```
ggt lab --strategy <name> --wfo
    ↓
generate_folds(eval_start, eval_end)      [rolling 12mo train / 3mo test windows]
    ↓
foreach fold:
  - sweep all param combos on train window
  - select best combo by composite score
  - evaluate on test window (OOS)
  - compute WFE = OOS_metric / IS_metric
  - check circuit breaker (halt if WFE < threshold)
    ↓
compute_anchor_set()                      [stable params across folds]
    ↓
print OOS summary + WFE + anchor params
```

### 4. Paper trading

```
ggt paper                                [daily cron, Mon-Fri]
    ↓
signal_runner.generate_signals()          [ensemble on current market data]
    ↓
feature_gate.filter(entries)              [LightGBM precision filter, drop < 0.50]
    ↓
risk.check_guardrails()                   [max positions, drawdown, daily loss]
    ↓
trader.execute(buys, sells)               [Alpaca paper orders, DAY TIF; poll for fills]
    ↓
persist.save(trades, snapshots)           [TimescaleDB; ledger only for actual fills]
    ↓
notifier.send(alerts)                     [Telegram]
```

### 5. Data ingestion (crypto)

```
ggt ingest --days N
    ↓
TimescaleDBLoader.fetch_ohlcv()           [full universe from DB]
    ↓
or: CachedExchangeLoader                  [CCXT live fetch + write-through to DB]
    ↓
persist to ohlcv table
```

### 6. Database administration

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

## Vectorbt portfolio simulation

`lab/simulate.py` provides two simulation functions:

**`simulate_weights()`** — for weight-based strategies (xs_momentum, dual_momentum):
```python
pf = vbt.Portfolio.from_orders(
    close=close,
    size=size,                       # target-percent weights
    size_type="targetpercent",
    cash_sharing=True,
    group_by=strategy_index,
)
```

**`simulate_signals()`** — for signal-based strategies (all signal strategies):
```python
pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=SIGNAL_POSITION_SIZE,       # fraction per entry (default 0.02)
    size_type="percent",
    cash_sharing=True,
    group_by=strategy_index,
)
```

Both functions batch all strategies into ONE vectorbt call via `group_by`, then split the results back out.

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

## Indicator library

`lab/strategies/indicators.py` provides all vectorized indicator functions used by signal strategies:

| Function | Purpose |
|---|---|
| `extract_close(ohlcv, symbols)` | Extract close prices from multi-index OHLCV |
| `extract_volume(ohlcv, symbols)` | Extract volume from multi-index OHLCV |
| `eligible_symbols(close, min_bars)` | Filter symbols with sufficient history |
| `bb_signals(close, period, std)` | Bollinger Band entry/exit signals |
| `bb_strength(close, period, std)` | BB conviction strength (distance below band) |
| `ema_signals(close, fast, slow)` | EMA crossover entry/exit signals |
| `ema_strength(close, fast, slow)` | EMA conviction strength (spread magnitude) |
| `rsi_signals(close, period, oversold, exit)` | RSI mean-reversion entry/exit signals |
| `rsi_strength(close, period, oversold)` | RSI conviction strength (depth below threshold) |
| `macd_signals(close, fast, slow, signal, window)` | MACD divergence entry/exit signals |
| `macd_strength(close, fast, slow, signal)` | MACD conviction strength |
| `volume_bb_signals(close, volume, ...)` | Volume-confirmed BB reversion signals |
| `volume_bb_strength(close, volume, ...)` | Volume-BB conviction strength |
| `mtf_signals(close, ...)` | Multi-timeframe reversion signals (weekly RSI + daily BB) |
| `mtf_strength(close, ...)` | Multi-timeframe conviction strength |

---

## Where to find things

| Path | Contents |
|---|---|
| `src/ggTrader/lab/` | Lab engine: CLI, data loading, walk-forward harness, metrics, persistence, strategy registry |
| `src/ggTrader/lab/strategies/` | Strategy implementations (momentum, signals, ensemble, conviction, indicators) |
| `src/ggTrader/paper/` | Paper trading: Alpaca broker, signal runner, trader, ML gate, risk, notifications, persistence |
| `src/ggTrader/data/` | OHLCV loaders (yfinance, Tiingo, TimescaleDB), data schemas, market constants |
| `src/ggTrader/utils/` | Config, paths, DB engine |
| `src/ggTrader/cli/` | CLI commands (main, ingest, db) |
| `scripts/ml_signal_screen.py` | Standalone ML pre-screen for evaluating signal quality |
| `data/universe/sp500_constituents_history.csv.gz` | Point-in-time S&P 500 membership (2,712 snapshots, 1996–present) |
| TimescaleDB `lab_runs` table | Lab run history (strategy, config, execution time, summary metrics) |
| TimescaleDB `lab_periods` table | Per-fold results (strategy, fold dates, per-fold metrics) |

---

*Back to [README.md](../README.md).*
