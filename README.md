# ggTrader

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-white?logo=pytest&logoColor=2f9fe3)](https://docs.pytest.org/en/stable/)

A **vectorbt-based research lab** for walk-forward optimization of trading strategies. Each evaluation run splits historical data into monthly folds, trains strategy parameters on in-sample data, validates on held-out out-of-sample data, and persists results to TimescaleDB.

The lab supports US equities (S&P 500 via yfinance) and crypto (via TimescaleDB OHLCV). All simulation is fully vectorized through vectorbt — no per-bar iteration.

## What you get

- **Walk-forward optimization** with monthly folds — in-sample training, out-of-sample validation, no lookahead.
- **Vectorized backtesting** via `vectorbt.Portfolio` — one grouped call simulates all strategies simultaneously with shared cash pools.
- **Point-in-time universes** — S&P 500 membership as of each selection date (2,712 snapshots, 1996–present), preventing survivorship bias.
- **Pluggable strategies** — implement the `Strategy` protocol (`select` + `to_targets`) and it's immediately available via the CLI.
- **TimescaleDB persistence** — all run results stored in `lab_runs` and `lab_periods` tables. No file-based state.

## Available strategies

| Name | Type | Description |
|------|------|-------------|
| `wfo_tournament` | Signal | EMA combo tournament per rebalance (4 fast/slow pairs, 70% IS window) |
| `ema_cross` | Signal | Simple whole-window EMA crossover |
| `xs_momentum` | Weight | Cross-sectional momentum (top-N by 12-1 momentum) |
| `dual_momentum` | Weight | Absolute + relative momentum filter |

## Quick start

```bash
# Install
pip install -e .

# Verify
ggt --help

# Run a strategy over SP500 (2021–present, top 50 stocks)
ggt lab --strategy wfo_tournament

# Smaller diagnostic run
ggt lab --strategy xs_momentum --top-n 10 --eval-start 2024-01-01
```

## Project layout

```
src/ggTrader/
├── lab/              # Research engine (vectorbt-first)
│   ├── cli.py        # CLI entry point
│   ├── data.py       # Universe selection + OHLCV loading
│   ├── harness.py    # Walk-forward driver
│   ├── metrics.py    # Sharpe, Calmar, max drawdown, win rate
│   ├── persist.py    # DB persistence
│   ├── simulate.py   # vbt.Portfolio wrappers (from_orders, from_signals)
│   ├── strategy.py   # Strategy protocol + LabConfig
│   └── strategies/   # Strategy implementations
├── data/             # Data loading infrastructure
│   ├── core/         # Base loader, SP500 constituents, constants
│   ├── historical/   # TimescaleDB loader + ingestor
│   └── live/         # yfinance loader
├── utils/            # Config, paths, DB engine
└── cli/              # ggt lab | ingest | db
```

| Other paths | Contents |
|---|---|
| `data/universe/` | SP500 constituent history, venue listing snapshots |
| `scripts/` | Data backfill utilities |
| `tests/lab/` | Lab test suite |

## Documentation

- [**Architecture**](docs/architecture.md) — module structure, data flows, strategy protocol
- [**CLI Reference**](docs/cli_reference.md) — `ggt` commands and flags
- [**Installation**](docs/installation.md) — TimescaleDB, environment, Docker
- [**Roadmap**](docs/roadmap.md) — research history and direction (archived)
- [**Changelog**](docs/changelog.md) — dated record of changes

## License

MIT. See [LICENSE](LICENSE).
