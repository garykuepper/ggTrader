# ggTrader

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-white?logo=pytest&logoColor=2f9fe3)](https://docs.pytest.org/en/stable/)

Welcome to ggTrader! This is a research platform (or "lab") designed to help you test, tune, and deploy trading strategies. 

It simulates how a trading strategy would have performed in the past using **Walk-Forward Optimization**—a realistic testing method where strategy parameters are tuned on past data and evaluated on subsequent "unseen" data on a rolling basis.

The platform supports both US equities (stocks like the S&P 500 via Yahoo Finance) and cryptocurrencies (loaded from a local database).

---

## Core Features 

- **Realistic Backtesting (Walk-Forward Optimization)**: Instead of cheating by testing rules on the same data used to create them, we split history into monthly segments. We find the best rules on a "training" segment (in-sample) and test them on the next "testing" segment (out-of-sample).
- **Super-Fast Simulations (Vectorization)**: Traditional backtesters test day-by-day in a slow loop. ggTrader uses the **vectorbt** library to calculate your entire portfolio's performance all at once, letting you run years of history across hundreds of stocks in seconds.
- **Constituent History (No Survivorship Bias)**: Most people test stock strategies using the current list of S&P 500 stocks. This is a mistake because it ignores companies that went bankrupt or got acquired. ggTrader tracks the historical "point-in-time" list of S&P 500 members to make your tests realistic.
- **Easy Strategy Customization**: Implement a simple Python blueprint (the `Strategy` protocol) and your new trading rules are immediately usable from the command line.
- **Database Storage**: Saves all test results directly to **TimescaleDB** (a time-series database) so you never lose your results.

---

## Available Strategies

Strategies are split into two kinds:
1. **Signal Strategies**: Decides **when** to buy or sell (e.g. buying when moving averages cross).
2. **Weight Strategies**: Decides **how much** of each stock to hold (e.g. putting equal amounts into the top 10 momentum stocks).

| Strategy Name | Type | Description |
|------|------|-------------|
| `ensemble` | Signal | **The deployed strategy.** Combines five indicators (Bollinger Bands, RSI, EMA cross, MACD divergence, volume-confirmed BB) and only trades when enough of them agree. This is what runs the live paper account. |
| `ensemble_conviction` | Signal | As above, but sizes each position by how many indicators agreed. |
| `bb_reversion` | Signal | Buys when price falls below a lower Bollinger Band and sells when it returns to the middle. |
| `rsi_reversion` | Signal | Buys when RSI signals oversold, sells on recovery. |
| `wfo_tournament` | Signal | Automatically finds the best moving average parameters by running a mini-competition on past data. |
| `ema_cross` | Signal | A classic strategy that buys when a fast moving average crosses above a slow moving average, and sells when it crosses below. |
| `xs_momentum` | Weight | Ranks stocks by their past 12-month returns and holds the top performers. |
| `dual_momentum` | Weight | Ranks stocks by momentum, but moves the entire portfolio to safe cash if the overall market is falling. |

The registry holds **36** strategies in total — the rest are closed NO-GO
research kept for reproducibility. See
[CLI Reference](docs/cli_reference.md) for the full list and
[Research Snapshot](docs/research/RESEARCH_SNAPSHOT.md) for each verdict.

---

## Quick Start

### 1. Installation
Install the project dependencies in your Python environment:
```bash
pip install -e .
```

### 2. Verify
Check that the command-line interface is working:
```bash
ggt --help
```

### 3. Run a Simulation
Simulate a strategy over the S&P 500 universe (from 2021 to present on the top 50 stocks):
```bash
ggt lab --strategy wfo_tournament
```

### 4. Diagnostic Run
Run a small, fast test run on just 10 stocks:
```bash
ggt lab --strategy xs_momentum --top-n 10 --eval-start 2024-01-01
```

---

## Folder Layout

```
src/ggTrader/
├── lab/              # The Research Engine (running simulations)
│   ├── cli.py        # Command-line entry point for simulation commands
│   ├── data.py       # Handles downloading/loading price data
│   ├── harness.py    # The simulation controller (handles rolling folds)
│   ├── metrics.py    # Calculates return, Sharpe ratio, and drawdowns
│   ├── persist.py    # Saves test results to the database
│   ├── simulate.py   # Runs the vectorbt backtesting math
│   ├── strategy.py   # Strategy interfaces and blueprints
│   └── strategies/   # Code for individual trading strategies
├── paper/            # Live Paper Trading (runs on cron against Alpaca)
│   ├── trader.py     # Orchestrates one trading cycle
│   ├── alpaca_broker.py  # Broker API wrapper (paper account)
│   ├── signal_runner.py  # Generates today's ensemble signals
│   ├── overlay.py    # Multi-sleeve blending and vol targeting
│   ├── risk.py       # Position caps, drawdown halt, margin pre-flight
│   ├── persist.py    # Writes paper_trades / paper_snapshots
│   └── ...           # notifier, feature_gate, split_check, dividend_check
├── data/             # Data Loaders (Yahoo Finance, database loaders)
├── utils/            # Shared utilities (database connections, config)
└── cli/              # Main CLI entry subcommands (ggt lab | paper | db | ingest*)
```

\* `ggt ingest` is a non-functional stub — see [CLI Reference §3](docs/cli_reference.md).

| Additional Files | Description |
|---|---|
| `data/universe/` | Historical S&P 500 membership records. |
| `scripts/` | Script utilities for cleaning and backfilling data. |
| `tests/lab/` | Code tests to ensure the simulator is working correctly. |

---

## Documentation

For a deeper dive, check out our detailed guides:
- [**Next Steps**](docs/next_steps.md) — The current worklist. Start here if you are picking up work.
- [**Roadmap**](docs/roadmap.md) — Goals, strategy status table, and project history.
- [**Research Snapshot**](docs/research/RESEARCH_SNAPSHOT.md) — Every strategy tried and its verdict.
- [**Installation Guide**](docs/installation.md) — How to set up Python, TimescaleDB, and Docker.
- [**CLI Reference**](docs/cli_reference.md) — A breakdown of all commands, flags, and parameters.
- [**Architecture Guide**](docs/architecture.md) — How the codebase is built, how data flows, and how the simulation works.
- [**Changelog**](docs/changelog.md) — A record of updates made to the project.
- [**Agent Guidelines**](AGENTS.md) — The single source of truth for AI assistants working in this repo.

---

## License

MIT.

> **Note:** this repository does not currently contain a `LICENSE` file — the
> link that used to be here dangled. The MIT declaration above is the only
> statement of license. Add a `LICENSE` file to make it enforceable.
