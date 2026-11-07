# ggTrader

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-white?logo=pytest&logoColor=2f9fe3)](https://docs.pytest.org/en/stable/)

Welcome to ggTrader! This is a research platform (or "lab") designed to help you test, tune, and deploy trading strategies. 

It simulates how a trading strategy would have performed in the past using **Walk-Forward Optimization**—a realistic testing method where strategy parameters are tuned on past data and evaluated on subsequent "unseen" data on a rolling basis.

The platform supports US equities (stocks like the S&P 500 via Yahoo Finance) for both research and live paper trading. Crypto is research-only: historical OHLCV loads from a local TimescaleDB, but there is no supported live-trading path for it — the crypto execution engines were removed, and crypto data ingestion (`ggt ingest`) was removed.

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
[Research Snapshot](research/RESEARCH_SNAPSHOT.md) for each verdict.

---

## Quick Start

### 1. Installation
Install the locked dependency set (Python 3.12, [uv](https://docs.astral.sh/uv/)):
```bash
uv sync --frozen --extra test --extra dev
source .venv/bin/activate
```

### 2. Verify
Check that the command-line interface is working:
```bash
ggt --help
```

### 3. Run a Simulation
Walk-forward a strategy over the S&P 500 universe on the vectorbt engine (frozen tape; pin both dates for any number you will quote):
```bash
ggt lab --strategy ensemble --wfo --eval-start 2021-01-31 --eval-end 2026-04-30
```

### 4. Diagnostic Run
Run a small, fast test run on just 10 stocks:
```bash
ggt lab --strategy ensemble --wfo --eval-end 2022-07-31 --max-stocks 40
```

Ten strategies are ported (see [CLI Reference §1](docs/cli_reference.md)); the other 28 are closed research that
runs at git tag `legacy-engine-vbt0285` (`scripts/legacy_env.sh`).

---

## Folder Layout

```
src/ggTrader/
├── lab/              # The Research Engine (running simulations)
│   ├── cli.py        # Command-line entry point for simulation commands
│   ├── data.py       # Handles downloading/loading price data
│   ├── wfo/          # The walk-forward (rolling folds, gates, anchor, breaker) on the engine
│   ├── persist.py    # Saves blend results to the database
│   ├── universe.py   # Point-in-time S&P 500 membership mask
│   ├── strategy.py   # LabConfig (shared research settings)
│   └── strategies/   # The ported strategies and the vote glue
├── engine/           # The only code that imports vectorbt (presets, simulation, statistics)
├── paper/            # Live Paper Trading (Alpaca; stopped by the owner since 2026-11-06, dry-run first, cron only on the owner's say-so)
│   ├── trader.py     # Orchestrates one trading cycle
│   ├── alpaca_broker.py  # Broker API wrapper (paper account)
│   ├── signal_runner.py  # Generates today's ensemble signals
│   ├── overlay.py    # Multi-sleeve blending and vol targeting
│   ├── risk.py       # Position caps, drawdown halt, margin pre-flight
│   ├── persist.py    # Writes paper_trades / paper_snapshots
│   └── ...           # notifier, split_check, dividend_check
├── data/             # Data Loaders (Yahoo Finance, database loaders)
├── utils/            # Shared utilities (database connections, config)
└── cli/              # Main CLI entry subcommands (ggt lab | paper | db)
```


| Additional Files | Description |
|---|---|
| `data/universe/` | Historical S&P 500 membership records. |
| `scripts/` | Script utilities for cleaning and backfilling data. |
| `tests/lab/` | Code tests to ensure the simulator is working correctly. |
| `tests/` | `scripts/run_tests.sh` runs the fast core (about two minutes); `scripts/run_tests.sh --full` runs everything and is the gate before a merge, a deploy or a vectorbt upgrade. |

---

## Documentation

The prospective intraday practice journal is activated as an isolated automatic
recorder. First observation: October 30, 2026; operational checkpoint: November 30.
It records hypothetical SPY positions and places no orders. See the
[activation record](research/intraday/spy-momentum-prospective-observer/2026-10-27-intraday-observer-activation.md) and
[operator commands](docs/cli_reference.md#prospective-intraday-practice-journal).

Historical intraday r8 is closed as **UNRESOLVED (data)** after parent-verified
collection and signal-blind validation found missing endpoint bars. No score was
computed. See the [validation report](research/intraday/spy-momentum-feasibility/2026-10-25-intraday-r8-validation-unresolved.md).
The prospective observer is separate from that closed study and production paper trading.

For a deeper dive, check out our detailed guides:
- [**Next Steps**](docs/next_steps.md) — The current worklist. Start here if you are picking up work.
- [**Roadmap**](docs/roadmap.md) — Goals, strategy status table, and project history.
- [**Research Snapshot**](research/RESEARCH_SNAPSHOT.md) — Every strategy tried and its verdict.
- [**Research**](research/README.md) — Where research reports live and how to find a study (moved from docs/research on 2026-11-05).
- [**Installation Guide**](docs/installation.md) — How to set up Python, TimescaleDB, and Docker.
- [**CLI Reference**](docs/cli_reference.md) — A breakdown of all commands, flags, and parameters.
- [**Architecture Guide**](docs/architecture.md) — How the codebase is built, how data flows, and how the simulation works.
- [**Changelog**](docs/changelog.md) — A record of updates made to the project.
- [**Agent Guidelines**](AGENTS.md) — The single source of truth for AI assistants working in this repo.

---

## License

MIT. See [LICENSE](LICENSE).

The [event-timing audit utility](research/audits/event-timing/event-timing-audit-implementation.md)
provides offline evidence replay and guarded read-only collection for the
registered PEAD/insider timing study. It computes no returns; full collection
remains blocked pending code review and an authorized contact.
