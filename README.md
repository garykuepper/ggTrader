# ggTrader

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-white?logo=pytest&logoColor=2f9fe3)](https://docs.pytest.org/en/stable/)

An algorithmic crypto trading bot. Each month it runs **walk-forward optimization** — fits strategy parameters on a training window of historical data, validates them on the next chunk of unseen ("out-of-sample") data, slides forward and repeats — across the most-traded coins, then trades the winning per-coin parameters live on **Binance.US** or **Kraken Pro**. The venue is a configuration switch (both wired through CCXT, a multi-exchange Python library). Live execution is currently migrating from Kraken Pro to Binance.US to capture round-trip fees that are roughly 12× lower.

The same code path runs research, backtest, and live execution — what you simulate is what trades.

## What you get

- **Walk-Forward Optimization (WFO)** with textbook overfitting controls — four aggregate gates ensure parameters that worked in training also work on unseen data, plus rank-based composite scoring (Sortino + Calmar + Profit Factor).
- **Multi-methodology** — WFO on momentum signals, cash-and-carry on dated futures, and funding-rate arbitrage on perpetual futures coexist behind the same data + execution layer. Adding a new methodology is implementing one protocol.
- **TimescaleDB** for time-series storage — historical and live data share one table; multi-venue Open/High/Low/Close/Volume (OHLCV) candles keyed by `(symbol, interval, venue)`.
- **VectorBT + Numba** for fast backtests across thousands of parameter combinations (just-in-time compiled, vectorized).
- **Live engine** for Binance.US and Kraken Pro — places a market or limit buy on entry, immediately places a venue-native trailing-stop (Kraken) or One-Cancels-Other (OCO) order (Binance.US) so positions stay protected even if our process dies. Daily-loss circuit breaker. Real-time Grafana dashboard.
- **Monthly auto-recalibration** that hot-reloads new parameters with no downtime.

## Documentation

- [**Architecture**](docs/architecture.md) — how the four layers fit together (defines vocabulary used across other docs)
- [**CLI Reference**](docs/cli_reference.md) — `ggt` subcommands and flags
- [**Installation**](docs/installation.md) — database, environment variables, Docker
- [**Live Trading**](docs/live_trading_guide.md) — deploying optimized parameters
- [**Roadmap**](docs/roadmap.md) — what's shipped and what's next
- [**Changelog**](docs/changelog.md) — dated record of changes

## Project layout

| Path | What's there |
|---|---|
| `src/ggTrader/` | Core engine, data adapters, strategies, command-line interface |
| `scripts/` | One-off operational tooling (universe regeneration, backfills, correlation matrices) |
| `results/research/` | Output of each research run — Markdown report, raw JSON, plots |
| `data/` | Live trader runtime artifacts (rendered dashboards). State of record lives in TimescaleDB. |

## Quick start

```bash
# Install
pip install -e .

# Run research — top 50 USD coins, 3 years history, 5 parallel workers
python ggt.py research --top 50

# Replay the latest research as a portfolio backtest
python ggt.py backtest

# Start live trading (always use --dry-run first to inspect what would happen)
python ggt.py trade --dry-run
```

See the [Installation Guide](docs/installation.md) for the full setup and the [CLI Reference](docs/cli_reference.md) for every command.

## License

MIT. See [LICENSE](LICENSE).
