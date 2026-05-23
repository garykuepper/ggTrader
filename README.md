# ggTrader

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-white?logo=pytest&logoColor=2f9fe3)](https://docs.pytest.org/en/stable/)

An algorithmic crypto trading bot. Each month it runs walk-forward optimization across the most-traded coins, picks per-coin parameters that hold up out-of-sample, and trades them live on Binance.US or Kraken Pro (CCXT-based; the venue is a config switch). Migrating live execution from Kraken Pro → Binance.US to capture the ~12× lower round-trip fees.

The same code path runs research, backtest, and live execution — what you simulate is what trades.

## What you get

- **Walk-forward optimization** with textbook-reset overfitting controls (4 aggregate gates: WFE, %profitable folds, parameter CV, DD ratio) + rank-based Sortino+Calmar+ProfitFactor composite scoring
- **Multi-methodology**: WFO momentum, cash-and-carry (CashAndCarryBTC), funding-rate arbitrage (FundingCarryBTC) — all behind the same data + execution layer
- **TimescaleDB** for time-series storage — historical and live data share one table; multi-venue OHLCV (Kraken + Binance.US) keyed by `(symbol, interval, venue)`
- **VectorBT + Numba** for fast backtests across thousands of parameter combinations
- **Live engine** for Binance.US and Kraken Pro, with native trailing-stop orders, daily-loss circuit breaker, and Grafana mirroring
- **Monthly auto-recalibration** that hot-reloads new parameters with no downtime

## Documentation

- [**Architecture**](docs/architecture.md) — how the four layers fit together
- [**CLI Reference**](docs/cli_reference.md) — `ggt` subcommands and flags
- [**Installation**](docs/installation.md) — database, env vars, Docker
- [**Live Trading**](docs/live_trading_guide.md) — deploying optimized parameters
- [**Roadmap**](docs/roadmap.md) — what's shipped and what's next
- [**Changelog**](docs/changelog.md) — dated record of changes

## Project layout

| Path | What's there |
|---|---|
| `src/ggTrader/` | Core engine, data adapters, strategies, CLI |
| `scripts/` | One-off operational tooling |
| `results/research/` | Output of each research run |
| `data/` | Live trader state, universe snapshots |

## Quick start

```bash
# Install
pip install -e .

# Run research (top 50 USD coins, 3 years history, 5 parallel workers)
python ggt.py research --top 50

# Replay the latest research as a portfolio backtest
python ggt.py backtest

# Start live trading (use --dry-run first to inspect signals)
python ggt.py trade --dry-run
```

See the [Installation Guide](docs/installation.md) for the full setup and the [CLI Reference](docs/cli_reference.md) for every command.

## License

MIT. See [LICENSE](LICENSE).
