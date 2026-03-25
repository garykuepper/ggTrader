# ggTrader

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Architecture: Clean](https://img.shields.io/badge/architecture-clean-orange.svg)](docs/architecture.md)
[![Database: TimescaleDB](https://img.shields.io/badge/database-TimescaleDB-00D3FF?logo=timescaledb&logoColor=white)](https://www.timescale.com/)
[![Backtest Engine: VectorBT](https://img.shields.io/badge/engine-VectorBT-orange.svg)](https://vectorbt.pro/)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-white?logo=pytest&logoColor=2f9fe3)](https://docs.pytest.org/en/stable/)

A sleek, high-performance algorithmic trading bot built for Kraken and other major exchanges. Designed for professional researchers and quantitative traders who demand speed, reproducibility, and robust optimization.

## 🚀 High-Level Overview

`ggTrader` provides a modular framework for developing, testing, and optimizing trading strategies. It leverages **TimescaleDB** for efficient time-series data management and **VectorBT** for lighting-fast backtesting.

### Key Features

- **Modular Architecture**: Clean separation between data ingestion, signal generation, and execution logic.
- **High-Performance Backtesting**: Integrated `FastBacktest.py` utilizing VectorBT and CuPy for GPU-accelerated simulations.
- **Advanced Optimization**: Built-in Walk-Forward Optimization (WFO) and sensitivity analysis to ensure strategy stability.
- **Robust Data Layer**: Uses TimescaleDB (PostgreSQL) for centralized, high-speed OHLCV storage and retrieval.
- **Professional Analytics**: Seamless integration with Jupyter Notebooks for deep-dive visualization and reporting.

## 📖 Documentation

- [**Unified CLI Guide**](docs/UNIFIED_PIPELINE.md): Start here for the `ggt` command reference.
- [**Architecture Guide**](docs/architecture.md): Deep dive into the project structure and data flow.
- [**Installation & Setup**](docs/installation.md): How to get `ggTrader` running on your local machine.
- [**Live Trading Guide**](docs/live_trading_guide.md): Deploying optimized strategies to Kraken.

## 📁 Project Structure

- `src/ggTrader/`: Core engine, data adapters, and strategy indicators.
- `scripts/`: Operational scripts (wrapped by `ggt`).
- `results/`: Standardized output for all strategy executions.
- `data/`: Local storage for cached data, exports, and symbol pools.

## ⚡ Quick Start

The unified `ggt` CLI is the recommended way to interact with the engine.

1. **Install Dependencies**:

   ```bash
   pip install -e .
   ```

2. **Run a Research Optimization (WFO)**:

   Fetches the top 50 liquid coins and runs a parallel 3-year WFO.

   ```bash
   python ggt.py research --top 50
   ```

3. **Run a Backtest**:

   Simulates signals for specific symbols using the latest optimized parameters.

   ```bash
   python ggt.py backtest --symbols BTC,ETH
   ```

4. **Production Recalibration**:

   Generates portfolio weights for live trading.

   ```bash
   python ggt.py production
   ```

5. **Start Live Trading**:

   Begins the execution heartbeat.

   ```bash
   python ggt.py trade --dry-run
   ```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built with ❤️ and powered by **Google Gemini** for advanced algorithmic coding and research optimization.*

For a detailed walkthrough, refer to the [**Unified CLI Guide**](docs/UNIFIED_PIPELINE.md) or [**Architecture Guide**](docs/architecture.md).
