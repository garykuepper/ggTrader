# ggTrader

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/garykuepper/ggTrader)
[![Backtest Engine](https://img.shields.io/badge/engine-VectorBT-orange.svg)](https://vectorbt.pro/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Database: TimescaleDB](https://img.shields.io/badge/database-TimescaleDB-00D3FF?logo=timescaledb&logoColor=white)](https://www.timescale.com/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-white?logo=pytest&logoColor=2f9fe3)](https://docs.pytest.org/en/stable/)
[![Static Types](https://img.shields.io/badge/types-Mypy-blue.svg)](http://mypy-lang.org/)

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
- [**Ingestion & DB performance**](docs/ingestion_optimization.md): Notes on TimescaleDB usage.

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

---
*For a detailed walkthrough, refer to the [Unified CLI Guide](docs/UNIFIED_PIPELINE.md) or [Architecture Guide](docs/architecture.md).*
