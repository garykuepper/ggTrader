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

- [**Architecture Guide**](docs/architecture.md): Deep dive into the project structure and data flow.
- [**Installation & Setup**](docs/installation.md): How to get `ggTrader` running on your local machine.
- [**Strategy Pipeline Guide**](docs/strategy_pipeline_guide.md): Comprehensive workflow for optimizing strategies across multiple cryptocurrencies.
- [**Analysis Guide**](docs/analysis_guide.md): Short WFO / backtest / sensitivity workflow notes (details live in the pipeline guide).
- [**Ingestion & DB performance**](docs/ingestion_optimization.md): Notes on data ingestion and TimescaleDB usage.
- [**Pipeline run history**](docs/pipeline_run_history.md): Pre-flight checklist, manual table, and default auto-append after each successful pipeline (`GGTRADER_APPEND_RUN_HISTORY=0` to disable for testing).

## 📁 Project Structure

- `src/ggTrader/`: Core package including `core` engine, `data` adapters, and `indicators`.
- `scripts/`: Operational scripts for backtesting, WFO, and sensitivity analysis.
- `notebooks/`: Research and visualization tools.
- `results/`: Standardized output for all strategy executions.
- `data/`: Local storage for cached data and exports.

## ⚡ Quick Start

1. **Clone & Setup** (editable install is **required** so `python scripts/...` can import `ggTrader`):

   ```bash
   pip install -e .
   ```

   This also installs **console commands** (on PATH inside the active venv): `ggtrader-backtest`, `ggtrader-wfo`, `ggtrader-sensitivity`, `ggtrader-pipeline`, `ggtrader-compare-strategies`, `ggtrader-view-results`, `ggtrader-pipeline-status`, `ggtrader-manage-data`, `ggtrader-manage-db`. They run the same code as the matching files under `scripts/`.

2. **Run a Backtest**:

   ```bash
   python scripts/run_backtest.py --symbols BTC-USD
   # or: ggtrader-backtest --symbols BTC-USD
   ```

   Use `--movers N` for a daily top-*N* mover mask, or `--params path/to/params.json` after WFO. Pass `--no-progress` to disable the VectorBT progress bar.

3. **Execute WFO**:

   ```bash
   python scripts/run_walk_forward_optimization.py
   ```

4. **Run Full Pipeline** (Per-Coin WFO → Validation → Report; optional `--sensitivity` for Phase 1):

   ```bash
   python scripts/run_full_pipeline.py
   ```

---
*For a more detailed breakdown, please refer to the [Strategy Pipeline Guide](docs/strategy_pipeline_guide.md) or [Architecture Guide](docs/architecture.md).*
