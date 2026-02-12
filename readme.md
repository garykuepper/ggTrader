# ggTrader

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Backtest Engine](https://img.shields.io/badge/engine-VectorBT-orange.svg)](https://vectorbt.pro/)

A sleek, high-performance algorithmic trading bot built for Kraken and other major exchanges. Designed for professional researchers and quantitative traders who demand speed, reproducibility, and robust optimization.

## 🚀 High-Level Overview

`ggTrader` provides a modular framework for developing, testing, and optimizing trading strategies. It leverages **DuckDB** for efficient data management and **VectorBT** for lighting-fast backtesting.

### Key Features

- **Modular Architecture**: Clean separation between data ingestion, signal generation, and execution logic.
- **High-Performance Backtesting**: Integrated `FastBacktest.py` utilizing VectorBT and CuPy for GPU-accelerated simulations.
- **Advanced Optimization**: Built-in Walk-Forward Optimization (WFO) and sensitivity analysis to ensure strategy stability.
- **Robust Data Layer**: Uses DuckDB for localized, high-speed OHLCV storage and retrieval.
- **Professional Analytics**: Seamless integration with Jupyter Notebooks for deep-dive visualization and reporting.

## 📖 Documentation

- [**Architecture Guide**](ARCHITECTURE.md): Deep dive into the project structure and data flow.
- [**Installation & Setup**](INSTALLATION.md): How to get `ggTrader` running on your local machine.

## 📁 Project Structure

- `src/ggTrader/`: Core package including `core` engine, `data` adapters, and `indicators`.
- `scripts/`: Operational scripts for backtesting, WFO, and sensitivity analysis.
- `notebooks/`: Research and visualization tools.
- `results/`: Standardized output for all strategy executions.
- `data/`: Local database storage (`.db` files and Parquet).

## ⚡ Quick Start

1. **Clone & Setup**:

   ```bash
   pip install -e .
   ```

2. **Run a Backtest**:

   ```bash
   python scripts/backtest/run_backtest.py --symbols BTC/USD
   ```

3. **Execute WFO**:

   ```bash
   python scripts/run_walk_forward_optimization.py
   ```

---
*For a more detailed breakdown, please refer to the [Architecture Guide](ARCHITECTURE.md).*
