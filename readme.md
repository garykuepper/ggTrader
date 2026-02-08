# ggTrader

A sleek algorithmic trading bot built for Kraken and other exchanges. Designed for professional researchers and traders.

## Project Goals
- **Modular Design**: Separated core logic, data management, and indicators.
- **Reproducibility**: Structured results management and parameter tracking.
- **Optimization**: Built-in Walk-Forward Optimization (WFO) and sensitivity analysis.
- **Professional Analytics**: Seamless integration with Jupyter Notebooks for visualization.

## Directory Structure
- `src/ggTrader/`: Core package
  - `core/`: Trading engine, portfolio, and simulation logic.
  - `data/`: Data adapters (Kraken specifically under `data/kraken/`).
  - `indicators/`: Signal generation logic.
  - `utils/`: Shared utilities including `ResultsManager`.
- `scripts/`: Operational scripts
  - `backtest/`: Main backtesting runners.
  - `run_walk_forward_optimization.py`: Primary parameter optimization script.
  - `run_sensitivity_analysis.py`: Tests parameter stability.
- `notebooks/`: Visualization and deep-dive analysis.
- `results/`: Output from scripts (timestamped folders).
- `data/`: Local storage for raw and parquet data.

## Key Scripts
- `python scripts/backtest/run_backtest.py`: Run a standard backtest. Supports `--params` to load optimal values.
- `python scripts/run_walk_forward_optimization.py`: Execute WFO to find stable parameters over time.
- `python scripts/run_sensitivity_analysis.py`: Analyze how strategy performance changes with parameter variations.

## Quick Start
1. Setup environment: `pip install -e .`
2. Run a backtest: `python scripts/backtest/run_backtest.py`
3. Optimize: `python scripts/run_walk_forward_optimization.py`