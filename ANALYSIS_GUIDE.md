# ggTrader Strategy Analysis & Optimization Guide

This guide explains how to use the suite of analysis and optimization tools located in the `scripts/` directory. These tools are designed to help you find robust parameters and validate your trading strategy.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Parameter Sensitivity Analysis](#1-parameter-sensitivity-analysis)
3. [Walk-Forward Optimization (WFO)](#2-walk-forward-optimization-wfo)
4. [Final Backtest](#3-final-backtest)

---

## Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install optuna pandas_ta plotly tabulate vectorbt
```

---

## 1. Parameter Sensitivity Analysis
**File:** `scripts/run_sensitivity_analysis.py`

Use this script to analyze how different parameter ranges affect the strategy's net profit. It uses **Global Optimization**, seeking parameters that perform well across all "Top Movers" simultaneously.

### How to Run
```bash
python scripts/run_sensitivity_analysis.py
```

### What it does:
- Samples parameter ranges for ADX, SAR, and ATR using Optuna.
- Runs a multi-asset backtest for each trial.
- Generates interactive Plotly visualizations (Parallel Coordinates, Slice plots) to show parameter importance and optimal regions.

---

## 2. Walk-Forward Optimization (WFO)
**File:** `scripts/run_walk_forward_optimization.py`

This script validates the strategy's robustness by simulating how it would have performed if parameters were re-optimized periodically.

### How to Run
```bash
python scripts/run_walk_forward_optimization.py
```

### Workflow:
1. **Train Window**: Optimizes parameters on a historical block (e.g., 90 days).
2. **Test Window**: Applies the best parameters to the *next* block of time (e.g., 30 days) that the optimizer didn't see.
3. **Roll**: Moves forward and repeats, aggregating results into a realistic performance report.

### Outputs:
- Performance summary for each "Out of Sample" window.
- Saves detailed results to `wfo_results.csv`.

---

## 3. Final Backtest
**File:** `scripts/run_final_backtest.py`

Run a single, high-detail backtest with specific parameters you've chosen (perhaps based on results from the Sensitivity Analysis).

### How to Run
```bash
python scripts/run_final_backtest.py
```

### Configuration:
To change the parameters, edit the `STRATEGY_PARAMS` dictionary at the top of the file:
```python
STRATEGY_PARAMS = {
    'adx_threshold': 25,
    'adx_length': 14,
    'sar_acceleration': 0.02,
    ...
}
```

### Outputs:
- Detailed portfolio metrics (Profit, Sharpe Ratio, Max Drawdown).
- Complete Trade History table.
- Breakdown of profit/loss per symbol.

---

## Troubleshooting
- **Missing Data**: If data is unavailable for certain symbols or dates, the engine will skip them and log a debug message.
- **Trial Failures**: If every trial fails with `-inf` or `nan`, check `error.log` for missing dependencies or data loading issues.
