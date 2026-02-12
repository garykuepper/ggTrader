# Installation & Execution Guide

Follow these steps to set up `ggTrader` and run your first backtest.

## 🛠️ Installation

### 1. Prerequisites

- **Python 3.8+**
- **Git**
- **Virtual Environment** (recommended)

### 2. Clone the Repository

```bash
git clone https://github.com/garykuepper/ggTrader.git
cd ggTrader
```

### 3. Setup Environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies in editable mode:

```bash
pip install -e .
```

### 4. GPU Acceleration (Optional but Recommended)

For high-performance backtesting with VectorBT, it is recommended to have a CUDA-enabled GPU and install `cupy`:

```bash
pip install cupy-cuda13x  # Replace '13x' with your CUDA version (e.g., 11x, 12x)
```

## ⚙️ Configuration

Copy the `.env.example` file (if provided) to `.env` and fill in your API credentials:

```bash
# Example .env entries
KRAKEN_API_KEY=your_key
KRAKEN_SECRET_KEY=your_secret
```

## 🚀 Running ggTrader

### Standard Backtest

Run a backtest for a specific symbol or set of symbols defined in your config:

```bash
python scripts/backtest/run_backtest.py
```

### Parameter Optimization

Execute Walk-Forward Optimization to find the most robust parameters over a historical period:

```bash
python scripts/run_walk_forward_optimization.py
```

### Sensitivity Analysis

Check the stability of your optimized parameters:

```bash
python scripts/run_sensitivity_analysis.py
```

## 🐳 External Dependencies

- **DuckDB**: Used for local data storage. The database files (`ggtrader.db`, `daily_movers.db`) will be created automatically in the root or `data/` directory upon ingestion.
- **VectorBT**: Core backtesting engine.

---
*Back to [README.md](readme.md)*
