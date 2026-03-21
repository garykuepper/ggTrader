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

Copy the `.env.example` file (if provided) to `.env` and fill in your credentials:

```bash
# Example .env entries
KRAKEN_API_KEY=your_key
KRAKEN_SECRET_KEY=your_secret
DATABASE_URL=postgresql://user:password@localhost:5433/ggtrader
```

## 🚀 Running ggTrader

### Standard Backtest

Run a backtest using the vectorized `FastBacktest` engine:

```bash
python scripts/run_backtest.py
```

### Dynamic Mover Backtest

Test with the top-N daily movers mask for dynamic universe filtering:

```bash
python scripts/run_backtest.py --movers 20
```

### Parameter Optimization

Execute Walk-Forward Optimization to find the most robust parameters:

```bash
python scripts/run_walk_forward_optimization.py
```

### Sensitivity Analysis

Check the stability of your optimized parameters:

```bash
python scripts/run_sensitivity_analysis.py
```

## Docker Compose Setup

### 1. Set up Database

- Run the database container:

     ```bash
     docker-compose up -d
     ```

- This starts a TimescaleDB instance on port 5433 with default credentials (`gary_admin`/`your_secure_password`).
- The application is configured to connect to this instance by default.

## 🐳 External Dependencies

- **PostgreSQL (TimescaleDB)**: Primary database for OHLCV data.
- **Results DB**: Stores backtest runs and WFO results in Postgres.
- **VectorBT**: Core backtesting engine (via `FastBacktest`).

---
*Back to [README.md](../readme.md)*
