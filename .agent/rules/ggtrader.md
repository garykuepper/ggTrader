---
trigger: always_on
---

# Python Coding Standards (Global)

1. **Strict PEP 8 Adherence**:
    - Use 4 spaces for indentation.
    - Maximum line length of 88-100 characters (Black/Ruff style).
    - Snake_case for functions/variables, CamelCase for classes, UPPER_CASE for constants.
    - **Snake_case for filenames**: All Python files must be lowercase with underscores (e.g., `trading.py`, not `Trading.py`).
    - Imports grouped: Standard lib, Third-party, Local (sorted alphabetically).

2. **Concise Documentation**:
    - **Module Docstrings**: One-line summary of the file's purpose at the top.
    - **Function/Class Docstrings**: Required for all public interfaces. brief summary of purpose.
        - *Format*: Google Style or concise text. Avoid verbose parameter lists for obvious types.
    - **Inline Comments**:
        - Explain *WHY*, not *WHAT*.
        - Keep them short and on the line above the code block.
        - Avoid obvious comments (e.g., `# Increment i` for `i += 1`).

3. **Type Hinting**:
    - Use type hints for all function arguments and return values.
    - Use `from typing import Optional, List, Dict, Any, Tuple` or modern syntax (`list[]`, `|`) where supported.

4. **Structure**:
    - Constants defined at the module level (top), not inside functions unless necessary.
    - Main execution logic inside `if __name__ == "__main__":` block.
    - **Single Source of Truth**: Core logic must live in `src/`. Do not duplicate definitions across files.
    - **Orchestration Boilerplate**: Scripts in `scripts/` must follow a standard template: `sys.path` setup, `main()` orchestration, and `argparse` for inputs. Every script must handle the `--help` flag gracefully.
    - **Results Management**: Use `ResultsManager` for all artifact, metric, and plot saving to ensure consistent output directory structures and metadata logging.

## ggTrader Project Specific Rules

1. **Vectorization First**:
    - Avoid iterating over rows in DataFrames for signal calculation.
    - Use `vectorbt`, `numpy`, or `pandas` vectorized operations for speed.
    - Use `SignalFactory.run()` for all indicator logic to support parameter broadcasting (grid search).

2. **Database as Source of Truth**:
    - Do not read raw CSVs directly in strategy code.
    - Use **TimescaleDB** (via `KrakenPostgresReader`) for all time-series and historical data.
    - Use `ResultDBManager` to save backtest/optimization results.

3. **Path Safety**:
    - Always use `os.path.join` or `pathlib.Path` for file paths.
    - Project root should be dynamically resolved relative to `src` or `scripts`, not hardcoded.

4. **Configuration & Data Loading**:
    - Use `CONSTANTS` dictionaries at module level for default configuration.
    - **Standardized Loading**: Always use `load_data_and_setup(CONSTANTS)` for data retrieval in scripts and notebooks.
    - **Flexible Symbols**: Support both `SYMBOLS` (direct list) and `SYMBOLS_FILE` (JSON path) in `CONSTANTS` to allow seamless switching between asset pools.

5. **Dependency Discipline**:
    - Update `pyproject.toml` immediately when adding new libraries.
    - Avoid `pip install` without recording the dependency.

6. **Import Strategy**:
    - Use absolute imports (`from ggTrader.core...`) rooted in `src` for cross-package access.
    - **Relative Imports in Init**: Always use relative imports within package `__init__.py` files (e.g., `from .module import Class`).
    - Avoid circular imports by careful module design (e.g. `utils` vs `core`).

7. **Resilient Scripts**:
    - Long-running scripts (ingestion/optimization) must handle exceptions (log & continue, do not crash).
    - Implement resumability (checkpoints) for any process taking >5 minutes.

## Jupyter Notebook Standards

1. **Imports from `src`**:
    - Notebooks must import core logic and indicators from `src`.
    - Do not define complex strategy classes or large functions inline.
    - Notebooks are for **orchestration**, **analysis**, and **visualization** only.

2. **Path Setup**:
    - Always include the standard `sys.path` setup block at the top to resolve project root.
    - Use `os.path.join` to locate files relative to the project root, not absolute paths.

3. **Sequential Execution**:
    - Notebooks must be runnable from top to bottom without errors.
    - Avoid relying on hidden state (out-of-order execution).

## Documentation Standards

1. **Single Source of Truth**:
    - Core architectural changes (e.g., database transitions) must be reflected across all documentation in `docs/` and `README.md`.
    - Supplemental documentation must reside in the `docs/` directory.

2. **Project Structure Alignment**:
    - Documentation must accurately reflect the directory structure (e.g., `src/`, `scripts/`, `docs/`).
    - Use relative links for cross-referencing between documentation files.

# ggTrader Project Consistency Rules

These rules ensure the codebase remains maintainable and that data artifacts (JSON, CSV, DB) are consistently formatted.

## 1. Symbol Normalization

All scripts, notebooks, and models MUST use standardized asset symbols (e.g., `BTC`, `ETH`).

- **Input**: Database or raw API data may contain Kraken-specific prefixed symbols (e.g., `XBT`, `XETH`, `XXLM`, `ZUSD`).
- **Processing**: Use the `SYMBOL_MAPPING` table (found in `generate_asset_pool.py` and consolidated in `data_manager.py`) to convert internal labels to standard tickers.
- **Output**: Any artifact saved to `data/` or results returned to the user must use the normalized symbol.

## 2. SQL Query Patterns

When querying the `ohlcv` table, always use standard PostgreSQL/TimescaleDB functions for asset separation:

- **Asset Part**: `split_part(symbol, '-', 1)`
- **Quote Part**: `split_part(symbol, '-', 2)`
- **Aggregation**: Prefer Notional Volume (`volume * close`) for cross-asset ranking to account for price discrepancies.

## 3. Data Persistence

Results, asset pools, and backtest metrics must follow these directory and naming conventions:

- **Pools**: `data/top_{N}_{QUOTE}_{DAYS}_movers.json`
- **Backtest Results**: Use the `ResultsManager` class to ensure timestamped and structured logs.
- **Relative Paths**: Always resolve paths relative to the project root or use `os.path.join`. Never hardcode absolute paths starting with `C:\`.

## 4. Error Handling & Resiliency

- **Fail Fast**: Data loading functions (like `load_data_and_setup`) must raise descriptive `ValueError` or `FileNotFoundError` exceptions if inputs are missing.
- **No None-Returns**: Avoid returning `None` from functions that are expected to return iterables (DataFrames, lists) to prevent "NoneType is not iterable" errors. Handle empty data by returning empty structures or raising exceptions.

## 5. Backtesting Architecture

- **FastBacktest** is the only engine for backtesting, sensitivity analysis, and WFO.
- Always pass `config=CONSTANTS` for portfolio-level settings; keep signal `params` separate.
- Use `cash_sharing=True` and `group_by=True` — never independent cash pools per symbol.
- The `Trading` engine is legacy (paper/live trading only). `WalkForwardOptimizer` (Optuna) is archived.

## 6. Dynamic Mover Masking

- Use `build_mover_mask(ohlcv, config, top_n=N)` for daily top-N mover masks.
- Pass masks to `FastBacktest(mover_mask=mask)` to zero out entries for non-qualifying symbols.
- Masks are precomputed via a single SQL query (`get_daily_mover_mask`), not per-bar.
