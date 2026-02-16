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
