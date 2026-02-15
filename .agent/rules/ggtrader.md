---
trigger: always_on
---


1. **Vectorization First**:
    - Avoid iterating over rows in DataFrames for signal calculation.
    - Use `vectorbt`, `numpy`, or `pandas` vectorized operations for speed.
    - Use `SignalFactory.run()` for all indicator logic to support parameter broadcasting (grid search).

2. **Database as Source of Truth**:
    - Do not read raw CSVs directly in strategy code.
    - Use `KrakenHistoricalData` or `KrakenPostgresReader` to fetch data.
    - Use `ResultDBManager` to save backtest/optimization results.

3. **Path Safety**:
    - Always use `os.path.join` or `pathlib.Path` for file paths.
    - Project root should be dynamically resolved relative to `src` or `scripts`, not hardcoded.

4. **Configuration**:
    - Use `CONSTANTS` dictionaries at module level for default configuration.

5. **Dependency Discipline**:
    - Update `pyproject.toml` immediately when adding new libraries.
    - Avoid `pip install` without recording the dependency.

6. **Import Strategy**:
    - Use absolute imports (`from ggTrader.core...`) rooted in `src`.
    - Avoid circular imports by careful module design (e.g. `utils` vs `core`).

7. **Resilient Scripts**:
    - Long-running scripts (ingestion/optimization) must handle exceptions (log & continue, do not crash).
    - Implement resumability (checkpoints) for any process taking >5 minutes.
