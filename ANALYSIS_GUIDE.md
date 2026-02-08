# ggTrader Analysis Guide

This guide describes the workflow for optimizing and validating trading strategies.

## Workflow: WFO to Backtest

1. **Run Walk-Forward Optimization (WFO)**:
   ```bash
   python scripts/run_walk_forward_optimization.py
   ```
   This script will save results in `results/run_wfo_YYYYMMDD_HHMMSS/`.
   The best overall parameters will be exported to `params.json` within that folder.

2. **Validate with Backtest**:
   Transfer the optimal parameters to a full backtest:
   ```bash
   python scripts/backtest/run_backtest.py --params results/run_wfo_XXXXXX/params.json
   ```

## Sensitivity Analysis
Check if your strategy is robust to small parameter changes:
```bash
python scripts/run_sensitivity_analysis.py
```
View generated heatmaps and contour plots in the `results/run_sensitivity_XXXXXX/plots/` directory.

## Results Management
Every run creates a timestamped folder in `results/`:
- `run_metadata.json`: Parameters used, symbols, and dates.
- `metrics.csv` / `trade_history.csv`: Performance data.
- `plots/`: Visualizations (equity curves, sensitivity maps).

## Notebook Integration
Use the notebooks in `notebooks/` for interactive exploration. Ensure you update imports using `scripts/update_notebook_imports.py` if you add new modules.
