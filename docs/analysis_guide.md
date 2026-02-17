# ggTrader Analysis Guide

This guide describes the workflow for optimizing and validating trading strategies.

## Workflow: WFO to Backtest

1. **Run Walk-Forward Optimization (WFO)**:

   ```bash
   python scripts/core/run_walk_forward_optimization.py
   ```

   This script will save results in `results/run_wfo_YYYYMMDD_HHMMSS/`.
   The best overall parameters will be exported to `params.json` within that folder.

2. **Validate with Backtest**:
   Transfer the optimal parameters to a full backtest:

   ```bash
   python scripts/core/run_backtest.py --params results/run_wfo_XXXXXX/params.json
   ```

3. **Dynamic Mover Backtest** (optional):
   Test with the top-N daily movers mask to simulate dynamic universe filtering:

   ```bash
   python scripts/core/run_backtest.py --params results/run_wfo_XXXXXX/params.json --movers 20
   ```

## Sensitivity Analysis

Check if your strategy is robust to small parameter changes:

```bash
python scripts/core/run_sensitivity_analysis.py
```

View generated heatmaps and contour plots in the `results/run_sensitivity_XXXXXX/plots/` directory.

## Engine Notes

All three workflows use `FastBacktest` with `config=CONSTANTS`:

- **Position sizing**: `PORTFOLIO_SHARE` controls per-trade allocation (shared capital pool).
- **Signal params**: Passed separately, support list values for broadcasting grids.
- **Mover mask**: Optional `--movers N` flag on `run_backtest.py` for daily top-N filtering.

## Results Management

Every run creates a timestamped folder in `results/`:

- `run_metadata.json`: Parameters used, symbols, and dates.
- `metrics.csv` / `trade_history.csv`: Performance data.
- `plots/`: Visualizations (equity curves, sensitivity maps).

## Notebook Integration

Use the notebooks in `notebooks/` for interactive exploration. They import core logic from `src/` and are for **orchestration, analysis, and visualization** only.

---
*Back to [README.md](../readme.md)*
