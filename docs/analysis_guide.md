# ggTrader Analysis Guide

Short workflow notes. **Authoritative command-line detail** for the full multi-coin pipeline (phases, flags, outputs) is in the [**Strategy Pipeline Guide**](strategy_pipeline_guide.md).

## Workflow: WFO → Backtest

1. **Walk-Forward Optimization (WFO)**:

   ```bash
   python scripts/run_walk_forward_optimization.py
   ```

   Optional: `--mode per_coin`, `--no-progress`. Results under `results/run_wfo_*` (or per-coin naming from the orchestrator).

2. **Validate with a full backtest** (load params from a WFO run folder):

   ```bash
   python scripts/run_backtest.py --params results/run_wfo_XXXXXX/params.json
   ```

3. **Optional: fixed symbol list or mover mask** on the same backtest:

   ```bash
   python scripts/run_backtest.py --symbols BTC-USD ETH-USD --movers 20
   ```

   `--movers 0` turns the mask off. Defaults for dates, symbol file, and fees live in [`ggTrader.utils.run_config`](../src/ggTrader/utils/run_config.py) (`backtest_script_config()`); override by editing that helper or the script if you need a permanent change.

## Sensitivity Analysis

```bash
python scripts/run_sensitivity_analysis.py
```

Use `--no-progress` when you do not want tqdm/VectorBT progress output. Plots: `results/run_sensitivity_*/plots/`.

## Engine Notes

- **Config**: Portfolio-level settings are the orchestrator `config` dict; signal parameters are passed separately and may use list values for grids.
- **Mover mask**: `USE_MOVERS` / `--movers N` builds a daily top-*N* mask when `N > 0`.

## Results Layout

Timestamped folders under `results/` typically include metadata, metrics, trades, and `plots/` where applicable. See the pipeline guide for pipeline-specific artifacts (`pipeline_report.md`, `status.txt`, etc.).

## Notebooks

Notebooks under `notebooks/` are for exploration and charts; core logic stays in `src/ggTrader`.

---
*Back to [README.md](../readme.md)*
