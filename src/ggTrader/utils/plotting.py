"""Plotting utilities for backtesting and optimization results."""

import os
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import optuna
    import optuna.visualization as vis

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# =============================================================================
# Shared Utilities
# =============================================================================


def _is_jupyter_environment() -> bool:
    """Detects if the script is executing within a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and "IPKernelApp" in shell.config:
            return True
    except Exception:
        pass
    return False


def _display_interactive(fig: Any) -> None:
    """Routes figure display based on the detected runtime environment."""
    if _is_jupyter_environment():
        try:
            from IPython.display import display

            display(fig)
            if hasattr(fig, "savefig"):  # Clean up Matplotlib memory
                plt.close(fig)
            return
        except ImportError:
            pass

    # Fallback for standard terminals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            if hasattr(fig, "show"):
                fig.show()
            else:
                plt.show()
        except Exception as e:
            print(f"Could not display figure interactively: {e}")


def _save_or_show_figure(fig: Any, filename: str, results_manager: Optional[Any] = None) -> None:
    """Handles saving or interactive display for both Plotly and Matplotlib figures."""
    if results_manager:
        save_dir = Path(results_manager.run_dir) / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename

        if hasattr(fig, "write_image"):  # Plotly
            try:
                fig.write_image(str(path))
            except Exception as e:
                print(f"Failed to save Plotly image {filename}: {e}.")
                fig.write_html(str(path.with_suffix(".html")))
        elif hasattr(fig, "savefig"):  # Matplotlib
            fig.savefig(str(path), dpi=150)
            plt.close(fig)
        print(f"Saved plot to {path}")
    else:
        _display_interactive(fig)


# =============================================================================
# Optimization Landscape Plotting
# =============================================================================


def _plot_df_heatmap(
    df: pd.DataFrame,
    params_to_plot: List[str],
    metric_name: str,
    results_manager: Optional[Any],
) -> None:
    """Generates a static Seaborn heatmap from a Pandas DataFrame."""
    missing = [p for p in params_to_plot if p not in df.columns]
    if missing or metric_name not in df.columns:
        print(f"Missing parameters or metric in DataFrame: {missing}")
        return

    if len(params_to_plot) < 2:
        print("Need at least 2 parameters for heatmap.")
        return

    p1, p2 = params_to_plot[0], params_to_plot[1]
    pivot = df.pivot_table(index=p1, columns=p2, values=metric_name, aggfunc="mean")

    vals = pivot.to_numpy(dtype=float, copy=True)
    if pivot.empty or vals.size == 0:
        print(f"Skipping heatmap: empty pivot for {p1} vs {p2}.")
        return
    if not np.isfinite(vals).any():
        print(
            f"Skipping heatmap: no finite {metric_name} values "
            f"(e.g. all NaN after MIN_TRADES filter)."
        )
        return

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f")
    plt.title(f"Heatmap: {metric_name} by {p1} & {p2}")

    _save_or_show_figure(fig, f"heatmap_{p1}_{p2}.png", results_manager)


def _plot_optuna_study(
    study: "optuna.Study",
    params_to_plot: List[str],
    metric_name: str,
    results_manager: Optional[Any],
) -> None:
    """Generates standard Optuna visualization plots."""
    try:
        fig_parallel = vis.plot_parallel_coordinate(study, params=params_to_plot)
        fig_parallel.update_layout(title=f"Parallel Coordinates: {metric_name}")
        _save_or_show_figure(fig_parallel, "parallel_coordinates.png", results_manager)
    except Exception as e:
        print(f"Could not generate parallel coordinate plot: {e}")

    try:
        fig_slice = vis.plot_slice(study, params=params_to_plot)
        fig_slice.update_layout(title=f"Slice Plot: Sensitivity for {metric_name}")
        _save_or_show_figure(fig_slice, "slice_plot.png", results_manager)
    except Exception as e:
        print(f"Could not generate slice plot: {e}")

    if len(params_to_plot) >= 2:
        try:
            fig_contour = vis.plot_contour(study, params=params_to_plot[:2])
            fig_contour.update_layout(title=f"Contour: {params_to_plot[0]} vs {params_to_plot[1]}")
            _save_or_show_figure(
                fig_contour, f"contour_{params_to_plot[0]}_{params_to_plot[1]}.png", results_manager
            )
        except Exception as e:
            print(f"Could not generate contour plot: {e}")

        # Fallback to static heatmap using trials DataFrame
        trials_df = study.trials_dataframe()
        if not trials_df.empty and "value" in trials_df.columns:
            trials_df.columns = [c.replace("params_", "") for c in trials_df.columns]
            trials_df.rename(columns={"value": metric_name}, inplace=True)
            _plot_df_heatmap(trials_df, params_to_plot, metric_name, results_manager)


def plot_optimization_landscape(
    study_or_df: Any,
    params_to_plot: List[str] = ["adx_threshold", "atr_multiplier"],
    metric_name: str = "Sharpe Ratio",
    results_manager: Optional[Any] = None,
) -> None:
    """
    Generates sensitivity plots for analyzing optimization results.

    Args:
        study_or_df: The completed optimization study or results DataFrame.
        params_to_plot: List of parameter names to visualize.
        metric_name: Label for the objective value.
        results_manager: Instance to handle plot saving.
    """
    if isinstance(study_or_df, pd.DataFrame):
        _plot_df_heatmap(study_or_df, params_to_plot, metric_name, results_manager)
    elif OPTUNA_AVAILABLE and isinstance(study_or_df, optuna.Study):
        _plot_optuna_study(study_or_df, params_to_plot, metric_name, results_manager)
    else:
        print(f"Unsupported object type or Optuna missing: {type(study_or_df)}")


# =============================================================================
# Walk-Forward Optimization (WFO) Plotting
# =============================================================================


def _extract_close_prices(ohlcv_data: Any) -> pd.Series:
    """Extracts the chronological close price series from OHLCV data."""
    close = (
        ohlcv_data.xs("close", axis=1, level=1, drop_level=True)
        if isinstance(ohlcv_data.columns, pd.MultiIndex)
        else ohlcv_data["close"]
    )
    if isinstance(close, pd.DataFrame):
        return close.iloc[:, 0]
    return close


def _calculate_fold_indices(
    total_len: int, n_splits: int, train_test_ratio: float
) -> List[Tuple[int, int, int, int]]:
    """
    Calculates train and test boundary indices, ensuring no test overlap.
    Returns: List of tuples -> (train_start, train_end, test_start, test_end).
    """
    test_len = int(total_len / (train_test_ratio + n_splits))
    train_len = int(test_len * train_test_ratio)
    folds = []

    for i in range(n_splits):
        start_idx = i * test_len
        train_end_idx = start_idx + train_len
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_len

        # Allow the last test fold to absorb rounding remainders
        if i == n_splits - 1:
            test_end_idx = total_len

        # Bounds Verification
        assert start_idx < train_end_idx, f"Fold {i}: Train set is invalid."
        assert test_start_idx < test_end_idx, f"Fold {i}: Test set is invalid."
        assert train_end_idx == test_start_idx, f"Fold {i}: Train/Test gap."

        if i > 0:
            prev_test_end = folds[i - 1][3]
            assert test_start_idx == prev_test_end, f"Fold {i}: Test overlap."

        folds.append((start_idx, train_end_idx, test_start_idx, test_end_idx))

    return folds


def _draw_wfo_plot(
    close_series: pd.Series,
    folds: List[Tuple[int, int, int, int]],
    n_splits: int,
    train_test_ratio: float,
) -> plt.Figure:
    """Generates the Matplotlib figure mapping out the WFO folds."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (tr_start_idx, tr_end_idx, te_start_idx, te_end_idx) in enumerate(folds):
        y_pos = i + 1  # Fixed: Sets Fold 1 at the bottom and counts upward

        tr_start = close_series.index[tr_start_idx]
        tr_end = close_series.index[tr_end_idx - 1]
        te_start = close_series.index[te_start_idx]
        te_end = close_series.index[te_end_idx - 1]

        ax.hlines(
            y_pos, tr_start, tr_end, color="blue", linewidth=8, label="Train" if i == 0 else ""
        )
        ax.hlines(
            y_pos, te_start, te_end, color="orange", linewidth=8, label="Test" if i == 0 else ""
        )

        label_text = (
            f"Fold {y_pos}: {tr_start.strftime('%Y-%m-%d')} to {te_end.strftime('%Y-%m-%d')}"
        )
        ax.text(
            tr_start,
            y_pos + 0.15,
            label_text,
            verticalalignment="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        f"Walk-Forward Splits (Folds: {n_splits}, Ratio: {train_test_ratio}:1)",
        fontsize=14,
    )
    ax.set_xlabel("Date")
    ax.set_yticks([])
    ax.set_ylim(0.5, n_splits + 0.8)
    ax.legend(loc="upper left", fontsize="small", ncol=2)  # Moved to avoid overlapping Fold 1
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    return fig


def plot_wfo_splits(
    ohlcv_data: Any,
    n_splits: int = 5,
    train_test_ratio: float = 3.0,
    results_manager: Optional[Any] = None,
) -> None:
    """
    Plots Walk-Forward Optimization sliding window splits.

    Args:
        ohlcv_data (Any): Input market data containing a 'close' column.
        n_splits (int): Number of Walk-Forward folds.
        train_test_ratio (float): Ratio of training window size to test window size.
        results_manager (Optional[Any]): ResultsManager instance for saving.
    """
    close_series = _extract_close_prices(ohlcv_data)
    folds = _calculate_fold_indices(len(close_series), n_splits, train_test_ratio)
    fig = _draw_wfo_plot(close_series, folds, n_splits, train_test_ratio)
    _save_or_show_figure(fig, "wfo_splits.png", results_manager)
