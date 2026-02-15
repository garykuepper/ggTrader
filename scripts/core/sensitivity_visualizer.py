import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
from pathlib import Path


def plot_optimization_landscape(
    study_or_df,
    params_to_plot=["adx_threshold", "atr_multiplier"],
    metric_name="Sharpe Ratio",
    results_manager=None,
):
    """
    Generates sensitivity plots for analyzing optimization results.

    Args:
        study_or_df (optuna.Study or pd.DataFrame): The completed optimization study or results DataFrame.
        params_to_plot (list): List of parameter names to visualize.
        metric_name (str): Label for the objective value.
        results_manager (ResultsManager, optional): Instance to handle plot saving.
    """

    # helper to handle showing or saving
    def handle_plot(fig, filename):
        if results_manager:
            # sys.path.append(
            #     os.path.abspath(
            #         os.path.join(os.path.dirname(__file__), "..", "..", "src")
            #     )
            # )
            path = results_manager.get_plot_path(filename)
            if hasattr(fig, "write_image"):  # Plotly
                # Default plotly save (requires kaleido)
                try:
                    fig.write_image(str(path))
                except Exception as e:
                    print(
                        f"Failed to save Plotly image {filename}: {e}. Make sure 'kaleido' is installed."
                    )
                    # Fallback to HTML if image saving fails
                    html_path = path.with_suffix(".html")
                    fig.write_html(str(html_path))
            elif hasattr(fig, "savefig"):  # Matplotlib
                fig.savefig(str(path))
                plt.close(fig)
        else:
            if hasattr(fig, "show"):
                fig.show()

    # Distinguish between Optuna Study and DataFrame
    if isinstance(study_or_df, pd.DataFrame):
        df = study_or_df
        # Assuming df has columns for params and the metric
        # Check if params exist
        missing = [p for p in params_to_plot if p not in df.columns]
        if missing:
            print(f"Missing parameters in DataFrame: {missing}")
            return

        if metric_name not in df.columns:
            # Try to guess metric if not found? Or just print warning
            print(f"Metric '{metric_name}' not in DataFrame columns: {df.columns}")
            return

        # 1. Custom Static Heatmap (Pandas/Seaborn)
        if len(params_to_plot) >= 2:
            p1 = params_to_plot[0]
            p2 = params_to_plot[1]

            # Pivot for heatmap
            # Aggregating by mean in case of duplicates
            pivot = df.pivot_table(
                index=p1, columns=p2, values=metric_name, aggfunc="mean"
            )

            fig_sns = plt.figure(figsize=(12, 10))
            sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f")
            plt.title(f"Heatmap: {metric_name} by {p1} & {p2}")
            handle_plot(fig_sns, f"heatmap_{p1}_{p2}.png")
        else:
            print("Need at least 2 parameters for heatmap.")

        # Can add 3D scatter or other plots here for DataFrame
        pass

    elif isinstance(study_or_df, optuna.Study):
        study = study_or_df
        # 1. Parallel Coordinate Plot
        try:
            fig_parallel = vis.plot_parallel_coordinate(study, params=params_to_plot)
            fig_parallel.update_layout(
                title=f"Parallel Coordinates: Impact on {metric_name}"
            )
            handle_plot(fig_parallel, "parallel_coordinates.png")
        except Exception as e:
            print(f"Could not generate parallel coordinate plot: {e}")

        # 2. Slice Plot
        try:
            fig_slice = vis.plot_slice(study, params=params_to_plot)
            fig_slice.update_layout(
                title=f"Slice Plot: Parameter Sensitivity for {metric_name}"
            )
            handle_plot(fig_slice, "slice_plot.png")
        except Exception as e:
            print(f"Could not generate slice plot: {e}")

        # 3. Contour Plot
        if len(params_to_plot) >= 2:
            try:
                fig_contour = vis.plot_contour(study, params=params_to_plot[:2])
                fig_contour.update_layout(
                    title=f"Contour Plot: {params_to_plot[0]} vs {params_to_plot[1]}"
                )
                handle_plot(
                    fig_contour, f"contour_{params_to_plot[0]}_{params_to_plot[1]}.png"
                )
            except Exception as e:
                print(f"Could not generate contour plot: {e}")

        # 4. Custom Static Heatmap (Pandas/Seaborn)
        trials_df = study.trials_dataframe()
        if trials_df.empty:
            print("No trials found in study.")
            return

        if "value" in trials_df.columns:
            # Clean up column names "params_x" -> "x"
            trials_df.columns = [
                col.replace("params_", "") for col in trials_df.columns
            ]
            trials_df.rename(columns={"value": metric_name}, inplace=True)

            if len(params_to_plot) >= 2:
                p1 = params_to_plot[0]
                p2 = params_to_plot[1]

                pivot = trials_df.pivot_table(
                    index=p1, columns=p2, values=metric_name, aggfunc="mean"
                )

                fig_sns = plt.figure(figsize=(12, 10))
                sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f")
                plt.title(f"Heatmap: {metric_name} by {p1} & {p2}")
                handle_plot(fig_sns, f"heatmap_{p1}_{p2}.png")
