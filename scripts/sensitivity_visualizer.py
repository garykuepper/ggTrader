import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_optimization_landscape(study, params_to_plot=['adx_threshold', 'atr_multiplier'], metric_name='Sharpe Ratio'):
    """
    Generates sensitivity plots for analyzing optimization results.
    
    Args:
        study (optuna.Study): The completed optimization study.
        params_to_plot (list): List of parameter names to visualize (e.g., ['adx_threshold', 'atr_multiplier']).
        metric_name (str): Label for the objective value.
    """
    
    # 1. Parallel Coordinate Plot (Good for seeing flow of high-perf trials)
    fig_parallel = vis.plot_parallel_coordinate(study, params=params_to_plot)
    fig_parallel.update_layout(title=f"Parallel Coordinates: Impact on {metric_name}")
    fig_parallel.show()

    # 2. Slice Plot (Individual parameter impact)
    fig_slice = vis.plot_slice(study, params=params_to_plot)
    fig_slice.update_layout(title=f"Slice Plot: Parameter Sensitivity for {metric_name}")
    fig_slice.show()

    # 3. Contour Plot (2D Interaction heatmap)
    if len(params_to_plot) >= 2:
        fig_contour = vis.plot_contour(study, params=params_to_plot[:2])
        fig_contour.update_layout(title=f"Contour Plot: {params_to_plot[0]} vs {params_to_plot[1]}")
        fig_contour.show()

    # 4. Custom Static Heatmap (Pandas/Seaborn)
    # Extracts trial data for custom analysis
    trials_df = study.trials_dataframe()
    if trials_df.empty:
        print("No trials found in study.")
        return

    if 'value' in trials_df.columns:
        # Clean column names (remove 'params_' prefix)
        trials_df.columns = [col.replace('params_', '') for col in trials_df.columns]
        
        # Pivot table for heatmap (example for 2 params)
        if len(params_to_plot) >= 2:
            p1 = params_to_plot[0]
            p2 = params_to_plot[1]
            
            # Binning continuous data if necessary
            pivot = trials_df.pivot_table(index=p1, columns=p2, values='value', aggfunc='mean')
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, cmap='viridis', annot=True, fmt=".2f")
            plt.title(f"Heatmap: {metric_name} by {p1} & {p2}")
            plt.show()

# Example Usage:
# study = optuna.create_study(...)
# study.optimize(...)
# plot_optimization_landscape(study, ['adx_threshold', 'atr_multiplier', 'sar_acceleration'])
