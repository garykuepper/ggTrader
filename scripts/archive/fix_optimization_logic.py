import pandas as pd
import vectorbt as vbt
import numpy as np

def select_best_parameters(trials_summary, best_params_keys):
    """
    Selects the best parameter set from trials_summary instead of averaging.
    
    Args:
        trials_summary (pd.DataFrame): DataFrame containing optimization trial results.
        best_params_keys (list): List of parameter names to extract.
        
    Returns:
        dict: A dictionary of the best parameter values.
    """
    # Sort by performance metric (e.g., Sharpe Ratio) descending
    # Ensure we get the absolute best run, not an average
    top_run = trials_summary.sort_values('Sharpe Ratio', ascending=False).iloc[0]
    
    selected_params = {}
    print(f"Selected Best Params from Trial #{top_run.name} with Sharpe: {top_run['Sharpe Ratio']:.4f}")
    
    for param in best_params_keys:
        selected_params[param] = top_run[param]
        
        # Type casting based on original types if needed
        # (e.g. ensure integers/booleans remain correct)
        if isinstance(top_run[param], (bool, np.bool_)):
            selected_params[param] = bool(top_run[param])
        elif isinstance(top_run[param], (int, np.integer)):
            selected_params[param] = int(top_run[param])
            
    return selected_params

# --- Example Usage Snippet for Notebook ---
# Replace the existing averaging loop with this:

"""
# ORIGINAL CODE (DON'T USE):
# top_runs = trials_summary.sort_values('Sharpe Ratio', ascending=False).groupby('symbol').head(top_10_pct)
# for param in best_params.keys():
#     params[param] = top_runs[param].mean()

# FIXED CODE:
top_run = trials_summary.sort_values('Sharpe Ratio', ascending=False).iloc[0]
print(f"Selected Best Params from Trial #{top_run.name} with Sharpe: {top_run['Sharpe Ratio']:.4f}")

for param in best_params.keys():
    params[param] = top_run[param]
"""
