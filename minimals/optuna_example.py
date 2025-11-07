import optuna
from optuna.samplers import GridSampler
from optuna.visualization import plot_slice, plot_contour, plot_optimization_history,plot_param_importances
from tabulate import tabulate
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Define the objective function
def objective(trial):
    # Example parameters to optimize
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    # Simple quadratic function (minimum at x=3, y=-2)
    return (x - 3) ** 2 + (y + 2) ** 2


# Define the grid to search
search_space = {
    "x": [-10, -5, 0, 5, 10],
    "y": [-10, -5, 0, 5, 10],
}

# Create a study with GridSampler
sampler = GridSampler(search_space)
study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective)

# Print results
print("Number of trials:", len(study.trials))
print("Best trial:")
print("  Value:", study.best_trial.value)
print("  Params:", study.best_trial.params)
trials = study.trials_dataframe().copy()
trials.sort_values("value", inplace=True)
print(tabulate(trials, headers="keys", tablefmt="github"))

# Plot optimization history
fig1 = plot_slice(study)
fig1.show()

fig2 = plot_contour(study)
fig2.show()

fig3 = plot_optimization_history(study)
fig3.show()

fig4 = plot_param_importances(study)
fig4.show()
