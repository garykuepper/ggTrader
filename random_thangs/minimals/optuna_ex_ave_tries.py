import optuna
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def blackbox(x, y):
    return (x - 2)**2 + (y + 3)**2
    # return (x - 2)**2 + (y + 3)**2 + np.random.normal(0, 0.1)

def objective(trial):
    x = trial.suggest_int('x', -10, 10)
    y = trial.suggest_int('y', -10, 10)
    return blackbox(x, y)

def get_best_trial_number(study):
    best_trial = study.best_trial
    return best_trial.number + 1

def average_trials_to_best(n_runs=100, n_trials=100):
    trial_numbers = []
    for _ in range(n_runs):
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=8)
        trial_numbers.append(get_best_trial_number(study))
    avg_trials = np.mean(trial_numbers)
    print(f"Average trial number to find best params over {n_runs} runs: {avg_trials:.2f}")
    return trial_numbers

if __name__ == "__main__":
    trial_numbers = average_trials_to_best(n_runs=100, n_trials=100)
    bin_width = 5
    bins = np.arange(1, max(trial_numbers) + bin_width + 1, bin_width)
    # plt.figure()
    # plt.hist(trial_numbers, bins=bins, edgecolor='black', align='left')
    # plt.xlabel('Trial number where best params found')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of trial numbers for best params')
    # plt.show()
    plt.figure()
    # trial_numbers is your data
    plt.hist(trial_numbers, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black')

    # Fit a normal distribution to the data
    mu, std = norm.fit(trial_numbers)

    # Generate x values for the bell curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    # Plot the bell curve
    plt.plot(x, p, 'k', linewidth=2)
    plt.title(f"Fit results: mu = {mu:.2f},  std = {std:.2f}")
    plt.xlabel('Trial number where best params found')
    plt.ylabel('Density')
    plt.show()