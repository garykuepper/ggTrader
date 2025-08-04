import optuna
import numpy as np

threshold = 0.1

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_value = float('inf')
        self.counter = 0

    def __call__(self, study, trial):
        if study.best_value < self.best_value:
            self.best_value = study.best_value
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"No improvement for {self.patience} trials. Stopping.")
            study.stop()

def blackbox(x, y):
    return (x - 2)**2 + (y + 3)**2 + np.random.normal(0, 0.1)

def objective(trial):
    x = trial.suggest_int('x', -10, 10)
    y = trial.suggest_int('y', -10, 10)
    return blackbox(x, y)

def stop_when_reached(study, trial):
    if study.best_value <= threshold:
        print(f"Threshold reached: {study.best_value}")
        study.stop()

if __name__ == "__main__":

    early_stopping = EarlyStopping(patience=70)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100, n_jobs=8)

    print("Best value: ", study.best_value)
    print("Best params: ", study.best_params)