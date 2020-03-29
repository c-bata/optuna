import time

import optuna
from optuna.signal_handler import handle_kill_signal


def objective(trial):
    x = trial.suggest_uniform("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    time.sleep(4)
    return x ** 2 + y


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, timeout=10)
    for t in study.trials:
        print(t)
