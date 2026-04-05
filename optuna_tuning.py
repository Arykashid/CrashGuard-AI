import optuna
import numpy as np

from lstm_model import build_lstm_model, train_model


def objective(trial, processed):

    X_train = processed["X_train"]
    y_train = processed["y_train"]

    X_val = processed["X_val"]
    y_val = processed["y_val"]

    window_size = X_train.shape[1]
    num_features = X_train.shape[2]

    forecast_horizon = y_train.shape[1]

    # ---- Hyperparameters to search ----

    units1 = trial.suggest_int("units1", 32, 128)

    units2 = trial.suggest_int("units2", 16, 64)

    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    batch = trial.suggest_categorical("batch_size", [16, 32, 64])

    # ---- Build model ----

    model = build_lstm_model(
        window_size=window_size,
        num_features=num_features,
        forecast_horizon=forecast_horizon,
        lstm_units=[units1, units2],
        dropout_rate=dropout,
        learning_rate=lr
    )

    # ---- Train ----

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=30,
        batch_size=batch
    )

    return min(history.history["val_loss"])


def run_optuna(processed, trials=20):

    study = optuna.create_study(direction="minimize")

    study.optimize(lambda trial: objective(trial, processed), n_trials=trials)

    return study