"""
Ablation Study Module
Systematically evaluates the contribution of each feature group.
Research-grade experiment showing WHY the model works.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ================= FEATURE GROUPS =================
ABLATION_CONFIGS = {
    "A — LSTM Base (cpu_usage only)": [
        "cpu_usage"
    ],
    "B — + Time Features": [
        "cpu_usage",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos"
    ],
    "C — + Lag Features": [
        "cpu_usage",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "lag1", "lag5", "lag10"
    ],
    "D — + Rolling Statistics (Full)": [
        "cpu_usage",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "lag1", "lag5", "lag10",
        "roll_mean_10", "roll_std_10"
    ]
}


# ================= SLIDING WINDOWS =================
def create_windows(data, target, window_size, horizon):
    X, y = [], []
    for i in range(window_size, len(data) - horizon + 1):
        X.append(data[i - window_size:i])
        y.append(target[i:i + horizon])
    return np.array(X), np.array(y)


# ================= SINGLE ABLATION RUN =================
def run_ablation_experiment(df, feature_cols, window_size=60,
                             forecast_horizon=1, epochs=30,
                             train_ratio=0.7, val_ratio=0.15):
    """
    Trains a small LSTM using only the specified feature_cols.
    Returns RMSE and MAE on the test set.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping

    # ---- Prepare features ----
    df = df.copy()

    # Add needed columns if missing
    if "hour_sin" not in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df.index / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df.index / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.index / 7)

    if "lag1" not in df.columns:
        df["lag1"] = df["cpu_usage"].shift(1)
        df["lag5"] = df["cpu_usage"].shift(5)
        df["lag10"] = df["cpu_usage"].shift(10)

    if "roll_mean_10" not in df.columns:
        df["roll_mean_10"] = df["cpu_usage"].rolling(10).mean()
        df["roll_std_10"] = df["cpu_usage"].rolling(10).std()

    df = df.dropna()

    # Only use specified feature columns
    available = [c for c in feature_cols if c in df.columns]
    values = df[available].values
    target = df[["cpu_usage"]].values

    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Leakage-safe scaling
    scaler = MinMaxScaler()
    scaler.fit(values[:train_end])
    scaled = scaler.transform(values)

    cpu_scaler = MinMaxScaler()
    cpu_scaler.fit(target[:train_end])

    X_all, y_all = create_windows(scaled, target, window_size, forecast_horizon)

    train_idx = train_end - window_size
    val_idx = val_end - window_size

    X_train = X_all[:train_idx]
    y_train = y_all[:train_idx]
    X_val = X_all[train_idx:val_idx]
    y_val = y_all[train_idx:val_idx]
    X_test = X_all[val_idx:]
    y_test = y_all[val_idx:]

    if len(X_train) < 10 or len(X_test) < 5:
        return None

    num_features = X_train.shape[2]

    # ---- Build small LSTM ----
    tf.random.set_seed(42)
    np.random.seed(42)

    model = Sequential([
        LSTM(32, input_shape=(window_size, num_features), return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(forecast_horizon)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # ---- Evaluate ----
    pred_scaled = model.predict(X_test, verbose=0)
    pred = cpu_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    true = y_test.flatten()

    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))

    return {"rmse": rmse, "mae": mae, "n_features": len(available)}


# ================= FULL ABLATION STUDY =================
def run_full_ablation(df, window_size=60, forecast_horizon=1, epochs=30):
    """
    Runs all 4 ablation experiments.
    Returns a DataFrame with results.
    """

    results = []

    for experiment_name, feature_cols in ABLATION_CONFIGS.items():
        print(f"Running: {experiment_name}...")

        try:
            result = run_ablation_experiment(
                df,
                feature_cols,
                window_size=window_size,
                forecast_horizon=forecast_horizon,
                epochs=epochs
            )

            if result is not None:
                results.append({
                    "Experiment": experiment_name,
                    "Features Used": ", ".join(feature_cols),
                    "Num Features": result["n_features"],
                    "RMSE": round(result["rmse"], 6),
                    "MAE": round(result["mae"], 6),
                })
            else:
                results.append({
                    "Experiment": experiment_name,
                    "Features Used": ", ".join(feature_cols),
                    "Num Features": len(feature_cols),
                    "RMSE": float("nan"),
                    "MAE": float("nan"),
                })

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "Experiment": experiment_name,
                "Features Used": ", ".join(feature_cols),
                "Num Features": len(feature_cols),
                "RMSE": float("nan"),
                "MAE": float("nan"),
            })

    df_results = pd.DataFrame(results)

    # Add improvement column
    if not df_results["RMSE"].isna().all():
        baseline = df_results["RMSE"].iloc[0]
        df_results["RMSE Improvement"] = df_results["RMSE"].apply(
            lambda x: f"{((baseline - x) / baseline * 100):+.1f}%" if x == x else "N/A"
        )

    return df_results
