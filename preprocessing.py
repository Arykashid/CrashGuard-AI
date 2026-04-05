"""
Data Preprocessing Module (Research Grade – Time Series Forecasting)

FIX APPLIED:
  - Target variable now uses scaled values (same range as input)
  - Previously: input was scaled (0-1) but target was raw CPU values
  - This mismatch was causing LSTM to predict flat values and lose to ARIMA
  - Now: both input and target are in same scaled range → LSTM learns correctly

Features:
- Leakage-safe scaling
- Temporal feature engineering
- Lag features
- Rolling statistics
- Sliding window creation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ================= VALIDATION =================
def validate_schema(df):
    required = ["timestamp", "cpu_usage"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")


# ================= MISSING VALUES =================
def handle_missing_values(df):
    df = df.copy()
    df["cpu_usage"] = df["cpu_usage"].interpolate(method="linear")
    df["cpu_usage"] = df["cpu_usage"].ffill().bfill()
    return df


# ================= TIME FEATURES =================
def add_time_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)

    if df["timestamp"].nunique() == 1:
        df["hour"] = (df.index // 360) % 24
        df["day_of_week"] = (df.index // (360 * 24)) % 7
    else:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# ================= LAG FEATURES =================
def add_lag_features(df):
    df = df.copy()
    df["lag1"] = df["cpu_usage"].shift(1)
    df["lag5"] = df["cpu_usage"].shift(5)
    df["lag10"] = df["cpu_usage"].shift(10)
    df["lag2"] = df["cpu_usage"].shift(2)
    df["lag3"] = df["cpu_usage"].shift(3)
    df["roll_mean_10"] = df["cpu_usage"].rolling(10).mean()
    df["roll_std_10"]  = df["cpu_usage"].rolling(10).std()
    df = df.dropna()
    return df


# ================= SLIDING WINDOWS =================
def create_sliding_windows(data, target, window_size, horizon):
    X, y = [], []
    for i in range(window_size, len(data) - horizon + 1):
        X.append(data[i - window_size:i])
        y.append(target[i:i + horizon])
    return np.array(X), np.array(y)


# ================= MAIN PIPELINE =================
def prepare_data(
    df,
    window_size=60,
    forecast_horizon=1,
    train_ratio=0.7,
    val_ratio=0.15
):
    validate_schema(df)

    df = handle_missing_values(df)
    df = add_time_features(df)
    df = add_lag_features(df)

    feature_cols = [
        "cpu_usage",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "lag1",
        "lag5",
        "lag10",
        "lag2",
        "lag3",
        "roll_mean_10",
        "roll_std_10"
    ]

    values    = df[feature_cols].values
    n_total   = len(values)
    train_end = int(n_total * train_ratio)
    val_end   = int(n_total * (train_ratio + val_ratio))

    # Fit scaler on train only (leakage-safe)
    scaler = MinMaxScaler()
    scaler.fit(values[:train_end])
    scaled_values = scaler.transform(values)

    # FIX: target must be scaled — same range as input features
    # Old code used raw cpu values as target → input/output mismatch
    # → LSTM predicted flat values → lost to ARIMA
    # Now both input and target are in 0-1 range → LSTM learns correctly
    scaled_target = scaled_values[:, 0:1]  # cpu_usage is column 0

    X_all, y_all = create_sliding_windows(
        scaled_values,
        scaled_target,
        window_size,
        forecast_horizon
    )

    window_offset = window_size
    train_idx     = train_end - window_offset
    val_idx       = val_end   - window_offset

    X_train = X_all[:train_idx]
    y_train = y_all[:train_idx]
    X_val   = X_all[train_idx:val_idx]
    y_val   = y_all[train_idx:val_idx]
    X_test  = X_all[val_idx:]
    y_test  = y_all[val_idx:]

    # Separate cpu scaler for correct inverse transform in evaluate.py
    cpu_scaler = MinMaxScaler()
    cpu_scaler.fit(df[["cpu_usage"]].values[:train_end])

    return {
        "X_train":      X_train,
        "y_train":      y_train,
        "X_val":        X_val,
        "y_val":        y_val,
        "X_test":       X_test,
        "y_test":       y_test,
        "scaler":       scaler,
        "cpu_scaler":   cpu_scaler,
        "feature_names": feature_cols,
        "original_data": df.reset_index(drop=True),
        "num_features":  len(feature_cols)
    }