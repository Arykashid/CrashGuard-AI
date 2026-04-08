"""
Data Preprocessing Module — CrashGuard AI

FIXES vs previous version:
  1. Added spike_flag feature
     (explicit signal that current value is statistically high
      → model gets early warning before a spike → spike RMSE drops)

  2. Added cpu_diff1, cpu_diff3 features
     (velocity and momentum of CPU — critical for spike detection
      → LSTM with only lag values can't tell if CPU is rising fast)

  3. N_FEATURES: 12 → 15
     (update live_monitor.py N_FEATURES constant to match)

Features (15 total, in exact order):
    0:  cpu_usage
    1:  hour_sin
    2:  hour_cos
    3:  dow_sin
    4:  dow_cos
    5:  lag1
    6:  lag5
    7:  lag10
    8:  lag2
    9:  lag3
    10: roll_mean_10
    11: roll_std_10
    12: spike_flag      ← NEW: 1 if cpu > rolling_mean + 2*rolling_std
    13: cpu_diff1       ← NEW: 1-step rate of change (velocity)
    14: cpu_diff3       ← NEW: 3-step rate of change (momentum)

WHY THESE 3 NEW FEATURES:
    Without them: LSTM sees cpu_usage=0.35 at t-1, t-2, t-3...
                  It cannot tell if CPU is about to spike or just stable high
    With them:    spike_flag=1 + cpu_diff1=+0.08 → "rising fast, spike forming"
                  → model can learn the spike signature explicitly
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# IMPORTANT: keep this in sync with live_monitor.py
N_FEATURES = 15


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
        df["hour"]        = (df.index // 360) % 24
        df["day_of_week"] = (df.index // (360 * 24)) % 7
    else:
        df["hour"]        = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# ================= LAG + SPIKE FEATURES =================
def add_lag_features(df):
    df = df.copy()

    # ── Lag features ──────────────────────────────────────────────
    df["lag1"]  = df["cpu_usage"].shift(1)
    df["lag2"]  = df["cpu_usage"].shift(2)
    df["lag3"]  = df["cpu_usage"].shift(3)
    df["lag5"]  = df["cpu_usage"].shift(5)
    df["lag10"] = df["cpu_usage"].shift(10)

    # ── Rolling statistics ─────────────────────────────────────────
    df["roll_mean_10"] = df["cpu_usage"].rolling(10).mean()
    df["roll_std_10"]  = df["cpu_usage"].rolling(10).std()

    # ── NEW: Spike flag ────────────────────────────────────────────
    # 1 if current cpu is statistically high (> mean + 2*std in a 20-step window)
    # Gives the model an explicit "spike is forming" signal
    roll_mean_20 = df["cpu_usage"].rolling(20).mean()
    roll_std_20  = df["cpu_usage"].rolling(20).std()
    df["spike_flag"] = (
        df["cpu_usage"] > (roll_mean_20 + 2.0 * roll_std_20)
    ).astype(float)

    # ── NEW: Rate of change features ──────────────────────────────
    # cpu_diff1: how much CPU changed in 1 step (velocity)
    # cpu_diff3: how much CPU changed in 3 steps (momentum)
    # Positive = rising, Negative = falling
    # These are the key signals for catching spikes early
    df["cpu_diff1"] = df["cpu_usage"].diff(1)
    df["cpu_diff3"] = df["cpu_usage"].diff(3)

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

    # ── Feature columns — ORDER MUST MATCH live_monitor.py ────────
    feature_cols = [
        "cpu_usage",        # 0
        "hour_sin",         # 1
        "hour_cos",         # 2
        "dow_sin",          # 3
        "dow_cos",          # 4
        "lag1",             # 5
        "lag5",             # 6
        "lag10",            # 7
        "lag2",             # 8
        "lag3",             # 9
        "roll_mean_10",     # 10
        "roll_std_10",      # 11
        "spike_flag",       # 12 ← NEW
        "cpu_diff1",        # 13 ← NEW
        "cpu_diff3",        # 14 ← NEW
    ]

    assert len(feature_cols) == N_FEATURES, (
        f"Feature count mismatch: {len(feature_cols)} != N_FEATURES={N_FEATURES}"
    )

    values  = df[feature_cols].values
    n_total = len(values)
    train_end = int(n_total * train_ratio)
    val_end   = int(n_total * (train_ratio + val_ratio))

    # ── Fit scaler on train only (leakage-safe) ────────────────────
    scaler = MinMaxScaler()
    scaler.fit(values[:train_end])
    scaled_values = scaler.transform(values)

    # Target = scaled cpu_usage (column 0) — same scale as input
    scaled_target = scaled_values[:, 0:1]

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

    # ── CPU-only scaler for inverse transform in evaluation ─────────
    cpu_scaler = MinMaxScaler()
    cpu_scaler.fit(df[["cpu_usage"]].values[:train_end])

    print(f"  Feature count: {len(feature_cols)} (N_FEATURES={N_FEATURES})")
    print(f"  Spike flag distribution: "
          f"{df['spike_flag'].mean()*100:.1f}% of points flagged")
    print(f"  cpu_diff1 range: [{df['cpu_diff1'].min():.4f}, {df['cpu_diff1'].max():.4f}]")

    return {
        "X_train":       X_train,
        "y_train":       y_train,
        "X_val":         X_val,
        "y_val":         y_val,
        "X_test":        X_test,
        "y_test":        y_test,
        "scaler":        scaler,
        "cpu_scaler":    cpu_scaler,
        "feature_names": feature_cols,
        "original_data": df.reset_index(drop=True),
        "num_features":  len(feature_cols)
    }
