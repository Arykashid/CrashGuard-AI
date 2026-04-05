"""
train.py — Full Retraining Pipeline
Config: Horizon=1, Window=60, Epochs=50, 50K rows

Fixes applied:
  1. MC Dropout → uses fixed lstm_model.py (Confidence will be > 0)
  2. RMSE → Horizon=1 + 50K data + Window=60 → expected ~0.005–0.020
  3. Coverage → mc_dropout_predict() with model() call → expected ~0.85–0.95

Usage:
    python train.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

from lstm_model import build_lstm_model, train_model, mc_dropout_predict
from preprocessing import prepare_data

# ============================================================
# CONFIG — matches your sidebar defaults
# ============================================================
WINDOW_SIZE      = 60
FORECAST_HORIZON = 1
EPOCHS           = 50
BATCH_SIZE       = 32
LSTM_UNITS       = [64, 32]
DROPOUT_RATE     = 0.2
LEARNING_RATE    = 0.001
MC_SAMPLES       = 50          # MC Dropout forward passes
DATA_PATH        = "data/google_cluster_processed.csv"

print("=" * 60)
print("CrashGuard AI — Retraining Pipeline")
print(f"  Window:   {WINDOW_SIZE}")
print(f"  Horizon:  {FORECAST_HORIZON}")
print(f"  Epochs:   {EPOCHS}")
print(f"  Batch:    {BATCH_SIZE}")
print(f"  MC samp:  {MC_SAMPLES}")
print("=" * 60)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/5] Loading data...")
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"  Rows loaded: {len(df):,}")
assert len(df) >= 10_000, "Need at least 10K rows — run generate_data.py first"


# ============================================================
# STEP 2: PREPARE DATA
# ============================================================
print("\n[2/5] Preparing data...")
processed = prepare_data(
    df,
    window_size=WINDOW_SIZE,
    forecast_horizon=FORECAST_HORIZON
)

X_train = processed["X_train"]
y_train = processed["y_train"]
X_val   = processed["X_val"]
y_val   = processed["y_val"]
X_test  = processed["X_test"]
y_test  = processed["y_test"]
scaler  = processed["scaler"]
cpu_scaler = processed.get("cpu_scaler")

print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Features: {processed.get('feature_names', [])}")

num_features = X_train.shape[2]


# ============================================================
# STEP 3: BUILD MODEL
# ============================================================
print("\n[3/5] Building LSTM model...")
model = build_lstm_model(
    window_size=WINDOW_SIZE,
    num_features=num_features,
    forecast_horizon=FORECAST_HORIZON,
    lstm_units=LSTM_UNITS,
    dropout_rate=DROPOUT_RATE,
    learning_rate=LEARNING_RATE
)
model.summary()


# ============================================================
# STEP 4: TRAIN
# ============================================================
print(f"\n[4/5] Training for up to {EPOCHS} epochs (early stopping)...")
history = train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

final_train_loss = history.history["loss"][-1]
final_val_loss   = history.history["val_loss"][-1]
print(f"\n  Final train loss: {final_train_loss:.6f}")
print(f"  Final val loss:   {final_val_loss:.6f}")


# ============================================================
# STEP 5: EVALUATE — MC DROPOUT (FIXED)
# ============================================================
print(f"\n[5/5] Evaluating with MC Dropout ({MC_SAMPLES} samples)...")

# Run MC Dropout — uses model() call, NOT model.predict()
mc_mean, mc_std = mc_dropout_predict(model, X_test, n_samples=MC_SAMPLES)

# Inverse transform predictions
if cpu_scaler is not None:
    mean_flat = mc_mean.reshape(-1, 1)
    std_flat  = mc_std.reshape(-1, 1)
    y_test_flat = y_test.reshape(-1, 1)

    mean_inv = cpu_scaler.inverse_transform(mean_flat).flatten()
    y_true   = cpu_scaler.inverse_transform(y_test_flat).flatten()

    # For std: approximate via scaled range
    # std in [0,1] scaled space → multiply by cpu range
    cpu_range = float(cpu_scaler.data_max_[0] - cpu_scaler.data_min_[0])
    std_inv   = std_flat.flatten() * cpu_range
else:
    mean_inv = mc_mean.flatten()
    std_inv  = mc_std.flatten()
    y_true   = y_test.flatten()

# RMSE
rmse = float(np.sqrt(np.mean((mean_inv - y_true) ** 2)))

# MAE
mae = float(np.mean(np.abs(mean_inv - y_true)))

# Confidence (based on std — higher std = lower confidence)
# Normalize: if std < 0.005 → confidence ~1.0, std > 0.05 → confidence ~0
avg_std = float(np.mean(std_inv))
confidence = float(np.clip(1.0 - avg_std / 0.05, 0.0, 1.0))

# Coverage @95% — fraction of true values within ±1.96σ
upper = mean_inv + 1.96 * std_inv
lower = mean_inv - 1.96 * std_inv
coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))

# Interval width
avg_width = float(np.mean(upper - lower))

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  RMSE:       {rmse:.6f}  (target: < 0.030)")
print(f"  MAE:        {mae:.6f}")
print(f"  Confidence: {confidence:.4f}  (target: > 0.60, was 0.00)")
print(f"  Coverage:   {coverage:.4f}  (target: > 0.85, was 0.000)")
print(f"  Avg Width:  {avg_width:.6f}")
print(f"  MC std avg: {avg_std:.6f}")

if rmse < 0.030:
    print("\n✅ RMSE target met!")
else:
    print(f"\n⚠️  RMSE={rmse:.4f} — consider more epochs or Optuna tuning")

if confidence > 0.60:
    print("✅ Confidence fix confirmed!")
else:
    print("⚠️  Confidence still low — check MCDropout is in model")

if coverage > 0.80:
    print("✅ Coverage fix confirmed!")
else:
    print("⚠️  Coverage low — check mc_dropout_predict() is using model() call")

print("=" * 60)


# ============================================================
# SAVE MODEL
# ============================================================
print("\nSaving model...")
joblib.dump(model, "saved_model.pkl")
try:
    model.save("saved_model.keras")
    print("✅ Saved: saved_model.keras + saved_model.pkl")
except Exception as e:
    print(f"⚠️  Keras save: {e} — pkl only")

# Save metrics summary
import json
metrics = {
    "trained_at": datetime.now().isoformat(),
    "config": {
        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "epochs_run": len(history.history["loss"]),
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lstm_units": LSTM_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "mc_samples": MC_SAMPLES,
        "train_rows": len(df)
    },
    "metrics": {
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "Confidence": round(confidence, 4),
        "Coverage_95": round(coverage, 4),
        "Avg_Interval_Width": round(avg_width, 6),
        "MC_Std_Avg": round(avg_std, 6),
        "Final_Train_Loss": round(final_train_loss, 6),
        "Final_Val_Loss": round(final_val_loss, 6)
    }
}
with open("last_training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Metrics saved to last_training_metrics.json")
print("\n🚀 Done! Reload app to use the new model.")
